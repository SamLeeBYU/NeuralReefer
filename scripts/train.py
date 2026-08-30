"""
Training script for Coral Reef Classification Pipeline.
Handles training of coral filter CNNs and ensembles.

Steps:
- Load MaskLoader dataset from disk
- Instantiate CoralFilter or CoralFilterEnsembler
- Train individual models and/or ensemble weights
- Save trained model weights and ensemble parameters

Run as a standalone script or import train
"""

from config import (
    VERSION, TRAIN_DIR, EXT,
    SAM2_CONFIG_PATH, SAM2_CHECKPOINT_PATH,
    TUNE_SEGMENTER, N_CALLS, K, VERBOSE,
    CREATE_MASK_DATASET, MASK_SIZE, TOLERANCE, MASK_DATA_PATH,

    TRAIN_CORAL_FILTER, M, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, SPLIT, FILTER_MODELS_DIR, PATIENCE,

    EVAL, SAVE_IMG, FIG_SIZE, METADATA, VAL_SIZE, SPATIAL_RADIUS, IMG_SIZE
)

import os
import torch
from data import MaskLoader
from filter import CoralFilterEnsembler
from segmenter import SAM2Segmenter, CoralSegmenter
from transforms import MASK_TRANSFORM
from pathlib import Path

import pandas as pd
import random
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import train_test_split

get_image_id = lambda path: path.split("\\")[-1].split("_")[0]

def union_mask(mask_list, shape=IMG_SIZE):
    """OR's together a list of boolean masks into a single mask of `shape`."""
    if len(mask_list) == 0:
        return np.zeros(shape, dtype=bool)
    return np.any(np.stack(mask_list), axis=0)

def pixel_confusion_metrics(true_mask, pred_mask):
    """
    Pixel-level IoU/Dice(=F1)/precision/recall between two boolean masks,
    computed directly from the raw TP/FP/FN pixel counts (no area
    normalization needed, since it cancels out of every ratio here) --
    unlike the overall LCC metrics in data_viz.R, which had to be
    reconstructed algebraically from aggregate accuracy/coverage numbers
    because the per-mask arrays weren't available there. Here we have the
    actual masks, so this is exact.

    precision/recall are NaN (not 0) when their denominator is zero, since
    "no positive predictions" or "no positive ground truth" makes the ratio
    undefined rather than 0 -- e.g. a taxonomy absent from both masks should
    not be scored as 0 precision.

    Returns:
        tp, fp, fn (int); iou, dice, precision, recall (float; iou/dice are
        1.0 when both masks are empty, matching the "correctly predicted
        nothing" convention)
    """
    tp = int(np.logical_and(true_mask, pred_mask).sum())
    fp = int(np.logical_and(~true_mask, pred_mask).sum())
    fn = int(np.logical_and(true_mask, ~pred_mask).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return tp, fp, fn, iou, dice, precision, recall

def enforce_spatial_independence(train_images, test_images, metadata, radius=SPATIAL_RADIUS):
    """
    Removes spatial leakage between the train/test split: any image within
    `radius` meters of an image in the opposite split is reassigned to the
    train set, since the coral cover in overlapping/adjacent photos is
    correlated and would otherwise violate the train/test independence
    assumption.

    Rather than the naive O(N^2) all-pairs distance check, this builds a
    KD-tree over the (projected, meter-scale) UTM coordinates and uses
    `cKDTree.query_pairs`, which only examines spatially nearby candidates
    (O(N log N) for images spread out over a reef transect, versus
    N*(N-1)/2 for brute force). Images are then grouped into connected
    "overlap clusters" (via union-find over the close pairs) rather than
    checked pairwise one at a time: if any single image in a cluster were
    left in test while another stayed in train, they'd still leak into each
    other transitively, so any cluster touching both splits is folded
    entirely into train.

    Args:
        train_images (list): training image paths.
        test_images (list): test image paths.
        metadata (pd.DataFrame): must contain 'image_id', 'NorthPhoto_UTM',
            'EastPhoto_UTM' columns.
        radius (float): exclusion radius in meters. If None, returns the
            split unchanged.

    Returns:
        train_images (list), test_images (list): the adjusted split.
    """
    if radius is None:
        return train_images, test_images

    all_images = train_images + test_images
    image_ids = [get_image_id(path) for path in all_images]
    split = np.array(["train"] * len(train_images) + ["test"] * len(test_images))

    coords = (
        metadata.set_index("image_id")[["NorthPhoto_UTM", "EastPhoto_UTM"]]
        .reindex(image_ids)
        .to_numpy(dtype=float)
    )

    valid = ~np.isnan(coords).any(axis=1)
    n_missing = int((~valid).sum())
    if n_missing and VERBOSE:
        print(f"Warning: {n_missing} image(s) have no usable GPS coordinates in metadata "
              f"and cannot be spatially checked; leaving their split assignment unchanged.")

    valid_idx = np.where(valid)[0]
    tree = cKDTree(coords[valid_idx])
    close_pairs = tree.query_pairs(r=radius)  # indices are local to valid_idx

    parent = list(range(len(valid_idx)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in close_pairs:
        union(i, j)

    clusters = defaultdict(list)
    for local_i in range(len(valid_idx)):
        clusters[find(local_i)].append(valid_idx[local_i])

    new_split = split.copy()
    n_reassigned = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        if len(set(new_split[members])) > 1:
            for m in members:
                if new_split[m] != "train":
                    new_split[m] = "train"
                    n_reassigned += 1

    if n_reassigned and VERBOSE:
        print(f"Reassigned {n_reassigned} test image(s) to train: within {radius}m of a "
              f"spatially connected train image.")

    new_train = [all_images[i] for i in range(len(all_images)) if new_split[i] == "train"]
    new_test = [all_images[i] for i in range(len(all_images)) if new_split[i] == "test"]

    return new_train, new_test

def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif ext == '.tsv':
        return pd.read_csv(file_path, sep='\t')
    elif ext == '.json':
        return pd.read_json(file_path)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    elif ext in ['.pkl', '.pickle']:
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def hold_out(images, val_size=VAL_SIZE, seed=42, metadata_path=METADATA, radius=SPATIAL_RADIUS):
    """
    Splits a list of images into training and validation sets, then, if
    `metadata_path` and `radius` are set, folds any test image within
    `radius` meters of a train image back into train so the split doesn't
    violate the train/test independence assumption (see
    `enforce_spatial_independence`).

    Args:
        images (list): List of image paths.
        val_size (float): Proportion of images to reserve for validation.
        seed (int): Random seed for reproducibility.
        metadata_path (str): Path to the metadata file with GPS coordinates.
            Set to None to skip spatial conditioning.
        radius (float): Exclusion radius in meters. Set to None to skip
            spatial conditioning.

    Returns:
        train_images (list), val_images (list)
    """
    train_images, val_images = train_test_split(
        images, test_size=val_size, random_state=seed, shuffle=True
    )

    if metadata_path is not None and radius is not None:
        metadata = load_data(metadata_path)
        metadata["image_id"] = metadata["filename"].str.split(".").str[0]
        train_images, val_images = enforce_spatial_independence(
            train_images, val_images, metadata, radius=radius
        )

    return train_images, val_images

def train(tune_segmenter: bool = TUNE_SEGMENTER,
          create_mask_dataset: bool = CREATE_MASK_DATASET,
          train_coral_filter: bool = TRAIN_CORAL_FILTER,
          eval: bool = EVAL
          ):

    images = [os.path.join(TRAIN_DIR, file) for file in os.listdir(TRAIN_DIR) if file.endswith(EXT)]

    if VAL_SIZE > 0:
        train_images, test_images = hold_out(images)
    else:
        train_images = test_images = images

    device = None #torch.device("cuda")

    checkpoint_path=SAM2_CHECKPOINT_PATH
    config_path=SAM2_CONFIG_PATH

    # SAM2 Hyperparameter Tuning #############################################################################################################################

    if tune_segmenter:

        segmenter = SAM2Segmenter(

            checkpoint_path=checkpoint_path,
            config_path=config_path,
            annotation_path = f"{TRAIN_DIR}/_annotations.coco.json",

            device = device

        )

        # ========================
        # Define the Search Space
        # ========================

        search_space = [
            # Large feature params (9)
            Integer(4, 8, name='large_points_per_side'),
            Categorical([36], name='large_points_per_batch'),
            Real(0.15, 0.35, name='large_pred_iou_thresh'),
            Real(0.1, 0.5, name='large_stability_score_thresh'),
            Real(0.9, 1.0, name='large_stability_score_offset'),
            Categorical([0], name='large_crop_n_layers'),
            Real(0.3, 0.5, name='large_box_nms_thresh'),
            Categorical([1], name='large_crop_n_points_downscale_factor'),
            Categorical([10000], name='large_min_mask_region_area'),

            # Small feature params (9)
            Integer(16, 32, name='small_points_per_side'),
            Categorical([64], name='small_points_per_batch'),
            Real(0.2, 0.4, name='small_pred_iou_thresh'),
            Real(0.3, 0.5, name='small_stability_score_thresh'),
            Real(0.9, 1.0, name='small_stability_score_offset'),
            Categorical([0], name='small_crop_n_layers'),
            Real(0.1, 0.5, name='small_box_nms_thresh'),
            Categorical([1], name='small_crop_n_points_downscale_factor'),
            Categorical([400], name='small_min_mask_region_area'),

            # NR params (7)
            Real(0, 0.2, name='overlap'),
            Integer(2, 16, name='clipLimit'),
            Integer(8, 10, name='tileGridSize'),
            Real(1.0, 2.0, name='redBoost'),
            Categorical([True], name='whiteBalance'),
            Real(0, 1.0, name='gamma')
        ]

        best_params = segmenter.tune(train_images, search_space, n_calls=N_CALLS, k_samples=K, verbose=VERBOSE)

    ###########################################################################################################################################################

    # Create Artificial Labeled Dataset for Coral Classification by
    # - Treat ground truth masks as positive samples (and there are no other positive samples other than the ground truth masks)
    # - Use the base SAM2 segmenter to find masks of other objects
    # - Any mask generated by the SAM2 segmenter that does not overlap significantly (determined by a tolerance hyperparameter) with a ground turth is a negative sample

    if create_mask_dataset:

        segmenter = SAM2Segmenter(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            annotation_path = f"{TRAIN_DIR}/_annotations.coco.json",
            device = device
        )

        maskloader = MaskLoader(test_images, segmentation_model=segmenter, tolerance=TOLERANCE, mask_size=MASK_SIZE)
        maskloader.save_data(MASK_DATA_PATH)

    # Train the model to filter out non-coral masks
    coral_filter = CoralFilterEnsembler(
        base_dataset=MASK_DATA_PATH if train_coral_filter else None,
        device=device,
        m=M, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, split=SPLIT
    )

    if train_coral_filter:
        coral_filter.train()
        coral_filter.train_ensemble()
        coral_filter.validate()
        coral_filter.save_models(FILTER_MODELS_DIR)

    #####################################################################################################################################

    if eval:

        #Once the ensemble model is trained, we can load back in all the parameter values for each model in the enemble as well as the ensemble weights and bias for the final ensemble method
        coral_filter.load_models(FILTER_MODELS_DIR)

        coral_segmenter = CoralSegmenter(config_path, checkpoint_path, coral_filter, annotation_path = f"{TRAIN_DIR}/_annotations.coco.json", device=device)
        #coral_segmenter.summary_stats(train_images)
        #coral_segmenter.summary_stats(test_images, output_file="data/test_data_summary.json")
        save_dir = Path(f"figures/segmentation.v.{VERSION}")
        save_dir.mkdir(parents=True, exist_ok=True)

        pixel_accuracies = np.zeros(len(test_images))

        coral_cover_true = np.zeros(len(test_images))
        coral_cover_pred = np.zeros(len(test_images))

        pct_bleached_true = np.zeros(len(test_images))
        pct_bleached_pred = np.zeros(len(test_images))

        genus_names = sorted(set(
            k.split(":")[0] for k in coral_segmenter.coral_filter.classes
            if ":bleached" in k or ":healthy" in k
        ))

        coral_cover_class_healthy_true = np.zeros((len(test_images), len(genus_names)))
        coral_cover_class_healthy_pred = np.zeros((len(test_images), len(genus_names)))
        coral_cover_class_true = np.zeros((len(test_images), len(genus_names)))
        coral_cover_class_pred = np.zeros((len(test_images), len(genus_names)))

        # Per-taxonomy pixel-level IoU/Dice(=F1)/precision/recall, computed
        # directly from the actual predicted/ground-truth masks (see
        # pixel_confusion_metrics). Macro-averaging and confidence intervals
        # are computed downstream, in data_viz.R, from this per-image CSV.
        taxonomy_records = []

        print(f"{'Idx':>4} | {'Acc':>6} | {'Avg Acc':>8} | {'True CC':>8} | {'Avg True CC':>12} | {'Pred CC':>8} | {'Avg Pred CC':>12}")
        print("-" * 78)

        for i, image in enumerate(test_images):
            masks, labels = coral_segmenter.predict(img_path = image, init_models=(i == 0), verbose=VERBOSE)
            pred_labels = coral_segmenter.coral_filter.get_class_names(labels, coral_segmenter.coral_filter.classes)
            genus_labels, bleach_labels, gt_masks = coral_segmenter.get_gt_masks(image)
            if genus_labels is not None:
                ml_labels = np.char.add(genus_labels, np.where(bleach_labels == 1, ":bleached", ":healthy"))
            else:
                ml_labels = np.array([])

            gt_masks = [segmentation for j, segmentation in enumerate(gt_masks) if genus_labels[j] != "noncoral"]
            gt_labels = [ml_labels[j] for j in range(len(ml_labels)) if genus_labels[j] != "noncoral"]

            pixel_accuracies[i] = coral_segmenter.accuracy(masks, gt_masks)
            coral_cover_true[i] = coral_segmenter.coral_cover(gt_masks, cs=coral_segmenter.crop_space)
            coral_cover_pred[i] = coral_segmenter.coral_cover(masks, cs=coral_segmenter.crop_space)

            # Bleached coverage
            pct_bleached_true[i] = coral_segmenter.coral_cover([
                gt_masks[j] for j in range(len(gt_labels)) if gt_labels[j].split(":")[-1] == "bleached"
            ], cs=coral_segmenter.crop_space)

            pct_bleached_pred[i] = coral_segmenter.coral_cover([
                masks[j] for j in range(len(pred_labels)) if pred_labels[j].split(":")[-1] == "bleached"
            ], cs=coral_segmenter.crop_space)

            # Class-wise cover (bleached + healthy together)
            for g, genus in enumerate(genus_names):
                coral_cover_class_true[i, g] = coral_segmenter.coral_cover([
                    gt_masks[j] for j in range(len(gt_labels)) if gt_labels[j].startswith(genus + ":")
                ], cs=coral_segmenter.crop_space)

                coral_cover_class_pred[i, g] = coral_segmenter.coral_cover([
                    masks[j] for j in range(len(pred_labels)) if pred_labels[j].startswith(genus + ":")
                ], cs=coral_segmenter.crop_space)

            # Class-wise healthy cover only
            cc_true_healthy = []
            cc_pred_healthy = []
            for k in range(coral_segmenter.coral_filter.k):  # exclude noncoral
                class_name = coral_segmenter.coral_filter.get_class_names([k], coral_segmenter.coral_filter.classes)[0]
                if not class_name.endswith(":healthy"):
                    continue

                cc_true_healthy.append(coral_segmenter.coral_cover([
                    gt_masks[j] for j in range(len(gt_labels)) if gt_labels[j] == class_name
                ], cs=coral_segmenter.crop_space))

                cc_pred_healthy.append(coral_segmenter.coral_cover([
                    masks[j] for j in range(len(pred_labels)) if pred_labels[j] == class_name
                ], cs=coral_segmenter.crop_space))

            coral_cover_class_healthy_true[i, :] = cc_true_healthy
            coral_cover_class_healthy_pred[i, :] = cc_pred_healthy

            # Per-taxonomy IoU/Dice/precision/recall (see pixel_confusion_metrics / union_mask above).
            image_id = get_image_id(image)

            def add_taxonomy_row(taxonomy, true_mask, pred_mask):
                tp, fp, fn, iou, dice, precision, recall = pixel_confusion_metrics(true_mask, pred_mask)
                taxonomy_records.append({
                    "image": image, "image_id": image_id, "taxonomy": taxonomy,
                    "tp_px": tp, "fp_px": fp, "fn_px": fn,
                    "iou": iou, "dice_f1": dice, "precision": precision, "recall": recall
                })

            add_taxonomy_row("all_coral", union_mask(gt_masks), union_mask(masks))

            add_taxonomy_row(
                "bleached",
                union_mask([gt_masks[j] for j in range(len(gt_labels)) if gt_labels[j].endswith(":bleached")]),
                union_mask([masks[j] for j in range(len(pred_labels)) if pred_labels[j].endswith(":bleached")]),
            )

            for genus in genus_names:
                add_taxonomy_row(
                    genus,
                    union_mask([gt_masks[j] for j in range(len(gt_labels)) if gt_labels[j].startswith(genus + ":")]),
                    union_mask([masks[j] for j in range(len(pred_labels)) if pred_labels[j].startswith(genus + ":")]),
                )
                add_taxonomy_row(
                    f"{genus}:healthy",
                    union_mask([gt_masks[j] for j in range(len(gt_labels)) if gt_labels[j] == f"{genus}:healthy"]),
                    union_mask([masks[j] for j in range(len(pred_labels)) if pred_labels[j] == f"{genus}:healthy"]),
                )

            print(f"{i:>4} | {pixel_accuracies[i]:6.3f} | {pixel_accuracies[:i+1].mean():8.3f} "
                f"| {coral_cover_true[i]:8.3f} | {coral_cover_true[:i+1].mean():12.3f} "
                f"| {coral_cover_pred[i]:8.3f} | {coral_cover_pred[:i+1].mean():12.3f}")

            if SAVE_IMG:
                save_path_pred = save_dir / f"{Path(image).name.split('_')[0]}_{pixel_accuracies[i]:.4f}.png"
                save_path_true = save_dir / f"{Path(image).name.split('_')[0]}_gt.png"
                coral_segmenter.show_masks(masks, coral_segmenter.color_map, pred_labels, show=False, save_path=save_path_pred)
                coral_segmenter.show_masks(gt_masks, coral_segmenter.color_map, gt_labels, show=False, save_path=save_path_true)
                #coral_segmenter.show_masks_side_by_side(gt_masks, masks, figsize=FIG_SIZE, save_path=save_path)

        print("-" * 78)

        predictions = {
            'image': test_images,
            'accuracy': pixel_accuracies,
            'coral_cover': coral_cover_true,
            'coral_cover_pred': coral_cover_pred,
            'pct_bleached_true': pct_bleached_true,
            'pct_bleached_pred': pct_bleached_pred
        }

        # Add genus-wise total coral cover (bleached + healthy)
        for g, genus in enumerate(genus_names):
            predictions[f'cover_true__{genus}'] = coral_cover_class_true[:, g]
            predictions[f'cover_pred__{genus}'] = coral_cover_class_pred[:, g]

        # Add genus-wise healthy coral cover only
        for g, genus in enumerate(genus_names):
            predictions[f'cover_healthy_true__{genus}'] = coral_cover_class_healthy_true[:, g]
            predictions[f'cover_healthy_pred__{genus}'] = coral_cover_class_healthy_pred[:, g]

        predictions['image_id'] = [get_image_id(img) for img in test_images]
        predictions_df = pd.DataFrame(predictions)
        metadata = load_data(METADATA)
        metadata['image_id'] = metadata['filename'].str.split('.').str[0]

        predictions_data = pd.merge(predictions_df, metadata, on='image_id', how='left')
        pd.DataFrame(predictions_data).to_csv(f"data/performance/coral_segmenter_predictions.v.{VERSION}.csv", index=False)

        taxonomy_df = pd.DataFrame(taxonomy_records)
        taxonomy_df.to_csv(f"data/performance/coral_segmenter_taxonomy_metrics.v.{VERSION}.csv", index=False)

if __name__ == "__main__":
    train(eval=True)