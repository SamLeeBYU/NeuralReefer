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

    EVAL, SAVE_IMG, FIG_SIZE, METADATA, VAL_SIZE, SPATIAL_RADIUS, CAMERA_HFOV_DEG, IMG_SIZE
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

get_image_id = lambda path: os.path.basename(path).split("_")[0]

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

def fov_radius_m(depth, hfov_deg=CAMERA_HFOV_DEG):
    """
    Nadir GoPro footprint radius on the seafloor (meters) from water depth.

    Args:
        depth (float): Depth_WaterSurface in meters (may be negative in metadata).
        hfov_deg (float): Full horizontal field of view in degrees.

    Returns:
        float: Footprint radius in meters, or NaN if depth is missing.
    """
    if depth is None or (isinstance(depth, float) and np.isnan(depth)):
        return float("nan")
    return abs(float(depth)) * np.tan(np.radians(hfov_deg / 2))

def build_overlap_components(coords, radii):
    """
    Group images whose seafloor footprints overlap (dist < r_i + r_j).

    Uses a KD-tree broad phase (pairs within 2 * max(r)) then filters to the
    variable-radius overlap rule, and union-find for connected components.

    Args:
        coords (np.ndarray): (N, 2) UTM coordinates.
        radii (np.ndarray): (N,) footprint radius per image in meters.

    Returns:
        list[list[int]]: Connected components as lists of local indices.
    """
    n = len(coords)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    max_r = float(np.max(radii))
    if max_r <= 0:
        return [[i] for i in range(n)]

    tree = cKDTree(coords)
    broad_pairs = tree.query_pairs(r=2 * max_r)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in broad_pairs:
        if np.linalg.norm(coords[i] - coords[j]) < radii[i] + radii[j]:
            union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    return list(clusters.values())

def assign_components_to_split(components, n_target_test, rng):
    """
    Assign overlap components to test so total test count is closest to target.

    Components are shuffled for randomness, then processed smallest-first so
    small components can fine-tune the final test count.

    Args:
        components (list[list[int]]): Each inner list is image indices in one component.
        n_target_test (int): Desired number of test images.
        rng (np.random.Generator): Random number generator.

    Returns:
        set[int]: Global image indices assigned to test.
    """
    if not components:
        return set()

    comps = [list(c) for c in components]
    rng.shuffle(comps)
    comps.sort(key=len)

    test_indices = set()
    for comp in comps:
        new_count = len(test_indices) + len(comp)
        old_diff = abs(len(test_indices) - n_target_test)
        new_diff = abs(new_count - n_target_test)
        if new_diff <= old_diff:
            test_indices.update(comp)

    return test_indices

def _spatial_radii_for_images(image_ids, metadata, radius=SPATIAL_RADIUS, hfov_deg=CAMERA_HFOV_DEG):
    """
    Per-image footprint radii and validity mask for spatial hold-out.

    When `radius` is set, all valid-coordinate images use that uniform radius.
    Otherwise radii come from depth via `fov_radius_m`.
    """
    meta = metadata.set_index("image_id")
    depths = meta["Depth_WaterSurface"].reindex(image_ids).to_numpy(dtype=float)

    if radius is not None:
        radii = np.full(len(image_ids), float(radius))
        valid = ~np.isnan(radii)
    else:
        radii = np.array([fov_radius_m(d, hfov_deg) for d in depths], dtype=float)
        valid = ~np.isnan(radii)

    return radii, valid

def enforce_spatial_independence(
    train_images,
    test_images,
    metadata,
    radius=SPATIAL_RADIUS,
    hfov_deg=CAMERA_HFOV_DEG,
):
    """
    Removes spatial leakage between an existing train/test split: any overlap
    component (footprints with dist < r_i + r_j, transitively) that touches
    both splits is folded entirely into train.

    Prefer `hold_out()` for new splits; this function adjusts a pre-existing
    random split using the same overlap rules.

    Args:
        train_images (list): training image paths.
        test_images (list): test image paths.
        metadata (pd.DataFrame): must contain 'image_id', 'NorthPhoto_UTM',
            'EastPhoto_UTM', and (unless radius is set) 'Depth_WaterSurface'.
        radius (float): Optional uniform radius override in meters. If None,
            depth-based FoV radii are used.
        hfov_deg (float): Horizontal FoV when using depth-based radii.

    Returns:
        train_images (list), test_images (list): the adjusted split.
    """
    if radius is None and hfov_deg is None:
        return train_images, test_images

    all_images = train_images + test_images
    image_ids = [get_image_id(path) for path in all_images]
    split = np.array(["train"] * len(train_images) + ["test"] * len(test_images))

    coords = (
        metadata.set_index("image_id")[["NorthPhoto_UTM", "EastPhoto_UTM"]]
        .reindex(image_ids)
        .to_numpy(dtype=float)
    )
    radii, radius_valid = _spatial_radii_for_images(image_ids, metadata, radius, hfov_deg)
    valid = (~np.isnan(coords).any(axis=1)) & radius_valid

    n_missing = int((~valid).sum())
    if n_missing and VERBOSE:
        print(
            f"Warning: {n_missing} image(s) have no usable GPS/depth in metadata "
            f"and cannot be spatially checked; leaving their split assignment unchanged."
        )

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return train_images, test_images

    components = build_overlap_components(coords[valid_idx], radii[valid_idx])
    local_to_global = {local: valid_idx[local] for local in range(len(valid_idx))}

    new_split = split.copy()
    n_reassigned = 0
    for comp in components:
        if len(comp) < 2:
            continue
        global_members = [local_to_global[local_i] for local_i in comp]
        if len(set(new_split[global_members])) > 1:
            for m in global_members:
                if new_split[m] != "train":
                    new_split[m] = "train"
                    n_reassigned += 1

    if n_reassigned and VERBOSE:
        print(
            f"Reassigned {n_reassigned} test image(s) to train: overlapping footprint "
            f"with a spatially connected train image."
        )

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

def hold_out(
    images,
    val_size=VAL_SIZE,
    seed=42,
    metadata_path=METADATA,
    radius=SPATIAL_RADIUS,
    hfov_deg=CAMERA_HFOV_DEG,
):
    """
    Splits images into train and test, enforcing spatial independence via
    non-overlapping GoPro footprints (dist >= r_train + r_test).

    Overlap components (transitively linked footprints) are assigned entirely
    to train or test via randomized component selection targeting `val_size`,
    rather than demoting test images after a random split.

    Args:
        images (list): List of image paths.
        val_size (float): Proportion of images to reserve for test.
        seed (int): Random seed for reproducibility.
        metadata_path (str): Path to metadata with UTM coords and depth.
            Set to None to skip spatial conditioning (plain random split).
        radius (float): Optional uniform radius override in meters. If None,
            depth-based FoV radii from `hfov_deg` are used.
        hfov_deg (float): Horizontal FoV when using depth-based radii.

    Returns:
        train_images (list), val_images (list)
    """
    use_spatial = metadata_path is not None and (radius is not None or hfov_deg is not None)
    if not use_spatial:
        train_images, val_images = train_test_split(
            images, test_size=val_size, random_state=seed, shuffle=True
        )
        return train_images, val_images

    metadata = load_data(metadata_path)
    metadata["image_id"] = metadata["filename"].str.split(".").str[0]

    image_ids = [get_image_id(path) for path in images]
    coords = (
        metadata.set_index("image_id")[["NorthPhoto_UTM", "EastPhoto_UTM"]]
        .reindex(image_ids)
        .to_numpy(dtype=float)
    )
    radii, radius_valid = _spatial_radii_for_images(image_ids, metadata, radius, hfov_deg)
    coord_valid = ~np.isnan(coords).any(axis=1)
    spatial_valid = coord_valid & radius_valid

    n_missing = int((~spatial_valid).sum())
    if n_missing and VERBOSE:
        print(
            f"Warning: {n_missing} image(s) have no usable GPS/depth in metadata "
            f"and cannot be spatially constrained; assigning them without overlap checks."
        )

    rng = np.random.default_rng(seed)
    n_target_test = int(round(len(images) * val_size))
    components = []

    valid_idx = np.where(spatial_valid)[0]
    if len(valid_idx) > 0:
        overlap_comps = build_overlap_components(coords[valid_idx], radii[valid_idx])
        for comp in overlap_comps:
            components.append([valid_idx[local_i] for local_i in comp])

    for i in np.where(~spatial_valid)[0]:
        components.append([int(i)])

    test_index_set = assign_components_to_split(components, n_target_test, rng)
    val_images = [images[i] for i in sorted(test_index_set)]
    train_images = [images[i] for i in range(len(images)) if i not in test_index_set]

    n_test = len(val_images)
    if VERBOSE:
        print(
            f"Spatial hold-out: {len(components)} component(s), "
            f"target test={n_target_test}, actual test={n_test} "
            f"({100 * n_test / len(images):.1f}%)."
        )
    if abs(n_test - n_target_test) > 0 and VERBOSE:
        print(
            f"Warning: could not reach target test size {n_target_test} "
            f"under spatial constraints (got {n_test})."
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