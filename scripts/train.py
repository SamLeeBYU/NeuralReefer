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

    EVAL, SAVE_IMG, FIG_SIZE, METADATA, VAL_SIZE
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

from skopt.space import Real, Integer, Categorical  
from sklearn.model_selection import train_test_split

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

def hold_out(images, val_size=VAL_SIZE, seed=42):
    """
    Splits a list of images into training and validation sets.

    Args:
        images (list): List of image paths.
        val_size (float): Proportion of images to reserve for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        train_images (list), val_images (list)
    """
    train_images, val_images = train_test_split(
        images, test_size=val_size, random_state=seed, shuffle=True
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

    device = torch.device("cuda")

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

        maskloader = MaskLoader(train_images, segmentation_model=segmenter, tolerance=TOLERANCE, mask_size=MASK_SIZE)
        maskloader.save_data(MASK_DATA_PATH)
            
    # Train the model to filter out non-coral masks
    coral_filter = CoralFilterEnsembler(
        base_dataset=MASK_DATA_PATH if train_coral_filter else None,
        device=device,
        m=M, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, split=SPLIT
    )

    if train_coral_filter:
        coral_filter.train()
        coral_filter.validate()
        coral_filter.save_models(FILTER_MODELS_DIR)

    #####################################################################################################################################

    if eval:

        #Once the ensemble model is trained, we can load back in all the parameter values for each model in the enemble as well as the ensemble weights and bias for the final ensemble method
        coral_filter.load_models(FILTER_MODELS_DIR)

        coral_segmenter = CoralSegmenter(config_path, checkpoint_path, coral_filter, annotation_path = f"{TRAIN_DIR}/_annotations.coco.json", device=device)
        #coral_segmenter.summary_stats(images)
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

        print(f"{'Idx':>4} | {'Acc':>6} | {'Avg Acc':>8} | {'True CC':>8} | {'Avg True CC':>12} | {'Pred CC':>8} | {'Avg Pred CC':>12}")
        print("-" * 78)

        for i, image in enumerate(test_images):
            masks, labels = coral_segmenter.predict(img_path = image, init_models=(i == 0), verbose=VERBOSE)
            pred_labels = coral_segmenter.coral_filter.get_class_names(labels, coral_segmenter.coral_filter.classes)
            genus_labels, bleach_labels, gt_masks = coral_segmenter.get_gt_masks(image)
            if genus_labels is not None:
                ml_labels = genus_labels + np.where(bleach_labels == 1, ":bleached", ":healthy")
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

        get_image_id = lambda path: path.split("\\")[-1].split("_")[0]
        predictions['image_id'] = [get_image_id(img) for img in test_images]
        predictions_df = pd.DataFrame(predictions)
        metadata = load_data(METADATA)
        metadata['image_id'] = metadata['filename'].str.split('.').str[0]
        
        predictions_data = pd.merge(predictions_df, metadata, on='image_id', how='left')
        pd.DataFrame(predictions_data).to_csv(f"data/performance/coral_segmenter_predictions.v.{VERSION}.csv", index=False)

if __name__ == "__main__":
    train(eval=True)