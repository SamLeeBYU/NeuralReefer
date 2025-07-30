"""

NeuralReefer: Coral Reef Segmentation and Classification

NeuralReefer implements machine learning modules for processing top-down coral reef imagery 
collected by the WHOI Yellowfin Surfzone ASV1 platform in Majuro, Marshall Islands through an automated pipeline. 
It includes functionality for segmenting coral regions from raw RGB imagery using the SAM2 segmentation model, 
filtering mask candidates using CNN-based coral classifiers, and constructing an ensemble filter 
to improve coral vs non-coral classification performance. Cropped mask regions are used as training 
data for downstream classification tasks including coral genus and bleaching status.

The pipeline consists of:
- Image preprocessing
- Ground-truth mask extraction from COCO annotations
- SAM2 mask prediction and filtering using a ResNet-based CNN
- Bootstrapped ensemble of CNN classifiers with logistic meta-learner
- Optional hyperparameter tuning via Bayesian optimization (skopt)

This work contributes to automating live coral cover (LCC) estimation and health monitoring 
using semantic segmentation and classification pipelines, thereby reducing manual labor and 
increasing scalability of reef ecosystem assessments.

Author: Sam Lee
Institution: Brigham Young University* / WHOI**  
Contact: sam.lee@whoi.edu**, slee039@byu.edu*, samlee.byu@gmail.com (personal)  
Date Created: June 2025

Use Cases (terminal):

Train the entire the entire pipeline with your own data
(make sure to adjust global parameters in config.py)

> python scripts/main.py --mode train

(For developers) show diagnostic plots

> python scripts/main.py --mode visualize --version 2.0

Main use case: Use NeuralReefer to obtain summary data on new coral images

> python scripts/main.py --mode inference --image_dir data/test/tabletops

"""

import argparse
import os
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

from config import SAM2_CONFIG_PATH, SAM2_CHECKPOINT_PATH, FILTER_MODELS_DIR, EXT, VERBOSE, M, SAVE_MASKS, METADATA

from utils import suppress_prints, restore_prints
from train import train, load_data
from visualize import plot_segmentation_summary, plot_coral_cover

from filter import CoralFilterEnsembler
from segmenter import CoralSegmenter

def inference(image_dir: str) -> pd.DataFrame:
    """
    Evaluates coral cover for a directory of images using a trained CoralSegmenter.

    Args:
        image_dir (str): Directory containing images to process.

    Returns:
        pd.DataFrame: DataFrame with image metadata and coral cover breakdown.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coral_filter = CoralFilterEnsembler(base_dataset=None, m=M, device=device)
    coral_filter.load_models(FILTER_MODELS_DIR)

    segmenter = CoralSegmenter(
        config_path=SAM2_CONFIG_PATH,
        checkpoint_path=SAM2_CHECKPOINT_PATH,
        coral_filter=coral_filter,
        device=device
    )

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(EXT)]
    genus_names = sorted(set(
        k.split(":")[0] for k in segmenter.coral_filter.classes
        if ":bleached" in k or ":healthy" in k
    ))

    save_dir = Path(f"{image_dir}/inference")
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, img_path in enumerate(tqdm(image_paths)):
        image_id = Path(img_path).stem.split('_')[0]

        if not VERBOSE: suppress_prints()
        masks, labels = segmenter.predict(img_path=img_path, init_models=(i==0), verbose=VERBOSE)
        if not VERBOSE: restore_prints()

        pred_labels = segmenter.coral_filter.get_class_names(labels, segmenter.coral_filter.classes)

        cover = segmenter.coral_cover(masks, cs=segmenter.crop_space)
        pct_bleached = segmenter.coral_cover(
            [masks[j] for j in range(len(pred_labels)) if pred_labels[j].endswith(":bleached")],
            cs=segmenter.crop_space
        )

        genus_cover = {}
        genus_cover_healthy = {}
        for genus in genus_names:
            genus_cover[genus] = segmenter.coral_cover(
                [masks[j] for j in range(len(pred_labels)) if pred_labels[j].startswith(f"{genus}:")],
                cs=segmenter.crop_space
            )
            genus_cover_healthy[genus] = segmenter.coral_cover(
                [masks[j] for j in range(len(pred_labels)) if pred_labels[j] == f"{genus}:healthy"],
                cs=segmenter.crop_space
            )

        record = {
            "image": img_path,
            "image_id": image_id,
            "coral_cover_pred": cover,
            "pct_bleached_pred": pct_bleached,
        }

        for g in genus_names:
            record[f"cover_pred__{g}"] = genus_cover[g]
            record[f"cover_healthy_pred__{g}"] = genus_cover_healthy[g]

        if SAVE_MASKS:
            segmenter.show_masks(masks, segmenter.color_map, pred_labels, show=False,
                                 save_path=save_dir / f"{image_id}.png")

        results.append(record)

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralReefer")
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "visualize-coral-cover", "inference"], help="Execution mode")
    #parser.add_argument("--version", type=str, required=False, help="Version identifier for visualizations")
    parser.add_argument("--image_dir", type=str, required=False, help="Directory of images to evaluate")
    parser.add_argument("--predictions_file", type=str, required=False, help="File path of evaluations (generated from mode=inference)")

    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "visualize":
        assert args.prediction_file
        plot_coral_cover(args.prediction_file)

    elif args.mode == "inference":
        assert args.image_dir, "Must provide --image_dir"
        predictions_data = inference(args.image_dir)
        output_file = f"{args.image_dir}/inference/inference.csv"

        if METADATA is not None:
            metadata = load_data(METADATA)
            metadata['image_id'] = metadata['filename'].str.split('.').str[0]
            
            predictions_data = pd.merge(predictions_data, metadata, on='image_id', how='left')

        predictions_data.to_csv(predictions_data, index=False)
        print(f"Saved coral cover estimates to {output_file}")