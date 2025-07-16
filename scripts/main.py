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

> python scripts/main.py --mode eval --image_dir data/test/tabletops


"""

import argparse
import os
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

from config import SAM2_CONFIG_PATH, SAM2_CHECKPOINT_PATH, FILTER_MODELS_DIR, EXT, VERBOSE, M, SAVE_MASKS

from utils import suppress_prints, restore_prints
from train import train
from visualize import plot_segmentation_summary

from filter import CoralFilterEnsembler
from segmenter import CoralSegmenter


def eval(image_dir: str) -> pd.DataFrame:
    """
    Evaluates coral cover for a directory of images using a trained CoralSegmenter.

    Args:
        image_dir (str): Directory containing images to process (all images with the extension EXT will be included)

    Returns:
        pd.DataFrame: DataFrame with columns: image, image_id, coral_cover
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

    results = []
    save_dir = Path(f"{image_dir}/eval")
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(image_paths))):
        img_path = image_paths[i]
        
        if not VERBOSE: suppress_prints()
        masks = segmenter.predict(img_path=img_path, init_models=(i==0), verbose=VERBOSE)
        if not VERBOSE: restore_prints()

        cover = segmenter.coral_cover(masks, cs=segmenter.crop_space)
        if VERBOSE:
            print(f"Predicted coral cover: {cover*100:.4f}%")
        image_id = Path(img_path).stem.split('_')[0]

        if SAVE_MASKS:
            segmenter.show_masks(masks, show=False, save_path=save_dir / f"{image_id}.png")

        results.append({"image": img_path, "image_id": image_id, "coral_cover": cover})

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralReefer")
    parser.add_argument("--mode", type=str, default="eval", choices=["train", "visualize", "eval"], help="Execution mode")
    parser.add_argument("--version", type=str, required=False, help="Version identifier for visualizations")
    parser.add_argument("--image_dir", type=str, required=False, help="Directory of images to evaluate")

    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "visualize":
        assert args.version, "Must provide --version"
        plot_segmentation_summary(version=args.version)

    elif args.mode == "eval":
        assert args.image_dir, "Must provide --image_dir"
        df = eval(args.image_dir)
        output_file = f"{args.image_dir}/eval/eval.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved coral cover estimates to {output_file}")