"""
Example script: run the trained NeuralReefer segmentation + classification
pipeline on the sample images in data/examples.

Equivalent to:
    python scripts/main.py --mode inference --image_dir data/examples

Requires:
- Trained filter models at config.FILTER_MODELS_DIR
- SAM2 checkpoint/config available at config.SAM2_CHECKPOINT_PATH / SAM2_CONFIG_PATH
  (config.SAM2_PATH must point to a local sam2 checkout)

Outputs:
- data/examples/inference/inference_statistics.csv  (per-image coral cover / bleaching estimates)
- data/examples/inference/<image_id>.png             (predicted mask overlays, if config.SAVE_MASKS)
- data/examples/annotations_coco.json                (predicted masks in COCO format, if config.SAVE_COCO)
"""

import pandas as pd

from config import METADATA
from main import inference
from train import load_data

IMAGE_DIR = "data/examples"

if __name__ == "__main__":
    output_file = f"{IMAGE_DIR}/inference/inference_statistics.csv"
    predictions_data = inference(IMAGE_DIR, output_file)

    if METADATA is not None:
        metadata = load_data(METADATA)
        metadata["image_id"] = metadata["filename"].str.split(".").str[0]
        predictions_data = pd.merge(predictions_data, metadata, on="image_id", how="left")

    predictions_data.to_csv(output_file, index=False)
    print(f"Saved coral cover estimates to {output_file}")
