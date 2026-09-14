"""
Adds the "genus_bleach" taxonomy rows (per-genus, per-bleach-status pixel
IoU/Dice/precision/recall -- see utils.compute_coco_taxonomy_metrics) to the
already-computed data/performance/inference/annotations_coco_{train,test}_metrics.csv
files, without re-running SAM2/the CNN ensemble: it re-scores the *already
exported* predicted COCO jsons from that same run
(data/performance/inference/annotations_coco_{train,test}.json) against the
ground-truth COCO (config.TRAIN_DIR/_annotations.coco.json) using only the
new taxonomy mode, then concatenates the result onto the existing metrics
CSVs (skipping recomputation of the lcc/bleached/genus rows already there).

This exists to give the paper's Table 5 (bias decomposition by BOTH genus
AND bleach status) a same-vintage source as everything else now pulled from
the Sep-12 data/performance/inference/ run, instead of falling back to the
older coral_segmenter_predictions.v.1.0.csv for that one table.

Run as a standalone script from the repository root:
    python scripts/compute_genus_bleach_metrics.py
"""
import os

import pandas as pd

from config import TRAIN_DIR
from utils import compute_coco_taxonomy_metrics

INFERENCE_DIR = "data/performance/inference"
GT_COCO = os.path.join(TRAIN_DIR, "_annotations.coco.json")


def add_genus_bleach(split_suffix):
    pred_coco = os.path.join(INFERENCE_DIR, f"annotations_coco{split_suffix}.json")
    metrics_csv = os.path.join(INFERENCE_DIR, f"annotations_coco{split_suffix}_metrics.csv")

    new_rows = compute_coco_taxonomy_metrics(pred_coco, GT_COCO, taxonomy="genus_bleach")

    existing = pd.read_csv(metrics_csv)
    if existing["taxonomy"].str.contains(":").any():
        print(f"{metrics_csv} already has genus_bleach rows -- skipping (delete them first to recompute).")
        return

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(metrics_csv, index=False)
    print(f"Wrote {metrics_csv} ({len(existing)} existing + {len(new_rows)} new genus_bleach rows)")


if __name__ == "__main__":
    # Only the train/test splits are needed (Table 5 is reported by split).
    # The unsplit "" file (data/performance/inference/annotations_coco.json)
    # is currently corrupted on disk (JSONDecodeError partway through
    # parsing) and isn't required for anything downstream -- skipped here
    # rather than worked around, since nothing reads it.
    for suffix in ("_train", "_test"):
        add_genus_bleach(suffix)
