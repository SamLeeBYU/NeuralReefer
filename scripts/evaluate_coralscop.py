"""
Scores CoralSCOP's aggregated COCO predictions (see aggregate_coralscop.py)
against ground truth for live coral cover (LCC), split into in-sample
(train) / out-of-sample (test) via data/metadata/train_test_split_metadata.csv.
Only taxonomy="lcc" is computed -- CoralSCOP has no genus/bleaching
breakdown to score (see utils.compute_coco_taxonomy_metrics).
"""
import json
import pandas as pd
from config import TRAIN_DIR
from utils import compute_coco_taxonomy_metrics

CORALSCOP_COCO = "data/performance/coralscop.coco.json"
GT_COCO = f"{TRAIN_DIR}/_annotations.coco.json"
SPLIT_METADATA = "data/metadata/train_test_split_metadata.csv"

coralscop = json.load(open(CORALSCOP_COCO))
split_of = dict(pd.read_csv(SPLIT_METADATA)[["filename", "split"]].values)

rows = []
for split in ("train", "test"):
    ids = {img["id"] for img in coralscop["images"] if split_of.get(img["file_name"]) == split}
    sub_coco = {
        "images": [img for img in coralscop["images"] if img["id"] in ids],
        "annotations": [a for a in coralscop["annotations"] if a["image_id"] in ids],
        "categories": coralscop["categories"],
    }
    df = compute_coco_taxonomy_metrics(sub_coco, GT_COCO, taxonomy="lcc")
    df["split"] = split
    rows.append(df)

metrics = pd.concat(rows, ignore_index=True)
metrics.to_csv("data/performance/coralscop_metrics.csv", index=False)

summary = metrics.groupby("split")[["iou", "dice_f1", "precision", "recall"]].mean()
summary.insert(0, "n", metrics.groupby("split").size())
summary = summary.rename(index={"train": "In-Sample", "test": "Out-of-Sample"})

print(summary)
with open("data/performance/coralscop_performance.txt", "w") as f:
    f.write(summary.to_string() + "\n")
