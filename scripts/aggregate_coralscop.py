"""
Concatenates a directory of CoralSCOP's (github.com/zhengziqiang/CoralSCOP)
per-image json prediction files into ONE combined COCO json, in exactly the
shape export_coco.COCOExporter produces for this pipeline's own predictions
-- built by actually calling that class's add_image/add_annotation
(decoding each RLE segmentation into a boolean mask via
utils._rasterize_coco_mask first), not a hand-rolled equivalent, so
bbox/area/iscrowd/supercategory and the polygon segmentation encoding match
this pipeline's own predicted COCO exports. Kept separate from
evaluate_coralscop.py (which only reads the result) since aggregation only
needs re-running when CoralSCOP produces new/updated per-image files, not
every time the comparison itself is run.

CoralSCOP's per-image files look like:
    {"image": {"image_filename": ..., "image_width": ..., "image_height": ..., "id": ...},
     "annotations": [{"id": ..., "image_id": ..., "category_id": ...,
                       "segmentation": {"size": [h, w], "counts": <RLE string>}}, ...],
     "categories": [{"id": 0, "name": "coral", ...}]}
-- a binary coral/not-coral segmenter (exactly one category), so every
annotation's category is relabeled CORALSCOP_CATEGORY_NAME ("coral:unknown")
rather than run through the ground truth's genus/bleach remap (that remap is
tuned to Roboflow's raw label spellings, not a third-party predictor's own
vocabulary). image_filename already matches this pipeline's ground-truth
file_name values exactly, so no normalization is needed to join them.

If json_dir has two files claiming the same image_filename (e.g. two
overlapping export batches), the later one (by sorted path) silently wins --
a warning is printed listing which ones.

Run as a standalone script from the repository root:
    python scripts/aggregate_coralscop.py --json_dir <dir of CoralSCOP *.json files>
"""
import os
import json
import argparse

from utils import _rasterize_coco_mask
from export_coco import COCOExporter

CORALSCOP_CATEGORY_NAME = "coral:unknown"


def load_coralscop_coco(json_dir):
    json_paths = sorted(os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json"))
    if not json_paths:
        raise ValueError(f"No .json files found in {json_dir}")

    path_by_filename = {}
    for path in json_paths:
        with open(path, "r") as f:
            file_name = json.load(f)["image"]["image_filename"]
        if file_name in path_by_filename:
            print(f"WARNING: {file_name} appears in both "
                  f"{os.path.basename(path_by_filename[file_name])} and {os.path.basename(path)} "
                  f"-- keeping the latter (sorted-path order).")
        path_by_filename[file_name] = path

    exporter = COCOExporter([CORALSCOP_CATEGORY_NAME])
    for file_name, path in sorted(path_by_filename.items()):
        with open(path, "r") as f:
            data = json.load(f)
        height, width = data["image"]["image_height"], data["image"]["image_width"]
        img_id = exporter.add_image(file_name, height, width)
        for ann in data["annotations"]:
            exporter.add_annotation(img_id, _rasterize_coco_mask(ann["segmentation"], height, width),
                                     CORALSCOP_CATEGORY_NAME)

    print(f"Loaded {len(exporter.images)} images / {len(exporter.annotations)} annotations from {json_dir}")
    return {"images": exporter.images, "annotations": exporter.annotations, "categories": exporter.categories}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate CoralSCOP's per-image jsons into one COCO file")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory of CoralSCOP's per-image *.json files")
    parser.add_argument("--output", type=str, default="data/performance/coralscop.coco.json")
    args = parser.parse_args()

    with open(args.output, "w") as f:
        json.dump(load_coralscop_coco(args.json_dir), f)
    print(f"Wrote {args.output}")
