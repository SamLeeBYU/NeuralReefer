"""
Reproduces the paper's full-dataset inference results: runs the trained
pipeline (the full ensemble, plus config.ABLATION_SUBMODEL's single-CNN
ablation, if set) over every image in TRAIN_DIR -- the same 552 annotated
Yellowfin images used throughout the paper (yellowfin_segment.v18i.coco-
segmentation/train, mirrored at TRAIN_DIR), split 441 train / 111 test --
producing the same per-image statistics and COCO exports
scripts/main.py's `--mode inference` CLI would, then partitions everything
into train/test subsets using the authoritative split recorded in
data/metadata/train_test_split_metadata.csv (the same split scripts/train.py
used to build the paper's held-out test set).

Requires (already-trained artifacts, produced by train.py/retrain_ensemble.py):
  - FILTER_MODELS_DIR/model_1.pth .. model_M.pth and ensemble.npz
  - The SAM2 checkpoint/config at SAM2_CHECKPOINT_PATH/SAM2_CONFIG_PATH
  - data/metadata/train_test_split_metadata.csv (filename/image_id -> split)
  - config.SAVE_COCO = True (to get the COCO exports this script splits)
  - TRAIN_DIR/_annotations.coco.json (ground truth, for the IoU/Dice scoring below)

Produces, under TRAIN_DIR/inference/ (all pre-split files come from
main.inference(); this script adds the *_train/*_test partitions, the
`split` column, and the *_metrics.csv IoU/Dice scoring):
  - inference_statistics.csv                                  (whole dataset, ensemble)
  - inference_statistics_train.csv / _test.csv                (split by metadata)
  - annotations_coco.json                                     (whole dataset, ensemble)
  - annotations_coco_train.json / _test.json                  (split by metadata)
  - annotations_coco_metrics.csv                              (whole dataset, ensemble; IoU/Dice
                                                                 vs ground truth -- utils.compute_
                                                                 coco_taxonomy_metrics, all three
                                                                 taxonomy modes: lcc/bleached/genus)
  - annotations_coco_train_metrics.csv / _test_metrics.csv    (same, per split)
  - inference_statistics_ablation_submodel{N}.csv[...]        (same, if ABLATION_SUBMODEL is set)
  - annotations_coco_ablation_submodel{N}.json[...]           (same, if ABLATION_SUBMODEL is set)
  - annotations_coco_ablation_submodel{N}_metrics.csv[...]    (same, if ABLATION_SUBMODEL is set)

This does NOT retrain anything -- it only runs inference with whatever is
currently in FILTER_MODELS_DIR, using whatever config.ABLATION_SUBMODEL is
set to (None skips the ablation output entirely, at zero extra cost).

WARNING: this runs SAM2 + the full CNN ensemble over all 552 images -- the
paper reports ~5s/image on a GPU; expect substantially longer on CPU/laptop
hardware. Supports resuming a partial run (see main.inference's docstring).
Run as a standalone script from the repository root:
    python scripts/replicate.py
"""
import os

# Force GPU/cuDNN determinism, to rule it out as a source of run-to-run drift
# on top of the fixed seeds elsewhere in the pipeline (see transforms.seeded_rng,
# config.MASK_TRANSFORM_AUGMENT_SEED). Without this, cuDNN is free to pick
# convolution/attention algorithms whose reduction order isn't bit-stable
# across runs, even on the same machine/GPU with identical seeds -- SAM2 is
# conv/attention-heavy, so this is the most likely place for that to bite.
# Must be set before any CUDA context is created, so this has to happen before
# torch (imported transitively below via `main`) touches the GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json

import pandas as pd
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# warn_only=True: SAM2 is third-party and may call an op with no deterministic
# CUDA implementation -- warn and fall back rather than hard-crashing the run.
torch.use_deterministic_algorithms(True, warn_only=True)

from config import TRAIN_DIR, ABLATION_SUBMODEL, SAVE_COCO
from main import inference
from utils import compute_coco_taxonomy_metrics

TRAIN_TEST_SPLIT_METADATA = "data/metadata/train_test_split_metadata.csv"
OUTPUT_DIR = os.path.join(TRAIN_DIR, "inference")
# Ground truth COCO annotations -- same path train.py's CoralSegmenter loads
# (see train.py's `annotation_path=f"{TRAIN_DIR}/_annotations.coco.json"`).
GT_COCO = os.path.join(TRAIN_DIR, "_annotations.coco.json")


def split_coco_by_metadata(coco_path, filename_to_split):
    """
    Partitions a COCO json (as written by export_coco.COCOExporter) into one
    new COCO json per distinct split value filename_to_split maps to (here,
    "train"/"test") -- filtering `images` by file_name and `annotations` by
    the image ids that survive, keeping `categories` shared/unchanged.
    Writes "<coco_path minus .json>_<split>.json" for each split value
    present. Returns {split_value: output_path}.
    """
    with open(coco_path) as f:
        coco = json.load(f)

    root, ext = os.path.splitext(coco_path)
    unmapped = sorted({img["file_name"] for img in coco["images"] if img["file_name"] not in filename_to_split})
    if unmapped:
        print(f"WARNING: {len(unmapped)} image(s) in {coco_path} have no entry in "
              f"{TRAIN_TEST_SPLIT_METADATA} and will be excluded from every split "
              f"(e.g. {unmapped[:3]})")

    written = {}
    for split_value in sorted(set(filename_to_split.values())):
        kept_images = [img for img in coco["images"] if filename_to_split.get(img["file_name"]) == split_value]
        kept_ids = {img["id"] for img in kept_images}
        kept_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in kept_ids]

        out_path = f"{root}_{split_value}{ext}"
        with open(out_path, "w") as f:
            json.dump({
                "images": kept_images,
                "annotations": kept_annotations,
                "categories": coco["categories"],
            }, f, indent=2)
        written[split_value] = out_path
        print(f"Wrote {out_path} ({len(kept_images)} images, {len(kept_annotations)} annotations)")

    return written


def evaluate_coco_predictions(coco_path, gt_coco_path=GT_COCO):
    """
    Scores one predicted COCO json (produced by main.inference/
    export_coco.COCOExporter, or a split_coco_by_metadata output) against
    the ground-truth COCO annotations, across all three taxonomy modes
    utils.compute_coco_taxonomy_metrics supports ("lcc", "bleached",
    "genus"). Only images present in BOTH files are scored (so calling this
    on an already train/test-split predicted json naturally restricts
    evaluation to that split -- no need to split gt_coco_path separately).

    This is decoupled from the live pipeline objects on purpose (see
    utils.py's module-level note), so the exact same call will later score
    a third-party predictor's COCO export (e.g. CoralSCOP) once it's
    adapted to this pipeline's category convention.

    Writes "<coco_path minus .json>_metrics.csv" (one row per image[,
    genus]) and returns it as a DataFrame.
    """
    metrics = pd.concat(
        [compute_coco_taxonomy_metrics(coco_path, gt_coco_path, taxonomy=mode)
         for mode in ("lcc", "bleached", "genus")],
        ignore_index=True,
    )
    root, ext = os.path.splitext(coco_path)
    out_path = f"{root}_metrics.csv"
    metrics.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(metrics)} rows)")
    return metrics


def join_split_metadata(stats_csv_path, split_df):
    """
    Merges the train/test split label from train_test_split_metadata.csv
    onto a stats CSV, then writes "<stats_csv_path minus .csv>_<split>.csv"
    for each split value present (mirroring split_coco_by_metadata's
    naming), in addition to overwriting stats_csv_path itself with the added
    `split` column for anyone who wants the unsplit file.

    Joins on the actual image FILENAME (derived from the stats CSV's
    `image` column), not `image_id` -- some distinct images in this dataset
    share the same image_id, so an image_id join would fan out those rows.
    filename is unique across all 552 images.

    Idempotent -- safe to call again on a stats_csv_path this already wrote.
    The stale `split` column is dropped before merging in a fresh one, so
    pandas doesn't rename both to `split_x`/`split_y`.
    """
    stats = pd.read_csv(stats_csv_path)
    stats["filename"] = stats["image"].apply(lambda p: os.path.basename(str(p).replace("\\", "/")))
    stats = stats.drop(columns=["split"], errors="ignore")
    merged = pd.merge(stats, split_df[["filename", "split"]], on="filename", how="left")
    unmapped = int(merged["split"].isna().sum())
    if unmapped:
        print(f"WARNING: {unmapped} row(s) in {stats_csv_path} did not match any "
              f"filename in {TRAIN_TEST_SPLIT_METADATA}")
    merged.to_csv(stats_csv_path, index=False)
    print(f"Joined split labels onto {stats_csv_path} ({len(merged)} rows)")

    root, ext = os.path.splitext(stats_csv_path)
    for split_value in sorted(merged["split"].dropna().unique()):
        out_path = f"{root}_{split_value}{ext}"
        merged[merged["split"] == split_value].to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({(merged['split'] == split_value).sum()} rows)")


def main():
    if not SAVE_COCO:
        print("WARNING: config.SAVE_COCO is False -- no COCO json will be produced "
              "for this script to split. Set SAVE_COCO = True in config.py to get it.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(OUTPUT_DIR, "inference_statistics.csv")
    print(f"Running inference over every image in {TRAIN_DIR} ...")
    inference(TRAIN_DIR, output_file, COCO_output_dir=OUTPUT_DIR, annotation_path=GT_COCO)

    split_meta = pd.read_csv(TRAIN_TEST_SPLIT_METADATA)
    filename_to_split = dict(zip(split_meta["filename"], split_meta["split"]))

    stats_paths = [output_file]
    coco_paths = [os.path.join(OUTPUT_DIR, "annotations_coco.json")]
    if ABLATION_SUBMODEL is not None:
        stats_paths.append(os.path.join(OUTPUT_DIR, f"inference_statistics_ablation_submodel{ABLATION_SUBMODEL}.csv"))
        coco_paths.append(os.path.join(OUTPUT_DIR, f"annotations_coco_ablation_submodel{ABLATION_SUBMODEL}.json"))

    for path in stats_paths:
        join_split_metadata(path, split_meta)

    if SAVE_COCO:
        for path in coco_paths:
            written = split_coco_by_metadata(path, filename_to_split)

            print(f"Scoring {path} against {GT_COCO} (IoU/Dice, all taxonomy modes) ...")
            evaluate_coco_predictions(path)
            for split_path in written.values():
                evaluate_coco_predictions(split_path)


if __name__ == "__main__":
    main()
