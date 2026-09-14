import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import pycocotools.mask as mask_utils

from time import perf_counter  # Runtime measurement
from contextlib import contextmanager

import warnings           # Suppress non-critical runtime warnings
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")

from config import REMAP_PATH, CLASSES_FILE

#For verbose functionality
def suppress_prints():
    """Redirects stdout to null to suppress print statements."""
    sys.stdout = open(os.devnull, 'w')

def restore_prints():
    """Restores normal stdout printing."""
    sys.stdout = sys.__stdout__

#The following method was written by ChatGPT 4o
#This helps convert python dictionaries to json-compatible objects
def convert_json_compat(obj):
    """
    Recursively converts numpy datatypes to native Python types
    so that they can be safely serialized to JSON.
    """
    if isinstance(obj, dict):
        return {k: convert_json_compat(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_compat(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_json_compat(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

@contextmanager
def timer(verbose=True):
    start = perf_counter()
    try:
        yield
    finally:
        if verbose:
            print(f"Time: {perf_counter() - start:2f}s")

# COCO-vs-COCO evaluation ############################################################################
# Decoupled from CoralSegmenter/CoralFilterEnsembler on purpose: these functions take only two COCO
# json structures (a prediction and a ground truth) and never touch the live model objects, so they
# can score ANY predictor's COCO export against ground truth -- not just this pipeline's own -- e.g.
# comparing against CoralSCOP once its output is adapted to COCO format. See train.py's
# pixel_confusion_metrics/union_mask/compute_pred_stats for the (model-coupled) equivalents this
# mirrors; the formulas are intentionally duplicated rather than imported to keep utils.py free of
# train.py's heavier dependencies (torch, skopt, ...), since several modules already import FROM
# utils.py and importing train.py here would risk a circular import.

def _load_coco(coco):
    """`coco` may be a path to a COCO json file or an already-loaded dict; either way, returns the dict."""
    if isinstance(coco, dict):
        return coco
    with open(coco, "r") as f:
        return json.load(f)


def _rasterize_coco_mask(segmentation, height, width):
    """Rasterizes one COCO annotation's `segmentation` into a single boolean
    mask of shape (height, width). Supports both:
      - polygon format: a list of flat [x1, y1, x2, y2, ...] point lists
        (possibly several parts for one annotation) -- this pipeline's own
        convention (export_coco.COCOExporter, Roboflow ground truth) --
        rasterized via cv2.fillPoly, matching segmenter.py's
        CoralSegmenter.get_gt_masks (generalized from that method's
        hardcoded 1024x1024 to each image's own recorded height/width).
      - RLE format: a dict with "size"/"counts" (pycocotools mask
        encoding) -- used by third-party predictors such as CoralSCOP --
        decoded via pycocotools.mask.decode.
    """
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle) > 0

    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask > 0


def _canonical_taxonomy_label(category_name, remap_dic):
    """
    Normalizes a category name to this pipeline's canonical
    "genus:bleached"/"genus:healthy"/"noncoral" form, matching
    CoralSegmenter.parse_annotations's convention: raw Roboflow ground-truth
    categories use a "_bleach" suffix and inconsistent genus spellings/
    synonyms, resolved via data/remap.json.

    A name already in canonical form (as produced by this pipeline's own
    export_coco.COCOExporter, or a third-party predictor already adapted to
    the convention) round-trips unchanged.
    """
    if category_name == "noncoral" or ":" in category_name:
        return category_name
    is_bleached = "bleach" in category_name.lower()
    cleaned = category_name.replace("_bleach", "")
    genus = remap_dic.get(cleaned, cleaned)
    if genus == "noncoral":
        return "noncoral"
    return f"{genus}:{'bleached' if is_bleached else 'healthy'}"


def _pixel_confusion_metrics(true_mask, pred_mask):
    """Pixel-level IoU/Dice(=F1)/precision/recall between two boolean masks --
    same formula as train.py's pixel_confusion_metrics (duplicated, see the
    module-level note above). precision/recall are NaN (not 0) when their
    denominator is zero; iou/dice are 1.0 when both masks are empty."""
    tp = int(np.logical_and(true_mask, pred_mask).sum())
    fp = int(np.logical_and(~true_mask, pred_mask).sum())
    fn = int(np.logical_and(true_mask, ~pred_mask).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return tp, fp, fn, iou, dice, precision, recall


def _annotations_by_file(coco, remap_dic):
    """Indexes one COCO structure's annotations by image file_name, with each
    annotation's category resolved to its canonical taxonomy label. Returns
    {file_name: (height, width, [(label, segmentation), ...])}."""
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image_id = {}
    for ann in coco["annotations"]:
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    result = {}
    for img in coco["images"]:
        anns = anns_by_image_id.get(img["id"], [])
        result[img["file_name"]] = (
            img["height"], img["width"],
            [(_canonical_taxonomy_label(categories[ann["category_id"]], remap_dic), ann["segmentation"])
             for ann in anns],
        )
    return result


def _taxonomy_union_mask(entries, height, width, predicate):
    """OR's together every rasterized annotation mask in `entries` whose
    canonical label satisfies `predicate` into one (height, width) mask."""
    mask = np.zeros((height, width), dtype=bool)
    for label, segmentation in entries:
        if predicate(label):
            mask |= _rasterize_coco_mask(segmentation, height, width)
    return mask


def compute_coco_taxonomy_metrics(pred_coco, gt_coco, taxonomy="lcc", remap_path=REMAP_PATH,
                                   classes_file=CLASSES_FILE, genus_names=None):
    """
    Computes pixel-level IoU/Dice/precision/recall between a predicted COCO
    annotation set and a ground-truth COCO annotation set, matching train.py
    eval()'s per-image taxonomy metrics, but decoupled from the live
    inference pipeline: it operates purely on two COCO structures (paths or
    already-loaded dicts), so it can score ANY predictor's COCO export
    against ground truth (e.g. CoralSCOP), not just this pipeline's own.

    Args:
        pred_coco: path to a predicted COCO json, or an already-loaded dict.
            Category names are assumed to already be in this pipeline's
            canonical "genus:bleached"/"genus:healthy"/"noncoral" form (as
            produced by export_coco.COCOExporter) -- adapt a third-party
            predictor's category names to that convention before calling this.
        gt_coco: path to the ground-truth COCO json (Roboflow export, raw
            genus/"_bleach" category names), or an already-loaded dict.
            Normalized to the same canonical form via remap_path + the
            "_bleach" suffix convention (matching
            CoralSegmenter.parse_annotations).
        taxonomy (str): one of:
            "lcc"      -- all coral (any genus/bleach status) vs noncoral,
                          one row per image (matches train.py's "all_coral").
            "bleached" -- bleached coral (any genus) vs everything else,
                          one row per image.
            "genus"    -- coral cover per genus (bleached + healthy
                          combined), one row per (image, genus).
            "genus_bleach" -- coral cover per (genus, bleach status) pair,
                          one row per (image, genus, bleach status), labeled
                          "<genus>:bleached"/"<genus>:healthy" -- the finer
                          cross-tabulation "genus" collapses, needed for a
                          bias decomposition by both genus AND bleach status
                          (e.g. the paper's Table 5).
        remap_path: path to the raw->genus remap json (default config.REMAP_PATH).
        classes_file: path to the canonical class dictionary (default
            config.CLASSES_FILE) used to derive the fixed genus list for
            taxonomy="genus", matching train.py's own genus_names (every
            genus known to the pipeline gets a row per image, regardless of
            whether it appears in that image) -- ignored if genus_names is
            given explicitly.
        genus_names: optional explicit list of genus names to score for
            taxonomy="genus", overriding the derivation from classes_file
            (e.g. if scoring a predictor that uses a different taxonomy).

    Returns:
        pandas.DataFrame with one row per (image[, genus]): image, taxonomy,
        tp_px, fp_px, fn_px, iou, dice_f1, precision, recall -- the same
        schema as train.py's taxonomy_records (minus image_id, since COCO
        structures are matched by file_name here, not this pipeline's
        image_id convention).

    Images present in only one of pred/gt are skipped (with a printed
    warning) rather than scored against an empty mask.
    """
    if taxonomy not in ("lcc", "bleached", "genus", "genus_bleach"):
        raise ValueError(f"taxonomy must be one of 'lcc', 'bleached', 'genus', 'genus_bleach' -- got {taxonomy!r}")

    pred = _load_coco(pred_coco)
    gt = _load_coco(gt_coco)

    with open(remap_path, "r") as f:
        remap_dic = json.load(f)

    pred_by_file = _annotations_by_file(pred, remap_dic)
    gt_by_file = _annotations_by_file(gt, remap_dic)

    common_files = sorted(set(pred_by_file) & set(gt_by_file))
    missing = sorted(set(pred_by_file) ^ set(gt_by_file))
    if missing:
        print(f"WARNING: {len(missing)} image(s) present in only one of pred/gt COCO structures "
              f"and will be skipped (e.g. {missing[:3]})")

    if taxonomy in ("genus", "genus_bleach"):
        if genus_names is None:
            with open(classes_file, "r") as f:
                classes = json.load(f)
            genus_names = sorted(set(
                k.split(":")[0] for k in classes if k.endswith(":bleached") or k.endswith(":healthy")
            ))
        if taxonomy == "genus":
            predicates = {genus: (lambda label, g=genus: label.startswith(f"{g}:")) for genus in genus_names}
        else:
            predicates = {}
            for genus in genus_names:
                predicates[f"{genus}:bleached"] = (lambda label, g=genus: label == f"{g}:bleached")
                predicates[f"{genus}:healthy"] = (lambda label, g=genus: label == f"{g}:healthy")
    elif taxonomy == "lcc":
        predicates = {"lcc": lambda label: label != "noncoral"}
    else:
        predicates = {"bleached": lambda label: label.endswith(":bleached")}

    records = []
    for file_name in common_files:
        gt_h, gt_w, gt_entries = gt_by_file[file_name]
        pred_h, pred_w, pred_entries = pred_by_file[file_name]

        for row_label, predicate in predicates.items():
            true_mask = _taxonomy_union_mask(gt_entries, gt_h, gt_w, predicate)
            pred_mask = _taxonomy_union_mask(pred_entries, pred_h, pred_w, predicate)
            if (pred_h, pred_w) != (gt_h, gt_w):
                # Puts the predicted mask on the ground-truth's pixel grid --
                # a no-op for this pipeline's own predictions (which always
                # match gt's IMG_SIZE), but lets a third-party predictor at a
                # different resolution still be compared.
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_w, gt_h),
                                        interpolation=cv2.INTER_NEAREST) > 0

            tp, fp, fn, iou, dice, precision, recall = _pixel_confusion_metrics(true_mask, pred_mask)
            records.append({
                "image": file_name, "taxonomy": row_label,
                "tp_px": tp, "fp_px": fp, "fn_px": fn,
                "iou": iou, "dice_f1": dice, "precision": precision, "recall": recall,
            })

    return pd.DataFrame(records)