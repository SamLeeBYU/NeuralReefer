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
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import SAM2_CONFIG_PATH, SAM2_CHECKPOINT_PATH, FILTER_MODELS_DIR, EXT, VERBOSE, M, SAVE_MASKS, SAVE_COCO, METADATA, ABLATION_SUBMODEL

from utils import suppress_prints, restore_prints
from train import train, load_data, union_mask, pixel_confusion_metrics
from visualize import plot_segmentation_summary, plot_coral_cover

from filter import CoralFilterEnsembler
from segmenter import CoralSegmenter

from export_coco import COCOExporter

V1_PREDICTIONS_CSV = "data/performance/v1.0_iou_specificity.csv"

def _load_v1_predictions(csv_path=V1_PREDICTIONS_CSV):
    """Loads reference per-image LCC predictions/IoU for the live spot-check
    comparison in _live_stats_line, keyed by filename (image_id is not
    unique in this dataset -- see the note in inference() below). dice_lcc
    is derived from the csv's tp/fp/fn_lcc columns. Returns {} if the file
    is missing, silently disabling the comparison."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    df["filename"] = df["image"].apply(lambda p: os.path.basename(str(p).replace("\\", "/")))
    out = {}
    for row in df.itertuples():
        denom = 2 * row.tp_lcc + row.fp_lcc + row.fn_lcc
        out[row.filename] = {
            "coral_cover_pred": row.coral_cover_pred,
            "pct_bleached_pred": row.pct_bleached_pred,
            "iou_lcc": row.iou_lcc,
            "dice_lcc": (2 * row.tp_lcc / denom) if denom > 0 else float("nan"),
        }
    return out

def _with_suffix(path: str, suffix: str) -> str:
    """Inserts `suffix` before a path's extension, e.g. ("a/b.csv", "_x") -> "a/b_x.csv"."""
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"

def _build_stats_record(segmenter, masks, pred_labels, genus_names, img_path, image_id):
    """Every masks/pred_labels-derived statistic inference() records for ONE
    image (coral cover, bleaching %, per-genus cover). Reused for both the
    main ensemble result and the ABLATION_SUBMODEL result. Mirrors
    train.py's compute_pred_stats for the eval() path."""
    cover = segmenter.coral_cover(masks, cs=segmenter.crop_space)
    pct_bleached = segmenter.coral_cover(
        [masks[j] for j in range(len(pred_labels)) if pred_labels[j].endswith(":bleached")],
        cs=segmenter.crop_space
    )

    record = {
        "image": img_path,
        "image_id": image_id,
        "coral_cover_pred": cover,
        "pct_bleached_pred": pct_bleached,
    }
    for genus in genus_names:
        record[f"cover_pred__{genus}"] = segmenter.coral_cover(
            [masks[j] for j in range(len(pred_labels)) if pred_labels[j].startswith(f"{genus}:")],
            cs=segmenter.crop_space
        )
        record[f"cover_healthy_pred__{genus}"] = segmenter.coral_cover(
            [masks[j] for j in range(len(pred_labels)) if pred_labels[j] == f"{genus}:healthy"],
            cs=segmenter.crop_space
        )
    return record

def _live_stats_line(image_id, cover_pred, pctb_pred, gt_stats, v1_stats=None):
    """One live progress line for the image just analyzed: predicted LCC/
    %bleached always; true LCC/%bleached and LCC IoU/Dice when ground truth
    is available (gt_stats is None otherwise); a reference run's LCC/
    %bleached/IoU/Dice for the same image when available (v1_stats is None
    otherwise)."""
    line = f"{image_id:>12} | pred LCC {cover_pred:6.3f} | pred %bleach {pctb_pred:6.3f}"
    if gt_stats is not None:
        line += (f" | true LCC {gt_stats['coral_cover_true']:6.3f}"
                 f" | true %bleach {gt_stats['pct_bleached_true']:6.3f}"
                 f" | LCC IoU {gt_stats['iou_lcc']:6.3f} | LCC Dice {gt_stats['dice_lcc']:6.3f}")
    if v1_stats is not None:
        line += (f" || v1 LCC {v1_stats['coral_cover_pred']:6.3f}"
                 f" | v1 %bleach {v1_stats['pct_bleached_pred']:6.3f}"
                 f" | v1 IoU {v1_stats['iou_lcc']:6.3f} | v1 Dice {v1_stats['dice_lcc']:6.3f}")
    return line


def inference(image_dir: str, output_file: str, COCO_output_dir: str | None = None,
              annotation_path: str | None = None) -> pd.DataFrame:
    """
    Evaluates coral cover for a directory of images using a trained CoralSegmenter.

    Args:
        image_dir (str): Directory containing images to process.
        output_file (str): CSV file where aggregated results are saved.
        COCO_output_dir (str): Directory to save COCO format annotations.
            If None, annotations will be saved in the same directory as images.
        annotation_path (str): Optional ground-truth COCO json (Roboflow
            convention, see CoralSegmenter.parse_annotations). When given,
            each image's live progress line also reports true LCC/%bleached
            and LCC IoU/Dice for any image that has a matching annotation
            (silently omitted for images that don't). Has no effect on what
            gets saved to output_file/COCO_output_dir -- ground truth is only
            used for this live printout, not persisted, since per-taxonomy
            IoU/Dice is already fully recoverable after the fact from the
            saved COCO export (see utils.compute_coco_taxonomy_metrics /
            scripts/replicate.py's evaluate_coco_predictions). None (default)
            for genuinely new, unannotated images.

    Returns:
        pd.DataFrame: DataFrame with image metadata and coral cover breakdown.

    Ablation study (see config.py's ABLATION_SUBMODEL): if set, this ALSO
    records the same statistics (and, if SAVE_COCO, the same COCO export)
    for that one submodel's predictions -- computed from
    segmenter.last_ablation_result, i.e. the SAME predict() call, no extra
    inference pass -- to files suffixed "_ablation_submodel{N}". NOTE: resume
    support (skipping images already in output_file) is keyed off the main
    output_file only; if ABLATION_SUBMODEL is toggled on between runs that
    resume a partially-completed output_file, already-done images won't
    retroactively get an ablation record.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coral_filter = CoralFilterEnsembler(base_dataset=None, m=M, device=device)
    coral_filter.load_models(FILTER_MODELS_DIR)

    segmenter = CoralSegmenter(
        config_path=SAM2_CONFIG_PATH,
        checkpoint_path=SAM2_CHECKPOINT_PATH,
        coral_filter=coral_filter,
        annotation_path=annotation_path,
        device=device
    )

    exporter = COCOExporter(coral_filter.classes)

    run_ablation = ABLATION_SUBMODEL is not None
    exporter_abl = COCOExporter(coral_filter.classes) if run_ablation else None
    output_file_abl = _with_suffix(output_file, f"_ablation_submodel{ABLATION_SUBMODEL}") if run_ablation else None

    output_dir = COCO_output_dir if COCO_output_dir is not None else image_dir
    output_json = f"{output_dir}/annotations_coco.json"
    output_json_abl = f"{output_dir}/annotations_coco_ablation_submodel{ABLATION_SUBMODEL}.json" if run_ablation else None

    v1_predictions = _load_v1_predictions()

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(EXT)]
    genus_names = sorted(set(
        k.split(":")[0] for k in segmenter.coral_filter.classes
        if ":bleached" in k or ":healthy" in k
    ))

    save_dir = Path(f"{image_dir}/inference")
    save_dir.mkdir(parents=True, exist_ok=True)

    images_done = set()
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        # Keyed by filename, not image_id: image_id is not unique in this
        # dataset (two distinct images can share the same id).
        images_done = set(existing["image"].apply(lambda p: os.path.basename(str(p).replace("\\", "/"))))
        results = existing.to_dict("records")
    else:
        results = []

    results_abl = []
    if run_ablation and os.path.exists(output_file_abl):
        results_abl = pd.read_csv(output_file_abl).to_dict("records")

    # Reload any COCO entries a prior (resumed) run already exported, so
    # exporter.save() below doesn't overwrite them with only this run's images.
    if SAVE_COCO and os.path.exists(output_json):
        exporter.load(output_json)
    if SAVE_COCO and run_ablation and os.path.exists(output_json_abl):
        exporter_abl.load(output_json_abl)

    # Whether large_feature_generator/small_feature_generator have been built
    # yet in this process (not the same as "first image in image_paths",
    # since a resumed run skips past already-done images first).
    models_initialized = False

    pbar = tqdm(image_paths)
    try:
        for img_path in pbar:
            image_id = Path(img_path).stem.split('_')[0]

            if os.path.basename(img_path) in images_done:
                continue

            if not VERBOSE: suppress_prints()
            masks, labels = segmenter.predict(img_path=img_path, init_models=(not models_initialized), verbose=VERBOSE)
            models_initialized = True
            if not VERBOSE: restore_prints()

            pred_labels = segmenter.coral_filter.get_class_names(labels, segmenter.coral_filter.classes)

            if SAVE_COCO:
                img = segmenter.load_image(img_path=img_path)
                H, W = img.shape[:2]
                img_id = exporter.add_image(os.path.basename(img_path), H, W)

                for j, mask in enumerate(masks):
                    if j >= len(pred_labels): continue
                    exporter.add_annotation(img_id, mask, pred_labels[j])

            stats_record = _build_stats_record(segmenter, masks, pred_labels, genus_names, img_path, image_id)
            results.append(stats_record)

            # Ground truth (when annotation_path was given and this image has
            # a matching entry), used only for the live progress line below --
            # never persisted, since per-taxonomy IoU/Dice is recoverable
            # from the saved COCO export (see utils.compute_coco_taxonomy_metrics).
            gt_stats = None
            if segmenter.annotations is not None:
                genus_labels, bleach_labels, gt_masks = segmenter.get_gt_masks(img_path)
                if genus_labels is not None:
                    ml_labels = np.char.add(genus_labels, np.where(bleach_labels == 1, ":bleached", ":healthy"))
                    gt_masks = [m for j, m in enumerate(gt_masks) if genus_labels[j] != "noncoral"]
                    gt_labels = [ml_labels[j] for j in range(len(ml_labels)) if genus_labels[j] != "noncoral"]

                    coral_cover_true = segmenter.coral_cover(gt_masks, cs=segmenter.crop_space)
                    pct_bleached_true = segmenter.coral_cover(
                        [gt_masks[j] for j in range(len(gt_labels)) if gt_labels[j].endswith(":bleached")],
                        cs=segmenter.crop_space
                    )
                    _, _, _, iou_lcc, dice_lcc, _, _ = pixel_confusion_metrics(union_mask(gt_masks), union_mask(masks))
                    gt_stats = {
                        "coral_cover_true": coral_cover_true,
                        "pct_bleached_true": pct_bleached_true,
                        "iou_lcc": iou_lcc,
                        "dice_lcc": dice_lcc,
                    }

            v1_stats = v1_predictions.get(os.path.basename(img_path))
            pbar.write(_live_stats_line(image_id, stats_record["coral_cover_pred"],
                                         stats_record["pct_bleached_pred"], gt_stats, v1_stats))

            if run_ablation and segmenter.last_ablation_result is not None:
                masks_abl, labels_abl = segmenter.last_ablation_result
                pred_labels_abl = segmenter.coral_filter.get_class_names(labels_abl, segmenter.coral_filter.classes)

                if SAVE_COCO:
                    img_id_abl = exporter_abl.add_image(os.path.basename(img_path), H, W)
                    for j, mask in enumerate(masks_abl):
                        if j >= len(pred_labels_abl): continue
                        exporter_abl.add_annotation(img_id_abl, mask, pred_labels_abl[j])

                results_abl.append(_build_stats_record(segmenter, masks_abl, pred_labels_abl, genus_names, img_path, image_id))

            if SAVE_MASKS:
                segmenter.show_masks(masks, segmenter.color_map, pred_labels, show=False,
                                        save_path=save_dir / f"{image_id}.png")

    finally:
        pd.DataFrame(results).to_csv(output_file, index=False)
        if run_ablation:
            pd.DataFrame(results_abl).to_csv(output_file_abl, index=False)

        if SAVE_COCO:
            exporter.save(output_json)
            if VERBOSE:
                print(f"Predicted masks saved to {output_json}")

            if run_ablation:
                exporter_abl.save(output_json_abl)
                if VERBOSE:
                    print(f"Ablation (submodel {ABLATION_SUBMODEL}) predicted masks saved to {output_json_abl}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralReefer")
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "visualize", "inference"], help="Execution mode")
    parser.add_argument("--version", type=str, required=False, help="Version identifier for visualizations")
    parser.add_argument("--image_dir", type=str, required=False, help="Directory of images to evaluate")
    parser.add_argument("--prediction_file", type=str, required=False, help="File path of evaluations (generated from mode=inference)")
    parser.add_argument("--COCOJSON_file", type=str, required=False, help="File path of predicted masks (generated from mode=inference)")
   
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "visualize":
        plot_coral_cover(version=args.prediction_file)

    elif args.mode == "inference":
        assert args.image_dir, "Must provide --image_dir"
        output_file = f"{args.image_dir}/inference/inference_statistics.csv"
        predictions_data = inference(args.image_dir, output_file, args.COCOJSON_file)

        if METADATA is not None:
            metadata = load_data(METADATA)
            metadata['image_id'] = metadata['filename'].str.split('.').str[0]
            
            predictions_data = pd.merge(predictions_data, metadata, on='image_id', how='left')

        predictions_data.to_csv(output_file, index=False)
        print(f"Saved coral cover estimates to {output_file}")
