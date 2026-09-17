"""
Regenerates BOTH the predicted-mask overlay (pred.png) and the ground-truth
overlay (gt.png) for the paper's Figure 9 qualitative examples, using the
CURRENT trained model artifacts (the existing figures/paper/examples/<ID>/
pred.png files predate the Sep-2025 classifier retrain reflected in Table 4/
confusion matrices) and a single shared, seeded color map. og.jpg (raw
photo) is untouched.

Also writes a single shared legend image (figures/paper/examples/legend.png)
mapping color -> coral class, so Figure 9 in the paper can show one legend
for all four panels instead of a separate one per panel (reviewer request
for a clearer/simpler legend).

CoralSegmenter._create_color_map (segmenter.py) assigns colors via
np.random.rand with NO fixed seed, so the mapping is only reproducible
within a single process/run -- this script seeds numpy once, builds ONE
CoralSegmenter (and therefore one segmenter.color_map), and reuses that
same color_map object for pred.png, gt.png, and the legend for every example
image, so colors are guaranteed consistent across all of them (previously
gt.png was rendered in a separate, unseeded run, so its colors did not match
pred.png's).

After rendering, the script copies each image's og.jpg/gt.png/pred.png into
paper/ as fig9{a,b,c,d}_{og,gt,pred}.{jpg,png} (matching the order of
EXAMPLE_IMAGES), and writes paper/fig9_metrics.tex with per-row lcc IoU,
Dice, Precision, and Recall macros for use in predictions.tex.

Run as a standalone script from the repository root:
    python scripts/render_examples.py
"""
import csv
import json
import os
import shutil

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import SAM2_CONFIG_PATH, SAM2_CHECKPOINT_PATH, FILTER_MODELS_DIR, M, TRAIN_DIR
from filter import CoralFilterEnsembler
from segmenter import CoralSegmenter

SEED = 42

EXAMPLES_DIR = "figures/paper/examples"
PAPER_DIR = "paper"
ANNOTATION_PATH = os.path.join(TRAIN_DIR, "_annotations.coco.json")

# Order determines a/b/c/d mapping in Figure 9.
# GPAB5827/GPAB5497: high-IoU (good) examples.
# GPAB2752/GPAB3067: low-IoU failure cases added at reviewer request.
EXAMPLE_IMAGES = {
    "GPAB5827": "yellowfin_segment.v18i.coco-segmentation/train/GPAB5827_jpg.rf.7cc4d07d38d46d7c26b93ba7539d1662.jpg",
    "GPAB5497": "yellowfin_segment.v18i.coco-segmentation/train/GPAB5497_jpg.rf.0ffa6911ae7ba47037807745fe6d0550.jpg",
    "GPAB2752": "yellowfin_segment.v18i.coco-segmentation/train/GPAB2752_jpg.rf.3d3a23e7d01325a3e36b5203981b2c6b.jpg",
    "GPAB3067": "yellowfin_segment.v18i.coco-segmentation/train/GPAB3067_jpg.rf.787b1d28ce4fe02d8dcdc53034e25912.jpg",
}

# lcc metrics CSVs (test split preferred; fall back to train split)
_METRICS_CSVS = [
    "yellowfin_segment.v18i.coco-segmentation/train/inference/annotations_coco_test_metrics.csv",
    "yellowfin_segment.v18i.coco-segmentation/train/inference/annotations_coco_train_metrics.csv",
]


def _load_lcc_metrics():
    """Returns {image_id: {iou, dice, precision, recall}} from the metrics CSVs,
    keyed by the GPAB base ID (e.g. 'GPAB5827')."""
    seen = {}
    for csv_path in _METRICS_CSVS:
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["taxonomy"] != "lcc":
                    continue
                base_id = row["image"].split("_jpg")[0]
                if base_id not in seen:
                    seen[base_id] = {
                        "iou":       float(row["iou"]),
                        "dice":      float(row["dice_f1"]),
                        "precision": float(row["precision"]) if row["precision"] else float("nan"),
                        "recall":    float(row["recall"])    if row["recall"]    else float("nan"),
                    }
    return seen


def pretty_label(label):
    if label == "noncoral":
        return None  # never shown in the legend -- rejected before final output
    genus, status = label.split(":")
    genus = genus.rstrip("_").replace("_", " ").title()
    return f"{genus} ({status.title()})"


def build_legend(color_map, save_path):
    """One shared legend for Figure 9: a genus x bleach-status color key,
    genera in a fixed order, bleached/healthy side by side."""
    genera = sorted({label.split(":")[0] for label in color_map if ":" in label})
    handles, labels = [], []
    for genus in genera:
        for status in ("healthy", "bleached"):
            key = f"{genus}:{status}"
            if key not in color_map:
                continue
            handles.append(mpatches.Patch(facecolor=color_map[key][:3], edgecolor="black", alpha=color_map[key][3]))
            labels.append(pretty_label(key))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.legend(handles, labels, loc="center", ncol=4, frameon=False, fontsize=11)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Wrote {save_path}")


def main():
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    coral_filter = CoralFilterEnsembler(base_dataset=None, m=M, device=device)
    coral_filter.load_models(FILTER_MODELS_DIR)

    segmenter = CoralSegmenter(
        config_path=SAM2_CONFIG_PATH,
        checkpoint_path=SAM2_CHECKPOINT_PATH,
        coral_filter=coral_filter,
        annotation_path=ANNOTATION_PATH,
        device=device,
    )

    models_initialized = False
    for image_id, img_path in EXAMPLE_IMAGES.items():
        example_dir = os.path.join(EXAMPLES_DIR, image_id)
        pred_save_path = os.path.join(example_dir, "pred.png")
        gt_save_path   = os.path.join(example_dir, "gt.png")

        if os.path.exists(pred_save_path) and os.path.exists(gt_save_path):
            print(f"Skipping {image_id}: pred.png and gt.png already exist")
            continue

        print(f"Running inference on {image_id} ({img_path})...")
        masks, labels = segmenter.predict(img_path=img_path, init_models=(not models_initialized), verbose=False)
        models_initialized = True
        pred_labels = segmenter.coral_filter.get_class_names(labels, segmenter.coral_filter.classes)

        os.makedirs(example_dir, exist_ok=True)
        segmenter.show_masks(masks, segmenter.color_map, pred_labels, show=False, save_path=pred_save_path, show_labels=True)
        print(f"Wrote {pred_save_path} ({len(masks)} masks)")

        # segmenter.image is still the same resized/augmented image predict()
        # just loaded, so the GT overlay lines up with the pred overlay above.
        genus_labels, bleached_labels, gt_masks = segmenter.get_gt_masks(img_path)
        if gt_masks.size == 0:
            print(f"WARNING: no ground-truth masks found for {image_id}; gt.png not regenerated")
            continue

        gt_labels = np.char.add(
            genus_labels.astype(str),
            np.where(bleached_labels == 1, ":bleached", ":healthy"),
        )
        segmenter.show_masks(gt_masks, segmenter.color_map, gt_labels, show=False, save_path=gt_save_path, show_labels=True)
        print(f"Wrote {gt_save_path} ({len(gt_masks)} masks)")

    build_legend(segmenter.color_map, os.path.join(EXAMPLES_DIR, "legend.png"))

    _publish_to_paper()


def _publish_to_paper():
    """Copies per-ID files into paper/ as fig9{a,b,c,d}_* and writes
    paper/fig9_metrics.tex with per-row lcc stats macros."""
    letters = list("abcd")
    lcc = _load_lcc_metrics()

    tex_lines = [
        "% Auto-generated by scripts/render_examples.py -- do not edit by hand.",
        "% Per-row lcc IoU/Dice/Precision/Recall for Figure 9 (predictions.tex).",
    ]

    for letter, (image_id, img_path) in zip(letters, EXAMPLE_IMAGES.items()):
        src_dir = os.path.join(EXAMPLES_DIR, image_id)

        # og.jpg
        og_src = img_path  # raw source image (never rewritten by this script)
        og_dst = os.path.join(PAPER_DIR, f"fig9{letter}_og.jpg")
        shutil.copy2(og_src, og_dst)
        print(f"Copied {og_src} -> {og_dst}")

        # gt.png / pred.png
        for suffix in ("gt.png", "pred.png"):
            src = os.path.join(src_dir, suffix)
            dst = os.path.join(PAPER_DIR, f"fig9{letter}_{suffix.replace('.png', '')}.png")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {src} -> {dst}")
            else:
                print(f"WARNING: {src} not found; skipping")

        # metrics
        m = lcc.get(image_id, {})
        iou  = f"{m.get('iou',  float('nan')):.2f}"
        dice = f"{m.get('dice', float('nan')):.2f}"
        prec = f"{m.get('precision', float('nan')):.2f}"
        rec  = f"{m.get('recall',    float('nan')):.2f}"
        print(f"{image_id} lcc metrics: IoU={iou}  Dice={dice}  Prec={prec}  Rec={rec}")
        tex_lines += [
            f"\\def\\figIX{letter}IoU{{{iou}}}",
            f"\\def\\figIX{letter}Dice{{{dice}}}",
            f"\\def\\figIX{letter}Prec{{{prec}}}",
            f"\\def\\figIX{letter}Rec{{{rec}}}",
        ]

    # legend
    legend_src = os.path.join(EXAMPLES_DIR, "legend.png")
    legend_dst = os.path.join(PAPER_DIR, "fig9_legend.png")
    if os.path.exists(legend_src):
        shutil.copy2(legend_src, legend_dst)
        print(f"Copied {legend_src} -> {legend_dst}")

    metrics_tex = os.path.join(PAPER_DIR, "fig9_metrics.tex")
    with open(metrics_tex, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Wrote {metrics_tex}")


if __name__ == "__main__":
    main()
