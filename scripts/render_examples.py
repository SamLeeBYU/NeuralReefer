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

Run as a standalone script from the repository root:
    python scripts/render_examples.py
"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import SAM2_CONFIG_PATH, SAM2_CHECKPOINT_PATH, FILTER_MODELS_DIR, M, TRAIN_DIR
from filter import CoralFilterEnsembler
from segmenter import CoralSegmenter

SEED = 42

EXAMPLES_DIR = "figures/paper/examples"
ANNOTATION_PATH = os.path.join(TRAIN_DIR, "_annotations.coco.json")
EXAMPLE_IMAGES = {
    "GPAB5497": "yellowfin_segment.v18i.coco-segmentation/train/GPAB5497_jpg.rf.0ffa6911ae7ba47037807745fe6d0550.jpg",
    "GPAB5827": "yellowfin_segment.v18i.coco-segmentation/train/GPAB5827_jpg.rf.7cc4d07d38d46d7c26b93ba7539d1662.jpg",
    "GPAB6428": "yellowfin_segment.v18i.coco-segmentation/train/GPAB6428_jpg.rf.b16d2b18243733f492c9084c55db4346.jpg",
    "GPAB7050": "yellowfin_segment.v18i.coco-segmentation/train/GPAB7050_jpg.rf.5d195395bfbe8803ccd8968785d9d541.jpg",
}


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
        print(f"Running inference on {image_id} ({img_path})...")
        masks, labels = segmenter.predict(img_path=img_path, init_models=(not models_initialized), verbose=False)
        models_initialized = True
        pred_labels = segmenter.coral_filter.get_class_names(labels, segmenter.coral_filter.classes)

        pred_save_path = os.path.join(EXAMPLES_DIR, image_id, "pred.png")
        os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
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
        gt_save_path = os.path.join(EXAMPLES_DIR, image_id, "gt.png")
        segmenter.show_masks(gt_masks, segmenter.color_map, gt_labels, show=False, save_path=gt_save_path, show_labels=True)
        print(f"Wrote {gt_save_path} ({len(gt_masks)} masks)")

    build_legend(segmenter.color_map, os.path.join(EXAMPLES_DIR, "legend.png"))


if __name__ == "__main__":
    main()
