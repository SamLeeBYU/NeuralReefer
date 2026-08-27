# Neural Reefer: Modular Coral Reef Segmentation and Classification from Top-Down Imagery

This repository implements an end-to-end deep learning pipeline for analyzing top-down coral reef imagery. It combines high-recall instance segmentation (SAM2), CNN-based filtering and multi-task classification (genus, bleaching, etc.) into a modular system tailored for ecological monitoring. Developed June-July 2025 at WHOI using Majuro reef data, Neural Reefer supports scalable estimation of live coral cover (LCC) and taxonomic/health classification from top-down imagery surveys.

---

## 🔧 Installation

Neural Reefer requires **Python 3.9+**. All commands below (and all commands in this README) must be run from the **repository root** — paths in `scripts/config.py` are relative to it, and will fail with `FileNotFoundError` if run from inside `scripts/`.

1. **Clone this repository** and `cd` into it:
   ```bash
   git clone https://github.com/your-repo/NeuralReefer.git
   cd NeuralReefer
   ```

2. **Create a virtual environment** (recommended) and install the Python dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate

   pip install -r requirements.txt
   ```
   This installs `torch`, `torchvision`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`,
   `scikit-optimize`, `Pillow`, `matplotlib`, `seaborn`, `tqdm`, `supervision`, `shapely`,
   `geopandas`, and `contextily`.
   - If you have an NVIDIA GPU, install a CUDA-enabled build of `torch`/`torchvision` **first**
     by following the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
     then run `pip install -r requirements.txt` — the CPU wheel installed by that command will
     be skipped since a compatible version is already present. Otherwise the default CPU-only
     build is installed automatically and inference/training will just run on CPU (slower, but
     functional).

3. **Clone and install SAM2** (the segmentation backbone — a separate repository, not on PyPI):
   ```bash
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2 && pip install -e . && cd ..
   ```
   Then download the pretrained checkpoint (from inside the `sam2` directory):
   ```bash
   cd sam2/checkpoints && ./download_ckpts.sh && cd ../..
   ```

4. **Point `scripts/config.py` at your local SAM2 checkout** — edit these two lines to the
   full path where you cloned it in step 3:
   ```python
   SAM2_PATH = "/path/to/sam2"  # Replace with the full path
   SAM2_CHECKPOINT_PATH = f"{SAM2_PATH}/checkpoints/sam2.1_hiera_large.pt"
   ```

5. **Run inference** — pretrained coral filter models, class labels, and tuned SAM2
   hyperparameters are already committed under `models/` and `data/segmentation/`, so no
   training is required to get started. Try it on the bundled example image:
   ```bash
   python scripts/example.py
   ```
   or point it at your own folder of images:
   ```bash
   python scripts/main.py --mode inference --image_dir path/to/images
   ```
   Results are written to `<image_dir>/inference/inference_statistics.csv` (per-image coral
   cover / bleaching / genus breakdown), with predicted mask overlays saved alongside as
   `<image_dir>/inference/<image_id>.png` and predicted masks in COCO format at
   `<image_dir>/annotations_coco.json`. For the bundled example this is
   `data/examples/inference/inference_statistics.csv`.

6. **(Optional) Retrain the pipeline on your own annotated data**, once your COCO
   annotations + images are placed in `data/train` (see `TRAIN_DIR`/`EXT` in `config.py`).
   Enable the stages you need in `config.py`:
   ```python
   TUNE_SEGMENTER = True
   CREATE_MASK_DATASET = True
   TRAIN_CORAL_FILTER = True
   ```
   then run:
   ```bash
   python scripts/main.py --mode train
   ```

---

## Pipeline Overview

**NeuralReefer** is a modular deep learning pipeline for coral reef segmentation and classification, supporting inference from raw RGB imagery to ecologically structured coral cover statistics. The pipeline proceeds in three primary stages:

1. **Segmentation with SAM2**:  
   A dual-stream SAM2 segmentation module identifies candidate coral objects across varying size scales. One stream targets large coral colonies with high precision, while the other prioritizes recall for small, fragmented structures. Hyperparameters for each stream are tuned jointly with preprocessing augmentations using a Monte Carlo optimization strategy. Outputs are merged by confidence-weighted ranking and non-maximum suppression.

2. **Coral Classification**:  
   Each candidate mask is cropped and passed to an ensemble of CNN classifiers (ResNet-34 backbone with MLP head), trained to distinguish between 13 mutually exclusive categories (6 coral genera × 2 bleaching statuses + 1 noncoral). Negative training examples are synthetically generated from false positive masks. Per-epoch augmentations are sampled from a surrogate distribution that emulates natural mask variability. An ensemble optimizer aggregates predictions via softmax-weighted logits using class-balanced focal loss.

3. **Cover Estimation**:  
   Accepted masks are aggregated by pixel area to compute coral cover statistics:
   - Total coral cover
   - Bleached vs. live proportions
   - Genus-specific and health-stratified cover

All modules can be executed independently or chained via `main.py` for end-to-end processing.

---

## 🏋️ Training Procedure

### Stage 1: SAM2 Segmentation

The segmentation stage uses a two-stream SAM2 framework:
- **Large-object stream**: optimized for high-precision identification of large colonies
- **Small-object stream**: optimized for high-recall detection of small or fragmented coral

Hyperparameters and augmentations are jointly optimized via Bayesian optimization (`skopt`) with a recall-weighted pixel-level scoring function. Candidate masks are merged by descending IoU confidence, weighted by classification scores, and pruned using area and overlap constraints.

### Stage 2: CNN Classification

Training data includes:
- **Positive samples**: manually annotated coral masks
- **Negative samples**: low-overlap masks from SAM2 to simulate false positives

Each cropped mask image is resized to $128 \times 128$, and passed to one of $M=5$ CNN classifiers. Each classifier has:
- A ResNet-34 backbone
- A 4-layer MLP head with ReLU, dropout, and softmax
- Augmentation at each epoch sampled from a learned prior over geometric distortions

To combat class imbalance, minority classes are oversampled and augmented. Classifiers are trained with Adam optimizer using categorical cross-entropy.

### Ensemble Aggregation

Instance-level logits are aggregated with a softmax-weighted sum:
```math
z_i = \sum_{m=1}^{M} W_m \odot z_i^{(m)}
```
where $W_m$ are learned per-class weights. The ensemble is trained on a held-out validation set using focal loss with $\gamma = 3.0$ and $\alpha_{\text{noncoral}} = 0.25$. This yields a binary coral detection accuracy of 91%, with 97.1% recall and 89.2% precision.

### Stage 3: Coral Cover Metrics

Final coral cover is computed by summing pixel areas of masks classified as coral. Bleaching severity is quantified as the proportion of coral pixels assigned to bleached classes. These metrics enable ecological summary at both image- and reef-level scales.

---

## 🚀 Usage Examples

All commands are run from the repository root.

### Run inference on the bundled example image

```bash
python scripts/example.py
```

### Run inference on a folder of images

```bash
python scripts/main.py --mode inference --image_dir path/to/images
```

### Train full pipeline (segmentation + filtering + classification)

```bash
python scripts/main.py --mode train
```

### Visualize performance metrics

```bash
python scripts/main.py --mode visualize --version <VERSION>
```

---

## 📞 Contact

- **Sam Lee** — University of Arizona
(Previous affiliation: Brigham Young University, WHOI)
- 📨 samlee@arizona.edu

---

## 🔬 Acknowledgments

This work was developed at Woods Hole Oceanographic Institution (WHOI) with support from:
- Dr. Calvin Quigley, Dr. Nathan Mollica, Dr. Anne Cohen
- WHOI Yellowfin ASV team (data collection)
- WHOI Annotation Team (Evii Tong, Robert Ronan)

Segmentation powered by [SAM2](https://github.com/facebookresearch/sam2).

