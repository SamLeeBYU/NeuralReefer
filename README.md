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
   `scikit-optimize`, `Pillow`, `matplotlib`, `seaborn`, `tqdm`, `pycocotools`, `supervision`,
   `shapely`, `geopandas`, and `contextily`.
   - `pycocotools` occasionally fails to build from source on Windows if no C++ build tools
     are installed; recent versions (>=2.0.7) ship prebuilt Windows wheels on PyPI, so a plain
     `pip install -r requirements.txt` should work. If it still fails to build, install the
     [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
     (or run `pip install pycocotools-windows` as a fallback) and retry.
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
   - **Windows:** `download_ckpts.sh` is a bash script and will not run under plain `cmd`/
     PowerShell. If Git for Windows is installed, run it through Git Bash instead:
     ```powershell
     cd sam2\checkpoints
     "C:\Program Files\Git\bin\bash.exe" download_ckpts.sh
     cd ..\..
     ```
     Otherwise, download just the checkpoint this pipeline actually uses
     (`sam2.1_hiera_large.pt`) directly, from inside `sam2\checkpoints`:
     ```powershell
     Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" -OutFile "sam2.1_hiera_large.pt"
     ```

4. **Point `scripts/config.py` at your local SAM2 checkout** — edit this line to the
   full path where you cloned it in step 3:
   ```python
   SAM2_PATH = "/path/to/sam2"  # Replace with the full path
   ```

5. **Run inference** — pretrained coral filter models, class labels/remap tables
   (`data/classes_v18.json`, `data/remap.json`), and tuned SAM2 hyperparameters are already
   committed in this repository (under `models/`, `data/`, and `data/segmentation/`
   respectively). Try it on the bundled example image:
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

## 📄 Reproducing the Paper's Results

These are the exact command-line steps to reproduce the paper's reported LCC IoU/Dice
numbers on another machine.

1. **Clone this repository** and `cd` into it:
   ```bash
   git clone https://github.com/SamLeeBYU/NeuralReefer.git
   cd NeuralReefer
   ```

2. **Set up the Python environment** (see Installation above for details):
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate; on Git Bash: source .venv/Scripts/activate
   pip install -r requirements.txt
   ```

3. **Install SAM2** and download its checkpoint (see the Windows note under Installation
   step 3 above if `./download_ckpts.sh` doesn't run on your machine):
   ```bash
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2 && pip install -e . && cd ..
   cd sam2/checkpoints && ./download_ckpts.sh && cd ../..
   ```

4. **Edit `SAM2_PATH` in `scripts/config.py`** to the full path where you cloned SAM2 in
   step 3. Should just be `SAM2_PATH = "sam2"` if these directions were followed.

5. **Download and extract the training data** from Zenodo
   (<https://doi.org/10.5281/zenodo.19373197>) into the repository root, so that
   `yellowfin_segment.v18i.coco-segmentation/train/` (containing the 552 annotated images and
   `_annotations.coco.json`) sits directly under `NeuralReefer/` — this must match `TRAIN_DIR`
   in `config.py`. This archive is imagery/annotations only — do **not** copy any
   `classes*.json`/`remap.json`-like files it may contain into `data/`; the ones the code
   actually reads (`data/classes_v18.json`, `data/remap.json`) already ship with the git
   repository from step 1 and must not be overwritten.

6. **Run the full reproduction script.** This uses the pretrained submodels + ensemble
   already committed under `models/filter_res34_5_v18/` — no training required — running SAM2
   and the CNN ensemble over all 552 images and scoring the result against ground truth:
   ```bash
   python scripts/replicate.py
   ```
   Depending on the CPU/GPU used, this will take anywhere from several hours to several days to complete.

7. **Print the held-out LCC IoU/Dice result** to compare against the paper/another machine:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('yellowfin_segment.v18i.coco-segmentation/train/inference/annotations_coco_test_metrics.csv')
   lcc = df[df['taxonomy'] == 'lcc']
   print(f\"Test-set LCC IoU: {lcc['iou'].mean():.4f}  |  Dice: {lcc['dice_f1'].mean():.4f}  (n={len(lcc)} images)\")
   "
   ```
   Or, in a single line:
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('yellowfin_segment.v18i.coco-segmentation/train/inference/annotations_coco_test_metrics.csv'); lcc = df[df['taxonomy'] == 'lcc']; iou = lcc['iou'].mean(); dice = lcc['dice_f1'].mean(); n = len(lcc); print(f'Test-set LCC IoU: {iou:.4f} | Dice: {dice:.4f} (n={n} images)')"
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

