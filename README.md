# Neural Reefer: Modular Coral Reef Segmentation and Classification from Top-Down Imagery

This repository implements an end-to-end deep learning pipeline for analyzing top-down coral reef imagery. It combines high-recall instance segmentation (SAM2), CNN-based filtering and multi-task classification (genus, bleaching, etc.) into a modular system tailored for ecological monitoring. Developed June-July 2025 at WHOI using Majuro reef data, Neural Reefer supports scalable estimation of live coral cover (LCC) and taxonomic/health classification from top-down imagery surveys.

---

## 🔧 Installation

To set up Neural Reefer from scratch, follow these steps:

1. **Clone the SAM2 repository** (required for segmentation):
   ```bash
   git clone https://github.com/facebookresearch/sam2.git
   ```

2. **Update `config.py`** to reflect your local SAM2 path:
   ```python
   SAM2_PATH = "/path/to/sam2"  # Replace with the full path
   SAM2_CHECKPOINT_PATH = f"{SAM2_PATH}/checkpoints/sam2.1_hiera_large.pt"
   ```

3. **Enable training mode** in `config.py` by setting:
   ```python
   TUNE_SEGMENTER = True
   CREATE_MASK_DATASET = True
   TRAIN_CORAL_FILTER = True
   ```

4. **Run training pipeline** from root directory:
   ```bash
   python scripts/main.py --mode train
   ```

5. **Run inference** on a folder of images (after training):
   ```bash
   python scripts/main.py --mode inference
   ```

Neural Reefer requires Python 3.9+ and PyTorch (>= 2.0). First, clone the repository:

```bash
git clone https://github.com/your-repo/NeuralReefer.git
cd NeuralReefer
```

Install dependencies (preferably in a new virtual environment):

If using CUDA, ensure compatible versions of PyTorch and torchvision are installed. Pretrained SAM2 weights must be downloaded separately from the [official repository](https://github.com/facebookresearch/sam2).

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

### Run inference on a folder of images

```bash
python main.py --mode eval --image_dir path/to/images
```

### Train full pipeline (segmentation + filtering + classification)

```bash
python main.py --mode train
```

### Visualize performance metrics

```bash
python main.py --mode visualize
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

