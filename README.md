# Neural Reefer: Modular Coral Reef Segmentation and Classification from Top-Down Imagery

This repository implements an end-to-end deep learning pipeline for analyzing top-down coral reef imagery. It combines high-recall instance segmentation (SAM2), CNN-based filtering (CoralFilter), and multi-task classification (genus, bleaching, etc.) into a modular system tailored for ecological monitoring. Developed June-July 2025 at WHOI using Majuro reef data, Neural Reefer supports scalable estimation of live coral cover (LCC) and taxonomic/health classification from top-down imagery surveys.

---

## Overview

The pipeline proceeds through four stages:

- **Segmentation** using [SAM2](https://github.com/facebookresearch/sam2) with dense prompting to extract candidate coral objects
- **Filtering** false positives using a bootstrapped CNN ensemble trained on true/false mask crops
- **Classification** of genus identity and bleaching severity using independent ResNet-based CNNs
- **Cover Estimation** using filtered and classified masks to compute ecological summaries (e.g., % live coral cover)

Each component can be trained and evaluated independently or integrated into a single `main.py` pipeline for evaluation and inference.

---

## Training Procedure

The training pipeline consists of three stages:

### 1. Segmentation with SAM2

The `SAM2Segmenter` class implements grid-based prompting for instance segmentation. All generated masks are merged using a confidence-weighted IoU filter and optionally tuned with Bayesian optimization (`segmenter.tune()`) to identify optimal hyperparameters for underwater imagery.

### 2. Coral Filter Ensemble

To reduce false positives from SAM2, each mask is passed through a trained CNN (`CoralClassifier`) using the `CoralFilter` class. A bootstrapped ensemble of these models is trained with early stopping, and a logistic meta-learner is fit using ROC-AUC scoring:

- **Training data**: crops labeled from COCO masks (positive) and non-overlapping predicted masks (negative)
- **Model**: ResNet18 backbone with MLP head and dropout
- **Ensemble**: `CoralFilterEnsembler` implements $m$ submodels and optimizes weights via logistic regression using OOS logits
- **Class imbalance**: mitigated with `pos_weight` loss scaling and robust validation split

### 3. Classification (Genus & Bleaching)

Filtered masks are then classified:

- **Genus**: Multiclass classifier with softmax cross-entropy loss
- **Bleaching**: Binary or ordinal classifier depending on data labels
- **Augmentation**: Random rotations, flips, and color normalization are applied through `MaskLoader` and `transforms.py`
- **Evaluation**: Confusion matrices and ROC-AUC metrics are given during training for each task

---

## Scripts and Modules

### `main.py`
- End-to-end interface
- Supports `train`, `eval`, `visualize` modes
- Input: top-down images or directories
- Output: coral cover metrics and mask overlays
```bash
python main.py --mode eval --image_dir data/test
```

### `segmenter.py`
- Defines `SAM2Segmenter` and `CoralSegmenter` classes
- Implements CLAHE, red-channel boosting, white balancing
- Supports mask generation, merging, tuning, and visualization

### `filter.py`
- Defines:
  - `CoralFilter`: wrapper for training a single CNN model
  - `CoralFilterEnsembler`: ensemble of $m$ CoralFilter models with bootstrapping
- Trains on `MaskLoader`-generated data, filters non-coral objects

### `classifier.py`
- Defines `CoralClassifier` (binary ResNet18) and provides templates for multi-task heads
- Modular classifier heads enable dropout, depth control, and label remapping

### `train.py`
- Controls full training pipeline (tuning, filtering, evaluation)

### `data.py`
- `MaskLoader`: PyTorch dataset to generate and load coral mask crops

### `transforms.py`
- Custom image transforms including CLAHE and standard normalization

### `config.py`
- Stores paths, hyperparameters, training constants

### `visualize.py`
- Plots accuracy, coral cover, and bias distributions

### `utils.py`
- Suppresses printing, tracks timing, and ensures JSON compatibility

---

## Usage

### Run End-to-End Inference
```bash
python main.py --mode eval --image_dir path/to/reef/images
```

### Train Pipeline
Edit `config.py` accordingly and run:
```bash
python main.py --mode train
```

### Visualize Diagnostics (for developers)
```bash
python main.py --mode visualize --version 2.0
```

---

## Citation and Contact

**Lead Developer**: Sam Lee  
**Affiliations**: Brigham Young University, Woods Hole Oceanographic Institution  
**Contact**:  
- 📧 samlee.byu@gmail.com (personal)  
- 📨 slee039@byu.edu  
- 🌊 sam.lee@whoi.edu  

---

## Acknowledgments

- SAM2 authors for foundational segmentation model  
- WHOI project oversight from Dr. Calvin Quigley, Dr. Nathan Mollica, Dr. Anne Cohen
- WHOI Yellowfin ASV team for coral survey data from Majuro
- WHOI Annotation Team for ground truth mask dataset: Evii Tong, Robert Ronan
