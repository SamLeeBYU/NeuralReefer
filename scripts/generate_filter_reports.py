"""
Evaluates the already-trained coral filter ensemble (loaded from
FILTER_MODELS_DIR) on the mask dataset at MASK_DATA_PATH, and writes:

  - <FILTER_MODELS_DIR>/model_performance.txt : in-sample / out-of-sample
    accuracy, precision, recall for each of the M submodels and the ensemble
  - <FILTER_MODELS_DIR>/confusion_matrix.txt  : the ensemble's full
    genus x bleach-status (+ noncoral) confusion matrix on the held-out
    split, plus its class-index labels (same text format `data_viz.R`
    already parses, so no downstream changes are needed)

This does NOT retrain anything -- it loads the already-saved model weights
and reconstructs a deterministic, reproducible split of the mask dataset to
evaluate them against, since the code that originally produced these two
files is no longer in the repository (filter.py's confusion-matrix logic is
commented out, and nothing ever wrote model_performance.txt -- see the
scripts/train.py conversation this was built from).

Split methodology -- three nested partitions of MASK_DATA_PATH, matching
CoralFilterEnsembler.train_ensemble()'s own split exactly (same seed=42,
same SPLIT/ENSEMBLE_SPLIT fractions):

  1. The full (oversampled) dataset splits SPLIT-wise into the submodels'
     own training pool ("in-sample" for the submodel rows -- NOTE: each
     submodel actually trained on its own *bootstrapped resample* of this
     partition, filter.py:91-92, not the literal partition) and the
     "ensemble pool" -- masks none of the 5 submodels ever trained on.
  2. The ensemble pool splits ENSEMBLE_SPLIT-wise into what the EM fit
     actually trained on ("in-sample" for the Ensemble row) and a final
     held-out slice.
  3. That final slice is used, uniformly, as "out-of-sample" for every row
     of the table AND the confusion matrix -- it's the one set untouched by
     both the submodels' training and the ensemble's own fit, so every
     reported out-of-sample number is computed on identical data.

This deliberately treats mask crops as exchangeable/independent regardless
of source image (an explicit, documented modeling assumption for fitting
and validating the classifier stage) -- the true, photo-independent test
of the full pipeline is the separate segmentation-level eval against the
111 held-out test images (scripts/train.py's eval()), which this script
does not touch.

Evaluation uses the deterministic MASK_TRANSFORM (no random augmentation)
rather than the training-time MASK_TRANSFORM_AUGMENT, and a fixed seed, so
re-running this script reproduces the same numbers every time.

The expensive step (forward-passing every submodel over the whole dataset)
is shared with scripts/retrain_ensemble.py via filter.extract_submodel_logits()
and LOGIT_CACHE_PATH -- this script's "in-sample"/"out-of-sample" split of
the FULL dataset is a superset of retrain_ensemble.py's own 30% ensemble
pool (same seed, same split fraction => the same held-out indices), so
whichever of the two scripts runs first and populates the cache saves the
other one the ~hours-long extraction cost.

Run as a standalone script from the repository root:
    python scripts/generate_filter_reports.py
"""

import os
import json

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from config import FILTER_MODELS_DIR, MASK_DATA_PATH, M, SPLIT, ENSEMBLE_SPLIT
from filter import CoralFilterEnsembler, extract_submodel_logits

SEED = 42
# Shared with scripts/retrain_ensemble.py -- same path, same cache.
LOGIT_CACHE_PATH = os.path.join(FILTER_MODELS_DIR, "submodel_logits_cache.npz")
USE_LOGIT_CACHE = True


def binary_coral_metrics(y_true_idx, y_pred_idx, noncoral_class):
    """Accuracy over all classes, plus precision/recall for the coral vs.
    noncoral binary sub-problem -- matching CoralFilter.test() /
    CoralFilterEnsembler.validate()'s existing metric definitions."""
    total = len(y_true_idx)
    accuracy = (y_pred_idx == y_true_idx).sum() / total if total > 0 else 0.0

    is_coral_true = y_true_idx != noncoral_class
    is_coral_pred = y_pred_idx != noncoral_class

    tp = int(np.logical_and(is_coral_true, is_coral_pred).sum())
    fn = int(np.logical_and(is_coral_true, ~is_coral_pred).sum())
    fp = int(np.logical_and(~is_coral_true, is_coral_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return accuracy, precision, recall


def format_performance_table(rows, ensemble_row, n_submodel_train, n_ensemble_train, n_oos):
    lines = []
    lines.append(f"{'':9s} {'In - Sample':^33s} {'Out-of-Sample':^33s}")
    lines.append(f"{'Model':9s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':6s} || "
                 f"{'Accuracy':8s} | {'Precision':9s} | {'Recall':6s} |")
    lines.append("=" * 77)
    for (name, in_acc, in_prec, in_rec, oos_acc, oos_prec, oos_rec) in rows:
        lines.append(f"{name:9s} | {in_acc:.4f}   | {in_prec:.4f}    | {in_rec:.4f} || "
                     f"{oos_acc:.4f}   | {oos_prec:.4f}    | {oos_rec:.4f} |")
    lines.append("-" * 77)
    name, in_acc, in_prec, in_rec, oos_acc, oos_prec, oos_rec = ensemble_row
    lines.append(f"{name:9s} | {in_acc:.4f}   | {in_prec:.4f}    | {in_rec:.4f} || "
                 f"{oos_acc:.4f}   | {oos_prec:.4f}    | {oos_rec:.4f} |")
    lines.append("")
    lines.append(" Notes: Each submodel's in-sample figures are computed on the full")
    lines.append(f" (non-bootstrapped) training partition it was drawn from ({n_submodel_train} masks) --")
    lines.append(" the exact bootstrapped resample each model actually trained on isn't")
    lines.append(" persisted, so this is the closest reproducible proxy, not the literal")
    lines.append(f" training set. The Ensemble row's in-sample figures are computed on the")
    lines.append(f" {n_ensemble_train} masks the EM fit actually trained on -- disjoint from every")
    lines.append(" submodel's training pool by construction (held out before any submodel")
    lines.append(f" training began). All out-of-sample statistics (every row, and the")
    lines.append(f" confusion matrix) are computed on the same shared {n_oos}-mask partition --")
    lines.append(" the one slice untouched by both submodel training and the ensemble's own")
    lines.append(" fit (same nested stratified splits, seed=42).")
    return "\n".join(lines) + "\n"


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Booting up NeuralReefer coral filter ensemble...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    ensembler = CoralFilterEnsembler(base_dataset=MASK_DATA_PATH, device=device, m=M, split=SPLIT)
    ensembler.load_models(FILTER_MODELS_DIR)

    dataset = ensembler.mask_data
    noncoral_class = ensembler.noncoral_class
    class_names = list(ensembler.classes.keys())
    k = len(class_names)

    # Expensive step, shared with retrain_ensemble.py -- sets the
    # deterministic MASK_TRANSFORM and evaluates every submodel over the
    # WHOLE dataset internally (see filter.extract_submodel_logits).
    full_logits, y_true_idx = extract_submodel_logits(
        ensembler.models, dataset, k,
        batch_size=128, cache_path=LOGIT_CACHE_PATH, use_cache=USE_LOGIT_CACHE,
    )

    # Level 1: submodels' own training pool vs. the ensemble pool (masks no
    # submodel ever trained on) -- the same split every CoralFilter builds
    # internally (filter.py:80-87).
    train_idx, ensemble_pool_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=SPLIT,
        stratify=y_true_idx,
        random_state=SEED,
    )
    # Level 2: within the ensemble pool, what the EM fit actually trained on
    # vs. a final held-out slice -- the same split train_ensemble() makes
    # internally (filter.py, train_ensemble()).
    ensemble_train_local, ensemble_oos_local = train_test_split(
        np.arange(len(ensemble_pool_idx)),
        test_size=ENSEMBLE_SPLIT,
        stratify=y_true_idx[ensemble_pool_idx],
        random_state=SEED,
    )
    ensemble_train_idx = ensemble_pool_idx[ensemble_train_local]
    oos_idx = ensemble_pool_idx[ensemble_oos_local]  # shared out-of-sample set for EVERYTHING below

    print(f"Submodel in-sample: {len(train_idx)} masks | "
          f"Ensemble in-sample: {len(ensemble_train_idx)} masks | "
          f"Out-of-sample (shared): {len(oos_idx)} masks")

    train_logits = full_logits[train_idx]
    ensemble_train_logits = full_logits[ensemble_train_idx]
    oos_logits = full_logits[oos_idx]

    rows = []
    for m in range(ensembler.m):
        train_pred_idx = np.argmax(train_logits[:, m, :], axis=1)
        oos_pred_idx = np.argmax(oos_logits[:, m, :], axis=1)

        in_acc, in_prec, in_rec = binary_coral_metrics(y_true_idx[train_idx], train_pred_idx, noncoral_class)
        oos_acc, oos_prec, oos_rec = binary_coral_metrics(y_true_idx[oos_idx], oos_pred_idx, noncoral_class)

        rows.append((f"  {m + 1}", in_acc, in_prec, in_rec, oos_acc, oos_prec, oos_rec))

    print("Evaluating ensemble...")
    # EMEnsembleOptimizer is pure NumPy (EM-fit, not gradient-trained) --
    # no device/eval-mode concept, predict_proba takes/returns numpy arrays.
    ens_train_probs = ensembler.ensemble_model.predict_proba(ensemble_train_logits)
    ens_oos_probs = ensembler.ensemble_model.predict_proba(oos_logits)

    ens_train_pred_idx = np.argmax(ens_train_probs, axis=1)
    ens_oos_pred_idx = np.argmax(ens_oos_probs, axis=1)

    ens_in_acc, ens_in_prec, ens_in_rec = binary_coral_metrics(y_true_idx[ensemble_train_idx], ens_train_pred_idx, noncoral_class)
    ens_oos_acc, ens_oos_prec, ens_oos_rec = binary_coral_metrics(y_true_idx[oos_idx], ens_oos_pred_idx, noncoral_class)
    ensemble_row = ("Ensemble", ens_in_acc, ens_in_prec, ens_in_rec, ens_oos_acc, ens_oos_prec, ens_oos_rec)

    perf_path = os.path.join(FILTER_MODELS_DIR, "model_performance.txt")
    with open(perf_path, "w") as f:
        f.write(format_performance_table(rows, ensemble_row, len(train_idx), len(ensemble_train_idx), len(oos_idx)))
    print(f"Wrote {perf_path}")

    # Confusion matrix: ensemble predictions on the shared out-of-sample
    # partition -- the same set every out-of-sample number above uses, so
    # the matrix and the table are directly comparable.
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true_idx[oos_idx], ens_oos_pred_idx):
        cm[t, p] += 1

    cm_path = os.path.join(FILTER_MODELS_DIR, "confusion_matrix.txt")
    with open(cm_path, "w") as f:
        f.write(repr(cm))
        f.write("\n\nlabels: ")
        f.write(json.dumps(ensembler.classes, indent=4))
        f.write("\n")
    print(f"Wrote {cm_path} (ensemble predictions, shared out-of-sample partition, {len(oos_idx)} masks)")


if __name__ == "__main__":
    main()
