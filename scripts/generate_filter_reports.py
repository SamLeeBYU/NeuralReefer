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
evaluate them against.

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

Evaluation uses MASK_TRANSFORM_AUGMENT (see filter.extract_submodel_logits),
since every submodel was only ever trained under it. Reproducibility comes
from config.MASK_TRANSFORM_AUGMENT_SEED (via transforms.seeded_rng), so
re-running this script still reproduces the same numbers every time.

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

from config import FILTER_MODELS_DIR, MASK_DATA_PATH, M, SPLIT, ENSEMBLE_SPLIT, ALL_ENSEMBLE_METHODS
from filter import CoralFilterEnsembler, extract_submodel_logits

SEED = 42
# Shared with scripts/retrain_ensemble.py -- same path, same cache.
LOGIT_CACHE_PATH = os.path.join(FILTER_MODELS_DIR, "submodel_logits_cache.npz")
USE_LOGIT_CACHE = True

# Which ENSEMBLE_METHOD's predictions confusion_matrix.txt reflects -- independent
# of config.ENSEMBLE_METHOD (the "production" method CoralFilterEnsembler.predict()
# loads). Set to whichever method model_performance.txt shows performing best.
CONFUSION_MATRIX_METHOD = "reweight"

# Short row labels for each config.ALL_ENSEMBLE_METHODS entry in the table.
ENSEMBLE_METHOD_LABELS = {
    "em": "EM", "linear": "Linear", "adam": "Adam",
    "reweight": "Reweight", "multinomial": "Multinom", "nn": "NeuralNet",
}


def load_ensemble_model(ensembler, method, dir):
    """Loads one of the method-suffixed combiners scripts/retrain_ensemble.py
    persists (ensemble_<method>.npz) into a fresh ensemble_model instance,
    without touching ensembler.ensemble_model/ensemble_method (unlike
    CoralFilterEnsembler.load_models(), which is only for the single
    canonical ensemble.npz). This script doesn't retrain anything -- if the
    file is missing, that method hasn't been fit yet."""
    path = os.path.join(dir, f"ensemble_{method}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run scripts/retrain_ensemble.py first to fit and "
            f"persist every method in config.ALL_ENSEMBLE_METHODS."
        )
    data = np.load(path)
    model = ensembler._make_ensemble_model(method)
    for name in ensembler._ENSEMBLE_PARAM_NAMES[method]:
        setattr(model, name, data[name])
    return model


def binary_coral_metrics(y_true_idx, y_pred_idx, noncoral_class):
    """Accuracy over all classes, plus precision/recall/F2 for the coral vs.
    noncoral binary sub-problem -- matching CoralFilter.test() /
    CoralFilterEnsembler.validate()'s existing metric definitions. F2
    weights recall 4x precision (beta=2), matching NEG_WEIGHT's emphasis on
    not missing coral (recall) over false-flagging noncoral."""
    total = len(y_true_idx)
    accuracy = (y_pred_idx == y_true_idx).sum() / total if total > 0 else 0.0

    is_coral_true = y_true_idx != noncoral_class
    is_coral_pred = y_pred_idx != noncoral_class

    tp = int(np.logical_and(is_coral_true, is_coral_pred).sum())
    fn = int(np.logical_and(is_coral_true, ~is_coral_pred).sum())
    fp = int(np.logical_and(~is_coral_true, is_coral_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f2


def format_performance_table(rows, ensemble_rows, n_submodel_train, n_ensemble_train, n_oos, cm_method_label):
    lines = []
    lines.append(f"{'':9s} {'In - Sample':^44s} {'Out-of-Sample':^44s}")
    lines.append(f"{'Model':9s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':6s} | {'F2':6s} || "
                 f"{'Accuracy':8s} | {'Precision':9s} | {'Recall':6s} | {'F2':6s} |")
    lines.append("=" * 99)
    for (name, in_acc, in_prec, in_rec, in_f2, oos_acc, oos_prec, oos_rec, oos_f2) in rows:
        lines.append(f"{name:9s} | {in_acc:.4f}   | {in_prec:.4f}    | {in_rec:.4f} | {in_f2:.4f} || "
                     f"{oos_acc:.4f}   | {oos_prec:.4f}    | {oos_rec:.4f} | {oos_f2:.4f} |")
    lines.append("-" * 99)
    for (name, in_acc, in_prec, in_rec, in_f2, oos_acc, oos_prec, oos_rec, oos_f2) in ensemble_rows:
        lines.append(f"{name:9s} | {in_acc:.4f}   | {in_prec:.4f}    | {in_rec:.4f} | {in_f2:.4f} || "
                     f"{oos_acc:.4f}   | {oos_prec:.4f}    | {oos_rec:.4f} | {oos_f2:.4f} |")
    lines.append("")
    lines.append(" Notes: Each submodel's in-sample figures are computed on the full")
    lines.append(f" (non-bootstrapped) training partition it was drawn from ({n_submodel_train} masks) --")
    lines.append(" the exact bootstrapped resample each model actually trained on isn't")
    lines.append(" persisted, so this is the closest reproducible proxy, not the literal")
    lines.append(f" training set. Each of the rows below the divider is a different")
    lines.append(" ENSEMBLE_METHOD (see config.py/config.ALL_ENSEMBLE_METHODS), all fit on")
    lines.append(f" the SAME {n_ensemble_train} masks -- disjoint from every submodel's training pool")
    lines.append(" by construction (held out before any submodel training began). All")
    lines.append(f" out-of-sample statistics (every row, and the confusion matrix) are")
    lines.append(f" computed on the same shared {n_oos}-mask partition -- the one slice")
    lines.append(" untouched by both submodel training and any ensemble fit (same nested")
    lines.append(f" stratified splits, seed=42). The confusion matrix reflects only the")
    lines.append(f" \"{cm_method_label}\" method (CONFUSION_MATRIX_METHOD in generate_filter_reports.py),")
    lines.append(" not all five -- chosen as the best-performing method above, independent")
    lines.append(" of config.py's ENSEMBLE_METHOD (the production default).")
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

    # Expensive step, shared with retrain_ensemble.py -- sets the seeded
    # MASK_TRANSFORM_AUGMENT and evaluates every submodel over the WHOLE
    # dataset internally (see filter.extract_submodel_logits).
    full_logits, y_true_idx = extract_submodel_logits(
        ensembler.models, dataset, k,
        batch_size=128, cache_path=LOGIT_CACHE_PATH, use_cache=USE_LOGIT_CACHE,
    )

    # Level 1: submodels' own training pool vs. the ensemble pool (masks no
    # submodel ever trained on), via CoralFilterEnsembler's own split method
    # so this can't diverge from what filter.py actually uses. legacy=True
    # matches the split the model_*.pth files on disk were trained under --
    # see CoralFilterEnsembler._submodel_pool_split and
    # scripts/verify_split_integrity.py. Drop once submodels are retrained
    # under the current split.
    train_idx, ensemble_pool_idx = ensembler._submodel_pool_split(legacy=True)
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

        in_acc, in_prec, in_rec, in_f2 = binary_coral_metrics(y_true_idx[train_idx], train_pred_idx, noncoral_class)
        oos_acc, oos_prec, oos_rec, oos_f2 = binary_coral_metrics(y_true_idx[oos_idx], oos_pred_idx, noncoral_class)

        rows.append((f"  {m + 1}", in_acc, in_prec, in_rec, in_f2, oos_acc, oos_prec, oos_rec, oos_f2))

    print("Evaluating ensemble methods...")
    # EMEnsembleOptimizer/etc. are pure NumPy (fit via EM/Newton/Adam, not
    # gradient-trained at inference time) -- no device/eval-mode concept,
    # predict_proba takes/returns numpy arrays.
    ensemble_rows = []
    cm_oos_pred_idx = None  # CONFUSION_MATRIX_METHOD's predictions, for the confusion matrix
    for method in ALL_ENSEMBLE_METHODS:
        is_default = method == ensembler.ensemble_method
        ens_model = ensembler.ensemble_model if is_default else load_ensemble_model(ensembler, method, FILTER_MODELS_DIR)

        ens_train_probs = ens_model.predict_proba(ensemble_train_logits)
        ens_oos_probs = ens_model.predict_proba(oos_logits)

        ens_train_pred_idx = np.argmax(ens_train_probs, axis=1)
        ens_oos_pred_idx = np.argmax(ens_oos_probs, axis=1)

        ens_in_acc, ens_in_prec, ens_in_rec, ens_in_f2 = binary_coral_metrics(y_true_idx[ensemble_train_idx], ens_train_pred_idx, noncoral_class)
        ens_oos_acc, ens_oos_prec, ens_oos_rec, ens_oos_f2 = binary_coral_metrics(y_true_idx[oos_idx], ens_oos_pred_idx, noncoral_class)
        ensemble_rows.append((ENSEMBLE_METHOD_LABELS[method], ens_in_acc, ens_in_prec, ens_in_rec, ens_in_f2, ens_oos_acc, ens_oos_prec, ens_oos_rec, ens_oos_f2))

        if method == CONFUSION_MATRIX_METHOD:
            cm_oos_pred_idx = ens_oos_pred_idx

    perf_path = os.path.join(FILTER_MODELS_DIR, "model_performance.txt")
    with open(perf_path, "w") as f:
        f.write(format_performance_table(
            rows, ensemble_rows, len(train_idx), len(ensemble_train_idx), len(oos_idx),
            ENSEMBLE_METHOD_LABELS[CONFUSION_MATRIX_METHOD],
        ))
    print(f"Wrote {perf_path}")

    # Confusion matrix: CONFUSION_MATRIX_METHOD's predictions on the shared
    # out-of-sample partition -- the same set every out-of-sample number
    # above uses, so the matrix and the table are directly comparable.
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true_idx[oos_idx], cm_oos_pred_idx):
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
