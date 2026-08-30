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

Split methodology: every CoralFilter submodel is built during training with
the same stratified split -- test_size=SPLIT, random_state=42 (filter.py,
CoralFilter.__init__) -- before bootstrap-resampling its own training
partition. This script reconstructs that same stratified split (same seed,
same SPLIT) once, shared across all submodels and the ensemble:

  - "out-of-sample" = the held-out partition, identical across every
    submodel by construction (same seed => same split), matching the
    original file's own claim that all submodels were scored on "the exact
    same set" of validation masks. The confusion matrix is computed here,
    on the ensemble's predictions.
  - "in-sample" = the remaining training partition. NOTE: each submodel
    was actually trained on its own *bootstrapped resample* of this
    partition (filter.py:91-92), and that specific resample isn't
    persisted anywhere -- only the final weights are. So this is the
    closest reproducible proxy (the full pool each model was drawn from),
    not the literal bootstrap draw used at training time.

Evaluation uses the deterministic MASK_TRANSFORM (no random augmentation)
rather than the training-time MASK_TRANSFORM_AUGMENT, and a fixed seed, so
re-running this script reproduces the same numbers every time.

Run as a standalone script from the repository root:
    python scripts/generate_filter_reports.py
"""

import os
import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import FILTER_MODELS_DIR, MASK_DATA_PATH, M, SPLIT
from filter import CoralFilterEnsembler
from transforms import MASK_TRANSFORM

SEED = 42


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


def predict_logits(submodel, dataset, indices, batch_size=128):
    """Forward-passes dataset.img_data[indices] through one submodel's raw
    network, matching the logit computation CoralFilterEnsembler already
    does internally in train_ensemble() (filter.py:337-343)."""
    device = submodel.device
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long)
            X = dataset.img_data[batch_idx].to(device)
            pred = submodel.model(X).squeeze(1)
            outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def format_performance_table(rows, ensemble_row, n_train, n_val):
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
    lines.append(f" (non-bootstrapped) training partition it was drawn from ({n_train} masks) --")
    lines.append(" the exact bootstrapped resample each model actually trained on isn't")
    lines.append(" persisted, so this is the closest reproducible proxy, not the literal")
    lines.append(f" training set. All out-of-sample statistics were collected on the same")
    lines.append(f" shared held-out partition ({n_val} masks), matching every submodel's")
    lines.append(" internal validation split (same stratified split, seed=42).")
    return "\n".join(lines) + "\n"


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Booting up NeuralReefer coral filter ensemble...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    ensembler = CoralFilterEnsembler(base_dataset=MASK_DATA_PATH, device=device, m=M, split=SPLIT)

    # Deterministic evaluation: use the non-augmented transform (no random
    # crop/rotation/flip) so re-running this script gives identical numbers.
    ensembler.mask_data.transform_fn = MASK_TRANSFORM
    ensembler.mask_data.resample()

    ensembler.load_models(FILTER_MODELS_DIR)

    dataset = ensembler.mask_data
    y_true_idx = torch.argmax(dataset.labels, dim=1).numpy()
    noncoral_class = ensembler.noncoral_class
    class_names = list(ensembler.classes.keys())
    k = len(class_names)

    # The same stratified split every CoralFilter builds internally
    # (filter.py:80-87), shared here across all submodels + the ensemble.
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=SPLIT,
        stratify=y_true_idx,
        random_state=SEED,
    )
    print(f"In-sample pool: {len(train_idx)} masks | Out-of-sample (shared, held-out): {len(val_idx)} masks")

    train_logits = np.zeros((len(train_idx), ensembler.m, k), dtype=np.float32)
    val_logits = np.zeros((len(val_idx), ensembler.m, k), dtype=np.float32)

    rows = []
    for m in range(ensembler.m):
        print(f"Evaluating submodel {m + 1}/{ensembler.m}...")
        submodel = ensembler.models[m]
        submodel.model.eval()

        train_logits[:, m, :] = predict_logits(submodel, dataset, train_idx)
        val_logits[:, m, :] = predict_logits(submodel, dataset, val_idx)

        train_pred_idx = np.argmax(train_logits[:, m, :], axis=1)
        val_pred_idx = np.argmax(val_logits[:, m, :], axis=1)

        in_acc, in_prec, in_rec = binary_coral_metrics(y_true_idx[train_idx], train_pred_idx, noncoral_class)
        oos_acc, oos_prec, oos_rec = binary_coral_metrics(y_true_idx[val_idx], val_pred_idx, noncoral_class)

        rows.append((f"  {m + 1}", in_acc, in_prec, in_rec, oos_acc, oos_prec, oos_rec))

    print("Evaluating ensemble...")
    # EMEnsembleOptimizer is pure NumPy (EM-fit, not gradient-trained) --
    # no device/eval-mode concept, predict_proba takes/returns numpy arrays.
    train_ens_probs = ensembler.ensemble_model.predict_proba(train_logits)
    val_ens_probs = ensembler.ensemble_model.predict_proba(val_logits)

    train_ens_pred_idx = np.argmax(train_ens_probs, axis=1)
    val_ens_pred_idx = np.argmax(val_ens_probs, axis=1)

    ens_in_acc, ens_in_prec, ens_in_rec = binary_coral_metrics(y_true_idx[train_idx], train_ens_pred_idx, noncoral_class)
    ens_oos_acc, ens_oos_prec, ens_oos_rec = binary_coral_metrics(y_true_idx[val_idx], val_ens_pred_idx, noncoral_class)
    ensemble_row = ("Ensemble", ens_in_acc, ens_in_prec, ens_in_rec, ens_oos_acc, ens_oos_prec, ens_oos_rec)

    perf_path = os.path.join(FILTER_MODELS_DIR, "model_performance.txt")
    with open(perf_path, "w") as f:
        f.write(format_performance_table(rows, ensemble_row, len(train_idx), len(val_idx)))
    print(f"Wrote {perf_path}")

    # Confusion matrix: ensemble predictions on the shared out-of-sample
    # (held-out) partition -- resolves the ambiguity of not knowing which
    # split the original file was computed on.
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true_idx[val_idx], val_ens_pred_idx):
        cm[t, p] += 1

    cm_path = os.path.join(FILTER_MODELS_DIR, "confusion_matrix.txt")
    with open(cm_path, "w") as f:
        f.write(repr(cm))
        f.write("\n\nlabels: ")
        f.write(json.dumps(ensembler.classes, indent=4))
        f.write("\n")
    print(f"Wrote {cm_path} (ensemble predictions, out-of-sample partition)")


if __name__ == "__main__":
    main()
