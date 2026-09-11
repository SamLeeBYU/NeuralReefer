"""
Exports, for every fitted ensemble_method in config.ALL_ENSEMBLE_METHODS, a
per-(submodel, class) matrix suitable for a heatmap in the style of
figures/paper/ensemble_interpretable_weights_heatmap.png ("Ensemble
Responsibility by Class") -- to data/performance/ensemble_weights.csv,
which data_viz.R reads to render one PNG per method into figures/ensemble/.

The underlying quantity differs by method, since their parameterizations
aren't all mixtures with a responsibility structure -- each is the closest
natural per-(model, class) quantity available for that method, labeled by
`value_type` so data_viz.R can pick an appropriate title/color scale:

  - "em", "adam", "reweight": mean E-step responsibility gamma_{i,m},
    averaged over examples with true class k -- this is EXACTLY the
    original figure's quantity (these three are all latent-class mixtures
    with a genuine per-example responsibility). model_share = alpha_m
    (the fitted mixture weight), matching the original figure's
    "Model N (XX.XX%)" row label.
  - "multinomial": the fitted beta_{m,k} calibration exponent directly (no
    alpha/mixture in this method at all, so no responsibility or
    model_share is defined -- diverges around 1.0, "no calibration change").
  - "linear": the self-class diagonal of submodel m's [K,K] coefficient
    block in the [MK,K] regression matrix W -- "how much does submodel m's
    own predicted probability for class k contribute to the ensemble's own
    prediction for class k". model_share = mean(|self-weight|) across
    classes, normalized to sum to 1 across models, as an analogous "share"
    (NOT alpha -- this method has no mixture weights).
  - "nn": mean absolute first-layer weight (W0) per input feature,
    reshaped to [M, K] -- an approximate "input sensitivity", NOT a
    responsibility or calibration weight (the network has no such
    structure) -- flagged as an approximation in its plot title downstream.

Run as a standalone script from the repository root:
    python scripts/export_ensemble_weights.py
"""
import os
import sys
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch

from config import FILTER_MODELS_DIR, MASK_DATA_PATH, M, SPLIT, ALL_ENSEMBLE_METHODS
from filter import CoralFilterEnsembler, extract_submodel_logits

SEED = 42
LOGIT_CACHE_PATH = os.path.join(FILTER_MODELS_DIR, "submodel_logits_cache.npz")
OUTPUT_CSV = "data/performance/ensemble_weights.csv"


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def responsibility_beta(logits, y_idx, alpha, beta):
    """EM/Adam's shared mixture: gamma_{i,m}, using their common
    beta-on-log-softmax parameterization."""
    u = log_softmax(logits)                       # [N, M, K]
    z = beta[None, :, :] * u
    tilde_p = softmax(z)                           # [N, M, K]
    N, Mm, K = tilde_p.shape
    tilde_p_y = tilde_p[np.arange(N), :, y_idx]    # [N, M]
    joint = alpha[None, :] * tilde_p_y
    return joint / joint.sum(axis=1, keepdims=True)  # [N, M]


def responsibility_reweight(logits, y_idx, alpha, W):
    """Reweight's Hadamard-reweight-and-renormalize mixture: gamma_{i,m}."""
    p = softmax(logits)                            # [N, M, K]
    weighted = W[None, :, :] * p
    tilde_p = weighted / np.clip(weighted.sum(axis=-1, keepdims=True), 1e-300, None)
    N, Mm, K = tilde_p.shape
    tilde_p_y = tilde_p[np.arange(N), :, y_idx]
    joint = alpha[None, :] * tilde_p_y
    return joint / joint.sum(axis=1, keepdims=True)


def mean_by_class(values_nm, y_idx, k):
    """values_nm: [N, M] -> [K, M] mean over examples sharing true class k."""
    out = np.zeros((k, values_nm.shape[1]))
    for c in range(k):
        mask = y_idx == c
        if mask.any():
            out[c] = values_nm[mask].mean(axis=0)
    return out


def main():
    device = torch.device("cpu")
    ensembler = CoralFilterEnsembler(base_dataset=MASK_DATA_PATH, device=device, m=M, split=SPLIT)
    ensembler.load_models(FILTER_MODELS_DIR)
    k = len(ensembler.classes)
    class_names = list(ensembler.classes.keys())  # index-ordered

    full_logits, y_true_idx = extract_submodel_logits(
        ensembler.models, ensembler.mask_data, k,
        batch_size=128, cache_path=LOGIT_CACHE_PATH, use_cache=True, verbose=True,
    )

    # Same in-sample ensemble-training pool model_performance.txt's
    # "In-Sample" ensemble rows are computed on.
    train_idx, ensemble_pool_idx = ensembler._submodel_pool_split(legacy=True)
    from sklearn.model_selection import train_test_split
    from config import ENSEMBLE_SPLIT
    ensemble_train_local, _ = train_test_split(
        np.arange(len(ensemble_pool_idx)), test_size=ENSEMBLE_SPLIT,
        stratify=y_true_idx[ensemble_pool_idx], random_state=SEED,
    )
    idx = ensemble_pool_idx[ensemble_train_local]
    logits, y_idx = full_logits[idx], y_true_idx[idx]

    rows = []
    for method in ALL_ENSEMBLE_METHODS:
        npz_path = os.path.join(FILTER_MODELS_DIR, f"ensemble_{method}.npz")
        if not os.path.exists(npz_path):
            print(f"Skipping {method!r} -- {npz_path} not found (not fit yet).")
            continue
        data = np.load(npz_path)
        print(f"Computing weights for {method!r}...")

        if method in ("em", "adam"):
            alpha, beta = data["alpha"], data["beta"]
            gamma = responsibility_beta(logits, y_idx, alpha, beta)      # [N, M]
            mat = mean_by_class(gamma, y_idx, k)                         # [K, M]
            model_share = alpha
            value_type = "responsibility"
        elif method == "reweight":
            alpha, W = data["alpha"], data["W"]
            gamma = responsibility_reweight(logits, y_idx, alpha, W)
            mat = mean_by_class(gamma, y_idx, k)
            model_share = alpha
            value_type = "responsibility"
        elif method == "multinomial":
            beta = data["beta"]                                          # [M, K]
            mat = beta.T                                                 # [K, M]
            model_share = None
            value_type = "calibration_beta"
        elif method == "linear":
            Wf, b = data["W"], data["b"]                                  # [MK, K], [K]
            Mm = ensembler.m
            W_blocks = Wf.reshape(Mm, k, k)                               # [M, K_in, K_out]
            self_weight = np.diagonal(W_blocks, axis1=1, axis2=2)         # [M, K]
            mat = self_weight.T                                           # [K, M]
            share = np.abs(self_weight).mean(axis=1)
            model_share = share / share.sum()
            value_type = "self_weight"
        elif method == "nn":
            W0 = data["W0"]                                               # [M*K, H1]
            Mm = ensembler.m
            sensitivity = np.abs(W0).mean(axis=1).reshape(Mm, k)          # [M, K]
            mat = sensitivity.T                                           # [K, M]
            share = sensitivity.mean(axis=1)
            model_share = share / share.sum()
            value_type = "input_sensitivity"
        else:
            print(f"Skipping {method!r} -- no visualization rule defined.")
            continue

        for model_i in range(mat.shape[1]):
            share_val = float(model_share[model_i]) if model_share is not None else np.nan
            for class_i in range(k):
                rows.append({
                    "method": method,
                    "model": model_i + 1,
                    "model_share": share_val,
                    "class_name": class_names[class_i],
                    "value": float(mat[class_i, model_i]),
                    "value_type": value_type,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV} ({len(df)} rows, methods={sorted(df['method'].unique())})")


if __name__ == "__main__":
    main()
