"""
Retrains ONLY the ensemble-combination stage (fit on top of the already-
trained CNN submodel weights on disk, model_1.pth .. model_M.pth) for each
method in METHODS_TO_FIT. Does not retrain the CNN submodels themselves.

Forward-passing every submodel over the full mask dataset (see
filter.extract_submodel_logits) is cached to LOGIT_CACHE_PATH; a second run
reuses the cache and skips straight to the (cheap) ensemble fit. Delete the
cache file, or set USE_LOGIT_CACHE = False, to force recomputation (needed
if the submodels themselves change). scripts/generate_filter_reports.py can
point at the same cache file and reuse it too.

Each fitted method is written to method-suffixed files
(ensemble_<method>.npz / ensemble_<method>_params.json). Whichever method
equals config.ENSEMBLE_METHOD (the production combiner
CoralFilterEnsembler.predict() actually loads) is ALSO saved to the
canonical, unsuffixed ensemble.npz/ensemble_params.json via
CoralFilterEnsembler.save_models() (which also resaves the submodel .pth
files). If config.ENSEMBLE_METHOD isn't in METHODS_TO_FIT, the canonical
files are left untouched.
"""

import os
import json
from glob import glob

import numpy as np
import torch

from config import FILTER_MODELS_DIR, MASK_DATA_PATH, M, SPLIT, RES, ALL_ENSEMBLE_METHODS
from filter import CoralFilterEnsembler, CoralFilter

LOGIT_CACHE_PATH = os.path.join(FILTER_MODELS_DIR, "submodel_logits_cache.npz")
USE_LOGIT_CACHE = True
# Which ensemble method(s) to fit/refit -- edit this directly to fit only a
# subset (e.g. METHODS_TO_FIT = ("nn",)) instead of every method in
# config.ALL_ENSEMBLE_METHODS.
METHODS_TO_FIT = ("reweight",) #ALL_ENSEMBLE_METHODS

# Extra starting point for the "reweight" fit, run alongside (not instead
# of) the usual random restarts -- see ClassReweightingOptimizer.fit's
# `init` argument. Set False to use random restarts only.
USE_V1_REWEIGHT_INIT = True

# Reference ensemble parameters (raw, pre-softmax), converted to
# ClassReweightingOptimizer's (alpha, W) via softmax(alpha, axis=0) and
# softmax(weights, axis=1) -- a lossless reconstruction, since W's overall
# per-model scale is unidentifiable in Eq. 3 (invariant to w_m -> c*w_m).
_V1_RAW_ALPHA = np.array([0.1788, 0.0431, 0.7765, 0.7211, 0.8744])
_V1_RAW_WEIGHTS = np.array([
    [0.8461, -1.5423, -0.43,   -1.3004, -0.5508, -0.8452, 0.1981, -0.724,  0.0662, -0.0692, -3.1862, -0.4719, -0.6114, 1.032,  -0.3565, -0.2227, -0.012],
    [-0.8856,-1.0064,  0.5213,  0.6038,  0.149,  -0.7613, 0.7178, -1.0958,-1.7784, -0.8274, -2.849,  -0.4572,  0.221, -0.211,  -0.2231, -0.1335, -0.3032],
    [0.9488, -1.7382,  0.5742,  0.3843, -0.7013, -0.6014, 0.5774, -0.231, -1.1547, -1.2876, -2.3541,  0.7474,  0.3516, 0.5202,  1.1449,  0.7429, -0.2235],
    [-2.3228,-0.4466,  0.5931,  0.705,  -0.1997, -1.0331,-3.3717, -0.4573, 0.1701,  0.0115, -1.5733, -0.0711,  1.2418, 0.0099,  0.4866, -0.4315, -1.1065],
    [-1.7427,-1.375,  -0.9271, -0.7692,  0.0041,  0.0079,-2.1586, -0.2456,-0.3476, -0.2411, -2.4085, -2.1649,  0.588, -1.3101, -0.7866, -0.9682,  0.0896],
])


def _v1_reweight_init():
    def softmax(x, axis=-1):
        z = x - x.max(axis=axis, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=axis, keepdims=True)
    return softmax(_V1_RAW_ALPHA), softmax(_V1_RAW_WEIGHTS, axis=1)


def save_ensemble_only(ensembler, method, dir):
    """Same arrays CoralFilterEnsembler.save_models() writes, but to
    method-suffixed filenames and without resaving the submodel .pth
    files (unchanged across ensemble_method values)."""
    param_names = ensembler._ENSEMBLE_PARAM_NAMES[method]
    params = {name: getattr(ensembler.ensemble_model, name) for name in param_names}
    np.savez(os.path.join(dir, f"ensemble_{method}.npz"), ensemble_method=method, **params)
    with open(os.path.join(dir, f"ensemble_{method}_params.json"), "w") as f:
        json.dump({
            "ensemble_method": method,
            **{k: np.round(v, 4).tolist() for k, v in params.items()},
        }, f, indent=2)


def main(methods=METHODS_TO_FIT):
    methods = list(methods)
    unknown = sorted(set(methods) - set(ALL_ENSEMBLE_METHODS))
    if unknown:
        raise ValueError(f"Unknown ensemble method(s) {unknown} -- expected a subset of {ALL_ENSEMBLE_METHODS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    print(f"Fitting: {methods}")

    ensembler = CoralFilterEnsembler(base_dataset=MASK_DATA_PATH, device=device, m=M, split=SPLIT)

    # Load the already-trained submodels
    model_files = sorted(glob(os.path.join(FILTER_MODELS_DIR, "model_*.pth")))
    assert len(model_files) == ensembler.m, (
        f"expected {ensembler.m} submodel files in {FILTER_MODELS_DIR}, found {len(model_files)}"
    )

    ensembler.models = []
    for model_file in model_files:
        model = CoralFilter(
            ensembler.base_model(pretrained=True, dim=ensembler.k, res=RES),
            ensembler.mask_data, device,
            batch_size=ensembler.batch_size, epochs=ensembler.epochs,
            lr=ensembler.lr, weight_decay=ensembler.weight_decay,
            split=ensembler.split, train=False,
        )
        model.load_model(model_file)
        ensembler.models.append(model)
        print(f"Loaded {model_file}")

    # The method save_models() writes to the canonical, unsuffixed
    # ensemble.npz/ensemble_params.json that CoralFilterEnsembler.predict()
    # loads in production -- handled specially below, if requested.
    canonical_method = ensembler.ensemble_method

    # legacy_split=True: recovers the ensemble pool using the submodels'
    # original stratify=one-hot-labels split rather than the current
    # default (argmax-stratify) one -- required for submodels trained
    # before CoralFilterEnsembler._submodel_pool_split existed (see
    # scripts/verify_split_integrity.py). Drop this once submodels are
    # retrained via CoralFilterEnsembler.train().
    for method in methods:
        print(f"\nFitting ensemble_method={method!r}...")
        ensemble_init = _v1_reweight_init() if (method == "reweight" and USE_V1_REWEIGHT_INIT) else None
        ensembler.train_ensemble(cache_path=LOGIT_CACHE_PATH, use_cache=USE_LOGIT_CACHE, ensemble_method=method, legacy_split=True, ensemble_init=ensemble_init)
        ensembler.validate()
        save_ensemble_only(ensembler, method, FILTER_MODELS_DIR)
        print(f"Wrote {FILTER_MODELS_DIR}/ensemble_{method}.npz and ensemble_{method}_params.json")

        if method == canonical_method:
            # Keeps the canonical ensemble.npz/ensemble_params.json (and
            # resaved submodel .pth files) in sync with this method's fit.
            ensembler.save_models(FILTER_MODELS_DIR)
            print(f"Wrote {FILTER_MODELS_DIR}/ensemble.npz and ensemble_params.json (canonical, method={method!r})")
            for name in ensembler._ENSEMBLE_PARAM_NAMES[method]:
                print(f"{name}:", getattr(ensembler.ensemble_model, name))


if __name__ == "__main__":
    main()
