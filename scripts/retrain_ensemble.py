"""
Retrains ONLY the ensemble-combination stage (alpha, W -- fit via the new
EM algorithm in classifier.EMEnsembleOptimizer) using the already-trained
CNN submodel weights on disk (model_1.pth .. model_M.pth). Does not retrain
the CNN submodels themselves.

The expensive part of this (forward-passing every submodel over the FULL
mask dataset, deterministically -- see filter.extract_submodel_logits) is
cached to LOGIT_CACHE_PATH. A second run reuses that cache and skips
straight to the EM fit (which itself takes well under a second per start),
so re-fitting with different EMEnsembleOptimizer settings (e.g. more
n_starts) after the first run is cheap. This cache covers the whole
dataset (not just this script's own 30% ensemble pool), so
scripts/generate_filter_reports.py can point at the SAME file and reuse it
too -- whichever of the two scripts runs first pays the extraction cost
once, for both. Delete the cache file (or set USE_LOGIT_CACHE = False
below) to force recomputation -- needed if the submodels themselves change.
"""

import os
from glob import glob

import torch

from config import FILTER_MODELS_DIR, MASK_DATA_PATH, M, SPLIT, RES
from filter import CoralFilterEnsembler, CoralFilter

# Shared with scripts/generate_filter_reports.py -- same path, same cache.
LOGIT_CACHE_PATH = os.path.join(FILTER_MODELS_DIR, "submodel_logits_cache.npz")
USE_LOGIT_CACHE = True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    ensembler = CoralFilterEnsembler(base_dataset=MASK_DATA_PATH, device=device, m=M, split=SPLIT)

    # Load the already-trained submodels directly
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

    ensembler.train_ensemble(cache_path=LOGIT_CACHE_PATH, use_cache=USE_LOGIT_CACHE)
    ensembler.validate()

    # save_models() writes both ensemble.npz and ensemble_params.json (and
    # re-saves the 5 submodel .pth files with identical, unchanged weights,
    # harmless) -- calling it directly here, rather than a bespoke np.savez,
    # keeps ensemble_params.json in sync with ensemble.npz on every rerun.
    ensembler.save_models(FILTER_MODELS_DIR)
    print(f"Wrote {FILTER_MODELS_DIR}/ensemble.npz and ensemble_params.json")
    print("alpha:", ensembler.ensemble_model.alpha)


if __name__ == "__main__":
    main()
