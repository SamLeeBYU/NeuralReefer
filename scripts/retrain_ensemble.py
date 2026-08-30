"""
Retrains ONLY the ensemble-combination stage (alpha, W -- fit via the new
EM algorithm in classifier.EMEnsembleOptimizer) using the already-trained
CNN submodel weights on disk (model_1.pth .. model_M.pth). Does not retrain
the CNN submodels themselves.

Needed because CoralFilterEnsembler's optimization method was rewritten
from a torch/Adam-trained EnsembleOptimizer to a pure-NumPy EM/MM fit
(EMEnsembleOptimizer): the two use incompatible parameterizations, so the
existing models/<...>/ensemble.pth (Adam-fit) cannot be loaded by the new
code, and load_models() will raise FileNotFoundError looking for
ensemble.npz until this is run once.

Writes ensemble.npz and ensemble_params.json via the real
CoralFilterEnsembler.save_models() (also re-saves the 5 submodel .pth
files with identical, unchanged weights -- harmless, keeps this script's
output in parity with the normal training path rather than a bespoke
partial save).

Run from the repository root:
    python scripts/retrain_ensemble.py
"""

import os
from glob import glob

import torch

from config import FILTER_MODELS_DIR, MASK_DATA_PATH, M, SPLIT, RES
from filter import CoralFilterEnsembler, CoralFilter


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    ensembler = CoralFilterEnsembler(base_dataset=MASK_DATA_PATH, device=device, m=M, split=SPLIT)

    # Load the already-trained submodels directly -- same loop as
    # CoralFilterEnsembler.load_models(), minus the (currently missing)
    # ensemble.npz load at the end.
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

    ensembler.train_ensemble()
    ensembler.validate()
    ensembler.save_models(FILTER_MODELS_DIR)
    print(f"Wrote {FILTER_MODELS_DIR}/ensemble.npz and ensemble_params.json")
    print("alpha:", ensembler.ensemble_model.alpha)


if __name__ == "__main__":
    main()
