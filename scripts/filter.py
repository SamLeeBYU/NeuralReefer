"""
Implements training and ensemble logic for binary coral classification.
Defines:
- CoralFilter: Wrapper to train and evaluate a single CoralClassifier model
- CoralFilterEnsembler: Bootstrapped ensemble of CoralFilter models, combined via a
  weighted-aggregation ensemble fit with Expectation-Maximization (see EMEnsembleOptimizer
  in classifier.py)
"""

import os
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import Counter

import torch
import torch.nn as nn
from torchvision.io import decode_image
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, confusion_matrix

from utils import convert_json_compat
from data import MaskLoader
from classifier import CoralClassifier, EMEnsembleOptimizer, LinearStackingOptimizer, AdamEnsembleOptimizer, ClassReweightingOptimizer, MultinomialRegressionOptimizer, NeuralNetEnsembleOptimizer, FocalLoss, create_loss_fn

from config import (
    VERBOSE, MASK_SIZE, FILTER_MODELS_DIR, CLASSES_FILE, PATIENCE, RES, NEG_WEIGHT, ENSEMBLE_SPLIT,
    N_STARTS, ENSEMBLE_METHOD, ABLATION_SUBMODEL, MASK_TRANSFORM_AUGMENT_SEED
)
from transforms import MASK_TRANSFORM_AUGMENT, seeded_rng

class CoralFilter:

    """
    CoralFilter

    Classifier that wraps a CoralClassifier model and provides training for n_classes.
    Defines evaluation, and inference utilities. Used to filter out false positive coral masks from SAM2 segmentation proposals.

    This class supports bootstrapping, stratified train/validation splitting, and early stopping.
    Training is done using a cross-entropy or focal loss with softmax activation, optimized via Adam.

    Args:
        model (CoralClassifier): Neural network model for binary classification.
        dataset (MaskLoader): Dataset of coral/non-coral image masks and labels.
        device (torch.device, optional): CUDA or CPU device. Auto-detects if not provided.
        loss_fn (nn.Module): Loss function. Defaults to binary CrossEntropyLoss.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate.
        weight_decay (float): L2 penalty for Adam optimizer.
        split (float): Validation set proportion for stratified sampling.
        train (bool): Whether to immediately split and prepare dataloaders.
        seed (int): Random seed for reproducibility.
        bootstrap (bool): If True, resample training data with replacement.
    """

    def __init__(self, model: CoralClassifier, dataset: MaskLoader, device=None, loss_fn=nn.CrossEntropyLoss(), epochs=15, batch_size=32, lr=1e-3, weight_decay=1e-4, split=0.3, train=True, seed=42, bootstrap=False, train_idx=None, val_idx=None):

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)

        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.dataset = dataset or MaskLoader()
        if self.dataset.classes:
            self.classes = self.dataset.classes
        else:
            with open(CLASSES_FILE, 'r') as f:
                self.classes = json.load(f)
        self.noncoral_class = self.classes['noncoral']

        if train:
            if train_idx is None or val_idx is None:
                # Fallback for standalone use (no explicit split passed in) -- stratifies on
                # the raw one-hot label array, which does NOT produce the same partition as
                # CoralFilterEnsembler.train_ensemble()'s stratify=argmax(labels) call (sklearn's
                # StratifiedShuffleSplit consumes its random_state differently depending on that
                # representation). Callers that need this split to agree with the ensembler's
                # held-out pool -- i.e. CoralFilterEnsembler.train() -- MUST pass train_idx/val_idx
                # explicitly (see CoralFilterEnsembler._submodel_pool_split).
                train_idx, val_idx = train_test_split(
                    np.arange(len(dataset)),
                    test_size = split,
                    stratify = self.dataset.labels.numpy(),
                    #Each submodel must be trained on a bootstrapped resample of the SAME pool
                    #to properly explore the sampling distribution
                    random_state=seed
                )

            # The pre-bootstrap pool this submodel was allowed to draw from, and its
            # complementary held-out pool -- stored so callers/tests can verify disjointness
            # against other submodels/the ensembler without reaching into DataLoader internals.
            self.train_pool_idx = np.asarray(train_idx)
            self.val_idx = np.asarray(val_idx)

            if bootstrap:
                train_idx = np.random.choice(train_idx, size=len(train_idx), replace=True)

            train_set = Subset(self.dataset, train_idx)
            val_set = Subset(self.dataset, val_idx)

            self.train_loader = DataLoader(train_set, batch_size=self.batch_size)
            self.test_loader = DataLoader(val_set, batch_size=self.batch_size)

            if VERBOSE:
                print(f"Stratified the data into {len(train_idx)} observations for training and {len(val_idx)} observations for validation.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def train(self, patience=5):

        """
        Trains the CoralClassifier model on the training set with early stopping.

        Args:
            patience (int): Number of epochs to wait without improvement before stopping.

        Tracks training loss, accuracy, and performs validation after each epoch.
        Saves the best model parameters based on validation loss.
        """

        size = len(self.train_loader)

        bad_epochs = 0
        best_val_loss = float('inf')
        #Loop through the dataset self.epochs # of times
        for epoch in range(self.epochs):

            print(f"Epoch {epoch+1}\n-------------------------------")
            self.dataset.resample()
            epoch_loss = 0.0

            correct = 0
            total = 0

            self.model.train()

            #Feed each batch of data through the model, compute loss, and apply back-propogation
            for batch, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X).squeeze(1)
                loss = self.loss_fn(pred, y.float())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

                y_class = torch.argmax(y, dim=1)
                probs = torch.softmax(pred, dim=1)           # shape: [B, K]
                preds = torch.argmax(probs, dim=1)           # shape: [B]
                correct += (preds == y_class).sum().item()
                total += y.size(0)

                if batch % 8 == 0:
                    loss, current = loss.item(), (batch+1)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            avg_loss = epoch_loss / len(self.train_loader)
            accuracy = correct / total if total > 0 else 0
            print(f"Epoch {epoch+1}, Avg. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            val_loss = self.test()

            #Early stopping mechanism
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                best_model = self.model.state_dict()
            else:
                bad_epochs += 1
                print(f"Validation loss did not improve for {bad_epochs} epoch(s).")

            if bad_epochs >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

        self.model.load_state_dict(best_model)

    def test(self):

        """
        Evaluates the model on the validation set.

        Returns:
            float: Average validation loss.

        Also prints accuracy and recall statistics. Recall is defined as:
            TP / (TP + FN) for identifying coral / noncoral objects
        """

        self.model.eval()
        num_batches = len(self.test_loader)
        test_loss = 0

        correct = 0
        total = 0

        true_positive = 0
        false_negative = 0
        false_positive = 0

        with torch.no_grad():
            for batch, (X, y) in enumerate(self.test_loader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X).squeeze(1)
                test_loss += self.loss_fn(pred, y.float()).item()

                y_class = torch.argmax(y, dim=1)
                probs = torch.softmax(pred, dim=1)           # shape: [B, K]
                preds = torch.argmax(probs, dim=1)           # shape: [B]
                correct += (preds == y_class).sum().item()
                total += y.size(0)

                true_positive += ((preds != self.noncoral_class) & (y_class != self.noncoral_class)).sum().item()
                false_negative += ((preds == self.noncoral_class) & (y_class != self.noncoral_class)).sum().item()
                false_positive += ((preds != self.noncoral_class) & (y_class == self.noncoral_class)).sum().item()

        test_loss /= num_batches
        accuracy = correct / total if total > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

        print(f"Test Error:\n"
            f"Avg loss: {test_loss:.6f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f}\n")

        return test_loss

    def predict(self, masks, img = None, img_path: str = None, mask_size=None, transform_fn=None):

        mask_size = mask_size or MASK_SIZE
        # MASK_TRANSFORM_AUGMENT, not the deterministic MASK_TRANSFORM: every submodel is
        # only ever trained under MASK_TRANSFORM_AUGMENT, so a plain resize with no crop
        # is out-of-distribution for them. Reproducibility comes from the caller wrapping
        # this in transforms.seeded_rng -- see CoralFilterEnsembler.predict/extract_submodel_logits.
        transform_fn = transform_fn or MASK_TRANSFORM_AUGMENT

        if img_path is not None:
            img = decode_image(img_path)

        masks = torch.tensor(masks)
        X = torch.stack([
            self.dataset.extract(img, mask, mask_size, tf=transform_fn).to(self.device)
            for mask in masks
        ])

        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
        return pred.cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

def _binary_coral_metrics(y_true_idx, y_pred_idx, noncoral_class):
    """Accuracy over all classes, plus precision/recall/F2 for the coral vs.
    noncoral binary sub-problem. Duplicated from
    generate_filter_reports.binary_coral_metrics (not imported -- that
    script already imports FROM this module, so importing back would invert
    the dependency) so extract_submodel_logits can print a per-submodel
    sanity check as each one finishes, rather than only after all finish."""
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


def extract_submodel_logits(models, dataset, k, batch_size=128, cache_path=None,
                             use_cache=True, verbose=True, seed=MASK_TRANSFORM_AUGMENT_SEED):
    """
    Forward-passes every model in `models` over the ENTIRE `dataset` (all
    samples, not a subset), using MASK_TRANSFORM_AUGMENT (not the
    deterministic MASK_TRANSFORM) -- every submodel is only ever trained
    under MASK_TRANSFORM_AUGMENT, so the deterministic transform is
    out-of-distribution for them. Reproducibility comes from `seed` (via
    transforms.seeded_rng), not from the transform being deterministic.

    Applies the transform to each mask individually (working from
    dataset.raw_data/raw_idx directly, not dataset.img_data/resample()) so
    every mask gets its own independent random draw, matching live
    inference's per-mask granularity (CoralFilter.predict -> MaskLoader.extract)
    rather than resample()'s chunked batch-transform, which shares one draw
    across an entire chunk. Each submodel gets its own continuing draw
    within the one seeded_rng scope below, matching
    CoralFilterEnsembler.predict()'s live-inference behavior.

    This is the expensive step shared by CoralFilterEnsembler.train_ensemble()
    and scripts/generate_filter_reports.py: each only needs a different
    index-based slice of the SAME full-dataset logits, not a different
    computation, so both can point cache_path at the same file and only the
    first to run pays this cost. NOTE: a cache built under a different
    transform is silently stale (same shape, wrong values) -- delete it or
    pass use_cache=False if the transform changes.

    Args:
        models (list[CoralFilter]): the trained, frozen submodels.
        dataset (MaskLoader): full dataset to evaluate every model over.
        k (int): number of classes (for cache shape validation).
        cache_path (str, optional): .npz path to cache/reuse the result at.
            None disables caching.
        use_cache (bool): set False to force recomputation even if a cache
            file exists (e.g. after retraining a submodel).
        seed (int): seeds MASK_TRANSFORM_AUGMENT for reproducibility;
            defaults to config.MASK_TRANSFORM_AUGMENT_SEED.

    Returns:
        logits ([N, len(models), k] float32), y_true ([N] int),
        N = len(dataset).
    """
    cache_hit = cache_path is not None and use_cache and os.path.exists(cache_path)

    if cache_hit:
        if verbose:
            print(f"Loading cached submodel logits from {cache_path}")
        cached = np.load(cache_path)
        logits, y_true = cached["logits"], cached["y_true"]
        expected_shape = (len(dataset), len(models), k)
        if logits.shape != expected_shape:
            raise ValueError(
                f"Cached logits at {cache_path} have shape {logits.shape} but expected "
                f"{expected_shape} for this dataset/model set -- the cache is stale "
                f"(e.g. from a different dataset or submodels). Delete it or pass use_cache=False."
            )
        return logits, y_true

    N = len(dataset)
    logits = np.zeros((N, len(models), k), dtype=np.float32)
    y_true = np.argmax(dataset.labels.numpy(), axis=1)
    noncoral_class = dataset.classes.get("noncoral") if hasattr(dataset, "classes") else None
    raw_idx = dataset.raw_idx  # handles oversampling's repeated indices, same as resample()

    with seeded_rng(seed):
        for m, filter_model in enumerate(tqdm(models, desc="Evaluating models", disable=not verbose)):
            filter_model.model.eval()
            with torch.no_grad():
                batch_bar = tqdm(range(0, N, batch_size), desc=f"Model {m+1}/{len(models)}", leave=False, disable=not verbose)
                for start in batch_bar:
                    end = min(start + batch_size, N)
                    raw_batch = dataset.raw_data[raw_idx[start:end]]
                    # One independent MASK_TRANSFORM_AUGMENT draw per mask,
                    # continuing this scope's seeded RNG sequence -- see docstring.
                    X = torch.stack([MASK_TRANSFORM_AUGMENT(img) for img in raw_batch]).to(filter_model.device)
                    pred = filter_model.model(X).squeeze(1)
                    logits[start:end, m, :] = pred.cpu().numpy()

            # Printed as soon as this submodel finishes so a broken submodel
            # (wrong weights, garbage logits, NaNs) is caught immediately.
            if verbose:
                pred_idx_m = np.argmax(logits[:, m, :], axis=1)
                if noncoral_class is not None:
                    acc, prec, rec, f2 = _binary_coral_metrics(y_true, pred_idx_m, noncoral_class)
                    print(f"  Submodel {m+1}/{len(models)} sanity check: "
                          f"accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} F2={f2:.4f}")
                else:
                    acc = (pred_idx_m == y_true).mean()
                    print(f"  Submodel {m+1}/{len(models)} sanity check: accuracy={acc:.4f} "
                          f"(no 'noncoral' class found on dataset -- skipping precision/recall/F2)")

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, logits=logits, y_true=y_true)
        if verbose:
            print(f"Cached submodel logits to {cache_path}")

    return logits, y_true


class CoralFilterEnsembler:

    def __init__(self, base_dataset: str, base_model = None, device=None, m=5, epochs=15, batch_size=32, lr=1e-3, weight_decay=1e-4, split=0.1, seed=42, ensemble_method=ENSEMBLE_METHOD):

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.base_dataset = base_dataset

        self.mask_data = None
        if self.base_dataset is not None: #e.g. if we're training the model
            self.mask_data = MaskLoader(load_file=self.base_dataset, balance=True)

            self.classes = self.mask_data.classes
            with open(CLASSES_FILE, 'w') as f:
                json.dump(self.classes, f, indent=4)
        else:
            with open(CLASSES_FILE, 'r') as f:
                self.classes = json.load(f)

        self.noncoral_class = self.classes["noncoral"]

        self.base_model = base_model or CoralClassifier
        self.m = m
        self.k = len(self.classes)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.split = split
        self.seed = seed
        self.ensemble_method = ensemble_method

        self.models = []
        self.ensemble_model = self._make_ensemble_model()
        self.last_ablation_proba = None  # set by predict() when config.ABLATION_SUBMODEL is set

    def _make_ensemble_model(self, method=None):
        method = method or self.ensemble_method
        if method == "em":
            return EMEnsembleOptimizer(self.m, self.k)
        elif method == "linear":
            return LinearStackingOptimizer(self.m, self.k)
        elif method == "adam":
            return AdamEnsembleOptimizer(self.m, self.k)
        elif method == "reweight":
            return ClassReweightingOptimizer(self.m, self.k)
        elif method == "multinomial":
            return MultinomialRegressionOptimizer(self.m, self.k)
        elif method == "nn":
            return NeuralNetEnsembleOptimizer(self.m, self.k)
        else:
            raise ValueError(f"Unknown ensemble_method '{method}' (expected 'em', 'linear', 'adam', 'reweight', 'multinomial', or 'nn').")

    def _submodel_pool_split(self, legacy=False):
        """
        The single, authoritative 70/30 split between the submodels' own
        training pool and the pool reserved for the ensemble stage --
        computed HERE ONLY and reused by both .train() (passed explicitly
        into every CoralFilter(...) it creates) and .train_ensemble() (as
        the ensemble pool), so the two stages cannot disagree about which
        masks are held out. Stratifying on argmax(labels) (integer class
        index) here does NOT reproduce the same split as stratifying on the
        raw one-hot label array (as CoralFilter.__init__'s own fallback
        split does when no train_idx/val_idx is passed in) -- sklearn's
        StratifiedShuffleSplit consumes its random_state differently
        depending on that representation, even with an identical
        seed/test_size. Passing indices through explicitly (as .train()
        does) avoids relying on both call sites reconstructing the same split.

        legacy=True reproduces the old stratify=one-hot-labels call instead
        -- use only when evaluating/retraining the ensemble against
        submodels that were trained under that split and cannot be
        retrained. Never use legacy=True for freshly-trained submodels.
        """
        y_true = np.argmax(self.mask_data.labels.numpy(), axis=1)
        stratify = self.mask_data.labels.numpy() if legacy else y_true
        train_pool_idx, val_pool_idx = train_test_split(
            np.arange(len(self.mask_data)),
            test_size=self.split,
            stratify=stratify,
            random_state=self.seed,
        )
        return train_pool_idx, val_pool_idx

    def train(self, ensemble_split=0.1, n_starts=N_STARTS, ensemble_method=None):
        train_pool_idx, val_pool_idx = self._submodel_pool_split()
        for i in range(self.m):
            print(f"Creating model {i+1}/{self.m}")
            model_i = CoralFilter(self.base_model(pretrained=True, dim=self.k, res=RES), self.mask_data, self.device,
                                  create_loss_fn(use_focal=False), #Generic CCE loss for each submodule
                                  batch_size=self.batch_size, epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay, split=self.split, train=True, seed=self.seed, bootstrap=True,
                                  train_idx=train_pool_idx, val_idx=val_pool_idx)
            model_i.train(patience=PATIENCE)
            self.models.append(model_i)

        # legacy_split=False: these submodels were just trained on
        # _submodel_pool_split()'s canonical (non-legacy) pool above, so
        # train_ensemble() must use that SAME pool, not the legacy one.
        self.train_ensemble(ensemble_split, n_starts=n_starts, ensemble_method=ensemble_method, legacy_split=False)

    def train_ensemble(self, ensemble_split=ENSEMBLE_SPLIT, cache_path=None, use_cache=True, n_starts=N_STARTS, ensemble_method=None, legacy_split=False, ensemble_init=None):
        """
        Args:
            ensemble_split (float): fraction of the ensemble-data pool held
                out as the ensemble's own out-of-sample set.
            cache_path (str, optional): passed through to
                extract_submodel_logits() -- caches the FULL-dataset submodel
                logits (not just this ensembler's 30% pool), so the same
                cache file can also be reused by scripts/generate_filter_reports.py,
                which needs the complementary 70% too. None (the default)
                disables caching entirely -- unchanged behavior for existing
                callers.
            use_cache (bool): set False to force recomputation even if a
                cache file exists at cache_path (e.g. after retraining a
                submodel, when the cache would be stale).
            n_starts (int): number of independent random initializations
                passed through to fit() on whichever ensemble_model is used
                -- the best of these by weighted log-likelihood is kept.
                Ignored for ensemble_method "linear" (a single closed-form
                solve) and "multinomial" (a single exact Newton solve to a
                unique global optimum -- see MultinomialRegressionOptimizer)
                -- neither has anything to restart.
            ensemble_method (str, optional): "em", "linear", "adam",
                "reweight", or "multinomial" -- overrides self.ensemble_method
                (set in __init__, from config.py's ENSEMBLE_METHOD) for this
                call only. See classifier.py's EMEnsembleOptimizer,
                LinearStackingOptimizer, AdamEnsembleOptimizer,
                ClassReweightingOptimizer, and MultinomialRegressionOptimizer.
            legacy_split (bool): passed through to _submodel_pool_split().
                Set True ONLY when self.models were loaded (not just
                trained in this same call) from submodels trained before
                the split-consistency fix -- see _submodel_pool_split's
                docstring. A .train() call already sets this correctly
                (False) when it calls train_ensemble() internally.
            ensemble_init (tuple(alpha0, W0), optional): passed through to
                ClassReweightingOptimizer.fit()'s `init` argument -- an
                EXTRA starting point run alongside the usual n_starts random
                restarts (not instead of them), see that method's docstring.
                Only meaningful for ensemble_method == "reweight" (the only
                method whose fit() accepts it); ignored otherwise.
        """
        if ensemble_method is not None:
            self.ensemble_method = ensemble_method
        #Now we weight each model that gives the best OOS ensemble performance
        full_logits, full_y_true = extract_submodel_logits(
            self.models, self.mask_data, self.k,
            batch_size=self.batch_size, cache_path=cache_path, use_cache=use_cache,
        )

        #We need to train the ensembler on the set of data that the submodels have not seen to maintain independence between models
        _, idx = self._submodel_pool_split(legacy=legacy_split)
        logits, y_true = full_logits[idx], full_y_true[idx]

        ensemble_train_idx, ensemble_test_idx = train_test_split(
            np.arange(len(idx)),
            test_size=ensemble_split,
            stratify=y_true,
            random_state=self.seed
        )

        # From here on the ensemble-combination stage is pure NumPy: the M
        # submodels are already trained and frozen, so their logits are just
        # fixed input data for the EM fit (see EMEnsembleOptimizer.fit).
        self.X_train, self.y_train_idx = logits[ensemble_train_idx], y_true[ensemble_train_idx]
        self.X_test, self.y_test_idx = logits[ensemble_test_idx], y_true[ensemble_test_idx]

        K = self.k

        #Alternatively, if you know the true distribution of coral classes across images you may substitute the class weights here
        nu = np.ones(K)
        nu[self.noncoral_class] = NEG_WEIGHT

        self.ensemble_model = self._make_ensemble_model()
        if self.ensemble_method in ("em", "reweight"):
            # Both are EM fits with a provably monotonic ascent on the
            # observed-data log-likelihood within a trajectory (exact
            # closed-form M-steps for "em"; closed-form alpha + MM
            # fixed-point W for "reweight" -- see ClassReweightingOptimizer)
            # -- no overfitting risk to guard against with a held-out check
            # DURING the fit, unlike "adam" below. self.X_test/self.y_test_idx
            # stay untouched until validate().
            fit_kwargs = {"init": ensemble_init} if (self.ensemble_method == "reweight" and ensemble_init is not None) else {}
            self.ensemble_model.fit(self.X_train, self.y_train_idx, nu, n_starts=n_starts, seed=self.seed, **fit_kwargs)
        elif self.ensemble_method in ("adam", "nn"):
            # Unlike "em"/"reweight" above and "linear"/"multinomial" below,
            # both are non-convex first-order fits (torch.optim.Adam), so
            # both use held-out early stopping via val_logits/val_y_idx --
            # self.X_test/self.y_test_idx ARE used during fitting here (for
            # model selection, not gradient computation), unlike every other
            # method where that split stays untouched until validate().
            self.ensemble_model.fit(
                self.X_train, self.y_train_idx, nu, n_starts=n_starts, seed=self.seed,
                val_logits=self.X_test, val_y_idx=self.y_test_idx,
            )
        else:
            # "linear" (closed-form) and "multinomial" (single exact Newton
            # solve to a unique global optimum) both need only one fit call,
            # no n_starts, no held-out early stopping.
            self.ensemble_model.fit(self.X_train, self.y_train_idx, nu, seed=self.seed)

    def validate(self):

        probs = self.ensemble_model.predict_proba(self.X_test)  # [N, K], numpy
        y_class = self.y_test_idx
        preds = np.argmax(probs, axis=1)

        correct = int((preds == y_class).sum())
        total = len(y_class)

        #For coral/non-coral
        true_positive = int(np.logical_and(preds != self.noncoral_class, y_class != self.noncoral_class).sum())
        false_negative = int(np.logical_and(preds == self.noncoral_class, y_class != self.noncoral_class).sum())
        false_positive = int(np.logical_and(preds != self.noncoral_class, y_class == self.noncoral_class).sum())

        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        accuracy = correct/total

        print(f"Ensemble model trained with out-of-sample accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    # Which arrays each ensemble_method's optimizer is parameterized by -- see
    # classifier.py for what each one means (EMEnsembleOptimizer/AdamEnsembleOptimizer:
    # alpha/beta, same multinomial-logit model, fit differently; LinearStackingOptimizer:
    # W/b, closed-form ridge-regression weights + bias; ClassReweightingOptimizer:
    # alpha/W, the original Hadamard-reweight-and-renormalize model's mixture
    # weights + per-submodel positive class-reweight vectors; MultinomialRegressionOptimizer:
    # beta only -- no alpha, see its docstring). save_models()/load_models()
    # use this to save/reconstruct the right arrays generically.
    _ENSEMBLE_PARAM_NAMES = {
        "em": ("alpha", "beta"),
        "adam": ("alpha", "beta"),
        "linear": ("W", "b"),
        "reweight": ("alpha", "W"),
        "multinomial": ("beta",),
        # 3 layers, matching NeuralNetEnsembleOptimizer's default
        # hidden_sizes=(32, 16) -- MK -> 32 -> 16 -> K. Changing hidden_sizes
        # would need updating this list to match (number of layers, not
        # sizes -- shapes are inferred from the saved arrays themselves).
        "nn": ("W0", "b0", "W1", "b1", "W2", "b2"),
    }

    def save_models(self, dir=None):
        dir = dir or FILTER_MODELS_DIR
        if not os.path.exists(dir):
            os.makedirs(dir)
        for i, model in tqdm(enumerate(self.models), desc="Saving models"):
            model.save_model(os.path.join(dir, f"model_{i+1}.pth"))

        # ensemble_method itself is saved alongside the arrays so load_models()
        # knows which class (and which arrays) to reconstruct.
        param_names = self._ENSEMBLE_PARAM_NAMES[self.ensemble_method]
        params = {name: getattr(self.ensemble_model, name) for name in param_names}

        np.savez(os.path.join(dir, "ensemble.npz"), ensemble_method=self.ensemble_method, **params)

        # Human-readable export of the same arrays -- a direct, faithful view of
        # the fitted ensemble, not merely a raw parameter dump.
        with open(os.path.join(dir, "ensemble_params.json"), "w") as f:
            json.dump({
                "ensemble_method": self.ensemble_method,
                **{k: np.round(v, 4).tolist() for k, v in params.items()},
            }, f, indent=2)

    def load_models(self, dir=None, dim=None):
        dim = dim or self.k
        dir = dir or FILTER_MODELS_DIR
        model_files = glob(os.path.join(dir, "model_*.pth"))
        if len(model_files) < 1:
            raise ValueError(f"No model files found in {dir}. Please train models first.")
        else:
            self.models = []
            for i in tqdm(range(self.m), desc="Loading models"):
                model_file = model_files[i]
                model = CoralFilter(self.base_model(pretrained=True, dim=dim, res=RES), self.mask_data, self.device,
                                    batch_size=self.batch_size, epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay, split=self.split, train=False)
                model.load_model(model_file)
                self.models.append(model)
        ensemble_data = np.load(os.path.join(dir, "ensemble.npz"))
        # str(...) : np.savez stores the ensemble_method string as a 0-d array
        self.ensemble_method = str(ensemble_data["ensemble_method"]) if "ensemble_method" in ensemble_data else "em"
        self.ensemble_model = self._make_ensemble_model()
        for name in self._ENSEMBLE_PARAM_NAMES[self.ensemble_method]:
            setattr(self.ensemble_model, name, ensemble_data[name])

    def predict(self, masks, img=None, img_path: str = None, mask_size=None):
        mask_size = mask_size or MASK_SIZE
        logits = np.zeros((len(masks), self.m, self.k), dtype=np.float32)
        # Seeded ONCE here (not inside CoralFilter.predict) so the 5 submodels'
        # MASK_TRANSFORM_AUGMENT draws continue one reproducible sequence
        # rather than each independently restarting from the same point.
        with seeded_rng(MASK_TRANSFORM_AUGMENT_SEED):
            for m in tqdm(range(self.m), desc="Classifying"):
                model = self.models[m]
                logits[:,m,:] = model.predict(masks, img, img_path, mask_size)

        # Stashed for diagnostics (e.g. re-running a different ensemble_model's
        # predict_proba against these SAME raw per-submodel logits).
        self.last_logits = logits

        # Ablation study (see config.py's ABLATION_SUBMODEL): from this SAME
        # forward pass, also compute one submodel's own softmax(logits), so a
        # single CNN can be compared against the full ensemble without a
        # second inference pass. Stashed as an attribute (read via
        # SAM2Segmenter.predict) rather than returned, so the ensemble
        # prediction remains this method's sole return value.
        self.last_ablation_proba = None
        if ABLATION_SUBMODEL is not None:
            # (Not all ensemble_model classes define a _softmax helper --
            # e.g. LinearStackingOptimizer doesn't -- so this is self-contained
            # rather than borrowed from self.ensemble_model.)
            z = logits[:, ABLATION_SUBMODEL - 1, :]
            z = z - z.max(axis=-1, keepdims=True)
            e = np.exp(z)
            self.last_ablation_proba = e / e.sum(axis=-1, keepdims=True)

        return self.ensemble_model.predict_proba(logits)

    @staticmethod
    def get_class_names(labels, class_dict):
        index_to_class = {v: k for k, v in class_dict.items()}
        return [index_to_class[label] for label in labels]