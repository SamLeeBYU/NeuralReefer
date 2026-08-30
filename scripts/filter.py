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
from classifier import CoralClassifier, EMEnsembleOptimizer, FocalLoss, create_loss_fn

from config import (
    VERBOSE, MASK_SIZE, FILTER_MODELS_DIR, CLASSES_FILE, PATIENCE, RES, NEG_WEIGHT, ENSEMBLE_SPLIT
)
from transforms import MASK_TRANSFORM, MASK_TRANSFORM_AUGMENT

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

    def __init__(self, model: CoralClassifier, dataset: MaskLoader, device=None, loss_fn=nn.CrossEntropyLoss(), epochs=15, batch_size=32, lr=1e-3, weight_decay=1e-4, split=0.3, train=True, seed=42, bootstrap=False):

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # if VERBOSE:
        #     print(f"using device: {self.device}")
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
            train_idx, val_idx = train_test_split(
                np.arange(len(dataset)),
                test_size = split,
                stratify = self.dataset.labels.numpy(),
                #It is imperative that each of these submodels are trained on a bootstrapped distribution
                #of the *same* training data to appropriately explore the sampling distribution
                random_state=seed
            )
            #array([45836, 65131, 20203, ..., 54551, 34767, 67438], shape=(51338,))
            #array([50013, 25265, 32029, ..., 29129, 63953, 48043], shape=(22003,))

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

def extract_submodel_logits(models, dataset, k, batch_size=128, cache_path=None,
                             use_cache=True, verbose=True):
    """
    Forward-passes every model in `models` over the ENTIRE `dataset` (all
    samples, not a subset), using the deterministic MASK_TRANSFORM (not the
    training-time MASK_TRANSFORM_AUGMENT) -- sets dataset.transform_fn and
    calls dataset.resample() internally -- so re-running this reproduces the
    same numbers every time.

    This is the expensive step shared by CoralFilterEnsembler.train_ensemble()
    and scripts/generate_filter_reports.py: each only needs a different
    index-based slice of the SAME full-dataset logits (train_ensemble()'s
    30% ensemble pool is a strict subset of generate_filter_reports.py's
    70/30 split of the whole dataset), not a different computation. Both
    can point cache_path at the same file, so only the first of the two to
    run ever pays this cost.

    Args:
        models (list[CoralFilter]): the trained, frozen submodels.
        dataset (MaskLoader): full dataset to evaluate every model over.
        k (int): number of classes (for cache shape validation).
        cache_path (str, optional): .npz path to cache/reuse the result at.
            None disables caching.
        use_cache (bool): set False to force recomputation even if a cache
            file exists (e.g. after retraining a submodel).

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

    dataset.transform_fn = MASK_TRANSFORM
    dataset.resample()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    N = len(dataset)
    logits = np.zeros((N, len(models), k), dtype=np.float32)
    y_true = np.argmax(dataset.labels.numpy(), axis=1)

    for m, filter_model in tqdm(enumerate(models), "Evaluating models", total=len(models)):
        filter_model.model.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(loader):
                X = X.to(filter_model.device)
                pred = filter_model.model(X).squeeze(1)
                logits[batch * batch_size: batch * batch_size + len(X), m, :] = pred.cpu().numpy()

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, logits=logits, y_true=y_true)
        if verbose:
            print(f"Cached submodel logits to {cache_path}")

    return logits, y_true


class CoralFilterEnsembler:

    def __init__(self, base_dataset: str, base_model = None, device=None, m=5, epochs=15, batch_size=32, lr=1e-3, weight_decay=1e-4, split=0.1, seed=42):

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.base_dataset = base_dataset

        self.mask_data = None
        if self.base_dataset is not None: #e.g. if we're training the model
            self.mask_data = MaskLoader(load_file=self.base_dataset, balance=True)
            #self.mask_loader = DataLoader(self.mask_data, batch_size=batch_size)

            self.classes = self.mask_data.classes
            with open(CLASSES_FILE, 'w') as f:
                json.dump(self.classes, f, indent=4)

            #Preliminary estimates suggest that our images contain 30-40% coral cover, on average
            #However, the data that will the models will be trained with is inflated with negative labels to increase sample size (resulting in 80:20 ratio of negative to positive labels)
            #Note that the inflated data *is* representative of the data that will ultimately be fed into the trained models because of *where* (the stage at which) the model is implemented

            #labels = self.mask_data.labels.numpy()
            # base_rates = np.mean(labels, axis=0)
            # weights = 1.0 / base_rates
            # weights = weights / weights.sum() * len(self.classes)
            # self.weight = torch.tensor(weights, device=device)

            #self.weight = torch.ones(len(self.classes), device=self.device)
            #self.weight[self.classes["noncoral"]] = NEG_WEIGHT
            #self.loss_fn = create_loss_fn(self.weight, use_focal=True, gamma=2.0, reduction='sum')

            #A better accuracy can generally be induced by balancing the weights (as the model regresses to the base rate), but that usually hinders recall
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

        self.models = []
        self.ensemble_model = EMEnsembleOptimizer(self.m, self.k)

    def train(self, ensemble_split=0.1):
        for i in range(self.m):
            print(f"Creating model {i+1}/{self.m}")
            model_i = CoralFilter(self.base_model(pretrained=True, dim=self.k, res=RES), self.mask_data, self.device,
                                  create_loss_fn(use_focal=False), #Generic CCE loss for each submodule
                                  batch_size=self.batch_size, epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay, split=self.split, train=True, seed=self.seed, bootstrap=True)
            model_i.train(patience=PATIENCE)
            self.models.append(model_i)

        self.train_ensemble(ensemble_split)

    def train_ensemble(self, ensemble_split=ENSEMBLE_SPLIT, cache_path=None, use_cache=True):
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
        """
        #Now we weight each model that gives the best OOS ensemble performance
        full_logits, full_y_true = extract_submodel_logits(
            self.models, self.mask_data, self.k,
            batch_size=self.batch_size, cache_path=cache_path, use_cache=use_cache,
        )

        #We need to train the ensembler on the set of data that the submodels have not seen to maintain independence between models
        _, idx = train_test_split(
            np.arange(len(self.mask_data)),
            test_size=self.split,
            stratify=full_y_true,
            random_state=self.seed
        )
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

        self.ensemble_model = EMEnsembleOptimizer(self.m, K)
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

    def save_models(self, dir=None):
        dir = dir or FILTER_MODELS_DIR
        if not os.path.exists(dir):
            os.makedirs(dir)
        for i, model in tqdm(enumerate(self.models), desc="Saving models"):
            model.save_model(os.path.join(dir, f"model_{i+1}.pth"))
        np.savez(os.path.join(dir, "ensemble.npz"), alpha=self.ensemble_model.alpha, weights=self.ensemble_model.weights)

        # Human-readable export of the same two arrays. Unlike the old
        # torch-based EnsembleOptimizer (whose raw nn.Parameter values were
        # pre-softmax and not directly interpretable -- softmax output is
        # always in (0,1), but the old ensemble_params.json contained
        # negative numbers, so it must have been dumping the unnormalized
        # parameters), EMEnsembleOptimizer.alpha/.weights ARE the exact
        # values used in predict_proba(): alpha sums to 1, each row of
        # weights sums to 1. So this export is a direct, faithful view of
        # the fitted ensemble, not merely a raw parameter dump.
        with open(os.path.join(dir, "ensemble_params.json"), "w") as f:
            json.dump({
                "alpha": np.round(self.ensemble_model.alpha, 4).tolist(),
                "weights": np.round(self.ensemble_model.weights, 4).tolist(),
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
        self.ensemble_model = EMEnsembleOptimizer(self.m, self.k)
        self.ensemble_model.alpha = ensemble_data["alpha"]
        self.ensemble_model.weights = ensemble_data["weights"]

    def predict(self, masks, img=None, img_path: str = None, mask_size=None):
        mask_size = mask_size or MASK_SIZE
        logits = np.zeros((len(masks), self.m, self.k), dtype=np.float32)
        for m in tqdm(range(self.m), desc="Classifying"):
            model = self.models[m]
            logits[:,m,:] = model.predict(masks, img, img_path, mask_size)

        return self.ensemble_model.predict_proba(logits)

    @staticmethod
    def get_class_names(labels, class_dict):
        index_to_class = {v: k for k, v in class_dict.items()}
        return [index_to_class[label] for label in labels]