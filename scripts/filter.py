"""
Implements training and ensemble logic for binary coral classification.
Defines:
- CoralFilter: Wrapper to train and evaluate a single CoralClassifier model
- CoralFilterEnsembler: Bootstrapped ensemble of CoralFilter models with logistic regression weighting
"""

import os
import json
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
from torchvision.io import decode_image 
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, confusion_matrix

from utils import convert_json_compat
from data import MaskLoader
from classifier import CoralClassifier

from config import VERBOSE, MASK_SIZE, FILTER_MODELS_DIR
from transforms import MASK_TRANSFORM

class CoralFilter:

    """
    CoralFilter

    A coral vs. non-coral binary classifier that wraps a CoralClassifier model and provides training,
    evaluation, and inference utilities. Used to filter out false positives from SAM2 segmentation proposals.

    This class supports bootstrapping, stratified train/validation splitting, and early stopping.
    Training is done using a binary cross-entropy loss with sigmoid activation, optimized via Adam.

    Args:
        model (CoralClassifier): Neural network model for binary classification.
        dataset (MaskLoader): Dataset of coral/non-coral image patches and labels.
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
        if VERBOSE:
            print(f"using device: {self.device}")
        self.model = model.to(self.device)

        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.dataset = dataset or MaskLoader()

        if train:
            train_idx, val_idx = train_test_split(
                np.arange(len(dataset)),
                test_size = split,
                stratify = self.dataset.labels.numpy(),
                #It is imperative that each of these submodels are trained on a bootstrapped distribution
                #of the *same* training data to appropriately explore the sampling distribution 
                random_state=seed
            )

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

                prob = torch.sigmoid(pred)
                preds = (prob >= 0.5).int()
                correct += (preds == y).sum().item()
                total += y.numel()

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
            TP / (TP + FN) for the positive (coral) class.
        """

        self.model.eval()
        num_batches = len(self.test_loader)
        test_loss = 0

        correct = 0
        total = 0

        true_positive = 0
        false_negative = 0

        with torch.no_grad():
            for batch, (X, y) in enumerate(self.test_loader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X).squeeze(1)
                test_loss += self.loss_fn(pred, y.float()).item()

                prob = torch.sigmoid(pred)
                preds = (prob >= 0.5).int()
                correct += (preds == y).sum().item()
                total += y.numel()

                true_positive += ((preds == 1) & (y == 1)).sum().item()
                false_negative += ((preds == 0) & (y == 1)).sum().item()

        test_loss /= num_batches
        accuracy = correct / total if total > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        print(f"Test Error:\n"
            f"Avg loss: {test_loss:.6f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Recall: {recall:.4f}\n")
        
        return test_loss

    def predict(self, masks, img = None, img_path: str = None, mask_size=None):
        
        mask_size = mask_size or MASK_SIZE

        if img_path is not None:
            img = decode_image(img_path)

        masks = torch.tensor(masks)
        X = torch.stack([
            self.dataset.extract(img, mask, mask_size, self.dataset.transform).to(self.device)
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

class CoralFilterEnsembler:

    def __init__(self, base_dataset: str, base_model = None, filter_transform=None, device=None, m=5, loss_fn=nn.CrossEntropyLoss(), epochs=15, batch_size=32, lr=1e-3, weight_decay=1e-4, split=0.1, seed=42):
        
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.base_dataset = base_dataset
        self.filter_transform = filter_transform or MASK_TRANSFORM

        self.mask_data = None
        if self.base_dataset is not None: #e.g. if we're training the model
            self.mask_data = MaskLoader(load_file=self.base_dataset, transform=self.filter_transform, randomAugment=True)
            self.mask_loader = DataLoader(self.mask_data, batch_size=batch_size)

            #Preliminary estimates suggest that our images contain 30-40% coral cover, on average
            #However, the data that will the models will be trained with is inflated with negative labels to increase sample size (resulting in 80:20 ratio of negative to positive labels)
            #Note that the inflated data *is* representative of the data that will ultimately be fed into the trained models because of *where* (the stage at which) the model is implemented
            labels = self.mask_data.labels.numpy()
            p = labels.sum()/len(labels) #proportion of data that are positive labels (e.g. % of masks that are coral in the training dataset)
            self.weight = torch.tensor([(1-p) / p], device=device)

            #A better accuracy can generally be induced by balancing the weights (as the model regresses to the base rate), but that usually hinders recall

        self.base_model = base_model or CoralClassifier
        self.m = m
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.split = split
        self.seed = seed
        
        self.models = []

        self.bias = 0
        self.ensemble_weights = np.ones(self.m, dtype=np.float32) / self.m

    def train(self, k=5, c_int=10, class_weight={0: 1, 1: 4}, submodel_patience=5, ensemble_split=0.3):
        for i in range(self.m):
            print(f"Creating model {i+1}/{self.m}")
            maskloader_i = MaskLoader(load_file=self.base_dataset, transform=self.filter_transform, randomAugment=True)
            model_i = CoralFilter(self.base_model(pretrained=True), maskloader_i, self.device, 
                                  nn.BCEWithLogitsLoss(pos_weight=self.weight), 
                                  batch_size=self.batch_size, epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay, split=self.split, train=True, seed=self.seed, bootstrap=True)
            model_i.train(patience=submodel_patience)
            self.models.append(model_i)

        self.train_weights(k, c_int, class_weight, ensemble_split)

    def train_weights(self, k=5, c_int=10, class_weight={0: 1, 1: 4}, ensemble_split=0.3):

        #Now we weight each model that gives the best OOS ensemble performance

        _, idx = train_test_split(
            np.arange(len(self.mask_data)),
            test_size=self.split,
            stratify=self.mask_data.labels.numpy(),
            random_state=self.seed
        )

        #We need to train the ensembler on the set of data that the submodels have not seen to maintain independence between models
        ensemble_dat = Subset(self.mask_data, idx)
        ensemble_loader = DataLoader(ensemble_dat, batch_size=self.batch_size, shuffle=False)

        logits = np.zeros((len(ensemble_dat), self.m), dtype=np.float32)
        y_true = self.mask_data.labels[idx].numpy()

        for m, filter_model in tqdm(enumerate(self.models), "Evaluating models"):
            filter_model.model.eval()
            with torch.no_grad():
                for batch, (X, y) in enumerate(ensemble_loader):
                    X, y = X.to(self.device), y.to(self.device)
                    pred = filter_model.model(X).squeeze(1)
                    logits[batch * self.batch_size : batch * self.batch_size + len(X), m] = pred.cpu().numpy()

        ensemble_train_idx, ensemble_test_idx = train_test_split(
            np.arange(len(idx)),
            test_size=ensemble_split,
            stratify=y_true,
        )

        X_train, X_test = logits[ensemble_train_idx], logits[ensemble_test_idx]
        y_train, y_test = y_true[ensemble_train_idx], y_true[ensemble_test_idx]

        clf = LogisticRegressionCV(
            Cs=c_int,
            cv=k,
            penalty='l2',
            scoring='roc_auc',
            solver='lbfgs',
            max_iter=1000,
            class_weight=class_weight
        )
        clf.fit(X_train, y_train)

        self.ensemble_weights = clf.coef_.flatten()
        self.bias = clf.intercept_[0]

        #We can now compute a weighted prediction like this:
        ensemble_logit = X_test @ self.ensemble_weights + self.bias
        ensemble_proba = 1 / (1 + np.exp(-ensemble_logit))

        y_hat = (ensemble_proba >= 0.5).astype(int)

        cm = confusion_matrix(y_test, y_hat)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(y_test, ensemble_proba)
        print(f"Ensemble model trained with accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}, Recall: {tp / (tp + fn):.4f}, Precision: {tp / (tp + fp):.4f}")

    def save_models(self, dir=None):
        dir = dir or FILTER_MODELS_DIR
        if not os.path.exists(dir):
            os.makedirs(dir)
        for i, model in tqdm(enumerate(self.models), desc="Saving models"):
            model.save_model(os.path.join(dir, f"model_{i+1}.pth"))

        ensemble_params = {
            "ensemble_weights": self.ensemble_weights.tolist(),
            "bias": self.bias
        }
        with open(os.path.join(dir, "ensemble_params.json"), 'w') as f:
            json.dump(convert_json_compat(ensemble_params), f, indent=4)

    def load_models(self, dir=None):
        dir = dir or FILTER_MODELS_DIR
        model_files = glob(os.path.join(dir, "model_*.pth"))
        if len(model_files) < 1:
            raise ValueError(f"No model files found in {dir}. Please train models first.")
        else:
            self.models = []
            for i in tqdm(range(self.m), desc="Loading models"):
                model_file = model_files[i]
                model = CoralFilter(self.base_model(pretrained=True), self.mask_data, self.device,
                                    batch_size=self.batch_size, epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay, split=self.split, train=False)
                model.load_model(model_file)
                self.models.append(model)
        ensemble_params_file = os.path.join(dir, "ensemble_params.json")
        with open(ensemble_params_file, 'r') as f:
            ensemble_params = json.load(f)
        self.ensemble_weights = np.array(ensemble_params["ensemble_weights"], dtype=np.float32)
        self.bias = ensemble_params["bias"]

    def predict(self, masks, img=None, img_path: str = None, mask_size=None):
        mask_size = mask_size or MASK_SIZE
        logits = np.array([model.predict(masks, img, img_path, mask_size).flatten() for model in self.models[0:self.m]])
        ensemble_logits = logits.T @ self.ensemble_weights + self.bias
        ensemble_proba = 1 / (1 + np.exp(-ensemble_logits))

        return ensemble_proba