"""
This module defines the neural network architectures for each classifier
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights
)

import torch.nn.functional as F

model_dict = {18: resnet18, 34: resnet34, 50: resnet50, 101: resnet101, 152: resnet152}
weights_dict = {18: ResNet18_Weights, 34: ResNet34_Weights, 50: ResNet50_Weights, 101: ResNet101_Weights, 152: ResNet152_Weights}

class CoralClassifier(nn.Module):

    # This module implements the CoralClassifier class, a deep convolutional neural network (CNN) optimized for
    # binary classification of segmented coral reef image crops. The model leverages a pretrained ResNet architecture,
    # replacing the final fully connected (FC) layer with a custom multi-layer perceptron (MLP) head composed of
    # ReLU activations and dropout regularization for improved generalization.

    # The last layer may be modified depending on the dimension of the output using the argument 'dim'

    def __init__(self, pretrained=True, dim=1, res=18):

        super(CoralClassifier, self).__init__()

        try:
            model = model_dict[res]
            weights = weights_dict[res]
        except KeyError:
            raise ValueError("Unsupported ResNet depth.")

        self.backbone = model(weights=weights.DEFAULT if pretrained else None)
        self.backbone.fc = self._create_fc(self.backbone.fc.in_features, dim=dim)

    @staticmethod
    def _create_fc(in_feautres, dim=1):

        return nn.Sequential(

            nn.Linear(in_feautres, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(128, dim)
        )

    def forward(self, x):
        return self.backbone(x)

class EMEnsembleOptimizer:
    """
    Weighted-aggregation ensemble combiner, fit via Expectation-Maximization
    with a closed-form Minorize-Maximize (MM) update for the per-submodel
    class-reweighting vectors. Pure NumPy -- no torch, no autodiff, no Adam.
    (Replaces the old gradient-descent `EnsembleOptimizer`.)

    Model (see the EM/MM derivation this implements):
        p_i^(m)        = softmax(logit_i^(m))                          -- fixed, pretrained submodel output
        w_m in R^K_++                                                   -- per-submodel class reweight
        tilde_p_{i,k}^(m)(w_m) = w_{m,k} p_{i,k}^(m) / sum_j w_{m,j} p_{i,j}^(m)   -- Eq. 1
        p_hat_i = sum_m alpha_m tilde_p_i^(m)(w_m),   alpha in simplex  -- Eq. 2

    The M submodels are already trained and frozen by the time this runs --
    their logits are just fixed input data for the EM fit, so there is
    nothing here that needs gradients or an optimizer in the torch sense.
    """

    def __init__(self, M, K):
        self.M = M
        self.K = K
        self.alpha = np.full(M, 1.0 / M)
        self.weights = np.ones((M, K))

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def _reweighted_probs(self, probs, weights=None):
        """Eq. 1. probs: [N, M, K] (already softmaxed) -> tilde_p: [N, M, K]."""
        weights = self.weights if weights is None else weights
        weighted = weights[None, :, :] * probs               # [N, M, K]
        return weighted / weighted.sum(axis=2, keepdims=True)

    def predict_proba(self, logits):
        """logits: [N, M, K] raw submodel logits -> p_hat: [N, K] (Eq. 2)."""
        probs = self._softmax(logits)
        tilde_p = self._reweighted_probs(probs)
        return (self.alpha[None, :, None] * tilde_p).sum(axis=1)

    def fit(self, logits, y_idx, nu, max_iter=200, mm_iters=5, tol=1e-6,
            pseudocount=1e-3, seed=42, verbose=True):
        """
        Fits alpha and W by EM. `logits` ([N, M, K], raw submodel outputs)
        and `y_idx` ([N], integer true class per sample) are fixed
        throughout -- only alpha and W are estimated. `nu` ([K]) is the
        per-class loss weight (nu_{y_i} in the derivation, e.g. down-
        weighting the noncoral class).
        """
        rng = np.random.default_rng(seed)
        N, M, K = logits.shape
        probs = self._softmax(logits)  # p_i^(m): fixed for the whole fit

        # Small random perturbation off uniform, not an exact symmetric
        # start -- a perfectly symmetric init is itself a (poor) stationary
        # point of this non-convex problem.
        self.alpha = np.full(M, 1.0 / M)
        self.weights = np.exp(0.01 * rng.standard_normal((M, K)))

        nu_i = nu[y_idx]  # [N], nu_{y_i}
        idx_n, idx_m = np.arange(N)[:, None], np.arange(M)[None, :]
        prev_ll = -np.inf

        for it in range(max_iter):
            # ---- E-step: responsibilities (Eq. 4) ----
            tilde_p = self._reweighted_probs(probs)                    # [N, M, K]
            tilde_p_y = tilde_p[idx_n, idx_m, y_idx[:, None]]           # [N, M] = tilde_p_{i,y_i}^{(m)}
            joint = self.alpha[None, :] * tilde_p_y                     # [N, M]
            p_hat_y = joint.sum(axis=1, keepdims=True)                  # [N, 1] = p_hat_{i,y_i}
            gamma = joint / p_hat_y                                      # [N, M]

            # Observed-data weighted log-likelihood (sum_i nu_yi log p_hat_iyi),
            # evaluated at the CURRENT (alpha, W) before this iteration's
            # updates -- EM guarantees this is non-decreasing across iterations.
            ll = float(np.sum(nu_i * np.log(np.clip(p_hat_y[:, 0], 1e-300, None))))

            # ---- M-step, alpha: closed form (Eq. 5) ----
            weighted_gamma = nu_i[:, None] * gamma                       # [N, M] = c_i for each m
            self.alpha = weighted_gamma.sum(axis=0) / nu_i.sum()

            # ---- M-step, W: per-submodel MM fixed point (Eq. 9) ----
            for m in range(M):
                c_m = weighted_gamma[:, m]                                # [N]
                n_k = np.bincount(y_idx, weights=c_m, minlength=K) + pseudocount
                w = self.weights[m].copy()
                for _ in range(mm_iters):
                    s_i = np.clip(probs[:, m, :] @ w, 1e-300, None)         # [N]
                    d_k = np.clip((probs[:, m, :] * (c_m / s_i)[:, None]).sum(axis=0), 1e-300, None)  # [K]
                    w = n_k / d_k
                self.weights[m] = w / w.sum()  # renormalize: eq. 1 is scale-invariant in w_m

            if verbose and (it % 10 == 0 or it == max_iter - 1):
                print(f"EM iter {it}: weighted log-lik = {ll:.4f}")

            if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1e-12):
                if verbose:
                    print(f"EM converged at iter {it} (delta log-lik = {ll - prev_ll:.2e})")
                break
            prev_ll = ll

        return self

#This code comes from https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1e-10
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        #In our case, we are reweighting the probabilities across models, so the resuling probability already lives in the simplex
        probs = inputs #F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets = targets.argmax(dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs+self.epsilon)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss
        if torch.isnan(loss).any() or torch.isnan(inputs).any() or torch.isnan(targets).any():
            raise ValueError("NaN detected in loss or model outputs/targets")

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def create_loss_fn(weight=None, use_focal=True, gamma=2.0, task_type='multi-class', reduction='mean'):
    """
    Creates a loss function for classification, supporting both CrossEntropyLoss and FocalLoss.

    :param weight: Tensor of per-class weights (used as `alpha` in FocalLoss or `weight` in CrossEntropyLoss)
    :param use_focal: Whether to use Focal Loss
    :param gamma: Focusing parameter for Focal Loss
    :param task_type: 'binary', 'multi-class', or 'multi-label'
    :param num_classes: Required for multi-class focal loss
    :param reduction: 'mean', 'sum', or 'none'
    """
    if use_focal:
        return FocalLoss(
            gamma=gamma,
            alpha=weight,
            reduction=reduction,
            task_type=task_type,
            num_classes=len(weight)
        )
    else:
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)