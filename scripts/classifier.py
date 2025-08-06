"""
This module defines the neural network architectures for each classifier
"""

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

class EnsembleOptimizer(torch.nn.Module):
    def __init__(self, M, K):
        super().__init__()
        self.M = M
        self.K = K
        self.weights = torch.nn.Parameter(torch.ones(M, K) / K)

    def forward(self, X):  # X: [N, M, K]
        # Normalize each model's weights across K classes
        norm_weights = self.weights / self.weights.sum(dim=1, keepdim=True)  # shape: [M, K]
        z = torch.einsum('nmk,mk->nk', X, norm_weights)  # Weighted sum
        return z

class EnsembleOptimizer(torch.nn.Module):
    def __init__(self, M, K):
        super().__init__()
        self.M = M
        self.K = K

        #Class-wise weights W: [M, K]
        self.weights = torch.nn.Parameter(torch.ones(M, K) / K)

        #Model-level weights alpha: [M]
        self.alpha = torch.nn.Parameter(torch.ones(M))

    def forward(self, X):  #X: [N, M, K]
        #Normalize W_m across K
        W_normalized = self.weights / self.weights.sum(dim=1, keepdim=True)  # [M, K]

        #Normalize alpha across M
        alpha_normalized = torch.softmax(self.alpha, dim=0)  # [M]
        alpha_expanded = alpha_normalized.unsqueeze(1)  # [M, 1]

        #Compute combined weight: T_{m,k} = alpha_m * W_{m,k}
        combined_weights = alpha_expanded * W_normalized  # [M, K]

        #Weighted sum over models: z_i = sum_m sum_k T_{m,k} * z_i^{(m,k)}
        z = torch.einsum('nmk,mk->nk', X, combined_weights)  # [N, K]
        return z


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
        probs = F.softmax(inputs, dim=1)

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