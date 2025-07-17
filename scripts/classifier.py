"""
This module defines the neural network architectures for each classifier
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CoralClassifier(nn.Module):

    # This module implements the CoralClassifier class, a deep convolutional neural network (CNN) optimized for 
    # binary classification of segmented coral reef image crops. The model leverages a pretrained ResNet18 architecture, 
    # replacing the final fully connected (FC) layer with a custom multi-layer perceptron (MLP) head composed of 
    # ReLU activations and dropout regularization for improved generalization.

    # The last layer may be modified depending on the dimension of the output using the argument 'dim'
    
    def __init__(self, pretrained=True, dim=1):

        super(CoralClassifier, self).__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
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