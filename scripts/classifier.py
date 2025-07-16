"""
This module defines the neural network architectures for each classifier
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

#Coral/Not Coral Image (Mask) Classifier
class CoralClassifier(nn.Module):

    # This module implements the CoralClassifier class, a deep convolutional neural network (CNN) optimized for 
    # binary classification of segmented coral reef image crops. The model leverages a pretrained ResNet18 architecture, 
    # replacing the final fully connected (FC) layer with a custom multi-layer perceptron (MLP) head composed of 
    # ReLU activations and dropout regularization for improved generalization.
    
    def __init__(self, pretrained=True):

        super(CoralClassifier, self).__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        self.backbone.fc = nn.Sequential(

            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)