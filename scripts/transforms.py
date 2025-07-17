"""
Image preprocessing custom transforms for coral reef imagery.
Includes CLAHE (Contrast Limited Adaptive Histogram Equalization)
and utilities for enhancing underwater image quality.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms

MASK_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#If you do not wish to use the ResNet normalization you may undo the normalization with this method:
def inv_norm(tensor):
    """
    Inverts the normalization applied by torchvision.transforms.Normalize with
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    Args:
        tensor (torch.Tensor): A normalized tensor of shape [3, H, W]

    Returns:
        torch.Tensor: A tensor of the same shape with normalization inverted
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean

class CLAHETransform:
    """
    Applies CLAHE to the luminance channel of an RGB image
    to enhance local contrast in underwater scenes.
    """
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, img):
        img = np.transpose(img.numpy(), (1, 2, 0))
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return img_clahe