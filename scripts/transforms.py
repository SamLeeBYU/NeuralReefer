"""
Image preprocessing custom transforms for coral reef imagery.
Includes CLAHE (Contrast Limited Adaptive Histogram Equalization)
and utilities for enhancing underwater image quality.
"""

import cv2
import numpy as np
from torchvision import transforms

MASK_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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