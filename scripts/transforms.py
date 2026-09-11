"""
Image preprocessing custom transforms for coral reef imagery.
Includes CLAHE (Contrast Limited Adaptive Histogram Equalization)
and utilities for enhancing underwater image quality.
"""

import cv2
import numpy as np
from contextlib import contextmanager

import torch
import torchvision.transforms.v2 as tv2

from config import MASK_SIZE

@contextmanager
def seeded_rng(seed):
    """
    Temporarily pins torch's global RNG to `seed`, restoring whatever state
    it had beforehand on exit -- makes a call site's use of
    MASK_TRANSFORM_AUGMENT (or anything else drawing from torch's global
    RNG) reproducible without affecting unrelated randomness elsewhere in a
    longer-running process.
    """
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)

MASK_TRANSFORM = tv2.Compose([
    tv2.ToDtype(torch.float32, scale=True),
    tv2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

MASK_TRANSFORM_AUGMENT = tv2.Compose([
    tv2.RandomResizedCrop(
        size=MASK_SIZE,
        scale=(0.8, 1.0),       
        ratio=(0.9, 1.1),        
        antialias=True
    ),
    tv2.RandomAffine(
        degrees=(0, 2),
        scale=(0.95, 1.05),     
        shear=(0, 5)
    ),
    tv2.RandomRotation(degrees=(0, 45)),
    tv2.RandomHorizontalFlip(p=0.5),
    tv2.RandomRotation(degrees=(0, 45)),
    tv2.RandomVerticalFlip(p=0.5),
    tv2.ToDtype(torch.float32, scale=True),
    tv2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

MASK_TRANSFORM_AUGMENT_AGGRESSIVE = tv2.Compose([
    tv2.RandomResizedCrop(size=MASK_SIZE, antialias=True),
    tv2.RandomAffine(
        degrees=(0, 5),
        scale=(0.9, 1.1),
        shear=(0, 10)
    ),
    tv2.RandomRotation(degrees=(5, 45)),
    tv2.RandomHorizontalFlip(p=0.5),
    tv2.RandomAffine(
        degrees=(0, 5),
        scale=(0.9, 1.1),
        shear=(0, 10)
    ),
    tv2.RandomRotation(degrees=(5, 45)),
    tv2.RandomVerticalFlip(p=0.5),
    tv2.RandomAffine(
        degrees=(0, 5),
        scale=(0.9, 1.1),
        shear=(0, 10)
    ),
    tv2.RandomRotation(degrees=(5, 45)),
    tv2.ToDtype(torch.float32, scale=True),
    tv2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def inv_norm(tensor):
    """
    Inverts the normalization applied by Normalize with
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    Args:
        tensor (torch.Tensor): A normalized tensor of shape [3, H, W]

    Returns:
        torch.Tensor: A tensor of the same shape with normalization inverted
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean