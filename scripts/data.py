"""
Dataset utilities for extracting and processing cropped mask regions
from coral reef images. Includes the MaskLoader class which builds a 
balanced dataset of coral vs. non-coral crops for downstream classification.
"""

from typing import List
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tv2
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import json
from collections import Counter

from transforms import MASK_TRANSFORM, MASK_TRANSFORM_AUGMENT, inv_norm
from config import MASK_SIZE, UPSAMPLE, CLASSES_FILE

class MaskLoader(Dataset):

    def __init__(self, images = None, load_file = None, transform_fn = None, segmentation_model = None, mask_size = None, tolerance = 0.1, balance=False):

        """
        PyTorch dataset class for constructing a labeled image crop dataset for binary coral classification.
        
        Supports two modes: loading preprocessed data from disk or generating it dynamically using a segmentation model.
        
        Creating the data set a priori (as is done here) and then loading in the data in the RAM come training comes with its limitations, but
        in general, greatly speeds up model training times
        
        Ground truth masks are treated as coral instances pertaining to genus/bleached category (defined in remap.json), while non-overlapping predicted masks are labeled as 'noncoral'.

        Args:
            images (List[str], optional): List of image paths to process.
            load_file (str, optional): Path to a saved torch dataset to load.
            segmentation_model (object, optional): Instance of SAM2Segmenter or compatible model for mask generation.
            mask_size (Tuple[int, int], optional): Size to which image crops are resized.
            tolerance (float): Overlap threshold for excluding predicted masks that resemble ground truth.
        """

        self.segmentation_model = segmentation_model
        self.mask_size = mask_size or MASK_SIZE
        self.transform_fn = transform_fn or MASK_TRANSFORM_AUGMENT

        self.labels = None
        self.img_data = None
        self.classes = None

        if load_file:
            self._load_dataset(load_file)
        elif images:
            self._create_dataset(images, tolerance)

        if balance:
            #This helps scale the gradient in our filter models appropriately to learn features of minority classes
            self._oversample(class_cap=UPSAMPLE)

    def __len__(self):
        #Needed for a pytorch data loader
        return len(self.labels)
    
    def __getitem__(self, idx):
        #Needed for a pytorch data loader
        X_i = self.img_data[idx]
        y_i = self.labels[idx]
        return X_i, y_i
    
    def _load_dataset(self, file_path):
        print(f"Loading dataset from {file_path}")
        data = torch.load(file_path, weights_only=False)

        labels = data["labels"]              # one-hot: shape [N, K]
        classes = data["classes"]            # dict: class → index
        raw_data = data["img_data"]

        label_indices = torch.argmax(labels, dim=1).tolist()
        class_counts = Counter(label_indices)

        #Filter out classes with less than 20 observations
        valid_class_indices = {cls for cls, count in class_counts.items() if count >= 20}
        keep_mask = [i for i, idx in enumerate(label_indices) if idx in valid_class_indices]

        self.labels = labels[keep_mask]
        self.raw_data = [raw_data[i] for i in keep_mask]

        old_to_new = {
            old_idx: new_idx for new_idx, old_idx in enumerate(sorted(valid_class_indices))
        }

        new_label_indices = torch.tensor([old_to_new[idx] for idx in label_indices if idx in valid_class_indices])
        inv_classes = {v: k for k, v in classes.items()}
        self.classes = {
            inv_classes[old]: new for old, new in old_to_new.items()
        }

        print(f"Remaining classes: {len(self.classes)}")

        self.class_distribution = self.get_class_distribution(self.labels)

        self.img_data = self.augment(data["img_data"], self.transform_fn)
        self.labels = torch.nn.functional.one_hot(new_label_indices, num_classes=len(old_to_new))

    def _oversample(self, class_cap=1000):
        print(f"Applying oversampling to balance minority classes (cap = {class_cap})...")

        labels_np = torch.argmax(self.labels, dim=1).numpy()
        class_counts = np.bincount(labels_np)
        new_imgs = []
        new_labels = []

        index_to_class = {v: k for k, v in self.classes.items()}
        for class_idx, count in enumerate(class_counts):
            if count >= class_cap:
                continue

            indices = np.where(labels_np == class_idx)[0]
            needed = class_cap - count

            sampled_idx = np.random.choice(indices, size=needed, replace=True)

            for i in tqdm(range(needed), desc=f"Oversampling class {index_to_class.get(class_idx, class_idx)}"):
                img = self.raw_data[sampled_idx[i]]
                label = self.labels[sampled_idx[i]]

                new_imgs.append(img)
                new_labels.append(label)

        if new_imgs:
            new_imgs_batch = torch.stack(new_imgs)     # Shape: [B, C, H, W]
            new_labels_batch = torch.stack(new_labels) # Shape: [B, K]

            new_augmented_imgs = self.augment(new_imgs_batch, self.transform_fn)

            self.raw_data = torch.cat([self.raw_data, new_imgs_batch])
            self.img_data = torch.cat([self.img_data, new_augmented_imgs])
            self.labels   = torch.cat([self.labels, new_labels_batch])

    def resample(self):
        #resample data from augmentation distribution
        self.img_data = self.augment(self.raw_data, self.transform_fn)

    @staticmethod
    def augment(images: torch.tensor, transform_fn):
        #This allows us to apply post-processing augmentations in a flexible way (so we don't have to apply augmentations in the _create_dataset method)
        return transform_fn(images.clone())

    def save_data(self, save_path):
        print(f"Saving dataset to {save_path}")
        torch.save({
            "classes": self.classes,
            "labels": self.labels,
            "img_data": self.img_data, 
        }, save_path)
        #Save classes to json CLASSES_FILE
        with open(CLASSES_FILE, 'w') as f:
            json.dump(self.classes, f, indent=4)
            
    def _create_dataset(self, images: List[str], tolerance=0.1, min_area=400):

        """
        Builds a labeled dataset by extracting coral (positive) and non-coral (negative) crops.

        Args:
            images (List[str]): List of image paths to process.
            tolerance (float): Overlap threshold for labeling predicted masks as non-coral.
            min_area (int): Minimum area required for predicted masks to be considered.
        """

        #Coral classification labels
        labels = []
        img_data = []

        gt_masks_set = []
        genus_labels_set = []
        bleached_labels_set = []

        for img_path in tqdm(images, desc="Loading Ground Truth Masks"):
            genus_labels, bleach_labels, gt_masks = self.segmentation_model.get_gt_masks(img_path)
            
            genus_labels_set.append(genus_labels)
            bleached_labels_set.append(bleach_labels)
            
            gt_masks_set.append(gt_masks)

        for i, x in tqdm(enumerate(images), desc="Extrapolating masks from image"):
            
            #Extract gt masks
            np_image = self.segmentation_model.resize_image(
                np.array(Image.open(x)), self.segmentation_model.img_size
            ).transpose((2, 0, 1)) #Resize if necessary
            torch_image = torch.tensor(np_image) #Needed to convert binary masks into a dataset of torch tensors

            #Here is the image if you want to see it (before transformations)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(torch_image.numpy().transpose((1, 2, 0)))
            # plt.show()

            for j, segmentation in enumerate(gt_masks_set[i]):
                torch_mask = torch.tensor(segmentation)
                img_data.append(self.extract(torch_image, torch_mask, self.mask_size))

                #Each ground truth mask automatically gets a genus/bleached label.
                #Dead and algae (among the masks in the gt set) will get 'noncoral' as defined in remap.json
                labels.append(genus_labels_set[i][j] + ":" + ("bleached" if bleached_labels_set[i][j] == 1 else "healthy"))

            #Create mask predictions
            all_pred_masks = self.segmentation_model.predict(img_path=x, init_models=(i==0), keep_all=True)
            all_pred_masks_filtered = [mask for mask in all_pred_masks if mask['segmentation'].sum() >= min_area]
            pred_masks = np.stack([mask['segmentation'] for mask in all_pred_masks_filtered])

            #Use the predicted masks to extract objects that are separate objects from the gt masks
            gt_map = np.any(gt_masks_set[i], axis=0)
            for j, pred_mask in enumerate(pred_masks):
                
                overlap = np.logical_and(pred_mask, gt_map).sum()/pred_mask.sum()

                # tolerance = how much of the overlapping crop (difference between predicted mask and actual mask) that we'd be willing to accept as a 'separate' object
                if overlap <= tolerance:
                    torch_mask = torch.tensor(pred_mask)
                    img_data.append(self.extract(torch_image, torch_mask, self.mask_size))
                    labels.append("noncoral")

        #Normalize all noncoral:bleached, noncoral:healthy (if any exist) -> noncoral to align with extrapolated mask labels
        labels = ['noncoral' if 'noncoral' in label else label for label in labels]
        
        #Create a label dictionary for the ML model
        self.classes = {label: idx for idx, label in enumerate(sorted(set(labels)))}

        #Save the data as tensors objects for pytorch models
        self.img_data = torch.stack(img_data)
        self.raw_data = torch.stack(img_data)
        self.labels = torch.nn.functional.one_hot(torch.tensor([self.classes[label] for label in labels]), num_classes=len(self.classes))
        self.class_distribution = self.get_class_distribution(self.labels)

    @staticmethod
    def get_class_distribution(labels: torch.tensor):
        class_indices = torch.argmax(labels, dim=1)                                   # shape: [N]
        class_counts = torch.bincount(class_indices, minlength=labels.shape[1])       # shape: [K]
        class_distribution = class_counts.float() / class_counts.sum()

        return class_distribution

    @staticmethod
    def extract(img, mask, output_size, padding=5, tf=None):
        
        segmentation = img * mask

        #create bounding box
        coords = torch.nonzero(mask, as_tuple=False)
        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values

        #crop to mask
        crop = segmentation[:, y_min:y_max, x_min:x_max]

        #make square by padding to center
        #padding parameter (=5) is relative to the scale of the image (1024x1024) in this case
        h, w = crop.shape[1:]
        max_side = max(h, w)
        pad_y = (max_side - h) // 2 + padding*((max_side-h) == 0)
        pad_x = (max_side - w) // 2 + padding*((max_side-w) == 0)
        padding = (pad_x, pad_y)

        squared_img = TF.pad(crop, padding, fill=0)

        if tf is not None:
            return tf(TF.resize(squared_img, [output_size[0], output_size[1]]))
        return TF.resize(squared_img, [output_size[0], output_size[1]])

    @staticmethod
    def visualize_mask(mask, invnorm=False):

        if invnorm:
            mask = inv_norm(mask)

        plt.figure(figsize=(4, 4))
        plt.imshow(mask.cpu().numpy().transpose((1, 2, 0)))
        plt.show()