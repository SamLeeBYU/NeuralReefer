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
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

from transforms import MASK_TRANSFORM, CLAHETransform
from config import MASK_SIZE

class MaskLoader(Dataset):

    def __init__(self, images = None, load_file = None, segmentation_model = None, mask_size = None, tolerance = 0.1, transform=None, randomAugment=False):

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
            transform (callable, optional): Transformations to apply to image crops for regularization and normalization (for ResNet)
            randomAugment (bool): Whether to apply random augmentations to the loaded dataset.
        """

        self.segmentation_model = segmentation_model
        self.mask_size = mask_size or MASK_SIZE

        self.labels = None
        self.img_data = None
        self.classes = None

        self.transform = transform or MASK_TRANSFORM

        if load_file:
            self._load_dataset(load_file, randomAugment)
        elif images:
            self._create_dataset(images, tolerance)    

    def __len__(self):
        #Needed for a pytorch data loader
        return len(self.labels)
    
    def __getitem__(self, idx):
        #Needed for a pytorch data loader
        X_i = self.img_data[idx]
        y_i = self.labels[idx]
        return X_i, y_i
    
    def _load_dataset(self, file_path, randomAugment=False):
        print(f"Loading dataset from {file_path}")
        data = torch.load(file_path, weights_only=False)

        #This helps the coral filter models learn on 'new' data that's representative of the true distribution of masks
        #This is necessary for ensemble learning
        self.transform, data["img_data"] = self.augment(data["img_data"], self.transform, random=randomAugment)

        self.labels = data["labels"]
        self.img_data = data["img_data"]
        self.classes = data["classes"]

        #Shuffle labels (is the model actually learning?)
        # indices = np.arange(len(self.labels))
        # np.random.shuffle(indices)

        # self.labels = self.labels[indices]

    @staticmethod
    def augment(images: torch.tensor, base_transforms, random=True):
        if random:
            #We create random augmentations that vary by each generation
            #We don't apply any colorization / color augmentations as those may be sensitive to the training data used
            augment_transforms = transforms.Compose([
                transforms.RandomRotation(degrees=np.random.uniform(5, 45)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=np.random.uniform(5, 45)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=np.random.uniform(5, 45)),
            ])
        else:
            #A 'less' random version of the above augmentation
            augment_transforms = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15)
            ])

        #This instance's new transformation are whatever we added from the random augmentation + the base augmentation that every
        #instance of this class experiences

        #This allows us to apply post-processing augmentations in a flexible way (so we don't have to apply augmentations in the _create_dataset method)
        return  transforms.Compose(base_transforms.transforms + augment_transforms.transforms), augment_transforms(images)

    def save_data(self, save_path):
        print(f"Saving dataset to {save_path}")
        torch.save({
            "classes": self.classes,
            "labels": self.labels,
            "img_data": self.img_data, 
        }, save_path)

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
                img_data.append(self.extract(torch_image, torch_mask, self.mask_size, self.transform))

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
                    img_data.append(self.extract(torch_image, torch_mask, self.mask_size, self.transform))
                    labels.append("noncoral")

        #Normalize all noncoral:bleached, noncoral:healthy (if any exist) -> noncoral to align with extrapolated mask labels
        labels = ['noncoral' if 'noncoral' in label else label for label in labels]
        
        #Create a label dictionary for the ML model
        self.classes = {label: idx for idx, label in enumerate(sorted(set(labels)))}

        #Save the data as tensors objects for pytorch models
        self.img_data = torch.stack(img_data)
        self.labels = torch.nn.functional.one_hot(torch.tensor([self.classes[label] for label in labels]), num_classes=len(self.classes))

    @staticmethod
    def extract(img, mask, output_size, transform_fn, padding=5):
        
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

        #NOTE: the transform_fn are the base augmentations we apply to each mask (namely normalizing the pixel values for ResNet)
        
        return transform_fn(TF.resize(squared_img, [output_size[0], output_size[1]]))
