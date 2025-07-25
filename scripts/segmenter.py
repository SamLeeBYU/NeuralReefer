"""
Defines the SAM2Segmenter class which wraps a SAM2 model for coral segmentation.
Handles loading checkpoints, parsing COCO annotations, generating masks, and applying
underwater-specific image augmentations (CLAHE, red boost, white balance, etc).

Defines a tuning method that searches the (potentially non-convex) parameter space
to find a (at least locally optimal) set of SAM2 and augmentation hyperparameters.

Also defines CoralSegmenter, a specialized subclass of SAM2Segmenter, preloaded with
project-specific configuration and hyperparameters for use in standard coral workflows.
Coral segmenter uses masks generated from the SAM2Segmenter and perserves the instances
it identifies to be coral masks.
"""

from io import BytesIO
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import itertools
import sys
from collections import defaultdict
import random
import time

import torch
from torchvision.io import decode_image
from PIL import Image
import matplotlib.pyplot as plt

from skopt import gp_minimize, gbrt_minimize, forest_minimize  # Optimization algorithms
from skopt.utils import use_named_args                         # Decorators for search
from skopt.learning import ExtraTreesRegressor                 # Surrogate model for Bayesian Optimization

from config import SAM2_PATH, HYPERPARAM_FILE, REMAP_PATH, MASK_SIZE, IMG_SIZE, VERBOSE, MIN_AREA, CROP_SPACE
from utils import convert_json_compat, restore_prints, suppress_prints, timer

sys.path.append(SAM2_PATH)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

#For the CoralSegmenter
from filter import CoralFilterEnsembler

class SAM2Segmenter:

    def __init__(self, config_path, checkpoint_path, annotation_path = None, remap_dic = None, large_feature_params = None, small_feature_params = None, nr_params=None, device=None, img_size=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if VERBOSE:
            print(f"using device: {self.device}")
        self.model = build_sam2(config_path, checkpoint_path, device=self.device)

        with open(REMAP_PATH, 'r') as f:
            remap_json = json.load(f)

        self.remap_dic = remap_dic or remap_json

        if annotation_path is not None:
            self.parse_annotations(annotation_path)

        self.large_feature_params = large_feature_params
        self.small_feature_params = small_feature_params
        self.nr_params = nr_params

        if self.large_feature_params is None or self.small_feature_params is None or self.nr_params is None:
            self.load_params()

        self.img_size = img_size or IMG_SIZE

    def load_params(self, hyperparam_file: str = HYPERPARAM_FILE):
        with open(hyperparam_file, "r") as f:
            hyperparameters = json.load(f)['params']

        self.large_feature_params = self.large_feature_params or {
            'points_per_side': hyperparameters['large_points_per_side'],
            'points_per_batch': hyperparameters['large_points_per_batch'],
            'pred_iou_thresh': hyperparameters['large_pred_iou_thresh'],
            'stability_score_thresh': hyperparameters['large_stability_score_thresh'],
            'stability_score_offset': hyperparameters['large_stability_score_offset'],
            'crop_n_layers': hyperparameters['large_crop_n_layers'],
            'box_nms_thresh': hyperparameters['large_box_nms_thresh'],
            'crop_n_points_downscale_factor': hyperparameters['large_crop_n_points_downscale_factor'],
            'min_mask_region_area': hyperparameters['large_min_mask_region_area']
        }
        self.small_feature_params = self.small_feature_params or {
            'points_per_side': hyperparameters['small_points_per_side'],
            'points_per_batch': hyperparameters['small_points_per_batch'],
            'pred_iou_thresh': hyperparameters['small_pred_iou_thresh'],
            'stability_score_thresh': hyperparameters['small_stability_score_thresh'],
            'stability_score_offset': hyperparameters['small_stability_score_offset'],
            'crop_n_layers': hyperparameters['small_crop_n_layers'],
            'box_nms_thresh': hyperparameters['small_box_nms_thresh'],
            'crop_n_points_downscale_factor': hyperparameters['small_crop_n_points_downscale_factor'],
            'min_mask_region_area': hyperparameters['small_min_mask_region_area']
        }
        self.nr_params = self.nr_params or {
            'overlap': hyperparameters['overlap'],
            'clipLimit': hyperparameters['clipLimit'],
            'tileGridSize': hyperparameters['tileGridSize'],
            'redBoost': hyperparameters['redBoost'],
            'whiteBalance': hyperparameters['whiteBalance'],
            'gamma': hyperparameters['gamma']
        }

        print(f"Loaded segmentation hyperparamters from {hyperparam_file}")

    # Parse COCO Annotations
    def parse_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        self.annotations = {
            'coco_id': [img['id'] for img in annotations['images']],
            'image_file': [img['file_name'] for img in annotations['images']],
            'image_id': [img['extra']['name'].split('.')[0].split('_')[0] for img in annotations['images']],
            'annotations': []
        }

        categories = np.array([cat['name'] for cat in annotations['categories']])
        #NOTE: this is how our team annotated the bleached corals; you'd need to adjust this for if you want to train the model on a different set of annotations
        categories_cleaned = np.array([cat.replace('_bleach', '') for cat in categories])
        bleached_categories = np.array(["bleach" in cat.lower() for cat in categories])
        for id in tqdm(self.annotations['coco_id'], desc="Parsing annotations"):
            annos = [anno for anno in annotations['annotations'] if anno['image_id'] == id]
            labels = [anno['category_id'] for anno in annos]
            genus = np.array([self.remap_dic.get(cat, cat) for cat in categories_cleaned[labels]])
            bleached = 1*bleached_categories[labels]
            n = len(annos)
            segmentations = [anno['segmentation'][0] for anno in annos]
            self.annotations['annotations'].append({
                'coco_id': id,
                'n': n,
                'genus': genus,
                'bleached': bleached,
                'segmentations': segmentations
            })
        print(f"Parsed {len(self.annotations['annotations'])} annotations from {annotation_path}.")

    @staticmethod
    def gen_params(param_dic, random=False, seed=42, n_samples=50):
       
        keys = param_dic.keys()
        param_grid = [dict(zip(keys, values)) for values in itertools.product(*param_dic.values())]

        #For randomized hyperparameter grid search
        if random:
            np.random.seed(seed)
            param_grid = np.random.choice(param_grid, n_samples, replace=False).tolist()

        return param_grid

    def load_image(self, img_path=None, img: Image = None, whiteBalance=True, redBoost=1, clipLimit=2.0, tileGridSize=8, gamma=0.8):
        if img_path is not None:
            image = cv2.imread(img_path)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            self.image = np.array(img)
        self.image = self.resize_image(self.image, self.img_size)

        return self.augment(self.image, whiteBalance=whiteBalance, redBoost=redBoost, clipLimit=clipLimit, tileGridSize=tileGridSize, gamma=gamma)
    
    @staticmethod
    def resize_image(img: np.array, size) -> np.array:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def get_gt_masks(self, img_path):

        if self.annotations is None:
            raise ValueError("No annotations loaded. Please load annotations first using `parse_annotations` method.")

        img_id = os.path.basename(img_path).split('.')[0].split('_')[0]
        annotation_loc = np.argwhere(np.array(self.annotations['image_id']) == img_id)
        if len(annotation_loc) < 1:
            print(f"No annotations found for image ID {img_id}.")
            return None, np.array([])
        annotation_loc = annotation_loc[0][0]
        masks = [
            np.array(polygon, dtype=np.int32).reshape(-1, 2) for polygon in self.annotations['annotations'][annotation_loc]['segmentations']
        ]
        genus_labels = self.annotations['annotations'][annotation_loc]['genus']
        bleached_labels = self.annotations['annotations'][annotation_loc]['bleached']

        if len(masks) < 1:
            return None, None, np.array([])
        else:
            return genus_labels, bleached_labels, np.stack([cv2.fillPoly(np.zeros((1024, 1024), dtype=np.int8), [mask], 1) > 0 for mask in masks])
        
    # Image Augmentation Methods ####################################################################################
    @staticmethod
    def white_balance(img):
        result = img.copy().astype(np.float32)
        avg = np.mean(result, axis=(0, 1))
        scale = avg.mean() / avg
        result *= scale
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def boost_red_by_blueshift(img, factor_range=(1.0, 1.4)):
        """
        Boosts red in proportion to how dominant the blue channel is.
        factor_range: tuple of (min boost, max boost)
        """
        img = img.astype(np.float32)
        blue = img[:, :, 2]
        red = img[:, :, 0]

        # Compute blue-red dominance map
        ratio = blue / (red + 1e-5)
        ratio = np.clip((ratio - 1), 0, 2) / 2  # normalize to [0,1]
        boost_mask = factor_range[0] + (factor_range[1] - factor_range[0]) * ratio

        img[:, :, 0] *= boost_mask
        return np.clip(img, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_clahe(img, clipLimit=2.0, tileGridSize=(8, 8)):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    @staticmethod
    def gamma_correction(img, gamma=0.8):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)    

    def augment(self, img, whiteBalance=True, redBoost = 1.3, clipLimit=2, tileGridSize=8, gamma=0.85):

        img = img.copy()

        if img is not None:
            if whiteBalance:
                img = self.white_balance(img)
            img = self.boost_red_by_blueshift(img, (0.9, redBoost))
            img = self.apply_clahe(img, clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
            img = self.gamma_correction(img, gamma)

        return img

    ############################################################################################################################

    def tune(self, images, search_space, tuning_model=None, k_samples=25, n_calls=100, verbose=True, out_file=HYPERPARAM_FILE):

        # Base Regressor for Decision Tree Optimization
        tuning_model = tuning_model or "ET"

        # This can also be a more finely tuned regressor like
        # ExtraTreesRegressor(
        #     n_estimators=200,
        #     max_depth=10,
        #     min_samples_split=3,
        #     max_features="sqrt",
        #     n_jobs=-1
        # )

        # Stratified Monte Carlo sampling #################################################################################

        if not verbose: suppress_prints()
        gt_masks_set = []
        for img_path in tqdm(images, desc="Loading Ground Truth Masks"):
            genus_labels, _, gt_masks = self.get_gt_masks(img_path)
            gt_masks_set.append([segmentation for j, segmentation in enumerate(gt_masks) if genus_labels[j] != "noncoral"])
        if not verbose: restore_prints()

        n_annotations = [len(gt_masks) for gt_masks in gt_masks_set]
        bin_edges = np.unique(np.percentile(n_annotations, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
        bin_labels = range(len(bin_edges))
        bins = np.digitize(n_annotations, bin_edges, right=True)

        strata = defaultdict(list)
        for idx, bin_id in enumerate(bins):
            strata[bin_id].append(idx)

        bin_sizes = [len(strata[b]) for b in bin_labels]
        bin_probs = np.array(bin_sizes) / sum(bin_sizes)
        per_bin = np.round(bin_probs * k_samples).astype(int)
        diff = k_samples - per_bin.sum()
        if diff != 0:
            per_bin[np.argmax(per_bin)] += diff

        print("Stratified sampling setup:")
        for b, size, alloc in zip(bin_labels, bin_sizes, per_bin):
            print(f"  Bin {b}: {size} images, allocating {alloc} samples")
        print()

        #########################################################################################################################

        iter_count = [1] 
        tuning_log = []

        @use_named_args(search_space)
        def objective(**params):

            large_keys = [s.name for s in search_space[:9]]
            small_keys = [s.name for s in search_space[9:18]]
            nr_keys = [s.name for s in search_space[18:]]

            large_feature_params = {k: params[k] for k in large_keys}
            small_feature_params = {k: params[k] for k in small_keys}
            nr_params = {k: params[k] for k in nr_keys}

            print(f"→ Trying parameters ({iter_count[0]}/{n_calls}):")
            print("  Large:", large_feature_params)
            print("  Small:", small_feature_params)
            print("  NR:   ", nr_params)

            batch_accuracies = np.zeros(k_samples)
            batch_recalls = np.zeros(k_samples)
            batch_times = np.zeros(k_samples)

            img_batch = []
            gt_masks_batch = []
            for b, n in zip(bin_labels, per_bin):
                stratum = strata[b]
                if len(stratum) == 0:
                    continue
                elif len(stratum) < n:
                    sampled = random.choices(stratum, k=n)
                else:
                    sampled = random.sample(stratum, k=n)
                if not verbose: suppress_prints()
                #print(sampled)
                if not verbose: restore_prints()
                img_batch.extend([images[i] for i in sampled])
                gt_masks_batch.extend([gt_masks_set[i] for i in sampled])

            for i, img in tqdm(enumerate(img_batch), desc="Evaluating Image Batch"):
                self.load_image(img)
                gt_masks = gt_masks_batch[i]
                if len(gt_masks) < 1:
                    gt_masks = np.zeros((1, 1024, 1024), dtype=bool)

                start = time.time()
                if not verbose: suppress_prints()
                pred_masks = self.predict(img, large_feature_params, small_feature_params,
                                        whiteBalance=nr_params["whiteBalance"],
                                        clipLimit=nr_params["clipLimit"],
                                        tileGridSize=nr_params["tileGridSize"],
                                        redBoost=nr_params["redBoost"],
                                        gamma=nr_params["gamma"],
                                        overlap=nr_params["overlap"],
                                        init_models=(i==0))
                if not verbose: restore_prints()
                end = time.time()

                batch_accuracies[i] = self.accuracy(pred_masks, gt_masks)
                batch_recalls[i] = self.recall(pred_masks, gt_masks)
                batch_times[i] = end-start

            acc = batch_accuracies.mean()
            rec = batch_recalls.mean()
            t = batch_times.mean()

            #This is the tentative 'loss function'
            score = 1/3 * (2 * rec + acc)

            print(f"→ ACC: {acc:.4f}, REC: {rec:.4f}, TIME: {t:.2f}, SCORE: {score:.4f}\n\n")
            iter_count[0] += 1

            tuning_log.append({
                "accuracy": float(acc),
                "recall": float(rec),
                "score": float(score),
                "time_per_image": float(t),
                "large_feature_params": large_feature_params,
                "small_feature_params": small_feature_params,
                "nr_params": nr_params
            })

            return -float(score)

        #This could easily be changed to a different minimizer if you wish
        result = forest_minimize(objective, search_space, tuning_model, n_calls=n_calls, n_initial_points=int(n_calls/10)+1, verbose=verbose)

        best_score = -result.fun
        best_params = dict(zip([dim.name for dim in search_space], result.x))

        print("\nBest Score:", best_score)
        for k, v in best_params.items():
            print(f"{k}: {v}")

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(
                convert_json_compat({
                    "score": best_score,
                    "params": best_params,
                    "tuning_log": tuning_log
                }),
                f,
                indent=4
            )

        return best_params
        
    # Automatic SAM2 Calibration Method (ASCM) ################################################################
    def predict(self, img_path=None, img: Image = None, 
                #SAM2 Hyperparameters
                large_feature_params=None, small_feature_params=None, 
                #Data Augmentation Parameters
                whiteBalance=None, redBoost = None, clipLimit = None, tileGridSize = None, gamma = None,
                #Merging parameters
                overlap=None, min_area=None,
                #Runtime arguments
                init_models=True, verbose=True, keep_all=False):
        
        min_area = min_area or MIN_AREA
        overlap = overlap or self.nr_params['overlap']

        #Load in the image using the augmentations needed for the segmentation model
        image = self.load_image(img_path=img_path, img=img, 
                                whiteBalance=whiteBalance or self.nr_params['whiteBalance'], 
                                redBoost=redBoost or self.nr_params['redBoost'], 
                                clipLimit=clipLimit or self.nr_params['clipLimit'], 
                                tileGridSize=tileGridSize or self.nr_params['tileGridSize'], 
                                gamma=gamma or self.nr_params['gamma']) 

        large_feature_params = large_feature_params or self.large_feature_params
        small_feature_params = small_feature_params or self.small_feature_params

        if init_models:
            self.large_feature_generator = SAM2AutomaticMaskGenerator(self.model, **large_feature_params)
            self.small_feature_generator = SAM2AutomaticMaskGenerator(self.model, **small_feature_params)

        with timer(verbose):

            #Find large features
            large_masks = self.large_feature_generator.generate(image)

            #Find small features
            small_masks = self.small_feature_generator.generate(image)

            masks = large_masks + small_masks

            #You could also run different data augmentations on the same model and generate different proposal masks here (and + it 'masks')

            if keep_all:
                if verbose:
                    print(f"Found {len(masks)} objects.")
                return masks #np.stack([mask['segmentation'] for mask in sorted_masks])
            else:
                #Filter out overlapping masks
                return self.merge(masks, min_area, overlap, verbose)[0]

    def merge(self, masks, weights=None, min_area=None, overlap=0.1, verbose=True):
        min_area = min_area or MIN_AREA

        if weights is None:
            weights = np.ones(len(masks))
        
        mask_idx = np.argsort([-m["predicted_iou"]*weights[i] for i, m in enumerate(masks)])
        sorted_masks = [masks[i] for i in mask_idx]
        kept = []

        if len(sorted_masks) > 0:
            seg_map = np.zeros_like(sorted_masks[0]['segmentation'], dtype=np.uint16)
            occupancy_mask = np.zeros_like(sorted_masks[0]['segmentation'], dtype=bool)

            for i in range(len(sorted_masks)):
                mask = sorted_masks[i]['segmentation']

                if mask.sum() < min_area:
                    continue

                if (mask*occupancy_mask).sum()/mask.sum() > overlap: 
                    #print("Overlapping mask... skipping")
                    continue

                mask[occupancy_mask] = 0
                seg_map[mask] = i+1
                occupancy_mask[mask] = 1
                kept.append(mask_idx[i])

            if verbose:
                print(f"Found {len(np.unique(seg_map))-1} objects.")
            return self.one_hot_encode(seg_map), np.array(kept)
        else:
            return np.zeros((1, 1024, 1024), dtype=bool), np.array(kept)
  
    @staticmethod
    def one_hot_encode(seg_map):
        unique_ids = np.unique(seg_map)
        if len(unique_ids) < 2:
            return np.zeros((0, *seg_map.shape), dtype=bool)
        else:
            return np.stack([seg_map == uid for uid in unique_ids if uid > 0])

    @staticmethod
    def recall(pmasks, gt_masks):

        p_map = np.any(pmasks, axis=0)
        gt_map = np.any(gt_masks, axis=0)
            
        tp = np.logical_and(gt_map, p_map).sum()
        fn = np.logical_and(gt_map, np.logical_not(p_map)).sum()

        if (tp + fn) == 0:
            return 1.0 
        return tp / (tp + fn)

    @staticmethod
    def accuracy(pmasks, gt_masks):

        p_map = np.any(pmasks, axis=0)
        gt_map = np.any(gt_masks, axis=0)

        return (p_map == gt_map).mean()

    @staticmethod
    def compute_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    # Visualization Methods ####################################################################################
    # These methods were adapted from the SAM2 documentation/demo code
    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    @staticmethod
    def _show_single_mask(ax, mask, color=(0, 1, 0, 0.4), label=None, borders=True):
        h, w = mask.shape
        img = np.zeros((h, w, 4))
        img[mask] = color
        if borders:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 0, 0.8), thickness=1)
        ax.imshow(img)

        if label:
            # Find bounding box of mask
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                x_center = (xs.min() + xs.max()) // 2
                y_center = (ys.min() + ys.max()) // 2
                ax.text(
                    x_center, y_center,
                    label,
                    fontsize=8,
                    color='black',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.8)
                )

    @staticmethod
    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        ax.imshow(img)

    def show_masks(self, masks, color_map=None, labels=None, return_image=False, show=True, save_path=None):

        if labels is not None:
            assert len(masks) == len(labels), "Each mask must have a corresponding label"
            if color_map is None:
                unique_labels = sorted(set(labels))
                color_map = {
                    label: np.random.rand(3).tolist() + [0.6]
                    for label in unique_labels
                }

        masks = [{'segmentation': mask, 'area': mask.sum()} for mask in masks]

        fig = plt.figure(figsize=(10, 10))
        ax = plt.Axes(fig, [0.05, 0.05, 0.9, 0.9])
        fig.add_axes(ax)
        ax.set_axis_off()

        ax.imshow(self.image)
        if labels is not None:
            for mask, label in zip(masks, labels):
                color = color_map[label]
                self._show_single_mask(ax, mask['segmentation'], color=color, label=label)
        else:
            self.show_anns(masks)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        if show:
            if return_image:
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=150)
                plt.close(fig)
                buf.seek(0)
                image = Image.open(buf).convert("RGB")
                return np.array(image)
            else:
                plt.show()

    def show_masks_side_by_side(self, gt_masks, pred_masks, iou_thresh=0.5, figsize=(8, 4.5), save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        h, w = self.image.shape[:2]

        gt_anns = [{'segmentation': mask, 'area': mask.sum()} for mask in gt_masks]
        pred_anns = [{'segmentation': mask, 'area': mask.sum()} for mask in pred_masks]

        color_map = {}
        used_preds = set()

        # Match predicted masks to ground truth by IoU
        for i, gmask in enumerate(gt_masks):
            best_iou = 0
            best_j = -1
            for j, pmask in enumerate(pred_masks):
                if j in used_preds:
                    continue
                iou = self.compute_iou(gmask, pmask)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            shared_color = np.random.rand(3).tolist() + [0.5]
            color_map[f'gt_{i}'] = shared_color
            if best_iou >= iou_thresh:
                color_map[f'pred_{best_j}'] = shared_color
                used_preds.add(best_j)

        for j in range(len(pred_masks)):
            if f'pred_{j}' not in color_map:
                color_map[f'pred_{j}'] = np.random.rand(3).tolist() + [0.5]

        ax = axes[0]
        ax.imshow(self.image)
        for i, ann in enumerate(gt_anns):
            self._show_single_mask(ax, ann['segmentation'], color=color_map[f'gt_{i}'])
        ax.set_title("Ground Truth")
        ax.axis("off")

        ax = axes[1]
        ax.imshow(self.image)
        for j, ann in enumerate(pred_anns):
            self._show_single_mask(ax, ann['segmentation'], color=color_map[f'pred_{j}'])
        ax.set_title("Predicted")
        ax.axis("off")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

class CoralSegmenter(SAM2Segmenter):

    def __init__(self, config_path, checkpoint_path, coral_filter: CoralFilterEnsembler, annotation_path = None,
                 large_feature_params=None, small_feature_params=None, nr_params=None, device=None, cs=None):
        
        super().__init__(config_path, checkpoint_path, annotation_path=annotation_path, large_feature_params=large_feature_params, small_feature_params=small_feature_params, nr_params=nr_params, device=device)
        self.coral_filter = coral_filter
        self.crop_space = cs or CROP_SPACE
        self.color_map = self._create_color_map(self.coral_filter.classes)

    # Automatic SAM2 Calibration Algorithm (ASCA) ################################################################
    def predict(self, img_path=None, img: Image=None, large_feature_params=None, small_feature_params=None,             
                whiteBalance=None,
                clipLimit=None,
                tileGridSize=None,
                redBoost=None,
                gamma=None,
                overlap=None,
                min_area=None, mask_size=None, coral_thresh=0.5, init_models=True, verbose=True):
        
        min_area = min_area or MIN_AREA
        mask_size = mask_size or MASK_SIZE
        overlap = overlap or self.nr_params['overlap']

        with timer(verbose):

            all_coral_masks = super().predict(img_path=img_path, img=img, large_feature_params=large_feature_params, small_feature_params=small_feature_params, 
                                                    #Augmentation hyperparameters
                                                    whiteBalance=whiteBalance or self.nr_params['whiteBalance'], 
                                                    redBoost=redBoost or self.nr_params['redBoost'], 
                                                    clipLimit=clipLimit or self.nr_params['clipLimit'], 
                                                    tileGridSize=tileGridSize or self.nr_params['tileGridSize'], 
                                                    gamma=gamma or self.nr_params['gamma'],
                                                #Consolidation parameter
                                                overlap=overlap, 
                                                min_area=min_area, init_models=init_models, verbose=verbose, keep_all=True)

            #SAM2Segmenter's predict method will load in the image -> self.image and resize if necessary
            image = torch.tensor(self.image.transpose((2, 0, 1)))

            #Format the segmentation masks as tensors for model input (this is technically a numpy array; but conversion from numpy to tensor is trivial)
            coral_masks_X = np.stack([mask['segmentation'] for mask in all_coral_masks])

            #NOTE: Computation time for this operation may vary depending on the size of the coral filter ensembler used (e.g. how many submodules there are)
            coral_classes_proba = self.coral_filter.predict(coral_masks_X, image, mask_size=mask_size)
            coral_classes_p = np.abs(np.max(coral_classes_proba, axis=1)-1e-3)
            weights = coral_classes_p #1/(1-coral_classes_p)
            coral_classes_preds = np.argmax(coral_classes_proba, axis=1)
            
            is_coral = coral_classes_preds != self.coral_filter.noncoral_class
            coral_masks = [mask for i, mask in enumerate(all_coral_masks) if is_coral[i]]

            #Merge masks (or 'consolidate' as Calvin says)
            coral_masks_merged, kept = self.merge(coral_masks, weights=weights[is_coral], min_area=min_area, overlap=overlap, verbose=verbose)
            if len(kept) > 0:
                #Corresponding class labels
                #np.array(list(self.coral_filter.classes.keys()))[labels]
                return coral_masks_merged, coral_classes_preds[is_coral][kept]
            else:
                return np.array([]), np.array([])

    @staticmethod
    def _create_color_map(classes):
        unique_labels = sorted(set(classes.keys()))
        label_to_color = {
            label: np.random.rand(3).tolist() + [0.6]
            for label in unique_labels
        }
        return label_to_color

    @staticmethod
    def coral_cover(masks, area=1024*1024, cs=0):
        return (np.sum(masks, axis=0).astype(bool)).sum() / (area - cs)