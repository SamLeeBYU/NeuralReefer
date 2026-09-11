"""
This module defines the COCOExporter class, which accumulates image metadata and
polygon-based segmentation masks during inference and writes them to a valid
COCO JSON annotation file. Compatible with standard tools like RoboFlow and
pycocotools.
"""

import json
import cv2
import numpy as np

import supervision

class COCOExporter:
    def __init__(self, class_names):
        self.images = []
        self.annotations = []
        self.categories = []
        self.image_id = 0
        self.ann_id = 0

        # Create a category map for consistent category IDs
        self.category_map = {name: idx + 1 for idx, name in enumerate(sorted(set(class_names)))}

        for name, idx in self.category_map.items():
            self.categories.append({
                "id": idx,
                "name": name,
                "supercategory": "coral"
            })

    def add_image(self, file_name, height, width, image_id=None):
        img_id = self.image_id if image_id is None else image_id
        self.images.append({
            "id": img_id,
            "file_name": file_name,
            "height": height,
            "width": width
        })
        # max(...)+1 (not a plain +1) so an explicit image_id can't leave
        # self.image_id trailing behind it and hand out a colliding id later.
        self.image_id = max(self.image_id, img_id) + 1
        return img_id

    def add_annotation(self, image_id, mask, label):
        segmentation = mask_to_poly(mask)
        if not segmentation:
            return

        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        area = float(np.sum(mask))

        self.annotations.append({
            "id": self.ann_id,
            "image_id": image_id,
            "category_id": self.category_map[label],
            "segmentation": segmentation,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": area,
            "iscrowd": 0
        })
        self.ann_id += 1

    def save(self, output_path):
        with open(output_path, "w") as f:
            json.dump({
                "images": self.images,
                "annotations": self.annotations,
                "categories": self.categories
            }, f, indent=2)

    def load(self, path):
        """
        Pre-populates this exporter from an already-written COCO json (its
        own prior save(), typically) -- for resuming a partially-completed
        inference run without losing the entries it already exported.
        `categories` is left as __init__ built it (assumed identical, since
        both come from the same class_names) rather than overwritten.
        Advances image_id/ann_id past the max id already present, so newly
        added entries can't collide with the reloaded ones.
        """
        with open(path, "r") as f:
            existing = json.load(f)
        self.images = existing["images"]
        self.annotations = existing["annotations"]
        if self.images:
            self.image_id = max(img["id"] for img in self.images) + 1
        if self.annotations:
            self.ann_id = max(ann["id"] for ann in self.annotations) + 1

def mask_to_poly(mask):
    array_polygons = supervision.mask_to_polygons(mask)
    segmentation = []
    for polygon in array_polygons:
        list_polygons = polygon.flatten().tolist()
        if len(list_polygons) >= 6:
            segmentation.append(list_polygons)
    return segmentation