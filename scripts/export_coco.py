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
        self.image_id += 1

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

#def binary_mask_to_polygons(mask):
    #contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # need to ensure EXT/SIMPLE isn't screwing things up
    #segmentation = []
    #for contour in contours:
        #if len(contour) >= 3:
            #poly = contour.flatten().astype(float).tolist()
            #if len(poly) >= 6:
                #segmentation.append(poly)
    #return segmentation

def mask_to_poly(mask):
    array_polygons = supervision.mask_to_polygons(mask)
    segmentation = []
    for polygon in array_polygons:
        list_polygons = polygon.flatten().tolist()
        if len(list_polygons) >= 6:
            segmentation.append(list_polygons)
    return segmentation