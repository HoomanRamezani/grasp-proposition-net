from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)
import os
import json
import cv2    # (pip install opencv-python)


def create_sub_masks(image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}

    object_ids = np.unique(image)
    print("ids", object_ids)

    #remove background
    object_ids = object_ids[1:]

    for i in object_ids:
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[np.where(image == i)] = 1
        sub_masks[i] = mask

    return sub_masks


def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)

    
    contours, hier = cv2.findContours(sub_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    segmentations = []
    for contour in contours:
        contour = np.squeeze(contour)
        #if contour has less than 3 point ignore
        if len(contour) < 3:
            continue
        # Make a polygon and simplify it
        poly = Polygon(contour)
        
        if(poly.is_empty ):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)
        #try:
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        '''
        except:
            polygons.pop()
            poly = Polygon(contour)
            
            if(poly.is_empty):
                continue
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
        '''
            
        segmentations.append(segmentation)
    
    return polygons, segmentations

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format
