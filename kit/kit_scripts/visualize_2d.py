#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset Obejct detection viewer.")
    parser.add_argument(
        "--data_root",
        type=str, 
        default="./dataset_ifl",
        help="Path to data.")
    parser.add_argument(
        "--scene",
        type=int, default=0,
        help="Specify which scene to load.")
    parser.add_argument(
        "--viewpt",
        type=int, default=0,
        help="Specify which viwpt to load.")
    args = parser.parse_args()

    PATH_TO_DATA = pathlib.Path(args.data_root)
    SCENE = args.scene
    VIEWPT = args.viewpt
    PATH_TO_SCENE = PATH_TO_DATA / f"scene{SCENE}"
    PATH_TO_RGB = PATH_TO_SCENE / f"{VIEWPT}_rgb.png"
    PATH_TO_DEPTH = PATH_TO_SCENE / f"{VIEWPT}.npz"
    
    color_bgr = cv2.imread(str(PATH_TO_RGB))
    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    with np.load(str(PATH_TO_DEPTH)) as data:
        depth = data['depth']
        instances_objects = data['instances_objects']
        instances_semantic = data['instances_semantic']
        occlusion = data['occlusion']
    
    height, width = depth.shape
    plt.subplot(1, 5, 1)
    plt.title('RGB')
    plt.imshow(color)
    plt.subplot(1, 5, 2)
    plt.title('Depth')
    plt.imshow(depth)
    plt.subplot(1, 5, 3)
    plt.title('Instances (Object Ids)')
    plt.imshow(instances_objects)
    plt.subplot(1, 5, 4)
    plt.title('Instances (Categories)')
    plt.imshow(instances_semantic)
    plt.subplot(1, 5, 5)
    plt.title('Occlusion')
    plt.imshow(occlusion)
    plt.show()
