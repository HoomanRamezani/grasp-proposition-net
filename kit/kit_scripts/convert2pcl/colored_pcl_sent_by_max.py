#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import pathlib
import h5py
import random

def random_rgb_color(seed, score=1.0):
    random.seed(seed)
    red = random.random()
    green = random.random()
    blue = random.random()
    return np.array([red, green, blue])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PCL Generation.")
    parser.add_argument(
        "--data_root",
        type=str, 
        default="data/",
        help="Path to data.")
    parser.add_argument(
        "--scene",
        type=int, default=0,
        help="Specify which scene to load.")
    parser.add_argument(
        "--viewpt",
        type=int, default=0,
        help="Specify which viwpt to load.")
    parser.add_argument(
        "--save_pcl",
        action="store_true",
        help="Set flag to save pcl in same location.")
    args = parser.parse_args()

    PATH_TO_DATA = pathlib.Path(args.data_root)
    SCENE = args.scene
    VIEWPT = args.viewpt

    PATH_TO_SCENE = PATH_TO_DATA / f"scene{SCENE}"
    PATH_TO_RGB = PATH_TO_SCENE / f"{VIEWPT}_rgb.png"
    PATH_TO_DEPTH = PATH_TO_SCENE / f"{VIEWPT}.npz"
    PATH_TO_CAMPARAMS = PATH_TO_SCENE / f"{VIEWPT}_camera_params.json"
    PATH_TO_HDF5 = PATH_TO_SCENE / f"{VIEWPT}_scene.hdf5"
    PATH_TO_PCL = PATH_TO_SCENE / f"{VIEWPT}_scene_pcl.ply"

    # load np arrays
    color_raw = o3d.io.read_image(str(PATH_TO_RGB))
    with np.load(str(PATH_TO_DEPTH)) as data:
        depth_np = data['depth']

    height, width = depth_np.shape

    print("height [px] x width [px]", depth_np.shape)
    print("depth min [cm]", np.amin(depth_np))
    print("depth max [cm]", np.amax(depth_np))

    depth_raw = o3d.geometry.Image(depth_np)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_raw, 
        depth=depth_raw,
        depth_scale=100.0, # important! cm -> m
        depth_trunc=5.0,
        convert_rgb_to_intensity=False)

    plt.subplot(1, 2, 1)
    plt.title('RGB')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth')
    plt.imshow(rgbd_image.depth)
    #plt.show()

    # TODO generate intrinsics from camera_params.json
    with open(str(PATH_TO_CAMPARAMS)) as json_file:
        f = json.load(json_file)
    fx = float(f['fx'])
    fy = float(f['fy'])


    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    camera_intrinsics.set_intrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=width/2,
        cy=height/2)

    print("intrinsic matrix : \n", camera_intrinsics.intrinsic_matrix)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics)
    
    if args.save_pcl:
        pcl_path = PATH_TO_PCL
        o3d.io.write_point_cloud(str(pcl_path), pcd)
        print(f"-> save pcl {pcl_path}")
    
    camera_cos = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([
        pcd,
        camera_cos])