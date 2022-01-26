#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT
# Modified by: Yuhao Chen, UW, 09/29/2021

# --*-- coding:utf-8 --*--
import math
import cv2
import os
import math
import h5py
import open3d as o3d
import json
import os
import numpy as np


def processScene(scene_path):
    PATH_TO_SCENE = scene_path

    for i in range(9):
        PATH_TO_RGB = os.path.join(PATH_TO_SCENE, str(i)+'_rgb.png')
        PATH_TO_DEPTH = os.path.join(PATH_TO_SCENE, str(i)+'.npz')
        PATH_TO_CAMPARAMS = os.path.join(PATH_TO_SCENE, str(i)+'_camera_params.json')
        PATH_TO_HDF5 = os.path.join(PATH_TO_SCENE, str(i)+'_scene.hdf5')
        PATH_TO_PCL = os.path.join(PATH_TO_SCENE, str(i)+'_pcl.ply')
    
        if os.path.exists(PATH_TO_PCL):
            continue
            
        # load np arrays
        color_raw = o3d.io.read_image(str(PATH_TO_RGB))
        with np.load(str(PATH_TO_DEPTH)) as data:
            depth_np = data['depth']

        height, width = depth_np.shape

        # print("height [px] x width [px]", depth_np.shape)
        # print("depth min [cm]", np.amin(depth_np))
        # print("depth max [cm]", np.amax(depth_np))

        depth_raw = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_raw, 
            depth=depth_raw,
            depth_scale=100.0, # important! cm -> m
            depth_trunc=5.0,
            convert_rgb_to_intensity=False)

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

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            camera_intrinsics)

        pcl_path = PATH_TO_PCL
        o3d.io.write_point_cloud(str(pcl_path), pcd)


def process(root, start, end):
    for i in range(start, end+1):
        scene = os.path.join(root, 'scene'+str(i))
        print("processing", scene)
        processScene(scene)
        
    return


if __name__ == "__main__":
    roots = ['/home/hooman.ramezani/Desktop/kit/kit']
            
    from multiprocessing import Process
    
    processNum = 16
    jobs = []
    for root in roots:
        total = 2501
        interval = int(total / processNum)
        starts = [j*interval for j in range(16)]
        ends = [(j+1)*interval for j in range(16)]
        ends[-1] = total
        
        for i in range(processNum):
            p = Process(target=process, args=(root, starts[i], ends[i],))
            jobs.append(p)
            p.start()
        

