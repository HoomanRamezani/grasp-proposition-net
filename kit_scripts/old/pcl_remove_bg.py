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

    for i in range(161):
        PATH_TO_RGB = os.path.join(PATH_TO_SCENE, str(i)+'_rgb.png')
        PATH_TO_DEPTH = os.path.join(PATH_TO_SCENE, str(i)+'.npz')
        PATH_TO_ILABEL = os.path.join(PATH_TO_SCENE, str(i)+'_ilabel.txt')
        PATH_TO_SLABEL = os.path.join(PATH_TO_SCENE, str(i)+'_slabel.txt')
        PATH_TO_CAMPARAMS = os.path.join(PATH_TO_SCENE, str(i)+'_camera_params.json')
        PATH_TO_HDF5 = os.path.join(PATH_TO_SCENE, str(i)+'_scene.hdf5')
        PATH_TO_PCL = os.path.join(PATH_TO_SCENE, str(i)+'_pcl.ply')

        #if os.path.isfile(PATH_TO_PCL):# and (os.path.getsize(PATH_TO_PCL) > 1):
        #    os.remove(PATH_TO_PCL)
        #    continue
        # load np arrays
        color_raw = o3d.io.read_image(str(PATH_TO_RGB))
        #rgb = cv2.imread(PATH_TO_RGB)
        with np.load(str(PATH_TO_DEPTH)) as data:
            depth_np = data['depth']
            semantic_labels = data['instances_semantic']
            semantic_labels = semantic_labels[:, :, np.newaxis]
            instance_labels = data['instances_objects']
            instance_labels = instance_labels[:, :, np.newaxis]
            
            
        height, width = depth_np.shape

        
        
        #print("height [px] x width [px]", depth_np.shape)
        #print("depth min [cm]", np.amin(depth_np))
        #print("depth max [cm]", np.amax(depth_np))

        
        sl_image = o3d.geometry.Image(semantic_labels)
        
        il_image = o3d.geometry.Image(instance_labels)
        
        depth_raw = o3d.geometry.Image(depth_np)
        #import pdb
        #pdb.set_trace()
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_raw, 
            depth=depth_raw,
            depth_scale=100.0, # important! cm -> m
            depth_trunc=5.0,
            convert_rgb_to_intensity=False)

        semantic_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=sl_image, 
            depth=depth_raw,
            depth_scale=100.0, # important! cm -> m
            depth_trunc=5.0,
            convert_rgb_to_intensity=False)

        instance_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=il_image, 
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
        pcd_s = o3d.geometry.PointCloud.create_from_rgbd_image(
            semantic_image,
            camera_intrinsics)
        pcd_i = o3d.geometry.PointCloud.create_from_rgbd_image(
            instance_image,
            camera_intrinsics)

        
        s_label = np.mean(np.asarray(pcd_s.colors), axis=1).astype(np.uint8)
        i_label = np.mean(np.asarray(pcd_i.colors), axis=1).astype(np.uint8)

        indices = np.where(s_label != 0)[0] #remove background
        pcd = pcd.select_by_index(indices)
        i_label = i_label[indices]
        s_label = s_label[indices]
        
        s_label.tofile(PATH_TO_SLABEL, sep='\n', format='%s')
        i_label.tofile(PATH_TO_ILABEL, sep='\n', format='%s')
        
        #camera_cos = o3d.geometry.TriangleMesh.create_coordinate_frame()
        #o3d.visualization.draw_geometries([
        #    pcd,
        #    camera_cos])
        pcl_path = PATH_TO_PCL
        o3d.io.write_point_cloud(str(pcl_path), pcd)
        
def process(root, start, end):
    for i in range(start, end+1):
        scene = os.path.join(root, 'scene'+str(i))
        print("processing", scene)
        processScene(scene)
        
    return


if __name__ == "__main__":
    roots = ['/pub2/y2863/data/kit-easy'] #['/pub2/y2863/data/kit', '/pub2/y2863/data/kit2']

    #processScene(os.path.join(roots[0], 'scene1'))


    #exit(0)
    from multiprocessing import  Process
    
    
    processNum = 16
    jobs = []
    for root in roots:
        total = 151
        interval = int(total / processNum)
        starts = [j*interval for j in range(16)]
        ends = [(j+1)*interval for j in range(16)]
        ends[-1] = total
        #process(root, 0, 2500)
        
        #continue
        for i in range(processNum):
            p = Process(target=process, args=(root, starts[i], ends[i],))
            jobs.append(p)
            p.start()
        
