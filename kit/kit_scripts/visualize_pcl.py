import numpy as np
import open3d as o3d

file_0 = "kit/scene0/0_pcl.ply"
file_1 = "kit/scene0/1_pcl.ply"
file_2 = "kit/scene0/2_pcl.ply"

print("Load a ply point cloud, print it, and render it")

pcd_0 = o3d.io.read_point_cloud(file_0)
pcd_1 = o3d.io.read_point_cloud(file_1)
pcd_2 = o3d.io.read_point_cloud(file_2)

print(pcd_0)
print(np.asarray(pcd_0.points))

o3d.visualization.draw_geometries([pcd_0])
