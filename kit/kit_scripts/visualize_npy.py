import numpy as np
import open3d as o3d

file_0 = "kit/scene0/0.npz"
file_1 = "kit/scene0/1.npz"
file_2 = "kit/scene0/2.npz"

print("Load a ply point cloud, print it, and render it")

array = np.load(file_0)["instances_semantic"]
pcd = o3d.geometry.PointCloud()
print(o3d.utility.Vector3dVector(array))

pcd.points = o3d.utility.Vector3dVector(array)

print(pcd)
print(np.asarray(pcd.points))

o3d.visualization.draw_geometries([pcd])
