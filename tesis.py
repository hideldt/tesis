# %%
#!pip install numpy
#!pip install pandas

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import KDTree



# %%
out_path = '/home/leo/Desktop/tesis/data/out_n.npy'

sam_path = '/home/leo/Desktop/tesis/data/sam_n.npy'
# %%
out_arr = np.load(out_path)
sam_path = np.load(sam_path)
# %%

def iqr_filter(distances,neighbors):
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr_val = q3 - q1
    lower_bound = q1 - 1.5 * iqr_val
    upper_bound = q3 + 1.5 * iqr_val
    output = []
    for i  in range(len(distances)):
        if lower_bound <= distances[i] <= upper_bound:
            output.append(neighbors[i].tolist())
            print(output)
    return np.mean(np.array(output), axis = 0).tolist()
        

import open3d as o3d
import numpy as np
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(sam_path)


blue_color = [64.0/255.0,101.0/255.0,139.0/255.0]
pcd.colors = o3d.utility.Vector3dVector(np.array([blue_color]*len(pcd.points)))

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=5))
pcd.normalize_normals()
normals_ = pcd.normals
normals_ = abs(np.asarray(normals_))
puntos = pcd.points 

kdtree = KDTree(puntos)
window_size = 5 
normal_filter = []
for query_point in (puntos,normals_):
    # Find nearest neighbors
    distances, nearest_neighbor_indices = kdtree.query(query_point, k=window_size)
    # Extract neighbors within window size
    neighbors = normals_[nearest_neighbor_indices]
    # Apply IQR filter
    nor_filt = iqr_filter(distances,neighbors)
    normal_filter.append(nor_filt)    


pcd.normals  = o3d.utility.Vector3dVector(np.array(normal_filter))   

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
# %%
