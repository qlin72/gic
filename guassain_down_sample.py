from plyfile import PlyData, PlyElement
import numpy as np

# 1. 读入
ply = PlyData.read("/home/qingran/Desktop/gic/output/pacnerf/elastic/0/point_cloud_fix_pcd/iteration_40000/point_cloud.ply")
vertex_data = ply['vertex'].data  # numpy structured array

# 2. 随机采样 20%
num = len(vertex_data)
keep_idx = np.random.choice(num, size=int(num * 0.2), replace=False)

# 3. 构造新的 structured array
new_data = vertex_data[keep_idx]

# 4. 写出新 PLY
elm = PlyElement.describe(new_data, 'vertex')
PlyData([elm], text=True).write("/home/qingran/Desktop/gic/output/pacnerf/elastic/0/point_cloud_fix_pcd/iteration_40000/point_cloud_down_sample.ply")