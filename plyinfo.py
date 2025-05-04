from plyfile import PlyData

# 读取 ply
ply = PlyData.read("/home/qingran/Desktop/gic/output/pacnerf/torus/point_cloud_fix_pcd/iteration_40000/point_cloud.ply")

# 打印原生 header
print(ply.header)

# 遍历每种 element（vertex, face, …）
for elem in ply.elements:
    props = [p.name for p in elem.properties]
    print(f"Element '{elem.name}': count={elem.count}, properties={props}")