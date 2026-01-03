import trimesh
import numpy as np
import open3d as o3d



if __name__ == "__main__":
    ply_file = "/home/cyrus/OpenHomie/HomieRL/legged_gym/resources/objs/O02@0015@00016/scan.ply"
    mesh = trimesh.load_mesh(ply_file)
    points, _ = trimesh.sample.sample_surface(mesh, count=2048)
    
    # 2. 创建 Open3D 点云对象 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # ... (其他可视化设置) ...
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) 

    # 3. 可视化
    print(f"正在加载 {points.shape} 个点的点云进行可视化...")
    o3d.visualization.draw_geometries([pcd], window_name="opoints")
