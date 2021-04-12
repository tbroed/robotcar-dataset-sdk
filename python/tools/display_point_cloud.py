import open3d as open3d
import numpy as np
import os
import sys
sys.path.insert(0, '')

from transform import build_se3_transform


print("Load a ply point cloud, print it, and render it")
# pcd = open3d.io.read_point_cloud("output/Point_Clouds/pc_all_gps.ply")

# Meeting 08.04.21:
#show that ICP works
#show that vo itself is already really good (del -target_num)
# pcd = open3d.io.read_point_cloud("output/Point_Clouds/pc_all_gps.ply")
# draw_registration_result(open3d.io.read_point_cloud("output/Point_Clouds/pc_all_vo.ply"), open3d.io.read_point_cloud("output/Point_Clouds/pc_all_gps.ply"),np.identity(4))
# pcd = open3d.io.read_point_cloud("output/Point_Clouds/pc_all_icp_vo.ply")
# pcd = open3d.io.read_point_cloud("output/Point_Clouds/pc_all_icp_only_p2plane.ply")
# draw_registration_result(open3d.io.read_point_cloud("output/Point_Clouds/pc_all_vo.ply"), open3d.io.read_point_cloud("output/Point_Clouds/pc_all_icp_vo.ply"),np.identity(4))
# pcd = open3d.io.read_point_cloud("output/Point_Clouds/RaLL_robotcar_map.ply")

# Neu:
pcd = open3d.io.read_point_cloud("output/Point_Clouds/pc_icp_every_10_frame.ply")
# draw_registration_result(open3d.io.read_point_cloud("output/Point_Clouds/pc_all_vo.ply"), open3d.io.read_point_cloud("output/Point_Clouds/pc_icp_every_10_frame.ply"),np.identity(4))
# pcd = open3d.io.read_point_cloud("output/Point_Clouds/pc_icp_200_400_every_10th.ply")


# pcd = open3d.io.read_point_cloud("pc_bright_color.ply")
print(pcd)
# open3d.visualization.draw_geometries([pcd])


# Pointcloud Visualisation using Open3D
vis = open3d.visualization.Visualizer()
vis.create_window(window_name=os.path.basename(__file__))
render_option = vis.get_render_option()
render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
render_option.point_color_option = open3d.visualization.PointColorOption.Default
coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
vis.add_geometry(coordinate_frame)

pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
vis.add_geometry(pcd)
view_control = vis.get_view_control()
params = view_control.convert_to_pinhole_camera_parameters()
params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])
view_control.convert_from_pinhole_camera_parameters(params)
vis.run()