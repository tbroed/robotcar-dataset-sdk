import open3d as o3d
import argparse
import os
import re
import numpy as np
from python.files_mapping.icp import get_icp_transform
from transform import build_se3_transform
from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud
import copy
from numpy import linalg as LA


def init_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=os.path.basename(__file__))
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
    render_option.point_color_option = o3d.visualization.PointColorOption.Default
    return vis


def display_single_pc(pointcloud):
    vis = init_visualizer()
    vis.add_geometry(pointcloud)
    vis.run()


def get_point_clouds(args, use_all=True):
    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', args.laser_dir).group(0)
    timestamps_path = os.path.join(args.laser_dir, os.pardir, lidar + '.timestamps')
    if use_all:
        timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
    else:
        timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)[200:400][::10]
    timestamps = timestamps.tolist()

    origin_time = int(timestamps[0])

    with open(os.path.join(args.extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses = get_poses(args.poses_file, args.extrinsics_dir, G_posesource_laser, timestamps, origin_time)

    timestamps = timestamps[1:]  # delete the added (in get_poses and more down in the code) origin timestamp again

    # test_icp_with_two_timestemps(args, timestamps, poses)

    point_clouds = []
    point_clouds_relative_poses = []
    for i, timestamp in enumerate(timestamps):
        if i % 100 is 0:
            print("iteration: ", i)
        pc = get_single_pc(args.laser_dir, timestamp)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            -np.ascontiguousarray(pc.transpose().astype(np.float64)))
        estimate_normals(pcd)#, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
        point_clouds.append(pcd)
        if i > 0:
            relative_pose = np.dot(inverse_transformation(poses[i - 1]), poses[i])
            point_clouds_relative_poses.append(relative_pose)

    optimized_relative_poses = optimize_with_icp(point_clouds, point_clouds_relative_poses)

    optimized_absolute_poses = []
    previous_pose = np.identity(4)
    for pose in optimized_relative_poses:
        absolute_pose = np.dot(previous_pose, pose)
        optimized_absolute_poses.append(absolute_pose)
        previous_pose = absolute_pose

    return point_clouds, optimized_absolute_poses #TODO: change back to optimized_relative_poses


def do_icp(source, target, trans_init, threshold=0.02, verbose=0):
    if verbose is 1:
        print("Initial alignment")
        evaluation = o3d.open3d.registration.evaluate_registration(
            source, target, threshold, trans_init)
        print(evaluation)

    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # reg_p2l = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    # mu, sigma = 0, 0.1  # mean and standard deviation
    # loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
    # p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    # reg_p2l = o3d.pipelines.registration.registration_icp(source, target,
    #                                                       threshold, trans_init,
    #                                                       p2l)

    if verbose is 1:
        print("ICP results:")
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)
    return reg_p2l


def get_single_pc(lidar_dir, start_time):
    scan_path = os.path.join(lidar_dir, str(start_time) + '.bin')
    if os.path.isfile(scan_path):
        ptcld = load_velodyne_binary(scan_path)
    else:
        scan_path = os.path.join(lidar_dir, str(start_time) + '.png')
        if not os.path.isfile(scan_path):
            print(str(start_time) + " not available")
        ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)
        ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)
    scan = ptcld[:3]

    # scan = np.dot(np.dot(poses[0], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))
    # pointcloud = np.hstack([pointcloud, scan])
    #
    # pointcloud = pointcloud[:, 1:]
    # if pointcloud.shape[1] == 0:
    #     raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return scan  # pointcloud, reflectance


def optimize_with_icp(point_clouds, point_clouds_relative_poses):
    point_clouds_icp_optimized_poses = []
    point_clouds_icp_optimized_poses.append(np.identity(4))
    for i in range(1, len(point_clouds)):
        if i % 100 is 0:
            print("icp iteration: ", i)
        target = point_clouds[i - 1]
        source = point_clouds[i]
        trans_init = point_clouds_relative_poses[i - 1]  # does not have an entry for the first pc
        opt_pose = do_icp(source, target, trans_init, threshold=0.2)
        point_clouds_icp_optimized_poses.append(opt_pose.transformation)
    return point_clouds_icp_optimized_poses


def get_poses(poses_file, extrinsics_dir, G_posesource_laser, timestamps, origin_time):
    poses_type = re.search('(vo|ins|rtk|radar_odometry)\.csv', poses_file).group(1)

    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)

        poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=(poses_type == 'rtk'))
    else:
        # sensor is VO, which is located at the main vehicle frame
        poses = interpolate_vo_poses(poses_file, timestamps, origin_time)
        # new_poses = []
        # for pose in poses:
        #     new_pose = np.dot(pose, G_posesource_laser)
        #     new_poses.append(new_pose)
    return poses


def draw_registration_result(source, target, transformation):
    vis = init_visualizer()

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # vis.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4459,
    #                                   front=[0.9288, -0.2951, -0.2242],
    #                                   lookat=[1.6784, 2.0612, 1.4451],
    #                                   up=[-0.3402, -0.9189, -0.1996])
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.run()
    vis.close()


def inverse_transformation(matrix):
    C = matrix[:3, :3]
    r = matrix[:3, 3]
    inverse = np.identity(4)
    inverse[:3, :3] = C.transpose()
    inverse[:3, 3] = np.array(np.dot(-C.transpose(), r)).squeeze()
    return inverse




def visualize_list_of_pc(point_clouds, optimized_relative_poses):
    vis = init_visualizer()
    optimized_absolute_poses = []
    previous_pose = np.identity(4)
    for pose in optimized_relative_poses:
        absolute_pose = np.dot(previous_pose, pose)
        optimized_absolute_poses.append(absolute_pose)

    for i,pc in enumerate(point_clouds):
        pc_temp = copy.deepcopy(pc)
        pc_temp.transform(optimized_absolute_poses[i])
        vis.add_geometry(pc_temp)
    vis.run()

def combine_point_clouds(point_clouds, poses):
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector()
    combined_pcd.normals = o3d.utility.Vector3dVector()
    for pc, pose in zip(point_clouds, poses):
        pc_temp = copy.deepcopy(pc)
        pc_temp.transform(pose)
        combined_pcd.points.extend(pc_temp.points)
        combined_pcd.normals.extend(pc_temp.normals)
    # display_single_pc(combined_pcd)
    return combined_pcd

def test_icp_with_two_timestemps(args, timestamps, poses):
    timestamp_target = int(timestamps[0])
    timestamp_source = int(timestamps[target_num])

    # TODO: do for 1. and 3. -> display -> do ICP -> display

    pointcloud_source = get_single_pc(args.laser_dir, timestamp_source)
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(
        -np.ascontiguousarray(pointcloud_source.transpose().astype(np.float64)))
    # display_single_pc(pcd_source)

    pointcloud_target = get_single_pc(args.laser_dir, timestamp_target)
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(
        -np.ascontiguousarray(pointcloud_target.transpose().astype(np.float64)))
    # display_single_pc(pcd_target)
    #
    #
    rel_pose = np.dot(inverse_transformation(poses[0]), poses[target_num])
    draw_registration_result(pcd_source, pcd_target, rel_pose)

    # pcd_source.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # pcd_target.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    estimate_normals(pcd_source, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    estimate_normals(pcd_target, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    refined_transformation = do_icp(pcd_source, pcd_target, rel_pose, threshold=0.2, verbose=1)
    print("difference due to icp ", rel_pose - refined_transformation.transformation)
    draw_registration_result(pcd_source, pcd_target, refined_transformation.transformation)


if __name__ == "__main__":
    target_num = 10

    parser = argparse.ArgumentParser(description='Build and display a pointcloud')
    parser.add_argument('--poses_file', type=str, default=None, help='File containing relative or absolute poses')
    parser.add_argument('--laser_dir', type=str, default=None, help='Directory containing LIDAR data')
    parser.add_argument('--extrinsics_dir', type=str, default=None,
                        help='Directory containing extrinsic calibrations')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--models_dir', type=str, help='Directory containing camera models')

    args = parser.parse_args()

    point_clouds, poses = get_point_clouds(args, use_all=False)

    point_cloud = combine_point_clouds(point_clouds, poses)

    o3d.io.write_point_cloud("output/Point_Clouds/pc_all_icp_vo.ply", point_cloud)

    display_single_pc(point_cloud)


    # visualize_list_of_pc(point_clouds, optimized_relative_poses)

    # # visualize v0/gps difference
    # draw_registration_result(o3d.io.read_point_cloud("output/Point_Clouds/pc_all_vo.ply"),
    #                          o3d.io.read_point_cloud("output/Point_Clouds/pc_all_gps.ply"),
    #                          np.identity(4))



    print("hier")
    # TODO: build_pose_graph()
    # TODO: do_graph_optimization()
    # build_lidar_map(point_clouds,pc_transformation)
    # save_map()
    # display_pc()
