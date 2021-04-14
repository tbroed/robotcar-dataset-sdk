import open3d as o3d
import argparse
import os
import re
import numpy as np
from python.files_mapping.icp import get_icp_transform
from python.files_mapping.pose_graph_optimization import PoseGraphOptimization
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


def accumulate_poses(relative_poses):
    absolute_poses = []
    previous_pose = np.identity(4)
    for pose in relative_poses:
        absolute_pose = np.dot(previous_pose, pose)
        absolute_poses.append(absolute_pose)
        previous_pose = absolute_pose
    return absolute_poses


def get_point_clouds(args, timestamps):
    origin_time = int(timestamps[0])
    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', args.laser_dir).group(0)
    with open(os.path.join(args.extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses = get_poses(args.poses_file, args.extrinsics_dir, G_posesource_laser, timestamps, origin_time)

    timestamps = timestamps[1:]  # delete the added (in get_poses and more down in the code) origin timestamp again

    point_clouds = []
    for i, timestamp in enumerate(timestamps):
        if i % 100 is 0:
            print("iteration: ", i)
        pc = get_single_pc(args.laser_dir, timestamp)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            -np.ascontiguousarray(pc.transpose().astype(np.float64)))
        point_clouds.append(pcd)

    optimized_relative_poses = optimize_with_icp(point_clouds, poses, timestamps)
    optimized_absolute_poses = accumulate_poses(optimized_relative_poses)

    return point_clouds, optimized_absolute_poses


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


def filter_pcl(pcl, filter_ego_vehicle=True, min_threshold=2.5, max_range=50, ground_level=2.5, reduce=False):
    # only keep point within the thresholds
    # reduce only keeps random 10% of the point clouds
    list_of_valid_points = []
    for i in range(pcl.shape[1]):
        if reduce:
            if np.random.rand() > 0.1:
                continue
        point = pcl[:, i]
        distance = LA.norm(point)
        reject_point = False
        if filter_ego_vehicle:
            if distance < min_threshold:
                # TODO: make box not circle
                reject_point = True
        if distance > max_range:
            reject_point = True
        if point[2] > ground_level:
            reject_point = True
        if not reject_point:
            list_of_valid_points.append(i)
    filtered_pcl = pcl[:, [list_of_valid_points]][:, 0, :]
    return filtered_pcl


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
    scan = filter_pcl(scan, filter_ego_vehicle=True, min_threshold=2.5, max_range=50, ground_level=2.5, reduce=False)
    return scan


def optimize_with_icp(point_clouds, poses_initial_guess, timestamps):
    point_clouds_icp_optimized_poses = [np.identity(4)]
    for i in range(1, len(point_clouds)):
        if i % 100 is 0:
            print("icp iteration: ", i)
        target = point_clouds[i - 1]
        source = point_clouds[i]

        # define entries for get_icp_transform
        src_pcl = np.array(source.points)
        dst_pcl = np.array(target.points)
        src_pose = poses_initial_guess[i]
        dst_pose = poses_initial_guess[i - 1]
        src_ts = timestamps[i]
        dst_ts = timestamps[i - 1]
        tmat, cc, fitness = get_icp_transform(src_pcl, dst_pcl,
                                              src_pose, dst_pose,
                                              src_ts, dst_ts, verbose=False, max_icp_distance=2) # TODO: adjust to 1.2 after 1m check
        # TODO: save to file to accumulate later
        point_clouds_icp_optimized_poses.append(tmat)
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
    # displays individual point clouds with their respective relative poses
    vis = init_visualizer()
    optimized_absolute_poses = []
    previous_pose = np.identity(4)
    for pose in optimized_relative_poses:
        absolute_pose = np.dot(previous_pose, pose)
        optimized_absolute_poses.append(absolute_pose)
    for i, pc in enumerate(point_clouds):
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


def build_pose_graph(poses):
    pgo = PoseGraphOptimization()
    for i, pose in enumerate(poses):
        if i is 0:
            pgo.add_vertex(i, pose, fixed=True)
        else:
            pgo.add_vertex(i, pose)
            relative_pose = np.dot(inverse_transformation(poses[i - 1]), poses[i])
            pgo.add_edge([i, i - 1], relative_pose)
        last = i
    # pgo.add_edge([last, 0], np.identity(4)) # forces a closure at the end
    return pgo


def downsample_pcl(pcl, rate):
    # reduces point cloud by the rate
    rate = 1. / rate
    points = pcl.points
    points_np = np.array(points)
    size = points_np.shape[0]
    sample = np.random.rand(size)
    keep = sample < rate
    sampled_points = points_np[keep]
    pcl.points = o3d.utility.Vector3dVector(sampled_points)
    if pcl.has_normals():
        pcl.normals = pcl.normals[keep]
    if pcl.has_colors():
        pcl.colors = pcl.colors[keep]
    return pcl


if __name__ == "__main__":
    down_sample_rate = 5
    display_result = True
    use_all_points = False
    start_ts = 32750
    end_ts = 33800
    stride = 20
    save_result = False
    save_result_name = "output/Point_Clouds/pc_test_circle.ply"
    build_graph = True
    save_poses = True
    save_poses_name = 'test.npy'
    load_poses = True
    load_poses_name = 'poses_gps_circle_32750_33800_20.npy'

    parser = argparse.ArgumentParser(description='Build and display a pointcloud')
    parser.add_argument('--poses_file', type=str, default=None, help='File containing relative or absolute poses')
    parser.add_argument('--laser_dir', type=str, default=None, help='Directory containing LIDAR data')
    parser.add_argument('--extrinsics_dir', type=str, default=None,
                        help='Directory containing extrinsic calibrations')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
    args = parser.parse_args()
    if not load_poses:
        if not use_all_points:
            point_clouds, poses = get_point_clouds(args, use_all=use_all_points, start_ts=start_ts, end_ts=end_ts,
                                                   stride=stride)
        else:
            point_clouds, poses = get_point_clouds(args)
        point_cloud = combine_point_clouds(point_clouds, poses)

        if save_result:
            o3d.io.write_point_cloud(save_result_name, point_cloud)
        if down_sample_rate > 0:
            point_cloud = downsample_pcl(point_cloud, rate=down_sample_rate)
        if display_result:
            display_single_pc(point_cloud)
        if save_poses:
            poses_np = np.array(poses)
            np.save(save_poses_name, poses_np)
    else:
        poses_np = np.load(load_poses_name)
        poses = poses_np
    if build_graph:
        # TODO: build_pose_graph()
        pgo = build_pose_graph(poses)
        pgo.visualize_in_plt(threeDim=True)

        # TODO: do_graph_optimization()
        print('Performing full BA:')
        pgo.optimize(max_iterations=20, verbose=True)
        pgo.visualize_in_plt()

    print("finished")
