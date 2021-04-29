import open3d as o3d
import argparse
import os
import re
import numpy as np
import yaml
from python.files_mapping.icp import get_icp_transform, get_initial_transform
from python.files_mapping.pose_graph_optimization import PoseGraphOptimization
from transform import build_se3_transform
from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud
import copy
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R


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


def filter_timestamps(list_of_timestamps, list_of_poses, loop_closure_ts, threshold_in_m=1):
    # only keep timestamps with poses 1m apart
    filtered_timestamps = [list_of_timestamps[0]]
    filtered_poses = [list_of_poses[0]]
    for i in range(1, len(list_of_timestamps)):
        r = np.array(list_of_poses[i][:3, 3] - filtered_poses[-1][:3, 3]).squeeze()
        distance = np.sqrt(r.dot(r))
        if list_of_timestamps[i] in loop_closure_ts:
            filtered_timestamps.append(list_of_timestamps[i])
            filtered_poses.append(list_of_poses[i])
        elif distance > threshold_in_m:
            filtered_timestamps.append(list_of_timestamps[i])
            filtered_poses.append(list_of_poses[i])
    # check if timestamps and poses are same length -> delete redundant first timestamp
    if len(filtered_timestamps) != filtered_poses and filtered_timestamps[0] == filtered_timestamps[1]:
        filtered_timestamps = filtered_timestamps[1:]
    return filtered_timestamps, filtered_poses


def get_point_clouds(config_setup, all_timestamps, lcts, stride=None):
    origin_time = int(all_timestamps[0])
    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', config_setup['laser_dir']).group(0)
    with open(os.path.join(config_setup['extrinsics_dir'], lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    all_poses = get_poses(config_setup['poses_file'], config_setup['extrinsics_dir'], G_posesource_laser,
                          all_timestamps, origin_time)

    filtered_timestamps, filtered_poses = filter_timestamps(all_timestamps[1:],
                                                            all_poses, lcts)  # delete the added (in get_poses and more down in the code) origin timestamp again

    if stride is not None:
        filtered_poses = filtered_poses[::stride]
        filtered_timestamps = filtered_timestamps[::stride]

    point_clouds = []
    print("collecting point clouds")
    for i, timestamp in enumerate(filtered_timestamps):
        if i % 100 is 0:
            print("iteration: ", i)
        pc = get_single_pc(config_setup['laser_dir'], timestamp)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            -np.ascontiguousarray(pc.transpose().astype(np.float64)))
        point_clouds.append(pcd)

    print("ICP optimization with direct neighbours")
    optimized_relative_poses, icp_fitness = optimize_with_icp(point_clouds, filtered_poses, filtered_timestamps)
    optimized_absolute_poses = accumulate_poses(optimized_relative_poses)

    return point_clouds, optimized_absolute_poses, filtered_timestamps, icp_fitness


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
    icp_fitness = [1]
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
                                              src_ts, dst_ts, verbose=False, max_icp_distance=0.1,
                                              tmp_folder='ICP_tmp/neighbours')
        point_clouds_icp_optimized_poses.append(tmat)
        icp_fitness.append(fitness)
    return point_clouds_icp_optimized_poses, icp_fitness


def get_poses(poses_file, extrinsics_dir, g_pose_source, timestamps, origin_time):
    poses_type = re.search('(vo|ins|rtk|radar_odometry)\.csv', poses_file).group(1)

    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            g_pose_source = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                            g_pose_source)

        poses_interpolated = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=(poses_type == 'rtk'))
    else:
        # sensor is VO, which is located at the main vehicle frame
        poses_interpolated = interpolate_vo_poses(poses_file, timestamps, origin_time)
    # new_poses = [] # TODO:check if this is needed
    # for pose in poses:
    #     new_pose = np.dot(pose, g_pose_source)
    #     new_poses.append(new_pose)
    return poses_interpolated


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


def build_pose_graph(poses, icp_fitness):
    pgo = PoseGraphOptimization()
    for i, pose in enumerate(poses):
        if i is 0:
            pgo.add_vertex(i, pose, fixed=True)
        else:
            pgo.add_vertex(i, pose)
            relative_pose = np.dot(inverse_transformation(poses[i - 1]), poses[i])
            if np.linalg.norm(relative_pose[:3,3]) <= 30:
                pgo.add_edge((i - 1, i), relative_pose, information=icp_fitness[i] * np.eye(6))
            else:
                print("no edge added at ", i)
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


def build_KD_Tree(config_setup, list_of_timestamps, two_dim=True):
    origin_time = int(list_of_timestamps[0])
    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', config_setup['laser_dir']).group(0)
    with open(os.path.join(config_setup['extrinsics_dir'], lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])
    gps_poses = get_poses(config_setup['gps_file'], config_setup['extrinsics_dir'], G_posesource_laser,
                          list_of_timestamps, origin_time)
    list_of_timestamps = list_of_timestamps[1:]
    gps_poses_np = np.array(gps_poses)
    if two_dim:
        points = gps_poses_np[:, [0, 1], 3]
        pcd_tree = o3d.geometry.KDTreeFlann(points.transpose())
    else:
        points = gps_poses_np[:, :3, 3]
        pcd_tree = o3d.geometry.KDTreeFlann(points.transpose())
    return pcd_tree, points, gps_poses, list_of_timestamps


def find_points_in_range(pgo_instance, timestamp, timestamp_id, radius=5):
    gps = get_gps_position()
    ids = get_neighbour_ids(gps)
    pgo_instance.add_vertex()
    for match_id in ids:
        match_pose = get_gps_pose(match_id)
        instance_pose = get_gps_pose(timestamp_id)
        transformation = np.dot(inverse_transformation(match_pose, instance_pose))
        optimized_transformation = icp(match_pose, instance_pose, transformation)
        pgo_instance.add_edge([match_id, vertex], optimized_transformation)


def get_timestamps(configurations, use_all=True):
    laser_dir = configurations['setup']['laser_dir']
    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', laser_dir).group(0)
    timestamps_path = os.path.join(laser_dir, os.pardir, lidar + '.timestamps')
    if use_all:
        timestamps_np = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
    else:
        start = configurations['data']['start_ts']
        end = configurations['data']['end_ts']
        timestamps_np = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)[start:end]
    timestamps_list = timestamps_np.tolist()
    return timestamps_list


def find_loop_closures(pgo_instance, kd_tree_instance, radius=5):
    # returns a list of indices of matches that lie withing the radius
    matches = []
    for vertex_id, vertex in pgo_instance.vertices().items():
        pivot_point = pgo_instance.vertex(vertex_id).estimate().matrix()[:3, 3]
        [k, idx, _] = kd_tree_instance.search_radius_vector_xd(np.expand_dims(pivot_point[[0, 1]], axis=1), radius)
        if idx:
            if idx[0] is not vertex_id:
                assert "indices do no match up"
        for match_id in idx:
            if np.abs((match_id - vertex_id)) > 20:
                # only use matched that are not to close to each other (find "real" loop closures)
                if (match_id, vertex_id) not in matches:
                    matches.append((vertex_id, match_id))
    return matches


def add_optimized_loop_closures(loaded_point_clouds, pgo_instance, loop_closures, poses_gps, list_of_timestamps,
                                fitness_threshold=0.4, do_ransac=False, rate=None, verbose=0):
    for vertex_id, match_id in loop_closures:
        if verbose:
            print('Checking for loop closure: %s to %s' % (list_of_timestamps[vertex_id] / 1e6,
                                                           list_of_timestamps[match_id] / 1e6))

        target = loaded_point_clouds[vertex_id]
        source = loaded_point_clouds[match_id]

        if rate is not None:
            target = downsample_pcl(target, rate=rate)
            source = downsample_pcl(source, rate=rate)

        # ICP with gps as initial transform
        src_pcl = copy.copy(np.array(source.points))
        dst_pcl = copy.copy(np.array(target.points))
        src_pose = poses_gps[vertex_id]
        dst_pose = poses_gps[match_id]
        src_ts = list_of_timestamps[vertex_id]
        dst_ts = list_of_timestamps[match_id]
        tmat, cc, fitness = get_icp_transform(src_pcl, dst_pcl, src_pose, dst_pose, src_ts, dst_ts,
                                              verbose=False, max_icp_distance=0.1,
                                              tmp_folder='ICP_tmp/lp_gps')
        if verbose:
            print("fitness gps: ", fitness)
            # draw_registration_result(target, source, tmat)

        # if gps result is not satisfying try to get a better transformation via RANSAC
        if do_ransac and fitness < 0.5:
            tmat_init, _ = get_initial_transform(src_pcl, dst_pcl, src_ts, dst_ts, verbose=verbose)
            tmat_ransac, _cc_ransac, fitness_ransac = get_icp_transform(src_pcl, dst_pcl, None, None, src_ts, dst_ts,
                                                                        verbose=False, max_icp_distance=0.1,
                                                                        tmat_init=tmat_init,
                                                                        tmp_folder='ICP_tmp/lp_ransac')
            print("fitness ransac: ", fitness_ransac) if verbose else None
            # draw_registration_result(target, source, np.linalg.inv(tmat_ransac))
            r = R.from_matrix(np.dot(np.linalg.inv(tmat), tmat_ransac)[:3, :3])
            if np.abs(np.sum(r.as_euler('zxy', degrees=True))) < 60:
                if fitness_ransac > fitness:
                    tmat = np.linalg.inv(tmat_ransac)
                    fitness = fitness_ransac
                    print("ransac used") if verbose else None
                else:
                    print("ransac fitness lower than gps fitness -> skipped ") if verbose else None
            else:
                print("ransac inverted -> skipped") if verbose else None

        # Add edge to pose graph if a good fit was found
        if fitness > fitness_threshold:
            # Display final registration result
            # draw_registration_result(target, source, tmat)
            pgo_instance.add_edge((match_id, vertex_id), tmat, information=fitness * np.eye(
                6))
            print("edge added between vertex ", src_ts, " and match ", dst_ts, "with fitness ",
                  fitness)
    return pgo_instance


def add_gps_anchors(point_clouds, pgo, list_of_loop_closures, gps_poses, timestamps, poses_vo, do_ransac):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build and display a pointcloud map')
    parser.add_argument('--config_path', type=str, help='Yaml file containing configurations')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, 'r'))
    display_down_sample_rate = config['data']['display_down_sample_rate']
    display_result = config['data']['display_result']

    timestamps = get_timestamps(config, use_all=config['data']['use_all_points'])
    point_clouds, poses, timestamps, icp_fitness = get_point_clouds(config['setup'], timestamps,
                                                                    stride=config['data']['stride_pc'])
    point_cloud = combine_point_clouds(point_clouds, poses)

    if config['data']['save_result']:
        o3d.io.write_point_cloud(config['data']['save_result_name'], point_cloud)
    if display_down_sample_rate > 0:
        point_cloud = downsample_pcl(point_cloud, rate=display_down_sample_rate)
    if display_result:
        display_single_pc(point_cloud)
    if config['data']['build_graph']:
        pgo = build_pose_graph(poses, icp_fitness)

        # Build KD Tree and find loop closures
        print("building KD Tree")
        kd_tree, gps_points, gps_poses = build_KD_Tree(config['setup'], timestamps)
        print("searching for loop closure candidates")
        list_of_loop_closures = find_loop_closures(pgo, kd_tree, radius=5)
        print("found ", len(list_of_loop_closures), " loop closures candidates")
        print(list_of_loop_closures)
        if config['data']['add_anchors']:
            pgo = add_gps_anchors(point_clouds, pgo, list_of_loop_closures, gps_poses, timestamps,
                                  poses_vo=poses, do_ransac=True)
        pgo = add_optimized_loop_closures(point_clouds, pgo, list_of_loop_closures, gps_poses, timestamps,
                                          do_ransac=True)
        # visualize before refinment
        pgo.visualize_in_plt()

        # Do graph optimization
        print('Performing full BA:')
        pgo.optimize(max_iterations=100)
        pgo.visualize_in_plt()

        if display_result or config['data']['save_result']:
            # visualize and save optimized point cloud
            pgo_poses = []
            for i in range(len(pgo.vertices())):
                pgo_poses.append(pgo.vertex(i).estimate().matrix())
            point_cloud = combine_point_clouds(point_clouds, pgo_poses)
            if display_down_sample_rate > 0:
                point_cloud = downsample_pcl(point_cloud, rate=display_down_sample_rate)
            if config['data']['save_result']:
                o3d.io.write_point_cloud(config['data']['save_result_name_graph_optimized'], point_cloud)
            if display_result:
                display_single_pc(point_cloud)
    print("finished")
