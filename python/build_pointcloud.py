################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import os
import re
import numpy as np

from image import load_image
from camera_model import CameraModel
from python.semantic_segmentation import get_model, predict_image
from python.tools.clolor_map import trainId2label, labels, label2color, create_ade20k_label_colormap
from transform import build_se3_transform
from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud

def find_closesd_timestamp(timestamp_path, radar_timestamp):
    # Find fitting radar data
    timestamps_in_file = []
    timstamp_distance = []
    with open(timestamp_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            timstamp_distance.append(np.abs(radar_timestamp - timestamp))
            timestamps_in_file.append(timestamp)
            # if start_time <= timestamp <= end_time:
            #     timestamps_radar.append(timestamp)
    idx_of_min_dist = np.where(timstamp_distance == np.amin(timstamp_distance))
    final_timestamp = timestamps_in_file[idx_of_min_dist[0][0]]
    return final_timestamp

def build_pointcloud(lidar_dir, poses_file, extrinsics_dir, start_time, end_time, origin_time=-1, segmented=False):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        lidar_dir (str): Directory containing LIDAR scans.
        poses_file (str): Path to a file containing pose information. Can be VO or INS data.
        extrinsics_dir (str): Directory containing extrinsic calibrations.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        numpy.ndarray: 3xn array of (x, y, z) coordinates of pointcloud
        numpy.array: array of n reflectance values or None if no reflectance values are recorded (LDMRS)

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """
    if origin_time < 0:
        origin_time = start_time

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', lidar_dir).group(0)
    timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestamps.append(timestamp)

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")

    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)

    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)

        poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=(poses_type == 'rtk'))
    else:
        # sensor is VO, which is located at the main vehicle frame
        poses = interpolate_vo_poses(poses_file, timestamps, origin_time)

    if segmented:
        color_map = create_ade20k_label_colormap()
        seg_model = get_model()
        pc_colour = np.array([[0], [0], [0]]).transpose()
    pointcloud = np.array([[0], [0], [0], [0]])
    if lidar == 'ldmrs':
        reflectance = None
    else:
        reflectance = np.empty((0))

    for i in range(0, len(poses)):
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.bin')
        if "velodyne" not in lidar:
            if not os.path.isfile(scan_path):
                continue

            scan_file = open(scan_path)
            scan = np.fromfile(scan_file, np.double)
            scan_file.close()

            scan = scan.reshape((len(scan) // 3, 3)).transpose()

            if lidar != 'ldmrs':
                # LMS scans are tuples of (x, y, reflectance)
                reflectance = np.concatenate((reflectance, np.ravel(scan[2, :])))
                scan[2, :] = np.zeros((1, scan.shape[1]))
        else:
            if os.path.isfile(scan_path):
                ptcld = load_velodyne_binary(scan_path)
            else:
                scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.png')
                if not os.path.isfile(scan_path):
                    continue
                ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)
                ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)

            reflectance = np.concatenate((reflectance, ptcld[3]))
            scan = ptcld[:3]

        if segmented:
            # TODO: add comments
            this_scan = np.dot(G_posesource_laser, np.vstack([scan, np.ones((1, scan.shape[1]))]))

            model = CameraModel(args.models_dir, args.image_dir)

            extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
            with open(extrinsics_path) as extrinsics_file:
                extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

            timestamp_scan = timestamps[i+1]
            image_timestamp = find_closesd_timestamp(os.path.join(args.image_dir, '../../stereo.timestamps'), timestamp_scan)
            print(image_timestamp)
            image_path = os.path.join(args.image_dir, str(image_timestamp) + '.png')
            image = load_image(image_path, model)
            # current_pointcloud = np.dot(G_camera_posesource, this_scan)
            uv, depth, pc_in_image = model.project(this_scan, image.shape, return_pc=True)

            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
            # plt.xlim(0, image.shape[1])
            # plt.ylim(image.shape[0], 0)
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()

            seg_iamge = predict_image(seg_model, image)
            print(np.unique(seg_iamge))

            this_pc_colour = np.zeros((pc_in_image[3,:].size,3))
            for j in range(pc_in_image[3,:].size):
                if uv[1,j] < seg_iamge.shape[0] and uv[0,j] < seg_iamge.shape[1]:
                    value = seg_iamge[int(uv[1, j]), int(uv[0, j])]
                    if value>18:
                        value=255
                    my_color = np.array(trainId2label[value].color)
                    # my_color = np.array(color_map[value])
                    this_pc_colour[j] = my_color#np.ones(3)*seg_iamge[int(uv[1,j]),int(uv[0,j])] / 21
            # TODO: make colours out of the labels - not just grey steps


            # project_image_in_radar(image_timestamp, cart_img, image, model, target_dim=(cart_pixel_width - 1) / 2,
            #                                               scale=cart_resolution,
            #                                               show=False, save=True)

            scan = np.dot(poses[i], pc_in_image) # only the labeled once are displayed
            # scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))
            pointcloud = np.hstack([pointcloud, scan])
            pc_colour = np.vstack([pc_colour, this_pc_colour])
        else:
            scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))
            pointcloud = np.hstack([pointcloud, scan])

    pointcloud = pointcloud[:, 1:]
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    if segmented:
        pc_colour = pc_colour[1:, :]
        return pointcloud, reflectance, pc_colour
    else:
        return pointcloud, reflectance


if __name__ == "__main__":
    import argparse
    import open3d

    parser = argparse.ArgumentParser(description='Build and display a pointcloud')
    parser.add_argument('--poses_file', type=str, default=None, help='File containing relative or absolute poses')
    parser.add_argument('--laser_dir', type=str, default=None, help='Directory containing LIDAR data')
    parser.add_argument('--extrinsics_dir', type=str, default=None,
                        help='Directory containing extrinsic calibrations')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--models_dir', type=str, help='Directory containing camera models')

    args = parser.parse_args()

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', args.laser_dir).group(0)
    timestamps_path = os.path.join(args.laser_dir, os.pardir, lidar + '.timestamps')
    with open(timestamps_path) as timestamps_file:
        start_time = int(next(timestamps_file).split(' ')[0])

    end_time = start_time + 1e8

    pointcloud, reflectance, pc_colour = build_pointcloud(args.laser_dir, args.poses_file,
                                               args.extrinsics_dir, start_time, end_time, segmented=True)

    if reflectance is not None:
        colours = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
        colours = 1 / (1 + np.exp(-10 * (colours - colours.mean())))
    else:
        colours = 'gray'

    colours = np.ones(reflectance.shape) / 1.5

    # Pointcloud Visualisation using Open3D
    vis = open3d.Visualizer()
    vis.create_window(window_name=os.path.basename(__file__))
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
    render_option.point_color_option = open3d.PointColorOption.Default
    coordinate_frame = open3d.geometry.create_mesh_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    pcd = open3d.geometry.PointCloud()
    my_colors = np.vstack((pointcloud[3],pointcloud[3],pointcloud[3])).transpose()
    #TODO: set the colour to RGB value from image -> later to segmented value
    pcd.points = open3d.utility.Vector3dVector(
        -np.ascontiguousarray(pointcloud[[1, 0, 2]].transpose().astype(np.float64)))
    # pcd.colors = open3d.utility.Vector3dVector(np.tile(colours[:, np.newaxis], (1, 3)).astype(np.float64))
    # label2color
    # labels
    # pcd.colors = open3d.utility.Vector3dVector(np.array([pc_colour[:,0],pc_colour[:,1],pc_colour[:,2]]).transpose())
    pcd.colors = open3d.utility.Vector3dVector(pc_colour)
    # pcd.colors = open3d.utility.Vector3dVector(np.ones((pointcloud.shape[1],3))/2)


    # import open3d as o3d
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array([[1, 2, 3], [4, 5, 6]]))
    #
    # # 0.2, 0.3 are the intensity of the two points
    # pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0]]))

    open3d.io.write_point_cloud("output/Point_Clouds/pc_test.ply", pcd)
    # pcd.paint_uniform_color([1, 0.706, 0])
    # Rotate pointcloud to align    displayed coordinate frame colouring
    pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()
