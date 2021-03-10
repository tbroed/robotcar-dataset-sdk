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
import matplotlib.pyplot as plt
import argparse

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel
from radar import load_radar, radar_polar_to_cartesian, create_radar_point_cloud


def find_closesd_radar_timestamp(radar_dir, image_timestamp):
    # Find fitting radar data
    timestamps_path_radar = os.path.join(radar_dir, os.pardir, 'radar.timestamps')
    timestamps_radar = []
    timstamp_distance = []
    with open(timestamps_path_radar) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            timstamp_distance.append(np.abs(image_timestamp - timestamp))
            timestamps_radar.append(timestamp)
            # if start_time <= timestamp <= end_time:
            #     timestamps_radar.append(timestamp)
    idx_of_min_dist = np.where(timstamp_distance == np.amin(timstamp_distance))
    radar_timestamp = timestamps_radar[idx_of_min_dist[0][0]]

    # if len(timestamps_radar) == 0:
    #     raise ValueError("No radar data in the given time bracket.")

    return radar_timestamp

def load_radar_image(radar_timestamp, radar_dir, cart_pixel_width=651):
    # TODO: load radar image and calculate point cloud (all x>0) -> display via LiDAR and adjust extrinsics

    # Cartesian Visualsation Setup
    # Resolution of the cartesian form of the radar scan in metres per pixel
    cart_resolution = .5
    # Cartesian visualisation size (used for both height and width)
    cart_pixel_width = cart_pixel_width # 651  # pixels
    interpolate_crossover = True

    filename = os.path.join(radar_dir, '../radar/' + str(radar_timestamp) + '.png')

    if not os.path.isfile(filename):
        raise FileNotFoundError("Could not find radar example: {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                        interpolate_crossover)

    return cart_img

parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
parser.add_argument('--image_dir', type=str, help='Directory containing images')
parser.add_argument('--radar_dir', type=str, help='Directory containing LIDAR scans')
parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses')
parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
parser.add_argument('--image_idx', type=int, help='Index of image to display')

args = parser.parse_args()

model = CameraModel(args.models_dir, args.image_dir)

for i in range(1,200,25):
    args.image_idx = i

    extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
    G_camera_vehicle = build_se3_transform(extrinsics)
    G_camera_posesource = None

    # laod radar extrinsics
    radar_extrinsics_path = os.path.join(args.extrinsics_dir, 'radar.txt')
    with open(radar_extrinsics_path) as radar_extrinsics_file:
        radar_extrinsics = [float(x) for x in next(radar_extrinsics_file).split(' ')]
    G_posesource_radar = build_se3_transform(radar_extrinsics)

    poses_type = re.search('(vo|ins|rtk)\.csv', args.poses_file).group(1)
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
    else:
        # VO frame and vehicle frame are the same
        G_camera_posesource = G_camera_vehicle


    timestamps_path = os.path.join(args.image_dir, os.pardir, model.camera + '.timestamps')
    if not os.path.isfile(timestamps_path):
        timestamps_path = os.path.join(args.image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

    timestamp = 0
    with open(timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            if i == args.image_idx:
                timestamp = int(line.split(' ')[0])

    radar_timestamp = find_closesd_radar_timestamp(args.radar_dir, timestamp)

    radar_image = load_radar_image(radar_timestamp, args.radar_dir, cart_pixel_width=151)

    pointcloud = create_radar_point_cloud(radar_image, G_posesource_radar, rel_intesnity_thresh=3, z_value=0)

    # pointcloud, reflectance = build_pointcloud(args.radar_dir, args.poses_file, args.extrinsics_dir,
    #                                            timestamp - 1e7, timestamp + 1e7, timestamp)


    # pointcloud = np.dot(G_camera_posesource, pointcloud)
    # X [0] has to be greater 0
    # X forward; Y to the right; Z downwards
    # pointcloud = np.array([[5,0,0,1],[5,0,1,1], [5,0,2,1], [5,0,3,1], [5,0,4,1]]).T
    # pointcloud = np.array([[10,0,-0.45,1],[1,0,-0.45,1], [2,0,-0.45,1], [5,0,-0.45,1]]).T



    image_path = os.path.join(args.image_dir, str(timestamp) + '.png')
    image = load_image(image_path, model)

    uv, depth = model.project(pointcloud, image.shape)

    plt.imshow(image)
    plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=20, c=depth, edgecolors='none', cmap='jet')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.xticks([])
    plt.yticks([])
    plt.show()


