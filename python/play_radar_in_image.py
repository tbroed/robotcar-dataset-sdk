################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import argparse
import os
from radar import load_radar, radar_polar_to_cartesian, create_radar_point_cloud
import numpy as np
import cv2
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel
import matplotlib.pyplot as plt

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

parser = argparse.ArgumentParser(description='Play back radar data from a given directory')

parser.add_argument('dir', type=str, help='Directory containing radar data.')
parser.add_argument('--image_dir', type=str, help='Directory containing images')
parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')

args = parser.parse_args()

model = CameraModel(args.models_dir, args.image_dir)


timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, 'radar.timestamps'))
if not os.path.isfile(timestamps_path):
    raise IOError("Could not find timestamps file")

# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .5 #1 #.25
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 101  # 501 # pixels
interpolate_crossover = True

title = "Radar Visualisation Example"

radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
cout = 0
for radar_timestamp in radar_timestamps:
    filename = os.path.join(args.dir, str(radar_timestamp) + '.png')

    if not os.path.isfile(filename):
        raise FileNotFoundError("Could not find radar example: {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                        interpolate_crossover)

    # Combine polar and cartesian for visualisation
    # The raw polar data is resized to the height of the cartesian representation
    # downsample_rate = 4
    # fft_data_vis = fft_data[:, ::downsample_rate]
    # resize_factor = float(cart_img.shape[0]) / float(fft_data_vis.shape[0])
    # fft_data_vis = cv2.resize(fft_data_vis, (0, 0), None, resize_factor, resize_factor)
    # vis = cv2.hconcat((fft_data_vis, fft_data_vis[:, :10] * 0 + 1, cart_img))
    # cv2.imshow(title, vis * 2.)  # The data is doubled to improve visualisation

    # laod radar extrinsics
    radar_extrinsics_path = os.path.join(args.extrinsics_dir, 'radar.txt')
    with open(radar_extrinsics_path) as radar_extrinsics_file:
        radar_extrinsics = [float(x) for x in next(radar_extrinsics_file).split(' ')]
    G_posesource_radar = build_se3_transform(radar_extrinsics)

    pointcloud = create_radar_point_cloud(cart_img, G_posesource_radar, radar_timestamp, rel_intesnity_thresh=3, z_value=0)

    # pc = create_radar_point_cloud(cart_img)

    cv2.waitKey(100)
    cout += 1
    # cv2.imwrite("output/radar_" + str(cout) + ".png", vis * 255.)

    image_timestamp = find_closesd_timestamp(os.path.join(args.dir, '../stereo.timestamps'), radar_timestamp)
    # TODO: adjust for movement (pose + ego motion in scan)

    image_path = os.path.join(args.image_dir, str(image_timestamp) + '.png')
    image = load_image(image_path, model)

    uv, depth = model.project(pointcloud, image.shape)

    plt.close("all")
    plt.imshow(image)
    plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=20, c=depth, edgecolors='none', cmap='jet')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./output/radar_in_image_25_m/img/" + str(radar_timestamp) + ".png")