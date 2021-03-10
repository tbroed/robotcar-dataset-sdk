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
###############################################################################

from typing import AnyStr, Tuple
import numpy as np
import cv2
import math

from scipy import ndimage
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import utils.ipm.utils as ipm_util


def load_radar(example_path: AnyStr) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Decode a single Oxford Radar RobotCar Dataset radar example
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset Example png
    Returns:
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        fft_data (np.ndarray): Radar power readings along each azimuth
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
    """
    # Hard coded configuration to simplify parsing code
    radar_resolution = np.array([0.0432], np.float32)
    encoder_size = 5600

    raw_example_data = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.

    return timestamps, azimuths, valid, fft_data, radar_resolution


def radar_polar_to_cartesian(azimuths: np.ndarray, fft_data: np.ndarray, radar_resolution: float,
                             cart_resolution: float, cart_pixel_width: int, interpolate_crossover=True) -> np.ndarray:
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_size (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
            below for a full explanation of how this is used.
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    Notes:
        After using the warping grid the output radar cartesian is defined as as follows where
        X and Y are the `real` world locations of the pixels in metres:
         If 'cart_pixel_width' is odd:
                        +------ Y = -1 * cart_resolution (m)
                        |+----- Y =  0 (m) at centre pixel
                        ||+---- Y =  1 * cart_resolution (m)
                        |||+--- Y =  2 * cart_resolution (m)
                        |||| +- Y =  cart_pixel_width // 2 * cart_resolution (m) (at last pixel)
                        |||| +-----------+
                        vvvv             v
         +---------------+---------------+
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+ <-- X = 0 (m) at centre pixel
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+
         <------------------------------->
             cart_pixel_width (pixels)
         If 'cart_pixel_width' is even:
                        +------ Y = -0.5 * cart_resolution (m)
                        |+----- Y =  0.5 * cart_resolution (m)
                        ||+---- Y =  1.5 * cart_resolution (m)
                        |||+--- Y =  2.5 * cart_resolution (m)
                        |||| +- Y =  (cart_pixel_width / 2 - 0.5) * cart_resolution (m) (at last pixel)
                        |||| +----------+
                        vvvv            v
         +------------------------------+
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         +------------------------------+
         <------------------------------>
             cart_pixel_width (pixels)
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
    Y, X = np.meshgrid(coords, -coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = azimuths[1] - azimuths[0]
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    return cart_img


def create_radar_point_cloud(radar_img, G_posesource_radar, radar_timestamp, rel_intesnity_thresh=5,
                             z_value=-0.45):  # , radar_cartesian, intesnity_thresh):
    radar_peaks = radar_to_radar_peaks(radar_img, radar_timestamp, rel_intesnity_thresh=rel_intesnity_thresh)
    shape = radar_img.shape
    radar_pc = np.array([[0], [0], [0], [0]])
    for peak in radar_peaks:
        # X forward; Y to the right; Z downwards
        x = ((shape[1] / 2) - peak[1]) / 2
        y = -((shape[0] / 2) - peak[0]) / 2
        z = z_value
        reflectance = 1
        if x > 0:
            scan = np.dot(G_posesource_radar, np.array([[x], [y], [z], [reflectance]]))
            radar_pc = np.hstack([radar_pc, scan])
    radar_pc = radar_pc[:, 1:]
    return radar_pc


def radar_to_radar_peaks(radar_cartesian, radar_timestamp, rel_intesnity_thresh=5):
    # detected_points = cfar2d(radar_cartesian, 10, 4, 0.5)
    # radar_pc should be a point cloud with shape Nx5 (x,y,z,intensity,ring)
    mean = np.mean(radar_cartesian)
    std = np.std(radar_cartesian)
    radar_peaks_coordinates = peak_local_max(radar_cartesian[:, :, 0], min_distance=1,
                                             threshold_abs=mean + (rel_intesnity_thresh * std))
    # radar_peaks = detect_peaks(radar_cartesian)
    plt.close("all")
    plt.imshow(radar_cartesian)  # , cmap=plt.cm.gray)
    plt.plot(radar_peaks_coordinates[:, 1], radar_peaks_coordinates[:, 0], 'r.')
    # plt.show()
    # plt.savefig("output/radar_in_image_25_m/cart/" + str(radar_timestamp) + ".png")
    radar_pc = np.zeros((radar_peaks_coordinates.shape[0], 2))
    for i, coord in enumerate(radar_peaks_coordinates):
        radar_pc[i] = [coord[1], coord[0]]

    return radar_pc


def pol2cart(rho, phi):
    x = rho * math.cos(math.radians(phi))
    y = rho * math.sin(math.radians(phi))
    return (x, y)

def project_image_in_radar(radar_timestamp, cart_img, image, model , target_dim=500, scale=0.1, save=None, show=None):
    target_dim = int(target_dim)
    TARGET_H, TARGET_W = target_dim, target_dim

    # Camera extrinsics left wide
    x, y, z, r, p, y = 0, 0, 1.52, 0, 0.05, 0

    R_veh2cam = np.transpose(ipm_util.rotation_from_euler(r, p, y))
    T_veh2cam = ipm_util.translation_matrix((-x, -y, -z))
    # Rotate to camera coodinates
    R = np.transpose(np.array([[0., 0., 1., 0.],
                               [1., 0., 0., 0.],
                               [0., -1., 0., 0.],
                               [0., 0., 0., 1.]]))
    RT = R @ R_veh2cam @ T_veh2cam
    extrinsic = RT

    # Define the plane on the region of interest (road)
    plane = ipm_util.Plane(0, -(0.1 * TARGET_H / 2), 0, 0, 0, 0, TARGET_H, TARGET_W, scale)
    # Resolution: 0.1m per pixel

    fx, fy = model.focal_length
    u0, v0 = model.principal_point

    # Intrinsic
    K = np.array([[fx, 0, u0, 0],
                  [0, fy, v0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    intrinsics = K

    warped1 = ipm_util.ipm_from_parameters(image, plane.xyz, intrinsics, extrinsic, TARGET_H, TARGET_W)
    rotated_img = ndimage.rotate(warped1, 90)


    # create overlay and safe it to output
    plt.close("all")
    plt.imshow(cart_img / cart_img.max())
    zeros = np.zeros([np.shape(cart_img)[0], np.shape(cart_img)[1], 3])
    overlay_shape = rotated_img.shape
    x_start = np.int((np.shape(cart_img)[0] - rotated_img.shape[0]) / 2)
    y_start = np.int((np.shape(cart_img)[1] - rotated_img.shape[1]) / 2)
    zeros[:overlay_shape[0], y_start:overlay_shape[1] + y_start, :] = rotated_img / 255
    plt.imshow(zeros, alpha=0.7)
    if show==True:
        plt.show()
    if save==True:
        plt.savefig("output/image_in_radar_100m/" + str(radar_timestamp) + ".png")
