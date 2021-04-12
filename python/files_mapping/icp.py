import os
import pickle
from typing import Optional, Tuple

import numpy as np

from python.files_mapping.open3d_helper import compute_normals, open3d_icp, open3d_ransac
# from hd_map.common.transform import compute_transformation, tmat2array


def get_initial_transform(src_pcl: np.ndarray, dst_pcl: np.ndarray,
                          src_pose: np.ndarray, dst_pose: np.ndarray,
                          src_ts: int, dst_ts: int,
                          voxel_size: float,
                          verbose: bool = False):
    swap_src_dst = False if src_ts < dst_ts else True
    if swap_src_dst:
        src_ts, dst_ts = dst_ts, src_ts
        src_pcl, dst_pcl = dst_pcl, src_pcl
        src_pose, dst_pose = dst_pose, src_pose

    src_normals = compute_normals(src_pcl[:, :3])
    dst_normals = compute_normals(dst_pcl[:, :3])
    tmat, cc = open3d_ransac(src_pcl[:, :3], src_normals, dst_pcl[:, :3], dst_normals, voxel_size=voxel_size,
                             verbose=verbose, src_name=str(src_ts / 1e6), dst_name=str(dst_ts / 1e6))

    return tmat, cc


def get_icp_transform(src_pcl: np.ndarray, dst_pcl: np.ndarray,
                      src_pose: np.ndarray, dst_pose: np.ndarray,
                      src_ts: int, dst_ts: int,
                      tmp_folder: str = None,
                      src_lidar: str = 'r', dst_lidar: str = 'r',
                      recompute: bool = False,
                      max_icp_distance: float = 1,
                      tmat_init: Optional[np.ndarray] = None,
                      check_swap: bool = True,
                      verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    # Avoid having two redundant files for 'src -> dst' and 'dst -> src'
    swap_src_dst = False if not check_swap else src_ts < dst_ts
    if swap_src_dst:
        src_ts, dst_ts = dst_ts, src_ts
        src_pcl, dst_pcl = dst_pcl, src_pcl
        src_pose, dst_pose = dst_pose, src_pose
        src_lidar, dst_lidar = dst_lidar, src_lidar
        tmat_init = np.linalg.inv(tmat_init) if tmat_init is not None else None

    # Manage storage of ICP results
    if tmp_folder is not None:
        icp_filename, file_exists = _get_icp_filename(src_ts, dst_ts, tmp_folder, src_lidar, dst_lidar)
        os.makedirs(tmp_folder, exist_ok=True)
        load_from_disk = False if recompute else os.path.isfile(icp_filename)
    else:
        load_from_disk = False

    if load_from_disk:
        with open(icp_filename, 'rb') as f:
            tmat, cc, fitness = pickle.load(f)
    else:
        if tmat_init is None:
            # Compute odometry-based transform between two consecutive poses --> initial guess for ICP
            # tmat_init = np.linalg.inv(src_pose) @ dst_pose  # --> correct transform
            tmat_init = np.linalg.inv(dst_pose) @ src_pose
        # Refine initial guess using ICP
        src_normals = compute_normals(src_pcl[:, :3])
        dst_normals = compute_normals(dst_pcl[:, :3])
        tmat, cc, fitness = open3d_icp(src_pcl[:, :3], src_normals, dst_pcl[:, :3], dst_normals, tmat_init,
                                       max_iter=100, max_icp_distance=max_icp_distance, verbose=verbose,
                                       src_name=str(src_ts / 1e6), dst_name=str(dst_ts / 1e6)) #TODO: set max_iter=5000 again

    # Do not overwrite existing files
    if tmp_folder is not None and not load_from_disk:
        with open(icp_filename, 'wb') as f:
            pickle.dump((tmat, cc, fitness), f, pickle.HIGHEST_PROTOCOL)

    # Undo swapping if src and dst were not ordered according to their timestamps
    if swap_src_dst:
        tmat = np.linalg.inv(tmat)
        cc = np.fliplr(cc)
    return tmat, cc, fitness


def compute_covariance(src_p, dst_p, n_iter=100, n_points=5000):
    assert n_points <= src_p.shape[0]

    rng = np.random.default_rng()

    tmat_list = np.zeros((n_iter, 6))
    for i in range(n_iter):
        # Random subset
        indices = rng.choice(src_p.shape[0], n_points, replace=False)
        src_subset = src_p[indices]
        dst_subset = dst_p[indices]

        # Compute transform: [rx, ry, rz, tx, ty, tz]
        tmat = tmat2array(compute_transformation(src_subset, dst_subset))
        tmat_list[i, :] = tmat

    # Compute covariances
    tmat_list -= np.mean(tmat_list, axis=0)
    cov_matrix = tmat_list.T @ tmat_list
    # cov = np.mean(cov_matrix, axis=0)
    return cov_matrix


def _get_icp_filename(
        src_ts: int, dst_ts: int,
        tmp_folder: str = None,
        src_lidar: str = 'r', dst_lidar: str = 'r',
        check_swap: bool = False) -> Tuple[str, bool]:
    # Avoid having two redundant files for 'src -> dst' and 'dst -> src'
    swap_src_dst = False if not check_swap else src_ts < dst_ts
    if swap_src_dst:
        src_ts, dst_ts = dst_ts, src_ts
        src_lidar, dst_lidar = dst_lidar, src_lidar

    icp_filename = os.path.join(tmp_folder, f'{src_lidar}_{str(src_ts)}-{dst_lidar}_{str(dst_ts)}.pkl')

    return icp_filename, os.path.exists(icp_filename)
