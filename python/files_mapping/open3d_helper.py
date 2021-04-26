import numpy as np
import open3d as o3d

# from hd_map.common.transform import print_tmat


def compute_normals(pcl):
    pcl_open3d = _pcl_to_open3d(pcl)
    pcl_open3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    normals = np.asarray(pcl_open3d.normals)
    return normals


def open3d_icp(src_p, src_n, dst_p, dst_n, t_init, max_iter=1000, src_name='src', dst_name='dst', max_icp_distance=1.0,
               relative_rmse=1e-10, relative_fitness=1e-10, verbose=False, point_to_point=False):
    if verbose:
        print('ICP: %s to %s' % (src_name, dst_name))

    src_open3d = _pcl_to_open3d(src_p, src_n)
    dst_open3d = _pcl_to_open3d(dst_p, dst_n)

    if point_to_point:
        reg_res = o3d.pipelines.registration.registration_icp(
            src_open3d, dst_open3d, max_icp_distance, t_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter,
                                                    relative_rmse=relative_rmse,
                                                    relative_fitness=relative_fitness))
    else:
        mu, sigma = 0, 0.1  # mean and standard deviation
        loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        reg_res = o3d.pipelines.registration.registration_icp(
            src_open3d, dst_open3d, max_icp_distance, t_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter,
                                                    relative_rmse=relative_rmse,
                                                    relative_fitness=relative_fitness))

    t_final = reg_res.transformation

    reg_init = o3d.pipelines.registration.evaluate_registration(src_open3d, dst_open3d, max_icp_distance, t_init)

    if verbose:
        print('init', t_init)
        print('icp', t_final)
        print(
            f'init: fitness: {round(reg_init.fitness, 3)}   inlier_rmse: {round(reg_init.inlier_rmse, 3)}   set: {np.asarray(reg_init.correspondence_set).shape[0]}')
        print(
            f'icp:  fitness: {round(reg_res.fitness, 3)}   inlier_rmse: {round(reg_res.inlier_rmse, 3)}   set: {np.asarray(reg_res.correspondence_set).shape[0]}')

    # nn_d = cKDTree(dst_p)
    # src_p_transformed = apply_transformation(t_final, src_p)
    # rng_d, dst_indx = nn_d.query(src_p_transformed, 1)
    # ok = rng_d < max_icp_distance
    # cc = np.stack((np.arange(src_p.shape[0])[ok], dst_indx[ok]), axis=1)

    cc = np.asarray(reg_res.correspondence_set)

    return t_final, cc, reg_res.fitness


def open3d_ransac(src_p, src_n, dst_p, dst_n, src_name='src', dst_name='dst', voxel_size=.25, verbose=False):
    if verbose:
        print('RANSAC: %s to %s' % (src_name, dst_name))

    src_open3d = _pcl_to_open3d(src_p, src_n)
    dst_open3d = _pcl_to_open3d(dst_p, dst_n)

    src_down, src_fpfh = _preprocess_point_cloud(src_open3d, voxel_size)
    dst_down, dst_fpfh = _preprocess_point_cloud(dst_open3d, voxel_size)
    distance_threshold = .5 * voxel_size

    reg_res = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        src_down, dst_down, src_fpfh, dst_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))

    t_final = reg_res.transformation

    if verbose:
        print('ransac', t_final)
        print(
            f'ransac: fitness: {round(reg_res.fitness, 3)}   inlier_rmse: {round(reg_res.inlier_rmse, 3)}   set: {np.asarray(reg_res.correspondence_set).shape[0]}')

    cc = np.asarray(reg_res.correspondence_set)

    return t_final, cc


def open3d_filter(pcl, voxel_size):
    pcl_open3d = _pcl_to_open3d(pcl)
    pcl_open3d_filtered = pcl_open3d.voxel_down_sample(voxel_size)
    pcl_filtered = _open3d_to_pcl(pcl_open3d_filtered)
    return pcl_filtered


def _pcl_to_open3d(pcl, normals=None):
    open3d_pcl = o3d.geometry.PointCloud()
    open3d_pcl.points = o3d.utility.Vector3dVector(pcl)
    if normals is not None:
        open3d_pcl.normals = o3d.utility.Vector3dVector(normals)
    return open3d_pcl


def _open3d_to_pcl(point_cloud):
    return np.asarray(point_cloud.points)


def _preprocess_point_cloud(pcd, voxel_size, verbose=False):
    print(":: Downsample with a voxel size %.3f." % voxel_size) if verbose else None
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal) if verbose else None
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature) if verbose else None
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
