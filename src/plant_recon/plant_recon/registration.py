"""Registration of a new view to the model of the plant being built (odom / TF pose as the initial guess + ICP)."""
from __future__ import annotations

import numpy as np


def _pcd(P):
    import open3d as o3d
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.ascontiguousarray(P, dtype=np.float64))
    return p


def rotation_angle(T) -> float:
    c = (np.trace(T[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def icp_refine(src, dst, max_dist=0.03, iters=40):
    """Point-to-point ICP of `src` onto `dst` (both in the same frame, so identity is the initial guess).

    Returns (T 4x4, fitness = share of src points with a partner within max_dist, inlier rmse [m])."""
    import open3d as o3d
    r = o3d.pipelines.registration.registration_icp(
        _pcd(src), _pcd(dst), max_dist, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters, relative_fitness=1e-6, relative_rmse=1e-6))
    return np.asarray(r.transformation), float(r.fitness), float(r.inlier_rmse)


def accept(T, fitness, min_fitness=0.25, max_shift=0.06, max_rot_deg=5.0):
    """Gate: a correction is applied only if the views clearly overlap and the correction is small.
    A bigger one means ICP slid on the thin leaves rather than found the real error."""
    return fitness >= min_fitness and np.linalg.norm(T[:3, 3]) <= max_shift and rotation_angle(T) <= max_rot_deg


def register_view(view_pts, model_pts, max_dist=0.03):
    """Refine `view_pts` against the accumulated `model_pts`; returns (points to merge, info dict)."""
    info = {'applied': False, 'fitness': 0.0, 'rmse': float('nan'), 'shift': 0.0, 'rot_deg': 0.0}
    if len(model_pts) < 2000 or len(view_pts) < 500:
        return view_pts, info
    T, fit, rmse = icp_refine(view_pts, model_pts, max_dist)
    info.update(fitness=fit, rmse=rmse, shift=float(np.linalg.norm(T[:3, 3])), rot_deg=rotation_angle(T))
    if accept(T, fit):
        info['applied'] = True
        return view_pts @ T[:3, :3].T + T[:3, 3], info
    return view_pts, info


def icp_multiscale(src, dst, scales=((0.04, 0.10), (0.02, 0.05), (0.01, 0.025)), iters=30):
    """Coarse-to-fine point-to-point ICP: (voxel, max correspondence distance) per level.  Returns (T, fitness at the
    finest level, inlier rmse).  The coarse levels let it recover the few-centimetre / few-degree misregistration that
    wheel odometry leaves between two stations."""
    from .geom import voxel_downsample
    T = np.eye(4)
    cur = np.asarray(src, float)
    fit, rmse = 0.0, float('nan')
    for vox, md in scales:
        dT, fit, rmse = icp_refine(voxel_downsample(cur, vox), voxel_downsample(dst, vox), md, iters)
        cur = cur @ dT[:3, :3].T + dT[:3, 3]
        T = dT @ T
    return T, fit, rmse


def register_multiscale(view_pts, model_pts, min_fitness=0.30, max_shift=0.25, max_rot_deg=10.0):
    """Correction (4x4, identity if rejected) of `view_pts` against `model_pts` + info."""
    info = {'applied': False, 'fitness': 0.0, 'rmse': float('nan'), 'shift': 0.0, 'rot_deg': 0.0}
    if len(model_pts) < 2000 or len(view_pts) < 500:
        return np.eye(4), info
    T, fit, rmse = icp_multiscale(view_pts, model_pts)
    info.update(fitness=fit, rmse=rmse, shift=float(np.linalg.norm(T[:3, 3])), rot_deg=rotation_angle(T))
    if fit >= min_fitness and info['shift'] <= max_shift and info['rot_deg'] <= max_rot_deg:
        info['applied'] = True
        return T, info
    return np.eye(4), info
