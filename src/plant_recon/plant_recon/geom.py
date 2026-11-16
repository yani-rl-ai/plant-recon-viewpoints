"""Geometry helpers shared by the dataset collector, the scan manager and the evaluation (pure numpy/scipy)."""
from __future__ import annotations

import json
import os

import numpy as np

# camera mounting constants of plant_sim (urdf/common.xacro): lens 15 cm in front of the tilt axis;
# optical frame = camera_link rotated rpy(-pi/2, 0, -pi/2) (x right, y down, z forward)
LENS_X = 0.015


def quat_to_R(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def rpy_to_R(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_from_msg(tf) -> np.ndarray:
    """4x4 from a geometry_msgs/TransformStamped (or anything with .transform.translation/.rotation)."""
    p, q = tf.transform.translation, tf.transform.rotation
    return make_T(quat_to_R(q.x, q.y, q.z, q.w), [p.x, p.y, p.z])


def invert_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


R_OPTICAL = rpy_to_R(-np.pi / 2, 0.0, -np.pi / 2)        # camera_link -> camera_optical_frame


def optical_pose_from_gt(T_world_model, T_model_tiltlink):
    """World pose of camera_optical_frame from Gazebo's model pose and the tilt-link pose (relative to the model)."""
    return T_world_model @ T_model_tiltlink @ make_T(np.eye(3), [LENS_X, 0, 0]) @ make_T(R_OPTICAL, [0, 0, 0])


def pixel_grid(K, shape):
    h, w = shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]


def backproject(depth, K, mask=None, near=0.3, far=8.0):
    """Points (n, 3) in the optical frame of the pixels with valid depth (and mask); also their pixel indices."""
    xn, yn = pixel_grid(K, depth.shape)
    ok = np.isfinite(depth) & (depth > near) & (depth < far)
    if mask is not None:
        ok &= mask
    z = depth[ok].astype(np.float64)
    return np.stack([xn[ok] * z, yn[ok] * z, z], axis=1), np.nonzero(ok)


def transform_points(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def voxel_downsample(P, voxel, *extra):
    """One point per voxel (mean); `extra` arrays (colours ...) are averaged the same way."""
    if len(P) == 0:
        return (P,) + tuple(extra)
    key = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = []
    for A in (P,) + tuple(extra):
        acc = np.zeros((len(cnt), A.shape[1]))
        np.add.at(acc, inv, A.astype(np.float64))
        out.append(acc / cnt[:, None])
    return tuple(out) if extra else out[0]


# --------------------------------------------------------------------------- ground truth of the world
class GroundTruth:
    """Exact plant geometry written by plant_sim/scripts/gen_world.py."""

    def __init__(self, gt_dir):
        from scipy.spatial import cKDTree
        d = np.load(os.path.join(gt_dir, 'plants_gt.npz'))
        self.points = d['points'].astype(np.float64)
        lab = d['label'].astype(np.int8)    # stored: 0 pot + soil, 1 foliage, 2 fruit
        lab[(lab == 0) & (d['points'][:, 2] > 0.2005)] = 3        # the soil disc lies at z = 0.203, the pot wall below 0.200
        self.label = lab                   # 0 pot, 1 foliage, 2 fruit, 3 soil   (class image value = label + 1)
        self.plant = d['plant']            # 0..5
        meta = json.load(open(os.path.join(gt_dir, 'plants.json')))
        self.meta = meta
        self.centres = np.array([p['xy'] for p in meta['plants']])
        self.fruits = [(k, np.array(f['center']), f['radius']) for k, p in enumerate(meta['plants']) for f in p['fruits']]
        self.tree = cKDTree(self.points)

    def label_pixels(self, depth, K, T_world_optical, tol=0.02, return_hits=False):
        """Per-pixel ground truth by comparing the measured surface point with the exact plant surface.

        Returns class (0 background, 1 pot, 2 foliage, 3 fruit, 4 soil), plant id (0 = none, else id + 1) and
        fruit id (0 = none, else index + 1 into self.fruits) images, all of the depth image's shape; with
        return_hits also the unique indices of the ground-truth points this view actually saw."""
        P, (rows, cols) = backproject(depth, K)
        W = transform_points(T_world_optical, P)
        dist, idx = self.tree.query(W, k=1, distance_upper_bound=tol, workers=-1)
        hit = np.isfinite(dist)
        cls = np.zeros(depth.shape, np.uint8)
        pid = np.zeros(depth.shape, np.uint8)
        fid = np.zeros(depth.shape, np.uint16)
        r, c = rows[hit], cols[hit]
        cls[r, c] = self.label[idx[hit]] + 1
        pid[r, c] = self.plant[idx[hit]] + 1
        fr = hit.copy()
        fr[hit] = self.label[idx[hit]] == 2
        if fr.any():
            Wf, pf = W[fr], self.plant[idx[fr]]
            best = np.zeros(len(Wf), np.uint16)
            bestd = np.full(len(Wf), np.inf)
            for j, (k, c0, rad) in enumerate(self.fruits):
                m = pf == k
                if m.any():
                    e = np.abs(np.linalg.norm(Wf[m] - c0, axis=1) - rad)
                    better = e < bestd[m]
                    sel = np.nonzero(m)[0][better]
                    best[sel], bestd[sel] = j + 1, e[better]
            fid[rows[fr], cols[fr]] = best
        if return_hits:
            return cls, pid, fid, np.unique(idx[hit])
        return cls, pid, fid
