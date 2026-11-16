"""Tests of the voxel-IoU comparison helpers (recon_metrics)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plant_recon.recon_metrics import shape_iou, voxel_iou  # noqa: E402


def _plane(n=4000, size=0.3, seed=0):
    r = np.random.default_rng(seed)
    return np.c_[r.uniform(0, size, (n, 2)), np.zeros(n)]


def test_voxel_iou_identical_is_one():
    P = _plane()
    assert voxel_iou(P, P, 0.01) == 1.0


def test_voxel_iou_disjoint_is_zero():
    P = _plane()
    assert voxel_iou(P, P + np.array([1.0, 0, 0]), 0.01) == 0.0


def test_voxel_iou_half_overlap():
    # two 10 x 10 slabs of voxels shifted by 5 voxels in x: 50 shared of 150 in the union.  A common anchor point (1 voxel
    # in both sets) puts the grid origin off the points, so none of them sits exactly on a voxel boundary (float rounding).
    g = np.indices((10, 10)).reshape(2, -1).T * 0.01 + 0.005
    A = np.r_[np.c_[g, np.full(len(g), 0.005)], [[-0.0075, -0.0075, -0.0075]]]
    B = np.r_[A[:-1] + np.array([0.05, 0, 0]), [[-0.0075, -0.0075, -0.0075]]]
    assert abs(voxel_iou(A, B, 0.01) - 51 / 151) < 1e-12


def test_voxel_iou_empty_is_zero():
    assert voxel_iou(np.zeros((0, 3)), _plane(), 0.01) == 0.0


def test_shape_iou_structure_and_classes():
    r = np.random.default_rng(0)
    gt = np.r_[_plane(5000), _plane(5000, seed=1) + np.array([0, 0, 0.5])]
    lab = np.r_[np.zeros(5000, int), np.ones(5000, int)]          # GT labels: 0 = pot, 1 = foliage
    L = np.r_[np.ones(5000, int), np.full(5000, 2)]               # class ids: 1 = pot, 2 = foliage
    obs = np.zeros(len(gt), bool)
    obs[:5000] = True
    out = shape_iou({'cloud': (gt, L), 'half': (gt[::2], L[::2])}, gt, lab, obs, {1: 'pot', 2: 'plant'}, voxels=(0.02,))
    assert out['cloud']['iou@20mm'] == 1.0 and out['cloud']['per_class']['pot']['iou@20mm'] == 1.0
    assert 0.0 < out['half']['iou@20mm'] <= 1.0
    assert 'iou_observed@20mm' in out['cloud']
    assert set(out['cloud']['per_class']) == {'pot', 'plant'}
