"""Metrics of a reconstruction against the exact ground truth of the Gazebo world.

  cloud_prf        precision / recall / F1 of a point set at distance thresholds (fixed in mm, NOT tied to the voxel size)
  graph_metrics    DBL-GNG nodes: distance to the true surface, coverage of the true surface, quantisation error
  fit_sphere / match_fruits   apple localisation: sphere fit per group of fruit points, match to the true apples
  voxel_iou / shape_iou       3D voxel IoU of the merged cloud and of the DBL-GNG nodes against the true surface
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree

TAUS = (0.01, 0.02, 0.05)


def cloud_prf(model, gt_pts, taus=TAUS, gt_tree=None, observable=None):
    """Precision = share of model points within tau of the true surface; recall = share of true surface points
    within tau of the model.  `observable` (indices into gt_pts) restricts recall to the part some camera saw."""
    gt_tree = gt_tree or cKDTree(gt_pts)
    d_m2g = gt_tree.query(model, workers=-1)[0]
    mt = cKDTree(model)
    d_g2m = mt.query(gt_pts, workers=-1)[0]
    out = {'n_model': int(len(model)), 'median_err_mm': float(np.median(d_m2g) * 1000),
           'p95_err_mm': float(np.percentile(d_m2g, 95) * 1000)}
    for tau in taus:
        p = float((d_m2g <= tau).mean())
        r = float((d_g2m <= tau).mean())
        f = 2 * p * r / max(p + r, 1e-12)
        out[f'precision@{tau * 1000:.0f}mm'], out[f'recall@{tau * 1000:.0f}mm'], out[f'f1@{tau * 1000:.0f}mm'] = p, r, f
        if observable is not None and len(observable):
            ro = float((d_g2m[observable] <= tau).mean())
            out[f'recall_observed@{tau * 1000:.0f}mm'] = ro
            out[f'f1_observed@{tau * 1000:.0f}mm'] = 2 * p * ro / max(p + ro, 1e-12)
    return out


def graph_metrics(nodes, edges, labels, gt_pts, data, gt_tree=None, taus=(0.02, 0.05), observable=None):
    """Quality of the DBL-GNG topological map of one plant."""
    gt_tree = gt_tree or cKDTree(gt_pts)
    d_n = gt_tree.query(nodes, workers=-1)[0]
    d_g = cKDTree(nodes).query(gt_pts, workers=-1)[0]
    qe = float(cKDTree(nodes).query(data, workers=-1)[0].mean())
    out = {'nodes': int(len(nodes)), 'edges': int(len(edges)), 'components': int(labels.max() + 1) if len(labels) else 0,
           'node_to_surface_mean_mm': float(d_n.mean() * 1000), 'node_to_surface_p95_mm': float(np.percentile(d_n, 95) * 1000),
           'quantization_error_mm': qe * 1000}
    for tau in taus:
        out[f'coverage@{tau * 1000:.0f}mm'] = float((d_g <= tau).mean())
        if observable is not None and len(observable):
            out[f'coverage_observed@{tau * 1000:.0f}mm'] = float((d_g[observable] <= tau).mean())
    return out


# --------------------------------------------------------------------------------------------- fruit
def fit_sphere(P, r_range=(0.025, 0.055)):
    """Least-squares sphere through the visible cap of a fruit; the radius is kept inside `r_range` (an apple's
    size is known) because a small cap alone leaves the centre ambiguous along the viewing direction."""
    c0 = P.mean(0)
    r0 = np.clip(np.linalg.norm(P - c0, axis=1).mean(), *r_range)

    def res(x):
        return np.linalg.norm(P - x[:3], axis=1) - np.clip(x[3], *r_range)
    s = least_squares(res, np.r_[c0, r0], loss='soft_l1', f_scale=0.004)
    return s.x[:3], float(np.clip(s.x[3], *r_range)), float(np.sqrt(np.mean(res(s.x) ** 2)))


def match_fruits(est, gt_centres, gate=0.05):
    """Hungarian matching of estimated to true apples; returns (errors of matched [m], FP, FN)."""
    if not est or len(gt_centres) == 0:
        return [], len(est), len(gt_centres)
    C = np.linalg.norm(np.array([e[0] for e in est])[:, None] - np.asarray(gt_centres)[None], axis=2)
    r, c = linear_sum_assignment(C)
    ok = [(i, j) for i, j in zip(r, c) if C[i, j] <= gate]
    return [float(C[i, j]) for i, j in ok], len(est) - len(ok), len(gt_centres) - len(ok)


# --------------------------------------------------------------------------------------------- voxel IoU
def voxel_iou(A, B, voxel):
    """IoU of the sets of occupied voxels of two point sets (|A and B| / |A or B|), on one shared grid."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    if len(A) == 0 or len(B) == 0:
        return 0.0
    o = np.minimum(A.min(0), B.min(0))
    ia, ib = np.floor((A - o) / voxel).astype(np.int64), np.floor((B - o) / voxel).astype(np.int64)
    m = np.maximum(ia.max(0), ib.max(0)) + 1
    ka = np.unique((ia[:, 0] * m[1] + ia[:, 1]) * m[2] + ia[:, 2])
    kb = np.unique((ib[:, 0] * m[1] + ib[:, 1]) * m[2] + ib[:, 2])
    inter = np.intersect1d(ka, kb, assume_unique=True).size
    return float(inter / (ka.size + kb.size - inter))


def iou_scores(pts, gt, gt_observed=None, voxels=(0.01, 0.02, 0.05)):
    """Voxel IoU against the whole true surface and (optionally) against only the part some camera saw."""
    out = {'points': int(len(pts))}
    for v in voxels:
        out[f'iou@{v * 1000:.0f}mm'] = voxel_iou(pts, gt, v)
        if gt_observed is not None and len(gt_observed):
            out[f'iou_observed@{v * 1000:.0f}mm'] = voxel_iou(pts, gt_observed, v)
    return out


def shape_iou(sets, gt_pts, gt_label, observable, classes, voxels=(0.01, 0.02, 0.05)):
    """`sets` = {method: (points in the GT frame, class id per point)}; `classes` = {class id: name} (GT label = id - 1).
    Returns {method: {overall scores, 'per_class': {name: scores}}}."""
    obs_all = gt_pts[observable] if observable is not None and observable.any() else None
    out = {}
    for name, (P, L) in sets.items():
        r = iou_scores(P, gt_pts, obs_all, voxels)
        r['per_class'] = {}
        for c, cname in classes.items():
            g = gt_pts[gt_label == c - 1]
            go = gt_pts[(gt_label == c - 1) & observable] if obs_all is not None else None
            if (L == c).sum() > 5 and len(g) > 20:
                r['per_class'][cname] = iou_scores(P[L == c], g, go, voxels)
        out[name] = r
    return out
