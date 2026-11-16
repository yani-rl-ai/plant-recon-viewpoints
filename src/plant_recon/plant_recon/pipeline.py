"""PlantReconstructor: from camera frames of ONE plant to its DBL-GNG topological map, with metrics.

Per frame:  segment (YOLO plant region + colour)  ->  3D points (depth)  ->  odom pose  ->  [ICP to the model]
            ->  merge (5 mm voxels)  ->  DBL-GNG update (paper Sec. 5.7: continue training, raise max nodes)
Metrics against the Gazebo ground truth are computed every frame (cloud) and at the end (graph, fruits).

The same class runs live inside scan_manager.py and offline from recorded frames (offline_recon.py), so every
ablation uses exactly the code that produced the live result.  Nothing in here reads the ground truth to build the
model; the ground truth only feeds the metrics (and the 'oracle' segmentation used as an upper bound).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .dbl_gng import DBLGNG
from .geom import backproject, transform_points, voxel_downsample
from .recon_metrics import cloud_prf, graph_metrics, match_fruits, shape_iou
from .registration import register_multiscale, register_view
from .segment import BG, FRUIT, POT, FOLIAGE, SOIL, hsv_classes, iou_prf, pick_instance, refine_in_region


@dataclass
class Config:
    seg: str = 'sam'                 # sam (MobileSAM + YOLO/geometric prompts) | hybrid | color | yolo_region | oracle
    use_icp: bool = True
    icp_mode: str = 'station'        # station: one correction per station (its tilt views share the robot pose) | per_view
    update_gng: bool = True          # False: skip the DBL-GNG updates (fast offline ablations)
    align_each_view: bool = True     # shape metric after rigid alignment to the ground truth at every view
    voxel: float = 0.005             # model voxel [m]
    gng_voxel: float = 0.01          # DBL-GNG input voxel
    node_frac: float = 0.10          # max nodes = node_frac * #input points, clipped to node_min..node_max
    node_min: int = 400
    node_max: int = 1500
    gng_starts: int = 10             # eta of Alg. 5
    gng_epochs: int = 10             # epochs per incremental update
    crop_radius: float = 0.95        # keep points within this horizontal distance of the plant prior [m]
    z_range: tuple = (0.012, 1.75)    # above the floor (SAM masks leak a little onto it), up to the top of the plants
    flying_rel: float = 0.02         # a depth pixel needs a neighbour within max(2 cm, 2 % of its depth)
    prior_px_gate: int = 90          # how far from the expected plant position (pixels) a YOLO instance may be
    seed: int = 0


def flying_pixel_ok(depth, rel=0.02, abs_min=0.02):
    """Depth pixels that have at least one 4-neighbour at nearly the same depth (removes mixed-depth edge pixels)."""
    z = depth.astype(np.float64)
    ok = np.isfinite(z)
    similar = np.zeros(z.shape, bool)
    tol = np.maximum(abs_min, rel * np.where(ok, z, 0.0))
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        n = np.roll(z, shift, axis=axis)
        similar |= np.isfinite(n) & (np.abs(n - z) < tol)
    return ok & similar


class PlantReconstructor:
    def __init__(self, plant_id, prior_xy, gt, segmenter=None, cfg: Config | None = None):
        self.k, self.prior_xy, self.gt = plant_id, np.asarray(prior_xy, float), gt
        self.seg, self.cfg = segmenter, cfg or Config()
        sel = np.nonzero(gt.plant == plant_id)[0]
        self.gt_idx0 = int(sel[0])
        self.gt_pts = gt.points[sel]
        self.gt_label = gt.label[sel]
        self.gt_tree = cKDTree(self.gt_pts)
        self.gt_fruit_centres = [f[1] for f in gt.fruits if f[0] == plant_id]
        self.observable = np.zeros(len(sel), bool)
        self.P = np.zeros((0, 3))
        self.C = np.zeros((0, 3))
        self.L = np.zeros(0, np.int8)
        self.gng: DBLGNG | None = None
        self.views: list[dict] = []
        self.last_overlay = None
        self._station, self._station_corr = None, np.eye(4)

    # ------------------------------------------------------------------ segmentation
    def segment(self, bgr, depth, K, T_odom_opt, T_gt_opt=None):
        """Returns class map (0 bg, 1 pot, 2 foliage, 3 fruit) of the target plant and the region used."""
        cfg = self.cfg
        if cfg.seg == 'oracle':
            cls, pid, _ = self.gt.label_pixels(depth, K, T_gt_opt)
            cls[pid != self.k + 1] = BG
            return cls, pid == self.k + 1
        if cfg.seg == 'sam':
            Ti = np.linalg.inv(T_odom_opt)

            def project(P):
                Pc = (Ti[:3, :3] @ np.asarray(P, float).T).T + Ti[:3, 3]
                z = Pc[:, 2]
                return K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2], K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2], z
            return self.seg(bgr, project, self.prior_xy)
        centre = np.array([self.prior_xy[0], self.prior_xy[1], 0.75, 1.0])
        pc = np.linalg.inv(T_odom_opt) @ centre
        uv = None
        if pc[2] > 0.1:
            uv = (K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2])
        h, w = depth.shape
        if cfg.seg == 'color':
            region = np.zeros((h, w), bool)
            if uv is not None:                                   # generous box around the expected plant
                r = int(K[0, 0] * 0.8 / max(pc[2], 0.3))
                region[max(0, int(uv[1]) - 2 * r):int(uv[1]) + 2 * r, max(0, int(uv[0]) - r):int(uv[0]) + r] = True
            return refine_in_region(bgr, region, grow=1), region
        dets = self.seg(bgr)
        inst = pick_instance(dets, uv, 'plant') if uv is not None else None
        if inst is None:
            return np.zeros((h, w), np.uint8), np.zeros((h, w), bool)
        region = inst['mask']
        if cfg.seg == 'yolo_region':
            cls = np.where(region, FOLIAGE, BG).astype(np.uint8)
            return cls, region
        return refine_in_region(bgr, region), region

    # ------------------------------------------------------------------ one frame
    def add_view(self, bgr, depth, K, T_odom_opt, T_gt_opt=None, station=None):
        cfg, t0 = self.cfg, time.perf_counter()
        info = {'view': len(self.views) + 1}
        cls, region = self.segment(bgr, depth, K, T_odom_opt, T_gt_opt)
        self.last_cls, self.last_region = cls, region          # for the live overlay
        info['t_seg'] = time.perf_counter() - t0
        info['plant_px'] = int((cls > 0).sum())
        if T_gt_opt is not None:
            gcls, gpid, _, hits = self.gt.label_pixels(depth, K, T_gt_opt, return_hits=True)
            gt_mask = gpid == self.k + 1
            info['seg_vs_gt'] = iou_prf(cls > 0, gt_mask)
            mine = hits[(hits >= self.gt_idx0) & (hits < self.gt_idx0 + len(self.observable))] - self.gt_idx0
            self.observable[mine] = True
        valid = flying_pixel_ok(depth, cfg.flying_rel) & (cls > 0)
        Pc, (rows, cols) = backproject(np.where(valid, depth, np.nan), K)
        P = transform_points(T_odom_opt, Pc)
        C = bgr[rows, cols][:, ::-1] / 255.0
        L = cls[rows, cols]
        d_xy = np.linalg.norm(P[:, :2] - self.prior_xy, axis=1)
        keep = (d_xy < cfg.crop_radius) & (P[:, 2] > cfg.z_range[0]) & (P[:, 2] < cfg.z_range[1])
        P, C, L = P[keep], C[keep], L[keep]
        info['points_raw'] = int(len(P))
        onehot = np.eye(5)[L][:, 1:]
        P, C, onehot = voxel_downsample(P, cfg.voxel, C, onehot)
        info['points_view'] = int(len(P))
        # --- registration to what has been built so far (odom pose = initial guess)
        t1 = time.perf_counter()
        if cfg.use_icp and len(self.P):
            new_station = station is None or station != getattr(self, '_station', None) or cfg.icp_mode == 'per_view'
            if new_station:
                Tc, icp = register_multiscale(voxel_downsample(P, 0.01), voxel_downsample(self.P, 0.01))
                self._station_corr, self._station = Tc, station
                info['icp'] = icp
            else:                                              # same robot pose as the previous view: same correction
                Tc = self._station_corr
                info['icp'] = {'applied': bool(np.any(Tc != np.eye(4))), 'reused': True}
            if info['icp']['applied']:
                P = transform_points(Tc, P)
        elif station is not None:
            self._station, self._station_corr = station, np.eye(4)
        info['t_icp'] = time.perf_counter() - t1
        # --- merge
        Lnew = onehot
        if len(self.P):
            old = np.eye(5)[self.L][:, 1:]
            P_all, C_all, oh_all = np.vstack([self.P, P]), np.vstack([self.C, C]), np.vstack([old, Lnew])
        else:
            P_all, C_all, oh_all = P, C, Lnew
        self.P, self.C, oh = voxel_downsample(P_all, cfg.voxel, C_all, oh_all)
        self.L = (np.argmax(oh, axis=1) + 1).astype(np.int8)
        info['model_points'] = int(len(self.P))
        # --- DBL-GNG
        t2 = time.perf_counter()
        if cfg.update_gng:
            self.update_gng()
        info['t_gng'] = time.perf_counter() - t2
        info['gng_nodes'] = int(len(self.gng.W)) if self.gng is not None else 0
        # --- metrics of the cloud
        if len(self.P) > 100:
            info['cloud'] = cloud_prf(self.P, self.gt_pts, gt_tree=self.gt_tree,
                                      observable=np.nonzero(self.observable)[0] if self.observable.any() else None)
        if cfg.align_each_view and len(self.P) > 500:   # shape quality independent of the odometry drift of the whole model
            from .registration import icp_refine
            T, _, _ = icp_refine(voxel_downsample(self.P, 0.01), self.gt_pts[::4], max_dist=0.25, iters=30)
            info['align_shift_mm'] = float(np.linalg.norm(T[:3, 3]) * 1000)
            info['cloud_aligned'] = cloud_prf(transform_points(T, self.P), self.gt_pts, taus=(0.01, 0.02), gt_tree=self.gt_tree)
        self.views.append(info)
        return info

    # ------------------------------------------------------------------ DBL-GNG
    def gng_input(self):
        return voxel_downsample(self.P, self.cfg.gng_voxel)

    def update_gng(self):
        X = self.gng_input()
        if len(X) < 3 * self.cfg.gng_starts + 10:
            return
        M = int(np.clip(self.cfg.node_frac * len(X), self.cfg.node_min, self.cfg.node_max))
        if self.gng is None:
            self.gng = DBLGNG(max_nodes=M, n_starts=self.cfg.gng_starts, extra_epochs=3, seed=self.cfg.seed).fit(X)
        else:                                                    # paper Sec. 5.7: keep the network, raise max nodes
            self.gng.partial_fit(X, n_epochs=self.cfg.gng_epochs, max_nodes=max(M, self.gng.M))

    def graph(self):
        g = self.gng.graph()
        return g.nodes, g.edges, g.components()

    # ------------------------------------------------------------------ fruit via the graph
    def node_classes(self, nodes):
        """Class of every graph node = majority class of the model points nearest to it."""
        idx = cKDTree(nodes).query(self.P, workers=-1)[1]
        votes = np.zeros((len(nodes), 5))
        np.add.at(votes, (idx, self.L.astype(int)), 1)
        return votes[:, 1:].argmax(1) + 1, idx

    def fruit_estimates(self):
        """Apples from the DBL-GNG: connected groups of fruit nodes -> their points -> sphere fit."""
        from .recon_metrics import fit_sphere
        out = {'gng': []}
        if self.gng is not None:
            nodes, edges, _ = self.graph()
            ncls, nearest = self.node_classes(nodes)
            fnode = ncls == FRUIT
            if fnode.any():
                from .dbl_gng import connected_components
                sub = np.nonzero(fnode)[0]
                remap = -np.ones(len(nodes), int)
                remap[sub] = np.arange(len(sub))
                e = np.array([(remap[a], remap[b]) for a, b in edges if fnode[a] and fnode[b]]).reshape(-1, 2)
                comp = connected_components(len(sub), e)
                node_comp = -np.ones(len(nodes), int)
                node_comp[sub] = comp
                pt_comp = node_comp[nearest]
                for c in range(comp.max() + 1):
                    pts = self.P[(pt_comp == c) & (self.L == FRUIT)]
                    if len(pts) >= 25:
                        ctr, r, rms = fit_sphere(pts)
                        out['gng'].append((ctr, r, len(pts), rms))
        return out

    # ------------------------------------------------------------------ voxel IoU: merged cloud and DBL-GNG nodes
    def shape_iou_report(self, R, t):
        """3D voxel IoU (1, 2, 5 cm) against the true surface, in the aligned frame, overall and per class, for
          cloud    every point of the merged model
          dbl_gng  the nodes of the incremental DBL-GNG"""
        Pa = self.P @ R.T + t
        sets = {'cloud': (Pa, self.L)}
        if self.gng is not None:
            nodes, _, _ = self.graph()
            ncls, _ = self.node_classes(nodes)
            sets['dbl_gng'] = (nodes @ R.T + t, ncls)
        return shape_iou(sets, self.gt_pts, self.gt_label, self.observable, {POT: 'pot', FOLIAGE: 'plant', FRUIT: 'apple', SOIL: 'soil'})

    # ------------------------------------------------------------------ end of the plant
    def finalize(self, offline_gng=True):
        """Final metrics, in three frames:
          absolute   the model in the frame it was built in (odom or map): includes ALL pose drift of the robot;
          aligned    after the best rigid alignment of the whole model to the ground truth surface: the quality of the
                     reconstruction itself (consistency of the views, segmentation, depth), independent of drift;
          anchored   translated so that the pot sits at the known slot position (rotation untouched).
        Graph, apple and per-class metrics use the aligned frame; apples are also scored in the absolute frame."""
        res = {'plant': self.k, 'views': len(self.views), 'model_points': int(len(self.P))}
        obs = np.nonzero(self.observable)[0] if self.observable.any() else None
        res['observable_gt_fraction'] = float(self.observable.mean())
        res['cloud'] = cloud_prf(self.P, self.gt_pts, gt_tree=self.gt_tree, observable=obs)
        R, t = np.eye(3), np.zeros(3)
        if len(self.P) > 500:
            from .registration import icp_refine
            T, fit, rmse = icp_refine(voxel_downsample(self.P, 0.01), self.gt_pts[::3], max_dist=0.5, iters=60)
            R, t = T[:3, :3], T[:3, 3]
            res['alignment_to_gt'] = {'shift_mm': float(np.linalg.norm(t) * 1000),
                                      'rot_deg': float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                                      'fitness': fit}
            res['cloud_aligned'] = cloud_prf(self.P @ R.T + t, self.gt_pts, gt_tree=self.gt_tree, observable=obs)
        # anchored to the plant slot (pot circle fitted on the pot-wall points of all views; soil surface height)
        pot = self.P[(self.L == POT) & (self.P[:, 2] < 0.2)]
        soil = self.P[self.L == SOIL]
        if len(pot) > 200:
            A = np.c_[2 * pot[:, 0], 2 * pot[:, 1], np.ones(len(pot))]
            cx, cy, _ = np.linalg.lstsq(A, (pot[:, :2] ** 2).sum(1), rcond=None)[0]
            dz = (0.203 - float(np.median(soil[:, 2]))) if len(soil) > 50 else 0.0
            shift = np.array([self.prior_xy[0] - cx, self.prior_xy[1] - cy, dz])
            res['anchor_shift_mm'] = (shift * 1000).tolist()
            res['cloud_anchored'] = cloud_prf(self.P + shift, self.gt_pts, gt_tree=self.gt_tree, observable=obs)
        # per class (aligned frame)
        Pa = self.P @ R.T + t
        res['per_class'] = {}
        for c, name in ((POT, 'pot'), (FOLIAGE, 'plant'), (FRUIT, 'apple'), (SOIL, 'soil')):
            mp, mg = Pa[self.L == c], self.gt_pts[self.gt_label == c - 1]
            if len(mp) > 20 and len(mg) > 20:
                pc = cloud_prf(mp, mg)
                pc['observed_gt_fraction'] = float(self.observable[self.gt_label == c - 1].mean()) if self.observable.any() else None
                res['per_class'][name] = pc
        # DBL-GNG graph (aligned frame)
        X = self.gng_input()
        Xa = X @ R.T + t
        if self.gng is not None:
            nodes, edges, comp = self.graph()
            res['gng_incremental'] = graph_metrics(nodes @ R.T + t, edges, comp, self.gt_pts, Xa, gt_tree=self.gt_tree, observable=obs)
            res['gng_incremental']['epochs_total'] = len(self.gng.history)
            res['gng_incremental']['movement_last_epoch'] = float(self.gng.history[-1]['movement'])
            res['gng_incremental']['nodes_added_per_epoch_max'] = int(max(h['added'] for h in self.gng.history))
        if offline_gng and len(X) > 3 * self.cfg.gng_starts + 10:
            M = int(np.clip(self.cfg.node_frac * len(X), self.cfg.node_min, self.cfg.node_max))
            t0 = time.perf_counter()
            g = DBLGNG(max_nodes=M, n_starts=self.cfg.gng_starts, seed=self.cfg.seed).fit(X)
            gr = g.graph()
            res['gng_offline'] = graph_metrics(gr.nodes @ R.T + t, gr.edges, gr.components(), self.gt_pts, Xa, gt_tree=self.gt_tree, observable=obs)
            res['gng_offline']['fit_seconds'] = time.perf_counter() - t0
            res['gng_offline']['epochs_total'] = len(g.history)
            res['gng_offline']['nodes_added_per_epoch_max'] = int(max(h['added'] for h in g.history))
            self.gng_offline = g
        res['shape_iou'] = self.shape_iou_report(R, t)
        # apples: DBL-GNG groups of fruit nodes, scored aligned and absolute
        est = self.fruit_estimates()
        res['fruit'] = {'n_true': len(self.gt_fruit_centres)}
        res['fruit_absolute'] = {'n_true': len(self.gt_fruit_centres)}
        for name, e in est.items():
            for key, (RR, tt) in (('fruit', (R, t)), ('fruit_absolute', (np.eye(3), np.zeros(3)))):
                ee = [(RR @ c + tt, r, n, rms) for c, r, n, rms in e]
                errs, fp, fn = match_fruits(ee, self.gt_fruit_centres, gate=0.05)
                res[key][name] = {'estimated': len(e), 'matched': len(errs), 'false_positive': fp, 'false_negative': fn,
                                  'mean_err_mm': float(np.mean(errs) * 1000) if errs else None,
                                  'max_err_mm': float(np.max(errs) * 1000) if errs else None}
        return res


def _rigid_from_points(A, B):
    """Rigid transform (Kabsch) that maps A onto B for corresponding rows (A: points before ICP, B: after)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T
