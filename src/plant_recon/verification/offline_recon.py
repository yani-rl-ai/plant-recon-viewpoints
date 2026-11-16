#!/usr/bin/env python3
"""Replay the recorded frames of a scan run through the SAME PlantReconstructor, with different settings.

    python3 offline_recon.py --run ~/plant_ws/results/scan_20261115_131500 --plant 1 [--full]

Every view of a run is stored by scan_manager.py (colour, depth, camera info, odom / map / ground-truth camera pose),
so ablations need no simulator.  Prints, per configuration: absolute F1 (model in the odom or map frame, drift included),
shape F1 after the best rigid alignment to the ground truth, F1 of only the part of the plant the cameras saw,
and how many station corrections ICP applied.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plant_recon.geom import GroundTruth  # noqa: E402
from plant_recon.pipeline import Config, PlantReconstructor  # noqa: E402
from plant_recon.segment import SamSegmenter  # noqa: E402


def load_views(pdir):
    out = []
    for f in sorted(glob.glob(f'{pdir}/view_*.npz')):
        d = np.load(f, allow_pickle=True)
        out.append({k: d[k] for k in d.files})
    return out


def run_config(name, views, gt, seg, cfg, pose='odom', verbose=False):
    k = int(views[0]['plant'])
    prior = views[0]['prior'] if pose == 'odom' else gt.centres[k]
    rec = PlantReconstructor(k, prior, gt, seg, cfg)
    t = time.time()
    icp_on = 0
    for v in views:
        T = v['T_odom_opt'] if pose == 'odom' else v['T_map_opt']
        info = rec.add_view(v['color'], v['depth'].astype(np.float32), v['K'], T, v['T_gt_opt'], station=int(v['station']))
        if info.get('icp', {}).get('applied') and not info['icp'].get('reused'):
            icp_on += 1
    res = rec.finalize(offline_gng=False)
    c, a = res['cloud'], res.get('cloud_aligned', {})
    perr = [np.linalg.norm((v['T_odom_opt'] if pose == 'odom' else v['T_map_opt'])[:3, 3] - v['T_gt_opt'][:3, 3]) * 1000 for v in views]
    print(f'{name:44s} abs F1@1cm {c["f1@10mm"]:.3f} @2cm {c["f1@20mm"]:.3f} | shape(aligned) @1cm {a.get("f1@10mm", 0):.3f} @2cm {a.get("f1@20mm", 0):.3f} '
          f'| observed-part abs @2cm {c.get("f1_observed@20mm", 0):.3f} | median err {c["median_err_mm"]:.1f} mm | station corrections {icp_on} '
          f'| cam-pose err median {np.median(perr):.0f} mm | {time.time() - t:.0f} s', flush=True)
    return res, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--plant', type=int, required=True)
    ap.add_argument('--sam', default=os.path.expanduser('~/plant_ws/models/mobile_sam.pt'))
    ap.add_argument('--yolo', default=os.path.expanduser('~/plant_ws/models/yolo26n-seg.pt'))
    args = ap.parse_args()
    gt = GroundTruth(os.path.expanduser('~/plant_ws/src/plant_sim/gt'))
    views = load_views(f'{args.run}/plant{args.plant}')
    print(f'{len(views)} recorded views of plant {args.plant} in {args.run}\n')
    sam = SamSegmenter(args.sam, args.yolo, gate_colour=True)
    base = dict(update_gng=False, align_each_view=False)
    run_config('odom, no ICP', views, gt, sam, Config(use_icp=False, **base))
    run_config('odom, old-style ICP (per view)', views, gt, sam, Config(icp_mode='per_view', **base))
    run_config('odom, multi-scale ICP per station', views, gt, sam, Config(icp_mode='station', **base))
    run_config('odom, multi-scale ICP per view', views, gt, sam, Config(icp_mode='per_view', **base))
    run_config('map (AMCL) pose, no ICP', views, gt, sam, Config(use_icp=False, **base), pose='map')
    run_config('map (AMCL) pose, multi-scale ICP per station', views, gt, sam, Config(icp_mode='station', **base), pose='map')
    run_config('odom, ICP per station, oracle segmentation', views, gt, None, Config(seg='oracle', icp_mode='station', **base))


if __name__ == '__main__':
    main()
