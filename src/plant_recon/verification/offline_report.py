#!/usr/bin/env python3
"""Final report of a scan run, recomputed from the recorded frames with the final pipeline code.

    python3 offline_report.py --run ~/plant_ws/results/scan_20261115_132000

For every plant: the 12 recorded views go through the SAME PlantReconstructor as in the live scan (MobileSAM
segmentation, 3D points, odom pose + per-station multi-scale ICP, incremental DBL-GNG), then the metrics against the
exact Gazebo ground truth.  A second pass uses the AMCL (map) camera pose instead of the odometry pose.
Writes REPORT.md and report.json into the run directory.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plant_recon.geom import GroundTruth  # noqa: E402
from plant_recon.pipeline import Config, PlantReconstructor  # noqa: E402
from plant_recon.segment import SamSegmenter  # noqa: E402
from offline_recon import load_views  # noqa: E402


def f(x, nd=3):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def replay(views, gt, seg, pose, gng):
    k = int(views[0]['plant'])
    prior = views[0]['prior'] if pose == 'odom' else gt.centres[k]
    cfg = Config(update_gng=gng, align_each_view=False)
    rec = PlantReconstructor(k, prior, gt, seg, cfg)
    seg_iou, perr, rerr = [], [], []
    for v in views:
        T = v['T_odom_opt'] if pose == 'odom' else v['T_map_opt']
        info = rec.add_view(v['color'], v['depth'].astype(np.float32), v['K'], T, v['T_gt_opt'], station=int(v['station']))
        seg_iou.append(info['seg_vs_gt']['iou'])
        Tg = v['T_gt_opt']
        perr.append(float(np.linalg.norm(T[:3, 3] - Tg[:3, 3]) * 1000))
        rerr.append(float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ Tg[:3, :3]) - 1) / 2, -1, 1)))))
    res = rec.finalize(offline_gng=gng)
    res.update(seg_iou_mean=float(np.mean(seg_iou)), pose_err_mm_median=float(np.median(perr)), pose_err_deg_median=float(np.median(rerr)))
    return res, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--sam', default=os.path.expanduser('~/plant_ws/models/mobile_sam.pt'))
    ap.add_argument('--yolo', default=os.path.expanduser('~/plant_ws/models/yolo26n-seg.pt'))
    ap.add_argument('--device', default='cpu', help='cpu (default: the GPU is usually busy with Gazebo/RViz) or 0')
    args = ap.parse_args()
    gt = GroundTruth(os.path.expanduser('~/plant_ws/src/plant_sim/gt'))
    seg = SamSegmenter(args.sam, args.yolo, device=(args.device if args.device == 'cpu' else int(args.device)), gate_colour=True)
    plants = sorted(int(d.split('plant')[-1]) for d in glob.glob(f'{args.run}/plant*') if glob.glob(f'{d}/view_*.npz'))
    out = {'odom': {}, 'map': {}}
    for k in plants:
        views = load_views(f'{args.run}/plant{k}')
        t = time.time()
        out['odom'][k], rec = replay(views, gt, seg, 'odom', True)
        out['map'][k], _ = replay(views, gt, seg, 'map', False)
        gh = rec.gng.history
        out['odom'][k]['gng_signature'] = {'epochs': len(gh), 'max_nodes_added_in_one_epoch': int(max(h['added'] for h in gh)),
                                           'final_nodes': int(len(rec.gng.W)), 'start_pairs': int(rec.gng.n_starts)}
        print(f'plant {k}: {len(views)} views replayed in {time.time() - t:.0f} s', flush=True)
    json.dump(out, open(f'{args.run}/report.json', 'w'), indent=1, default=float)

    L = []
    L.append(f'# Scan report: {os.path.basename(args.run)}\n')
    L.append('Camera + odometry reconstruction (MobileSAM segmentation, odom pose, per-station ICP, incremental DBL-GNG) of the '
             'plants, scored against the exact Gazebo ground truth. "absolute" = in the frame the model was built in (includes the '
             'odometry drift of the robot); "aligned" = after the best rigid alignment of the whole model to the true surface '
             '(quality of the reconstruction itself); "anchored" = translated so the pot sits at its slot.\n')
    L.append('## 1. Point cloud vs ground truth (all classes)\n')
    L.append('| plant | views | points | F1@1cm abs | F1@2cm abs | F1@5cm abs | F1@1cm anchored | F1@1cm aligned | F1@2cm aligned | F1@5cm aligned | F1@2cm aligned, part seen | precision@1cm aligned | recall@1cm aligned | median err aligned [mm] | AMCL-pose F1@2cm aligned |')
    L.append('|' + '---|' * 15)
    for k in plants:
        r, m = out['odom'][k], out['map'][k]
        c, a, an = r['cloud'], r['cloud_aligned'], r.get('cloud_anchored', {})
        L.append(f'| {k} | {r["views"]} | {r["model_points"]} | {f(c["f1@10mm"])} | {f(c["f1@20mm"])} | {f(c["f1@50mm"])} | {f(an.get("f1@10mm"))} | '
                 f'{f(a["f1@10mm"])} | {f(a["f1@20mm"])} | {f(a["f1@50mm"])} | {f(a.get("f1_observed@20mm"))} | {f(a["precision@10mm"])} | '
                 f'{f(a["recall@10mm"])} | {f(a["median_err_mm"], 1)} | {f(m["cloud_aligned"]["f1@20mm"])} |')
    mean = lambda key, fld: np.mean([out['odom'][k][key][fld] for k in plants])                          # noqa: E731
    L.append(f'| **mean** | | | {f(mean("cloud", "f1@10mm"))} | {f(mean("cloud", "f1@20mm"))} | {f(mean("cloud", "f1@50mm"))} | - | '
             f'{f(mean("cloud_aligned", "f1@10mm"))} | {f(mean("cloud_aligned", "f1@20mm"))} | {f(mean("cloud_aligned", "f1@50mm"))} | - | - | - | - | - |\n')
    L.append('## 2. Per class, aligned frame (F1 at 2 cm; precision / recall at 1 cm)\n')
    L.append('| plant | pot F1@2cm | plant F1@2cm | apple F1@2cm | soil F1@2cm | pot P/R@1cm | plant P/R@1cm | apple P/R@1cm | soil P/R@1cm |')
    L.append('|' + '---|' * 9)
    for k in plants:
        pc = out['odom'][k]['per_class']
        row = f'| {k} |'
        for n in ('pot', 'plant', 'apple', 'soil'):
            row += f' {f(pc.get(n, {}).get("f1@20mm"))} |'
        for n in ('pot', 'plant', 'apple', 'soil'):
            v = pc.get(n)
            row += f' {f(v["precision@10mm"], 2) + "/" + f(v["recall@10mm"], 2) if v else "-"} |'
        L.append(row)
    L.append('\n## 3. DBL-GNG topological map (aligned frame)\n')
    L.append('| plant | nodes | edges | components | node-to-surface mean [mm] | p95 [mm] | quantization error [mm] | coverage@2cm (seen part) | coverage@5cm (seen part) | incremental epochs | max nodes added in one epoch | offline fit nodes | offline node-to-surface [mm] | offline fit [s] |')
    L.append('|' + '---|' * 14)
    for k in plants:
        g, o = out['odom'][k]['gng_incremental'], out['odom'][k].get('gng_offline', {})
        L.append(f'| {k} | {g["nodes"]} | {g["edges"]} | {g["components"]} | {f(g["node_to_surface_mean_mm"], 2)} | {f(g["node_to_surface_p95_mm"], 1)} | '
                 f'{f(g["quantization_error_mm"], 1)} | {f(g.get("coverage_observed@20mm"))} | {f(g.get("coverage_observed@50mm"))} | {g["epochs_total"]} | '
                 f'{g["nodes_added_per_epoch_max"]} | {o.get("nodes", "-")} | {f(o.get("node_to_surface_mean_mm"), 2)} | {f(o.get("fit_seconds"), 1)} |')
    L.append('\n## 4. Apple localisation (centre error vs the true apple centres, matched within 5 cm)\n')
    L.append('| plant | true apples | DBL-GNG matched | DBL-GNG FP | DBL-GNG mean err [mm] | DBL-GNG matched, absolute frame |')
    L.append('|' + '---|' * 6)
    tm = tt = 0
    for k in plants:
        fr, fa = out['odom'][k]['fruit'], out['odom'][k]['fruit_absolute']
        tm += fr['gng']['matched']; tt += fr['n_true']
        L.append(f'| {k} | {fr["n_true"]} | {fr["gng"]["matched"]} | {fr["gng"]["false_positive"]} | {f(fr["gng"]["mean_err_mm"], 1)} | '
                 f'{fa["gng"]["matched"]} |')
    L.append(f'| **all** | {tt} | {tm} | | | |\n')
    L.append('## 5. Segmentation and camera pose\n')
    L.append('| plant | MobileSAM mask IoU vs GT (mean over views) | odom camera-pose error median [mm] / [deg] | AMCL camera-pose error median [mm] / [deg] | alignment shift / rotation to GT [mm / deg] |')
    L.append('|---|---|---|---|---|')
    for k in plants:
        r, m = out['odom'][k], out['map'][k]
        al = r.get('alignment_to_gt', {})
        L.append(f'| {k} | {f(r["seg_iou_mean"], 2)} | {r["pose_err_mm_median"]:.0f} / {r["pose_err_deg_median"]:.1f} | {m["pose_err_mm_median"]:.0f} / {m["pose_err_deg_median"]:.1f} | '
                 f'{al.get("shift_mm", float("nan")):.0f} / {al.get("rot_deg", float("nan")):.1f} |')
    open(f'{args.run}/REPORT.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
