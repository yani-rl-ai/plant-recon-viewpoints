#!/usr/bin/env python3
"""Compare a saved 2D occupancy map with the exact world geometry (walls and pots) and draw an overlay.

    python3 check_map.py [--map ~/plant_ws/src/plant_sim/maps/plant_arena.yaml] [--png out.png]

Ground truth (gt/plants.json + the arena of gen_world.py): inner wall faces at |x| = |y| = 5.0 m and one
pot of radius 0.11 m at every plant position (the laser plane, 0.185 m above the floor, only meets the
pots - 0.20 m high - and the thin stems, never the canopy).  Exits non-zero if a check fails.
"""
import argparse
import json
import math
import os
import re
import sys

import cv2
import numpy as np

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
POT_R = 0.11
FAILS = 0


def report(ok, msg):
    global FAILS
    FAILS += 0 if ok else 1
    print(('PASS  ' if ok else 'FAIL  ') + msg)


def load_map(yaml_path):
    txt = open(yaml_path).read()
    get = lambda k: re.search(rf'^{k}:\s*(.+)$', txt, re.M).group(1).strip()          # noqa: E731
    res = float(get('resolution'))
    origin = [float(v) for v in get('origin').strip('[]').split(',')]
    occ_t, free_t, negate = float(get('occupied_thresh')), float(get('free_thresh')), int(get('negate'))
    img = cv2.imread(os.path.join(os.path.dirname(yaml_path), get('image')), cv2.IMREAD_UNCHANGED)
    p = img.astype(float) / 255.0
    occ_prob = p if negate else 1.0 - p
    occ = occ_prob > occ_t
    # nav2's saver writes free = 254, occupied = 0, unknown = 205.  205 gives occ_prob 0.196, which is BELOW
    # free_thresh 0.25, so the thresholds alone would call unknown cells free: use the pixel value instead.
    free = (img >= 250) if not negate else (img <= 5)
    return img, occ, free, res, origin


def cell_centres(shape, res, origin):
    h, w = shape
    xs = origin[0] + (np.arange(w) + 0.5) * res
    ys = origin[1] + (h - np.arange(h) - 0.5) * res
    return np.meshgrid(xs, ys)


def dist_to_gt(x, y, plants, half):
    """Distance of points to the nearest ground-truth surface (inner wall faces, pot cylinders)."""
    d = np.minimum(np.abs(half - np.abs(x)), np.abs(half - np.abs(y)))
    for px, py in plants:
        d = np.minimum(d, np.abs(np.hypot(x - px, y - py) - POT_R))
    return d


def fit_circle(P):
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    b = (P ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx, cy, math.sqrt(max(c + cx * cx + cy * cy, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default=os.path.join(PKG, 'maps', 'plant_arena.yaml'))
    ap.add_argument('--png', default=None)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(PKG, 'gt', 'plants.json')))
    plants = [tuple(p['xy']) for p in meta['plants']]
    half = meta['arena'] / 2

    img, occ, free, res, origin = load_map(args.map)
    X, Y = cell_centres(occ.shape, res, origin)
    print(f'map {occ.shape[1]} x {occ.shape[0]} cells, resolution {res} m, origin {origin}, '
          f'{occ.sum()} occupied / {free.sum()} free / {(~occ & ~free).sum()} unknown cells')

    # 1. every occupied cell must lie on a real surface
    ox, oy = X[occ], Y[occ]
    d = dist_to_gt(ox, oy, plants, half)
    report(np.median(d) < 0.05 and np.percentile(d, 95) < 0.10,
           f'occupied cells vs real surfaces: median {np.median(d) * 100:.1f} cm, 95 % {np.percentile(d, 95) * 100:.1f} cm, '
           f'max {d.max() * 100:.1f} cm (cell = {res * 100:.0f} cm)')
    spurious = d > 0.15
    report(spurious.sum() <= 0.01 * occ.sum(), f'spurious occupied cells (> 15 cm from any real surface): {spurious.sum()} '
                                               f'of {occ.sum()}')

    # 2. walls complete
    t = np.arange(-half, half + 1e-9, 0.05)
    wall_pts = np.concatenate([np.c_[t, np.full_like(t, half)], np.c_[t, np.full_like(t, -half)],
                               np.c_[np.full_like(t, half), t], np.c_[np.full_like(t, -half), t]])
    from_wall = np.array([np.min(np.hypot(ox - a, oy - b)) for a, b in wall_pts])
    report((from_wall < 0.08).mean() > 0.95, f'walls: {(from_wall < 0.08).mean() * 100:.1f} % of the 40 m of wall perimeter '
                                              f'has an occupied cell within 8 cm')

    # 3. pots
    for k, (px, py) in enumerate(plants):
        m = np.hypot(ox - px, oy - py) < 0.35
        if m.sum() < 5:
            report(False, f'pot {k} at ({px:+.1f}, {py:+.1f}): only {m.sum()} occupied cells nearby')
            continue
        P = np.c_[ox[m], oy[m]]
        cx, cy, r = fit_circle(P)
        ang = np.sort(np.degrees(np.arctan2(P[:, 1] - py, P[:, 0] - px)))
        gaps = np.diff(np.r_[ang, ang[0] + 360])
        arc = 360 - gaps.max()
        err = math.hypot(cx - px, cy - py)
        report(err < 0.05 and abs(r - POT_R) < 0.05 and arc > 120,
               f'pot {k} at ({px:+.1f}, {py:+.1f}): {m.sum():2d} cells, fitted centre error {err * 100:.1f} cm, '
               f'radius {r * 100:.1f} cm (true {POT_R * 100:.0f}), seen over {arc:.0f} deg of its circumference')

    # 4. the interior was explored
    inside = (np.abs(X) < half - 0.2) & (np.abs(Y) < half - 0.2)
    for px, py in plants:
        inside &= np.hypot(X - px, Y - py) > POT_R + 0.1
    known = (free[inside]).mean()
    report(known > 0.95, f'interior explored: {known * 100:.1f} % of the free floor is mapped as free space')

    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        h, w = img.shape
        ax.imshow(img, cmap='gray', vmin=0, vmax=255,
                  extent=[origin[0], origin[0] + w * res, origin[1], origin[1] + h * res])
        for k, (px, py) in enumerate(plants):
            ax.add_patch(plt.Circle((px, py), POT_R, fill=False, color='lime', lw=1.2))
            ax.text(px + 0.18, py + 0.18, f'plant {k}', color='lime', fontsize=8)
        ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], '-', color='cyan', lw=0.8)
        ax.set_title(f'{os.path.basename(args.map)}: saved map (grey/black) vs true walls (cyan) and pots (green circles)')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        fig.tight_layout()
        fig.savefig(args.png, dpi=90)
        print('overlay:', args.png)
    print('RESULT:', 'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
