#!/usr/bin/env python3
"""Reproduce the numbers of Siow et al. 2024 (Tables 3-5, 7, 8, Fig. 8/10) with plant_recon.dbl_gng.

Datasets: the benchmark files of https://github.com/CornerSiow/DBL-GNG (folder dataset/).

    python3 reproduce_paper.py --datasets /path/to/DBL-GNG/dataset --runs 10 --out /tmp/dbl_fig

Paper protocol (Sec. 4.3): learn until max nodes, then 10 more epochs; mean of 10 runs.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plant_recon.dbl_gng import DBLGNG  # noqa: E402

FILES = {"Agg": "Aggregation.txt", "D31": "D31.txt", "R15": "R15.txt",
         "Path": "pathbased.txt", "Sp": "spiral.txt", "H72D": "H72D.txt"}
T3 = {"Agg": (68, 7), "D31": (93, 31), "R15": (45, 15), "Path": (30, 3), "Sp": (96, 3), "H72D": (360, 72)}
P_QE = {"Agg": .981, "D31": .627, "R15": .246, "Path": 1.288, "Sp": .449, "H72D": .485}      # Table 4
P_TIME = {"Agg": .053, "D31": .054, "R15": .023, "Path": .029, "Sp": .076, "H72D": .852}      # Table 5
P_CUT = {"Agg": .45, "D31": .65, "R15": .65, "Path": .55, "Sp": .10, "H72D": .45}            # Table 7
P_ACC_TUNED = {"Agg": .968, "D31": .963, "R15": .994, "Path": .805, "Sp": 1.0, "H72D": .901}  # Table 8
P_ACC_015 = {"Agg": .808, "D31": .158, "R15": .730, "Path": .367, "Sp": .998, "H72D": .436}   # Table 8


def accuracy(true, pred):
    """One-to-one (Hungarian) matched clustering accuracy."""
    t = np.unique(true, return_inverse=True)[1]
    p = np.unique(pred, return_inverse=True)[1]
    cont = np.zeros((t.max() + 1, p.max() + 1))
    np.add.at(cont, (t, p), 1)
    r, c = linear_sum_assignment(-cont)
    return cont[r, c].sum() / len(t)


def cluster_acc(model, X, y, p, inclusive):
    gr = model.graph(cut_p=p, inclusive=inclusive)
    comp = gr.components()
    idx, _ = model.nearest_node(X, gr.nodes)
    return accuracy(y, comp[idx]), comp.max() + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    names = [args.only] if args.only else list(FILES)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    print("=== Table 4 / 5: quantization error and time (mean over %d runs) ===" % args.runs)
    print(f"{'set':5s} {'QE ours':>16s} {'QE paper':>9s} {'time ours':>10s} {'t paper':>8s} "
          f"{'ep->M':>6s} {'starts hit clusters':>20s}")
    rows = {}
    for name in names:
        raw = np.loadtxt(os.path.join(args.datasets, FILES[name]))
        X, y = raw[:, :2], raw[:, 2].astype(int)
        M, eta = T3[name]
        qe, tm, ep, hit = [], [], [], []
        models = []
        for r in range(args.runs):
            g = DBLGNG(max_nodes=M, n_starts=eta, cut_p=0.15, seed=r)
            t0 = time.perf_counter()
            g.fit(X)
            tm.append(time.perf_counter() - t0)
            qe.append(g.quantization_error(X))
            ep.append(next(h["epoch"] for h in g.history if h["nodes"] >= M))
            near, _ = g.nearest_node(g.start_nodes, nodes=X)      # label of the datum nearest to each start
            hit.append(len(set(y[near])) / len(set(y)))
            models.append(g)
        rows[name] = (X, y, models)
        print(f"{name:5s} {np.mean(qe):8.3f} ±{np.std(qe):5.3f} {P_QE[name]:9.3f} {np.mean(tm):9.3f}s "
              f"{P_TIME[name]:7.3f}s {np.mean(ep):6.1f} {np.mean(hit) * 100:19.0f}%")

    print("\n=== Table 8: clustering accuracy (Hungarian-matched), mean over %d runs ===" % args.runs)
    print("edge-cut rule: 'paper' keeps S > P (Eq. 33), 'code' keeps S >= P (authors' repo)")
    print(f"{'set':5s} {'p':>5s} | {'paper':>6s} {'code':>6s} | {'paper-ref':>9s}   || "
          f"{'p=.15':>5s} | {'paper':>6s} {'code':>6s} | {'paper-ref':>9s}")
    for name, (X, y, models) in rows.items():
        out = {}
        for tag, p in (("tuned", P_CUT[name]), ("p15", 0.15)):
            for incl in (False, True):
                a = [cluster_acc(g, X, y, p, incl)[0] for g in models]
                out[(tag, incl)] = np.mean(a)
        print(f"{name:5s} {P_CUT[name]:5.2f} | {out[('tuned', False)]:6.3f} {out[('tuned', True)]:6.3f} | "
              f"{P_ACC_TUNED[name]:9.3f}   || {0.15:5.2f} | {out[('p15', False)]:6.3f} {out[('p15', True)]:6.3f} | "
              f"{P_ACC_015[name]:9.3f}")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(15, 9))
        for ax, (name, (X, y, models)) in zip(axs.ravel(), rows.items()):
            g = models[0]
            gr = g.graph(cut_p=P_CUT[name], inclusive=False)
            ax.scatter(X[:, 0], X[:, 1], s=4, c="violet")
            for i, j in gr.edges:
                ax.plot(*gr.nodes[[i, j]].T, "k-", lw=1)
            ax.plot(*gr.nodes.T, "k.", ms=5)
            ax.plot(*g.start_nodes.T, "r+", ms=8)
            ax.set_title(f"{name}  M={T3[name][0]} eta={T3[name][1]}  p={P_CUT[name]}  (red + = Alg.5 starts)")
            ax.set_aspect("equal")
        fig.tight_layout()
        path = os.path.join(args.out, "topological_maps.png")
        fig.savefig(path, dpi=80)
        print("figure:", path)


if __name__ == "__main__":
    main()
