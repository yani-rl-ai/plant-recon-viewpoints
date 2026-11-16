"""Distributed Batch Learning Growing Neural Gas (DBL-GNG).

Implementation written from the equations of

    C. Z. Siow, A. A. Saputra, T. Obo, N. Kubota,
    "Distributed Batch Learning of Growing Neural Gas for Quick and Efficient
    Clustering", Mathematics 2024, 12(12), 1909. https://doi.org/10.3390/math12121909

Comments such as "Eq. 22" or "Alg. 2" refer to that paper. The authors' reference
code (https://github.com/CornerSiow/DBL-GNG, MIT) was only used afterwards to
cross-check the result (see ``verification/``); no code was copied from it.

What makes it DBL-GNG (and not a plain batch GNG with edge aging):

* Alg. 5  distributed initialisation: ``n_starts`` pairs of nodes spread over the
          data (next start is drawn from the *farthest* batch of the data).
* Alg. 1  full-batch learning with matrix calculations: winner / second winner for
          every datum, accumulated winner error (Eq. 22), delta movement of winners
          (Eq. 23) and of their neighbours (Eq. 24), activation counts (Eq. 25-26)
          and *edge strength* (Eq. 27).  There is NO edge age.
* Alg. 2  network update: nodes move by the *averaged* delta (Eq. 28); edges are
          rebuilt from the edge strength of this epoch (Eq. 29), i.e. edges are
          reset every epoch; isolated nodes are removed (Eq. 10); errors are
          discounted (Eq. 11); with 10 % probability inactive nodes (A1 == 0) are
          removed (Eq. 31).
* Alg. 3  distributed growth: g = #(E > P_E(0.85)) nodes are inserted per epoch
          (Eq. 34), each between the highest-error node and its highest-error
          neighbour (Eq. 13-16, 18, 32).
* Alg. 4  edge cutting by the percentile of the edge strength (Eq. 33).

``fit`` follows Alg. 6 and the stopping rule of Sec. 4.3: learn until the maximum
number of nodes is reached, then train ``extra_epochs`` (=10) more epochs, then cut.

Deviations / safeguards (none of them change the algorithm on well-behaved data):

* The sums over all data of Eq. 22-27 are computed with ``bincount`` and one
  adjacency product instead of building the n x m identity-matrix products.
  They are mathematically identical; ``tests/test_dbl_gng.py`` checks this against
  a literal transcription of the equations.
* Nodes created by Alg. 3 get an activation count of 1 (the paper does not say
  what A1 of a node that did not take part in the batch is; without it they would
  be deleted as "inactive" at once).  The reference code does the same.
* q1 = argmax(E) is taken among nodes that have at least one neighbour, and growth
  stops when no such node with E > 0 exists (the paper has no rule for that case).
* ``partial_fit`` (continue training an existing network on new data, optionally
  with a larger ``max_nodes``) is only the usage described in Sec. 5.7 for dynamic
  data; it re-uses exactly the same epoch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    """Topological map: node positions and undirected edges (i < j)."""

    nodes: np.ndarray            # (k, F)
    edges: np.ndarray            # (e, 2) int, i < j
    index: np.ndarray            # (k,) index of each node in the (un-cut) network

    def components(self) -> np.ndarray:
        """Connected-component label of every node."""
        return connected_components(len(self.nodes), self.edges)


def connected_components(n: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in np.asarray(edges, dtype=int).reshape(-1, 2):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


class DBLGNG:
    """Distributed Batch Learning Growing Neural Gas.

    Parameters follow Sec. 4.2 / Table 2 of the paper.

    max_nodes      M      maximum number of nodes (custom).
    n_starts       eta    number of starting node pairs (custom, Alg. 5).
    alpha          winner learning rate           (0.5)
    beta           neighbour learning rate        (0.01)
    rho            error discount of q1, q2       (0.5)
    delta          error discount of all nodes    (0.5)
    eps            division guard of Eq. 28       (1e-4)
    growth_p       percentile of Eq. 34           (0.85)
    cut_p          percentile of Eq. 33           (0.15); ``None`` = no cutting
    cut_inclusive  True : keep edges with S >= P  (default, as in the authors' code)
                   False: keep edges with S >  P  (Eq. 33 read literally)
                   Edge strengths are integer counts with many ties at the percentile,
                   so the two rules cut very different amounts.  Table 8 of the paper
                   is only reproduced with ">=" (Spiral, p = 0.10: 0.99 vs 0.60 with
                   ">"; the paper reports 1.000), see verification/reproduce_paper.py.
    inactive_prob  probability of removing inactive nodes each epoch (0.1)
    extra_epochs   epochs trained after M is reached (10, Sec. 4.3)
    chunk          rows of X processed at once (memory only, result is identical)
    """

    def __init__(self, max_nodes: int, n_starts: int = 1, alpha: float = 0.5,
                 beta: float = 0.01, rho: float = 0.5, delta: float = 0.5,
                 eps: float = 1e-4, growth_p: float = 0.85,
                 cut_p: float | None = 0.15, cut_inclusive: bool = True,
                 inactive_prob: float = 0.1, extra_epochs: int = 10,
                 max_epochs: int = 500, chunk: int = 8192, seed: int | None = None):
        self.M = int(max_nodes)
        self.n_starts = int(n_starts)
        self.alpha, self.beta = alpha, beta
        self.rho, self.delta, self.eps = rho, delta, eps
        self.growth_p, self.cut_p, self.cut_inclusive = growth_p, cut_p, cut_inclusive
        self.inactive_prob = inactive_prob
        self.extra_epochs, self.max_epochs, self.chunk = extra_epochs, max_epochs, chunk
        self.rng = np.random.default_rng(seed)
        self.W = None            # (m, F) node positions
        self.adj = None          # (m, m) bool, edge matrix c
        self.E = None            # (m,)   accumulated error
        self.S = None            # (m, m) edge strength of the last epoch
        self.A1 = None           # (m,)   winner activation count of the last epoch
        self.A2 = None           # (m,)   neighbour activation count of the last epoch
        self.start_nodes = None  # (2*eta, F) nodes created by Alg. 5
        self.history: list[dict] = []

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _as_data(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be (n_samples, n_features)")
        return X

    def _subset(self, keep: np.ndarray) -> None:
        """Eq. 30: delete the nodes with keep == False from every variable."""
        self.W = self.W[keep]
        self.E = self.E[keep]
        self.A1 = self.A1[keep]
        self.A2 = self.A2[keep]
        self.S = self.S[np.ix_(keep, keep)]
        self.adj = self.adj[np.ix_(keep, keep)]

    # --------------------------------------------------- Alg. 5 initialisation
    def init_distributed(self, X) -> None:
        X = self._as_data(X)
        D, F = X.shape
        eta = self.n_starts
        B = D // eta                                     # Eq. 35
        if B < 3:
            raise ValueError("too few data points for the requested n_starts")
        if self.M <= 2 * eta:
            raise ValueError("max_nodes must be larger than 2 * n_starts")
        Xc = X.copy()                                    # "copy entire data as X"
        nodes = np.empty((2 * eta, F))
        for i in range(eta):
            if i == 0:
                # "We first randomly select a starting point from the data"
                pick = int(self.rng.integers(len(Xc)))
            else:
                # Eq. 36: random datum of the batch at the end (= farthest) of X
                pick = len(Xc) - B + int(self.rng.integers(B))
            w = Xc[pick]
            # Eq. 37: distance up to the constant |w|^2 (does not change the order)
            d = np.einsum("ij,ij->i", Xc, Xc) - 2.0 * (Xc @ w)
            order = np.argsort(d, kind="stable")         # Eq. 38 nearest -> farthest
            nodes[2 * i] = w
            nodes[2 * i + 1] = Xc[order[2]]              # Eq. 39 third nearest
            Xc = Xc[order[B:]]                           # Eq. 41 drop nearest batch
        m = 2 * eta
        self.W = nodes
        self.adj = np.zeros((m, m), dtype=bool)
        idx = np.arange(eta)
        self.adj[2 * idx, 2 * idx + 1] = True            # Eq. 40
        self.adj[2 * idx + 1, 2 * idx] = True
        self.E = np.zeros(m)
        self.S = np.zeros((m, m), dtype=np.float32)
        self.A1 = np.ones(m)
        self.A2 = np.ones(m)
        self.start_nodes = nodes.copy()
        self.history = []

    # ----------------------------------------------------- Alg. 1 batch learning
    def _batch_learning(self, X: np.ndarray) -> dict:
        """Alg. 1 on the whole of X (processed in chunks; sums are exact)."""
        m, F = self.W.shape
        W = self.W
        w2 = np.einsum("ij,ij->i", W, W)
        A1 = np.zeros(m)
        Q = np.zeros((m, F))                  # sum of the data won by each node
        err = np.zeros(m)                     # sum of winner distances
        cnt = np.zeros(m * m)                 # (s1, s2) co-occurrence counts
        for start in range(0, len(X), self.chunk):
            x = X[start:start + self.chunk]
            n = len(x)
            rows = np.arange(n)
            # Eq. 19: all distances without a loop
            d2 = np.einsum("ij,ij->i", x, x)[:, None] + w2[None, :] - 2.0 * (x @ W.T)
            d = np.sqrt(np.maximum(d2, 0.0))
            s1 = np.argmin(d, axis=1)                         # Eq. 20
            d1 = d[rows, s1]
            d[rows, s1] = np.inf                              # Eq. 21
            s2 = np.argmin(d, axis=1)
            A1 += np.bincount(s1, minlength=m)                # Eq. 25
            err += np.bincount(s1, weights=d1, minlength=m)   # inner sum of Eq. 22
            for f in range(F):
                Q[:, f] += np.bincount(s1, weights=x[:, f], minlength=m)
            cnt += np.bincount(s1 * m + s2, minlength=m * m)  # Eq. 27
        cnt = cnt.reshape(m, m)
        S = (cnt + cnt.T).astype(np.float32)                  # Eq. 27 (symmetric)
        adjf = self.adj.astype(np.float64)
        A2 = adjf.T @ A1                                      # Eq. 26
        dW1 = self.alpha * (Q - A1[:, None] * W)              # Eq. 23
        dW2 = self.beta * (adjf.T @ Q - A2[:, None] * W)      # Eq. 24
        self.E = self.E + self.alpha * err                    # Eq. 22
        self.A1, self.A2, self.S = A1, A2, S
        return {"dW1": dW1, "dW2": dW2, "qe": float(err.sum() / max(len(X), 1))}

    # ------------------------------------------------------ Alg. 2 update network
    def _update_network(self, dW1: np.ndarray, dW2: np.ndarray) -> float:
        move = dW1 / (self.A1 + self.eps)[:, None] + dW2 / (self.A2 + self.eps)[:, None]
        self.W = self.W + move                                 # Eq. 28
        self.adj = self.S > 0                                  # Eq. 29
        self._subset(self.adj.any(axis=1))                     # Eq. 10 isolated nodes
        self.E = self.E * self.delta                           # Eq. 11
        if self.rng.random() > 1.0 - self.inactive_prob:       # 10 % chance
            self._subset(self.A1 > 0)                          # Eq. 31 inactive nodes
        return float(np.abs(move).sum())                       # "total movement"

    # --------------------------------------------------------- Alg. 3 growth
    def _grow(self) -> int:
        m, F = self.W.shape
        g = int(np.sum(self.E > np.quantile(self.E, self.growth_p)))   # Eq. 34
        g = min(g, self.M - m)
        if g <= 0:
            return 0
        cap = m + g
        W = np.empty((cap, F)); W[:m] = self.W
        E = np.zeros(cap); E[:m] = self.E
        A1 = np.ones(cap); A1[:m] = self.A1
        A2 = np.ones(cap); A2[:m] = self.A2
        adj = np.zeros((cap, cap), dtype=bool); adj[:m, :m] = self.adj
        S = np.zeros((cap, cap), dtype=np.float32); S[:m, :m] = self.S
        deg = np.zeros(cap, dtype=int); deg[:m] = self.adj.sum(axis=1)
        n = m
        for _ in range(g):
            masked = np.where(deg[:n] > 0, E[:n], -np.inf)
            q1 = int(np.argmax(masked))                        # Eq. 13
            if not np.isfinite(masked[q1]) or E[q1] <= 0.0:
                break
            nb = np.flatnonzero(adj[q1, :n])
            q2 = int(nb[np.argmax(E[nb])])                     # Eq. 14
            q3 = n
            W[q3] = 0.5 * (W[q1] + W[q2])                      # Eq. 15
            adj[q1, q2] = adj[q2, q1] = False                  # Eq. 16
            adj[q1, q3] = adj[q3, q1] = adj[q2, q3] = adj[q3, q2] = True
            deg[q3] = 2                                        # q1, q2 keep their degree
            E[q1] *= self.rho                                  # Eq. 18
            E[q2] *= self.rho
            E[q3] = 0.5 * (E[q1] + E[q2])
            S[q1, q2] = S[q2, q1] = 0.0                        # Eq. 32
            S[q1, q3] = S[q3, q1] = S[q2, q3] = S[q3, q2] = 1.0
            n += 1
        self.W, self.E, self.A1, self.A2 = W[:n], E[:n], A1[:n], A2[:n]
        self.adj, self.S = adj[:n, :n], S[:n, :n]
        return n - m

    # ------------------------------------------------------------------ epoch
    def _epoch(self, X: np.ndarray) -> None:
        res = self._batch_learning(X)                          # Alg. 1
        movement = self._update_network(res["dW1"], res["dW2"])  # Alg. 2
        added = self._grow()                                   # Alg. 3
        self.history.append({"epoch": len(self.history) + 1, "nodes": len(self.W),
                             "added": added, "movement": movement, "qe": res["qe"]})

    # -------------------------------------------------------------- Alg. 6 fit
    def fit(self, X) -> "DBLGNG":
        X = self._as_data(X)
        self.init_distributed(X)
        while len(self.W) < self.M and len(self.history) < self.max_epochs:
            self._epoch(X)
        self.reached_max_nodes = len(self.W) >= self.M
        for _ in range(self.extra_epochs):
            self._epoch(X)
        return self

    def partial_fit(self, X, n_epochs: int = 1, max_nodes: int | None = None) -> "DBLGNG":
        """Continue training the existing network on (new) data, Sec. 5.7.

        ``max_nodes`` may be raised, as the paper suggests when new structure appears.
        """
        if self.W is None:
            raise RuntimeError("call fit() or init_distributed() first")
        X = self._as_data(X)
        if max_nodes is not None:
            self.M = int(max_nodes)
        for _ in range(n_epochs):
            self._epoch(X)
        return self

    # -------------------------------------------------------- Alg. 4 edge cutting
    def graph(self, cut_p="default", inclusive: bool | None = None) -> Graph:
        """Topological map after edge cutting (Alg. 4, Eq. 33).

        Non-destructive: the training state is left untouched, so training can go on.
        """
        p = self.cut_p if cut_p == "default" else cut_p
        inclusive = self.cut_inclusive if inclusive is None else inclusive
        W, adj, S = self.W, self.adj.copy(), self.S
        index = np.arange(len(W))
        if p is not None:
            keep = self.A1 > 0                                 # Eq. 31 inactive nodes
            W, adj, S, index = W[keep], adj[np.ix_(keep, keep)], S[np.ix_(keep, keep)], index[keep]
            strengths = S[S > 0]
            if strengths.size:
                thr = np.quantile(strengths, p)                # Eq. 33
                adj &= (S >= thr) if inclusive else (S > thr)
            keep = adj.any(axis=1)                             # Eq. 10 isolated nodes
            W, adj, index = W[keep], adj[np.ix_(keep, keep)], index[keep]
        iu, ju = np.nonzero(np.triu(adj, 1))
        return Graph(nodes=W.copy(), edges=np.stack([iu, ju], axis=1), index=index)

    # ---------------------------------------------------------------- metrics
    def nearest_node(self, X, nodes: np.ndarray | None = None):
        """Index and distance of the nearest node for every datum."""
        X = self._as_data(X)
        W = self.W if nodes is None else nodes
        w2 = np.einsum("ij,ij->i", W, W)
        idx = np.empty(len(X), dtype=int)
        dist = np.empty(len(X))
        for s in range(0, len(X), self.chunk):
            x = X[s:s + self.chunk]
            d2 = np.einsum("ij,ij->i", x, x)[:, None] + w2[None, :] - 2.0 * (x @ W.T)
            i = np.argmin(d2, axis=1)
            idx[s:s + self.chunk] = i
            dist[s:s + self.chunk] = np.sqrt(np.maximum(d2[np.arange(len(x)), i], 0.0))
        return idx, dist

    def quantization_error(self, X, nodes: np.ndarray | None = None) -> float:
        """Eq. 42: mean distance of every datum to its nearest node."""
        return float(self.nearest_node(X, nodes)[1].mean())
