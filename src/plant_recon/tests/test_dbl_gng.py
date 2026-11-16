"""Unit tests for plant_recon.dbl_gng.

``literal_*`` helpers transcribe the paper's equations verbatim (identity-matrix
products of size n x m).  The production code uses bincount / one adjacency product
instead; the tests assert that both give identical numbers.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plant_recon.dbl_gng import DBLGNG, connected_components  # noqa: E402


# ---------------------------------------------------------------- literal paper
def literal_batch_learning(W, adj, X, alpha, beta):
    """Eq. 19-27 exactly as written (X: n x F, W: m x F, adj: m x m)."""
    n, _ = X.shape
    m = len(W)
    Im = np.eye(m)
    d2 = (X ** 2).sum(1, keepdims=True) - 2.0 * (X @ W.T) + (W ** 2).sum(1)   # Eq. 19
    d = np.sqrt(np.clip(d2, 0.0, None))
    s1 = np.argmin(d, axis=1)                                                  # Eq. 20
    dd = d.copy()
    dd[np.arange(n), s1] = dd.max()                                            # Eq. 21
    s2 = np.argmin(dd, axis=1)
    E_add = (Im[s1] * d).sum(0) * alpha                                        # Eq. 22
    dW1 = (Im[s1].T @ X - (W.T * Im[s1].sum(0)).T) * alpha                     # Eq. 23
    c_s1 = adj[s1].astype(float)                                               # n x m
    dW2 = (c_s1.T @ X - (W.T * c_s1.sum(0)).T) * beta                          # Eq. 24
    A1 = Im[s1].sum(0)                                                         # Eq. 25
    A2 = c_s1.sum(0)                                                           # Eq. 26
    c_hat = np.zeros((m, m))                                                   # Eq. 27
    c_hat[s1, s2] = 1
    c_hat[s2, s1] = 1
    c_tilde = Im[s1] + Im[s2]
    c_tilde = c_tilde.T @ c_tilde
    S = c_hat * c_tilde
    return E_add, dW1, dW2, A1, A2, S


def blobs(centers, n_per, sigma, rng):
    return np.vstack([c + sigma * rng.standard_normal((n_per, len(c))) for c in centers])


# ------------------------------------------------------------------------ tests
@pytest.mark.parametrize("chunk", [10_000, 37])
def test_batch_learning_equals_literal_equations(chunk):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((600, 3))
    m = 25
    W = rng.standard_normal((m, 3))
    adj = rng.random((m, m)) < 0.2
    adj = np.triu(adj, 1)
    adj = adj | adj.T
    g = DBLGNG(max_nodes=100, n_starts=2, alpha=0.5, beta=0.01, chunk=chunk)
    g.W, g.adj, g.E = W.copy(), adj.copy(), np.zeros(m)
    res = g._batch_learning(X)
    E_add, dW1, dW2, A1, A2, S = literal_batch_learning(W, adj, X, 0.5, 0.01)
    np.testing.assert_allclose(g.E, E_add, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res["dW1"], dW1, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res["dW2"], dW2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(g.A1, A1)
    np.testing.assert_allclose(g.A2, A2)
    np.testing.assert_allclose(g.S, S)
    assert np.allclose(g.S, g.S.T)               # edge strength is symmetric
    assert np.all(np.diag(g.S) == 0)


def test_distributed_initialisation_covers_all_clusters():
    rng = np.random.default_rng(3)
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 25], [25, 5]], float)
    X = blobs(centers, 400, 0.6, rng)
    for seed in range(5):
        g = DBLGNG(max_nodes=40, n_starts=6, seed=seed)
        g.init_distributed(X)
        assert g.W.shape == (12, 2)
        cl = np.argmin(((g.W[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        # every cluster receives exactly one start pair (both nodes of a pair together)
        assert sorted(set(cl)) == list(range(6))
        assert np.all(cl[0::2] == cl[1::2])
        # pair = two distinct data points joined by an edge (Eq. 40), nothing else linked
        assert g.adj.sum() == 12 and np.all(g.adj[0::2, 1::2].diagonal())
        assert not np.any(np.all(g.W[0::2] == g.W[1::2], axis=1))


def test_growth_rule_alg3():
    g = DBLGNG(max_nodes=50, rho=0.5)
    g.W = np.array([[0.0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    g.adj = np.zeros((5, 5), bool)
    for i in range(4):
        g.adj[i, i + 1] = g.adj[i + 1, i] = True
    g.E = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    g.S = (g.adj * 3).astype(np.float32)
    g.A1, g.A2 = np.ones(5), np.ones(5)
    # P_E(0.85) of [1..5] = 4.4 -> exactly one node (E=5) above it -> g = 1  (Eq. 34)
    assert g._grow() == 1
    assert len(g.W) == 6
    q1, q2, q3 = 1, 2, 5                       # q1 = argmax E, q2 = its best neighbour
    np.testing.assert_allclose(g.W[q3], [1.5, 0])                 # Eq. 15
    assert not g.adj[q1, q2] and not g.adj[q2, q1]               # Eq. 16
    assert g.adj[q1, q3] and g.adj[q3, q1] and g.adj[q2, q3] and g.adj[q3, q2]
    np.testing.assert_allclose(g.E[[q1, q2, q3]], [2.5, 1.0, 1.75])   # Eq. 18
    assert g.S[q1, q2] == 0 and g.S[q1, q3] == 1 and g.S[q3, q2] == 1  # Eq. 32
    assert g.A1[q3] == 1


def literal_growth(W, adj, E, S, A1, A2, M, p, rho):
    """Alg. 3 / Eq. 13-18, 32, 34 transcribed literally: one node appended at a time."""
    W, adj, E, S = W.copy(), adj.copy(), E.copy(), S.copy()
    A1, A2 = A1.copy(), A2.copy()
    g = int(np.sum(E > np.quantile(E, p)))                         # Eq. 34
    for _ in range(g):
        m = len(W)
        if not m < M:
            break
        q1 = int(np.argmax(E))                                     # Eq. 13
        nbrs = np.flatnonzero(adj[q1])
        q2 = int(nbrs[np.argmax(E[nbrs])])                         # Eq. 14
        W = np.vstack([W, 0.5 * (W[q1] + W[q2])])                  # Eq. 15
        E = np.append(E, 0.0)
        A1, A2 = np.append(A1, 1.0), np.append(A2, 1.0)
        adj = np.pad(adj, ((0, 1), (0, 1)))
        S = np.pad(S, ((0, 1), (0, 1)))
        q3 = m
        adj[q1, q2] = adj[q2, q1] = False                          # Eq. 16
        for a, b in ((q1, q3), (q2, q3)):
            adj[a, b] = adj[b, a] = True
            S[a, b] = S[b, a] = 1.0                                # Eq. 32
        S[q1, q2] = S[q2, q1] = 0.0
        E[q1] *= rho                                               # Eq. 18
        E[q2] *= rho
        E[q3] = 0.5 * (E[q1] + E[q2])
    return W, adj, E, S, A1, A2


@pytest.mark.parametrize("seed", range(6))
def test_preallocated_growth_equals_literal_alg3(seed):
    rng = np.random.default_rng(seed)
    m = 40
    W = rng.standard_normal((m, 3))
    adj = np.triu(rng.random((m, m)) < 0.15, 1)
    adj = adj | adj.T
    adj |= np.eye(m, k=1, dtype=bool) | np.eye(m, k=-1, dtype=bool)    # every node has a neighbour
    E = rng.random(m) * 10
    S = (adj * rng.integers(1, 9, (m, m))).astype(np.float32)
    S = np.maximum(S, S.T)
    A1, A2 = rng.integers(1, 5, m).astype(float), rng.integers(1, 5, m).astype(float)
    M = m + 4 if seed % 2 else 500           # exercise the "stop at M" branch as well
    ref = literal_growth(W, adj, E, S, A1, A2, M, 0.85, 0.5)
    g = DBLGNG(max_nodes=M, rho=0.5, growth_p=0.85)
    g.W, g.adj, g.E, g.S, g.A1, g.A2 = W.copy(), adj.copy(), E.copy(), S.copy(), A1.copy(), A2.copy()
    g._grow()
    for got, want in zip((g.W, g.adj, g.E, g.S, g.A1, g.A2), ref):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_growth_never_exceeds_max_nodes():
    g = DBLGNG(max_nodes=7)
    g.W = np.random.default_rng(0).standard_normal((5, 2))
    g.adj = np.ones((5, 5), bool) ^ np.eye(5, dtype=bool)
    g.E = np.arange(1.0, 6.0)
    g.S = g.adj.astype(np.float32)
    g.A1 = g.A2 = np.ones(5)
    g.growth_p = 0.0                       # would ask for 4 nodes
    assert g._grow() == 2 and len(g.W) == 7


def test_update_network_alg2():
    g = DBLGNG(max_nodes=50, delta=0.5, inactive_prob=0.0)
    g.W = np.array([[0.0, 0], [1, 0], [5, 5], [9, 9]])
    g.E = np.array([2.0, 4.0, 6.0, 8.0])
    g.A1 = np.array([3.0, 1.0, 0.0, 0.0])
    g.A2 = np.array([1.0, 3.0, 0.0, 0.0])
    g.adj = np.ones((4, 4), bool)                  # old edges are irrelevant (Eq. 29)
    g.S = np.zeros((4, 4), np.float32)
    g.S[0, 1] = g.S[1, 0] = 2.0                    # only 0-1 got strength; 2 and 3 isolated
    dW1 = np.array([[3.0, 0], [0, 0], [0, 0], [0, 0]])
    dW2 = np.array([[0.0, 0], [-3, 0], [0, 0], [0, 0]])
    g._update_network(dW1, dW2)
    # isolated nodes 2 and 3 removed (Eq. 10), edges rebuilt from S (Eq. 29)
    assert len(g.W) == 2 and g.adj.tolist() == [[False, True], [True, False]]
    np.testing.assert_allclose(g.W[0], [3.0 / (3 + 1e-4), 0])     # Eq. 28
    np.testing.assert_allclose(g.W[1], [1 - 3.0 / (3 + 1e-4), 0])
    np.testing.assert_allclose(g.E, [1.0, 2.0])                    # Eq. 11, delta = 0.5


def test_inactive_nodes_removed_with_probability():
    def run(prob):
        g = DBLGNG(max_nodes=50, inactive_prob=prob, seed=0)
        g.W = np.array([[0.0, 0], [1, 0], [2, 0]])
        g.E = np.ones(3)
        g.A1 = np.array([5.0, 0.0, 5.0])           # node 1 is only ever a second winner
        g.A2 = np.zeros(3)
        g.S = np.zeros((3, 3), np.float32)
        g.S[0, 1] = g.S[1, 0] = g.S[1, 2] = g.S[2, 1] = 1.0
        g.adj = g.S > 0
        g._update_network(np.zeros((3, 2)), np.zeros((3, 2)))
        return len(g.W)
    assert run(0.0) == 3 and run(1.0) == 2


@pytest.mark.parametrize("inclusive,expected", [(False, {(0, 1)}), (True, {(0, 1), (2, 3), (1, 2)})])
def test_edge_cutting_eq33(inclusive, expected):
    g = DBLGNG(max_nodes=50, cut_p=0.5)
    g.W = np.arange(8, dtype=float).reshape(4, 2)
    g.A1 = np.ones(4)
    g.S = np.zeros((4, 4), np.float32)
    for (i, j), v in {(0, 1): 9, (1, 2): 2, (2, 3): 2, (0, 3): 1}.items():
        g.S[i, j] = g.S[j, i] = v
    g.adj = g.S > 0
    # positive strengths [1, 1, 2, 2, 9, 9] (each edge twice) -> P(0.5) = 2
    edges = {tuple(e) for e in g.graph(inclusive=inclusive).edges.tolist()}
    assert edges == expected


def test_cut_removes_isolated_and_inactive_nodes():
    g = DBLGNG(max_nodes=50, cut_p=0.0)
    g.W = np.arange(10, dtype=float).reshape(5, 2)
    g.A1 = np.array([1.0, 1, 1, 0, 1])
    g.S = np.zeros((5, 5), np.float32)
    g.S[0, 1] = g.S[1, 0] = 5
    g.S[1, 2] = g.S[2, 1] = 1
    g.S[3, 4] = g.S[4, 3] = 5               # node 3 inactive -> its only edge disappears
    g.adj = g.S > 0
    gr = g.graph(inclusive=False)           # p = 0 -> strict ">" cuts the weakest (1-2)
    assert gr.index.tolist() == [0, 1]      # 2 and 4 become isolated, 3 inactive
    assert gr.edges.tolist() == [[0, 1]]


def test_fit_runs_alg6_and_is_deterministic():
    rng = np.random.default_rng(5)
    centers = np.array([[0, 0, 0], [6, 0, 0], [0, 6, 0], [3, 3, 6]], float)
    X = blobs(centers, 500, 0.5, rng)
    a = DBLGNG(max_nodes=60, n_starts=4, seed=7).fit(X)
    b = DBLGNG(max_nodes=60, n_starts=4, seed=7).fit(X)
    assert a.reached_max_nodes
    np.testing.assert_array_equal(a.W, b.W)
    assert np.isfinite(a.W).all() and len(a.W) <= 60
    # the 10 extra epochs of Sec. 4.3 were run after the maximum was reached
    n_reach = next(h["epoch"] for h in a.history if h["nodes"] >= 60)
    assert len(a.history) == n_reach + a.extra_epochs
    assert a.history[-1]["qe"] < a.history[0]["qe"]
    # every cluster is represented by nodes after training
    cl = np.argmin(((a.W[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    assert set(cl) == {0, 1, 2, 3}
    # learning stability (paper Sec. 5.5, Fig. 17-19): because the delta is averaged
    # over the whole batch, the node movement decays towards zero with more epochs
    # (the 10 extra epochs alone are not enough for that, 60 more are)
    a.partial_fit(X, n_epochs=60)
    mv = [h["movement"] for h in a.history]
    assert mv[-1] < 1e-3 * max(mv)
    qe = [h["qe"] for h in a.history]
    assert qe[-1] <= qe[n_reach + a.extra_epochs - 1] + 1e-9


def test_partial_fit_adapts_to_new_data():
    rng = np.random.default_rng(9)
    X1 = blobs([[0, 0]], 800, 0.5, rng)
    X2 = blobs([[0, 0], [20, 20]], 800, 0.5, rng)
    g = DBLGNG(max_nodes=30, n_starts=2, seed=1).fit(X1)
    q_before = g.quantization_error(X2)
    g.partial_fit(X2, n_epochs=25, max_nodes=60)
    assert g.quantization_error(X2) < 0.2 * q_before
    assert g.W[:, 0].max() > 15


def test_connected_components():
    labels = connected_components(6, np.array([[0, 1], [1, 2], [4, 5]]))
    assert labels[0] == labels[1] == labels[2] and labels[4] == labels[5]
    assert len(set(labels)) == 3
