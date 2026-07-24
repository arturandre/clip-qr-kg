#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focused experiments for Graph Memory (GM) — binary scenarios only.

Datasets:
    • make_moons
    • make_circles

Baselines:
    • Logistic Regression (Linear)
    • kNN
    • Label Spreading (sklearn)
    • LP(harmonic) — closed-form harmonic Label Propagation

Graph Memory (GM) variants:
    • GM (reliability-weighted diffusion over prototype graph)
    • GM-flat (same as GM but reliability r = 1)
    • GM-instance (prototypes = instances)
    • GM-centroid (one prototype per class, no edges)
    • GM→LP(instance) — degenerate to instance-graph harmonic LP
    • (Optional) kNN-budget — kNN trained on |prototypes| samples

Key CLI flags:
    --dataset {moons,circles}
    --imbalance R     (R>1 applies a long-tail ratio to TRAIN only)
    --budget-match    (add kNN with same memory budget as GM)
"""

# ================================================================
# Imports and setup
# ================================================================

import os, csv, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from contextlib import contextmanager

# Progress & rich output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as rich_traceback_install
rich_traceback_install(show_locals=False)
console = Console()

# sklearn
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph, NearestNeighbors
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import KMeans  , MiniBatchKMeans
from sklearn.metrics import accuracy_score

# ================================================================
# Utilities
# ================================================================

import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_transition_matrix(
    P,                      # [K, d] prototype coords (float32/64)
    yc,                     # [K] dominant class per prototype (int) or None
    r=None,                 # [K] reliability in [0,1] or None
    pi=None,                # [K] purity in [0,1] or None (needed for purity-aware damping)
    k_graph=10,
    use_mutual=True,        # keep only reciprocal edges
    use_local_scale=True,   # self-tuned kernel; else global median scale
    dist_cap_quantile=0.95, # prune very long edges (None to disable)
    cross_class_factor=None,# if set in (0,1), uniform damping for cross-class edges;
                            # if None and (yc,pi) provided, use purity-aware damping
    reliability_gamma=1.0,  # 0 disables reliability gating
    eps=1e-12
):
    P = np.asarray(P)
    K = int(P.shape[0])
    if K <= 1:
        return np.eye(K, dtype=P.dtype)

    k_eff = int(np.clip(k_graph, 1, max(1, K-1)))

    # --- kNN for adjacency pattern & local scales ---
    nn = NearestNeighbors(n_neighbors=min(k_eff+1, K), algorithm="auto").fit(P)
    dists, idxs = nn.kneighbors(P, return_distance=True)  # each row: [self, k neighbors]

    # local scales sigma_c: distance to its k-th neighbor (excl self)
    if use_local_scale:
        # if k_eff >= number of available neighbors, take the farthest available
        kth = np.minimum(k_eff, dists.shape[1]-1)
        sigma = dists[np.arange(K), kth].clip(min=eps)  # [K]
    else:
        sigma = np.full(K, np.median(dists[:, 1:].ravel()) + eps, dtype=P.dtype)

    # --- weighted adjacency (Zelnik-Manor self-tuning kernel) ---
    A = np.zeros((K, K), dtype=P.dtype)
    for c in range(K):
        for j in range(1, dists.shape[1]):  # skip self at position 0
            c2 = int(idxs[c, j])
            d = float(dists[c, j])
            # exp( - d^2 / (sigma_c * sigma_c2) )
            w = np.exp(-(d*d) / (float(sigma[c]) * float(sigma[c2]) + eps))
            if w > 0.0:
                # keep the strongest weight if duplicate via different paths
                if w > A[c, c2]:
                    A[c, c2] = w

    # --- mutual kNN (optional) ---
    if use_mutual:
        A = np.minimum(A, A.T)

    # --- class-aware damping ---
    if yc is not None:
        yc = np.asarray(yc)
        cross = (yc[:, None] != yc[None, :])
        if cross_class_factor is not None and 0.0 < cross_class_factor < 1.0:
            A[cross] *= float(cross_class_factor)
        elif pi is not None:
            # purity-aware: keep boundary edges where at least one endpoint is ambiguous
            pi = np.asarray(pi)
            eta = np.minimum(pi[:, None], pi[None, :])  # low when one endpoint ambiguous
            # scale cross-class edges by (1 - eta): 1 for highly ambiguous, ~0 for very pure
            A[cross] *= (1.0 - eta[cross])

    # --- reliability gating (optional) ---
    if r is not None and reliability_gamma > 0.0:
        r = np.asarray(r, dtype=P.dtype).clip(min=0.0)
        R = np.power(r + eps, reliability_gamma)[:, None]
        A *= (R @ R.T)

    # --- distance cap (optional) ---
    if dist_cap_quantile is not None:
        # compute pairwise distances only where A>0
        rows, cols = np.where(A > 0)
        if rows.size > 0:
            d_eff = np.linalg.norm(P[rows] - P[cols], axis=1)
            cap = np.quantile(d_eff, dist_cap_quantile)
            mask_long = (d_eff > cap)
            A[rows[mask_long], cols[mask_long]] = 0.0

    # remove self-loops if any and symmetrize for stability
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, A.T)

    # --- ensure every row has at least one outgoing edge (avoid NaNs) ---
    row_sum = A.sum(axis=1, keepdims=True)
    dead = (row_sum.squeeze(-1) <= eps)
    if np.any(dead):
        # connect dead node to its single nearest neighbor
        nn1 = NearestNeighbors(n_neighbors=min(2, K), algorithm="auto").fit(P)
        d1, i1 = nn1.kneighbors(P[dead], return_distance=True)
        # i1 includes self at position 0; take the next if available
        for idx_dead, neighbors in zip(np.where(dead)[0], i1):
            tgt = neighbors[1] if len(neighbors) > 1 else neighbors[0]
            if idx_dead != tgt:
                A[idx_dead, tgt] = 1.0
                A[tgt, idx_dead] = max(A[tgt, idx_dead], 1.0)

    # --- row-normalize to stochastic S ---
    row_sum = A.sum(axis=1, keepdims=True) + eps
    S = (A / row_sum).astype(P.dtype)
    return S



def set_seed(seed: int):
    np.random.seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def l2n(X):
    """Row-wise L2 normalize."""
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def robust_sigmoid_normalize(values):
    """Robust centering by IQR, then sigmoid. Stable under outliers."""
    x = np.asarray(values, float)
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = max(q75 - q25, 1e-12)
    z = (x - med) / iqr
    return sigmoid(z)

import math

EPS = 1e-12

def onehot(y, n_classes=2):
    Y = np.zeros((len(y), n_classes), float)
    Y[np.arange(len(y)), y] = 1.0
    return Y

def nll(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())

def brier(probs, y):
    Y = onehot(y, probs.shape[1])
    return float(((probs - Y)**2).sum(axis=1).mean())

def ece(probs, y, n_bins=15):
    """Binary ECE with max prob as confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf >= lo) & (conf < hi) if i < n_bins-1 else (conf >= lo) & (conf <= hi)
        if not np.any(m): continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.mean()) * abs(acc_bin - conf_bin)
    return float(ece)

def softmax_log_scaled(probs, T=1.0):
    """Apply temperature scaling in log-prob space: softmax(log p / T)."""
    L = np.log(np.clip(probs, EPS, 1.0))
    L = L / max(T, 1e-6)
    # numeric-stable softmax
    Lm = L.max(axis=1, keepdims=True)
    expL = np.exp(L - Lm)
    return expL / (expL.sum(axis=1, keepdims=True) + EPS)

def fit_temperature(probs_val, y_val, T_init=1.0):
    """1D search over T to minimize NLL on calibration set."""
    # search over logT for stability
    grid = np.linspace(-2.0, 2.0, 41)  # T in ~[0.135, 7.39]
    best_T, best_loss = T_init, math.inf
    for logT in grid:
        T = float(np.exp(logT))
        q = softmax_log_scaled(probs_val, T)
        loss = nll(q, y_val)
        if loss < best_loss:
            best_loss, best_T = loss, T
    # local refine around best
    for step in [0.5, 0.25, 0.1]:
        c = math.log(best_T)
        cand = [c-2*step, c-step, c, c+step, c+2*step]
        for logT in cand:
            T = float(np.exp(logT))
            q = softmax_log_scaled(probs_val, T)
            loss = nll(q, y_val)
            if loss < best_loss:
                best_loss, best_T = loss, T
    return best_T, best_loss


@contextmanager
def section(msg: str):
    console.rule(f"[bold cyan]{msg}")
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        console.print(Panel.fit(f"[green]{msg}[/green] finished in [bold]{dt:.3f}s[/bold]"))

# ================================================================
# Data (binary, non-linear)
# ================================================================

def gen_moons(n_samples=2000, seed=0, noise=0.15):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X, y

def gen_circles(n_samples=2000, seed=0, factor=0.4, noise=0.0):
    X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=seed)
    return X, y

def apply_long_tail(Xtr, ytr, imbalance, seed):
    """Apply simple long-tail ratio to TRAIN (binary)."""
    if imbalance <= 1.0:
        return Xtr, ytr
    rng = np.random.RandomState(seed + 10)
    classes = np.unique(ytr)
    counts = np.bincount(ytr)
    maxc = counts.max()
    # geometric decay targets across sorted classes (0..C-1)
    target = [int(round(maxc / (imbalance ** (i / max(1, len(classes)-1))))) for i in range(len(classes))]
    keep = []
    for c, tgt in zip(classes, target):
        idx = np.where(ytr == c)[0]
        k = min(len(idx), max(5, tgt))
        keep.append(rng.choice(idx, size=k, replace=False))
    keep = np.concatenate(keep)
    return Xtr[keep], ytr[keep]

# ================================================================
# Reliability metrics (for GM)
# ================================================================

def prototype_purity(assign, y, K, n_classes):
    yc, pi = np.zeros(K, int), np.zeros(K, float)
    for c in range(K):
        idx = np.where(assign == c)[0]
        if len(idx) == 0:     # empty cluster
            continue
        hist = np.bincount(y[idx], minlength=n_classes)
        yc[c] = np.argmax(hist)
        pi[c] = hist.max() / len(idx)
    return yc, pi

def silhouette_normalized(X, assign, K):
    """Slow but transparent cluster-wise silhouette in [0,1]."""
    s = np.zeros(K)
    for c in range(K):
        idx = np.where(assign == c)[0]
        if len(idx) <= 1:
            s[c] = 0.5
            continue
        Xi = X[idx]
        d_within = np.sqrt(((Xi[:, None, :] - Xi[None, :, :])**2).sum(-1))
        a_i = d_within.mean(axis=1)
        # nearest other cluster mean distance
        b_i = np.inf * np.ones_like(a_i)
        for c2 in range(K):
            if c2 == c: continue
            idx2 = np.where(assign == c2)[0]
            if len(idx2) == 0: continue
            Xj = X[idx2]
            d_cross = np.sqrt(((Xi[:, None, :] - Xj[None, :, :])**2).sum(-1))
            b_i = np.minimum(b_i, d_cross.mean(axis=1))
        sil = (b_i - a_i) / np.maximum(a_i, b_i)
        s[c] = np.clip(((sil + 1) * 0.5).mean(), 0, 1)
    return s

def dispersion(X, assign, P, K):
    v = np.zeros(K)
    for c in range(K):
        idx = np.where(assign == c)[0]
        if len(idx) == 0: continue
        Xi = X[idx]
        v[c] = ((Xi - P[c])**2).sum(axis=1).mean()
    return v

def margins(P, yc):
    """Distance to nearest prototype of a different class."""
    K = len(P)
    m = np.zeros(K)
    for c in range(K):
        mask = (yc != yc[c])
        if not np.any(mask): continue
        d = np.linalg.norm(P[mask] - P[c], axis=1)
        m[c] = d.min()
    return m

def instability(X, assign, P, sigma_noise=0.05, trials=3):
    """Fraction of assigned samples that switch prototype under tiny noise."""
    K = len(P)
    nn = NearestNeighbors(n_neighbors=1).fit(P)
    changed, total = np.zeros(K), np.zeros(K)
    for _ in range(trials):
        Xd = X + np.random.normal(scale=sigma_noise, size=X.shape)
        nn2 = nn.kneighbors(Xd, return_distance=False).ravel()
        for i, (c0, c1) in enumerate(zip(assign, nn2)):
            total[c0] += 1
            if c1 != c0:
                changed[c0] += 1
    rho = np.divide(changed, total, out=np.zeros_like(changed), where=total>0)
    return rho

def reliability_full(s, m, v, rho, pi, lam=(1,1,1,1,1)):
    l1,l2,l3,l4,l5 = lam
    m_bar = robust_sigmoid_normalize(m)
    v_bar = robust_sigmoid_normalize(v)
    z = l1*s + l2*m_bar - l3*rho - l4*v_bar + l5*pi
    return sigmoid(z)

# ================================================================
# Graph Memory (GM)
# ================================================================

from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

# ---------- Fast helpers (subsampled reliability) ----------

def _cluster_subsample_indices(assign: np.ndarray, cap: int, seed: int, K: int):
    """
    Return a list of length K; each entry is a 1D np.ndarray of indices (possibly empty).
    """
    rng = np.random.default_rng(seed)
    subs = []
    for c in range(K):
        I = np.where(assign == c)[0]
        if I.size == 0:
            subs.append(np.empty(0, dtype=np.int64))
            continue
        if cap is None or I.size <= cap:
            subs.append(I.astype(np.int64, copy=False))
        else:
            subs.append(rng.choice(I, size=cap, replace=False).astype(np.int64))
    return subs


def _silhouette_approx(X: np.ndarray, assign: np.ndarray, subs, K: int) -> np.ndarray:
    """
    Robust, bounded-cost silhouette proxy per cluster.
    - X: (N, d) float32/float64
    - assign: (N,) ints
    - subs: list of length K; entries are 1D arrays of indices (possibly empty)
    Returns s_c in [0,1], with guards for small/empty clusters.
    """
    s_c = np.zeros(K, dtype=np.float32)

    # ensure each entry is an ndarray (so len() works and downstream ops are consistent)
    subs = [np.asarray(idx, dtype=np.int64) if idx is not None else np.empty(0, dtype=np.int64) for idx in subs]

    for c in range(K):
        I = subs[c]
        n_i = len(I)
        if n_i <= 1:
            # undefined or singleton: neutral silhouette (0.5 in [0,1] scale)
            s_c[c] = 0.5
            continue

        Xi = X[I].astype(np.float32, copy=False)
        # intra-cluster distances (avoid self-dist; subtract 1 in mean denom)
        d_within = np.sqrt(((Xi[:, None, :] - Xi[None, :, :]) ** 2).sum(-1) + 1e-8)
        # set diagonal to 0 explicitly; mean over (n_i-1) neighbors
        np.fill_diagonal(d_within, 0.0)
        a_i = d_within.sum(axis=1) / max(1, n_i - 1)

        # nearest-other-cluster average distance for each point in cluster c
        b_i = np.full(n_i, np.inf, dtype=np.float32)
        for c2 in range(K):
            if c2 == c:
                continue
            J = subs[c2]
            if len(J) == 0:
                continue
            Xj = X[J].astype(np.float32, copy=False)
            d_cross = np.sqrt(((Xi[:, None, :] - Xj[None, :, :]) ** 2).sum(-1) + 1e-8)
            # mean distance from each Xi to cluster c2
            b_i = np.minimum(b_i, d_cross.mean(axis=1))

        # if still inf (no other non-empty clusters), fall back to neutral
        if not np.isfinite(b_i).any():
            s_c[c] = 0.5
            continue

        sil_i = (b_i - a_i) / (np.maximum(a_i, b_i) + 1e-12)
        # map from [-1,1] to [0,1]
        s_c[c] = float(np.clip(((sil_i + 1.0) * 0.5).mean(), 0.0, 1.0))

    return s_c


def _dispersion_sub(X: np.ndarray, assign: np.ndarray, P: np.ndarray, subs, K: int) -> np.ndarray:
    """Mean squared distance to prototype, computed on subsamples only."""
    v = np.zeros(K, dtype=np.float32)
    for c in range(K):
        I = subs[c]
        if I.size == 0: 
            v[c] = 0.0
            continue
        Xi = X[I]
        v[c] = float(((Xi - P[c][None, :]) ** 2).sum(axis=1).mean())
    return v.astype(np.float32)

# ---------- GM ----------

@dataclass
class GMConfig:
    K: int
    knn_graph: int = 10
    # more stable near overlaps
    # attaching each query to its attach_k nearest prototypes
    attach_k: int = 8 
    alpha: float = 0.1 # gentler diffusion by default
    beta: float = 1.0
    use_norm: bool = False
    lam1: float = 1.0
    lam2: float = 1.0
    lam3: float = 1.0
    lam4: float = 1.0
    lam5: float = 1.0
    instab_sigma: float = 0.05
    instab_trials: int = 1

    # Speed knobs
    use_minibatch_kmeans: bool = True
    kmeans_batch_size: int = 4096
    reliability_sample_cap: int = 512  # subsample per cluster for reliability
    disable_instability: bool = True # default off for speed
    diffusion_iters: int = 20  # 0 = closed form inverse, >0 = run T iterations (no inverse)
    dtype: str = "float32"

    # make reliability act in absolute (not canceled by normalization)
    absolute_reliability: bool = True

    # encourage local class coverage at attachment (helps boundary calibration)
    enforce_opposite_seed: bool = True
    coverage_search_factor: int = 2   # search up to 2*attach_k for opposite proto

class GraphMemory:
    """Prototype graph with reliability-aware seeding and diffusion."""
    def __init__(self, cfg: GMConfig):
        self.cfg = cfg
        self.P = None      # [K, d] (possibly normalized if use_norm)
        self.yc = None     # [K] prototype dominant class
        self.pi = None     # [K] purity
        self.r  = None     # [K] reliability per prototype
        self.S  = None     # [K, K] row-stochastic transition among prototypes

    def fit(self, X, y):
        # ----- types -----
        X = np.asarray(X, dtype=np.float32 if self.cfg.dtype == "float32" else np.float64)
        y = np.asarray(y, dtype=np.int32)
        K = int(min(self.cfg.K, len(X)))
        k_graph = (self.cfg.knn_graph if isinstance(self.cfg.knn_graph, int)
                else int(np.clip(np.ceil(3*np.log2(K)), 6, 20)))


        # ----- k-means (mini-batch by default) -----
        if self.cfg.use_minibatch_kmeans:
            km = MiniBatchKMeans(
                n_clusters=K, batch_size=self.cfg.kmeans_batch_size,
                n_init="auto", random_state=0, reassignment_ratio=0.01
            ).fit(X)
        else:
            km = KMeans(n_clusters=K, n_init="auto", random_state=0).fit(X)
        P = km.cluster_centers_.astype(np.float32)
        assign = km.labels_.astype(np.int32)

        # ----- labels & purity -----
        nC = int(y.max()) + 1
        yc, pi = prototype_purity(assign, y, K, nC)  # (K,), (K,)
        self.pi = pi.astype(np.float32)

        # ----- reliability (fast: subsampled) -----
        subs = _cluster_subsample_indices(assign, cap=self.cfg.reliability_sample_cap, seed=0, K=K)
        s = _silhouette_approx(X, assign, subs, K)
        v = _dispersion_sub(X, assign, P, subs, K)                 # (K,)
        m = margins(P, yc).astype(np.float32)                      # (K,)

        if self.cfg.disable_instability:
            rho = np.zeros(K, dtype=np.float32)
        else:
            # Instability on subsamples union with 1 trial (cheap). Reuse your 'instability' if present.
            I_all = np.concatenate([I for I in subs if I.size > 0]) if any(len(I) > 0 for I in subs) else np.arange(min(len(X), K))
            rho = instability(
                X[I_all], assign[I_all], P,
                sigma_noise=self.cfg.instab_sigma,
                trials=self.cfg.instab_trials
            ).astype(np.float32)

        r = reliability_full(s, m, v, rho, pi,
                             lam=(self.cfg.lam1, self.cfg.lam2, self.cfg.lam3, self.cfg.lam4, self.cfg.lam5)
                            ).astype(np.float32)

        # --- store normalized or raw prototypes as configured ---
        Pn = l2n(P) if self.cfg.use_norm else P
        self.P  = Pn.astype(np.float32)
        self.yc = yc.astype(np.int32)
        self.r  = r.astype(np.float32)


        # --- prototype graph (mutual, locally scaled, purity-aware damping) ---
        self.S = build_transition_matrix(
            P=self.P,
            yc=self.yc,
            r=self.r,                 # or None to ignore reliability
            pi=self.pi,                 # make purity available for damping
            k_graph=k_graph,
            use_mutual=True,
            use_local_scale=True,
            dist_cap_quantile=0.95,
            cross_class_factor=None,   # purity-aware cross-class damping (keeps boundary edges where needed)
            reliability_gamma=1.0
        )

        return self

    def _z0_matrix(self, Xq: np.ndarray) -> np.ndarray:
        """
        Seeding: reliability-weighted RBF to the top-k nearest prototypes.
        If absolute_reliability=True, DO NOT normalize the weights — keep
        the absolute magnitude so low reliability lowers total mass.
        If enforce_opposite_seed=True and all-k neighbors are same class,
        replace the farthest same-class neighbor by the nearest opposite-class
        within a wider search window (up to coverage_search_factor * k).
        """
        Xq = np.asarray(Xq, dtype=np.float32 if self.cfg.dtype == "float32" else np.float64)
        K = len(self.P); Q = len(Xq)
        k = min(self.cfg.attach_k, K)

        # nearest for initial pool
        nn_k = NearestNeighbors(n_neighbors=k).fit(self.P)
        d, idx = nn_k.kneighbors(Xq, return_distance=True)  # (Q,k)

        # wider search (for coverage) if needed
        if self.cfg.enforce_opposite_seed and K > k:
            k_wide = min(self.cfg.coverage_search_factor * k, K)
            nn_w = NearestNeighbors(n_neighbors=k_wide).fit(self.P)
            d_w, idx_w = nn_w.kneighbors(Xq, return_distance=True)
        else:
            d_w, idx_w = d, idx

        Z0 = np.zeros((K, Q), dtype=np.float32)
        for i in range(Q):
            ii = idx[i].copy()
            di = d[i].copy()
            labs = self.yc[ii]

            # ensure at least two classes locally (if available)
            if self.cfg.enforce_opposite_seed and np.all(labs == labs[0]):
                cand_w = idx_w[i]
                cand_w_d = d_w[i]
                opp = cand_w[self.yc[cand_w] != labs[0]]
                if opp.size > 0:
                    # inject nearest opposite, replace farthest same-class
                    far_pos = np.argmax(di)
                    rep = opp[0]
                    # replace
                    ii[far_pos] = rep
                    di[far_pos] = cand_w_d[np.where(cand_w == rep)[0][0]]
                    labs = self.yc[ii]

            w = np.exp(-self.cfg.beta * (di ** 2)).astype(np.float32) * (self.r[ii] + 1e-8)

            if self.cfg.absolute_reliability:
                # keep absolute magnitude (do not normalize)
                Z0[ii, i] = w
            else:
                # normalize to 1 (classic behavior)
                wn = w / (w.sum() + 1e-8)
                Z0[ii, i] = wn

        return Z0  # (K, Q)
    
    def _diffuse(self, Z0: np.ndarray) -> np.ndarray:
        """Apply either iterative diffusion or closed-form."""
        a = float(self.cfg.alpha)
        K, Q = Z0.shape
        if self.cfg.diffusion_iters and self.cfg.diffusion_iters > 0:
            Z = Z0.copy()
            T = int(self.cfg.diffusion_iters)
            for _ in range(T):
                Z = a * (self.S @ Z) + (1.0 - a) * Z0
            return Z
        else:
            I = np.eye(K, dtype=np.float32)
            M = np.linalg.inv(I - a * self.S).astype(np.float32)
            return (1.0 - a) * (M @ Z0)

    def predict_proba(self, Xq, n_classes: int) -> np.ndarray:
        """
        Returns *normalized* class probabilities (sum to 1).
        For a separate confidence scalar that reflects absolute reliability mass,
        use predict_proba_with_confidence.
        """
        Z0 = self._z0_matrix(Xq)              # (K, Q)
        Z  = self._diffuse(Z0)                # (K, Q)

        Q = Z.shape[1]
        probs = np.zeros((Q, n_classes), dtype=np.float32)
        for c in range(n_classes):
            probs[:, c] = Z[self.yc == c, :].sum(axis=0)

        # normalize across classes (decision probabilities)
        s = probs.sum(axis=1, keepdims=True)
        zero_mask = (s.squeeze() == 0)  # samples with no prototype affinity
        if np.any(zero_mask):
            # set uniform probabilities for those queries
            probs[zero_mask, :] = 1.0 / n_classes
            s[zero_mask, :] = 1.0

        probs /= s
        return probs.astype(np.float32)
    
    def predict_proba_with_confidence(self, Xq, n_classes: int):
        """
        Returns (probs, conf), where probs are normalized class probabilities,
        and conf is a scalar per query capturing the *absolute* mass if
        absolute_reliability=True. This preserves the semantics that low
        reliability lowers confidence even without opposite-class seeds.
        """
        Z0 = self._z0_matrix(Xq)              # (K, Q)
        Z  = self._diffuse(Z0)                # (K, Q)

        Q = Z.shape[1]
        probs = np.zeros((Q, n_classes), dtype=np.float32)
        for c in range(n_classes):
            probs[:, c] = Z[self.yc == c, :].sum(axis=0)

        # confidence proxy = total diffused mass per query (before class-norm)
        # If absolute_reliability=True, this varies; otherwise ~constant.
        conf = probs.sum(axis=1).astype(np.float32)  # shape: (Q,)

        # normalize to get final probabilities for decisions
        s = conf[:, None] + 1e-8
        probs = (probs / s).astype(np.float32)
        return probs, conf.astype(np.float32)

    def predict(self, Xq, n_classes: int) -> np.ndarray:
        return self.predict_proba(Xq, n_classes).argmax(axis=1)

# ================================================================
# GM degenerates
# ================================================================

def gm_flat_from(gm: GraphMemory) -> GraphMemory:
    """Set all reliabilities to 1, keep same graph/prototypes."""
    flat = GraphMemory(gm.cfg)
    flat.P, flat.yc, flat.S = gm.P.copy(), gm.yc.copy(), gm.S.copy()
    flat.r = np.ones_like(gm.yc, dtype=float)
    return flat

def gm_instance(Xtr, ytr, knn_graph=10, alpha=0.6, beta=1.0, attach_k=3, diffusion_iters=20):
    """Prototypes = instances; graph built over all instances."""
    gm = GraphMemory(GMConfig(K=len(Xtr), knn_graph=knn_graph, alpha=alpha, beta=beta, attach_k=attach_k, diffusion_iters=diffusion_iters))
    gm.P, gm.yc = Xtr.copy(), ytr.copy()
    gm.r = np.ones(len(Xtr))
    A = kneighbors_graph(Xtr, n_neighbors=min(knn_graph, len(Xtr)-1),
                         mode="distance", include_self=False).toarray()
    A = np.exp(-beta * (A**2))
    A = np.maximum(A, A.T)
    D = A.sum(axis=1, keepdims=True) + 1e-12
    gm.S = A / D
    return gm

def gm_centroid(Xtr, ytr):
    """One prototype per class; no edges (pure nearest-centroid)."""
    classes = np.unique(ytr)
    means = np.vstack([Xtr[ytr==c].mean(0) for c in classes])
    gm = GraphMemory(GMConfig(K=len(means), knn_graph=1, alpha=0.0, attach_k=1))
    gm.P, gm.yc = means, classes
    gm.r = np.ones(len(means))
    gm.S = np.zeros((len(means), len(means)))
    return gm

# ================================================================
# Harmonic Label Propagation (true LP baseline)
# ================================================================

from sklearn.semi_supervised import LabelSpreading, LabelPropagation

def sklearn_ssl_predict(XL, yL, XU, method="spreading", n_neighbors=10, alpha=0.2, gamma=None):
    """
    Run scikit-learn SSL transductively over [XL; XU].
    method: "spreading" (recommended) or "propagation"
    Returns predicted labels for XU.
    """
    X_all = np.r_[XL, XU]
    y_all = np.r_[yL, -np.ones(len(XU), dtype=int)]
    if method == "spreading":
        model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=30)
    elif method == "propagation":
        model = LabelPropagation(kernel="knn", n_neighbors=n_neighbors, max_iter=1000)
    else:
        raise ValueError("method must be 'spreading' or 'propagation'")
    model.fit(X_all, y_all)
    return model.transduction_[-len(XU):]

def gm_to_sklearn_ssl(Xtr, ytr, Xte, method="spreading", n_neighbors=10, alpha=0.2):
    """Degenerate GM → use scikit SSL on the instance graph."""
    return sklearn_ssl_predict(Xtr, ytr, Xte, method=method, n_neighbors=n_neighbors, alpha=alpha)

def lp_labels_on_grid_fast(Xtr, ytr, Z, n_neighbors=10, alpha=0.2):
    return sklearn_ssl_predict(Xtr, ytr, Z, method="spreading", n_neighbors=n_neighbors, alpha=alpha)

def probs_linear(Xtr, ytr, Xte):
    m = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    return m.predict_proba(Xte)

def probs_knn(Xtr, ytr, Xte, k=10):
    m = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    return m.predict_proba(Xte)

def probs_labels_spreading_transductive(XL, yL, XU, n_neighbors=10, alpha=0.2):
    X_all = np.r_[XL, XU]
    y_all = np.r_[yL, -np.ones(len(XU), dtype=int)]
    lp = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=30)
    lp.fit(X_all, y_all)
    return lp.label_distributions_[-len(XU):]  # normalized probs


def probs_gm(model, Xq, n_classes=2):
    return model.predict_proba(Xq, n_classes)

def all_metrics(probs, y):
    # NLL / Brier / ECE: binary or multi-class (here binary)
    p = np.clip(probs, 1e-12, 1.0)
    acc = float((p.argmax(1) == y).mean())
    # NLL
    nll = float(-np.log(p[np.arange(len(y)), y]).mean())
    # Brier
    Y = np.zeros_like(p); Y[np.arange(len(y)), y] = 1.0
    brier = float(((p - Y)**2).sum(axis=1).mean())
    # ECE (15 bins, max prob as confidence)
    conf = p.max(axis=1); pred = p.argmax(axis=1); correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, 16); ece = 0.0
    for i in range(15):
        lo, hi = bins[i], bins[i+1]
        m = (conf >= lo) & (conf < hi if i < 14 else conf <= hi)
        if not np.any(m): continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return dict(acc=acc, nll=nll, brier=brier, ece=float(ece))



# ================================================================
# Baselines
# ================================================================

def baseline_linear():  return LogisticRegression(max_iter=500)
def baseline_knn(k=10): return KNeighborsClassifier(n_neighbors=k)

# ================================================================
# Plotting
# ================================================================

# ============================================================
# Grid-based probability maps for smoothness visualization
# ============================================================


def make_grid(Xtr, Xte, n=300, pad=1.0):
    """
    Create a 2D grid covering both training and test sets with padding.
    Returns (xx, yy, grid) where grid is [N^2, 2].
    """
    x_min, x_max = np.r_[Xtr[:, 0], Xte[:, 0]].min() - pad, np.r_[Xtr[:, 0], Xte[:, 0]].max() + pad
    y_min, y_max = np.r_[Xtr[:, 1], Xte[:, 1]].min() - pad, np.r_[Xtr[:, 1], Xte[:, 1]].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n),
                         np.linspace(y_min, y_max, n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def probs_on_grid_knn(Xtr, ytr, xx, yy, k=10):
    """
    Compute kNN probability map over a 2D grid.
    Returns a [H, W] map of P(class=1).
    """
    grid = np.c_[xx.ravel(), yy.ravel()]
    knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    P = knn.predict_proba(grid)
    # Ensure column 1 corresponds to positive class (handle sklearn convention)
    if P.shape[1] == 1:
        P = np.hstack([1 - P, P])
    return P[:, 1].reshape(xx.shape)


def probs_on_grid_ls(Xtr, ytr, xx, yy, n_neighbors=10, alpha=0.2):
    """
    Compute LabelSpreading (transductive) probabilities on a 2D grid.
    Uses the existing helper probs_labels_spreading_transductive.
    """
    grid = np.c_[xx.ravel(), yy.ravel()]
    P = probs_labels_spreading_transductive(Xtr, ytr, grid,
                                            n_neighbors=n_neighbors, alpha=alpha)
    if P.shape[1] == 1:
        P = np.hstack([1 - P, P])
    return P[:, 1].reshape(xx.shape)


def probs_on_grid_gm(gm, xx, yy, n_classes=2):
    """
    Compute GraphMemory probabilities on a 2D grid.
    """
    grid = np.c_[xx.ravel(), yy.ravel()]
    P = probs_gm(gm, grid, n_classes)
    if P.shape[1] == 1:
        P = np.hstack([1 - P, P])
    return P[:, 1].reshape(xx.shape)


def _grid_spacings(xx, yy):
    # uniform grids -> use mean step; protect against zeros
    dx = float(np.mean(np.diff(xx[0, :]))) if xx.shape[1] > 1 else 1.0
    dy = float(np.mean(np.diff(yy[:, 0]))) if yy.shape[0] > 1 else 1.0
    dx = dx if abs(dx) > 1e-12 else 1.0
    dy = dy if abs(dy) > 1e-12 else 1.0
    return dx, dy

def decision_smoothness(pmap, xx, yy):
    """Mean squared gradient magnitude of class-1 prob over the grid."""
    dx, dy = _grid_spacings(xx, yy)
    # gradient expects spacings in the order of axes: (dy, dx)
    gy, gx = np.gradient(pmap, dy, dx, edge_order=2)
    return float(np.nanmean(gx**2 + gy**2))

def grad_magnitude_map(pmap, xx, yy):
    """√(gx²+gy²) map for visualization (not averaged)."""
    dx, dy = _grid_spacings(xx, yy)
    gy, gx = np.gradient(pmap, dy, dx, edge_order=2)
    gm = np.sqrt(np.maximum(gx**2 + gy**2, 0.0))
    # replace any NaNs/Infs that might slip through
    return np.nan_to_num(gm, nan=0.0, posinf=0.0, neginf=0.0)



def plot_decision(ax, predict_fn, Xtr, ytr, Xte, yte, title):
    x_min, x_max = np.r_[Xtr[:,0], Xte[:,0]].min()-1, np.r_[Xtr[:,0], Xte[:,0]].max()+1
    y_min, y_max = np.r_[Xtr[:,1], Xte[:,1]].min()-1, np.r_[Xtr[:,1], Xte[:,1]].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,300), np.linspace(y_min,y_max,300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)
    cmap_bg = ListedColormap(["#209922","#88879d"])
    cmap_pts= ListedColormap(["#0284c7","#ea580c"])
    ax.contourf(xx,yy,Z,alpha=0.35,cmap=cmap_bg,levels=[-0.5,0.5,1.5])
    ax.scatter(Xtr[:,0],Xtr[:,1],c=ytr,s=12,alpha=0.7,cmap=cmap_pts,edgecolors="k",linewidths=0.2)
    ax.scatter(Xte[:,0],Xte[:,1],c=yte,s=18,alpha=0.9,cmap=cmap_pts,marker="x")
    ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([])

# ================================================================
# Timing helpers
# ================================================================

def time_fit_predict(clf, Xtr, ytr, Xte):
    t0 = perf_counter(); clf.fit(Xtr, ytr); t_fit = perf_counter() - t0
    t0 = perf_counter(); yhat = clf.predict(Xte); t_pred = perf_counter() - t0
    return yhat, t_fit, t_pred

# ================================================================
# Budget helper (stratified subset)
# ================================================================

def stratified_quota_indices(y, total, rng, min_each=1):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if total <= 0 or counts.sum() == 0:
        return np.array([], dtype=int)
    alloc = np.zeros_like(counts)
    # guarantee at least one from each class (if possible)
    for i, c in enumerate(classes):
        alloc[i] = min(min_each, counts[i])
    rem = total - int(alloc.sum())
    if rem > 0:
        # distribute remainder proportional to availability
        room = counts - alloc
        if room.sum() > 0:
            w = room / room.sum()
            add = np.minimum(room, np.floor(w * rem).astype(int))
            alloc += add
            rem2 = rem - int(add.sum())
            if rem2 > 0:
                # greedy fill remaining
                order = np.argsort(-room)
                for i in order:
                    if rem2 == 0: break
                    if room[i] - add[i] > 0:
                        alloc[i] += 1; rem2 -= 1
    chosen = []
    for i, c in enumerate(classes):
        k = int(min(alloc[i], counts[i]))
        if k <= 0: continue
        pool = np.where(y == c)[0]
        chosen.append(rng.choice(pool, size=k, replace=False))
    return np.concatenate(chosen) if chosen else np.array([], int)

# ================================================================
# Reporting helpers
# ================================================================

def rows_to_table(rows):
    table = Table(title="Summary (acc / time)", show_lines=False, header_style="bold magenta")
    cols = ["tag","method","acc","t_fit(s)","t_pred(s)","n_train","n_test","K"]
    for c in cols: table.add_column(c)
    for r in rows:
        table.add_row(
            r["tag"], r["method"], f'{r["acc"]:.4f}',
            f'{r["t_fit"]:.6f}', f'{r["t_pred"]:.6f}',
            str(r["n_train"]), str(r["n_test"]), str(r.get("K","-"))
        )
    return table

# ================================================================
# Experiment runner
# ================================================================

def run_block(tag, X, y, seed, outdir, imbalance, budget_match):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    #scaler = StandardScaler().fit(Xtr)
    #Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    nC = 2
    if imbalance > 1.0:
        Xtr, ytr = apply_long_tail(Xtr, ytr, imbalance, seed)

    results = []
    results_cal = []
    results_smooth = []
    plots_dir = Path(outdir) / "plots"
    ensure_dir(plots_dir)

    # ---------- Baselines (predictions + times) ----------
    y_lin, tfit_lin, tpred_lin = time_fit_predict(baseline_linear(), Xtr, ytr, Xte)
    y_knn, tfit_knn, tpred_knn = time_fit_predict(baseline_knn(k=10), Xtr, ytr, Xte)

    # LabelSpreading (transductive over train+test)
    t0 = perf_counter()
    P_tst_ls = probs_labels_spreading_transductive(Xtr, ytr, Xte, n_neighbors=10, alpha=0.2)
    t_ls = perf_counter() - t0
    y_lps = P_tst_ls.argmax(1)

    # GM & degenerates
    #Kproto = min(max(256, len(Xtr)//40), len(Xtr))
    Kproto = max(32, min(120, len(Xtr)//10))
    #Kproto = 32
    gm_cfg = GMConfig(
        K=Kproto, knn_graph=10,
        attach_k=8,                    # local attachment; 5–10 is robust
        alpha=0.5,                     # mild diffusion
        beta=0.1,                      # sharp RBF falloff
        use_minibatch_kmeans=True,
        kmeans_batch_size=8192,
        reliability_sample_cap=512,
        disable_instability=False,     # speedup; instability seldom changes ranking
        diffusion_iters=20,            # avoids matrix inverse
        absolute_reliability=True,     # prototypes' reliabilities are not normalized away
        enforce_opposite_seed=True,    # handle mixed boundaries myopia
        coverage_search_factor=2,       # check small window for opposite class
        dtype="float32"
    )
    t0 = perf_counter(); gm = GraphMemory(gm_cfg).fit(Xtr, ytr); tfit_gm = perf_counter()-t0
    t0 = perf_counter(); y_gm = gm.predict(Xte, nC); tpred_gm = perf_counter()-t0

    gm_flat = gm_flat_from(gm)
    t0 = perf_counter(); y_gmflat = gm_flat.predict(Xte, nC); tpred_gmflat = perf_counter()-t0
    tfit_gmflat = 0.0  # re-use gm fit

    t0 = perf_counter(); gmi = gm_instance(Xtr, ytr, knn_graph=10, alpha=0.2, beta=1.0, attach_k=10); tfit_gmi=perf_counter()-t0
    t0 = perf_counter(); y_gmi = gmi.predict(Xte, nC); tpred_gmi=perf_counter()-t0

    t0 = perf_counter(); gmc = gm_centroid(Xtr, ytr); tfit_gmc=perf_counter()-t0
    t0 = perf_counter(); y_gmc = gmc.predict(Xte, nC); tpred_gmc=perf_counter()-t0

    # ---------- Accuracies ----------
    acc_lin  = accuracy_score(yte, y_lin)
    acc_knn  = accuracy_score(yte, y_knn)
    acc_lps  = accuracy_score(yte, y_lps)
    acc_gm   = accuracy_score(yte, y_gm)
    acc_gmf  = accuracy_score(yte, y_gmflat)
    acc_gmi  = accuracy_score(yte, y_gmi)
    acc_gmc  = accuracy_score(yte, y_gmc)

    # ---------- Calibration metrics (probs for ALL methods) ----------
    P_lin  = probs_linear(Xtr, ytr, Xte)
    P_knn  = probs_knn(Xtr, ytr, Xte, k=10)
    P_gm   = probs_gm(gm,      Xte, nC)
    P_gmf  = probs_gm(gm_flat, Xte, nC)
    P_gmi  = probs_gm(gmi,     Xte, nC)
    P_gmc  = probs_gm(gmc,     Xte, nC)

    cal_rows = [
        dict(tag=tag, method="Linear",          **all_metrics(P_lin,  yte)),
        dict(tag=tag, method="kNN",             **all_metrics(P_knn,  yte)),
        dict(tag=tag, method="LabelSpreading",  **all_metrics(P_tst_ls,    yte)),
        dict(tag=tag, method=f"GM (P={Kproto})",              **all_metrics(P_gm,   yte)),
        dict(tag=tag, method=f"GM-flat (P={Kproto})",         **all_metrics(P_gmf,  yte)),
        dict(tag=tag, method="GM-instance",     **all_metrics(P_gmi,  yte)),
        dict(tag=tag, method="GM-centroid",     **all_metrics(P_gmc,  yte)),
    ]

    rows = [
        dict(tag=tag, method="Linear",         acc=acc_lin,  t_fit=tfit_lin,   t_pred=tpred_lin,   n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="kNN",            acc=acc_knn,  t_fit=tfit_knn,   t_pred=tpred_knn,   n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="LabelSpreading", acc=acc_lps,  t_fit=0.0,        t_pred=t_ls,        n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method=f"GM (P={Kproto})",             acc=acc_gm,   t_fit=tfit_gm,    t_pred=tpred_gm,    n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method=f"GM-flat (P={Kproto})",        acc=acc_gmf,  t_fit=tfit_gmflat,t_pred=tpred_gmflat,n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method="GM-instance",    acc=acc_gmi,  t_fit=tfit_gmi,   t_pred=tpred_gmi,   n_train=len(Xtr), n_test=len(Xte), K=len(Xtr)),
        dict(tag=tag, method="GM-centroid",    acc=acc_gmc,  t_fit=tfit_gmc,   t_pred=tpred_gmc,   n_train=len(Xtr), n_test=len(Xte), K=2),
    ]

    # ---------- Budget-matched kNN (|train| = #prototypes) ----------
    if budget_match:
        P = Kproto
        rng = np.random.RandomState(seed + 20)
        keep = stratified_quota_indices(y=ytr, total=P, rng=rng, min_each=1)
        # if rounding left us short, fill randomly
        if len(keep) < P:
            pool = np.setdiff1d(np.arange(len(ytr)), np.array(keep, int), assume_unique=False)
            extra = min(P - len(keep), len(pool))
            if extra > 0:
                keep.extend(rng.choice(pool, size=extra, replace=False).tolist())
        keep = np.array(keep, dtype=int)

        Xsub, ysub = Xtr[keep], ytr[keep]
        t0 = perf_counter(); knn_b = KNeighborsClassifier(n_neighbors=10).fit(Xsub, ysub); tfit_knnb = perf_counter()-t0
        t0 = perf_counter(); y_knnb = knn_b.predict(Xte); tpred_knnb = perf_counter()-t0
        acc_knnb = accuracy_score(yte, y_knnb)
        rows.append(
            dict(tag=tag, method=f"kNN-budget(P={P})", acc=acc_knnb,
                 t_fit=tfit_knnb, t_pred=tpred_knnb, n_train=len(Xsub), n_test=len(Xte), K=P)
        )
        # calibration for budget kNN
        P_knnb = knn_b.predict_proba(Xte)
        cal_rows.append(dict(tag=tag, method=f"kNN-budget(P={P})", **all_metrics(P_knnb, yte)))

    results.extend(rows)
    results_cal.extend(cal_rows)

    # ---------- Plots ----------
    # --- Smoothness maps (GM / kNN / LabelSpreading) ---
    xx, yy, grid = make_grid(Xtr, Xte, n=300, pad=1.0)

    pmap_knn = probs_on_grid_knn(Xtr, ytr, xx, yy, k=10)
    pmap_ls  = probs_on_grid_ls(Xtr, ytr, xx, yy, n_neighbors=10, alpha=0.2)
    pmap_gm  = probs_on_grid_gm(gm, xx, yy, n_classes=nC)
    pmap_gm_flat  = probs_on_grid_gm(gm_flat, xx, yy, n_classes=nC)

    sm_knn = decision_smoothness(pmap_knn, xx, yy)
    sm_ls  = decision_smoothness(pmap_ls,  xx, yy)
    sm_gm  = decision_smoothness(pmap_gm,  xx, yy)
    sm_gm_flat  = decision_smoothness(pmap_gm_flat,  xx, yy)

    results_smooth.append(dict(tag=tag, method=f"GM (P={Kproto})",             smooth=sm_gm))
    results_smooth.append(dict(tag=tag, method=f"GM-flat (P={Kproto})",        smooth=sm_gm_flat))
    results_smooth.append(dict(tag=tag, method="kNN",            smooth=sm_knn))
    results_smooth.append(dict(tag=tag, method="LabelSpreading", smooth=sm_ls))

    # Visualize gradient magnitude
    gm_knn = grad_magnitude_map(pmap_knn, xx, yy)
    gm_ls  = grad_magnitude_map(pmap_ls,  xx, yy)
    gm_gm  = grad_magnitude_map(pmap_gm,  xx, yy)

    vmax = np.percentile(np.r_[gm_knn.ravel(), gm_ls.ravel(), gm_gm.ravel()], 99)

    fig_s, axs_s = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)

    im0 = axs_s[0].imshow(gm_gm, origin="lower",
                        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                        cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axs_s[0].set_title(f"GM (P={Kproto}) ∥∇p∥ (mean²={sm_gm:.3f})"); 

    im1 = axs_s[1].imshow(gm_knn, origin="lower",
                        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                        cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axs_s[1].set_title(f"kNN ∥∇p∥ (mean²={sm_knn:.3f})"); 

    im2 = axs_s[2].imshow(gm_ls, origin="lower",
                        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                        cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axs_s[2].set_title(f"LabelSpreading ∥∇p∥ (mean²={sm_ls:.3f})"); 
    
    axs_s[0].set_xticks([]); axs_s[0].set_yticks([])
    axs_s[1].set_xticks([]); axs_s[1].set_yticks([])
    axs_s[2].set_xticks([]); axs_s[2].set_yticks([])

    fig_s.colorbar(im2, ax=axs_s.ravel().tolist(), shrink=0.85, pad=0.02)
    fig_s.suptitle(f"{tag} - Decision Smoothness (lower is smoother)")
    smooth_path = str((Path(outdir) / "plots" / f"{tag}_smoothness.png").resolve())
    plt.savefig(smooth_path, dpi=220)
    plt.close(fig_s)


    # ---------- Decision boundary plots ----------
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    #fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    # plot_decision(axs[0,0], lambda Z: baseline_linear().fit(Xtr, ytr).predict(Z), Xtr, ytr, Xte, yte, f"Linear ({acc_lin:.3f})")
    #plot_decision(axs[0,1], lambda Z: baseline_knn(k=10).fit(Xtr, ytr).predict(Z), Xtr, ytr, Xte, yte, f"kNN ({acc_knn:.3f})")
    plot_decision(axs[1], lambda Z: baseline_knn(k=10).fit(Xtr, ytr).predict(Z), Xtr, ytr, Xte, yte, f"kNN ({acc_knn:.3f})")
    # plot_decision(
    #     axs[0,2],
    #     lambda Z: lp_labels_on_grid_fast(Xtr, ytr, Z, n_neighbors=10, alpha=0.2),
    #     Xtr, ytr, Xte, yte, f"LabelSpreading ({acc_lps:.3f})"
    # )
    plot_decision(
        axs[2],
        lambda Z: lp_labels_on_grid_fast(Xtr, ytr, Z, n_neighbors=10, alpha=0.2),
        Xtr, ytr, Xte, yte, f"LabelSpreading ({acc_lps:.3f})"
    )
    plot_decision(axs[0], lambda Z: gm.predict(Z, nC), Xtr, ytr, Xte, yte, f"GM ({acc_gm:.3f})")
    # plot_decision(axs[1,1], lambda Z: gm_flat.predict(Z, nC),  Xtr, ytr, Xte, yte, f"GM-flat ({acc_gmf:.3f})")
    # plot_decision(axs[1,2], lambda Z: gm_centroid(Xtr, ytr).predict(Z, nC), Xtr, ytr, Xte, yte, f"GM-centroid ({acc_gmc:.3f})")
    plt.suptitle(tag, fontsize=11)
    ensure_dir(plots_dir)
    plt.tight_layout(rect=[0,0,1,0.97])
    fig_path = str(plots_dir / f"{tag}.png")
    plt.savefig(fig_path, dpi=220); plt.close()

    return results, results_cal, results_smooth, fig_path


# ================================================================
# Suite
# ================================================================

def run_suite(dataset: str, outdir: str, seed: int, imbalance: float, budget_match: bool):
    ensure_dir(outdir)
    set_seed(seed)

    # Data
    if dataset == "moons":
        X, y = gen_moons(n_samples=4000, seed=seed, noise=0.25)
    elif dataset == "circles":
        X, y = gen_circles(n_samples=4000, seed=seed, factor=0.1, noise=0.25)
    else:
        raise ValueError("--dataset must be 'moons' or 'circles'")

    tag = f"{dataset}_2class_fixed"
    rows, cal_rows, smooth_rows, fig_path = run_block(tag, X, y, seed, outdir, imbalance, budget_match)

    # Save CSV/JSON
    csv_path = os.path.join(outdir, f"{dataset}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag","method","acc","t_fit","t_pred","n_train","n_test","K"])
        for r in rows:
            wr.writerow([r["tag"], r["method"], f'{r["acc"]:.6f}', f'{r["t_fit"]:.8f}',
                         f'{r["t_pred"]:.8f}', r["n_train"], r["n_test"], r.get("K","-")])
            
    cal_csv_path = os.path.join(outdir, f"{dataset}_calibration.csv")
    with open(cal_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag","method","acc","nll","brier","ece"])
        for r in cal_rows:
            wr.writerow([
                r["tag"],
                r["method"],
                f'{r["acc"]:.6f}',
                f'{r["nll"]:.6f}',
                f'{r["brier"]:.6f}',
                f'{r["ece"]:.6f}',
            ])

    # Save per-run smooth CSV 
    smooth_csv_path = os.path.join(outdir, f"{dataset}_smooth.csv")
    with open(smooth_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag","method","smooth"])
        for r in smooth_rows:
            wr.writerow([
                r["tag"],
                r["method"],
                f'{r["smooth"]:.6f}',
            ])


    with open(os.path.join(outdir, f"{dataset}_summary.json"), "w") as jf:
        json.dump(rows, jf, indent=2)

    console.print(rows_to_table(rows))
    console.print(Panel.fit(
        f"[bold green]Saved[/bold green] CSV → [white]{Path(csv_path).resolve()}[/white]\n"
        f"[bold green]Plot[/bold green] → [white]{Path(fig_path).resolve()}[/white]"
    ))

# ================================================================
# CLI
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["moons","circles"], required=True,
                    help="Binary non-linear datasets.")
    ap.add_argument("--outdir", type=str, default="runs_toy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imbalance", type=float, default=1.0,
                    help="Long-tail ratio (max/min) applied to TRAIN only. >1 enables long-tail.")
    ap.add_argument("--budget-match", action="store_true",
                    help="Also run kNN with training size equal to GM's #prototypes (memory budget).")
    args = ap.parse_args()

    with section(f"Running suite: {args.dataset}  (long-tail={args.imbalance}  budget={args.budget_match})"):
        run_suite(args.dataset, args.outdir, args.seed, args.imbalance, args.budget_match)

if __name__ == "__main__":
    main()
