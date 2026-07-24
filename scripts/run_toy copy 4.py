#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focused experiments for Graph Memory (GM) — only scenarios where GM shines:

- Datasets (non-linear): moons, circles
- Regimes:
  * clean (no noise sweep)
  * long-tail (class imbalance on training set via --imbalance > 1)
  * budget (adds budget-matched kNN using the same memory as GM)
- Baselines: Linear, kNN, Label Propagation (LabelSpreading)
- GM + Degenerates: instance-prototypes, no-edges (centroids), flat-propagation
- Reliability: r_c = sigmoid( λ1 s_c + λ2 \bar{m}_c - λ3 ρ_c - λ4 \bar{v}_c + λ5 π_c )

Usage:
  python run_toy_gm_focus.py --dataset moons --outdir runs_toy --imbalance 8 --budget-match
  python run_toy_gm_focus.py --dataset circles --outdir runs_toy
"""

import os
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
from typing import Tuple, List, Dict
from time import perf_counter
from contextlib import contextmanager
from pathlib import Path

# Progress / pretty output
from tqdm.auto import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as rich_traceback_install
rich_traceback_install(show_locals=False)
console = Console()

from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph, NearestNeighbors
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int):
    np.random.seed(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def robust_sigmoid_normalize(values):
    x = np.asarray(values).astype(np.float64)
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = max(q75 - q25, 1e-12)
    z = (x - med) / iqr
    return sigmoid(z)

@contextmanager
def section(msg: str):
    console.rule(f"[bold cyan]{msg}")
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        console.print(Panel.fit(f"[green]{msg}[/green] finished in [bold]{dt:.3f}s[/bold]"))

# ---------------------------
# Data generators (non-linear only)
# ---------------------------

def gen_moons_pairs(n_samples=2000, n_pairs=2, gap=6.0, seed=0, noise_fixed=0.15):
    """Multiple non-overlapping moon pairs; classes = 2 * n_pairs. Fixed noise (no sweeps)."""
    rng = np.random.RandomState(seed)
    per_pair = n_samples // n_pairs
    Xs, ys = [], []
    for p in range(n_pairs):
        Xm, Ym = make_moons(n_samples=per_pair, noise=noise_fixed, random_state=rng.randint(10_000))
        tx = (p % 3) * gap
        ty = (p // 3) * gap
        Xm = Xm + np.array([tx, ty])[None, :]
        Ym = Ym + 2*p
        Xs.append(Xm); ys.append(Ym)
    return np.vstack(Xs), np.concatenate(ys)

def gen_circles_pairs(n_samples=2000, n_pairs=2, gap=6.0, seed=0, factor=0.4):
    """Tile several 2-class circle pairs (strictly non-linear). No noise."""
    rng = np.random.RandomState(seed)
    per_pair = n_samples // n_pairs
    Xs, ys = [], []
    for p in range(n_pairs):
        Xc, yc = make_circles(n_samples=per_pair, factor=factor, noise=0.0, random_state=rng.randint(10_000))
        tx = (p % 3) * gap
        ty = (p // 3) * gap
        Xc = Xc + np.array([tx, ty])[None, :]
        yc = yc + 2*p
        Xs.append(Xc); ys.append(yc)
    return np.vstack(Xs), np.concatenate(ys)

# ---------------------------
# Long-tail maker (train only)
# ---------------------------

def apply_long_tail(Xtr, ytr, im_ratio, seed):
    """Impose class imbalance: max/min ≈ im_ratio across sorted classes."""
    if im_ratio <= 1.0:
        return Xtr, ytr
    rng = np.random.RandomState(seed+11)
    C = int(ytr.max())+1
    counts = np.bincount(ytr, minlength=C).astype(int)
    maxc = counts.max()
    targets = [max(5, min(int(round(maxc / (im_ratio ** (i/max(1,C-1)) ))), counts[i])) for i in range(C)]
    keep = []
    for c in range(C):
        idx = np.where(ytr==c)[0]
        if len(idx) <= targets[c]:
            keep += idx.tolist()
        else:
            keep += rng.choice(idx, size=targets[c], replace=False).tolist()
    keep = np.array(keep, int)
    return Xtr[keep], ytr[keep]

# ---------------------------
# Reliability metrics (full)
# ---------------------------

def compute_cluster_assignments(P, X):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(P)
    d, idx = nbrs.kneighbors(X, return_distance=True)
    return idx.ravel(), d.ravel()

def prototype_purity(assign, y, K, n_classes):
    yc = np.zeros(K, dtype=int)
    pi = np.zeros(K, dtype=float)
    for c in range(K):
        ii = np.where(assign == c)[0]
        if len(ii) == 0:
            yc[c] = 0; pi[c] = 0.0
            continue
        hist = np.bincount(y[ii], minlength=n_classes)
        yc[c] = int(np.argmax(hist))
        pi[c] = float(hist.max()) / max(1.0, float(len(ii)))
    return yc, pi

def silhouette_normalized(X, assign, K):
    idxs = [np.where(assign == c)[0] for c in range(K)]
    s_c = np.zeros(K, dtype=float)
    for c in range(K):
        I = idxs[c]
        if len(I) <= 1:
            s_c[c] = 0.5
            continue
        Xi = X[I]
        d_within = np.sqrt(((Xi[:, None, :] - Xi[None, :, :]) ** 2).sum(-1) + 1e-12)
        a_i = (d_within.sum(axis=1)) / max(1, (len(I)-1))
        b_i = np.full(len(I), np.inf, dtype=float)
        for c2 in range(K):
            if c2 == c or len(idxs[c2]) == 0:
                continue
            Xj = X[idxs[c2]]
            d_cross = np.sqrt(((Xi[:, None, :] - Xj[None, :, :]) ** 2).sum(-1) + 1e-12)
            mean_to_c2 = d_cross.mean(axis=1)
            b_i = np.minimum(b_i, mean_to_c2)
        sil_i = (b_i - a_i) / (np.maximum(a_i, b_i) + 1e-12)
        s_c[c] = np.clip(((sil_i + 1.0) * 0.5).mean(), 0.0, 1.0)
    return s_c

def dispersion(X, assign, P, K):
    v = np.zeros(K, dtype=float)
    for c in range(K):
        I = np.where(assign == c)[0]
        if len(I) == 0:
            v[c] = 0.0
            continue
        Xi = X[I]
        v[c] = float(((Xi - P[c][None, :]) ** 2).sum(axis=1).mean())
    return v

def margins(P, yc):
    K = len(P)
    m = np.zeros(K, dtype=float)
    for c in range(K):
        mask = (yc != yc[c])
        if not np.any(mask):
            m[c] = 0.0
            continue
        d = np.linalg.norm(P[mask] - P[c][None, :], axis=1)
        m[c] = float(d.min()) if len(d) else 0.0
    return m

def instability(X, assign, P, sigma_noise=0.05, trials=3):
    K = len(P)
    n = len(X)
    changed = np.zeros(K, dtype=float)
    counts  = np.zeros(K, dtype=float)
    nbrs = NearestNeighbors(n_neighbors=1).fit(P)
    for _ in range(trials):
        Xd = X + np.random.normal(scale=sigma_noise, size=X.shape)
        nn = nbrs.kneighbors(Xd, return_distance=False).ravel()
        for i in range(n):
            c0 = assign[i]
            counts[c0] += 1.0
            if nn[i] != c0:
                changed[c0] += 1.0
    rho = np.zeros(K, dtype=float)
    mask = counts > 0
    rho[mask] = changed[mask] / counts[mask]
    return rho

def reliability_full(s_c, m_c, v_c, rho_c, pi_c, lam=(1,1,1,1,1)):
    lam1, lam2, lam3, lam4, lam5 = lam
    m_bar = robust_sigmoid_normalize(m_c)
    v_bar = robust_sigmoid_normalize(v_c)
    z = (lam1 * s_c + lam2 * m_bar - lam3 * rho_c - lam4 * v_bar + lam5 * pi_c)
    return sigmoid(z)

# ---------------------------
# Graph Memory (full)
# ---------------------------

@dataclass
class GMConfig:
    K: int
    knn_graph: int = 10
    attach_k: int = 3
    alpha: float = 0.6
    beta: float = 1.0
    use_norm: bool = False
    lam1: float = 1.0
    lam2: float = 1.0
    lam3: float = 1.0
    lam4: float = 1.0
    lam5: float = 1.0
    instab_sigma: float = 0.05
    instab_trials: int = 3

class GraphMemory:
    def __init__(self, cfg: GMConfig):
        self.cfg = cfg
        self.P = None
        self.yc = None
        self.pi = None
        self.r = None
        self.S = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        K = min(self.cfg.K, len(X))
        km = KMeans(n_clusters=K, n_init="auto", random_state=0).fit(X)
        P = km.cluster_centers_
        assign = km.labels_

        n_classes = int(y.max()) + 1
        yc, pi = prototype_purity(assign, y, K, n_classes)

        s = silhouette_normalized(X, assign, K)
        v = dispersion(X, assign, P, K)
        m = margins(P, yc)
        rho = instability(X, assign, P, sigma_noise=self.cfg.instab_sigma, trials=self.cfg.instab_trials)
        r = reliability_full(s, m, v, rho, pi,
                             lam=(self.cfg.lam1, self.cfg.lam2, self.cfg.lam3, self.cfg.lam4, self.cfg.lam5))

        Pn = l2n(P) if self.cfg.use_norm else P
        A = kneighbors_graph(Pn, n_neighbors=min(self.cfg.knn_graph, K-1),
                             mode="distance", include_self=False).toarray()
        A = np.exp(-self.cfg.beta * (A ** 2))
        A = np.maximum(A, A.T)
        D = A.sum(axis=1, keepdims=True) + 1e-12
        S = A / D

        self.P, self.yc, self.pi, self.r, self.S = P, yc, pi, r, S
        return self

    def _z0(self, xq: np.ndarray):
        d = np.linalg.norm(self.P - xq[None, :], axis=1)
        idx = np.argsort(d)[:min(self.cfg.attach_k, len(self.P))]
        w = np.exp(-self.cfg.beta * (d[idx] ** 2)) * (self.r[idx] + 1e-12)
        z0 = np.zeros(len(self.P))
        z0[idx] = w / (w.sum() + 1e-12)
        return z0

    def predict_proba(self, Xq: np.ndarray, n_classes: int) -> np.ndarray:
        probs = np.zeros((len(Xq), n_classes))
        I = np.eye(len(self.P))
        M = np.linalg.inv(I - self.cfg.alpha * self.S)
        for i, x in enumerate(Xq):
            z0 = self._z0(x)
            z = (1 - self.cfg.alpha) * (M @ z0)
            for c in range(n_classes):
                probs[i, c] = z[self.yc == c].sum()
            s = probs[i].sum()
            probs[i] = probs[i] / (s + 1e-12)
        return probs

    def predict(self, Xq: np.ndarray, n_classes: int) -> np.ndarray:
        return np.argmax(self.predict_proba(Xq, n_classes), axis=1)

# ---------------------------
# Degenerate variants
# ---------------------------

def gm_instance_prototypes(Xtr, ytr, knn_graph=10, alpha=0.6, beta=1.0, attach_k=3):
    gm = GraphMemory(GMConfig(K=len(Xtr), knn_graph=knn_graph, alpha=alpha, beta=beta, attach_k=attach_k))
    gm.P = Xtr.copy()
    gm.yc = ytr.copy()
    gm.pi = np.ones(len(Xtr))
    gm.r  = np.ones(len(Xtr))
    A = kneighbors_graph(Xtr, n_neighbors=min(knn_graph, len(Xtr)-1),
                         mode="distance", include_self=False).toarray()
    A = np.exp(-beta * (A ** 2))
    A = np.maximum(A, A.T)
    D = A.sum(axis=1, keepdims=True) + 1e-12
    gm.S = A / D
    return gm

def gm_no_edges_centroids(Xtr, ytr, n_classes, alpha=0.0):
    means = []
    for c in range(n_classes):
        idx = np.where(ytr == c)[0]
        means.append(Xtr[idx].mean(axis=0))
    means = np.vstack(means)
    gm = GraphMemory(GMConfig(K=n_classes, knn_graph=1, alpha=alpha, attach_k=1))
    gm.P = means
    gm.yc = np.arange(n_classes)
    gm.pi = np.ones(n_classes)
    gm.r  = np.ones(n_classes)
    gm.S  = np.zeros((n_classes, n_classes))
    return gm

def gm_flat_propagation(Xtr, ytr, K=32, knn_graph=10, alpha=0.6, beta=1.0, attach_k=3):
    gm = GraphMemory(GMConfig(K=K, knn_graph=knn_graph, alpha=alpha, beta=beta, attach_k=attach_k)).fit(Xtr, ytr)
    gm.r = np.ones_like(gm.r)
    return gm

# ---------------------------
# Baselines
# ---------------------------

def baseline_linear():
    return LogisticRegression(max_iter=500, C=1.0)

def baseline_knn(k=10):
    return KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)

def baseline_lp(n_neighbors=10):
    return LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=0.2, max_iter=50)

# ---------------------------
# Plotting
# ---------------------------

def plot_decision(ax, predict_fn, Xtr, ytr, Xte, yte, title, n_classes):
    x_min, x_max = np.r_[Xtr[:,0], Xte[:,0]].min()-1.0, np.r_[Xtr[:,0], Xte[:,0]].max()+1.0
    y_min, y_max = np.r_[Xtr[:,1], Xte[:,1]].min()-1.0, np.r_[Xtr[:,1], Xte[:,1]].max()+1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)

    cmap_bg = ListedColormap(["#f0f9ff", "#fff7ed", "#f3e8ff", "#ecfeff", "#fef2f2", "#eff6ff", "#fefce8", "#e5f6f0"])
    cmap_pts = ListedColormap(["#0284c7", "#ea580c", "#7c3aed", "#0891b2", "#dc2626", "#2563eb", "#ca8a04", "#059669"])

    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg, levels=np.arange(n_classes+1)-0.5)
    ax.scatter(Xtr[:,0], Xtr[:,1], c=ytr, s=12, alpha=0.75, cmap=cmap_pts, edgecolors="k", linewidths=0.2, label="train")
    ax.scatter(Xte[:,0], Xte[:,1], c=yte, s=18, alpha=0.9, cmap=cmap_pts, marker="x", label="test")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def lp_labels_on_grid(Xtr, ytr, Z, n_neighbors):
    X_all = np.r_[Xtr, Z]
    y_all = np.r_[ytr, -np.ones(len(Z), dtype=int)]
    lp = baseline_lp(n_neighbors=n_neighbors)
    lp.fit(X_all, y_all)
    return lp.transduction_[-len(Z):]


# ---------------------------
# Timing helpers
# ---------------------------

def time_fit_predict(clf, Xtr, ytr, Xte):
    t0 = perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = perf_counter() - t0
    t0 = perf_counter()
    yhat = clf.predict(Xte)
    t_pred = perf_counter() - t0
    return yhat, t_fit, t_pred

def time_lp_fit_predict(lp_model, Xtr, ytr, Xte, yte):
    X_all = np.r_[Xtr, Xte]
    y_all = np.r_[ytr, -np.ones_like(yte)]
    t0 = perf_counter()
    lp_model.fit(X_all, y_all)
    t_fit = perf_counter() - t0
    t0 = perf_counter()
    yhat = lp_model.transduction_[-len(yte):]
    t_pred = perf_counter() - t0
    return yhat, t_fit, t_pred

def time_gm_fit_predict(gm_cfg, Xtr, ytr, Xte, nC):
    gm = GraphMemory(gm_cfg)
    t0 = perf_counter()
    gm.fit(Xtr, ytr)
    t_fit = perf_counter() - t0
    t0 = perf_counter()
    yhat = gm.predict(Xte, nC)
    t_pred = perf_counter() - t0
    return gm, yhat, t_fit, t_pred

# ---------------------------
# Budget helper (stratified subset)
# ---------------------------

def stratified_quota_indices(y, total, rng,
                             min_each=1,
                             max_per_class=None,
                             weights=None):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    C = len(classes)
    caps = np.array([min(counts[i], (max_per_class.get(int(c), counts[i]) if max_per_class else counts[i]))
                     for i, c in enumerate(classes)], dtype=int)
    if total <= 0 or caps.sum() == 0:
        return np.array([], dtype=int)
    mins = np.minimum(min_each, caps)
    min_sum = int(mins.sum())
    if total < min_sum:
        order = np.argsort(-caps)[:total]
        alloc = np.zeros(C, dtype=int); alloc[order] = 1
    else:
        alloc = mins.copy()
        rem = total - int(alloc.sum())
        if weights is None:
            w = caps / (caps.sum() + 1e-12)
        else:
            w = np.array([weights.get(int(c), 0.0) for c in classes], float)
            w = (w / w.sum()) if w.sum() > 0 else caps / (caps.sum() + 1e-12)
        room = caps - alloc
        quotas = w * rem
        base = np.minimum(room, np.floor(quotas).astype(int))
        alloc += base
        rem2 = rem - int(base.sum())
        if rem2 > 0:
            frac = quotas - np.floor(quotas)
            cand = np.where(room - base > 0)[0]
            if cand.size > 0:
                order = cand[np.argsort(-frac[cand])]
                take = min(rem2, order.size)
                alloc[order[:take]] += 1
    chosen = []
    for idx_c, c in enumerate(classes):
        k = int(alloc[idx_c])
        if k <= 0: continue
        pool_idx = np.where(y == c)[0]
        k = min(k, pool_idx.size)
        if k > 0:
            chosen.append(np.random.default_rng().choice(pool_idx, size=k, replace=False))
    if not chosen:
        return np.array([], dtype=int)
    return np.concatenate(chosen)

# ---------------------------
# Experiment runners (only non-linear + long-tail + budget)
# ---------------------------

def prep_split(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte

def _rows_to_rich_table(rows: List[Dict]) -> Table:
    table = Table(title="Summary (acc / time)", show_lines=False, header_style="bold magenta")
    cols = ["tag","method","acc","t_fit(s)","t_pred(s)","n_train","n_test","K","n_classes"]
    for c in cols:
        table.add_column(c)
    for r in rows:
        table.add_row(
            r["tag"], r["method"],
            f'{r["acc"]:.4f}',
            f'{r["t_fit"]:.6f}',
            f'{r["t_pred"]:.6f}',
            str(r["n_train"]), str(r["n_test"]),
            str(r.get("K","-")), str(r["n_classes"])
        )
    return table

def _make_dataset(dataset: str, seed: int, mode: str):
    """mode in {'2class','multiclass'} with fixed non-linear configs (no noise/std sweeps)."""
    if dataset == "moons":
        if mode == "2class":
            X, y = gen_moons_pairs(n_samples=2000, n_pairs=1, seed=seed, noise_fixed=0.15)
            y = y - y.min()  # keep 2 classes
        else:
            X, y = gen_moons_pairs(n_samples=3000, n_pairs=3, seed=seed, noise_fixed=0.15)  # 6 classes
    elif dataset == "circles":
        if mode == "2class":
            X, y = gen_circles_pairs(n_samples=2000, n_pairs=1, seed=seed)  # 2 classes
            y = y - y.min()
        else:
            X, y = gen_circles_pairs(n_samples=3000, n_pairs=3, seed=seed)  # 6 classes
    else:
        raise ValueError("--dataset must be 'moons' or 'circles'")
    return X, y

def _run_block(tag, X, y, seed, all_rows, plots_dir, gm_knn_graph, lp_n_neighbors, gm_attach_k=3):
    Xtr, Xte, ytr, yte = prep_split(X, y, seed)
    nC = len(np.unique(y))
    ntr, nte = len(Xtr), len(Xte)

    # Optional long-tail on train (controlled by global args via closure)
    if _ARGS.imbalance and _ARGS.imbalance > 1.0:
        Xtr, ytr = apply_long_tail(Xtr, ytr, _ARGS.imbalance, seed)

    # Baselines (timed)
    y_lin, tfit_lin, tpred_lin = time_fit_predict(baseline_linear(), Xtr, ytr, Xte)
    y_knn, tfit_knn, tpred_knn = time_fit_predict(baseline_knn(k=10), Xtr, ytr, Xte)
    y_lp,  tfit_lp,  tpred_lp  = time_lp_fit_predict(baseline_lp(n_neighbors=lp_n_neighbors), Xtr, ytr, Xte, yte)

    # GM + degenerates (timed)
    Kproto = int(min(8*nC if nC>2 else 32, max(2*nC if nC>2 else 8, len(Xtr)//10)))
    gm_cfg = GMConfig(K=Kproto, knn_graph=gm_knn_graph, attach_k=gm_attach_k, alpha=0.2, beta=1.0)

    gm, y_gm, tfit_gm, tpred_gm = time_gm_fit_predict(gm_cfg, Xtr, ytr, Xte, nC)

    t0 = perf_counter()
    gm_flat = gm_flat_propagation(Xtr, ytr, K=Kproto, knn_graph=gm_knn_graph, alpha=0.2, beta=1.0, attach_k=gm_attach_k)
    tfit_gmflat = perf_counter() - t0
    t0 = perf_counter()
    y_gmflat = gm_flat.predict(Xte, nC)
    tpred_gmflat = perf_counter() - t0

    t0 = perf_counter()
    gm_inst = gm_instance_prototypes(Xtr, ytr, knn_graph=gm_knn_graph, alpha=0.6, beta=1.0, attach_k=gm_attach_k)
    tfit_gminst = perf_counter() - t0
    t0 = perf_counter()
    y_gminst = gm_inst.predict(Xte, nC)
    tpred_gminst = perf_counter() - t0

    t0 = perf_counter()
    gm_cent = gm_no_edges_centroids(Xtr, ytr, n_classes=nC, alpha=0.0)
    tfit_gmcent = perf_counter() - t0
    t0 = perf_counter()
    y_gmcent = gm_cent.predict(Xte, nC)
    tpred_gmcent = perf_counter() - t0

    # Accuracies
    acc_lin = accuracy_score(yte, y_lin)
    acc_knn = accuracy_score(yte, y_knn)
    acc_lp  = accuracy_score(yte, y_lp)
    acc_gm     = accuracy_score(yte, y_gm)
    acc_gmflat = accuracy_score(yte, y_gmflat)
    acc_gminst = accuracy_score(yte, y_gminst)
    acc_gmcent = accuracy_score(yte, y_gmcent)

    # Collect rows
    rows_here = [
        dict(tag=tag, method="Linear", acc=acc_lin, t_fit=tfit_lin, t_pred=tpred_lin, n_train=ntr, n_test=nte, K="-", n_classes=nC),
        dict(tag=tag, method="kNN", acc=acc_knn, t_fit=tfit_knn, t_pred=tpred_knn, n_train=ntr, n_test=nte, K="-", n_classes=nC),
        dict(tag=tag, method="LP", acc=acc_lp, t_fit=tfit_lp, t_pred=tpred_lp, n_train=ntr, n_test=nte, K="-", n_classes=nC),
        dict(tag=tag, method="GM", acc=acc_gm, t_fit=tfit_gm, t_pred=tpred_gm, n_train=ntr, n_test=nte, K=Kproto, n_classes=nC),
        dict(tag=tag, method="GM-flat", acc=acc_gmflat, t_fit=tfit_gmflat, t_pred=tpred_gmflat, n_train=ntr, n_test=nte, K=Kproto, n_classes=nC),
        dict(tag=tag, method="GM-instance", acc=acc_gminst, t_fit=tfit_gminst, t_pred=tpred_gminst, n_train=ntr, n_test=nte, K=len(Xtr), n_classes=nC),
        dict(tag=tag, method="GM-centroid", acc=acc_gmcent, t_fit=tfit_gmcent, t_pred=tpred_gmcent, n_train=ntr, n_test=nte, K=nC, n_classes=nC),
    ]

    # Budget-matched kNN (|train| = #prototypes)
    if _ARGS.budget_match:
        P = Kproto
        rng = np.random.RandomState(seed + 20)
        keep_tr = stratified_quota_indices(y=ytr, total=P, rng=rng, min_each=1)
        Xsub_tr, ysub_tr = Xtr[keep_tr], ytr[keep_tr]
        t0 = perf_counter()
        knn_b = baseline_knn(k=10)
        knn_b.fit(Xsub_tr, ysub_tr)
        tfit_knnb = perf_counter() - t0
        t0 = perf_counter()
        y_knnb = knn_b.predict(Xte)
        tpred_knnb = perf_counter() - t0
        acc_knnb = accuracy_score(yte, y_knnb)
        rows_here.append(
            dict(tag=tag, method=f"kNN-budget(P={P})", acc=acc_knnb,
                 t_fit=tfit_knnb, t_pred=tpred_knnb, n_train=len(Xsub_tr), n_test=nte, K=P, n_classes=nC)
        )

    all_rows.extend(rows_here)

    # Plots
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    plot_decision(axs[0,0], lambda Z: baseline_linear().fit(Xtr, ytr).predict(Z), Xtr, ytr, Xte, yte, f"Linear ({acc_lin:.3f})", nC)
    plot_decision(axs[0,1], lambda Z: baseline_knn(k=10).fit(Xtr, ytr).predict(Z), Xtr, ytr, Xte, yte, f"kNN ({acc_knn:.3f})", nC)
    plot_decision(
        axs[0,2],
        lambda Z: lp_labels_on_grid(Xtr, ytr, Z, n_neighbors=lp_n_neighbors),
        Xtr, ytr, Xte, yte, f"LP ({acc_lp:.3f})", nC
    )
    plot_decision(axs[1,0], lambda Z: gm.predict(Z, nC),       Xtr, ytr, Xte, yte, f"GM ({acc_gm:.3f})", nC)
    plot_decision(axs[1,1], lambda Z: gm_flat.predict(Z, nC),  Xtr, ytr, Xte, yte, f"GM-flat ({acc_gmflat:.3f})", nC)
    plot_decision(axs[1,2], lambda Z: gm_cent.predict(Z, nC),  Xtr, ytr, Xte, yte, f"GM-centroid ({acc_gmcent:.3f})", nC)
    plt.suptitle(tag, fontsize=11)
    ensure_dir(plots_dir)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(os.path.join(plots_dir, f"{tag}.png"), dpi=220)
    plt.close()

def run_suite(dataset: str, outdir: str, seed: int):
    ensure_dir(outdir)
    set_seed(seed)
    all_rows: List[Dict] = []

    plots_dir = os.path.join(outdir, "plots")
    ensure_dir(plots_dir)

    # 2-class (fixed, non-linear)
    console.print(Panel.fit(f"[bold]Dataset:[/bold] {dataset} — [cyan]2-class (fixed non-linear)[/cyan]"))
    X, y = _make_dataset(dataset, seed, mode="2class")
    tag = f"{dataset}_2class_fixed"
    _run_block(tag, X, y, seed, all_rows, plots_dir, gm_knn_graph=10, lp_n_neighbors=10, gm_attach_k=10)

    # Multiclass (fixed, non-linear)
    console.print(Panel.fit(f"[bold]Dataset:[/bold] {dataset} — [cyan]Multiclass (fixed non-linear)[/cyan]"))
    X, y = _make_dataset(dataset, seed+101, mode="multiclass")
    tag = f"{dataset}_multiclass_fixed"
    _run_block(tag, X, y, seed+101, all_rows, plots_dir, gm_knn_graph=12, lp_n_neighbors=15, gm_attach_k=10)

    # Save CSV/JSON
    csv_path = os.path.join(outdir, f"{dataset}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag","method","acc","t_fit","t_pred","n_train","n_test","K","n_classes"])
        for r in all_rows:
            wr.writerow([r["tag"], r["method"], f'{r["acc"]:.6f}', f'{r["t_fit"]:.8f}', f'{r["t_pred"]:.8f}',
                         r["n_train"], r["n_test"], r.get("K","-"), r["n_classes"]])

    with open(os.path.join(outdir, f"{dataset}_summary.json"), "w") as jf:
        json.dump(all_rows, jf, indent=2)

    console.print(_rows_to_rich_table(all_rows))
    console.print(Panel.fit(
        f"[bold green]Saved[/bold green] CSV → [white]{csv_path}[/white]\n"
        f"[bold green]Plots[/bold green] → [white]{Path(plots_dir).resolve()}[/white]"
    ))

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["moons", "circles"], required=True,
                    help="Only non-linear datasets kept.")
    ap.add_argument("--outdir", type=str, default="runs_toy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imbalance", type=float, default=1.0,
                    help="Long-tail ratio (max/min) applied to TRAIN only. >1 enables long-tail.")
    ap.add_argument("--budget-match", action="store_true",
                    help="Also run kNN with training size equal to GM's #prototypes (memory budget).")
    global _ARGS
    _ARGS = ap.parse_args()
    with section(f"Running suite: { _ARGS.dataset } (long-tail={_ARGS.imbalance}  budget={_ARGS.budget_match})"):
        run_suite(_ARGS.dataset, _ARGS.outdir, _ARGS.seed)

if __name__ == "__main__":
    _ARGS = None
    main()
