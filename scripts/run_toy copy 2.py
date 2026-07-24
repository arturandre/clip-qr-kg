#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy experiments for Graph Memory (GM) with full reliability implementation.

- Datasets: blobs, moons
- Sweeps:
  * 2-class with increasing complexity (blobs: tighter std; moons: decreasing noise)
  * multi-class (blobs: more centers; moons: more moon pairs => 2*pairs classes)
- Baselines: Linear, kNN, Label Propagation (LabelSpreading)
- GM + Degenerates: instance-prototypes (DkNN-like), no-edges (nearest-centroid), flat-propagation (LP-like)
- Full reliability r_c = sigmoid( λ1 s_c + λ2 \bar{m}_c - λ3 ρ_c - λ4 \bar{v}_c + λ5 π_c )

Usage:
  python run_toy.py --dataset moons --outdir runs_toy
  python run_toy.py --dataset blobs --outdir runs_toy
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
from typing import Tuple

from sklearn.datasets import make_blobs, make_moons
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
    """Return array of same shape mapped by sigma((x-med)/IQR).
       IQR fallback to MAD/const if degenerate."""
    x = np.asarray(values).astype(np.float64)
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = max(q75 - q25, 1e-12)
    z = (x - med) / iqr
    return sigmoid(z)

# ---------------------------
# Data generators
# ---------------------------

def gen_blobs(n_samples=2000, n_classes=2, cluster_std=1.0, seed=0):
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2,
                      cluster_std=cluster_std, random_state=seed)
    return X, y

def gen_moons_pairs(n_samples=2000, n_pairs=1, noise=0.2, gap=6.0, seed=0):
    """Build multiple non-overlapping moon pairs; classes = 2 * n_pairs."""
    rng = np.random.RandomState(seed)
    per_pair = n_samples // n_pairs
    Xs, ys = [], []
    for p in range(n_pairs):
        Xm, Ym = make_moons(n_samples=per_pair, noise=noise, random_state=rng.randint(10_000))
        tx = (p % 3) * gap
        ty = (p // 3) * gap
        Xm = Xm + np.array([tx, ty])[None, :]
        Ym = Ym + 2*p
        Xs.append(Xm); ys.append(Ym)
    return np.vstack(Xs), np.concatenate(ys)

# ---------------------------
# Reliability metrics (full)
# ---------------------------

def compute_cluster_assignments(P, X):
    """Nearest prototype index for each sample."""
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
    """s_c in [0,1]: average over cluster of (sil(i)+1)/2.
       sil(i) = (b(i)-a(i)) / max(a(i), b(i)) with a(i): intra, b(i): nearest other cluster."""
    # Precompute per-cluster member lists
    idxs = [np.where(assign == c)[0] for c in range(K)]
    # Precompute cluster means for b(i) approximation: we'll use average distance to each cluster
    s_c = np.zeros(K, dtype=float)
    for c in range(K):
        I = idxs[c]
        if len(I) <= 1:
            s_c[c] = 0.5  # neutral
            continue
        Xi = X[I]
        # Intra distances a(i)
        # pairwise distances within Xi
        d_within = np.sqrt(((Xi[:, None, :] - Xi[None, :, :]) ** 2).sum(-1) + 1e-12)
        # exclude self by using (sum - 0) / (n-1)
        a_i = (d_within.sum(axis=1)) / max(1, (len(I)-1))
        # Nearest other cluster average distance b(i)
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
    """v_c = mean squared distance to centroid (unbounded)."""
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
    """m_c = min distance to other-class prototype center (unbounded)."""
    K = len(P)
    m = np.zeros(K, dtype=float)
    for c in range(K):
        # distances to all prototypes of different dominant class
        mask = (yc != yc[c])
        if not np.any(mask):
            m[c] = 0.0
            continue
        d = np.linalg.norm(P[mask] - P[c][None, :], axis=1)
        m[c] = float(d.min()) if len(d) else 0.0
    return m

def instability(X, assign, P, sigma_noise=0.05, trials=3):
    """ρ_c = fraction of samples re-assigned to a different prototype under small Gaussian perturbations."""
    K = len(P)
    n = len(X)
    changed = np.zeros(K, dtype=float)
    counts  = np.zeros(K, dtype=float)
    nbrs = NearestNeighbors(n_neighbors=1).fit(P)
    for t in range(trials):
        Xd = X + np.random.normal(scale=sigma_noise, size=X.shape)
        nn = nbrs.kneighbors(Xd, return_distance=False).ravel()
        # count per cluster
        for i in range(n):
            c0 = assign[i]
            counts[c0] += 1.0
            if nn[i] != c0:
                changed[c0] += 1.0
    rho = np.zeros(K, dtype=float)
    mask = counts > 0
    rho[mask] = changed[mask] / counts[mask]
    return rho

def reliability_full(s_c, m_c, v_c, rho_c, pi_c,
                     lam=(1,1,1,1,1)):
    """Apply robust normalization to m_c and v_c only, then combine."""
    lam1, lam2, lam3, lam4, lam5 = lam
    m_bar = robust_sigmoid_normalize(m_c)   # [0,1]
    v_bar = robust_sigmoid_normalize(v_c)   # [0,1]
    # s_c, rho_c, pi_c already in [0,1]
    z = (lam1 * s_c
         + lam2 * m_bar
         - lam3 * rho_c
         - lam4 * v_bar
         + lam5 * pi_c)
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
    use_norm: bool = False  # set True only for cosine encoders
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

        # full reliability
        s = silhouette_normalized(X, assign, K)
        v = dispersion(X, assign, P, K)
        m = margins(P, yc)
        rho = instability(X, assign, P, sigma_noise=self.cfg.instab_sigma, trials=self.cfg.instab_trials)
        r = reliability_full(s, m, v, rho, pi,
                             lam=(self.cfg.lam1, self.cfg.lam2, self.cfg.lam3, self.cfg.lam4, self.cfg.lam5))

        # graph
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
        M = np.linalg.inv(I - self.cfg.alpha * self.S)  # (I - αS)^-1
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
# Degenerate variants (use SAME reliability machinery where applicable)
# ---------------------------

def gm_instance_prototypes(Xtr, ytr, knn_graph=10, alpha=0.6, beta=1.0, attach_k=3):
    """Each sample is a prototype; reliability defaults to 1 to emulate DkNN spirit."""
    gm = GraphMemory(GMConfig(K=len(Xtr), knn_graph=knn_graph, alpha=alpha, beta=beta, attach_k=attach_k))
    gm.P = Xtr.copy()
    gm.yc = ytr.copy()
    gm.pi = np.ones(len(Xtr))
    gm.r  = np.ones(len(Xtr))  # instance "prototypes": no region stats
    A = kneighbors_graph(Xtr, n_neighbors=min(knn_graph, len(Xtr)-1),
                         mode="distance", include_self=False).toarray()
    A = np.exp(-beta * (A ** 2))
    A = np.maximum(A, A.T)
    D = A.sum(axis=1, keepdims=True) + 1e-12
    gm.S = A / D
    return gm

def gm_no_edges_centroids(Xtr, ytr, n_classes, alpha=0.0):
    """Nearest-class-mean: one prototype per class; r_c=1; α=0 (no diffusion)."""
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
    """Same prototypes as GM, but r_c ≡ 1 → LP-like propagation."""
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
    # Practical steady-state LP (knn kernel); sklearn handles the graph internally.
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

# ---------------------------
# Experiment runners
# ---------------------------

def prep_split(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte

def run_suite(dataset: str, outdir: str, seed: int):
    ensure_dir(outdir)
    set_seed(seed)
    results = []

    if dataset == "blobs":
        two_class_settings = [(2, s) for s in (1.6, 1.2, 0.9, 0.6)]
        multiclass_settings = [3, 5, 8]   # classes
    elif dataset == "moons":
        two_class_settings = [(1, n) for n in (0.35, 0.25, 0.18, 0.12)]  # pairs=1; decreasing noise
        multiclass_settings = [2, 3, 4]   # pairs => classes=2*pairs
    else:
        raise ValueError("--dataset must be 'blobs' or 'moons'")

    # 2-class sweep
    for idx, setting in enumerate(two_class_settings):
        if dataset == "blobs":
            n_classes, std = setting
            X, y = gen_blobs(n_samples=2000, n_classes=n_classes, cluster_std=std, seed=seed+idx)
            tag = f"blobs_2class_std{std}"
        else:
            n_pairs, noise = setting
            X, y = gen_moons_pairs(n_samples=2000, n_pairs=n_pairs, noise=noise, seed=seed+idx)
            # keep only first pair for 2-class
            mask = y < 2
            X, y = X[mask], y[mask]
            y = y - y.min()
            tag = f"moons_2class_noise{noise}"

        Xtr, Xte, ytr, yte = prep_split(X, y, seed)
        nC = len(np.unique(y))

        # Baselines
        lin = baseline_linear().fit(Xtr, ytr)
        knn = baseline_knn(k=10).fit(Xtr, ytr)
        lp  = baseline_lp(n_neighbors=10).fit(np.r_[Xtr, Xte], np.r_[ytr, -np.ones_like(yte)])

        # GM + degenerates (full r_c in GM)
        Kproto = min(32, max(8, len(Xtr)//8))
        gm = GraphMemory(GMConfig(K=Kproto, knn_graph=10, attach_k=3, alpha=0.6, beta=1.0)).fit(Xtr, ytr)
        gm_flat = gm_flat_propagation(Xtr, ytr, K=Kproto, knn_graph=10, alpha=0.6, beta=1.0)
        gm_inst = gm_instance_prototypes(Xtr, ytr, knn_graph=10, alpha=0.6, beta=1.0, attach_k=3)
        gm_cent = gm_no_edges_centroids(Xtr, ytr, n_classes=nC, alpha=0.0)

        # Accuracies
        acc_lin = accuracy_score(yte, lin.predict(Xte))
        acc_knn = accuracy_score(yte, knn.predict(Xte))
        acc_lp  = accuracy_score(yte, lp.transduction_[-len(yte):])
        acc_gm     = accuracy_score(yte, gm.predict(Xte, nC))
        acc_gmflat = accuracy_score(yte, gm_flat.predict(Xte, nC))
        acc_gminst = accuracy_score(yte, gm_inst.predict(Xte, nC))
        acc_gmcent = accuracy_score(yte, gm_cent.predict(Xte, nC))

        for name, acc in [("Linear", acc_lin), ("kNN", acc_knn), ("LP", acc_lp),
                          ("GM", acc_gm), ("GM-flat", acc_gmflat),
                          ("GM-instance", acc_gminst), ("GM-centroid", acc_gmcent)]:
            results.append((tag, name, acc))

        # Plots
        fig, axs = plt.subplots(2, 3, figsize=(11, 7))
        plot_decision(axs[0,0], lambda Z: lin.predict(Z), Xtr, ytr, Xte, yte, f"Linear ({acc_lin:.3f})", nC)
        plot_decision(axs[0,1], lambda Z: knn.predict(Z), Xtr, ytr, Xte, yte, f"kNN ({acc_knn:.3f})", nC)
        plot_decision(axs[0,2], lambda Z: lp.predict(Z),   Xtr, ytr, Xte, yte, f"LP ({acc_lp:.3f})", nC)
        plot_decision(axs[1,0], lambda Z: gm.predict(Z, nC),       Xtr, ytr, Xte, yte, f"GM ({acc_gm:.3f})", nC)
        plot_decision(axs[1,1], lambda Z: gm_flat.predict(Z, nC),  Xtr, ytr, Xte, yte, f"GM-flat ({acc_gmflat:.3f})", nC)
        plot_decision(axs[1,2], lambda Z: gm_cent.predict(Z, nC),  Xtr, ytr, Xte, yte, f"GM-centroid ({acc_gmcent:.3f})", nC)
        plt.suptitle(tag, fontsize=11)
        ensure_dir(os.path.join(outdir, "plots"))
        plt.tight_layout(rect=[0,0,1,0.97])
        plt.savefig(os.path.join(outdir, "plots", f"{tag}.png"), dpi=220)
        plt.close()

    # Multi-class sweep
    for idx, nlev in enumerate(multiclass_settings):
        if dataset == "blobs":
            X, y = gen_blobs(n_samples=3000, n_classes=nlev, cluster_std=1.0, seed=seed+100+idx)
            tag = f"blobs_multiclass_C{nlev}"
            nC = nlev
        else:
            X, y = gen_moons_pairs(n_samples=3000, n_pairs=nlev, noise=0.2, seed=seed+100+idx)
            tag = f"moons_multiclass_pairs{nlev}"
            nC = 2*nlev

        Xtr, Xte, ytr, yte = prep_split(X, y, seed)

        lin = baseline_linear().fit(Xtr, ytr)
        knn = baseline_knn(k=10).fit(Xtr, ytr)
        lp  = baseline_lp(n_neighbors=15).fit(np.r_[Xtr, Xte], np.r_[ytr, -np.ones_like(yte)])

        Kproto = min(8*nC, max(2*nC, len(Xtr)//12))
        gm = GraphMemory(GMConfig(K=Kproto, knn_graph=12, attach_k=3, alpha=0.6, beta=1.0)).fit(Xtr, ytr)
        gm_flat = gm_flat_propagation(Xtr, ytr, K=Kproto, knn_graph=12, alpha=0.6, beta=1.0)
        gm_inst = gm_instance_prototypes(Xtr, ytr, knn_graph=12, alpha=0.6, beta=1.0, attach_k=3)
        gm_cent = gm_no_edges_centroids(Xtr, ytr, n_classes=nC, alpha=0.0)

        acc_lin = accuracy_score(yte, lin.predict(Xte))
        acc_knn = accuracy_score(yte, knn.predict(Xte))
        acc_lp  = accuracy_score(yte, lp.transduction_[-len(yte):])
        acc_gm     = accuracy_score(yte, gm.predict(Xte, nC))
        acc_gmflat = accuracy_score(yte, gm_flat.predict(Xte, nC))
        acc_gminst = accuracy_score(yte, gm_inst.predict(Xte, nC))
        acc_gmcent = accuracy_score(yte, gm_cent.predict(Xte, nC))

        for name, acc in [("Linear", acc_lin), ("kNN", acc_knn), ("LP", acc_lp),
                          ("GM", acc_gm), ("GM-flat", acc_gmflat),
                          ("GM-instance", acc_gminst), ("GM-centroid", acc_gmcent)]:
            results.append((tag, name, acc))

        fig, axs = plt.subplots(2, 3, figsize=(11, 7))
        plot_decision(axs[0,0], lambda Z: lin.predict(Z), Xtr, ytr, Xte, yte, f"Linear ({acc_lin:.3f})", nC)
        plot_decision(axs[0,1], lambda Z: knn.predict(Z), Xtr, ytr, Xte, yte, f"kNN ({acc_knn:.3f})", nC)
        plot_decision(axs[0,2], lambda Z: lp.predict(Z),   Xtr, ytr, Xte, yte, f"LP ({acc_lp:.3f})", nC)
        plot_decision(axs[1,0], lambda Z: gm.predict(Z, nC),       Xtr, ytr, Xte, yte, f"GM ({acc_gm:.3f})", nC)
        plot_decision(axs[1,1], lambda Z: gm_flat.predict(Z, nC),  Xtr, ytr, Xte, yte, f"GM-flat ({acc_gmflat:.3f})", nC)
        plot_decision(axs[1,2], lambda Z: gm_cent.predict(Z, nC),  Xtr, ytr, Xte, yte, f"GM-centroid ({acc_gmcent:.3f})", nC)
        plt.suptitle(tag, fontsize=11)
        ensure_dir(os.path.join(outdir, "plots"))
        plt.tight_layout(rect=[0,0,1,0.97])
        plt.savefig(os.path.join(outdir, "plots", f"{tag}.png"), dpi=220)
        plt.close()

    # Save CSV
    csv_path = os.path.join(outdir, f"{dataset}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag", "method", "acc"])
        for row in results:
            wr.writerow(row)
    print(f"[OK] Saved summary to {csv_path}")
    print(f"[OK] Plots in {os.path.join(outdir, 'plots')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["blobs", "moons"], required=True)
    ap.add_argument("--outdir", type=str, default="runs_toy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_suite(args.dataset, args.outdir, args.seed)

if __name__ == "__main__":
    main()
