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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# ================================================================
# Utilities
# ================================================================

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

def gen_circles(n_samples=2000, seed=0, factor=0.4):
    X, y = make_circles(n_samples=n_samples, factor=factor, noise=0.0, random_state=seed)
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

@dataclass
class GMConfig:
    K: int
    knn_graph: int = 10
    attach_k: int = 3
    alpha: float = 0.6
    beta: float = 1.0
    use_norm: bool = False
    # reliability weights could be exposed here if you want

class GraphMemory:
    """Prototype graph with reliability-weighted diffusion."""
    def __init__(self, cfg: GMConfig):
        self.cfg = cfg
        self.P = None      # [K, d]
        self.yc = None     # prototype dominant class
        self.r  = None     # reliability per prototype
        self.S  = None     # row-stochastic transition among prototypes

    def fit(self, X, y):
        K = min(self.cfg.K, len(X))
        km = KMeans(n_clusters=K, n_init="auto", random_state=0).fit(X)
        P, assign = km.cluster_centers_, km.labels_

        nC = len(np.unique(y))
        yc, pi = prototype_purity(assign, y, K, nC)
        s = silhouette_normalized(X, assign, K)
        v = dispersion(X, assign, P, K)
        m = margins(P, yc)
        rho = instability(X, assign, P)
        r = reliability_full(s, m, v, rho, pi)

        Pn = l2n(P) if self.cfg.use_norm else P
        A = kneighbors_graph(Pn, n_neighbors=min(self.cfg.knn_graph, K-1),
                             mode="distance", include_self=False).toarray()
        A = np.exp(-self.cfg.beta * (A**2))
        A = np.maximum(A, A.T)
        D = A.sum(axis=1, keepdims=True) + 1e-12
        S = A / D

        self.P, self.yc, self.r, self.S = P, yc, r, S
        return self

    def _z0(self, xq):
        d = np.linalg.norm(self.P - xq, axis=1)
        idx = np.argsort(d)[:min(self.cfg.attach_k, len(self.P))]
        w = np.exp(-self.cfg.beta * (d[idx]**2)) * (self.r[idx] + 1e-12)
        z = np.zeros(len(self.P))
        z[idx] = w / (w.sum() + 1e-12)
        return z

    def predict_proba(self, Xq, n_classes):
        I = np.eye(len(self.P))
        M = np.linalg.inv(I - self.cfg.alpha * self.S)  # diffusion kernel
        probs = np.zeros((len(Xq), n_classes))
        for i, x in enumerate(Xq):
            z0 = self._z0(x)
            z = (1 - self.cfg.alpha) * (M @ z0)
            for c in range(n_classes):
                probs[i, c] = z[self.yc == c].sum()
            probs[i] /= probs[i].sum() + 1e-12
        return probs

    def predict(self, Xq, n_classes):
        return np.argmax(self.predict_proba(Xq, n_classes), axis=1)

# ================================================================
# GM degenerates
# ================================================================

def gm_flat_from(gm: GraphMemory) -> GraphMemory:
    """Set all reliabilities to 1, keep same graph/prototypes."""
    flat = GraphMemory(gm.cfg)
    flat.P, flat.yc, flat.S = gm.P.copy(), gm.yc.copy(), gm.S.copy()
    flat.r = np.ones_like(gm.yc, dtype=float)
    return flat

def gm_instance(Xtr, ytr, knn_graph=10, alpha=0.6, beta=1.0, attach_k=3):
    """Prototypes = instances; graph built over all instances."""
    gm = GraphMemory(GMConfig(K=len(Xtr), knn_graph=knn_graph, alpha=alpha, beta=beta, attach_k=attach_k))
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

def _row_stochastic_knn(X, k=10, beta=1.0):
    A = kneighbors_graph(X, n_neighbors=min(k, len(X)-1),
                         mode="distance", include_self=False).toarray()
    W = np.exp(-beta * (A**2))
    W = np.maximum(W, W.T)
    D = W.sum(1, keepdims=True) + 1e-12
    return W / D  # S

def lp_harmonic_predict(XL, yL, XU, n_classes, k=10, beta=1.0, alpha=0.2):
    """
    Closed-form harmonic LP (Zhu & Ghahramani):
        (I - α S_UU) F_U = α S_UL Y_L
    where Y_L is one-hot of labeled nodes, labels are clamped.
    """
    Xall = np.r_[XL, XU]
    S = _row_stochastic_knn(Xall, k=k, beta=beta)
    nL, nU = len(XL), len(XU)
    S_UL, S_UU = S[nL:, :nL], S[nL:, nL:]

    YL = np.zeros((nL, n_classes))
    YL[np.arange(nL), yL] = 1.0
    IUU = np.eye(nU)
    F_U = np.linalg.solve(IUU - alpha * S_UU, alpha * (S_UL @ YL))
    F_U /= F_U.sum(1, keepdims=True) + 1e-12
    yhat = np.argmax(F_U, axis=1)
    return yhat, F_U

def gm_as_lp(Xtr, ytr, Xte, n_classes, k=10, beta=1.0, alpha=0.2):
    """Degenerate GM → instance-graph harmonic LP."""
    yhat, _ = lp_harmonic_predict(Xtr, ytr, Xte, n_classes, k, beta, alpha)
    return yhat

# ================================================================
# Baselines
# ================================================================

def baseline_linear():  return LogisticRegression(max_iter=500)
def baseline_knn(k=10): return KNeighborsClassifier(n_neighbors=k)
def baseline_lp(n_neighbors=10): return LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=0.2, max_iter=50)

# ================================================================
# Plotting
# ================================================================

def plot_decision(ax, predict_fn, Xtr, ytr, Xte, yte, title):
    x_min, x_max = np.r_[Xtr[:,0], Xte[:,0]].min()-1, np.r_[Xtr[:,0], Xte[:,0]].max()+1
    y_min, y_max = np.r_[Xtr[:,1], Xte[:,1]].min()-1, np.r_[Xtr[:,1], Xte[:,1]].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,300), np.linspace(y_min,y_max,300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)
    cmap_bg = ListedColormap(["#f0f9ff","#fff7ed"])
    cmap_pts= ListedColormap(["#0284c7","#ea580c"])
    ax.contourf(xx,yy,Z,alpha=0.35,cmap=cmap_bg,levels=[-0.5,0.5,1.5])
    ax.scatter(Xtr[:,0],Xtr[:,1],c=ytr,s=12,alpha=0.7,cmap=cmap_pts,edgecolors="k",linewidths=0.2)
    ax.scatter(Xte[:,0],Xte[:,1],c=yte,s=18,alpha=0.9,cmap=cmap_pts,marker="x")
    ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([])

def lp_harmonic_on_grid(Xtr, ytr, Z, n_neighbors=10, alpha=0.2):
    """Predict labels for grid points Z via harmonic LP (treat Z as unlabeled)."""
    yhat, _ = lp_harmonic_predict(Xtr, ytr, Z, n_classes=2, k=n_neighbors, beta=1.0, alpha=alpha)
    return yhat

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
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    nC = 2
    if imbalance > 1.0:
        Xtr, ytr = apply_long_tail(Xtr, ytr, imbalance, seed)

    results = []
    plots_dir = Path(outdir) / "plots"
    ensure_dir(plots_dir)

    # ---------- Baselines ----------
    y_lin, tfit_lin, tpred_lin = time_fit_predict(baseline_linear(), Xtr, ytr, Xte)
    y_knn, tfit_knn, tpred_knn = time_fit_predict(baseline_knn(k=10), Xtr, ytr, Xte)

    # sklearn LabelSpreading (transductive over train+test)
    t0 = perf_counter()
    X_all = np.r_[Xtr, Xte]
    y_all = np.r_[ytr, -np.ones_like(yte)]
    lp = baseline_lp(n_neighbors=10)
    lp.fit(X_all, y_all)
    tfit_lps = perf_counter() - t0
    t0 = perf_counter()
    y_lps = lp.transduction_[-len(yte):]
    tpred_lps = perf_counter() - t0

    # Our harmonic LP (transductive over train+test)
    t0 = perf_counter()
    y_lph, _ = lp_harmonic_predict(Xtr, ytr, Xte, n_classes=nC, k=10, beta=1.0, alpha=0.2)
    t_lph = perf_counter() - t0

    # ---------- GM & degenerates ----------
    # Pick K prototypes relative to training size (light but expressive)
    Kproto = max(8, min(32, len(Xtr)//10))
    gm_cfg = GMConfig(K=Kproto, knn_graph=10, attach_k=10, alpha=0.2, beta=1.0)
    t0 = perf_counter(); gm = GraphMemory(gm_cfg).fit(Xtr, ytr); tfit_gm = perf_counter()-t0
    t0 = perf_counter(); y_gm = gm.predict(Xte, nC); tpred_gm = perf_counter()-t0

    gm_flat = gm_flat_from(gm)
    t0 = perf_counter(); y_gmflat = gm_flat.predict(Xte, nC); tpred_gmflat = perf_counter()-t0
    tfit_gmflat = 0.0  # reuse gm fit

    t0 = perf_counter(); gmi = gm_instance(Xtr, ytr, knn_graph=10, alpha=0.2, beta=1.0, attach_k=10); tfit_gmi=perf_counter()-t0
    t0 = perf_counter(); y_gmi = gmi.predict(Xte, nC); tpred_gmi=perf_counter()-t0

    t0 = perf_counter(); gmc = gm_centroid(Xtr, ytr); tfit_gmc=perf_counter()-t0
    t0 = perf_counter(); y_gmc = gmc.predict(Xte, nC); tpred_gmc=perf_counter()-t0

    # GM→LP(instance) — harmonic LP on instance graph
    t0 = perf_counter(); y_gm_as_lp = gm_as_lp(Xtr, ytr, Xte, n_classes=nC, k=10, beta=1.0, alpha=0.2); t_gm_as_lp = perf_counter()-t0

    # ---------- Accuracies ----------
    acc_lin  = accuracy_score(yte, y_lin)
    acc_knn  = accuracy_score(yte, y_knn)
    acc_lps  = accuracy_score(yte, y_lps)
    acc_lph  = accuracy_score(yte, y_lph)
    acc_gm   = accuracy_score(yte, y_gm)
    acc_gmf  = accuracy_score(yte, y_gmflat)
    acc_gmi  = accuracy_score(yte, y_gmi)
    acc_gmc  = accuracy_score(yte, y_gmc)
    acc_gmlp = accuracy_score(yte, y_gm_as_lp)

    rows = [
        dict(tag=tag, method="Linear",         acc=acc_lin,  t_fit=tfit_lin,  t_pred=tpred_lin,  n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="kNN",            acc=acc_knn,  t_fit=tfit_knn,  t_pred=tpred_knn,  n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="LabelSpreading", acc=acc_lps,  t_fit=tfit_lps,  t_pred=tpred_lps,  n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="LP(harmonic)",   acc=acc_lph,  t_fit=0.0,       t_pred=t_lph,      n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="GM",             acc=acc_gm,   t_fit=tfit_gm,   t_pred=tpred_gm,   n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method="GM-flat",        acc=acc_gmf,  t_fit=tfit_gmflat,t_pred=tpred_gmflat,n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method="GM-instance",    acc=acc_gmi,  t_fit=tfit_gmi,  t_pred=tpred_gmi,  n_train=len(Xtr), n_test=len(Xte), K=len(Xtr)),
        dict(tag=tag, method="GM-centroid",    acc=acc_gmc,  t_fit=tfit_gmc,  t_pred=tpred_gmc,  n_train=len(Xtr), n_test=len(Xte), K=2),
        dict(tag=tag, method="GM→LP(instance)",acc=acc_gmlp, t_fit=0.0,       t_pred=t_gm_as_lp, n_train=len(Xtr), n_test=len(Xte), K=len(Xtr)),
    ]
    results.extend(rows)

    # ---------- Optional budget-matched kNN ----------
    if budget_match:
        P = Kproto
        rng = np.random.RandomState(seed + 20)
        keep = stratified_quota_indices(ytr, total=P, rng=rng, min_each=1)
        Xsub, ysub = Xtr[keep], ytr[keep]
        t0 = perf_counter(); knnb = KNeighborsClassifier(n_neighbors=10).fit(Xsub, ysub); tfit_knnb = perf_counter()-t0
        t0 = perf_counter(); y_knnb = knnb.predict(Xte); tpred_knnb = perf_counter()-t0
        acc_knnb = accuracy_score(yte, y_knnb)
        results.append(dict(tag=tag, method=f"kNN-budget(P={P})", acc=acc_knnb,
                            t_fit=tfit_knnb, t_pred=tpred_knnb, n_train=len(Xsub), n_test=len(Xte), K=P))

    # ---------- Plots ----------
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))

    plot_decision(axs[0,0], lambda Z: baseline_linear().fit(Xtr, ytr).predict(Z), Xtr, ytr, Xte, yte, f"Linear ({acc_lin:.3f})")
    plot_decision(axs[0,1], lambda Z: baseline_knn(k=10).fit(Xtr, ytr).predict(Z),  Xtr, ytr, Xte, yte, f"kNN ({acc_knn:.3f})")
    plot_decision(axs[0,2], lambda Z: lp_harmonic_on_grid(Xtr, ytr, Z, n_neighbors=10, alpha=0.2), Xtr, ytr, Xte, yte, f"LP-harm ({acc_lph:.3f})")

    plot_decision(axs[1,0], lambda Z: gm.predict(Z, nC),       Xtr, ytr, Xte, yte, f"GM ({acc_gm:.3f})")
    plot_decision(axs[1,1], lambda Z: gm_flat.predict(Z, nC),  Xtr, ytr, Xte, yte, f"GM-flat ({acc_gmf:.3f})")
    plot_decision(axs[1,2], lambda Z: gm_centroid(Xtr, ytr).predict(Z, nC), Xtr, ytr, Xte, yte, f"GM-centroid ({acc_gmc:.3f})")

    plt.suptitle(tag, fontsize=11)
    ensure_dir(plots_dir)
    plt.tight_layout(rect=[0,0,1,0.97])
    fig_path = str(plots_dir / f"{tag}.png")
    plt.savefig(fig_path, dpi=220); plt.close()

    return results, fig_path

# ================================================================
# Suite
# ================================================================

def run_suite(dataset: str, outdir: str, seed: int, imbalance: float, budget_match: bool):
    ensure_dir(outdir)
    set_seed(seed)

    # Data
    if dataset == "moons":
        X, y = gen_moons(n_samples=2000, seed=seed, noise=0.15)
    elif dataset == "circles":
        X, y = gen_circles(n_samples=2000, seed=seed, factor=0.4)
    else:
        raise ValueError("--dataset must be 'moons' or 'circles'")

    tag = f"{dataset}_2class_fixed"
    rows, fig_path = run_block(tag, X, y, seed, outdir, imbalance, budget_match)

    # Save CSV/JSON
    csv_path = os.path.join(outdir, f"{dataset}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag","method","acc","t_fit","t_pred","n_train","n_test","K"])
        for r in rows:
            wr.writerow([r["tag"], r["method"], f'{r["acc"]:.6f}', f'{r["t_fit"]:.8f}',
                         f'{r["t_pred"]:.8f}', r["n_train"], r["n_test"], r.get("K","-")])

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
