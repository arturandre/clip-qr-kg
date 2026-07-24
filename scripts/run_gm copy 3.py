#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Memory (GM) — binary-only toy suite with inductive LabelSpreading.

Datasets:
  - make_moons
  - make_circles

Baselines:
  - Logistic Regression (Linear)
  - kNN
  - LabelSpreading (INDUCTIVE in sklearn: fit on TRAIN only, predict on TEST)

GM variants:
  - GM (reliability-weighted diffusion over prototype graph)
  - GM-flat (same graph, reliability r=1)
  - GM-instance (prototypes = instances, large K)
  - GM-centroid (one prototype per class, no edges)

Outputs:
  - <outdir>/<dataset>_summary.csv (acc + timing)
  - <outdir>/<dataset>_calibration.csv (acc, nll, brier, ece)
  - <outdir>/plots/<tag>.png (decision visuals)

Usage:
  python scripts/run_gm.py --dataset moons --outdir runs_toy --seed 0
  python scripts/run_gm.py --dataset circles --outdir runs_toy --seed 0 --imbalance 8
"""

import os, csv, json, argparse, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from contextlib import contextmanager

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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score

EPS = 1e-12

# =========================
# Utilities
# =========================

def set_seed(seed: int):
    np.random.seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def l2n(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def robust_sigmoid_normalize(values):
    x = np.asarray(values, float)
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = max(q75 - q25, 1e-12)
    z = (x - med) / iqr
    return sigmoid(z)

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
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    out = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf >= lo) & (conf < hi if i < n_bins-1 else conf <= hi)
        if not np.any(m): continue
        out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)

def all_metrics(probs, y):
    p = np.clip(probs, 1e-12, 1.0)
    return dict(
        acc=float((p.argmax(1) == y).mean()),
        nll=nll(p, y),
        brier=brier(p, y),
        ece=ece(p, y)
    )

@contextmanager
def section(msg: str):
    console.rule(f"[bold cyan]{msg}")
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        console.print(Panel.fit(f"[green]{msg}[/green] finished in [bold]{dt:.3f}s[/bold]"))

# =========================
# Data (binary only)
# =========================

def gen_moons(n_samples=2000, seed=0, noise=0.15):
    Xtr, ytr = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    Xte, yte = make_moons(n_samples=n_samples, noise=noise, random_state=seed+10)
    return Xtr, ytr, Xte, yte

def gen_circles(n_samples=2000, seed=0, factor=0.4, noise=0.0):
    Xtr, ytr = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=seed)
    Xte, yte = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=seed+10)
    return Xtr, ytr, Xte, yte

def apply_long_tail(Xtr, ytr, imbalance, seed):
    """Impose max/min ≈ imbalance over classes in TRAIN (binary)."""
    if imbalance <= 1.0:
        return Xtr, ytr
    rng = np.random.RandomState(seed + 10)
    classes = np.unique(ytr)
    counts = np.bincount(ytr, minlength=classes.max()+1)
    maxc = counts.max()
    targets = [int(round(maxc / (imbalance ** (i / max(1, len(classes)-1))))) for i in range(len(classes))]
    keep = []
    for c, tgt in zip(classes, targets):
        idx = np.where(ytr == c)[0]
        k = min(len(idx), max(5, tgt))
        keep.append(rng.choice(idx, size=k, replace=False))
    keep = np.concatenate(keep)
    return Xtr[keep], ytr[keep]

# =========================
# Reliability metrics (GM)
# =========================

def prototype_purity(assign, y, K, n_classes):
    yc, pi = np.zeros(K, int), np.zeros(K, float)
    for c in range(K):
        idx = np.where(assign == c)[0]
        if len(idx) == 0: continue
        hist = np.bincount(y[idx], minlength=n_classes)
        yc[c] = int(np.argmax(hist))
        pi[c] = float(hist.max()) / float(len(idx))
    return yc, pi

def margins(P, yc):
    K = len(P)
    m = np.zeros(K, float)
    for c in range(K):
        mask = (yc != yc[c])
        if not np.any(mask): continue
        d = np.linalg.norm(P[mask] - P[c], axis=1)
        m[c] = float(d.min())
    return m

# ---- Fast subsampled reliability pieces ----

def _cluster_subsample_indices(assign: np.ndarray, cap: int, seed: int = 123):
    K = int(assign.max()) + 1
    rng = np.random.default_rng(seed)
    subs = []
    for c in range(K):
        I = np.where(assign == c)[0]
        if I.size > cap:
            I = rng.choice(I, size=cap, replace=False)
        subs.append(I.astype(np.int32))
    return subs

def _silhouette_approx(X: np.ndarray, assign: np.ndarray, subs, K: int) -> np.ndarray:
    s_c = np.zeros(K, dtype=np.float32)
    for c in range(K):
        I = subs[c]
        if I.size <= 1:
            s_c[c] = 0.5; continue
        Xi = X[I]
        d_within = np.sqrt(((Xi[:, None, :] - Xi[None, :, :])**2).sum(-1) + 1e-8)
        a_i = d_within.mean(axis=1)
        b_i = np.full(I.size, np.inf, dtype=np.float32)
        for c2 in range(K):
            if c2 == c or subs[c2].size == 0: continue
            Xj = X[subs[c2]]
            d_cross = np.sqrt(((Xi[:, None, :] - Xj[None, :, :])**2).sum(-1) + 1e-8)
            b_i = np.minimum(b_i, d_cross.mean(axis=1))
        sil = (b_i - a_i) / (np.maximum(a_i, b_i) + 1e-8)
        s_c[c] = np.clip(((sil + 1.0) * 0.5).mean(), 0.0, 1.0)
    return s_c

def _dispersion_sub(X: np.ndarray, P: np.ndarray, subs, K: int) -> np.ndarray:
    v = np.zeros(K, dtype=np.float32)
    for c in range(K):
        I = subs[c]
        if I.size == 0: continue
        Xi = X[I]
        v[c] = float(((Xi - P[c][None, :])**2).sum(axis=1).mean())
    return v

# =========================
# Graph Memory (fast)
# =========================

@dataclass
class GMConfig:
    K: int
    knn_graph: int = 10
    attach_k: int = 3
    alpha: float = 0.6
    beta: float = 1.0
    use_norm: bool = False

    # speed knobs
    use_minibatch_kmeans: bool = True
    kmeans_batch_size: int = 8192
    dtype: str = "float32"
    reliability_sample_cap: int = 512
    diffusion_iters: int = 20   # 0 = inverse; >0 = iterative (recommended)

class GraphMemory:
    """Prototype graph with reliability-weighted diffusion (fast, scalable)."""
    def __init__(self, cfg: GMConfig):
        self.cfg = cfg
        self.P = None      # [K, d]
        self.yc = None     # [K]
        self.r  = None     # [K]
        self.S  = None     # [K, K]
        self._M = None     # cached inverse if diffusion_iters == 0
        self._nnP = None

    def _prepare(self):
        """Finalize internal structures after P/yc/r/S are set."""
        K = len(self.P)
        # clamp attach_k to [1, K]
        k_attach = max(1, min(self.cfg.attach_k, K))
        self._nnP = NearestNeighbors(n_neighbors=k_attach).fit(self.P)
        if self.cfg.diffusion_iters == 0:
            I = np.eye(K, dtype=np.float32)
            self._M = np.linalg.inv(I - self.cfg.alpha * self.S).astype(np.float32)
        else:
            self._M = None
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32 if self.cfg.dtype=="float32" else np.float64)
        y = np.asarray(y, dtype=np.int32)
        K = min(self.cfg.K, len(X))

        # KMeans
        if self.cfg.use_minibatch_kmeans:
            km = MiniBatchKMeans(
                n_clusters=K, batch_size=self.cfg.kmeans_batch_size,
                n_init="auto", random_state=0, reassignment_ratio=0.01
            ).fit(X)
        else:
            km = KMeans(n_clusters=K, n_init="auto", random_state=0).fit(X)

        P = km.cluster_centers_.astype(np.float32)
        assign = km.labels_.astype(np.int32)

        # Reliability ingredients
        nC = int(y.max()) + 1
        yc, pi = prototype_purity(assign, y, K, nC)
        subs = _cluster_subsample_indices(assign, self.cfg.reliability_sample_cap)
        s = _silhouette_approx(X, assign, subs, K).astype(np.float32)
        v = _dispersion_sub(X, P, subs, K).astype(np.float32)
        m = margins(P, yc).astype(np.float32)
        # normalized combination
        m_bar = robust_sigmoid_normalize(m)
        v_bar = robust_sigmoid_normalize(v)
        z = 1.0*s + 1.0*m_bar - 1.0*0.0 - 1.0*v_bar + 1.0*pi   # rho disabled (expensive)
        r = sigmoid(z).astype(np.float32)

        # Prototype graph
        Pn = l2n(P) if self.cfg.use_norm else P
        A = kneighbors_graph(Pn, n_neighbors=min(self.cfg.knn_graph, K-1),
                             mode="distance", include_self=False).toarray().astype(np.float32)
        A = np.exp(-self.cfg.beta * (A**2)).astype(np.float32)
        A = np.maximum(A, A.T)
        S = (A / (A.sum(axis=1, keepdims=True) + 1e-8)).astype(np.float32)

        self.P, self.yc, self.r, self.S = Pn, yc, r, S
        return self._prepare()
        # self._nnP = NearestNeighbors(n_neighbors=min(self.cfg.attach_k, K)).fit(self.P)

        # if self.cfg.diffusion_iters == 0:
        #     I = np.eye(K, dtype=np.float32)
        #     self._M = np.linalg.inv(I - self.cfg.alpha * self.S).astype(np.float32)
        # else:
        #     self._M = None
        # return self

    def _build_Z0(self, Xq: np.ndarray) -> np.ndarray:
        Xq = np.asarray(Xq, dtype=np.float32 if self.cfg.dtype=="float32" else np.float64)
        Q, K = len(Xq), len(self.P)
        # self._nnP already set with clamped k in _prepare()
        d, idx = self._nnP.kneighbors(Xq, return_distance=True)
        Z0 = np.zeros((K, Q), dtype=np.float32)
        for i in range(Q):
            w = np.exp(-self.cfg.beta * (d[i]**2)).astype(np.float32) * (self.r[idx[i]] + 1e-8)
            w = w / (w.sum() + 1e-8)
            Z0[idx[i], i] = w
        return Z0


    def predict_proba(self, Xq, n_classes: int) -> np.ndarray:
        if self._nnP is None:   # safety net (e.g., for degenerate builders)
          self._prepare()
        Z0 = self._build_Z0(Xq)  # [K, Q]
        if self.cfg.diffusion_iters == 0:
            Z = (1.0 - self.cfg.alpha) * (self._M @ Z0)
        else:
            Z = Z0.copy()
            a = self.cfg.alpha
            for _ in range(self.cfg.diffusion_iters):
                Z = a * (self.S @ Z) + (1.0 - a) * Z0
        Q = Z.shape[1]
        probs = np.zeros((Q, n_classes), dtype=np.float32)
        for c in range(n_classes):
            probs[:, c] = Z[self.yc == c, :].sum(axis=0)
        probs /= (probs.sum(axis=1, keepdims=True) + 1e-8)
        return probs

    def predict(self, Xq, n_classes: int) -> np.ndarray:
        return np.argmax(self.predict_proba(Xq, n_classes), axis=1)

# =========================
# GM degenerates
# =========================

def gm_flat_from(gm: GraphMemory) -> GraphMemory:
    flat = GraphMemory(gm.cfg)
    flat.P, flat.yc, flat.S = gm.P.copy(), gm.yc.copy(), gm.S.copy()
    flat.r = np.ones_like(gm.yc, dtype=np.float32)
    return flat._prepare()

def gm_instance(Xtr, ytr, knn_graph=10, alpha=0.6, beta=1.0, attach_k=3):
    gm = GraphMemory(GMConfig(K=len(Xtr), knn_graph=knn_graph, alpha=alpha, beta=beta, attach_k=attach_k))
    gm.P = Xtr.astype(np.float32).copy()
    gm.yc = ytr.astype(np.int32).copy()
    gm.r  = np.ones(len(Xtr), dtype=np.float32)
    A = kneighbors_graph(gm.P, n_neighbors=max(1, min(knn_graph, len(Xtr)-1)),
                         mode="distance", include_self=False).toarray().astype(np.float32)
    A = np.exp(-beta * (A**2)).astype(np.float32)
    A = np.maximum(A, A.T)
    gm.S = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    return gm._prepare()


def gm_centroid(Xtr, ytr):
    classes = np.unique(ytr)
    means = np.vstack([Xtr[ytr==c].mean(0) for c in classes]).astype(np.float32)
    gm = GraphMemory(GMConfig(K=len(means), knn_graph=1, alpha=0.0, attach_k=1))
    gm.P, gm.yc = means, classes.astype(np.int32)
    gm.r = np.ones(len(means), dtype=np.float32)
    gm.S = np.zeros((len(means), len(means)), dtype=np.float32)
    return gm._prepare()

# =========================
# Baselines & LS (inductive)
# =========================

def baseline_linear():  return LogisticRegression(max_iter=500)
def baseline_knn(k=10): return KNeighborsClassifier(n_neighbors=k)

def ls_proba_inductive(Xtr, ytr, Xte, n_neighbors=10, alpha=0.2):
    """Inductive sklearn LabelSpreading: fit on TRAIN only; predict on TEST."""
    ls = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=30)
    ls.fit(Xtr, ytr)
    return ls.predict_proba(Xte)

def probs_linear(Xtr, ytr, Xte):
    return baseline_linear().fit(Xtr, ytr).predict_proba(Xte)

def probs_knn(Xtr, ytr, Xte, k=10):
    return baseline_knn(k=k).fit(Xtr, ytr).predict_proba(Xte)

# =========================
# Plotting
# =========================

def plot_decision(ax, predict_fn, Xtr, ytr, Xte, yte, title):
    x_min, x_max = np.r_[Xtr[:,0], Xte[:,0]].min()-1, np.r_[Xtr[:,0], Xte[:,0]].max()+1
    y_min, y_max = np.r_[Xtr[:,1], Xte[:,1]].min()-1, np.r_[Xtr[:,1], Xte[:,1]].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,300), np.linspace(y_min,y_max,300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)
    cmap_bg = ListedColormap(["#f0f9ff", "#fff7ed"])
    cmap_pts= ListedColormap(["#0284c7","#ea580c"])
    ax.contourf(xx,yy,Z,alpha=0.35,cmap=cmap_bg,levels=[-0.5,0.5,1.5])
    ax.scatter(Xtr[:,0],Xtr[:,1],c=ytr,s=12,alpha=0.75,cmap=cmap_pts,edgecolors="k",linewidths=0.2,label="train")
    ax.scatter(Xte[:,0],Xte[:,1],c=yte,s=18,alpha=0.9,cmap=cmap_pts,marker="x",label="test")
    ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([])

# =========================
# Timing
# =========================

def time_fit_predict(clf, Xtr, ytr, Xte):
    t0 = perf_counter(); clf.fit(Xtr, ytr); t_fit = perf_counter() - t0
    t0 = perf_counter(); yhat = clf.predict(Xte); t_pred = perf_counter() - t0
    return yhat, t_fit, t_pred

# =========================
# Reporting
# =========================

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

# =========================
# Experiment block
# =========================

def run_block(tag, Xtr, Xte, ytr, yte, Xcal, ycal, seed, outdir, imbalance, budget_match):
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte, Xcal = scaler.transform(Xtr), scaler.transform(Xte), scaler.transform(Xcal)
    nC = 2
    if imbalance > 1.0:
        Xtr, ytr = apply_long_tail(Xtr, ytr, imbalance, seed)

    results, cal_rows = [], []
    plots_dir = Path(outdir) / "plots"
    ensure_dir(plots_dir)

    # ----- Baselines (timed) -----
    y_lin, tfit_lin, tpred_lin = time_fit_predict(baseline_linear(), Xtr, ytr, Xte)
    y_knn, tfit_knn, tpred_knn = time_fit_predict(baseline_knn(k=10), Xtr, ytr, Xte)

    # LS inductive
    t0 = perf_counter()
    ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=30); ls.fit(Xtr, ytr)
    tfit_lps = perf_counter() - t0
    t1 = perf_counter(); P_tst_ls = ls.predict_proba(Xte); tpred_lps = perf_counter() - t1
    y_lps = P_tst_ls.argmax(1)

    # ----- GM & degenerates -----
    # Scale K with data size (binary heuristic)
    Kproto = int(np.clip(len(Xtr)//150, 64, 256))
    gm_cfg = GMConfig(K=Kproto, knn_graph=12, attach_k=10, alpha=0.1, beta=1.5,
                      use_minibatch_kmeans=True, kmeans_batch_size=8192,
                      reliability_sample_cap=512, diffusion_iters=20, dtype="float32")
    t0 = perf_counter(); gm = GraphMemory(gm_cfg).fit(Xtr, ytr); tfit_gm = perf_counter()-t0
    t0 = perf_counter(); y_gm = gm.predict(Xte, nC); tpred_gm = perf_counter()-t0

    gm_flat = gm_flat_from(gm)
    t0 = perf_counter(); y_gmflat = gm_flat.predict(Xte, nC); tpred_gmflat = perf_counter()-t0
    tfit_gmflat = 0.0

    t0 = perf_counter(); gmi = gm_instance(Xtr, ytr, knn_graph=12, alpha=0.1, beta=1.5, attach_k=10); tfit_gmi=perf_counter()-t0
    t0 = perf_counter(); y_gmi = gmi.predict(Xte, nC); tpred_gmi=perf_counter()-t0

    t0 = perf_counter(); gmc = gm_centroid(Xtr, ytr); tfit_gmc=perf_counter()-t0
    t0 = perf_counter(); y_gmc = gmc.predict(Xte, nC); tpred_gmc=perf_counter()-t0

    # ----- Accuracies -----
    acc_lin  = accuracy_score(yte, y_lin)
    acc_knn  = accuracy_score(yte, y_knn)
    acc_lps  = accuracy_score(yte, y_lps)
    acc_gm   = accuracy_score(yte, y_gm)
    acc_gmf  = accuracy_score(yte, y_gmflat)
    acc_gmi  = accuracy_score(yte, y_gmi)
    acc_gmc  = accuracy_score(yte, y_gmc)

    # ----- Calibration metrics -----
    P_lin  = probs_linear(Xtr, ytr, Xte)
    P_knn  = probs_knn(Xtr, ytr, Xte, k=10)
    P_gm   = gm.predict_proba(Xte, nC)
    P_gmf  = gm_flat.predict_proba(Xte, nC)
    P_gmi  = gmi.predict_proba(Xte, nC)
    P_gmc  = gmc.predict_proba(Xte, nC)

    cal_rows.extend([
        dict(tag=tag, method="Linear",          **all_metrics(P_lin,  yte)),
        dict(tag=tag, method="kNN",             **all_metrics(P_knn,  yte)),
        dict(tag=tag, method="LabelSpreading",  **all_metrics(P_tst_ls, yte)),
        dict(tag=tag, method="GM",              **all_metrics(P_gm,   yte)),
        dict(tag=tag, method="GM-flat",         **all_metrics(P_gmf,  yte)),
        dict(tag=tag, method="GM-instance",     **all_metrics(P_gmi,  yte)),
        dict(tag=tag, method="GM-centroid",     **all_metrics(P_gmc,  yte)),
    ])

    # ----- Rows (acc/time) -----
    rows = [
        dict(tag=tag, method="Linear",         acc=acc_lin,  t_fit=tfit_lin,   t_pred=tpred_lin,   n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="kNN",            acc=acc_knn,  t_fit=tfit_knn,   t_pred=tpred_knn,   n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="LabelSpreading", acc=acc_lps,  t_fit=tfit_lps,   t_pred=tpred_lps,   n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="GM",             acc=acc_gm,   t_fit=tfit_gm,    t_pred=tpred_gm,    n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method="GM-flat",        acc=acc_gmf,  t_fit=tfit_gmflat,t_pred=tpred_gmflat,n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method="GM-instance",    acc=acc_gmi,  t_fit=tfit_gmi,   t_pred=tpred_gmi,   n_train=len(Xtr), n_test=len(Xte), K=len(Xtr)),
        dict(tag=tag, method="GM-centroid",    acc=acc_gmc,  t_fit=tfit_gmc,   t_pred=tpred_gmc,   n_train=len(Xtr), n_test=len(Xte), K=2),
    ]
    results.extend(rows)

    # ----- Plots -----
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    plot_decision(axs[0,0], lambda Z: baseline_linear().fit(Xtr, ytr).predict(Z), Xtr, ytr, Xte, yte, f"Linear ({acc_lin:.3f})")
    plot_decision(axs[0,1], lambda Z: baseline_knn(k=10).fit(Xtr, ytr).predict(Z),  Xtr, ytr, Xte, yte, f"kNN ({acc_knn:.3f})")
    plot_decision(axs[0,2], lambda Z: LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2, max_iter=30).fit(Xtr, ytr).predict(Z),
                  Xtr, ytr, Xte, yte, "LabelSpreading (inductive)")
    plot_decision(axs[1,0], lambda Z: gm.predict(Z, nC),       Xtr, ytr, Xte, yte, f"GM ({acc_gm:.3f})")
    plot_decision(axs[1,1], lambda Z: gm_flat.predict(Z, nC),  Xtr, ytr, Xte, yte, f"GM-flat ({acc_gmf:.3f})")
    plot_decision(axs[1,2], lambda Z: gm_centroid(Xtr, ytr).predict(Z, nC), Xtr, ytr, Xte, yte, f"GM-centroid ({acc_gmc:.3f})")

    plt.suptitle(tag, fontsize=11)
    ensure_dir(plots_dir)
    plt.tight_layout(rect=[0,0,1,0.97])
    fig_path = str(plots_dir / f"{tag}.png")
    plt.savefig(fig_path, dpi=220); plt.close()

    return results, cal_rows, fig_path

# =========================
# Suite
# =========================

def run_suite(dataset: str, outdir: str, seed: int, imbalance: float, budget_match: bool):
    ensure_dir(outdir)
    set_seed(seed)

    if dataset == "moons":
        Xtr, ytr, Xte, yte = gen_moons(n_samples=2000, seed=seed, noise=0.25)
    elif dataset == "circles":
        Xtr, ytr, Xte, yte = gen_circles(n_samples=2000, seed=seed, factor=0.4, noise=0.1)
    else:
        raise ValueError("--dataset must be 'moons' or 'circles'")

    # 10% of TRAIN as calibration (not used for temperature now; kept for parity)
    Xtr, Xcal, ytr, ycal = train_test_split(Xtr, ytr, test_size=0.10, random_state=seed+777, stratify=ytr)

    tag = f"{dataset}_2class_fixed"
    rows, cal_rows, fig_path = run_block(tag, Xtr, Xte, ytr, yte, Xcal, ycal, seed, outdir, imbalance, budget_match)

    # Save summaries
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
                r["tag"], r["method"],
                f'{r["acc"]:.6f}', f'{r["nll"]:.6f}', f'{r["brier"]:.6f}', f'{r["ece"]:.6f}',
            ])

    with open(os.path.join(outdir, f"{dataset}_summary.json"), "w") as jf:
        json.dump(rows, jf, indent=2)

    console.print(rows_to_table(rows))
    console.print(Panel.fit(
        f"[bold green]Saved[/bold green] CSV → [white]{Path(csv_path).resolve()}[/white]\n"
        f"[bold green]Calibration[/bold green] → [white]{Path(cal_csv_path).resolve()}[/white]\n"
        f"[bold green]Plot[/bold green] → [white]{Path(fig_path).resolve()}[/white]"
    ))

# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["moons","circles"], required=True,
                    help="Binary non-linear datasets.")
    ap.add_argument("--outdir", type=str, default="runs_toy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imbalance", type=float, default=1.0,
                    help="Long-tail ratio (max/min) applied to TRAIN only. >1 enables long-tail.")
    ap.add_argument("--budget-match", action="store_true",
                    help="(Kept for API parity, unused here).")
    args = ap.parse_args()

    with section(f"Running suite: {args.dataset}  (long-tail={args.imbalance}  budget={args.budget_match})"):
        run_suite(args.dataset, args.outdir, args.seed, args.imbalance, args.budget_match)

if __name__ == "__main__":
    main()
