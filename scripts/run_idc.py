#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDC-focused experiments (binary) using Graph Memory (GM) and baselines.

Pipeline:
  1) Load IDC patches from HuggingFace (cached locally).
  2) Extract features with ImageNet-pretrained ResNet-50 (GPU if available).
  3) UMAP -> 2D embedding (fast) for all methods (keeps smoothness maps consistent).
  4) Train/test split (uses dataset's official splits: train/test).
  5) Run Linear, kNN, LabelSpreading, GM (+flat/instance/centroid), kNN-budget.
  6) Report accuracy/time, calibration (NLL/Brier/ECE), smoothness (mean ||∇p||^2).
  7) Save decision plots and smoothness heatmaps.

Usage:
  python scripts/run_idc.py --outdir runs_idc --train-per-class 8000 --test-per-class 8000 \
      --backbone resnet50 --batch-size 256 --epochs 2 --budget-match
"""

import hashlib
import os, csv, json, argparse, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset

# ----- import your GM tooling and utilities from the toy script -----
# Make sure run_gm.py is importable (same dir or PYTHONPATH)
from run_gm import (
    # core GM
    GMConfig, GraphMemory, gm_flat_from, gm_instance, gm_centroid,
    # baselines & metrics
    baseline_linear, baseline_knn, probs_labels_spreading_transductive,
    probs_gm, probs_linear, probs_knn, all_metrics,
    # plots & extras
    make_grid, decision_smoothness, grad_magnitude_map, plot_decision,
    rows_to_table, section, ensure_dir, stratified_quota_indices
)

import umap

# ---------- Cache helpers ----------
def _hash_cfg(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:10]

def make_cache_key(backbone: str, img_size: int, aug_tag: str, extra: str = ""):
    raw = f"{backbone}|img{img_size}|seed0|aug={aug_tag}|{extra}"
    return f"{backbone}_{_hash_cfg(raw)}"

def cache_paths(cache_dir: str, key: str):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    feat_npz = cache_dir / f"{key}_features.npz"
    ckpt_pt  = cache_dir / f"{key}_best.pt"
    meta_json= cache_dir / f"{key}_meta.json"
    return feat_npz, ckpt_pt, meta_json


# ---------------------------------------------------------
# Data & features
# ---------------------------------------------------------

IMG_SIZE = 96


def _set_torch():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)

def load_idc_subset(train_per_class=8000, test_per_class=8000, cache_dir=None, seed=42):
    """
    Load a stratified subset from the IDC HF dataset with local caching.
    Returns PIL-based HF datasets for train/test with {'image','label'}.
    """
    ds = load_dataset("dbzadnen/breast-histopathology-images", cache_dir=cache_dir)

    rng = np.random.RandomState(seed)
    def stratify_split(split_name, k_per_class):
        split = ds[split_name]
        labels = np.array([int(r["label"]) for r in split])
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        k0 = min(k_per_class, len(idx0))
        k1 = min(k_per_class, len(idx1))
        keep = np.concatenate([
            rng.choice(idx0, k0, replace=False),
            rng.choice(idx1, k1, replace=False),
        ])
        rng.shuffle(keep)
        return split.select(keep)

    tr = stratify_split("train",      train_per_class)
    te = stratify_split("test",       test_per_class)
    return tr, te


def make_umap_2d(Xtr_f, Xte_f, seed=0, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="euclidean", random_state=seed
    )
    reducer.fit(Xtr_f)
    Xtr_2d = reducer.transform(Xtr_f)
    Xte_2d = reducer.transform(Xte_f)
    return Xtr_2d, Xte_2d


class IDCWrap(torch.utils.data.Dataset):
    def __init__(self, hf_split, tf):
        self.hf = hf_split
        self.tf = tf
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        r = self.hf[int(i)]
        img = r["image"].convert("RGB")
        y = int(r["label"])
        return self.tf(img), y

def build_backbone(name="resnet50", weights="DEFAULT", device=None, num_classes=2):
    """
    Return (backbone, head), where backbone outputs a feature vector and head is a small linear head.
    If train_fc=True, a linear head is trained for a few epochs to adapt; backbone stays frozen.
    """
    if name.lower() == "resnet50":
        backbone = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features
    elif name.lower() == "resnet18":
        backbone = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features
    else:
        raise ValueError("--backbone must be resnet18 or resnet50")

    head = nn.Linear(feat_dim, num_classes)
    backbone.fc = head
    backbone = backbone.to(device)
    return backbone

@torch.no_grad()
def extract_features(encoder_body: nn.Module, loader, device):
    encoder_body.eval().to(device)
    Xs, Ys = [], []
    for xb, yb in tqdm(loader, desc="Extracting features", leave=False):
        xb = xb.to(device, non_blocking=True)
        feats = encoder_body(xb).flatten(1)
        Xs.append(feats.cpu().numpy())
        Ys.append(yb.numpy())
    return np.concatenate(Xs, 0), np.concatenate(Ys, 0)

def features_or_cache(
    cache_dir: str,
    backbone_key: str,
    features_key: str,
    build_backbone_fn,  # returns nn.Module with logits head
    dl_tr, dl_val, dl_te,
    device, epochs=5, lr=1e-3, wd=1e-4,
):
    feat_npz, _, meta_json = cache_paths(cache_dir, features_key)
    _, ckpt_pt, _ = cache_paths(cache_dir, backbone_key)

    # Fast path: if features exist, just load and SKIP backbone entirely
    if feat_npz.exists():
        z = np.load(feat_npz)
        meta = json.load(open(meta_json)) if meta_json.exists() else {}
        print(f"[cache] Loaded features from {feat_npz}")
        return (z["Xtr_f"], z["ytr"], z["Xte_f"], z["yte"], meta)

    # Otherwise: build backbone, (optionally) train head, save best ckpt
    backbone = build_backbone_fn().to(device)
    if not ckpt_pt.exists():
        if epochs > 0:
            print(f"[train] training head for {epochs} epochs")
            best_va = train_backbone(backbone, dl_tr, dl_val, device, epochs=epochs, lr=lr, wd=wd, ckpt_path=ckpt_pt)
            print(f"[train] best val acc: {best_va:.4f}")
        else:
            # still evaluate once for logging
            va_loss, va_acc = evaluate_backbone(backbone, dl_val, device)
            print(f"[eval] val acc (frozen): {va_acc:.4f}")
    else:
        backbone.load_state_dict(torch.load(ckpt_pt, map_location="cpu"))

    # Swap classifier for feature body
    # Assumes torchvision-style backbones where the last module is the classifier
    # Adapt if your backbone structure differs.
    modules = list(backbone.children())[:-1]
    encoder_body = nn.Sequential(*modules)

    # Extract & cache features
    Xtr_f, ytr = extract_features(encoder_body, dl_tr, device)
    Xte_f, yte = extract_features(encoder_body, dl_te, device)

    np.savez_compressed(feat_npz, Xtr_f=Xtr_f, ytr=ytr, Xte_f=Xte_f, yte=yte)
    meta = dict(
        ckpt=str(ckpt_pt.resolve()),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    return (Xtr_f, ytr, Xte_f, yte, meta)


@torch.no_grad()
def evaluate_backbone(encoder: nn.Module, loader, device):
    encoder.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = encoder(xb)
        loss = ce(logits, yb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.numel()
        loss_sum+= loss.item()*yb.numel()
    return (loss_sum/total, correct/total)

def train_backbone(
    backbone: nn.Module, dl_tr, dl_val, device,
    epochs=5, lr=1e-3, wd=1e-4, ckpt_path: Path | None = None
):
    backbone.to(device)
    opt = torch.optim.AdamW(backbone.parameters(), lr=lr, weight_decay=wd)
    ce  = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_va = -1.0
    best_state = None

    for ep in range(epochs):
        backbone.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for xb, yb in tqdm(dl_tr, desc=f"Train ep {ep+1}/{epochs}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = backbone(xb)
                loss = ce(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss   += loss.item() * yb.numel()
            running_correct+= (logits.argmax(1) == yb).sum().item()
            running_total  += yb.numel()

        tr_loss = running_loss / max(1, running_total)
        tr_acc  = running_correct / max(1, running_total)

        va_loss, va_acc = evaluate_backbone(backbone, dl_val, device)
        print(f"[{ep+1:02d}/{epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.cpu() for k, v in backbone.state_dict().items()}
            if ckpt_path:
                torch.save(best_state, ckpt_path)

    # load best (from RAM or disk) back into model
    if best_state is not None:
        backbone.load_state_dict(best_state)
    elif ckpt_path and ckpt_path.exists():
        backbone.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return best_va


def standardize_train_test(Xtr: np.ndarray, Xte: np.ndarray):
    """
    Standardize train/test features using only train statistics.
    """
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte)


def embedding_smoothness(P: np.ndarray, X: np.ndarray, k: int = 15, beta: float = 1.0) -> float:
    """
    Compute smoothness of class probability function p(x) on a point cloud (embedding space).

    Args:
        P:  [N, C] array of predicted probabilities (for 2-class problems, [N] or [N,2]).
        X:  [N, D] feature vectors (standardized embedding space).
        k:  number of neighbors for local connectivity (default 15).
        beta: Gaussian bandwidth factor; smaller = smoother kernel (default 1.0).

    Returns:
        Scalar Dirichlet energy (mean squared gradient proxy):
            (1/|E|) * sum_{i<j} w_ij * (p_i - p_j)^2
        where w_ij = exp(-beta * ||x_i - x_j||^2).
    """
    # Handle binary and multi-class cases
    if P.ndim == 2 and P.shape[1] > 1:
        # Use class-1 probabilities for binary problems
        p = P[:, 1]
    else:
        p = P.ravel()

    N = len(p)
    if N <= 2:
        return 0.0

    # Compute kNN distances (symmetric)
    A = kneighbors_graph(X, n_neighbors=min(k, N - 1), mode="distance", include_self=False).toarray()
    W = np.exp(-beta * (A ** 2))
    W = np.maximum(W, W.T)  # symmetrize

    # Dirichlet energy numerator
    diff = p[:, None] - p[None, :]
    energy = 0.5 * np.sum(W * (diff ** 2))
    norm = 0.5 * np.sum(W) + 1e-12

    return float(energy / norm)


# ---------------------------------------------------------
# Experiments on 2-D embedding (UMAP) to enable smoothness maps
# ---------------------------------------------------------

def run_idc_block(tag, Xtr_f, ytr, Xte_f, yte, outdir, seed, imbalance, budget_match,
                  alpha, beta, kproto):
    """
    High-D authoritative path: all training/metrics on X*_f (original embeddings).
    Optional UMAP 2-D is used ONLY for qualitative figures.
    """
    plots_dir = Path(outdir) / "plots"; ensure_dir(str(plots_dir))

    # ---------- High-D standardization ----------
    # Xtr, Xte = standardize_train_test(Xtr_f, Xte_f)
    Xtr, Xte = Xtr_f, Xte_f
    if imbalance > 1.0:
        Xtr, ytr = apply_long_tail_binary(Xtr, ytr, imbalance, seed)

    # ---------- Baselines (High-D) ----------
    t0 = time.perf_counter(); lin = baseline_linear().fit(Xtr, ytr); tfit_lin = time.perf_counter()-t0
    t0 = time.perf_counter(); y_lin = lin.predict(Xte); tpred_lin = time.perf_counter()-t0

    t0 = time.perf_counter(); knn = baseline_knn(k=10).fit(Xtr, ytr); tfit_knn = time.perf_counter()-t0
    t0 = time.perf_counter(); y_knn = knn.predict(Xte); tpred_knn = time.perf_counter()-t0

    # LabelSpreading (transductive over train+test) in High-D
    t0 = time.perf_counter()
    P_tst_ls = probs_labels_spreading_transductive(Xtr, ytr, Xte, n_neighbors=10, alpha=0.2)
    t_ls = time.perf_counter() - t0
    y_ls = P_tst_ls.argmax(1)

    # ---------- GM & degenerates (High-D) ----------
    # heuristic: ~N/25 prototypes, clipped to [64, 256]
    #Kproto = int(np.clip(len(Xtr) // 25, 64, 256))
    knn_graph=12
    attach_k=8
    diffusion_iters=20
    Kproto = kproto
    alpha=alpha
    beta=beta

    gm_cfg = GMConfig(
        K=Kproto,
        knn_graph=knn_graph,              
        attach_k=attach_k,                # local attachment; lets reliability/coverage do work
        alpha=alpha,                 # mild diffusion
        beta=beta,                  # RBF falloff; 1.5–2.0 both work, keep 1.5 first
        use_norm=False,             # L2-normalize prototypes (cosine-friendly in high-D)

        use_minibatch_kmeans=True,
        kmeans_batch_size=8192,
        reliability_sample_cap=512,
        disable_instability=False,  # speed; re-enable only for a robustness ablation
        diffusion_iters=diffusion_iters,
        dtype="float32",

        absolute_reliability=True,     # preserve absolute mass → better NLL/ECE
        enforce_opposite_seed=True,    # inject nearby opposite-class seed if top-k are same class
        coverage_search_factor=2
    )

    t0 = time.perf_counter(); gm = GraphMemory(gm_cfg).fit(Xtr, ytr); tfit_gm = time.perf_counter()-t0
    t0 = time.perf_counter(); y_gm = gm.predict(Xte, 2); tpred_gm = time.perf_counter()-t0


    gm_flat = gm_flat_from(gm)
    t0 = time.perf_counter(); y_gmflat = gm_flat.predict(Xte, 2); tpred_gmflat = time.perf_counter()-t0
    tfit_gmflat = 0.0

    t0 = time.perf_counter(); gmi = gm_instance(Xtr, ytr, knn_graph=knn_graph, alpha=alpha, beta=beta, attach_k=attach_k, diffusion_iters=diffusion_iters); tfit_gmi = time.perf_counter()-t0
    t0 = time.perf_counter(); y_gmi = gmi.predict(Xte, 2); tpred_gmi = time.perf_counter()-t0

    t0 = time.perf_counter(); gmc = gm_centroid(Xtr, ytr); tfit_gmc = time.perf_counter()-t0
    t0 = time.perf_counter(); y_gmc = gmc.predict(Xte, 2); tpred_gmc = time.perf_counter()-t0

    # ---------- Accuracies (High-D) ----------
    acc_lin = accuracy_score(yte, y_lin)
    acc_knn = accuracy_score(yte, y_knn)
    acc_ls  = accuracy_score(yte, y_ls)
    acc_gm  = accuracy_score(yte, y_gm)
    acc_gmf = accuracy_score(yte, y_gmflat)
    acc_gmi = accuracy_score(yte, y_gmi)
    acc_gmc = accuracy_score(yte, y_gmc)

    # ---------- Calibration (High-D) ----------
    P_lin = lin.predict_proba(Xte)
    P_knn = knn.predict_proba(Xte)
    P_gm  = probs_gm(gm,      Xte, 2)
    P_gmf = probs_gm(gm_flat, Xte, 2)
    P_gmi = probs_gm(gmi,     Xte, 2)
    P_gmc = probs_gm(gmc,     Xte, 2)

    cal_rows = [
        dict(tag=tag, method="Linear",          **all_metrics(P_lin, yte)),
        dict(tag=tag, method="kNN",             **all_metrics(P_knn, yte)),
        dict(tag=tag, method="LabelSpreading",  **all_metrics(P_tst_ls, yte)),
        dict(tag=tag, method="GM",              **all_metrics(P_gm,  yte)),
        dict(tag=tag, method="GM-flat",         **all_metrics(P_gmf, yte)),
        dict(tag=tag, method="GM-instance",     **all_metrics(P_gmi, yte)),
        dict(tag=tag, method="GM-centroid",     **all_metrics(P_gmc, yte)),
    ]

    # ---------- Embedding-space smoothness (High-D, authoritative) ----------
    smooth_rows = [
        dict(tag=tag, method="Linear",          smooth=embedding_smoothness(P_lin, Xte, k=15, beta=1.0)),
        dict(tag=tag, method="kNN",             smooth=embedding_smoothness(P_knn, Xte, k=15, beta=1.0)),
        dict(tag=tag, method="LabelSpreading",  smooth=embedding_smoothness(P_tst_ls, Xte, k=15, beta=1.0)),
        dict(tag=tag, method="GM",              smooth=embedding_smoothness(P_gm,  Xte, k=15, beta=1.0)),
        dict(tag=tag, method="GM-flat",         smooth=embedding_smoothness(P_gmf, Xte, k=15, beta=1.0)),
        dict(tag=tag, method="GM-instance",     smooth=embedding_smoothness(P_gmi, Xte, k=15, beta=1.0)),
        dict(tag=tag, method="GM-centroid",     smooth=embedding_smoothness(P_gmc, Xte, k=15, beta=1.0)),
    ]

    # ---------- Budget-matched kNN (High-D) ----------
    rows = [
        dict(tag=tag, method="Linear",         acc=acc_lin, t_fit=tfit_lin, t_pred=tpred_lin, n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="kNN",            acc=acc_knn, t_fit=tfit_knn, t_pred=tpred_knn, n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="LabelSpreading", acc=acc_ls,  t_fit=0.0,     t_pred=t_ls,      n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="GM",             acc=acc_gm,  t_fit=tfit_gm, t_pred=tpred_gm,  n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method="GM-flat",        acc=acc_gmf, t_fit=tfit_gmflat, t_pred=tpred_gmflat, n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method="GM-instance",    acc=acc_gmi, t_fit=tfit_gmi, t_pred=tpred_gmi, n_train=len(Xtr), n_test=len(Xte), K=len(Xtr)),
        dict(tag=tag, method="GM-centroid",    acc=acc_gmc, t_fit=tfit_gmc, t_pred=tpred_gmc, n_train=len(Xtr), n_test=len(Xte), K=2),
    ]
    if budget_match:
        P = Kproto
        rng = np.random.RandomState(seed + 20)
        keep = stratified_quota_indices(ytr, total=P, rng=rng, min_each=1)
        Xsub, ysub = Xtr[keep], ytr[keep]
        t0 = time.perf_counter(); knnb = baseline_knn(k=10).fit(Xsub, ysub); tfit_knnb = time.perf_counter()-t0
        t0 = time.perf_counter(); y_knnb = knnb.predict(Xte); tpred_knnb = time.perf_counter()-t0
        rows.append(dict(tag=tag, method=f"kNN-budget(P={P})", acc=accuracy_score(yte, y_knnb),
                         t_fit=tfit_knnb, t_pred=tpred_knnb, n_train=len(Xsub), n_test=len(Xte), K=P))
        cal_rows.append(dict(tag=tag, method=f"kNN-budget(P={P})", **all_metrics(knnb.predict_proba(Xte), yte)))
        smooth_rows.append(dict(tag=tag, method=f"kNN-budget(P={P})",
                                smooth=embedding_smoothness(knnb.predict_proba(Xte), Xte, k=15, beta=1.0)))

    # ---------- 2-D VISUALS ONLY (UMAP) ----------
    Xtr_2d, Xte_2d = make_umap_2d(Xtr, Xte, seed=seed, n_neighbors=15, min_dist=0.1)

    # Fit a separate GM in 2-D just for visualization
    gm2d_cfg = GMConfig(
        K=Kproto,
        knn_graph=knn_graph, attach_k=attach_k, alpha=alpha, beta=beta,
        use_minibatch_kmeans=True, kmeans_batch_size=4096,
        reliability_sample_cap=256, disable_instability=True,
        diffusion_iters=20, dtype="float32"
    )
    gm2d = GraphMemory(gm2d_cfg).fit(Xtr_2d, ytr)

    # Train light models in 2-D ONLY for plotting
    knn2d = KNeighborsClassifier(n_neighbors=10).fit(Xtr_2d, ytr)
    # LS 2-D for qualitative comparison
    P_grid_ls_2d_fn = lambda Z: probs_labels_spreading_transductive(
        Xtr_2d, ytr, Z, n_neighbors=10, alpha=0.2).argmax(1)

    # Decision plots (qualitative)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    plot_decision(axs[0], lambda Z: gm2d.predict(Z, 2), Xtr_2d, ytr, Xte_2d, yte, "GM (2-D viz)")
    plot_decision(axs[1], lambda Z: knn2d.predict(Z), Xtr_2d, ytr, Xte_2d, yte, "kNN (2-D viz)")
    plot_decision(axs[2], lambda Z: P_grid_ls_2d_fn(Z), Xtr_2d, ytr, Xte_2d, yte, "LabelSpreading (2-D viz)")
    fig_path = str((plots_dir / f"{tag}_umap_2d_viz.png").resolve()); plt.savefig(fig_path, dpi=220); plt.close(fig)

    # Optional: gradient maps in 2-D (qualitative)
    xx, yy, _ = make_grid(Xtr_2d, Xte_2d, n=300, pad=1.0)
    p_knn2d = knn2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
    p_ls2d  = probs_labels_spreading_transductive(Xtr_2d, ytr, np.c_[xx.ravel(), yy.ravel()], n_neighbors=10, alpha=0.2)[:,1].reshape(xx.shape)
    p_gm2d  = probs_gm(gm2d, np.c_[xx.ravel(), yy.ravel()], 2)[:,1].reshape(xx.shape)  # NB: gm is trained in high-D; this is just a viz proxy

    gm_knn = grad_magnitude_map(p_knn2d, xx, yy)
    gm_ls  = grad_magnitude_map(p_ls2d,  xx, yy)
    gm_gm  = grad_magnitude_map(p_gm2d,  xx, yy)
    vmax = np.percentile(np.r_[gm_knn.ravel(), gm_ls.ravel(), gm_gm.ravel()], 99)
    fig_s, axs_s = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    axs_s[0].imshow(gm_gm, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axs_s[0].set_title("GM ∥∇p∥ (2-D viz)")
    axs_s[1].imshow(gm_knn, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axs_s[1].set_title("kNN ∥∇p∥ (2-D viz)")
    axs_s[2].imshow(gm_ls, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axs_s[2].set_title("LS ∥∇p∥ (2-D viz)")
    fig_s.colorbar(axs_s[2].images[0], ax=axs_s.ravel().tolist(), shrink=0.85, pad=0.02)
    smooth_path = str((plots_dir / f"{tag}_umap_2d_smoothness.png").resolve()); plt.savefig(smooth_path, dpi=220); plt.close(fig_s)

    return rows, cal_rows, smooth_rows, fig_path


def apply_long_tail_binary(Xtr, ytr, imbalance, seed):
    if imbalance <= 1.0:
        return Xtr, ytr
    rng = np.random.RandomState(seed + 17)
    idx0 = np.where(ytr == 0)[0]
    idx1 = np.where(ytr == 1)[0]
    # make class 1 smaller by factor
    target1 = max(50, int(round(len(idx0) / imbalance)))
    keep1 = rng.choice(idx1, size=min(target1, len(idx1)), replace=False)
    keep = np.concatenate([idx0, keep1])
    rng.shuffle(keep)
    return Xtr[keep], ytr[keep]

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="runs_idc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", type=str, default=str(Path.home()/".cache"/"hf_idc"))
    ap.add_argument("--train-per-class", type=int, default=8000)
    ap.add_argument("--test-per-class", type=int, default=8000)
    ap.add_argument("--backbone", type=str, choices=["resnet18","resnet50"], default="resnet18")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5, help="Linear head epochs (backbone frozen)")
    ap.add_argument("--imbalance", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--kproto", type=float, default=32)
    ap.add_argument("--budget-match", action="store_true")
    args = ap.parse_args()

    device = _set_torch()
    ensure_dir(args.outdir)
    ensure_dir(args.cache_dir)

    with section(f"Backbone: {args.backbone} (head epochs={args.epochs}) (train: {args.train_per_class})"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- Prepare cache key / paths ----
        aug_tag = "hflip_vflip_jit"
        backbone_key = make_cache_key(args.backbone, IMG_SIZE, aug_tag,
                            extra=f"bs{args.batch_size}_ep{args.epochs}_trainsize{args.train_per_class}")
        features_key = make_cache_key(args.backbone, IMG_SIZE, aug_tag,
                            extra=f"bs{args.batch_size}_ep{args.epochs}_trainsize{args.train_per_class}_testsize{args.test_per_class}")
        feat_npz, ckpt_pt, meta_json = cache_paths(args.cache_dir, features_key)

        # ---- Try to load cached features first ----
        if feat_npz.exists():
            print(f"[cache] Found cached features: {feat_npz}")
            z = np.load(feat_npz)
            meta = json.load(open(meta_json)) if meta_json.exists() else {}
            Xtr_f, ytr, Xte_f, yte = z["Xtr_f"], z["ytr"], z["Xte_f"], z["yte"]
            print(f"[cache] Loaded Xtr_f={Xtr_f.shape}, Xte_f={Xte_f.shape}")
        else:
            # ---------- Load dataset only if cache miss ----------
            with section("Load IDC subsets"):
                tr_hf, te_hf = load_idc_subset(
                    train_per_class=args.train_per_class,
                    test_per_class=args.test_per_class,
                    cache_dir=args.cache_dir,
                    seed=args.seed
                )

            # ---------- Transforms ----------
            tf_tr = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(p=0.1),
                T.ColorJitter(0.1,0.1,0.1,0.05),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            tf_ev = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

            # ---------- Split train → train/val ----------
            full_train = IDCWrap(tr_hf, tf_tr)
            val_ratio = 0.10
            n_tr = len(full_train)
            n_val = max(1, int(round(val_ratio * n_tr)))
            n_tr_main = n_tr - n_val
            tr_set, va_set = random_split(
                full_train, [n_tr_main, n_val],
                generator=torch.Generator().manual_seed(args.seed)
            )

            dl_tr = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=False)
            dl_va = DataLoader(
                IDCWrap(getattr(locals(), "va_hf", None) or va_set, tf_ev)
                if isinstance(va_set, IDCWrap) else va_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False
            )
            dl_te = DataLoader(IDCWrap(te_hf, tf_ev), batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

            # ---------- Build backbone & extract features ----------
            def build_backbone_fn():
                # Returns model WITH classification head
                return build_backbone(args.backbone, device=device)

            Xtr_f, ytr, Xte_f, yte, meta = features_or_cache(
                cache_dir=args.cache_dir,
                backbone_key=backbone_key, # Checks if the backbone needs to be trained or loaded from cache
                features_key=features_key, # Checks if train/test samples needs to be computed or loaded from cache.
                build_backbone_fn=build_backbone_fn,
                dl_tr=dl_tr, dl_val=dl_va, dl_te=dl_te,
                device=device,
                epochs=args.epochs, lr=1e-3, wd=1e-4,
            )

        print(f"✅ IDC feature extraction complete: Xtr_f={Xtr_f.shape}, Xte_f={Xte_f.shape}")

    with section("Run IDC block"):
        tag = "idc"
        rows, cal_rows, smooth_rows, fig_path = run_idc_block(
            tag, Xtr_f, ytr, Xte_f, yte,
            outdir=args.outdir, seed=args.seed,
            imbalance=args.imbalance, budget_match=args.budget_match,
            alpha=args.alpha, beta=args.beta, kproto=args.kproto,
        )

    # Save CSVs
    csv_path = os.path.join(args.outdir, "idc_summary.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag","method","acc","t_fit","t_pred","n_train","n_test","K"])
        for r in rows:
            wr.writerow([r["tag"], r["method"], f'{r["acc"]:.6f}', f'{r["t_fit"]:.8f}',
                         f'{r["t_pred"]:.8f}', r["n_train"], r["n_test"], r.get("K","-")])

    cal_csv_path = os.path.join(args.outdir, "idc_calibration.csv")
    with open(cal_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag","method","acc","nll","brier","ece"])
        for r in cal_rows:
            wr.writerow([r["tag"], r["method"], f'{r["acc"]:.6f}', f'{r["nll"]:.6f}', f'{r["brier"]:.6f}', f'{r["ece"]:.6f}'])

    smooth_csv_path = os.path.join(args.outdir, "idc_smooth.csv")
    with open(smooth_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag","method","smooth"])
        for r in smooth_rows:
            wr.writerow([r["tag"], r["method"], f'{r["smooth"]:.6f}'])

    # Console summary
    from rich.console import Console; from rich.panel import Panel
    console = Console()
    console.print(rows_to_table(rows))
    console.print(Panel.fit(
        f"[bold green]Saved[/bold green] CSV → [white]{Path(csv_path).resolve()}[/white]\n"
        f"[bold green]Plot[/bold green] → [white]{Path(fig_path).resolve()}[/white]"
    ))

# Helpers (local)
def apply_long_tail_binary(Xtr, ytr, imbalance, seed):
    if imbalance <= 1.0: return Xtr, ytr
    rng = np.random.RandomState(seed + 7)
    idx0 = np.where(ytr==0)[0]; idx1 = np.where(ytr==1)[0]
    target1 = max(100, int(round(len(idx0)/imbalance)))
    keep1 = rng.choice(idx1, size=min(target1, len(idx1)), replace=False)
    keep = np.concatenate([idx0, keep1]); rng.shuffle(keep)
    return Xtr[keep], ytr[keep]

if __name__ == "__main__":
    main()
