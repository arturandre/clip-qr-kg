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

import os, csv, json, argparse, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv
from torch.utils.data import DataLoader

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

# ---------------------------------------------------------
# Data & features
# ---------------------------------------------------------

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

def build_backbone(name="resnet50", weights="DEFAULT", device=None, train_fc=False, num_classes=2):
    """
    Return (encoder, head), where encoder outputs a feature vector and head is a small linear head.
    If train_fc=True, a linear head is trained for a few epochs to adapt; backbone stays frozen.
    """
    if name.lower() == "resnet50":
        m = tv.resnet50(weights=weights)
        feat_dim = m.fc.in_features
    elif name.lower() == "resnet18":
        m = tv.resnet18(weights=weights)
        feat_dim = m.fc.in_features
    else:
        raise ValueError("--backbone must be resnet18 or resnet50")

    # encoder: everything except the final fc
    encoder = nn.Sequential(*(list(m.children())[:-1]))  # outputs [B, C, 1, 1]
    for p in encoder.parameters():
        p.requires_grad = False

    head = nn.Linear(feat_dim, num_classes)
    encoder = encoder.to(device)
    head = head.to(device)
    return encoder, head, feat_dim

@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader, device):
    encoder.eval()
    Xs, Ys = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        feats = encoder(xb).flatten(1)
        Xs.append(feats.cpu().numpy())
        Ys.append(yb.numpy())
    X = np.concatenate(Xs, 0)
    y = np.concatenate(Ys, 0).astype(np.int64)
    return X, y

def train_head(encoder, head, loader, device, epochs=2, lr=1e-3, wd=1e-4):
    encoder.eval()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    for ep in tqdm(range(epochs), desc="Training encoder"):
        head.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.no_grad():
                feats = encoder(xb).flatten(1)
            logits = head(feats)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

# ---------------------------------------------------------
# Experiments on 2-D embedding (UMAP) to enable smoothness maps
# ---------------------------------------------------------

def run_idc_block(tag, Xtr_d, ytr, Xte_d, yte, outdir, seed, imbalance, budget_match):
    """
    Xtr_d/Xte_d are *2-D* embeddings (UMAP projected).
    Runs the same experiment suite as toys, including smoothness maps.
    """
    # Standardize the 2-D space
    scaler = StandardScaler().fit(Xtr_d)
    Xtr, Xte = scaler.transform(Xtr_d), scaler.transform(Xte_d)

    if imbalance > 1.0:
        Xtr, ytr = apply_long_tail_binary(Xtr, ytr, imbalance, seed)

    results, cal_rows, smooth_rows = [], [], []
    plots_dir = Path(outdir) / "plots"
    ensure_dir(str(plots_dir))

    # ---------- Baselines ----------
    # Linear/kNN on 2-D
    t0 = time.perf_counter(); lin = baseline_linear().fit(Xtr, ytr); tfit_lin = time.perf_counter()-t0
    t0 = time.perf_counter(); y_lin = lin.predict(Xte); tpred_lin = time.perf_counter()-t0

    t0 = time.perf_counter(); knn = baseline_knn(k=10).fit(Xtr, ytr); tfit_knn = time.perf_counter()-t0
    t0 = time.perf_counter(); y_knn = knn.predict(Xte); tpred_knn = time.perf_counter()-t0

    # LabelSpreading (transductive over train+test)
    t0 = time.perf_counter()
    P_tst_ls = probs_labels_spreading_transductive(Xtr, ytr, Xte, n_neighbors=10, alpha=0.2)
    t_ls = time.perf_counter() - t0
    y_lps = P_tst_ls.argmax(1)

    # ---------- GM & degenerates ----------
    Kproto = max(32, min(256, len(Xtr)//15))
    gm_cfg = GMConfig(
        K=Kproto, knn_graph=12, attach_k=12, alpha=0.1, beta=1.5,
        use_minibatch_kmeans=True, kmeans_batch_size=8192,
        reliability_sample_cap=512, disable_instability=True,
        diffusion_iters=20, dtype="float32"
    )
    t0 = time.perf_counter(); gm = GraphMemory(gm_cfg).fit(Xtr, ytr); tfit_gm = time.perf_counter()-t0
    t0 = time.perf_counter(); y_gm = gm.predict(Xte, 2); tpred_gm = time.perf_counter()-t0

    gm_flat = gm_flat_from(gm)
    t0 = time.perf_counter(); y_gmflat = gm_flat.predict(Xte, 2); tpred_gmflat = time.perf_counter()-t0
    tfit_gmflat = 0.0

    t0 = time.perf_counter(); gmi = gm_instance(Xtr, ytr, knn_graph=12, alpha=0.1, beta=1.5, attach_k=12); tfit_gmi=time.perf_counter()-t0
    t0 = time.perf_counter(); y_gmi = gmi.predict(Xte, 2); tpred_gmi=time.perf_counter()-t0

    t0 = time.perf_counter(); gmc = gm_centroid(Xtr, ytr); tfit_gmc=time.perf_counter()-t0
    t0 = time.perf_counter(); y_gmc = gmc.predict(Xte, 2); tpred_gmc=time.perf_counter()-t0

    # ---------- Accuracies ----------
    acc_lin = accuracy_score(yte, y_lin)
    acc_knn = accuracy_score(yte, y_knn)
    acc_ls  = accuracy_score(yte, y_lps)
    acc_gm  = accuracy_score(yte, y_gm)
    acc_gmf = accuracy_score(yte, y_gmflat)
    acc_gmi = accuracy_score(yte, y_gmi)
    acc_gmc = accuracy_score(yte, y_gmc)

    # ---------- Calibration (2-D space) ----------
    P_lin = lin.predict_proba(Xte)
    P_knn = knn.predict_proba(Xte)
    P_gm  = probs_gm(gm, Xte, 2)
    P_gmf = probs_gm(gm_flat, Xte, 2)
    P_gmi = probs_gm(gmi, Xte, 2)
    P_gmc = probs_gm(gmc, Xte, 2)

    cal_rows += [
        dict(tag=tag, method="Linear",          **all_metrics(P_lin, yte)),
        dict(tag=tag, method="kNN",             **all_metrics(P_knn, yte)),
        dict(tag=tag, method="LabelSpreading",  **all_metrics(P_tst_ls, yte)),
        dict(tag=tag, method="GM",              **all_metrics(P_gm,  yte)),
        dict(tag=tag, method="GM-flat",         **all_metrics(P_gmf, yte)),
        dict(tag=tag, method="GM-instance",     **all_metrics(P_gmi, yte)),
        dict(tag=tag, method="GM-centroid",     **all_metrics(P_gmc, yte)),
    ]

    rows = [
        dict(tag=tag, method="Linear",         acc=acc_lin, t_fit=tfit_lin, t_pred=tpred_lin, n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="kNN",            acc=acc_knn, t_fit=tfit_knn, t_pred=tpred_knn, n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="LabelSpreading", acc=acc_ls,  t_fit=0.0,     t_pred=t_ls,      n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag=tag, method="GM",             acc=acc_gm,  t_fit=tfit_gm, t_pred=tpred_gm,  n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method="GM-flat",        acc=acc_gmf, t_fit=tfit_gmflat, t_pred=tpred_gmflat, n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag=tag, method="GM-instance",    acc=acc_gmi, t_fit=tfit_gmi, t_pred=tpred_gmi, n_train=len(Xtr), n_test=len(Xte), K=len(Xtr)),
        dict(tag=tag, method="GM-centroid",    acc=acc_gmc, t_fit=tfit_gmc, t_pred=tpred_gmc, n_train=len(Xtr), n_test=len(Xte), K=2),
    ]

    # ---------- Budget-matched kNN ----------
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

    # ---------- Smoothness maps ----------
    xx, yy, _ = make_grid(Xtr, Xte, n=300, pad=1.0)
    # p(class=1) maps
    p_knn = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
    p_ls  = probs_labels_spreading_transductive(Xtr, ytr, np.c_[xx.ravel(), yy.ravel()], n_neighbors=10, alpha=0.2)[:,1].reshape(xx.shape)
    p_gm  = probs_gm(gm, np.c_[xx.ravel(), yy.ravel()], 2)[:,1].reshape(xx.shape)
    p_gm_flat  = probs_gm(gm_flat, np.c_[xx.ravel(), yy.ravel()], 2)[:,1].reshape(xx.shape)

    sm_knn = decision_smoothness(p_knn, xx, yy)
    sm_ls  = decision_smoothness(p_ls,  xx, yy)
    sm_gm  = decision_smoothness(p_gm,  xx, yy)
    sm_gm_flat  = decision_smoothness(p_gm_flat,  xx, yy)
    smooth_rows = [
        dict(tag=tag, method="GM",             smooth=sm_gm),
        dict(tag=tag, method="GM-flat",             smooth=sm_gm_flat),
        dict(tag=tag, method="kNN",            smooth=sm_knn),
        dict(tag=tag, method="LabelSpreading", smooth=sm_ls),
    ]

    # visualize gradient magnitude
    gm_knn = grad_magnitude_map(p_knn, xx, yy)
    gm_ls  = grad_magnitude_map(p_ls,  xx, yy)
    gm_gm  = grad_magnitude_map(p_gm,  xx, yy)
    vmax = np.percentile(np.r_[gm_knn.ravel(), gm_ls.ravel(), gm_gm.ravel()], 99)

    fig_s, axs_s = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    axs_s[0].imshow(gm_gm, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                    cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axs_s[0].set_title(f"GM ∥∇p∥ (mean²={sm_gm:.3f})")
    axs_s[1].imshow(gm_knn, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                    cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axs_s[1].set_title(f"kNN ∥∇p∥ (mean²={sm_knn:.3f})")
    axs_s[2].imshow(gm_ls, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                    cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axs_s[2].set_title(f"LabelSpreading ∥∇p∥ (mean²={sm_ls:.3f})")
    for ax in axs_s: ax.set_xticks([]); ax.set_yticks([])
    fig_s.colorbar(axs_s[2].images[0], ax=axs_s.ravel().tolist(), shrink=0.85, pad=0.02)
    smooth_path = str((Path(outdir) / "plots" / f"{tag}_smoothness.png").resolve())
    plt.savefig(smooth_path, dpi=220); plt.close(fig_s)

    # decision plots
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    plot_decision(axs[0], lambda Z: gm.predict(Z, 2), Xtr, ytr, Xte, yte, f"GM ({acc_gm:.3f})")
    plot_decision(axs[1], lambda Z: baseline_knn(k=10).fit(Xtr, ytr).predict(Z), Xtr, ytr, Xte, yte, f"kNN ({acc_knn:.3f})")
    plot_decision(axs[2], lambda Z: (probs_labels_spreading_transductive(Xtr, ytr, Z, n_neighbors=10, alpha=0.2)).argmax(1),
                  Xtr, ytr, Xte, yte, f"LabelSpreading ({acc_ls:.3f})")
    fig_path = str((Path(outdir) / "plots" / f"{tag}.png").resolve())
    plt.savefig(fig_path, dpi=220); plt.close(fig)

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
    ap.add_argument("--backbone", type=str, choices=["resnet18","resnet50"], default="resnet50")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2, help="Linear head epochs (backbone frozen)")
    ap.add_argument("--imbalance", type=float, default=1.0)
    ap.add_argument("--budget-match", action="store_true")
    args = ap.parse_args()

    device = _set_torch()
    ensure_dir(args.outdir)
    ensure_dir(args.cache_dir)

    with section("Load IDC subsets"):
        tr_hf, te_hf = load_idc_subset(
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
            cache_dir=args.cache_dir,
            seed=args.seed
        )

    tf_tr = T.Compose([
        T.Resize((96,96)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tf_ev = T.Compose([
        T.Resize((96,96)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    dl_tr = DataLoader(IDCWrap(tr_hf, tf_tr), batch_size=args.batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=False)
    dl_te = DataLoader(IDCWrap(te_hf, tf_ev), batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, drop_last=False)

    with section(f"Backbone: {args.backbone} (head epochs={args.epochs})"):
        encoder, head, feat_dim = build_backbone(args.backbone, device=device, train_fc=True)
        if args.epochs > 0:
            train_head(encoder, head, dl_tr, device, epochs=args.epochs, lr=1e-3, wd=1e-4)

    with section("Extract features"):
        Xtr_f, ytr = extract_features(encoder, dl_tr, device)
        Xte_f, yte = extract_features(encoder, dl_te, device)


    with section("UMAP to 2-D (for smoothness/plots)"):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=args.seed
        )
        reducer.fit(Xtr_f)
        Xtr_2d = reducer.transform(Xtr_f)
        Xte_2d = reducer.transform(Xte_f)


    with section("Run IDC block (2-D)"):
        tag = "idc_umap2d"
        rows, cal_rows, smooth_rows, fig_path = run_idc_block(
            tag, Xtr_2d, ytr, Xte_2d, yte,
            outdir=args.outdir, seed=args.seed,
            imbalance=args.imbalance, budget_match=args.budget_match
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
