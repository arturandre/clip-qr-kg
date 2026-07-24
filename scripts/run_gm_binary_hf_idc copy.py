#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GM on IDC (HF) — Binary-only, calibration-aware evaluation on deep features.

- Loads IDC from Hugging Face: dbzadnen/breast-histopathology-images
- Trains a light classifier head (ResNet18 by default) for a few epochs
- Converts the trained net into a pure feature encoder
- Extracts TRAIN / TEST embeddings (balanced per class by sampling caps)
- Runs GM + baselines + LabelSpreading on the embeddings
- Reports Accuracy + NLL + Brier + ECE for all methods (no temperature)

Usage:
  python scripts/run_gm_binary_hf_idc.py \
    --outdir runs_idc_hf \
    --backbone resnet18 \
    --epochs 5 --batch-size 256 --num-workers 4 \
    --train-per-class 20000 --eval-per-class 20000 \
    --seed 0 --imbalance 1
"""

import os, math, argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
import torchvision.transforms as T
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from rich.console import Console
from rich.panel import Panel

import os, sys
# --- make project root importable (../) so 'graphmemory' is found ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==== import your existing utilities from the toy script ====
# Adjust this import path/name to match your repo.
from scripts.run_gm import (
    GMConfig, GraphMemory, gm_flat_from, gm_centroid,
    baseline_linear, baseline_knn,
    probs_labels_spreading_transductive, probs_gm,
    all_metrics, apply_long_tail, rows_to_table
)

console = Console()

# ---------------------------
# HF IDC loader → embeddings
# ---------------------------

class _WrapHFDataset(Dataset):
    def __init__(self, ds_split, keep_idx, tf):
        self.ds_split = ds_split
        self.keep = np.array(keep_idx, dtype=int)
        self.tf = tf
    def __len__(self): return len(self.keep)
    def __getitem__(self, i):
        r = self.ds_split[int(self.keep[i])]
        img = r['image'].convert('RGB')
        y = int(r['label'])
        return self.tf(img), y

def _get_resnet(backbone: str):
    backbone = backbone.lower()
    if backbone == "resnet18":
        m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError("--backbone must be resnet18|resnet50 for this HF script")
    return m

@torch.inference_mode()
def _extract_encoder_from_classifier(trained_model: nn.Module) -> nn.Module:
    """Take a trained ResNet classifier and return the conv trunk + avgpool as an encoder."""
    enc = _get_resnet("resnet18") if isinstance(trained_model, tv.models.ResNet) and trained_model.fc.in_features == 512 \
          else _get_resnet("resnet50")
    # copy weights back
    enc.load_state_dict(trained_model.state_dict(), strict=False)
    # encoder = everything except final fc
    encoder = nn.Sequential(
        *(list(enc.children())[:-1])  # conv stem → avgpool
    ).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder

@torch.inference_mode()
def _featurize(encoder: nn.Module, dl: DataLoader, device: str):
    Xs, Ys = [], []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        feats = encoder(xb).flatten(1)  # (N, D)
        Xs.append(feats.cpu().numpy())
        Ys.append(yb.numpy())
    X = np.concatenate(Xs, 0); y = np.concatenate(Ys, 0)
    return X, y

def load_idc_embeddings_hf(
    train_per_class=20000,
    eval_per_class=20000,
    backbone="resnet18",
    epochs=5,
    batch_size=256,
    num_workers=8,
    prefetch_factor=4,
    device="cuda",
    seed=42,
):
    import torchvision as tv
    import torchvision.transforms as T
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    torch.backends.cudnn.benchmark = True  # speed up convs for fixed shapes
    use_cuda = (device == "cuda") and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")

    rng = np.random.RandomState(seed)
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

    print('Loading dataset')
    ds = load_dataset("dbzadnen/breast-histopathology-images")

    def strat_indices(split, k_per_class):
        lab = np.array([r["label"] for r in ds[split]])
        i0 = np.where(lab==0)[0]; i1 = np.where(lab==1)[0]
        k0 = min(k_per_class, len(i0)); k1 = min(k_per_class, len(i1))
        keep = np.concatenate([rng.choice(i0, k0, replace=False),
                               rng.choice(i1, k1, replace=False)])
        rng.shuffle(keep)
        return keep

    class Wrap(Dataset):
        def __init__(self, split, keep_idx, tf):
            self.split = ds[split]
            self.keep = np.array(keep_idx, dtype=int)
            self.tf = tf
        def __len__(self): return len(self.keep)
        def __getitem__(self, i):
            r = self.split[int(self.keep[i])]
            img = r["image"].convert("RGB")
            y = int(r["label"])
            return self.tf(img), y

    tr_idx = strat_indices("train", train_per_class)
    te_idx = strat_indices("test", eval_per_class)

    print('Loading training dataloader')
    dl_tr = DataLoader(
        Wrap("train", tr_idx, tf_tr),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_cuda,
        persistent_workers=(num_workers > 0), prefetch_factor=prefetch_factor
    )
    print('Loading test dataloader')
    dl_te = DataLoader(
        Wrap("test", te_idx, tf_ev),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda,
        persistent_workers=(num_workers > 0), prefetch_factor=prefetch_factor
    )

    # --- Backbone with 2-class head for quick finetune ---
    print(f'Loading backbone: {backbone}')
    if backbone.lower() == "resnet18":
        m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone.lower() == "resnet50":
        m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError("backbone must be resnet18 or resnet50")

    in_dim = m.fc.in_features
    m.fc = nn.Linear(in_dim, 2)
    m = m.to(dev)
    m.to(memory_format=torch.channels_last)  # better Tensor Core paths
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("Training the backbone")
    m.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for xb, yb in dl_tr:
            xb = xb.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                logits = m(xb)
                loss = ce(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()); n += 1
        print(f"[ep {ep+1}/{epochs}] loss={tot/max(1,n):.4f}")

    # --- Build encoder directly from the trained model (no reloading) ---
    print("Building encoder")
    m.eval()
    encoder = nn.Sequential(*(list(m.children())[:-1])).to(dev).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print("Encoder builded")

    @torch.inference_mode()
    def fe(dl):
        Xs, Ys = [], []
        for xb, yb in dl:
            xb = xb.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                feats = encoder(xb)           # (N, C, 1, 1)
            feats = feats.flatten(1).float()  # (N, D)
            Xs.append(feats.cpu().numpy())
            Ys.append(yb.numpy())
        return np.concatenate(Xs, 0), np.concatenate(Ys, 0)

    Xtr, Ytr = fe(dl_tr)
    Xte, Yte = fe(dl_te)

    # Standardize on TRAIN
    scaler_np = StandardScaler().fit(Xtr)
    Xtr = scaler_np.transform(Xtr)
    Xte = scaler_np.transform(Xte)

    return Xtr, Ytr.astype(int), Xte, Yte.astype(int)

# ---------------------------
# Evaluation on embeddings
# ---------------------------

def evaluate_embeddings_binary(Xtr, ytr, Xte, yte, outdir: str, seed: int, imbalance: float):
    os.makedirs(outdir, exist_ok=True)
    if imbalance and imbalance > 1.0:
        Xtr, ytr = apply_long_tail(Xtr, ytr, imbalance, seed)
        console.print(f"[yellow]Applied long-tail[/yellow] (ratio≈{imbalance}).")

    # ==== Fit methods ====
    # Linear
    print("Running baseline linear")
    t0=time.perf_counter(); lin=baseline_linear().fit(Xtr,ytr); tfit_lin=time.perf_counter()-t0
    t0=time.perf_counter(); y_lin=lin.predict(Xte); tpred_lin=time.perf_counter()-t0
    P_lin=lin.predict_proba(Xte)

    # kNN
    print("Running baseline knn")
    t0=time.perf_counter(); knn=baseline_knn(k=10).fit(Xtr,ytr); tfit_knn=time.perf_counter()-t0
    t0=time.perf_counter(); y_knn=knn.predict(Xte); tpred_knn=time.perf_counter()-t0
    P_knn=knn.predict_proba(Xte)

    # LabelSpreading (transductive)
    print("Running baseline LabelSpreading")
    t0=time.perf_counter(); P_ls=probs_labels_spreading_transductive(Xtr,ytr,Xte,n_neighbors=15,alpha=0.2); t_ls=time.perf_counter()-t0
    y_ls = P_ls.argmax(1)

    print("Running GM + degenerates")
    # GM + degenerates
    print("Running GM base")
    Kproto = max(32, min(256, len(Xtr)//20))
    #gm_cfg = GMConfig(K=Kproto, knn_graph=15, attach_k=15, alpha=0.1, beta=1.5)
    gm_cfg = GMConfig(
        K=max(32, min(256, len(Xtr)//20)),
        knn_graph=12,
        attach_k=10,
        alpha=0.1,
        beta=1.5,
        use_minibatch_kmeans=True,
        kmeans_batch_size=8192,
        reliability_sample_cap=512,
        disable_instability=True,     # turn on if you really need it
        diffusion_iters=20,           # avoids matrix inverse (fast + stable)
        dtype="float32"
    )

    t0=time.perf_counter(); gm=GraphMemory(gm_cfg).fit(Xtr,ytr); tfit_gm=time.perf_counter()-t0
    t0=time.perf_counter(); y_gm=gm.predict(Xte,2); tpred_gm=time.perf_counter()-t0
    P_gm=probs_gm(gm,Xte,2)

    print("Running GM flat (no edges)")
    gm_f = gm_flat_from(gm)
    t0=time.perf_counter(); y_gmf=gm_f.predict(Xte,2); tpred_gmf=time.perf_counter()-t0
    P_gmf=probs_gm(gm_f,Xte,2)

    print("Running GM centroid (1 prototype per class)")
    t0=time.perf_counter(); gmc=gm_centroid(Xtr,ytr); tfit_gmc=time.perf_counter()-t0
    t0=time.perf_counter(); y_gmc=gmc.predict(Xte,2); tpred_gmc=time.perf_counter()-t0
    P_gmc=probs_gm(gmc,Xte,2)

    # ==== Acc summary ====
    rows = [
        dict(tag="IDC(HF)", method="Linear",         acc=float((y_lin==yte).mean()),  t_fit=tfit_lin,  t_pred=tpred_lin,  n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag="IDC(HF)", method="kNN",            acc=float((y_knn==yte).mean()),  t_fit=tfit_knn,  t_pred=tpred_knn,  n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag="IDC(HF)", method="LabelSpreading", acc=float((y_ls==yte).mean()),   t_fit=0.0,       t_pred=t_ls,       n_train=len(Xtr), n_test=len(Xte), K="-"),
        dict(tag="IDC(HF)", method="GM",             acc=float((y_gm==yte).mean()),   t_fit=tfit_gm,   t_pred=tpred_gm,   n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        dict(tag="IDC(HF)", method="GM-flat",        acc=float((y_gmf==yte).mean()),  t_fit=0.0,       t_pred=tpred_gmf,  n_train=len(Xtr), n_test=len(Xte), K=Kproto),
        # dict(tag="IDC(HF)", method="GM-instance",    acc=float((y_gmi==yte).mean()),  t_fit=tfit_gmi,  t_pred=tpred_gmi,  n_train=len(Xtr), n_test=len(Xte), K=len(Xtr)),
        dict(tag="IDC(HF)", method="GM-centroid",    acc=float((y_gmc==yte).mean()),  t_fit=tfit_gmc,  t_pred=tpred_gmc,  n_train=len(Xtr), n_test=len(Xte), K=2),
    ]

    # ==== Calibration summary (native probs) ====
    cal_rows = [
        dict(tag="IDC(HF)", method="Linear",         **all_metrics(P_lin,  yte)),
        dict(tag="IDC(HF)", method="kNN",            **all_metrics(P_knn,  yte)),
        dict(tag="IDC(HF)", method="LabelSpreading", **all_metrics(P_ls,   yte)),
        dict(tag="IDC(HF)", method="GM",             **all_metrics(P_gm,   yte)),
        dict(tag="IDC(HF)", method="GM-flat",        **all_metrics(P_gmf,  yte)),
        # dict(tag="IDC(HF)", method="GM-instance",    **all_metrics(P_gmi,  yte)),
        dict(tag="IDC(HF)", method="GM-centroid",    **all_metrics(P_gmc,  yte)),
    ]

    # Save CSVs
    Path(outdir).mkdir(parents=True, exist_ok=True)
    acc_csv = Path(outdir) / "idc_hf_summary.csv"
    with acc_csv.open("w", newline="") as f:
        import csv
        wr = csv.writer(f)
        wr.writerow(["tag","method","acc","t_fit","t_pred","n_train","n_test","K"])
        for r in rows:
            wr.writerow([r["tag"], r["method"], f'{r["acc"]:.6f}',
                         f'{r["t_fit"]:.6f}', f'{r["t_pred"]:.6f}',
                         r["n_train"], r["n_test"], r["K"]])

    cal_csv = Path(outdir) / "idc_hf_calibration.csv"
    with cal_csv.open("w", newline="") as f:
        import csv
        wr = csv.writer(f)
        wr.writerow(["tag","method","acc","nll","brier","ece"])
        for r in cal_rows:
            wr.writerow([r["tag"], r["method"],
                         f'{r["acc"]:.6f}', f'{r["nll"]:.6f}',
                         f'{r["brier"]:.6f}', f'{r["ece"]:.6f}'])

    console.print(rows_to_table(rows))
    console.print(Panel.fit(
        f"[bold green]Saved[/bold green] ACC → {acc_csv.resolve()}\n"
        f"[bold green]Saved[/bold green] CAL → {cal_csv.resolve()}"
    ))

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="runs_idc_hf")
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--train-per-class", type=int, default=20000, help="Cap per class for TRAIN split")
    ap.add_argument("--eval-per-class", type=int, default=20000, help="Cap per class for TEST split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--imbalance", type=float, default=1.0)
    args = ap.parse_args()

    # HF → embeddings
    print("Computing embeddings")
    Xtr, ytr, Xte, yte = load_idc_embeddings_hf(
        train_per_class=args.train_per_class,
        eval_per_class=args.eval_per_class,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )

    # Evaluate (binary)
    print("Evaluating embeddings")
    evaluate_embeddings_binary(
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
        outdir=args.outdir, seed=args.seed, imbalance=args.imbalance
    )

if __name__ == "__main__":
    main()
