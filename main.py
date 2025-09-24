#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python main.py --dataset cifar10 --encoder vit_b_16 --baselines linear,knn,lp \
#    --batch-size 256 --cache-dir ./cache --metrics acc,ece,nll --ood-dataset stl10

"""
Baseline runner for Prototype-Relation Graph Memory paper.

What this file provides (vanilla baselines only):
- Embedding extraction & caching (torchvision encoders; frozen)
- Baselines: Linear Probe (multinomial LR), kNN (uniform vote), Label Propagation (sklearn)
- Metrics: Accuracy, NLL, ECE; optional OOD AUROC via MSP from the same model
- Results CSV + console table
- DkNN stub + minimal GraphMemoryClassifier scaffold (no tweaks)

Deliberate non-tweaks (to keep baselines vanilla):
- kNN uses uniform voting (no distance weights)
- OOD uses MSP only (no energy/ODIN/etc.)
- LP uses sklearn LabelPropagation with either 'rbf' (default) or 'knn' kernel
"""

import argparse, os, random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision as tv
import torchvision.transforms as T

from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# --------------------------- Utils -------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def np_softmax(logits, axis=1):
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

# ------------------------ Datasets & Transforms ------------------------------

DATASETS = {"cifar10", "cifar100", "stl10"}  # extend as needed

def make_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

def load_dataset(name: str, root: str, split: str, transform):
    name = name.lower()
    if name == "cifar10":
        train = (split == "train")
        return tv.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if name == "cifar100":
        train = (split == "train")
        return tv.datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    if name == "stl10":
        split_map = {"train": "train", "test": "test"}
        return tv.datasets.STL10(root=root, split=split_map[split], download=True, transform=transform)
    raise ValueError(f"Unknown dataset: {name}")

def num_classes_for(name: str) -> int:
    return {"cifar10": 10, "cifar100": 100, "stl10": 10}[name]

# -------------------------- Encoders -----------------------------------------

def build_encoder(name: str, pretrained: bool = True, freeze: bool = True, device=None):
    name = name.lower()
    device = device or get_device()
    if name == "resnet50":
        model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT if pretrained else None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "vit_b_16":
        model = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        feature_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
    elif name == "vit_b_32":
        model = tv.models.vit_b_32(weights=tv.models.ViT_B_32_Weights.DEFAULT if pretrained else None)
        feature_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
    else:
        raise ValueError(f"Unknown encoder: {name}")
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.eval().to(device)
    return model, feature_dim

@torch.no_grad()
def extract_embeddings(encoder, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for x, y in tqdm(loader, desc="Extracting", leave=False):
        x = x.to(device)
        out = encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        feats.append(out.detach().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)

def cache_path(cache_dir, dataset, encoder, split):
    return os.path.join(cache_dir, f"{dataset}__{encoder}__{split}.npz")

def get_or_make_embeddings(dataset, encoder_name, data_root, cache_dir,
                           batch_size, num_workers, img_size):
    device = get_device()
    encoder, _ = build_encoder(encoder_name, pretrained=True, freeze=True, device=device)
    tfm = make_transforms(img_size)
    out = {}
    for split in ["train", "test"]:
        cp = cache_path(cache_dir, dataset, encoder_name, split)
        if os.path.exists(cp):
            z = np.load(cp); out[split] = (z["X"], z["y"])
        else:
            ds = load_dataset(dataset, data_root, split, tfm)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
            X, y = extract_embeddings(encoder, loader, device)
            ensure_dir(cache_dir)
            np.savez_compressed(cp, X=X, y=y)
            out[split] = (X, y)
    return out["train"], out["test"]

# ---------------------------- Metrics ----------------------------------------

def compute_accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))

def compute_nll(y_true, proba) -> float:
    eps = 1e-12
    p = np.clip(proba, eps, 1.0 - eps)
    return float(log_loss(y_true, p, labels=np.arange(p.shape[1])))

def compute_ece(y_true, proba, n_bins: int = 15) -> float:
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)

def compute_ood_auroc(proba_in, proba_out) -> float:
    in_msp = proba_in.max(axis=1)
    out_msp = proba_out.max(axis=1)
    if out_msp.mean() > in_msp.mean():
        print("[WARN] OOD MSP > IN MSP on average; check pipeline/scaling/model reuse.")
    y = np.concatenate([np.ones_like(in_msp), np.zeros_like(out_msp)])  # 1 = IN, 0 = OUT
    s = np.concatenate([in_msp, out_msp])
    return float(roc_auc_score(y, s))

# -------------------------- Baselines (vanilla) ------------------------------

def run_linear_probe(Xtr, ytr, Xte, yte,
                     C_values=(0.01, 0.1, 1.0, 10.0),
                     max_iter=1000, tol=1e-4, cv=3, scale=True):
    """
    Vanilla linear probe: multinomial LR with L2, lbfgs, CV over C by NLL.
    Standardizes features by default (common practice). Disable via scale=False if needed.
    """
    print("Running linear probe")
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(Xtr.astype("float32"))
        Xte_s = scaler.transform(Xte.astype("float32"))
    else:
        scaler = None
        Xtr_s, Xte_s = Xtr, Xte

    clf = LogisticRegressionCV(
        Cs=list(C_values),
        cv=cv,
        solver="lbfgs",
        penalty="l2",
        scoring="neg_log_loss",  # choose C by NLL
        max_iter=max_iter,
        tol=tol,
        n_jobs=-1,
        refit=True
    )
    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)
    return {
        "acc": compute_accuracy(yte, proba.argmax(1)),
        "nll": compute_nll(yte, proba),
        "ece": compute_ece(yte, proba),
        "C": float(clf.C_[0]),
        "proba": proba,
        "estimator": clf,
        "scaler": scaler,
    }

def run_knn(Xtr, ytr, Xte, yte, k_values=(1, 5, 10, 20), metric="cosine"):
    """
    Vanilla kNN: uniform voting (no distance weights).
    Metric defaults to cosine (standard for embedding spaces).
    """
    best = None
    best_est = None
    for k in tqdm(k_values, desc="Running kNN"):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)  # uniform weights (default)
        knn.fit(Xtr, ytr)
        proba = knn.predict_proba(Xte)
        acc = compute_accuracy(yte, proba.argmax(1))
        nll = compute_nll(yte, proba)
        ece = compute_ece(yte, proba)
        cand = {"acc": acc, "nll": nll, "ece": ece, "k": float(k), "proba": proba}
        if best is None or cand["nll"] < best["nll"]:
            best, best_est = cand, knn
    best.update({"estimator": best_est})
    return best

def run_label_propagation(Xtr, ytr, Xte, yte,
                          n_neighbors=10,
                          gamma=None,
                          kernel="rbf",        # default to sklearn's typical RBF
                          scale=True,
                          max_iter=1000):
    """
    Vanilla Label Propagation (sklearn).
    - kernel='rbf' (default) or 'knn'
    - When 'rbf', gamma can be set; if None, use 1/d heuristic after scaling.
    - Transductive: we fit on [train + test] with test labels = -1.
    """
    if scale:
        scaler = StandardScaler().fit(Xtr.astype("float32"))
        Xtr_s = scaler.transform(Xtr.astype("float32"))
        Xte_s = scaler.transform(Xte.astype("float32"))
    else:
        Xtr_s, Xte_s = Xtr, Xte

    X_all = np.concatenate([Xtr_s, Xte_s], axis=0)
    y_all = np.concatenate([ytr, -np.ones_like(yte)])

    if kernel == "knn":
        lp = LabelPropagation(kernel="knn", n_neighbors=n_neighbors, max_iter=max_iter)
    elif kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / Xtr_s.shape[1]  # simple heuristic
        lp = LabelPropagation(kernel="rbf", gamma=float(gamma), max_iter=max_iter)
    else:
        raise ValueError("kernel must be 'rbf' or 'knn'.")

    lp.fit(X_all, y_all)

    proba_all = lp.label_distributions_
    proba_te = proba_all[len(ytr):]
    classes_ = lp.classes_.astype(int)

    C = int(max(ytr.max(), yte.max())) + 1
    proba = np.zeros((len(yte), C), dtype=float)
    proba[:, classes_] = np.clip(proba_te, 1e-12, 1.0)

    yhat = proba.argmax(1)
    return {
        "acc": compute_accuracy(yte, yhat),
        "nll": compute_nll(yte, proba),
        "ece": compute_ece(yte, proba),
        "n_neighbors": n_neighbors,
        "gamma": gamma if kernel == "rbf" else None,
        "kernel": kernel,
        "proba": proba,
    }

def run_dknn_stub(*args, **kwargs):
    raise NotImplementedError(
        "DkNN requires multi-layer features. Extract intermediate activations at several layers "
        "and run kNN per layer with conformity scoring. TODO for future baseline."
    )

# ---------------------- Graph Memory (YOUR METHOD) ---------------------------

class GraphMemoryClassifier:
    """
    Minimal scaffold (no tweaks yet).
    - fit(): build per-class KMeans prototypes; reliability=1.0 (placeholder)
    - predict_proba(): reliability-weighted nearest-prototype voting (no propagation)
    Replace with your full Prototype-Relation Graph Memory.
    """
    def __init__(self,
                 prototypes_per_class: int = 4,
                 neighbor_k_attach: int = 10,
                 graph_sparsity_M: int = 20,
                 alpha: float = 1/3, beta: float = 1/3, gamma: float = 1/3,
                 geometry_bandwidth: float = 1.0,
                 reliability_rho: float = 1.0,
                 seed: int = 42):
        self.Ky = prototypes_per_class
        self.K_attach = neighbor_k_attach
        self.M = graph_sparsity_M
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.tau = geometry_bandwidth
        self.rho = reliability_rho
        self.seed = seed
        self.proto_mu_ = None
        self.proto_y_ = None
        self.proto_reliability_ = None

    def fit(self, Xtr: np.ndarray, ytr: np.ndarray,
            Xval: Optional[np.ndarray] = None, yval: Optional[np.ndarray] = None):
        from sklearn.cluster import KMeans
        rng = np.random.RandomState(self.seed)
        classes = np.unique(ytr)
        protos, proto_labels = [], []
        for cls in classes:
            Xc = Xtr[ytr == cls]
            k = min(self.Ky, len(Xc))
            km = KMeans(n_clusters=k, n_init=10, random_state=rng).fit(Xc)
            protos.append(km.cluster_centers_)
            proto_labels.append(np.full(k, cls, dtype=int))
        self.proto_mu_ = np.concatenate(protos, axis=0)
        self.proto_y_  = np.concatenate(proto_labels, axis=0)
        self.proto_reliability_ = np.ones(self.proto_mu_.shape[0], dtype=float)  # placeholder
        return self

    def predict_proba(self, Xte: np.ndarray) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        P = self.proto_mu_.shape[0]
        C = int(self.proto_y_.max()) + 1
        S = cosine_similarity(Xte, self.proto_mu_)  # (n, P)
        K = min(self.K_attach, P)
        idx = np.argpartition(-S, kth=K-1, axis=1)[:, :K]
        sims = np.take_along_axis(S, idx, axis=1)
        rel  = self.proto_reliability_[idx] ** self.rho
        lab  = self.proto_y_[idx]
        weights = sims * rel
        proba = np.zeros((Xte.shape[0], C), dtype=float)
        for c in range(C):
            proba[:, c] = (weights * (lab == c)).sum(axis=1)
        proba_sum = proba.sum(axis=1, keepdims=True) + 1e-12
        return proba / proba_sum

# ---------------------------- Runner -----------------------------------------

def run_experiment(args):
    set_seed(args.seed)
    (Xtr, ytr), (Xte, yte) = get_or_make_embeddings(
        dataset=args.dataset, encoder_name=args.encoder, data_root=args.data_root,
        cache_dir=args.cache_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=args.img_size
    )
    results = []
    baselines = [b.strip().lower() for b in args.baselines.split(",")]

    # Optional OOD set
    Xood, yood = None, None
    if args.ood_dataset:
        (Xo_tr, yo_tr), (Xo_te, yo_te) = get_or_make_embeddings(
            dataset=args.ood_dataset, encoder_name=args.encoder, data_root=args.data_root,
            cache_dir=args.cache_dir, batch_size=args.batch_size, num_workers=args.num_workers,
            img_size=args.img_size
        )
        Xood, yood = Xo_te, yo_te

    # ---- Baselines ----
    if "linear" in baselines:
        res = run_linear_probe(Xtr, ytr, Xte, yte,
                               C_values=(0.01, 0.1, 1.0, 10.0),
                               max_iter=1000, tol=1e-4, cv=3, scale=True)
        if args.ood_dataset:
            if res["scaler"] is not None:
                Xood_s = res["scaler"].transform(Xood)
                proba_out = res["estimator"].predict_proba(Xood_s)
            else:
                proba_out = res["estimator"].predict_proba(Xood)
            res["ood_auroc_msp"] = compute_ood_auroc(res["proba"], proba_out)
        res.update(model="linear_probe")
        results.append(res)

    if "knn" in baselines:
        res = run_knn(Xtr, ytr, Xte, yte, k_values=(1, 5, 10, 20), metric="cosine")
        if args.ood_dataset:
            proba_out = res["estimator"].predict_proba(Xood)
            res["ood_auroc_msp"] = compute_ood_auroc(res["proba"], proba_out)
        res.update(model="knn")
        results.append(res)

    if "lp" in baselines:
        res = run_label_propagation(Xtr, ytr, Xte, yte,
                                    n_neighbors=args.lp_k,
                                    gamma=args.lp_gamma,
                                    kernel=args.lp_kernel,
                                    scale=True,
                                    max_iter=1000)
        # LP is transductive; omit AUROC unless refit on [train+test+ood]
        if args.ood_dataset:
            res["ood_auroc_msp"] = np.nan
        res.update(model="label_propagation")
        results.append(res)

    if "dknn" in baselines:
        try:
            _ = run_dknn_stub()
        except NotImplementedError as e:
            print(f"[WARN] DkNN not implemented: {e}")

    # ---- Graph Memory (minimal scaffold; not a baseline tweak) ----
    if "graphmemory" in baselines:
        gm = GraphMemoryClassifier(
            prototypes_per_class=args.gm_prototypes_per_class,
            neighbor_k_attach=args.gm_attach_k,
            graph_sparsity_M=args.gm_graph_M,
            alpha=args.gm_alpha, beta=args.gm_beta, gamma=args.gm_gamma,
            geometry_bandwidth=args.gm_tau,
            reliability_rho=args.gm_reliability_rho,
            seed=args.seed
        )
        gm.fit(Xtr, ytr)
        proba_in = gm.predict_proba(Xte)
        res = {
            "model": "graph_memory_minimal",
            "acc":  compute_accuracy(yte, proba_in.argmax(1)),
            "nll":  compute_nll(yte, proba_in),
            "ece":  compute_ece(yte, proba_in),
            "proba": proba_in
        }
        if args.ood_dataset:
            proba_out = gm.predict_proba(Xood)
            res["ood_auroc_msp"] = compute_ood_auroc(proba_in, proba_out)
        results.append(res)

    # ---- Save & print ----
    for r in results:
        r.pop("proba", None)
        r.pop("estimator", None)
        r.pop("scaler", None)
    df = pd.DataFrame(results).sort_values(by="nll")
    ensure_dir(args.out_dir)
    out_csv = os.path.join(args.out_dir, f"results__{args.dataset}__{args.encoder}.csv")
    df.to_csv(out_csv, index=False)
    print("\n=== RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")

# ---------------------------- CLI --------------------------------------------

def build_cli():
    p = argparse.ArgumentParser(description="Vanilla baselines for Graph Memory experiments")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(list(DATASETS)))
    p.add_argument("--ood-dataset", type=str, default=None, choices=sorted(list(DATASETS)))
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--cache-dir", type=str, default="./cache")
    p.add_argument("--out-dir", type=str, default="./runs")
    p.add_argument("--encoder", type=str, default="vit_b_16",
                   choices=["vit_b_16", "vit_b_32", "resnet50"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--baselines", type=str, default="linear,knn,lp",
                   help="comma-separated: linear,knn,lp,dknn,graphmemory")
    # LP params (vanilla)
    p.add_argument("--lp-k", type=int, default=10)
    p.add_argument("--lp-kernel", type=str, default="rbf", choices=["rbf", "knn"])
    p.add_argument("--lp-gamma", type=float, default=None)
    # Graph Memory params (scaffold only)
    p.add_argument("--gm-prototypes-per-class", type=int, default=4)
    p.add_argument("--gm-attach-k", type=int, default=10)
    p.add_argument("--gm-graph-M", type=int, default=20)
    p.add_argument("--gm-alpha", type=float, default=1/3)
    p.add_argument("--gm-beta", type=float, default=1/3)
    p.add_argument("--gm-gamma", type=float, default=1/3)
    p.add_argument("--gm-tau", type=float, default=1.0)
    p.add_argument("--gm-reliability-rho", type=float, default=1.0)
    # misc
    p.add_argument("--metrics", type=str, default="acc,ece,nll")
    p.add_argument("--seed", type=int, default=42)
    return p

if __name__ == "__main__":
    args = build_cli().parse_args()
    run_experiment(args)
