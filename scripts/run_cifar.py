# FILE: scripts/run_cifar.py
"""CIFAR-10 pipeline using unified GraphMemory (frozen encoder -> embeddings -> eval).
Usage:
  python scripts/run_cifar.py --out runs/cifar10.csv
"""
import argparse, os
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss

import os, sys
# --- make project root importable (../) so 'graphmemory' is found ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graphmemory.graph_memory import GraphMemory, GraphMemoryConfig


def get_embeddings(device='cuda'):
    tf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    tr = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf)
    te = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf)
    dl_tr = DataLoader(tr, batch_size=256, shuffle=False, num_workers=4)
    dl_te = DataLoader(te, batch_size=256, shuffle=False, num_workers=4)
    enc = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.IMAGENET1K_V1).to(device).eval()
    def fe(dl):
        Xs, Ys = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                feats = enc(xb)
                Xs.append(feats.cpu().numpy()); Ys.append(yb.numpy())
        return np.concatenate(Xs,0), np.concatenate(Ys,0)
    Xtr, Ytr = fe(dl_tr); Xte, Yte = fe(dl_te)
    return Xtr, Ytr, Xte, Yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/cifar10.csv')
    args = ap.parse_args()

    Xtr, Ytr, Xte, Yte = get_embeddings(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Baselines
    lr = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, Ytr)
    knn = KNeighborsClassifier(n_neighbors=20).fit(Xtr, Ytr)

    # GraphMemory
    cfg = GraphMemoryConfig(n_prototypes=200, min_support=20, knn_edges=20, graph_alpha=0.6)
    gm = GraphMemory(cfg)
    gm.build(Xtr, Ytr, random_state=42)
    gm.smooth_reliability()

    # Evaluate
    def eval_model(predict_fn):
        preds = predict_fn(Xte)
        acc = accuracy_score(Yte, preds)
        # pseudo-probabilities for logloss via one-vs-all using scores if available
        return acc

    acc_lr = eval_model(lambda X: lr.predict(X))
    acc_knn = eval_model(lambda X: knn.predict(X))
    acc_gm  = eval_model(lambda X: np.array([gm.predict_with_graph(x)[0] for x in X]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('model,acc\n')
        f.write(f'linear,{acc_lr:.4f}\n')
        f.write(f'knn,{acc_knn:.4f}\n')
        f.write(f'graphmemory,{acc_gm:.4f}\n')
    print('Saved', args.out)

if __name__ == '__main__':
    main()

