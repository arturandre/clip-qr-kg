# FILE: scripts/run_toy.py
"""Synthetic 2D experiments using a unified GraphMemory.
Usage:
  python scripts/run_toy.py --dataset moons --noise 0.2 --n-samples 2000 --protos 24 --seed 7
Produces a PNG with decision regions and prototype graph overlay.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans



import os, sys
# --- make project root importable (../) so 'graphmemory' is found ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graphmemory.graph_memory import GraphMemory, GraphMemoryConfig, _l2n

def build_data(name: str, n: int, noise: float, seed: int):
    if name == 'moons':
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    elif name == 'blobs':
        X, y = make_blobs(n_samples=n, centers=6, cluster_std=1.0+noise, random_state=seed)
        y = (y % 2)  # binary for visualization parity
    else:
        raise ValueError('dataset not supported')
    Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    return Xtr, Xte, Ytr, Yte


def plot_regions(ax, clf, X, title: str):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    ZZ = clf(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, ZZ, alpha=0.2, levels=[-0.5,0.5,1.5], colors=['#4C72B0','#DD8452'])
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='moons')
    ap.add_argument('--noise', type=float, default=0.2)
    ap.add_argument('--n-samples', type=int, default=2000)
    ap.add_argument('--protos', type=int, default=24)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='runs/toy.png')
    args = ap.parse_args()

    Xtr, Xte, Ytr, Yte = build_data(args.dataset, args.n_samples, args.noise, args.seed)

    # Baselines
    logreg = LogisticRegression(max_iter=1000).fit(Xtr, Ytr)
    knn = KNeighborsClassifier(n_neighbors=15).fit(Xtr, Ytr)

    # GraphMemory on train
    cfg = GraphMemoryConfig(n_prototypes=args.protos, min_support=10, knn_edges=8, graph_alpha=0.6)
    gm = GraphMemory(cfg)
    gm.build(Xtr, Ytr, random_state=args.seed)
    gm.smooth_reliability()

    # Simple wrappers for decision region plotting
    def clf_logreg(X): return logreg.predict(X)
    def clf_knn(X): return knn.predict(X)
    def clf_gm(X):
        preds = [gm.predict_with_graph(x)[0] for x in X]
        return np.array(preds)

    # t-SNE for viz (samples only), then set prototype 2D by cluster means
    Xs = np.vstack([Xtr, Xte]); Ys = np.hstack([Ytr, Yte])
    X2 = TSNE(n_components=2, init='pca', perplexity=30, learning_rate='auto', random_state=args.seed).fit_transform(Xs)
    # Reconstruct train assignment for placement
    km = KMeans(n_clusters=gm.P.shape[0], n_init='auto', random_state=args.seed).fit(Xtr)
    assign_tr = km.labels_
    gm.set_positions(X2[:len(Xtr)], assign_tr)

    fig, ax = plt.subplots(1, 3, figsize=(14,4))
    plot_regions(ax[0], clf_logreg, Xs, 'Linear (LR)')
    plot_regions(ax[1], clf_knn, Xs, 'kNN')
    plot_regions(ax[2], clf_gm, Xs, 'GraphMemory')
    for a in ax:
        a.scatter(X2[:len(Xtr),0], X2[:len(Xtr),1], s=3, alpha=0.2, c=Ytr, cmap='coolwarm')
        if gm.P2 is not None:
            # draw edges
            rows, cols = gm.A.nonzero()
            for i, j in zip(rows, cols):
                if i < j:
                    (x1,y1),(x2,y2) = gm.P2[i], gm.P2[j]
                    a.plot([x1,x2],[y1,y2], color='k', alpha=0.2, linewidth=1)
            a.scatter(gm.P2[:,0], gm.P2[:,1], s=60+40*gm.reliability, edgecolor='k', linewidths=0.6,
                      c=gm.dom_class, cmap='coolwarm', marker='o')
        a.axis('off')
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches='tight', pad_inches=0)
    print('Saved', args.out)

if __name__ == '__main__':
    main()