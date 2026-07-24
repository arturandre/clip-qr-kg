import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

def l2n(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

def build_prototypes(X, Y, K=16, min_support=8, seed=7):
    km = KMeans(n_clusters=min(K,len(X)), n_init="auto", random_state=seed).fit(X)
    P, assign = km.cluster_centers_, km.labels_
    support = np.array([np.sum(assign==i) for i in range(P.shape[0])])
    keep = support >= min_support
    if keep.sum() < P.shape[0]:
        kept = np.where(keep)[0]
        P = P[keep]
        nn = NearestNeighbors(n_neighbors=1).fit(P)
        new_assign = np.full_like(assign, -1)
        for idx, lab in enumerate(assign):
            new_assign[idx] = np.where(kept==lab)[0][0] if lab in kept else nn.kneighbors(X[idx:idx+1], return_distance=False)[0,0]
        assign = new_assign
        support = np.array([np.sum(assign==i) for i in range(P.shape[0])])
    dom = np.zeros(P.shape[0], dtype=int)
    for i in range(P.shape[0]):
        ys = Y[assign==i]
        dom[i] = int(np.argmax(np.bincount(ys, minlength=2))) if len(ys)>0 else 0
    return P, assign, support, dom

def row_norm(W):
    d = np.array(W.sum(axis=1)).ravel() + 1e-8
    return (1.0/d)[:,None] * W

def diffuse_steps(S, z0, alpha=0.6, steps=(0,1,3,10)):
    zs = []
    for t in steps:
        z = z0.copy()
        for _ in range(t):
            z = (1 - alpha) * z0 + alpha * S.dot(z)
        zs.append(z)
    return zs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--steps", nargs="+", type=int, default=[0,1,3,10])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k_attach", type=int, default=2)
    args = ap.parse_args()

    # data and prototypes
    X, Y = make_moons(n_samples=2000, noise=0.2, random_state=args.seed)
    P, assign, support, dom = build_prototypes(X, Y, K=args.K, min_support=8, seed=args.seed)

    # prototype graph (sparser to slow mixing)
    A = kneighbors_graph(l2n(P), n_neighbors=min(6, max(1,P.shape[0]-1)),
                         mode="connectivity", metric="cosine", include_self=False).toarray()
    # NOTE: no symmetrization -> slower, directional mixing
    S = row_norm(A)

    # pick a boundary query: nearest 2 prototypes with opposite classes if possible
    nnP2 = NearestNeighbors(n_neighbors=2).fit(P)
    idx = None
    for i in range(len(X)):
        d_idx = nnP2.kneighbors(X[i:i+1], return_distance=False)[0]
        i1, i2 = d_idx
        if dom[i1] != dom[i2]:
            idx = i; break
    if idx is None: idx = 0
    xq = X[idx]

    # query→prototype attachments (k_attach)
    nnPk = NearestNeighbors(n_neighbors=min(args.k_attach, P.shape[0])).fit(P)
    q_idx = nnPk.kneighbors(xq[None,:], return_distance=False)[0]
    # query affinity only to attached prototypes (sharper source)
    d_all = np.linalg.norm(P[q_idx] - xq, axis=1)
    beta = 30.0
    w_local = np.exp(-beta * d_all**2)
    z0 = np.zeros(P.shape[0])
    z0[q_idx] = w_local / (w_local.sum() + 1e-8)

    # diffuse for all steps, then use ONE global scale
    zs = diffuse_steps(S, z0, alpha=args.alpha, steps=args.steps)
    z_stack = np.stack(zs, axis=0)
    z_global_max = z_stack.max() + 1e-8  # single scale across panels

    # plot each step
    cmap = get_cmap("viridis")
    for t, z in zip(args.steps, zs):
        # no per-panel normalization; use the same vmax
        z_vis = z / z_global_max

        plt.figure(figsize=(6.2,5.2))
        # samples (faint)
        plt.scatter(X[Y==0,0], X[Y==0,1], s=4, alpha=0.18, label="Class 0")
        plt.scatter(X[Y==1,0], X[Y==1,1], s=4, alpha=0.18, label="Class 1")

        # prototype graph edges
        rows, cols = np.nonzero(A)
        for i,j in zip(rows, cols):
            if i != j:
                plt.plot([P[i,0], P[j,0]], [P[i,1], P[j,1]], linewidth=0.8, alpha=0.22, color='k')

        # query→prototype attachments (dashed)
        for j in q_idx:
            plt.plot([xq[0], P[j,0]], [xq[1], P[j,1]], linestyle="--", linewidth=1.2, alpha=0.6)

        # activation as color + halo size (fixed global scale)
        sizes = 40 + 80 * (support / (support.max()+1e-8))
        halo = 300 * z_vis
        # halos first
        plt.scatter(P[:,0], P[:,1], s=halo, alpha=0.22, c=z_vis, cmap=cmap, marker='o')
        # cores by dominant class
        plt.scatter(P[dom==0,0], P[dom==0,1], s=sizes[dom==0], marker='o', edgecolor='k', linewidths=0.5, label="Proto dom 0")
        plt.scatter(P[dom==1,0], P[dom==1,1], s=sizes[dom==1], marker='X', edgecolor='k', linewidths=0.5, label="Proto dom 1")

        # query
        plt.scatter([xq[0]],[xq[1]], marker='*', s=180, label=f"Query (t={t})")

        # tiny class-mass bars to make the change obvious
        mass0 = float(z[dom==0].sum()); mass1 = float(z[dom==1].sum())
        total = mass0 + mass1 + 1e-8
        plt.text(0.02, 0.98, f"mass: C0={mass0/total:.2f}  C1={mass1/total:.2f}",
                 transform=plt.gca().transAxes, ha='left', va='top')

        plt.legend(loc="lower right", fontsize=8, frameon=True)
        plt.axis('off'); plt.gca().set_frame_on(False)
        plt.tight_layout()
        out = f"gm_step_{t}.png"
        plt.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0)
        plt.close()
        print("Saved", out)

if __name__ == "__main__":
    main()
