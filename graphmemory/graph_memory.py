# FILE: graphmemory/graph_memory.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import silhouette_samples
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import expit


def _l2n(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def _soft_bound(x: np.ndarray, ref: Optional[float] = None, scale: Optional[float] = None, invert: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if ref is None:
        ref = np.median(x)
    if scale is None:
        scale = np.percentile(x, 75) - np.percentile(x, 25)
    scale = max(scale, 1e-8)
    z = (x - ref) / scale
    s = expit(-z) if invert else expit(z)
    return s


@dataclass
class GraphMemoryConfig:
    n_prototypes: int = 24            # total prototypes (joint clustering)
    min_support: int = 30             # drop tiny clusters
    knn_edges: int = 12               # edges per prototype
    graph_alpha: float = 0.6          # diffusion strength for smoothing / voting
    neighbor_bary_k: int = 100        # for 2D placement (if used)
    reliability_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # sil, mar, instab, purity


class GraphMemory:
    """
    Unified prototype–graph memory built on embeddings X (N x D) with labels Y (N,)
    Provides: build(), reliability(), smooth_reliability(), predict_with_graph()
    Notes:
      - Joint KMeans over all classes (no per-class filtering)
      - Reliability uses sigmoid/soft-bounded metrics (stable in [0,1])
      - Edges in FEATURE space (cosine); diffusion for voting/smoothing
    """
    def __init__(self, cfg: GraphMemoryConfig):
        self.cfg = cfg
        self.P: Optional[np.ndarray] = None        # (P x D) prototype features
        self.P2: Optional[np.ndarray] = None       # (P x 2) optional 2D positions
        self.dom_class: Optional[np.ndarray] = None  # (P,) dominant class per proto
        self.purity: Optional[np.ndarray] = None
        self.entropy: Optional[np.ndarray] = None
        self.support: Optional[np.ndarray] = None
        self.reliability: Optional[np.ndarray] = None
        self.A = None                               # adjacency (csr)
        self._solve_lin = None                      # cached linear solver for (I - alpha S)
        self._S = None

    # ---------- BUILD ----------
    def build(self, X: np.ndarray, Y: np.ndarray, random_state: int = 42) -> None:
        X = np.asarray(X); Y = np.asarray(Y)
        K = min(self.cfg.n_prototypes, len(X))
        km = KMeans(n_clusters=K, n_init='auto', random_state=random_state).fit(X)
        P = km.cluster_centers_
        assign = km.labels_
        nP = P.shape[0]

        # Drop tiny clusters, reassign their members to nearest kept prototype
        support = np.array([np.sum(assign == i) for i in range(nP)])
        keep = support >= self.cfg.min_support
        if keep.sum() < nP:
            kept_ids = np.where(keep)[0]
            remap = {old: i for i, old in enumerate(kept_ids)}
            P = P[keep]
            nP = P.shape[0]
            assign2 = np.array([remap.get(a, -1) for a in assign])
            dropped = np.where(assign2 == -1)[0]
            if len(dropped) > 0:
                nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(_l2n(P))
                _, idx = nbrs.kneighbors(_l2n(X[dropped]), return_distance=True)
                assign2[dropped] = idx[:, 0]
            assign = assign2
            support = np.array([np.sum(assign == i) for i in range(nP)])

        # Composition
        C = int(np.max(Y)) + 1
        counts = np.zeros((nP, C), dtype=int)
        for i in range(nP):
            yi = Y[assign == i]
            for c in range(C):
                counts[i, c] = int(np.sum(yi == c))
        totals = counts.sum(axis=1) + 1e-8
        purity = (counts.max(axis=1) / totals).astype(float)
        p = counts / totals[:, None]
        entropy = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0), axis=1) / np.log(2)
        dom = counts.argmax(axis=1)

        # 2D placement: mean of members in an external 2D map (passed later via set_positions())
        # Here we keep a placeholder; caller may compute and set P2.

        # Edges in feature space (cosine)
        A = kneighbors_graph(_l2n(P), n_neighbors=max(1, min(self.cfg.knn_edges, nP - 1)),
                             mode='distance', metric='cosine', include_self=False)
        A = 0.5 * (A + A.T)
        self.P, self.A = P, A.tocsr()
        self.support, self.dom_class = support, dom
        self.purity, self.entropy = purity, entropy

        # Precompute row-normalized S and linear solver for diffusion
        d = np.array(self.A.sum(axis=1)).ravel() + 1e-8
        Dinv = diags(1.0 / d)
        self._S = Dinv @ self.A
        I = diags(np.ones(self.A.shape[0]))
        M = (I - self.cfg.graph_alpha * self._S).tocsr()
        self._solve_lin = lambda b: np.asarray(spsolve(M, b)).ravel()

        # Reliability (soft-bounded)
        self.reliability = self._compute_reliability(X, Y, assign)

    def set_positions(self, X2: np.ndarray, assign: np.ndarray) -> None:
        """Set 2D positions by mean of member coordinates from an external 2D layout X2."""
        nP = self.P.shape[0]
        P2 = np.zeros((nP, 2))
        for i in range(nP):
            idx = np.where(assign == i)[0]
            if len(idx) == 0:
                # fallback: nearest sample by cosine
                nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(_l2n(X2))
                j = nbrs.kneighbors(_l2n(self.P[i:i + 1]), return_distance=False)[0, 0]
                P2[i] = X2[j]
            else:
                P2[i] = X2[idx].mean(axis=0)
        self.P2 = P2

    # ---------- METRICS & RELIABILITY ----------
    def _compute_reliability(self, X: np.ndarray, Y: np.ndarray, assign: np.ndarray) -> np.ndarray:
        P = self.P; nP = P.shape[0]
        # Silhouette per sample on prototype assignments
        valid = [i for i in range(nP) if np.sum(assign == i) > 1]
        if len(valid) >= 2:
            sil_all = silhouette_samples(_l2n(X), assign, metric='cosine')
        else:
            sil_all = np.zeros(len(X))
        proto_sil = np.zeros(nP); proto_disp = np.zeros(nP)
        for i in range(nP):
            idx = np.where(assign == i)[0]
            proto_sil[i] = sil_all[idx].mean() if len(idx) > 0 else 0.0
            if len(idx) > 0:
                diffs = _l2n(X[idx]) - _l2n(P[i:i + 1])
                proto_disp[i] = np.sqrt((diffs ** 2).sum(axis=1).mean())
            else:
                proto_disp[i] = 0.0
        # Margin (gap to nearest rival center)
        nbrsP = NearestNeighbors(n_neighbors=min(3, nP), metric='cosine').fit(_l2n(P))
        distP, _ = nbrsP.kneighbors(_l2n(P), return_distance=True)
        proto_margin = distP[:, 1] if nP > 1 else np.ones(nP)
        # Instability via tiny noise
        rng = np.random.RandomState(123)
        Xp = _l2n(X + rng.normal(0, 0.01, size=X.shape))
        nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(_l2n(P))
        _, a2 = nn.kneighbors(Xp, return_distance=True)
        switch = (a2[:, 0] != assign).astype(np.float32)
        proto_instab = np.zeros(nP)
        for i in range(nP):
            idx = np.where(assign == i)[0]
            proto_instab[i] = switch[idx].mean() if len(idx) > 0 else 0.0
        # Soft-bounded metrics in [0,1]
        sil_b = (proto_sil + 1.0) / 2.0
        mar_b = _soft_bound(proto_margin)
        ins_b = _soft_bound(proto_instab, invert=True)
        pur_b = self.purity
        # Standardize each bounded metric before combining
        def z01(v):
            return (v - v.mean()) / (v.std() + 1e-8)
        w_sil, w_mar, w_ins, w_pur = self.cfg.reliability_weights
        zsum = (w_sil * z01(sil_b) + w_mar * z01(mar_b)
                - w_ins * z01(ins_b) + w_pur * z01(pur_b))
        return expit(zsum)

    # ---------- GRAPH UTILITIES ----------
    def smooth_reliability(self) -> np.ndarray:
        """Diffuse reliabilities over the prototype graph: r <- (I - alpha S)^{-1} ((1-alpha) r)."""
        r = self.reliability
        alpha = self.cfg.graph_alpha
        b = (1 - alpha) * r
        r_s = self._solve_lin(b)
        self.reliability = np.clip(r_s, 0, 1)
        return self.reliability

    def predict_with_graph(self, x_feat: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """Inductive prediction via prototype diffusion (ego-activation).
        Returns: (pred_class, class_scores[C], proto_activations[P])
        """
        P = _l2n(self.P)
        x = _l2n(x_feat[None, :])
        from sklearn.metrics.pairwise import cosine_distances
        d = cosine_distances(x, P).ravel()
        beta = 50.0
        wq = np.exp(-beta * d ** 2)
        z0 = wq * (self.reliability if self.reliability is not None else 1.0)
        z = self._solve_lin(z0)
        C = int(self.dom_class.max() + 1)
        Yp = np.eye(C)[self.dom_class]
        scores = z @ Yp
        return int(np.argmax(scores)), scores, z

    def cross_class_ambiguity(self) -> np.ndarray:
        """Compute cross-class edge mass ratio per prototype (0..1)."""
        W = self.A.tocsr()
        rows, cols = W.nonzero()
        vals = np.asarray(W[rows, cols]).ravel()
        cross = (self.dom_class[rows] != self.dom_class[cols]).astype(float)
        num = np.zeros(W.shape[0]); den = np.zeros(W.shape[0])
        for i, j, w, cr in zip(rows, cols, vals, cross):
            num[i] += (w if cr else 0.0)
            den[i] += w
        return num / (den + 1e-8)