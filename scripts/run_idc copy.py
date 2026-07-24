# FILE: scripts/run_idc.py
"""IDC (HF) pipeline with unified GraphMemory. Trains small head (ResNet18) to get embeddings,
then builds GraphMemory, computes reliability (soft-bounded), and exports static & interactive plots.
Usage:
  python scripts/run_idc.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import torch, torchvision as tv
import torchvision.transforms as T

import os, sys
# --- make project root importable (../) so 'graphmemory' is found ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graphmemory.graph_memory import GraphMemory, GraphMemoryConfig


def load_idc_embeddings(sample_per_class_train=20000, sample_per_class_eval=20000, device='cuda'):
    tf_tr = T.Compose([T.Resize((96,96)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_ev = T.Compose([T.Resize((96,96)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds = load_dataset('dbzadnen/breast-histopathology-images')
    def strat_indices(split, k):
        lab = np.array([r['label'] for r in ds[split]])
        i0 = np.where(lab==0)[0]; i1 = np.where(lab==1)[0]
        k0 = min(k, len(i0)); k1 = min(k, len(i1))
        rng = np.random.RandomState(42)
        keep = np.concatenate([rng.choice(i0, k0, replace=False), rng.choice(i1, k1, replace=False)])
        rng.shuffle(keep); return keep
    # Build Torch datasets on-the-fly
    class Wrap(torch.utils.data.Dataset):
        def __init__(self, split, tf, k):
            self.split=split; self.keep=strat_indices(split, k); self.tf=tf
        def __len__(self): return len(self.keep)
        def __getitem__(self, i):
            r = ds[self.split][int(self.keep[i])]
            img = r['image']; y = int(r['label'])
            img = img.convert('RGB'); img = self.tf(img)
            return img, y
    tr = Wrap('train', tf_tr, sample_per_class_train)
    va = Wrap('validation', tf_ev, sample_per_class_eval)
    te = Wrap('test', tf_ev, sample_per_class_eval)
    dl_tr = DataLoader(tr, batch_size=256, shuffle=True,  num_workers=4, pin_memory=True)
    dl_te = DataLoader(te, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Small head: ResNet18
    dev = device if torch.cuda.is_available() else 'cpu'
    m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    in_dim = m.fc.in_features
    m.fc = torch.nn.Linear(in_dim, 2)
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    m.train()
    for ep in range(5):  # short training for demo
        for xb, yb in dl_tr:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(); logits = m(xb); loss = ce(logits, yb); loss.backward(); opt.step()
    m.eval()

    # Feature extractor
    enc = tv.models.resnet18(weights=None)
    enc.fc = torch.nn.Linear(enc.fc.in_features, 2)
    enc.load_state_dict(m.state_dict())
    encoder = torch.nn.Sequential(*list(enc.children())[:-1]).to(dev).eval()

    def fe(dl):
        Xs, Ys = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(dev)
                feats = encoder(xb).flatten(1)
                Xs.append(feats.cpu().numpy())
                Ys.append(yb.cpu().numpy())
        return np.concatenate(Xs,0), np.concatenate(Ys,0)

    Xtr, Ytr = fe(dl_tr); Xte, Yte = fe(dl_te)
    return Xtr, Ytr, Xte, Yte


def main():
    Xtr, Ytr, Xte, Yte = load_idc_embeddings()

    cfg = GraphMemoryConfig(n_prototypes=24, min_support=30, knn_edges=12, graph_alpha=0.6)
    gm = GraphMemory(cfg)
    gm.build(Xtr, Ytr, random_state=42)
    gm.smooth_reliability()

    # t-SNE for visualization (samples only, train)
    X2 = TSNE(n_components=2, init='pca', perplexity=30, learning_rate='auto', random_state=42).fit_transform(Xtr)
    # place prototypes by mean of member coords
    # rebuild KMeans with same K to get train-assignments (for placement only)
    km = KMeans(n_clusters=gm.P.shape[0], n_init='auto', random_state=42).fit(Xtr)
    gm.set_positions(X2, km.labels_)

    # Static figure (axes off)
    plt.figure(figsize=(9,8))
    m0, m1 = (Ytr==0), (Ytr==1)
    plt.scatter(X2[m0,0], X2[m0,1], s=3, alpha=0.22, label='IDC-')
    plt.scatter(X2[m1,0], X2[m1,1], s=3, alpha=0.22, label='IDC+')
    rows, cols = gm.A.nonzero()
    for i,j in zip(rows, cols):
        if i<j:
            (x1,y1),(x2,y2) = gm.P2[i], gm.P2[j]
            plt.plot([x1,x2],[y1,y2], linewidth=1, alpha=0.3, color='k')
    plt.scatter(gm.P2[:,0], gm.P2[:,1], s=60+40*gm.reliability, edgecolor='k', linewidths=0.8,
                c=gm.dom_class, cmap='coolwarm', marker='o')
    plt.axis('off'); plt.gca().set_frame_on(False)
    os.makedirs('runs', exist_ok=True)
    plt.savefig('runs/idc_tsne_joint_proto.png', dpi=220, bbox_inches='tight', pad_inches=0)
    print('Saved runs/idc_tsne_joint_proto.png')

    # Interactive HTML
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=X2[m0,0], y=X2[m0,1], mode='markers', marker=dict(size=3, opacity=0.22), name='IDC-'))
    fig.add_trace(go.Scattergl(x=X2[m1,0], y=X2[m1,1], mode='markers', marker=dict(size=3, opacity=0.22), name='IDC+'))
    edge_x, edge_y = [], []
    for i,j in zip(rows, cols):
        if i<j:
            edge_x += [gm.P2[i,0], gm.P2[j,0], None]
            edge_y += [gm.P2[i,1], gm.P2[j,1], None]
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='rgba(50,50,50,0.3)'), name='Edges', hoverinfo='skip'))
    # Hover
    hover = []
    amb = gm.cross_class_ambiguity()
    for i in range(gm.P.shape[0]):
        hover.append("<br>".join([
            f"<b>Proto {i}</b>",
            f"dom class: {int(gm.dom_class[i])}",
            f"support: {int(gm.support[i])}",
            f"purity: {float(gm.purity[i]):.3f}",
            f"entropy: {float(gm.entropy[i]):.3f}",
            f"reliability: {float(gm.reliability[i]):.3f}",
            f"ambiguity(edge): {float(amb[i]):.3f}",
        ]))
    sizes = 16 + 24 * gm.reliability
    for c, symbol in [(0,'circle'),(1,'x')]:
        idx = np.where(gm.dom_class==c)[0]
        fig.add_trace(go.Scatter(x=gm.P2[idx,0], y=gm.P2[idx,1], mode='markers',
                                 marker=dict(size=sizes[idx], line=dict(width=1, color='black'), symbol=symbol),
                                 name=f'Proto dom {c}', text=[hover[i] for i in idx], hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(title='IDC — GraphMemory (interactive)', template='plotly_white', dragmode='pan')
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.write_html('runs/idc_graphmemory.html', include_plotlyjs='cdn')
    print('Saved runs/idc_graphmemory.html')

if __name__ == '__main__':
    main()