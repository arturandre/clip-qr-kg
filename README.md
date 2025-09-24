## Prototype-Relation Graph Memory — Baselines & Scaffold

This repo provides **vanilla baselines** and a clean scaffold to evaluate a Prototype-Relation Graph Memory on top of frozen vision embeddings. It extracts & caches encoder features, runs **Linear Probe**, **kNN**, **Label Propagation**, reports **Accuracy / NLL / ECE**, optional **MSP OOD AUROC**, and includes a minimal **Graph Memory** stub + ablation hooks.

> ⚖️ Baseline policy: no “secret sauce.”
> kNN = uniform vote; Linear = multinomial LR with L2; LP = sklearn LabelPropagation (rbf/knn). OOD uses MSP from the **same fitted model**. DkNN is stubbed.

---

## What’s included

* Embedding extraction (torchvision: ViT-B/16, ViT-B/32, ResNet-50) and **on-disk caching**.
* Baselines: **Linear Probe**, **kNN**, **Label Propagation**.
* Metrics: **Top-1**, **NLL**, **ECE**; **MSP OOD AUROC** (IN vs OOD).
* **GraphMemoryClassifier** scaffold to plug in your method.
* Reproducible CSV outputs.

Supported datasets: `cifar10`, `cifar100`, `stl10` (extendable).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U torch torchvision torchaudio \
    scikit-learn numpy pandas tqdm scipy
```

---

## Quickstart

```bash
# Extract embeddings (cached) + run vanilla baselines
python main.py --dataset cifar10 --encoder vit_b_16 \
  --baselines linear,knn,lp \
  --batch-size 256 --cache-dir ./cache --out-dir ./runs

# With OOD (MSP AUROC): ID=CIFAR-10, OOD=STL-10
python main.py --dataset cifar10 --ood-dataset stl10 \
  --encoder vit_b_16 --baselines linear,knn,lp
```

Outputs a CSV like:

```
./runs/results__cifar10__vit_b_16.csv
acc, nll, ece, C, ood_auroc_msp, model, k, n_neighbors, gamma, kernel
```

---

## Baseline details (vanilla)

* **Linear Probe**: multinomial LogisticRegressionCV (`lbfgs`, L2), C chosen by **NLL**. Feature standardization is on (common practice).
* **kNN**: uniform voting, default `metric=cosine`.
* **Label Propagation (LP)**: sklearn `LabelPropagation` with `--lp-kernel {rbf,knn}` (default `rbf`). LP is **transductive**; we omit OOD AUROC unless refitted with OOD included.

> Note: DkNN requires multi-layer features; a stub is provided.

---

## OOD evaluation (MSP)

* IN/OOD probabilities must come from the **same trained estimator**.
* We compute AUROC using **Maximum Softmax Probability (MSP)** as the score.

---

## Reproducing the “go/no-go” baselines

```bash
# CIFAR-10 (ID) vs STL-10 (OOD)
python main.py --dataset cifar10 --ood-dataset stl10 \
  --encoder vit_b_16 --baselines linear,knn,lp \
  --batch-size 256 --num-workers 4
```

* Expect linear ≈ 94–96% acc on strong embeddings; kNN ≈ 92–95%; LP competitive when configured (rbf/knn).

---

## Using the Graph Memory scaffold

`GraphMemoryClassifier` is minimal and **not** part of vanilla baselines. Implement in-place:

* `fit(...)`: build prototypes, reliability, sparse inter-prototype edges.
* `predict_proba(...)`: attach probe → aggregate / propagate.
* Add ablations that **degenerate** to kNN / ProtNet / LP to show strict supersets.

---

## Project structure

```
main.py          # CLI runner, baselines, scaffold
cache/           # cached embeddings (*.npz)
runs/            # results CSVs
data/            # datasets (downloaded by torchvision)
```

---

## Tips & troubleshooting

* **Slow linear probe?** We use `lbfgs` + standardization + CV over C (fast for typical sizes).
* **LP accuracy too low?** Try `--lp-kernel knn --lp-k 10` (still vanilla).
* **Cached files:** remove `./cache/*.npz` to re-extract.

---

## License

MIT (add your preferred license).

## Citation

If you reference baselines or the scaffold, please cite appropriate standard works (e.g., logistic regression, kNN, Label Propagation, DkNN).
