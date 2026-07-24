# Plan: Align WSI with Genomics to Infer TNBC Receptor Status (ER−/PR−/HER2−)

## 0) Goal

* Learn **the same clinical facts** (ER, PR, HER2 status) from **both** modalities (WSI and genomics) and **prove** cross-modal agreement.
* Validate a **multimodal knowledge graph (KG)** where a “receptor_status” entity is supported by WSI and genomics evidence.

---

## 1) Data & Splits

* **Cohorts:** TCGA-BRCA (WSI + RNA-seq + mutations + CNV + clinical), TCIA WSIs when available, METABRIC for external validation (omics; WSI if accessible).
* **Splits:** patient-level stratified 70/15/15 (train/val/test). Hold out **entire sites** if possible for domain generalization.
* **Exclusions:** missing labels for ER/PR/HER2; WSIs with <50% tumor area.

---

## 2) Ground-Truth Labels (clinical “anchor”)

* From TCGA clinical pathology: **ER, PR, HER2 IHC/FISH** → map to {Positive/Negative}.
* Define **TNBC = ER− & PR− & HER2−** as the primary endpoint label.

---

## 3) Genomic Modality: Feature Engineering + Label Rules

* **RNA-seq:** logCPM + ComBat; genes: **ESR1 (ER), PGR (PR), ERBB2 (HER2)**. Add pathway scores (GSVA) for ER signaling, HER2/ERBB pathway.
* **CNV:** ERBB2 amplification status (GISTIC + thresholds).
* **Mutation:** drivers (TP53, PIK3CA) as covariates (not labels).
* **Genomic label proxy rules (when clinical missing/noisy):**

  * ER+: high ESR1 expression + ER pathway score high.
  * PR+: high PGR expression.
  * HER2+: **ERBB2 amp** or **ERBB2 high** RNA (and, if available, HER2 FISH).
  * **TNBC proxy:** ESR1 low ∧ PGR low ∧ ERBB2 not amplified and low RNA.
* Store **confidence** per genomic label (e.g., distance from thresholds).

---

## 4) WSI Modality: Preprocess + Features + Model

* **Preprocess:** tissue mask → tile @20× (256–512 px) → stain normalization (Macenko).
* **Self-supervised encoder:** DINOv2/RetCCL/CTransPath → tile embeddings.
* **MIL head (CLAM/ABMIL):** slide-level prediction for ER, PR, HER2 (multi-task).
* **WSI features (for KG evidence):**

  * TIL density/spatial stats.
  * Nuclear morphology, mitotic proxies, necrosis %, tumor–stroma interface.
  * Glandular/tubule patterns where present.
* **WSI label source:** primary = clinical anchor; optionally **knowledge distillation** from genomics proxy to handle sparse IHC.

---

## 5) Cross-Modal Alignment (Same Feature Across Modalities)

* **Target variables to align:** `ER_status`, `PR_status`, `HER2_status`, and `TNBC`.
* **Contrastive alignment:** train a projection for WSI and genomics such that matched pairs are close (CLIP-style), while keeping **supervised heads** for receptor labels.
* **Canonical correlation (CCA) / DCCA:** sanity check alignment strength.
* **Calibration:** temperature scaling per modality to make probabilities comparable.

---

## 6) Multimodal Fusion Models

* **Late fusion:** concat(W S I slide embedding, genomic vector) → MLP → multi-task heads: ER, PR, HER2, TNBC, survival (optional).
* **Missing-modality training:** random drop one modality during training; masking tokens; MIWAE/VAE imputation for genomics if needed.
* **Uncertainty:** MC-Dropout/Deep Ensembles; expose per-modality uncertainty.

---

## 7) Evaluation (Per Modality + Cross-Modal)

* **Per label:** AUROC, AUPRC, accuracy, F1 for ER/PR/HER2, and TNBC.
* **Cross-modal agreement:** Cohen’s κ, Matthews CC between WSI-only vs Genomics-only calls.
* **Against clinical anchor:** both modalities and fused model vs clinical labels.
* **Discordance audit:** confusion matrices; stratify by tumor purity, TILs, site, stage.
* **External validation:** METABRIC (omics) + any external WSI set; report drop.

---

## 8) KG Integration

* **Entities:** `Patient`, `Specimen`, `WSI`, `RNA`, `CNV`, `ReceptorStatus (ER/PR/HER2)`, `TNBC`.
* **Edges:**

  * `WSI --supports--> ReceptorStatus` with attributes: prob, tiles, attention heatmaps.
  * `RNA/CNV --supports--> ReceptorStatus` with attributes: ESR1/PGR/ERBB2 levels, CNV call, confidence.
  * `ReceptorStatus --defines--> TNBC`.
* **Provenance:** store model version, thresholds, calibration params, WSI tiles used.

---

## 9) Error Analysis & Robustness

* **Borderline cases:** CPS-like low-positive zones; inspect attention maps vs pathologist review.
* **Domain shift:** site-held-out, scanner type, staining variability.
* **Tumor purity control:** include purity as covariate; re-evaluate when purity <40%.
* **Sensitivity tests:** perturb stain, tile sampling, tile size.

---

## 10) Clinical Conclusions Parity Check

* **Goal:** show that **WSI-only** and **Genomics-only** lead to **same clinical call** (ER/PR/HER2/TNBC) with high κ (e.g., κ≥0.7).
* **When they disagree:** require fused model or human review; log to KG with conflict flag.
* **Report:** per-patient parity table: WSI call, Genomics call, Clinical anchor, fused call, uncertainties.

---

## 11) Survival/Outcome (Optional but useful)

* **Models:** DeepSurv/DeepHit using modality features; check calibration (IBS, GH-ECE).
* **Uplift/Causal:** simple treatment classes (chemo/IO/ADC proxies) to estimate differential benefit; use as **priors** for your virtual-trial simulator.

---

## 12) Deliverables

* **Repro repo:** loaders (WSI, RNA, CNV), training scripts (WSI-only, Genomics-only, Fusion), evaluation suite, visualization (WSI attention tiles vs ESR1/PGR/ERBB2 scores).
* **KG export:** JSON/Neo4j schema + ingest scripts.
* **Report:** parity metrics, model cards, failure modes, site-shift results.

---

## 13) Acceptance Criteria

* AUROC ≥ 0.85 for ER and HER2; ≥ 0.8 for PR (harder).
* κ (WSI vs Genomics) ≥ 0.7 for ER/HER2; ≥ 0.6 for PR.
* TNBC call accuracy ≥ 0.9 vs clinical anchor on held-out set.
* Calibrated probabilities (ECE ≤ 0.05) per modality.

---

## 14) Next Steps

* Add **self-training**: use high-confidence cross-modal agreements to pseudo-label low-resource sites.
* Add **radiomics** (if TCIA scans present) for tri-modal fusion.
* Deploy **uncertainty-gated** inference: route discordant/uncertain cases to expert review.
