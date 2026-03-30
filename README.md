# The Geometry of Cross-Model Representation Bridging

> **A note on terminology:** In this project we use **"bridge"** (or "mapping") to refer to learned linear transformations between two models' activation spaces. This is unrelated to "AI alignment" in the safety sense. The codebase internally uses `alignment` in variable/function names following the convention in the representation learning literature (Procrustes alignment, CKA alignment, etc.), but all user-facing text uses "bridge" to avoid confusion.

## Motivation

Recent work on cross-model transfer — steering vectors (2503.04429), model stitching (2506.06609), universal SAEs (2502.03714) — demonstrates that learned mappings can transfer interpretability tools across architectures. But these papers skip measuring representational similarity before attempting transfer, making it hard to distinguish genuine shared structure from overfitting.

The **Platonic Representation Hypothesis** (Huh et al., 2024) predicts convergence to shared representations regardless of architecture. The **Aristotelian critique** (Chun et al., 2026) argues that standard CKA overstates this convergence due to dimensionality inflation, and advocates for permutation-calibrated measurements.

**Our contribution** is to characterize the *geometry* of cross-model representation bridging: we measure how similar representations are (debiased CKA with permutation calibration), learn bridge mappings at multiple ranks, and test whether the bridge carries functional task signal via probe transfer. We test both cross-architecture pairs (Gemma vs Qwen, Llama vs Pythia) and a within-family positive control (Llama-1B vs Llama-3B).

---

## Key Results

### 1. Cross-family CKA is weak, within-family is strong

![CKA Heatmaps](outputs/plots/all_cka_heatmaps.png)
*Debiased CKA heatmaps across four cross-family evaluations. All values remain below 0.22.*

We use **debiased CKA** — not the standard (biased) CKA used in the original Platonic Representation Hypothesis paper. The debiased HSIC estimator (Song et al., 2012) avoids inflated similarity scores in the high-dimensional regime (d = 1536~3072, n = 5000~10000).

| Eval | Model Pair | Type | Max CKA | Mean CKA |
|------|-----------|------|---------|----------|
| A | Llama-1B vs Pythia-1.4B | Cross-family | 0.208 | 0.053 |
| B | Gemma-2B vs Qwen-1.5B | Cross-family | 0.222 | 0.112 |
| C | Llama-1B vs Llama-3B | **Within-family** | **0.914** | **0.605** |
| D | Llama-3B vs Pythia-2.8B | Cross-family | 0.181 | 0.052 |
| E | Llama-3B vs Gemma-2B | Cross-family | 0.184 | 0.101 |

### 2. Within-family positive control (Eval C)

![Eval C CKA Heatmap](outputs/eval_c/cka/cka_heatmap.png)
*Llama-1B vs Llama-3B: CKA ranges from 0.18 to 0.91. Layers at matching relative depths align strongly.*

Within-family CKA (max 0.91, mean 0.60) is **4~9x higher** than any cross-family pair, validating that our pipeline detects real similarity when it exists.

### 3. Permutation tests confirm statistical significance

![Corrected Permutation Tests](outputs/plots/corrected_permutation_tests.png)
*Both max and mean CKA significantly exceed the null for all 5 evals (p < 0.002, 500 permutations).*

We test **both** the max-CKA layer pair and the mean CKA across **all 81 layer pairs** to rule out cherry-picking. For each of 500 permutations, we shuffle sample indices (breaking input correspondence between models), compute the full 9x9 CKA matrix, and record the max and mean.

| Eval | Type | Observed CKA | Null 95th | Obs/Null ratio | p |
|------|------|-------------|-----------|----------------|---|
| A | Max | 0.208 | 0.001 | 202x | < 0.002 |
| A | Mean | 0.053 | 0.0003 | 168x | < 0.002 |
| B | Max | 0.222 | 0.001 | 210x | < 0.002 |
| B | Mean | 0.112 | 0.0004 | 284x | < 0.002 |
| C | Max | **0.914** | 0.001 | **865x** | < 0.002 |
| C | Mean | **0.605** | 0.0003 | **1835x** | < 0.002 |
| D | Max | 0.181 | 0.001 | 172x | < 0.002 |
| D | Mean | 0.052 | 0.0003 | 192x | < 0.002 |
| E | Max | 0.184 | 0.001 | 156x | < 0.002 |
| E | Mean | 0.101 | 0.0004 | 289x | < 0.002 |

The observed CKA values exceed the null 95th percentile by **156~1835x**, confirming they reflect genuine representational similarity rather than finite-sample or dimensionality artifacts. All p-values are < 0.002 (0 of 500 null permutations exceeded the observed value in any test).

**The mean CKA test is critical:** the *average* CKA across all 81 layer pairs is orders of magnitude above the null, confirming the overall similarity structure is real — not an artifact of cherry-picking the best layer pair.

### 4. Binary probe transfer: frozen general bridge vs task-specific

An initial experiment using task-specific alignment showed cross-arch transfer beating chance (81% on AG News). However, critic review identified a flaw: the alignment was trained on the same task data as the probe, making it a weaker claim. We corrected this with a **dual-approach design**:

- **Frozen (general):** Load alignment learned on pile-10k (general text), freeze it, apply to task activations. Tests: "does the general cross-model structure carry task signal?"
- **Task-specific:** Learn alignment on task data (same as v1). Tests: "can you build task-specific bridges?"

![Dual Probe Transfer](outputs/dual_probe_transfer/dual_probe_comparison.png)
*Red = frozen pile-10k alignment (stronger claim). Blue = task-specific alignment. Within-family (right) shows clear signal; cross-arch frozen (left) shows signal for AG News.*

**Corrected results (frozen general alignment):**

| Task | Cross-Arch Frozen | Within-Family Frozen (ridge) | Chance |
|------|-------------------|------------------------------|--------|
| AG News (topic) | **71.3%** (p ≈ 0.002) | **97.7%** | 51.3% |
| ToxiGen (toxicity) | **67.0%** (p ≈ 0.004) | **73.9%** | 63.0% |
| SST-2 (sentiment) | 55.0% (not significant) | **73.7%** | 53.7% |

**Key findings:**
- **General (frozen) bridge carries cross-arch signal for topic classification** — AG News at 71.3% is +20pp above chance (p ≈ 0.002, 3 seeds).
- **Sentiment does not reliably transfer cross-architecture** — 55.0% vs 53.7% chance is not statistically significant.
- **Within-family frozen bridge approaches native accuracy** at high rank (97.7% AG News at ridge).

*Note: Task-specific cross-arch bridge performed at chance level. The mapping quality is extremely poor (test loss ~0.96--1.0, explaining <7% of target variance), so mapped activations cluster near the target mean, yielding chance-level probe accuracy. The frozen bridge captures marginally more shared structure due to training on 10k diverse samples (vs 4k task-specific samples), which is enough to preserve coarse topic signal.*

### 5. Next-token prediction probe transfer

We train a logistic regression probe on Model A's activations to predict next tokens (top-500 most frequent, covering ~55% of data), then transfer via bridge to Model B.

![Probe Transfer Comparison](outputs/plots/probe_transfer_comparison.png)
*Left: within-family transfer retains up to 93% of oracle accuracy. Right: cross-family transfer achieves ~0%.*

| | Cross-Family (Eval B) | Within-Family (Eval C) |
|---|---|---|
| Model A baseline | 72.5% | 63.9% |
| Ridge transfer | 0.2% | **92.9%** |
| Model B oracle | 77.8% | 63.4% |

Within-family ridge bridge retains **93%** of oracle accuracy. Cross-family retains ~0%. The 32k-class task is too demanding for the weak cross-model signal.

---

## Methods

### Debiased CKA + Permutation Calibration

Standard CKA can overstate similarity in high-dimensional settings (Chun et al., 2026). We use the **debiased HSIC estimator** throughout (Song et al., 2012). To calibrate, we generate a null distribution by permuting sample indices (500 permutations), breaking input correspondence while preserving marginal statistics. Both observed and null CKA use the same debiased estimator. We test both max and mean CKA across all 81 layer pairs.

### Dual-Approach Probe Transfer

Our initial binary probe experiment (v1) used a bridge learned on task data, which allowed the bridge to overfit to task-relevant features. Evidence: transfer accuracy exceeded source probe accuracy on SST-2 (63% vs 58%), which is logically impossible for a faithful mapping. The corrected design (v2) uses a **frozen pile-10k bridge** applied to task activations, properly testing whether the general cross-model structure carries task signal.

### Bridge Methods

| Method | Description |
|--------|-------------|
| Orthogonal Procrustes | SVD-based rotation (matched dims only) |
| Ridge Regression | Full-rank linear projection with L2 |
| LASSO | Sparse bridge with L1 |
| Low-Rank | W = AB factorization at ranks {4, 8, 16, 32, 64, 128, 256} |

---

## Conclusions

1. **Cross-family CKA is weak (0.10~0.22) but statistically genuine** — both max and mean CKA across all 81 layer pairs significantly exceed the permutation null (p < 0.002), with observed/null ratios of 156~284x.
2. **Within-family CKA is 4~9x higher (0.91)**, validating our methodology and showing convergence occurs within architecture families.
3. **General cross-model bridge carries coarse semantic signal** — frozen pile-10k bridge achieves 71% on cross-arch topic classification (+20pp above chance). Sentiment does not reliably transfer.
4. **Task-specific cross-arch bridge produces chance-level results** because the mapping quality is very poor (~7% explained variance) and 4k task-specific samples provide less diverse training signal than 10k pile-10k samples.
5. **Fine-grained prediction (32k-class next-token) fails completely** cross-architecture but succeeds within-family (93% of native accuracy).
6. **The Platonic Representation Hypothesis is not supported at 1~3B scale** for cross-family pairs, but a weaker form holds: models share coarse document-level features (topic, toxicity) regardless of architecture.

---

## References

- Huh et al. (2024). [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987). ICML 2024.
- Chun et al. (2026). [Revisiting the Platonic Representation Hypothesis: An Aristotelian View](https://arxiv.org/abs/2602.14486).
- Kornblith et al. (2019). [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414). ICML 2019.
- Song et al. (2012). Feature selection via dependence maximization. JMLR.
- [Activation Space Interventions Transfer](https://arxiv.org/abs/2503.04429) (2025).
- [Model Stitching for Linear Features](https://arxiv.org/abs/2506.06609) (2025).

## Setup & Usage

```bash
git clone https://github.com/jaehoonlee0829/cross-model-alignment-geometry.git
cd cross-model-alignment-geometry
pip install -r requirements.txt

# Corrected permutation tests (GPU-accelerated)
python scripts/run_corrected_permutation_tests.py --config configs/phase_b.yaml

# Dual-approach probe transfer (frozen vs task-specific)
python scripts/run_dual_probe_transfer.py

# Next-token probe transfer
python scripts/run_probing.py --config configs/eval_c.yaml
```

## License

MIT
