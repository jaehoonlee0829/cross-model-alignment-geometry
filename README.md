# The Geometry of Cross-Model Alignment

## Motivation

Recent work on cross-model transfer --- steering vectors (2503.04429), model stitching (2506.06609), and universal SAEs (2502.03714) --- demonstrates that learned mappings between activation spaces can transfer interpretability tools across architectures. But these papers use full-rank mappings and never ask: **what is the minimum dimensionality of the shared signal?** They also skip measuring representational similarity before attempting transfer, making it hard to distinguish genuine structure from overfitting.

The **Platonic Representation Hypothesis** (Huh et al., 2024) predicts that sufficiently large neural networks converge to shared representations regardless of architecture. The **Aristotelian critique** (Chun et al., 2026) argues that standard CKA overstates this convergence due to dimensionality inflation, and advocates for permutation-calibrated measurements.

**Our contribution** is to characterize the *geometry* of cross-model alignment: we measure how similar representations are (debiased CKA with permutation calibration), learn alignment mappings at multiple ranks, and test whether the alignment carries functional task signal via probe transfer. We test both cross-architecture pairs (Gemma vs Qwen, Llama vs Pythia) and a within-family positive control (Llama-1B vs Llama-3B).

---

## Key Results

### 1. Cross-family CKA is weak, within-family is strong

![CKA Heatmaps](outputs/plots/all_cka_heatmaps.png)
*Debiased CKA heatmaps across four cross-family evaluations. All values remain below 0.22.*

We use **debiased CKA** --- not the standard (biased) CKA used in the original Platonic Representation Hypothesis paper. The debiased HSIC estimator (Song et al., 2012) zeros out kernel matrix diagonals and applies finite-sample corrections, avoiding inflated similarity scores in the high-dimensional regime (d = 1536--3072, n = 5000--10000).

| Eval | Model Pair | Type | Max CKA | Mean CKA |
|------|-----------|------|---------|----------|
| A | Llama-1B vs Pythia-1.4B | Cross-family | 0.208 | 0.053 |
| B | Gemma-2B vs Qwen-1.5B | Cross-family | 0.222 | 0.112 |
| C | Llama-1B vs Llama-3B | **Within-family** | **0.914** | **0.605** |
| D | Llama-3B vs Pythia-2.8B | Cross-family | 0.181 | 0.052 |
| E | Llama-3B vs Gemma-2B | Cross-family | 0.184 | 0.101 |

![CKA Scale Comparison](outputs/plots/cka_scale_comparison.png)
*CKA does not increase with model scale from 1B to 3B.*

### 2. Within-family positive control (Eval C)

![Eval C CKA Heatmap](outputs/eval_c/cka/cka_heatmap.png)
*Llama-1B vs Llama-3B: CKA values range from 0.18 to 0.91. Layers at matching relative depths align strongly, confirming our pipeline detects real similarity when it exists.*

Within-family CKA (max 0.91, mean 0.60) is **4--9x higher** than any cross-family pair. This validates our methodology: the weak cross-family scores (0.1--0.2) reflect genuine lack of similarity, not a measurement artifact.

### 3. Permutation calibration confirms statistical significance

To assess whether observed CKA scores reflect genuine structure (not dimensionality artifacts), we perform **permutation tests**: shuffle sample indices to break the correspondence between models, compute CKA on shuffled data (200--1000 permutations), and compare to the observed CKA.

We report **Cohen's d** (effect size) = (observed CKA - null mean) / null std. This measures how many standard deviations the observed CKA lies above the null distribution. Cohen's d > 0.8 is conventionally "large"; our values exceed 100.

![Permutation Tests](outputs/plots/permutation_tests.png)
*All observed CKA values (blue) are >100 standard deviations above the null distribution (red). p < 0.005 for all tested pairs (limited by 200 permutations).*

| Eval | Layer Pair | Observed CKA | Null Mean | Null Std | Cohen's d |
|------|-----------|-------------|-----------|----------|-----------|
| A | L13 -> L23 | 0.190 | -0.00005 | 0.00141 | 134.8 |
| B | L18 -> L23 | 0.216 | -0.00008 | 0.00144 | 149.8 |
| C | L0 -> L0 | 0.915 | -0.00017 | 0.00133 | **687.0** |
| D | L16 -> L31 | 0.159 | -0.00006 | 0.00148 | 107.4 |
| E | L16 -> L18 | 0.173 | -0.00012 | 0.00151 | 114.6 |

p-values are bounded by the number of permutations: with 200 permutations, the minimum reportable p-value is 1/200 = 0.005. No permuted CKA exceeded the observed value in any test, giving p < 0.005 for all pairs. The signal is statistically genuine --- but statistical significance does not imply practical significance.

### 4. Alignment: low-rank outperforms full-rank (bias-variance tradeoff)

![Alignment Comparison](outputs/plots/alignment_comparison_all.png)
*Alignment quality across all evals. Low-rank methods achieve lower test loss than full-rank ridge/LASSO due to reduced overfitting.*

| Method | Rank | Train Loss | Test Loss | Explained Var |
|--------|------|------------|-----------|---------------|
| Low-rank | 4 | 0.949 | 0.979 | 4.3% |
| Low-rank | 8 | 0.938 | **0.977** | 4.7% |
| Low-rank | 32 | 0.902 | 0.985 | 3.1% |
| Low-rank | 128 | 0.852 | 1.015 | -3.1% |
| Ridge | full | 0.749 | 1.129 | -26.6% |

Full-rank ridge achieves lower *train* loss (0.749) but dramatically worse *test* loss (1.129) due to overfitting. Low-rank methods generalize better because they have fewer parameters relative to the number of training samples. This is a standard bias-variance tradeoff: with 8000 training samples and d = 1536--3072, full-rank methods have millions of parameters and overfit severely.

![Rank Ablation](outputs/plots/rank_ablation_zoomed.png)
*Rank-vs-sample-size ablation (Eval B, L18->L23). At all sample sizes, lower ranks generalize better --- a bias-variance phenomenon, not evidence of intrinsic low-dimensional structure.*

### 5. Next-token prediction probe transfer

We train a logistic regression probe on Model A's activations to predict next tokens (top-500 most frequent, covering ~55% of data), then transfer it via alignment to Model B.

![Probe Transfer Comparison](outputs/plots/probe_transfer_comparison.png)
*Left: within-family transfer retains up to 93% of oracle accuracy. Right: cross-family transfer achieves ~0%.*

| | Cross-Family (Eval B) | Within-Family (Eval C) |
|---|---|---|
| Model A baseline | 72.5% | 63.9% |
| Rank 32 transfer | 0.1% | 55.9% |
| Ridge transfer | 0.2% | **92.9%** |
| Model B oracle | 77.8% | 63.4% |

Within-family ridge alignment retains **93%** of oracle accuracy for next-token prediction. Cross-family retains essentially **0%**. The 32k-class task is too demanding for the weak cross-model signal.

### 6. Binary probe transfer: cross-architecture DOES carry signal

Binary classification is a more sensitive test --- even a weak directional signal can push accuracy above the 50% chance baseline. We test three tasks: SST-2 sentiment, ToxiGen toxicity, AG News topic.

![Binary Probe Transfer](outputs/binary_probe_transfer/binary_probe_results.png)
*Cross-architecture transfer (left column) beats chance on ALL three tasks. Within-family (right column) approaches native accuracy.*

| Task | Cross-Arch (Gemma->Qwen) | 95% CI | Within-Family (Llama 1B->3B) | Chance |
|------|--------------------------|--------|------------------------------|--------|
| AG News (topic) | **81.4%** | [76.6%, 86.2%] | 93.7% | 51.3% |
| SST-2 (sentiment) | **63.2%** | [58.9%, 67.4%] | 78.6% | 53.7% |
| ToxiGen (toxicity) | **71.6%** | [66.2%, 77.1%] | 76.1% | 63.0% |

**Statistical significance (cross-architecture):** All three cross-arch results are significant despite using only 3 random seeds (conservative t-test with 2 degrees of freedom). AG News: t(2) = 26.7, p = 0.0007, Cohen's d = 18.9. SST-2: t(2) = 9.6, p = 0.005, Cohen's d = 6.8. ToxiGen: t(2) = 6.8, p = 0.011, Cohen's d = 4.8. Binomial tests on individual predictions give p < 1e-9 for all three tasks. The 95% confidence intervals do not overlap with the majority-class baseline for any task.

Cross-architecture alignment carries real task signal for binary classification. The next-token failure was about task granularity (32k classes scatter probability mass across thousands of wrong tokens), not absence of functional signal. Topic detection transfers best (81.4%), suggesting the shared structure encodes global document-level semantics more strongly than fine-grained features.

---

## Methods Summary

### Debiased CKA

Standard CKA (Kornblith et al., 2019) can overstate similarity in high-dimensional settings (Chun et al., 2026). We use the **debiased HSIC estimator** throughout, which zeros out kernel matrix diagonals and applies bias-correction terms, providing more conservative estimates.

### Permutation Calibration (Aristotelian-style)

Standard CKA scores are hard to interpret in isolation --- a CKA of 0.2 could be "high" or "low" depending on dimensionality and sample size. Following Chun et al. (2026), we calibrate CKA against a null distribution:

1. **Compute observed CKA** between activation matrices X (model A) and Y (model B) using the debiased HSIC estimator
2. **Generate null distribution:** For each of 200 permutations, randomly shuffle the sample indices of Y (breaking the one-to-one correspondence between models while preserving each model's marginal statistics), then compute CKA on the shuffled pair
3. **Compare:** If the observed CKA is merely an artifact of dimensionality or marginal statistics, it should fall within the null distribution. If it reflects genuine shared structure, it should be far above

We report two statistics:
- **Cohen's d** = (observed CKA - null_mean) / null_std --- the effect size, measuring how many standard deviations above the null the observed value falls. By convention, d > 0.8 is "large"; our values range from 107 to 687.
- **p-value** = fraction of permuted CKA values >= observed CKA. With 200 permutations, the minimum reportable p-value is 1/200 = 0.005. In all tests, zero permutations exceeded the observed CKA, giving p < 0.005.

**Crucially:** Both observed and null CKA use the same debiased HSIC estimator, ensuring an apples-to-apples comparison. This controls for the dimensionality inflation confound: if CKA were inflated by high dimensionality alone, the null distribution would show similarly inflated values.

### Alignment Methods

| Method | Description |
|--------|-------------|
| Orthogonal Procrustes | SVD-based rotation, preserves geometry (matched dims only) |
| Ridge Regression | Full-rank linear projection with L2 regularization |
| LASSO | Sparse alignment with L1 regularization |
| Low-Rank | W = AB factorization at ranks {4, 8, 16, 32, 64, 128, 256} |

**Evaluation:** Normalized Frobenius residual = `||XW - Y||_F / ||Y - mean(Y)||_F`. A score of 1.0 means no better than predicting the mean. This is a dimensionless ratio, comparable across different d_model values.

### Binary Probe Transfer

1. Train logistic regression on source model activations (sentiment/topic/toxicity)
2. Learn alignment mapping: target -> source activation space
3. Apply source probe to aligned target activations
4. Compare transfer accuracy vs 50% chance baseline

---

## Relation to Prior Work

| | Prior Work #1 (2503.04429) | Prior Work #2 (2506.06609) | **Ours** |
|---|---|---|---|
| **Alignment** | Affine / autoencoder (full-rank) | Affine (full-rank) | Rank sweep [4--256] + Procrustes + ridge + LASSO |
| **Transfers** | Steering vectors | SAE weights + probes | Next-token + binary probes (3 tasks) |
| **Models** | Cross-architecture | Within-family only | Both (cross + within-family control) |
| **CKA baseline** | None | None | Debiased CKA + permutation calibration |
| **Key question** | Does transfer work? | Does stitching save FLOPs? | What geometry does the shared signal have? |

---

## Conclusions

1. **Cross-family CKA is weak** (0.10--0.22) but statistically significant (Cohen's d > 100). Within-family CKA is 4--9x higher (0.91).
2. **CKA does not increase with scale** from 1B to 3B parameters.
3. **Low-rank alignment outperforms full-rank** due to the bias-variance tradeoff --- full-rank methods have too many parameters and overfit.
4. **Cross-architecture alignment carries coarse semantic signal** (sentiment 63%, topic 81%, toxicity 72% on binary tasks) but fails on fine-grained prediction (0% on next-token).
5. **Within-family alignment retains 93%** of native probe accuracy for next-token prediction.
6. **The Platonic Representation Hypothesis is not supported** at 1--3B scale for cross-family pairs, but a weaker form holds: models share coarse semantic features regardless of architecture.

---

## References

- Huh et al. (2024). [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987). ICML 2024.
- Chun et al. (2026). [Revisiting the Platonic Representation Hypothesis: An Aristotelian View](https://arxiv.org/abs/2602.14486).
- Kornblith et al. (2019). [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414). ICML 2019.
- Song et al. (2012). Feature selection via dependence maximization. JMLR.
- Karvonen et al. (2025). [Activation Oracles](https://arxiv.org/abs/2512.15674).
- [Activation Space Interventions Transfer](https://arxiv.org/abs/2503.04429) (2025).
- [Model Stitching for Linear Features](https://arxiv.org/abs/2506.06609) (2025).

## Setup & Usage

```bash
git clone https://github.com/jaehoonlee0829/cross-model-activation-oracles.git
cd cross-model-activation-oracles
pip install -r requirements.txt

# CKA + permutation tests
python scripts/run_cka.py --config configs/eval_c.yaml

# Alignment
python scripts/run_alignment.py --config configs/eval_c.yaml --alignment-only

# Rank ablation
python scripts/run_rank_ablation.py

# Binary probe transfer
python scripts/run_binary_probe_extraction.py
python scripts/run_binary_probe_transfer.py

# Next-token probe transfer
python scripts/run_probing.py --config configs/eval_c.yaml
```

## Project Structure

```
cross-model-activation-oracles/
├── configs/                            # Eval configs (A/B/C/D/E)
├── scripts/
│   ├── run_extraction.py               # Activation extraction (GPU)
│   ├── run_cka.py                      # CKA + permutation tests
│   ├── run_alignment.py                # Alignment mapping
│   ├── run_rank_ablation.py            # Rank-vs-sample-size ablation
│   ├── run_probing.py                  # Next-token probe transfer
│   ├── run_binary_probe_extraction.py  # Binary probe activation extraction
│   └── run_binary_probe_transfer.py    # Binary probe transfer experiments
├── src/
│   ├── cka_analysis.py                 # Debiased CKA + permutation_test_cka()
│   ├── procrustes_alignment.py         # Procrustes, ridge, LASSO, low-rank (+GPU)
│   ├── linear_probing.py              # Linear probe train/transfer
│   ├── activation_extraction.py        # Residual stream extraction
│   └── config.py                       # Configuration
├── outputs/                            # All results, plots, CSVs
├── RESEARCH_REPORT.md                  # Full academic-style report
└── generate_plots.py                   # Reproducible figure generation
```

## License

MIT
