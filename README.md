# The Geometry of Cross-Model Alignment

**Do independently trained language models converge to shared representations? If so, how much signal do they share, and in how many dimensions?**

## Key Results

### Cross-family CKA is weak (0.10--0.22), within-family is strong (0.91)

![CKA Heatmaps](outputs/plots/all_cka_heatmaps.png)

Five evaluations across four architecture families (Llama, Pythia, Gemma, Qwen) at 1--3B scale. Cross-family similarity maxes at CKA = 0.22; within-family (Llama-1B vs Llama-3B) reaches 0.91.

### The shared subspace is ~4--8 dimensional

![Rank Ablation](outputs/plots/rank_ablation_zoomed.png)

Low-rank alignment at rank 4--8 outperforms all higher ranks and full-rank ridge. The optimal rank does NOT increase with sample size, ruling out regularization artifacts. The cross-model signal is genuinely confined to approximately 4--8 dimensions out of 1536--3072 total.

### Binary probe transfer works cross-architecture

![Binary Probe Transfer](outputs/binary_probe_transfer/binary_probe_results.png)

Cross-architecture alignment preserves coarse semantic signal for binary classification:

| Task | Cross-Arch (Gemma->Qwen) | Within-Family (Llama 1B->3B) | Chance |
|------|--------------------------|------------------------------|--------|
| AG News (topic) | **81.4%** | 93.7% | 51.3% |
| SST-2 (sentiment) | **63.2%** | 78.6% | 53.7% |
| ToxiGen (toxicity) | **71.6%** | 76.1% | 63.0% |

### Within-family vs cross-family probe transfer

![Probe Transfer Comparison](outputs/plots/probe_transfer_comparison.png)

Next-token prediction probe transfer: within-family retains 93% of oracle accuracy via ridge alignment. Cross-family retains ~0% for fine-grained prediction, but binary classification shows real signal.

## Research Questions

1. **Do architecturally distinct LLMs develop similar internal representations?** Measured via debiased CKA with Aristotelian-style permutation calibration.
2. **What is the dimensionality of cross-model structure?** Systematic rank sweeps [4--256] + rank-vs-sample-size ablation.
3. **Does the shared subspace carry task-relevant signal?** Binary probe transfer across 3 tasks.
4. **Does representational similarity increase with model scale?** Compared at 1B and 3B.
5. **Can our pipeline detect strong alignment when it should exist?** Within-family positive control (Eval C).

## Evaluations

| Eval | Model A | Model B | d_model | Type | Max CKA |
|------|---------|---------|---------|------|---------|
| A | Llama-3.2-1B | Pythia-1.4B | 2048 / 2048 | Cross-family | 0.208 |
| B | Gemma-2-2B | Qwen2.5-1.5B | 2304 / 1536 | Cross-family | 0.222 |
| C | Llama-3.2-1B | Llama-3.2-3B | 2048 / 3072 | Within-family | **0.914** |
| D | Llama-3.2-3B | Pythia-2.8B | 3072 / 2560 | Cross-family | 0.181 |
| E | Llama-3.2-3B | Gemma-2-2B | 3072 / 2304 | Cross-family | 0.184 |

## Methods

### CKA Similarity (Debiased + Permutation-Calibrated)

We use **debiased CKA** (Kornblith et al., 2019) with **Aristotelian-style permutation calibration** (Chun et al., 2026). Standard CKA can overstate similarity in high-dimensional settings. Our approach computes calibrated CKA (Cohen's d effect size) by comparing observed CKA against a null distribution from shuffled data.

### Alignment Methods

| Method | Description | When Used |
|--------|-------------|-----------|
| Orthogonal Procrustes | SVD-based rotation, preserves geometry | Matched dims only (Eval A) |
| Ridge Regression | Full-rank linear projection with L2 | All evals |
| LASSO | Sparse alignment with L1 | All evals |
| Low-Rank (LoRA-style) | W = AB factorization at rank k | Ranks {4, 8, 16, 32, 64, 128, 256} |

**Evaluation metric:** Normalized Frobenius residual = `||XW - Y||_F / ||Y - mean(Y)||_F`. Score of 1.0 = no better than mean prediction.

### Binary Probe Transfer

1. Train logistic regression probe on source model activations (sentiment/topic/toxicity)
2. Learn alignment: target -> source activation space at given rank
3. Apply source probe to aligned target activations
4. Compare transfer accuracy vs chance baseline

## Relation to Prior Work

| | Prior Work #1 (2503.04429) | Prior Work #2 (2506.06609) | **Ours** |
|---|---|---|---|
| **Alignment** | Affine / autoencoder (full-rank) | Affine (full-rank) | **Rank sweep [4--256]** + Procrustes + ridge + LASSO |
| **Transfers** | Steering vectors | SAE weights + probes | **Binary probes (3 tasks)** |
| **Models** | Cross-architecture | Within-family only | **Both** (cross + within-family control) |
| **CKA baseline** | None | None | **Debiased CKA + permutation calibration** |
| **Rank analysis** | None | None | **Systematic sweep + sample-size ablation** |
| **Key question** | Does transfer work? | Does stitching save FLOPs? | **What dimensionality IS the shared signal?** |

## Key Findings

1. **Cross-family CKA is weak** (0.10--0.22) but statistically significant (Cohen's d > 100, p = 0.000).
2. **Within-family CKA is strong** (0.91), validating our pipeline.
3. **The shared subspace has ~4--8 intrinsic dimensions.** Optimal rank stays low regardless of sample size = genuine structure, not regularization.
4. **CKA does not increase with scale** from 1B to 3B parameters.
5. **Cross-architecture alignment carries coarse semantic signal** (sentiment 63%, topic 81%, toxicity 72% on binary tasks).
6. **Cross-architecture fails on fine-grained tasks** (0% on 32k-class next-token prediction).
7. **Within-family alignment retains 93%** of native probe accuracy for next-token prediction.
8. **The Platonic Representation Hypothesis is not supported** at 1--3B scale for cross-family pairs, but a weaker form holds within architecture families.

## References

- Huh et al. (2024). [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987). ICML 2024.
- Chun et al. (2026). [Revisiting the Platonic Representation Hypothesis: An Aristotelian View](https://arxiv.org/abs/2602.14486).
- Kornblith et al. (2019). [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414). ICML 2019.
- Karvonen et al. (2025). [Activation Oracles](https://arxiv.org/abs/2512.15674).
- [Activation Space Interventions Transfer](https://arxiv.org/abs/2503.04429) (2025).
- [Model Stitching for Linear Features](https://arxiv.org/abs/2506.06609) (2025).

## Setup

```bash
git clone https://github.com/jaehoonlee0829/cross-model-activation-oracles.git
cd cross-model-activation-oracles
pip install -r requirements.txt
```

## Usage

```bash
# Extract activations
python scripts/run_extraction.py --config configs/eval_c.yaml

# Compute CKA + permutation tests
python scripts/run_cka.py --config configs/eval_c.yaml

# Learn alignment mappings
python scripts/run_alignment.py --config configs/eval_c.yaml --alignment-only

# Rank-vs-sample-size ablation
python scripts/run_rank_ablation.py

# Binary probe transfer (3 tasks x 2 pairs x 10 ranks)
python scripts/run_binary_probe_extraction.py
python scripts/run_binary_probe_transfer.py

# Next-token probe transfer
python scripts/run_probing.py --config configs/eval_c.yaml
```

## Project Structure

```
cross-model-activation-oracles/
├── configs/
│   ├── eval_c.yaml                     # Within-family (Llama-1B vs 3B)
│   ├── phase_a.yaml                    # Cross-family evals A/B/D/E
│   └── ...
├── scripts/
│   ├── run_extraction.py               # Activation extraction
│   ├── run_cka.py                      # CKA + permutation tests
│   ├── run_alignment.py                # Alignment mapping
│   ├── run_rank_ablation.py            # Rank-vs-sample-size ablation
│   ├── run_probing.py                  # Next-token probe transfer
│   ├── run_binary_probe_extraction.py  # Binary probe activation extraction
│   └── run_binary_probe_transfer.py    # Binary probe transfer experiments
├── src/
│   ├── cka_analysis.py                 # CKA + permutation_test_cka()
│   ├── procrustes_alignment.py         # Procrustes, ridge, LASSO, low-rank (+GPU)
│   ├── linear_probing.py              # Linear probe train/transfer
│   ├── activation_extraction.py        # Residual stream extraction
│   └── config.py                       # Configuration
├── outputs/
│   ├── plots/                          # All generated figures
│   ├── binary_probe_transfer/          # Binary probe results
│   └── eval_c/, phase_a/, ...          # Per-eval results
├── RESEARCH_REPORT.md                  # Full academic-style report
├── generate_plots.py                   # Reproducible figure generation
└── requirements.txt
```

## License

MIT
