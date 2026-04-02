# Experiment Log — Cross-Model Activation Oracles

Date: 2026-03-28 to 2026-03-29
GPU: NVIDIA A40 (48GB)
Pod: RunPod

## Research Question

Can activation oracles generalize across different model architectures? We test this by:
1. Extracting residual stream activations from model pairs
2. Measuring representational similarity via CKA
3. Learning alignment mappings (Procrustes, ridge, low-rank, LASSO)
4. Testing if cross-model signal is low-dimensional

## Results Summary

### Phase A: Llama-3.2-1B vs Pythia-1.4B (d=2048, matched dims)
- CKA: mean=0.053, max=0.208 (Llama L13 <-> Pythia L23)
- Best alignment: Procrustes L15->L23, R^2=0.069, test_loss=0.965
- Permutation tests: all p=0.000, effect sizes >100

### Phase B: Gemma-2-2B vs Qwen2.5-1.5B (d=2304 vs 1536, mismatched)
- CKA: mean=0.112, max=0.222 (Gemma L18 <-> Qwen L23)
- Best alignment: low_rank rank=32, R^2=0.069, test_loss=0.965
- KEY FINDING: Low-rank (r=32) BEATS full ridge and LASSO
- Higher rank (128, 256) causes OVERFITTING
- Permutation tests: all p=0.000, effect sizes >130

### Phase D: Llama-3.2-3B vs Pythia-2.8B (d=3072 vs 2560, bigger scale)
- CKA: mean=0.052, max=0.181 (Llama-3B L16 <-> Pythia-3B L31)
- CKA does NOT increase with model scale (0.181 vs 0.208 at 1B)
- Permutation tests: all p=0.000, effect sizes >100

### Phase E: Llama-3.2-3B vs Gemma-2-2B (d=3072 vs 2304, cross-family)
- CKA: mean=0.101, max=0.184 (Llama-3B L16 <-> Gemma-2B L18)
- Cross-family comparison shows similar weak CKA
- Permutation tests: all p=0.000, effect sizes >110

## Key Findings

1. **CKA is consistently WEAK** (0.10-0.22) across ALL model pairs and scales
2. **CKA does NOT increase with model size** — 3B models show similar or LOWER CKA than 1B models
3. **All CKA scores are statistically significant** (p=0.000, effect sizes >100) — the weak signal is real, not noise
4. **Low-rank alignment (rank 32) BEATS full ridge** — cross-model relationship is low-dimensional
5. **Higher rank causes overfitting** — more parameters != better alignment
6. **Platonic Representation Hypothesis NOT supported** at 1-3B scale with these architectures
7. **Late layers match best** — all models' best CKA pairs involve the final/penultimate layer

## Statistical Tests

- CKA permutation tests: 200 permutations, 1000 samples per test
- All 10 tested layer pairs across 4 phases: p=0.000 (none exceeded observed CKA)
- Effect sizes (Cohen's d): range 100-150, extremely large

## Files Generated

### Intermediary CSVs (outputs/intermediary/)
- activation_stats_*.csv — Per-layer activation statistics for all models
- cka_matrix_phase_{a,b,d,e}.csv — Full CKA similarity matrices
- alignment_results_phase_{a,b,d,e}.csv — Alignment method comparison
- permutation_test_phase_{a,b,d,e}.csv — Statistical significance tests

### Plots (outputs/plots/)
- all_cka_heatmaps.png — 2x2 grid of CKA heatmaps
- cka_scale_comparison.png — Max/mean CKA by model scale
- phase_b_lowrank_analysis.png — Low-rank dimension sweep
- alignment_comparison_all.png — Method comparison across phases
- permutation_tests.png — Statistical significance visualization

### Raw Data (not committed — too large)
- outputs/phase_{a,b,d,e}/activations/*.pt — Raw activation tensors
- outputs/phase_{a,b,d,e}/cka/*.npz — CKA matrices
- outputs/phase_{a,b,d,e}/alignment/*.npz — Learned alignment mappings

## Implications

The Platonic Representation Hypothesis predicts that neural network representations converge
"up to rotation" as models scale. Our results suggest this convergence is NOT yet occurring at
1-3B scale, at least between architecturally distinct models trained on different data.

Possible explanations:
- 1-3B may be too small for convergence (need 10B+)
- Architecture differences (Llama vs Pythia vs Gemma vs Qwen) may matter more than scale
- Training data differences may dominate architectural similarity
- The hypothesis may hold within model families but not across them

The finding that low-rank alignment beats full ridge suggests that whatever cross-model structure
exists is confined to a low-dimensional subspace (~32 dimensions out of 1536-3072).

---

## Cross-Tokenizer Fix Experiments (2026-04-02)

### Motivation

The original NTP probe transfer (Phase B) had a confound: raw token IDs were treated as shared
classes across Gemma and Qwen, but different tokenizers assign different IDs to different tokens.
Two experiments fix this.

### Experiment A: Matched-Token NTP

Built cross-tokenizer vocabulary mapping (83,499 shared tokens via exact string match after
normalization of SentencePiece ▁ and tiktoken Ġ prefixes). Relabeled to top-500 shared classes.

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| Source native (Gemma) | 66.8% | 82.1% |
| Target oracle (Qwen) | 75.3% | 86.2% |
| Cross-model oracle | 10.3% | 20.2% |
| Best low-rank (r128/r256) | 4.6% | 15.9% |
| Ridge (full) | 4.9% | 18.0% |

Key: Cross-model oracle only 10.3% — models fundamentally disagree on next-token predictions.
Bridge captures ~half of existing agreement. Tokenizer fix did NOT change the conclusion.

### Experiment B: POS Tag Prediction (Tokenizer-Independent)

Used spaCy Universal POS tags (17 classes) as tokenizer-independent labels.

**Cross-arch (Gemma L18 → Qwen L23):**
- Source: 40.4%, Oracle: 37.8%, Cross-model oracle: 21.2%
- Best transfer: r4 at 29.9% (79% of oracle)
- Ridge: 23.1% (61%)

**Within-family (Llama 1B L15 → 3B L27):**
- Source: 45.5%, Oracle: 48.5%
- Best transfer: r128 at 49.3% (101.6% — slight valid-sample mask artifact)
- Ridge: 47.1% (97.1%)

### Complexity Gradient Established

Binary (~70%) → POS 17-class (~79%) → NTP 500-class (~6%)
Coarse linguistic structure transfers cross-arch; fine-grained token identity does not.

### Critic Analysis (3 independent reviews)

**Statistical:** No error bars, single-seed estimates. POS differences between ranks within noise.
**Design:** Missing majority-class baseline (~20%), no random-bridge or shuffled-label controls.
POS labels confounded by tokenizer-dependent truncation positions.
**Interpretation:** Low-rank r4 > ridge is regularization, not low-dimensional structure.
Cross-model oracle conflates tokenizer boundary agreement with representational alignment.

### Files Generated

- outputs/vocab_mapping.json — Cross-tokenizer vocabulary mapping (83,499 shared tokens)
- outputs/phase_b/probing/labels_gemma_shared.npy — Gemma shared-class labels
- outputs/phase_b/probing/labels_qwen_shared.npy — Qwen shared-class labels
- outputs/phase_b/probing/top_shared_tokens.npy — Top-500 shared classes
- outputs/phase_b/probing/matched_token_probe_results.csv — Experiment A results
- outputs/pos_labels_gemma.npy — POS labels for Gemma
- outputs/pos_labels_qwen.npy — POS labels for Qwen
- outputs/pos_labels_llama-1b.npy — POS labels for Llama-1B
- outputs/pos_labels_llama-3b.npy — POS labels for Llama-3B
- outputs/pos_probe_results.csv — Experiment B results
