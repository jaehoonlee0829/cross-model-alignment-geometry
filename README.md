# Cross-Model Activation Oracles

**Can activation oracles generalize across different model architectures?**

## Research Question

Activation oracles ([Anthropic, Dec 2025](https://www.anthropic.com)) can read a model's internal representations and answer questions about them — but only within the same architecture. A Gemma oracle reads Gemma activations; it can't read Qwen activations.

But if the [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) holds — that neural networks converge to similar representations regardless of architecture — then there should exist a mapping between activation spaces that makes cross-model oracle transfer possible.

## Hypothesis

If two models develop similar internal representations (measurable via CKA), we can learn a rotation/projection mapping between their activation spaces, and use it to make one model's oracle read another model's activations.

## Method

### Phase 1: CKA Diagnostic
1. Pick two models (e.g., Gemma-2-2B and Qwen2.5-1.5B)
2. Run both on 10K shared prompts, extract residual stream activations at matching relative layer depths
3. Compute the full CKA (Centered Kernel Alignment) matrix across all layer pairs
4. Identify which layers correspond between architectures

### Phase 2: Alignment
5. For best-matching layer pairs: learn a mapping between activation spaces
   - **Orthogonal Procrustes** (SVD-based) — when d_model matches
   - **Learned linear projection** (ridge regression) — for any d_model pair
6. Evaluate mapping quality: explained variance, normalized residual

### Phase 3: Oracle Transfer
7. Apply the mapping to transform Model A's activations into Model B's space
8. Inject mapped activations into Model B and run Model B's oracle
9. Compare oracle outputs to baseline (oracle on native activations)

## Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/cross-model-activation-oracles.git
cd cross-model-activation-oracles

# GPU environment setup (run on your cloud GPU)
bash scripts/setup_gpu_env.sh

# Or manual install
pip install -r requirements.txt
```

## Usage

```bash
# Step 1: Extract activations from both models
python scripts/run_extraction.py --config configs/default.yaml

# Step 2: Compute CKA similarity matrix + heatmap
python scripts/run_cka.py --config configs/default.yaml

# Step 3: Learn alignment (without oracle test)
python scripts/run_alignment.py --config configs/default.yaml --alignment-only

# Step 4: Full pipeline with oracle transfer (requires oracle adapter)
python scripts/run_alignment.py --config configs/default.yaml
```

## Project Structure

```
cross-model-activation-oracles/
├── configs/
│   └── default.yaml              # Experiment configuration
├── scripts/
│   ├── setup_gpu_env.sh          # GPU cloud environment setup
│   ├── run_extraction.py         # Step 1: Extract activations
│   ├── run_cka.py                # Step 2: CKA analysis
│   └── run_alignment.py          # Step 3-4: Alignment + oracle transfer
├── src/
│   ├── config.py                 # Configuration dataclasses
│   ├── activation_extraction.py  # Residual stream extraction with hooks
│   ├── cka_analysis.py           # CKA computation + visualization
│   ├── procrustes_alignment.py   # Procrustes + linear projection alignment
│   └── oracle_transfer_test.py   # Oracle injection + transfer testing
├── notebooks/                    # Exploration notebooks
├── tests/                        # Unit tests
├── requirements.txt
└── README.md
```

## Key Design Decisions

**d_model mismatch handling:** Gemma-2-2B has d_model=2304 while Qwen2.5-1.5B has d_model=1536. Standard orthogonal Procrustes requires matching dimensions (the orthogonality constraint W^T W = I forces W to be square). We implement both:
- Orthogonal Procrustes for same-d_model pairs (preserves geometry)
- Ridge regression linear projection for mismatched pairs (more flexible)

**Last-token extraction:** We extract at the last (non-padding) token position because that's where next-token prediction happens and where the model's representation is richest.

**Debiased CKA:** We use the debiased HSIC estimator (Kornblith et al., 2019) which is more reliable for finite samples.

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required: `HF_TOKEN` (for gated models like Gemma). Optional: `WANDB_API_KEY` for experiment tracking.

## Related Work & Literature

### Foundation — Core Concepts

| Paper | Why It Matters |
|-------|---------------|
| [Activation Oracles](https://arxiv.org/abs/2512.15674) — Karvonen et al., Dec 2025 | THE paper. Trains LLMs as flexible activation explainers that accept neural activations as input. Shows generalization beyond training distribution. Our project asks: can this generalize across *architectures*? |
| [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) — Huh et al., 2024 (ICML) | Theoretical motivation. Different neural networks converge toward shared statistical representations of reality, differing "only by rotation." If true, cross-model oracles should work. |
| [The Linear Representation Hypothesis](https://arxiv.org/abs/2311.03658) — Park, Nanda et al., 2023 (ICLR 2025) | High-level concepts exist as linear directions in representation space. If this holds across models, linear mappings (Procrustes) should suffice for oracle transfer. |

### Methodology — Metrics & Alignment Techniques

| Paper | Why It Matters |
|-------|---------------|
| [Similarity of Neural Network Representations Revisited (CKA)](https://arxiv.org/abs/1905.00414) — Kornblith et al., 2019 (ICML) | Introduced Centered Kernel Alignment. Invariant to orthogonal transforms. Our primary metric for measuring cross-model representational similarity. |
| [SVCCA](https://arxiv.org/abs/1706.05806) — Raghu et al., 2017 (NeurIPS) | Earlier representation comparison via SVD + canonical correlation. CKA supersedes it, but SVCCA established the conceptual framework. |
| [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087) — Lample, Conneau et al., 2018 (ICLR) | Unsupervised alignment of embedding spaces using adversarial learning + Procrustes refinement. Directly applicable to aligning activation spaces between models. |
| [Representation Engineering](https://arxiv.org/abs/2310.01405) — Zou et al., 2023 | Top-down approach to reading/controlling model internals at the representation level. Complementary to oracle approach. |
| [How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255) — Heimersheim et al., 2024 | Best practices for activation replacement experiments. Critical for our injection-based oracle transfer test design. |

### Closest Prior Work — The Novelty Gap (READ THIS FIRST)

These papers do almost exactly what we're proposing, but for different tools. Our contribution is specifically about **oracle** transfer — none of these test LLM-based generative oracles.

| Paper | What They Transfer | Gap vs. Our Work |
|-------|-------------------|-----------------|
| [Activation Space Interventions Can Be Transferred Between LLMs](https://arxiv.org/abs/2503.04429) — ICML 2025 | **Steering vectors** across Llama/Qwen/Gemma via learned activation alignment | Same alignment machinery, but steering ≠ oracle. Steering is a causal intervention; oracles are generative readers. |
| [Transferring Linear Features Across Language Models](https://arxiv.org/abs/2506.06609) — NeurIPS 2025 | **SAEs, probes, steering vectors** across Pythia/GPT2/Gemma via affine stitching maps | Closest threat. Transfers probes (classifiers), but probes are simple linear classifiers — oracles are LLMs with richer generative capabilities. |
| [Theseus: Transporting Task Vectors across Architectures](https://arxiv.org/abs/2602.12952) — Feb 2026 | **Task vectors** (weight-space) via orthogonal Procrustes | Exact same Procrustes technique, but in weight space not activation space. Doesn't test activation-based tools. |
| [Universal Sparse Autoencoders](https://arxiv.org/abs/2502.03714) — ICML 2025 | **SAE dictionaries** that decode activations from multiple architectures to shared concept space | Cross-model activation interpretation, but via SAE dictionary learning, not oracle LLMs. |
| [Cross-model Transferability among LLMs](https://arxiv.org/abs/2501.02009) — ACL 2025 | **Concept steering** via linear transforms between LLM representation spaces | Linear alignment + cross-model, but steering only. No oracle readability test. |

### Related — Cross-Architecture Transfer

| Paper | Why It Matters |
|-------|---------------|
| [Relative Representations Enable Zero-Shot Latent Space Communication](https://arxiv.org/abs/2209.15430) — Moschella et al., 2023 (ICLR) | Shows angle-based relative representations are invariant across different latent spaces. Could provide an alternative alignment method to Procrustes. |
| [Revisiting Model Stitching](https://arxiv.org/abs/2106.07682) — Bansal et al., 2021 (NeurIPS) | Connects frozen layers from different networks with trainable mapping. Reveals similarities CKA misses. Practical stitching technique for cross-model transfer. |
| [GER-steer: Global Evolutionary Steering](https://arxiv.org/abs/2603.12298) — March 2026 | Cross-architecture steering evaluation on Qwen, Llama, Gemma. Latest evidence on cross-model representation alignment. |

### Supporting — Probes & Safety Applications

| Paper | Why It Matters |
|-------|---------------|
| [Simple Probes Catch Sleeper Agents](https://www.anthropic.com/research/probes-catch-sleeper-agents) — Anthropic, 2024 | Linear classifiers on residual stream activations achieve >99% AUROC detecting deceptive behavior. Rich information in activations is linearly accessible. |
| [Cost-Effective Constitutional Classifiers](https://alignment.anthropic.com/2025/cheap-monitors/) — Anthropic, 2025 | Probe classifiers reusing model activations for safety monitoring. Practical downstream application of activation reading. |

## License

MIT
