# Cross-Model Activation Oracles: Do Neural Networks Converge to Shared Representations?

## Abstract

The Platonic Representation Hypothesis posits that neural networks, regardless of architecture, converge toward shared statistical representations of reality that differ only by orthogonal transformation. If true, this would enable cross-model interpretability tools --- for instance, an activation oracle trained to read one model's internal representations could, via a learned mapping, read another model's representations. We test this hypothesis at the 1--3B parameter scale by measuring representational similarity (via Centered Kernel Alignment) and learning alignment mappings (Orthogonal Procrustes, ridge regression, LASSO, and low-rank factorization) between four distinct model pairs spanning three architecture families (Llama, Pythia, Gemma, Qwen). Across all experiments, CKA similarity is consistently weak (max 0.10--0.22), alignment quality is poor (test losses near 1.0, explained variance below 7%), and scaling from 1B to 3B parameters does not improve representational convergence. However, all CKA scores are statistically significant (p < 0.005, Cohen's d > 100), and low-rank alignment at rank 32 outperforms full-rank ridge regression, indicating that the cross-model signal, while weak, is real and confined to a low-dimensional subspace. These results suggest that the Platonic Representation Hypothesis does not hold at 1--3B scale across architecturally distinct model families.

## 1. Introduction

### 1.1 Motivation

Activation oracles (Karvonen et al., 2025) are LLMs trained to accept neural network activations as input and produce natural-language descriptions of what those activations encode. They represent a promising direction for mechanistic interpretability, but are currently architecture-specific: an oracle trained on Gemma activations cannot read Qwen activations.

The Platonic Representation Hypothesis (Huh et al., 2024) offers a potential solution. If sufficiently large neural networks converge toward the same representation of reality --- differing only by a rotation or linear transformation --- then a learned mapping between activation spaces could enable cross-model oracle transfer. One model's oracle could read another model's internal states, given the right alignment.

### 1.2 Research Questions

1. **Do architecturally distinct LLMs develop similar internal representations?** We measure representational similarity using Centered Kernel Alignment (CKA) across all layer pairs of each model pair.
2. **Can we learn high-quality alignment mappings between activation spaces?** We test orthogonal Procrustes, ridge regression, LASSO, and low-rank factorization methods.
3. **Does representational similarity increase with model scale?** We compare results at 1B and 3B parameters.
4. **What is the dimensionality of cross-model structure?** We use low-rank alignment sweeps to probe whether shared structure is confined to a low-dimensional subspace.

## 2. Methods

### 2.1 Models and Data

We conducted four experimental evaluations, each comparing a different model pair. Models were selected to span distinct architecture families, training corpora, and parameter scales.

| Eval | Model A | Model B | d_model A | d_model B | Layers A | Layers B | Dims Match |
|-------|---------|---------|-----------|-----------|----------|----------|------------|
| A | Llama-3.2-1B (Meta) | Pythia-1.4B (EleutherAI) | 2048 | 2048 | 16 | 24 | Yes |
| B | Gemma-2-2B (Google) | Qwen2.5-1.5B (Alibaba) | 2304 | 1536 | 26 | 28 | No |
| D | Llama-3.2-3B (Meta) | Pythia-2.8B (EleutherAI) | 3072 | 2560 | 28 | 32 | No |
| E | Llama-3.2-3B (Meta) | Gemma-2-2B (Google) | 3072 | 2304 | 28 | 26 | No |

**Dataset.** All experiments used the NeelNanda/pile-10k dataset (a 10,000-prompt subset of The Pile). We extracted residual stream activations at the last (non-padding) token position from each prompt, using a maximum sequence length of 128 tokens and batch size of 32 (16 for 3B-scale models). Activations were stored in float32.

**Layer sampling.** We extracted activations at 9 relative layer depths per model: fractions 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, and 1.0 of total depth, yielding 9 x 9 = 81 CKA comparisons per eval.

**Compute.** All experiments were run on a single NVIDIA A40 (48 GB) GPU via RunPod, with random seed 42.

### 2.2 CKA Similarity Analysis

**Centered Kernel Alignment (CKA)** (Kornblith et al., 2019) measures the similarity of two representation matrices, invariant to orthogonal transformations and isotropic scaling. Given activation matrices X (n x d_a) and Y (n x d_b), CKA is defined as:

    CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

where K = XX^T and L = YY^T are linear kernel matrices, and HSIC is the Hilbert-Schmidt Independence Criterion. CKA scores range from 0 (no similarity) to 1 (identical up to linear transformation).

We used the **debiased HSIC estimator** (Song et al., 2012), which provides more reliable estimates for finite samples by zeroing out kernel matrix diagonals and applying bias-correction terms. For each eval, we subsampled 5,000 of the 10,000 prompts for computational efficiency.

**Permutation tests.** To assess statistical significance, we ran 200 permutations of the sample indices for each of the top 2--3 layer pairs per eval, computing CKA on 1,000 samples per permutation. The p-value is the fraction of permuted CKA values exceeding the observed CKA. Effect size is reported as Cohen's d: (observed - null_mean) / null_std.

### 2.3 Alignment Methods

We tested four alignment approaches for learning a mapping W such that X @ W approximates Y, where X and Y are activation matrices from the source and target models respectively.

**Orthogonal Procrustes** (Eval A only). When d_model matches, we solve for the orthogonal matrix W = argmin ||XW - Y||_F subject to W^T W = I. The closed-form solution is W = UV^T where USV^T = SVD(X^T Y) (Schonemann, 1966). Data is centered before fitting. This method preserves geometric structure (distances and angles).

**Ridge regression (linear projection).** For any d_model pair, we solve W = argmin ||XW - Y||_F^2 + lambda ||W||_F^2, with closed-form solution W = (X^T X + lambda I)^{-1} X^T Y. Regularization lambda = 1e-4.

**LASSO (L1 sparse).** We solve W = argmin ||XW - Y||_F^2 + lambda ||W||_1 via iterative soft-thresholding (20 iterations), warm-started from a ridge solution. Regularization lambda = 1e-3. Sparsity in W reveals which source dimensions contribute most to the alignment.

**Low-rank factorization.** We decompose W = AB where A is (d_source x rank) and B is (rank x d_target), learned via alternating least squares (5 iterations) with regularization lambda = 1e-4. We tested ranks 32, 64, 128, and 256. This is analogous to a LoRA-style decomposition and tests whether the cross-model relationship is low-dimensional.

**Evaluation metrics.** All methods are evaluated on a held-out test set (80/20 train/test split, random seed 42):

- **Test loss** (normalized Frobenius residual): ||XW - Y||_F / ||Y - mean(Y)||_F. A score of 1.0 means the mapping is no better than predicting the mean; scores below 1.0 indicate learning.
- **Explained variance**: 1 - (residual_var / total_var), where variance is computed over the test set. Higher is better; 0 means no variance explained.

## 3. Results

### 3.1 CKA Similarity

CKA similarity was weak across all four model pairs. The following table summarizes the CKA statistics per eval.

| Eval | Model Pair | Max CKA | Best Layer Pair (A -> B) | Mean CKA | Min CKA |
|-------|-----------|---------|--------------------------|----------|---------|
| A | Llama-1B vs Pythia-1.4B | 0.208 | L13 -> L23 | 0.053 | 0.010 |
| B | Gemma-2B vs Qwen-1.5B | 0.222 | L18 -> L23 | 0.112 | 0.036 |
| D | Llama-3B vs Pythia-2.8B | 0.181 | L16 -> L31 | 0.052 | 0.006 |
| E | Llama-3B vs Gemma-2B | 0.184 | L16 -> L18 | 0.101 | 0.029 |

**CKA does not increase with scale.** Comparing Eval A (1B scale, max CKA = 0.208) to Eval D (3B scale, max CKA = 0.181), representational similarity is equivalent or slightly *lower* at 3B, directly contradicting the prediction of the Platonic Representation Hypothesis that similarity should grow with model capacity.

**Late layers match best.** Across all evals, the highest CKA scores involved late layers of both models. For example, in Eval A the best pair was Llama layer 13 (81% depth) paired with Pythia layer 23 (final layer). This is consistent with the expectation that later layers encode more abstract, task-relevant features.

**Top CKA layer pairs per eval (selected):**

| Eval | Layer A | Layer B | CKA |
|-------|---------|---------|-----|
| A | 13 | 23 | 0.208 |
| A | 11 | 23 | 0.198 |
| A | 9 | 23 | 0.195 |
| B | 18 | 23 | 0.222 |
| B | 21 | 23 | 0.208 |
| B | 15 | 23 | 0.196 |
| D | 16 | 31 | 0.181 |
| D | 23 | 31 | 0.177 |
| D | 20 | 31 | 0.176 |
| E | 16 | 18 | 0.184 |
| E | 23 | 18 | 0.183 |
| E | 20 | 18 | 0.176 |

### 3.2 Alignment Quality

Alignment quality was uniformly poor across all methods and phases. Test losses near 1.0 indicate that the learned mappings are barely better than predicting the target mean.

**Eval A: Orthogonal Procrustes (matched dims, d=2048)**

| Source Layer (Llama) | Target Layer (Pythia) | Train Loss | Test Loss | Explained Var |
|---------------------|-----------------------|------------|-----------|---------------|
| 15 | 23 | 0.931 | 0.965 | 0.069 |
| 13 | 23 | 0.958 | 0.980 | 0.039 |
| 11 | 23 | 0.972 | 0.986 | 0.028 |
| 9 | 23 | 0.980 | 0.990 | 0.020 |
| 7 | 23 | 0.986 | 0.994 | 0.012 |

The best result (Llama L15 -> Pythia L23) explains only 6.9% of target variance. Even with perfectly matched dimensions and an orthogonality-preserving mapping, the alignment captures very little of the target representation.

**Eval B: Method comparison (Gemma-2B -> Qwen-1.5B, best layer pair L18 -> L23)**

| Method | Rank | Train Loss | Test Loss | Explained Var |
|--------|------|------------|-----------|---------------|
| Low-rank | 32 | 0.902 | 0.965 | 0.069 |
| Low-rank | 64 | 0.879 | 0.970 | 0.060 |
| Low-rank | 128 | 0.852 | 0.982 | 0.036 |
| Low-rank | 256 | 0.819 | 1.004 | -0.008 |
| Ridge | full | 0.749 | 1.062 | -0.126 |
| LASSO | full | 0.749 | 1.062 | -0.126 |

**Key finding: Low-rank (rank=32) beats full ridge regression.** The full-rank methods (ridge and LASSO) show substantially lower train loss (0.749 vs 0.902) but much higher test loss (1.062 vs 0.965), indicating severe overfitting. Low-rank rank=32 achieves the best test loss across all methods. Increasing rank monotonically degrades test performance, with rank=256 overfitting to the point of negative explained variance on test data. This pattern was consistent across all tested layer pairs.

**Eval D: Llama-3B -> Pythia-2.8B (best layer pair L27 -> L31)**

| Method | Rank | Train Loss | Test Loss | Explained Var |
|--------|------|------------|-----------|---------------|
| Low-rank | 32 | 0.818 | 1.008 | -0.017 |
| Low-rank | 64 | 0.799 | 1.013 | -0.026 |
| Low-rank | 128 | 0.779 | 1.022 | -0.044 |
| Low-rank | 256 | 0.755 | 1.037 | -0.075 |
| Ridge | full | 0.674 | 1.111 | -0.234 |
| LASSO | full | 0.674 | 1.111 | -0.234 |

At 3B scale, even the best method (low-rank rank=32) fails to achieve positive explained variance on the test set. The alignment is even weaker than at 1B scale.

**Eval E: Llama-3B -> Gemma-2B (L16 -> L18)**

| Method | Rank | Train Loss | Test Loss | Explained Var |
|--------|------|------------|-----------|---------------|
| Low-rank | 32 | 0.894 | 1.006 | -0.010 |
| Ridge | full | 0.710 | 1.163 | -0.351 |

Cross-family alignment (Meta vs Google) shows similar patterns: massive overfitting with full-rank methods, and the best low-rank result still near chance level.

### 3.3 Permutation Tests

All tested layer pairs showed CKA scores far exceeding the null distribution, confirming that the weak representational similarity is statistically genuine.

| Eval | Layer Pair (A -> B) | Observed CKA | Null Mean | Null Std | p-value | Cohen's d | n_perm |
|-------|--------------------:|-------------:|----------:|---------:|--------:|----------:|-------:|
| A | L13 -> L23 | 0.190 | -0.00005 | 0.00141 | 0.000 | 134.8 | 200 |
| A | L11 -> L23 | 0.180 | -0.00003 | 0.00141 | 0.000 | 127.8 | 200 |
| A | L15 -> L23 | 0.153 | -0.00009 | 0.00134 | 0.000 | 114.2 | 200 |
| B | L18 -> L23 | 0.216 | -0.00008 | 0.00144 | 0.000 | 149.8 | 200 |
| B | L21 -> L23 | 0.196 | -0.00010 | 0.00144 | 0.000 | 136.4 | 200 |
| B | L15 -> L23 | 0.186 | -0.00008 | 0.00138 | 0.000 | 135.2 | 200 |
| D | L16 -> L31 | 0.159 | -0.00006 | 0.00148 | 0.000 | 107.4 | 200 |
| D | L23 -> L31 | 0.154 | 0.00008 | 0.00153 | 0.000 | 100.6 | 200 |
| E | L16 -> L18 | 0.173 | -0.00012 | 0.00151 | 0.000 | 114.6 | 200 |
| E | L23 -> L18 | 0.170 | -0.00018 | 0.00154 | 0.000 | 110.9 | 200 |

Across all 10 tested layer pairs from 4 evals, no single permutation out of 200 produced a CKA value exceeding the observed CKA (p = 0.000 in all cases). The null distribution means hover near zero (as expected for shuffled data), with standard deviations around 0.0014. Effect sizes (Cohen's d) range from 100.6 to 149.8, all classified as extremely large. The 95th percentile of the null distribution was approximately 0.002 in all tests, roughly two orders of magnitude below observed values.

**Interpretation.** The signal is real --- these models do share some representational structure that is not present in random permutations. However, statistical significance does not imply practical significance. CKA values of 0.15--0.22 indicate very weak similarity in absolute terms.

## 4. Discussion

### 4.1 The Platonic Representation Hypothesis Is Not Supported at 1--3B Scale

Our results provide evidence against the Platonic Representation Hypothesis at the 1--3B parameter scale. The key findings are:

1. **CKA similarity is consistently weak** (0.10--0.22 maximum) across all four model pairs, spanning three distinct architecture families trained on different data.
2. **Scaling does not help.** The 3B model pairs (Evals D and E) show equivalent or lower CKA than the 1B pairs (Eval A), contrary to the hypothesis that convergence should increase with scale.
3. **Alignment mappings capture very little variance.** The best alignment (Procrustes, Eval A) explains only 6.9% of target variance. At 3B scale, no method achieves positive explained variance on the test set.

### 4.2 Cross-Model Structure Is Real but Low-Dimensional

Despite the overall weakness, the cross-model signal is not noise. Two pieces of evidence support this:

1. **Permutation tests.** All observed CKA values are hundreds of standard deviations above the null distribution (Cohen's d > 100). The signal is extremely robust statistically.
2. **Low-rank outperforms full-rank.** In Eval B, low-rank alignment at rank 32 achieves test loss of 0.965, while full ridge regression achieves 1.062. This implies the cross-model relationship is confined to approximately 32 shared dimensions out of 1536--2304 total. Higher ranks introduce overfitting, not useful structure.

This suggests that while models do not converge to the same overall representation, they may share a small number of common features --- perhaps corresponding to basic linguistic structures (syntax, common entity types) that any language model must encode.

### 4.3 Implications for Cross-Model Interpretability

These results have practical implications for cross-model activation oracle transfer:

- **Direct transfer is currently infeasible.** With alignment quality near chance level, mapping one model's activations into another's space would produce essentially meaningless inputs to an oracle.
- **Low-dimensional bridging may be viable.** If the shared subspace (~32 dimensions) encodes interpretable features, one could project both models' activations into this shared subspace and train oracles there. This would sacrifice resolution for universality.
- **Larger scale may be necessary.** The Platonic Representation Hypothesis may require models significantly larger than 3B parameters, or models trained on similar data distributions, before convergence emerges.

### 4.4 Limitations

Several limitations constrain the interpretation of these results:

1. **Scale.** We tested models up to 3B parameters. The Platonic Representation Hypothesis may only manifest at 10B+ scale, where models have sufficient capacity to converge on similar representations.
2. **Training data.** Our model pairs were trained on different corpora (e.g., The Pile for Pythia vs proprietary data for Llama and Gemma). Training data differences may dominate architectural similarity.
3. **Activation extraction.** We extracted only last-token residual stream activations. Mean-pooled or attention-specific representations might reveal different similarity patterns.
4. **Linear methods.** All alignment methods are linear. Nonlinear mappings (e.g., neural stitching layers) might capture structure that linear methods miss.
5. **Layer sampling.** We sampled 9 layers per model at uniform relative depth. Finer-grained sampling might reveal narrow regions of higher similarity.
6. **Dataset.** We used a single dataset (pile-10k) for activation extraction. Domain-specific prompts (e.g., code, mathematics) might elicit more convergent representations in those domains.

## 5. Key Findings

1. CKA similarity between architecturally distinct LLMs at 1--3B scale is consistently weak, with maximum values of 0.10--0.22 across all four tested model pairs.

2. CKA does not increase with model scale: 3B model pairs (max CKA = 0.181) show similar or lower similarity than 1B pairs (max CKA = 0.208), contradicting the Platonic Representation Hypothesis at this scale.

3. All CKA scores are highly statistically significant (p = 0.000, Cohen's d = 100--150 across 10 tested layer pairs), confirming the cross-model signal is real despite being weak.

4. Low-rank alignment at rank 32 outperforms full-rank ridge regression on held-out data (test loss 0.965 vs 1.062 in Eval B), demonstrating that cross-model structure is confined to a low-dimensional subspace of approximately 32 dimensions.

5. Higher-rank alignment (128, 256) and full-rank methods (ridge, LASSO) consistently overfit: lower train loss but higher test loss than rank 32, confirming the low-dimensional nature of the shared structure.

6. The best alignment mapping (Orthogonal Procrustes, Eval A, Llama L15 -> Pythia L23) explains only 6.9% of target variance, far too low for practical cross-model oracle transfer.

7. Late layers consistently produce the highest CKA scores across all model pairs, consistent with deeper layers encoding more abstract, potentially universal features.

## References

- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *ICML 2024*. arXiv:2405.07987.
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*. arXiv:1905.00414.
- Karvonen, A., et al. (2025). Activation Oracles. arXiv:2512.15674.
- Park, K., Nanda, N., et al. (2023). The Linear Representation Hypothesis and the Geometry of Large Language Models. *ICLR 2025*. arXiv:2311.03658.
- Schonemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*, 31(1), 1--10.
- Song, L., Smola, A., Gretton, A., Bedo, J., & Borgwardt, K. (2012). Feature selection via dependence maximization. *JMLR*, 13, 1393--1434.
- Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. *NeurIPS 2017*. arXiv:1706.05806.
- Lample, G., Conneau, A., Denoyer, L., & Ranzato, M. (2018). Word Translation Without Parallel Data. *ICLR 2018*. arXiv:1710.04087.
- Bansal, Y., Nakkiran, P., & Barak, B. (2021). Revisiting Model Stitching to Compare Neural Representations. *NeurIPS 2021*. arXiv:2106.07682.
