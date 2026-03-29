#!/usr/bin/env python3
"""GPU-accelerated corrected CKA permutation tests.

Uses torch on GPU for all kernel/HSIC operations. Tests both max and mean CKA.
Consistent n=5000 subsampling. Reports z-scores (not Cohen's d).
"""

import argparse, csv, time
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import Config
from src.activation_extraction import load_activations

SUBSAMPLE_N = 5000
N_PERMUTATIONS = 500
SUBSAMPLE_SEED = 42
PERM_SEED = 137
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def hsic_debiased_gpu(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Debiased HSIC on GPU using O(n^2) trace trick."""
    n = K.shape[0]
    K_t = K.clone()
    L_t = L.clone()
    K_t.fill_diagonal_(0)
    L_t.fill_diagonal_(0)

    t1 = (K_t * L_t).sum()
    t2 = (K_t.sum() * L_t.sum()) / ((n - 1) * (n - 2))
    t3 = 2 * (K_t.sum(dim=0) @ L_t.sum(dim=0)) / (n - 2)
    return (t1 + t2 - t3) / (n * (n - 3))


def cka_from_kernels_gpu(K: torch.Tensor, L: torch.Tensor,
                          hsic_kk: torch.Tensor, hsic_ll: torch.Tensor) -> float:
    """CKA from precomputed kernels with cached self-HSIC."""
    hsic_kl = hsic_debiased_gpu(K, L)
    denom = torch.sqrt(hsic_kk * hsic_ll)
    return (hsic_kl / denom).item() if denom > 1e-10 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    act_dir = config.output_dir / "activations"

    print(f"Loading activations from {act_dir}...")
    acts_a = load_activations(config.model_a.alias, act_dir)
    acts_b = load_activations(config.model_b.alias, act_dir)

    n_total = next(iter(acts_a.values())).shape[0]
    rng_sub = np.random.default_rng(SUBSAMPLE_SEED)
    indices = rng_sub.choice(n_total, size=min(SUBSAMPLE_N, n_total), replace=False)

    layers_a = sorted(acts_a.keys())
    layers_b = sorted(acts_b.keys())
    n_pairs = len(layers_a) * len(layers_b)
    n = len(indices)

    print(f"Device: {DEVICE}, n={n}, {len(layers_a)}x{len(layers_b)}={n_pairs} pairs")

    # Precompute kernels on GPU
    print("Precomputing kernels on GPU...")
    kernels_a, kernels_b = {}, {}
    for la in layers_a:
        X = acts_a[la][indices].to(dtype=torch.float64, device=DEVICE)
        kernels_a[la] = X @ X.T
    for lb in layers_b:
        Y = acts_b[lb][indices].to(dtype=torch.float64, device=DEVICE)
        kernels_b[lb] = Y @ Y.T

    # Precompute self-HSIC (permutation-invariant)
    self_a = {la: hsic_debiased_gpu(kernels_a[la], kernels_a[la]) for la in layers_a}
    self_b = {lb: hsic_debiased_gpu(kernels_b[lb], kernels_b[lb]) for lb in layers_b}

    # Observed CKA matrix
    print("Computing observed CKA matrix...")
    obs_matrix = np.zeros((len(layers_a), len(layers_b)))
    for i, la in enumerate(layers_a):
        for j, lb in enumerate(layers_b):
            obs_matrix[i, j] = cka_from_kernels_gpu(
                kernels_a[la], kernels_b[lb], self_a[la], self_b[lb])

    obs_max = float(obs_matrix.max())
    obs_mean = float(obs_matrix.mean())
    max_idx = np.unravel_index(obs_matrix.argmax(), obs_matrix.shape)
    best_la, best_lb = layers_a[max_idx[0]], layers_b[max_idx[1]]
    print(f"Observed: max={obs_max:.4f} (L{best_la}->L{best_lb}), mean={obs_mean:.4f}")

    # Null distributions
    rng = np.random.default_rng(PERM_SEED)
    null_maxes, null_means = [], []
    t0 = time.time()

    for p in range(args.n_perms):
        perm = rng.permutation(n)
        perm_t = torch.from_numpy(perm).to(DEVICE)

        null_matrix = np.zeros((len(layers_a), len(layers_b)))
        for i, la in enumerate(layers_a):
            K_perm = kernels_a[la][perm_t][:, perm_t]
            for j, lb in enumerate(layers_b):
                null_matrix[i, j] = cka_from_kernels_gpu(
                    K_perm, kernels_b[lb], self_a[la], self_b[lb])

        null_maxes.append(float(null_matrix.max()))
        null_means.append(float(null_matrix.mean()))

        if (p + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (p + 1) / elapsed
            eta = (args.n_perms - p - 1) / rate
            print(f"  Perm {p+1}/{args.n_perms} | {rate:.1f}/s | ETA: {eta:.0f}s")

    null_maxes = np.array(null_maxes)
    null_means = np.array(null_means)
    elapsed = time.time() - t0

    # Stats (corrected p-value formula)
    max_z = (obs_max - null_maxes.mean()) / null_maxes.std() if null_maxes.std() > 0 else float('inf')
    mean_z = (obs_mean - null_means.mean()) / null_means.std() if null_means.std() > 0 else float('inf')
    max_p = (1 + np.sum(null_maxes >= obs_max)) / (1 + args.n_perms)
    mean_p = (1 + np.sum(null_means >= obs_mean)) / (1 + args.n_perms)

    print(f"\n{'='*70}")
    print(f"CORRECTED PERMUTATION TESTS: {config.model_a.alias} vs {config.model_b.alias}")
    print(f"n={n}, {n_pairs} pairs, {args.n_perms} perms, {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"\nMAX CKA (L{best_la}->L{best_lb}):")
    print(f"  Observed: {obs_max:.4f} | Null 95th: {np.percentile(null_maxes, 95):.5f} | p={max_p:.4f}")
    print(f"  (z-score: {max_z:.1f} — inflated by tight null)")
    print(f"\nMEAN CKA (all {n_pairs} pairs):")
    print(f"  Observed: {obs_mean:.4f} | Null 95th: {np.percentile(null_means, 95):.6f} | p={mean_p:.4f}")
    print(f"  (z-score: {mean_z:.1f} — inflated by tight null)")
    print(f"{'='*70}")

    # Save
    out_dir = config.output_dir / "cka"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "corrected_permutation_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in [
            ("observed_max", obs_max), ("observed_mean", obs_mean),
            ("best_layer_a", best_la), ("best_layer_b", best_lb),
            ("null_max_mean", null_maxes.mean()), ("null_max_std", null_maxes.std()),
            ("null_mean_mean", null_means.mean()), ("null_mean_std", null_means.std()),
            ("max_z_score", max_z), ("mean_z_score", mean_z),
            ("max_p_value", max_p), ("mean_p_value", mean_p),
            ("null_max_95th", np.percentile(null_maxes, 95)),
            ("null_mean_95th", np.percentile(null_means, 95)),
            ("n_permutations", args.n_perms), ("n_samples", n),
            ("n_layer_pairs", n_pairs), ("elapsed_seconds", elapsed),
        ]:
            w.writerow([k, v])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
