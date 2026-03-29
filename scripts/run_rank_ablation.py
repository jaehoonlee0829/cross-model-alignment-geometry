#!/usr/bin/env python3
"""Rank-vs-sample-size ablation: is rank 32 genuine structure or regularization artifact?

Uses Eval B (Gemma-2B vs Qwen-1.5B) best layer pair (L18 -> L23).
Tests all combinations of sample sizes and ranks with multiple seeds.

Usage:
    python scripts/run_rank_ablation.py
    python scripts/run_rank_ablation.py --config configs/phase_b.yaml --layer-a 18 --layer-b 23
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.activation_extraction import load_activations
from src.procrustes_alignment import low_rank_alignment, linear_projection_alignment


def run_ablation(
    X: np.ndarray,
    Y: np.ndarray,
    sample_sizes: list[int],
    ranks: list[int],
    seeds: list[int],
    regularization: float = 1e-4,
    train_fraction: float = 0.8,
) -> list[dict]:
    """Run rank-vs-sample-size ablation."""
    results = []
    total = len(sample_sizes) * (len(ranks) + 1) * len(seeds)
    done = 0

    for n_samples in sample_sizes:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=n_samples, replace=False)
            X_sub = X[idx]
            Y_sub = Y[idx]

            n_train = int(n_samples * train_fraction)
            perm = rng.permutation(n_samples)
            train_idx, test_idx = perm[:n_train], perm[n_train:]
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            Y_train, Y_test = Y_sub[train_idx], Y_sub[test_idx]

            # Low-rank at each rank
            for rank in ranks:
                result = low_rank_alignment(
                    X_train, Y_train, X_test, Y_test,
                    rank=rank, regularization=regularization,
                )
                results.append({
                    "n_samples": n_samples,
                    "rank": rank,
                    "method": f"low_rank_r{rank}",
                    "seed": seed,
                    "train_loss": result.train_loss,
                    "test_loss": result.test_loss,
                    "explained_variance": result.explained_variance,
                })
                done += 1
                print(f"  [{done}/{total}] n={n_samples}, rank={rank}, seed={seed}: "
                      f"test_loss={result.test_loss:.4f}")

            # Full ridge baseline
            result = linear_projection_alignment(
                X_train, Y_train, X_test, Y_test,
                regularization=regularization,
            )
            results.append({
                "n_samples": n_samples,
                "rank": "full",
                "method": "ridge",
                "seed": seed,
                "train_loss": result.train_loss,
                "test_loss": result.test_loss,
                "explained_variance": result.explained_variance,
            })
            done += 1
            print(f"  [{done}/{total}] n={n_samples}, rank=full, seed={seed}: "
                  f"test_loss={result.test_loss:.4f}")

    return results


def plot_ablation(results: list[dict], save_path: Path):
    """Plot rank-vs-test-loss, one line per sample size."""
    fig, ax = plt.subplots(figsize=(12, 7))

    sample_sizes = sorted(set(r["n_samples"] for r in results))
    ranks = sorted(set(r["rank"] for r in results if r["rank"] != "full"))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sample_sizes)))

    for idx, n in enumerate(sample_sizes):
        means = []
        stds = []
        for rank in ranks:
            losses = [r["test_loss"] for r in results
                      if r["n_samples"] == n and r["rank"] == rank]
            means.append(np.mean(losses))
            stds.append(np.std(losses))

        ax.errorbar(ranks, means, yerr=stds, marker="o", capsize=4,
                     linewidth=2, label=f"n={n}", color=colors[idx])

    # Ridge baseline per sample size
    for idx, n in enumerate(sample_sizes):
        ridge_losses = [r["test_loss"] for r in results
                        if r["n_samples"] == n and r["rank"] == "full"]
        if ridge_losses:
            ax.axhline(y=np.mean(ridge_losses), color=colors[idx],
                        linestyle="--", alpha=0.4)

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="No alignment")
    ax.set_xlabel("Low-Rank Dimension", fontsize=12)
    ax.set_ylabel("Test Loss (lower = better)", fontsize=12)
    ax.set_title("Rank-vs-Sample-Size Ablation (Eval B: Gemma-2B -> Qwen-1.5B, L18->L23)\n"
                 "If optimal rank stays ~32 regardless of n -> real structure\n"
                 "If optimal rank increases with n -> regularization artifact",
                 fontsize=11)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Rank-vs-sample-size ablation")
    parser.add_argument("--config", type=str, default="configs/phase_b.yaml")
    parser.add_argument("--layer-a", type=int, default=18)
    parser.add_argument("--layer-b", type=int, default=23)
    args = parser.parse_args()

    from src.config import Config
    config = Config.from_yaml(args.config)

    act_dir = config.output_dir / "activations"
    print(f"Loading activations from {act_dir}...")
    acts_a = load_activations(config.model_a.alias, act_dir)
    acts_b = load_activations(config.model_b.alias, act_dir)

    X = acts_a[args.layer_a].numpy()
    Y = acts_b[args.layer_b].numpy()
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    sample_sizes = [500, 1000, 2000, 5000, 8000]
    ranks = [4, 8, 16, 32, 64, 128, 256]
    seeds = [42, 123, 456]

    print(f"\nRunning ablation: {len(sample_sizes)} sample sizes x "
          f"{len(ranks)+1} methods x {len(seeds)} seeds = "
          f"{len(sample_sizes) * (len(ranks)+1) * len(seeds)} fits\n")

    t0 = time.time()
    results = run_ablation(
        X, Y,
        sample_sizes=sample_sizes,
        ranks=ranks,
        seeds=seeds,
        regularization=config.alignment.low_rank_regularization,
        train_fraction=config.alignment.train_fraction,
    )
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Save CSV
    out_dir = Path("outputs/intermediary")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rank_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV to {csv_path}")

    # Plot
    plot_dir = Path("outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation(results, plot_dir / "rank_ablation.png")


if __name__ == "__main__":
    main()
