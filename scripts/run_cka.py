#!/usr/bin/env python3
"""Compute CKA similarity between two models' activations.

Usage:
    python scripts/run_cka.py --config configs/default.yaml
    python scripts/run_cka.py --config configs/default.yaml --kernel rbf

Requires: run_extraction.py has been run first.
"""

import argparse
from pathlib import Path

import numpy as np

from src.config import Config
from src.activation_extraction import load_activations
from src.cka_analysis import (
    compute_cka_matrix,
    find_best_layer_pairs,
    plot_cka_heatmap,
    print_cka_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Compute CKA similarity matrix")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--kernel", type=str, default=None, help="Override kernel type")
    parser.add_argument("--threshold", type=float, default=0.3, help="CKA threshold for best pairs")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    kernel = args.kernel or config.cka.kernel

    act_dir = config.output_dir / "activations"

    # Load pre-extracted activations
    print(f"Loading activations from {act_dir}...")
    acts_a = load_activations(config.model_a.alias, act_dir)
    acts_b = load_activations(config.model_b.alias, act_dir)

    print(f"Model A ({config.model_a.alias}): {len(acts_a)} layers")
    print(f"Model B ({config.model_b.alias}): {len(acts_b)} layers")

    # Compute CKA matrix
    cka_matrix, layers_a, layers_b = compute_cka_matrix(
        acts_a, acts_b,
        kernel=kernel,
        debiased=config.cka.debiased,
        subsample_n=config.cka.subsample_n,
    )

    # Print summary
    print_cka_summary(cka_matrix, layers_a, layers_b, config.model_a.alias, config.model_b.alias)

    # Find best layer pairs
    best_pairs = find_best_layer_pairs(cka_matrix, layers_a, layers_b, threshold=args.threshold)
    print(f"\nBest layer pairs (CKA >= {args.threshold}):")
    for la, lb, score in best_pairs:
        print(f"  Layer {la} ({config.model_a.alias}) <-> Layer {lb} ({config.model_b.alias}): CKA={score:.4f}")

    # Save results
    results_dir = config.output_dir / "cka"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save heatmap
    plot_cka_heatmap(
        cka_matrix, layers_a, layers_b,
        alias_a=config.model_a.alias,
        alias_b=config.model_b.alias,
        save_path=results_dir / "cka_heatmap.png",
    )

    # Save raw matrix
    np.savez(
        results_dir / "cka_results.npz",
        cka_matrix=cka_matrix,
        layers_a=layers_a,
        layers_b=layers_b,
        kernel=kernel,
    )

    # Save best pairs
    np.savez(
        results_dir / "best_layer_pairs.npz",
        pairs=np.array(best_pairs),
    )

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
