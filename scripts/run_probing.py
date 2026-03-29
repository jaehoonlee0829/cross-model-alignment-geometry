#!/usr/bin/env python3
"""Linear probe transfer experiment.

Tests whether low-rank alignment preserves task-relevant signal by:
1. Training a linear probe (next-token prediction) on Model A's activations
2. Transferring via rank-k alignment to Model B's space
3. Evaluating transferred probe on Model B

Usage:
    python scripts/run_probing.py --config configs/phase_b.yaml
    python scripts/run_probing.py --config configs/eval_c.yaml --layer-a 15 --layer-b 27
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.config import Config
from src.activation_extraction import load_activations, load_prompts
from src.linear_probing import LinearProbe, extract_next_token_labels
from src.procrustes_alignment import load_alignment


def main():
    parser = argparse.ArgumentParser(description="Linear probe transfer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--layer-a", type=int, default=None, help="Specific source layer")
    parser.add_argument("--layer-b", type=int, default=None, help="Specific target layer")
    parser.add_argument("--epochs", type=int, default=30, help="Probe training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device for probe training")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    act_dir = config.output_dir / "activations"
    align_dir = config.output_dir / "alignment"

    # Load activations
    print(f"Loading activations from {act_dir}...")
    acts_a = load_activations(config.model_a.alias, act_dir)
    acts_b = load_activations(config.model_b.alias, act_dir)

    # Extract or load next-token labels for both models
    labels_dir = config.output_dir / "probing"
    labels_dir.mkdir(parents=True, exist_ok=True)

    labels_a_path = labels_dir / f"labels_{config.model_a.alias}.npy"
    labels_b_path = labels_dir / f"labels_{config.model_b.alias}.npy"

    texts = load_prompts(config.extraction)

    if labels_a_path.exists():
        print(f"Loading cached labels from {labels_a_path}")
        labels_a = np.load(labels_a_path)
    else:
        labels_a = extract_next_token_labels(
            config.model_a.name, texts,
            max_seq_len=config.extraction.max_seq_len,
            batch_size=config.extraction.batch_size,
        )
        np.save(labels_a_path, labels_a)

    if labels_b_path.exists():
        print(f"Loading cached labels from {labels_b_path}")
        labels_b = np.load(labels_b_path)
    else:
        labels_b = extract_next_token_labels(
            config.model_b.name, texts,
            max_seq_len=config.extraction.max_seq_len,
            batch_size=config.extraction.batch_size,
        )
        np.save(labels_b_path, labels_b)

    # Use Model B's labels for evaluation (what does Model B predict?)
    # The probe tests: can we predict Model B's behavior from Model A's activations?

    # Determine layer pairs
    if args.layer_a is not None and args.layer_b is not None:
        layer_pairs = [(args.layer_a, args.layer_b)]
    else:
        # Use best CKA pairs
        cka_dir = config.output_dir / "cka"
        pairs_data = np.load(cka_dir / "best_layer_pairs.npz")
        raw_pairs = pairs_data["pairs"]
        layer_pairs = [(int(row[0]), int(row[1])) for row in raw_pairs[:3]]

    # Find available ranks from alignment directory
    available_ranks = set()
    for f in align_dir.iterdir():
        if "low_rank_rank" in f.name:
            rank = int(f.name.split("rank")[-1].replace(".npz", ""))
            available_ranks.add(rank)
    ranks = sorted(available_ranks)
    print(f"Available ranks: {ranks}")

    # Train/test split (same as alignment)
    n = next(iter(acts_a.values())).shape[0]
    rng = np.random.default_rng(config.seed)
    n_train = int(n * config.alignment.train_fraction)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    # Filter to valid labels and reduce to top-K most frequent tokens
    # Full vocab (~256k for Gemma) is too large for logistic regression on CPU
    TOP_K = 500  # Use top-500 most frequent tokens as classes

    # Find most common tokens across both models
    all_labels = np.concatenate([labels_a[labels_a >= 0], labels_b[labels_b >= 0]])
    token_counts = np.bincount(all_labels)
    top_tokens = np.argsort(token_counts)[-TOP_K:]
    top_token_set = set(top_tokens)

    # Remap to contiguous class IDs; mark rare tokens as -1
    token_to_class = {t: i for i, t in enumerate(sorted(top_tokens))}
    n_classes = len(token_to_class)
    print(f"Using top {n_classes} most frequent tokens as classes "
          f"(covers {token_counts[top_tokens].sum() / len(all_labels) * 100:.1f}% of data)")

    def remap(labels):
        return np.array([token_to_class.get(l, -1) for l in labels])

    results = []

    for la, lb in layer_pairs:
        print(f"\n{'='*60}")
        print(f"Layer {la} ({config.model_a.alias}) -> Layer {lb} ({config.model_b.alias})")
        print(f"{'='*60}")

        X_a = acts_a[la].numpy()
        X_b = acts_b[lb].numpy()

        # Train probe on Model A
        y_train_raw = remap(labels_a[train_idx])
        valid_train = y_train_raw >= 0
        X_train = X_a[train_idx][valid_train]
        y_train = y_train_raw[valid_train]

        print(f"  Training probe on Model A ({len(X_train)} samples, {n_classes} classes)...")
        probe = LinearProbe(X_a.shape[1], n_classes, device=args.device)
        losses = probe.fit(X_train, y_train, epochs=args.epochs)
        print(f"  Final training loss: {losses[-1]:.4f}")

        # Evaluate probe on Model A (held-out)
        y_test_a = remap(labels_a[test_idx])
        valid_test_a = y_test_a >= 0
        baseline = probe.evaluate(X_a[test_idx][valid_test_a], y_test_a[valid_test_a])
        print(f"  Baseline (Model A test): top1={baseline.accuracy_top1:.4f}, "
              f"top5={baseline.accuracy_top5:.4f}")

        results.append({
            "layer_a": la, "layer_b": lb, "method": "baseline_model_a",
            "rank": "N/A",
            "top1_accuracy": baseline.accuracy_top1,
            "top5_accuracy": baseline.accuracy_top5,
            "loss": baseline.loss,
        })

        # Transfer via each rank
        y_test_b = remap(labels_b[test_idx])
        valid_test_b = y_test_b >= 0

        # Direct probe on Model B (oracle upper bound)
        y_train_b = remap(labels_b[train_idx])
        valid_train_b = y_train_b >= 0

        print(f"  Training oracle probe on Model B ({valid_train_b.sum()} samples)...")
        probe_b = LinearProbe(X_b.shape[1], n_classes, device=args.device)
        probe_b.fit(X_b[train_idx][valid_train_b], y_train_b[valid_train_b], epochs=args.epochs)
        oracle = probe_b.evaluate(X_b[test_idx][valid_test_b], y_test_b[valid_test_b])

        # Transfer test: map Model A activations into Model B's space via alignment,
        # then evaluate using Model B's native probe.
        # This tests: "Does rank-k alignment preserve enough structure for Model B's
        # probe to read Model A's mapped activations?"
        # Use Model A's labels for evaluation (what does Model A encode?)
        y_test_a_b = remap(labels_a[test_idx])  # Model A's predictions
        valid_test_ab = y_test_a_b >= 0

        for rank in ranks:
            align_file = align_dir / f"layer_{la}_to_{lb}_low_rank_rank{rank}.npz"
            if not align_file.exists():
                continue

            alignment = load_alignment(align_file)
            W = alignment.W @ alignment.W_B  # Reconstruct full W = A @ B

            # Map Model A test activations into Model B's space
            X_a_mapped = (X_a[test_idx] - alignment._X_mean) @ W + alignment._Y_mean

            result = probe_b.evaluate(X_a_mapped[valid_test_ab], y_test_a_b[valid_test_ab])
            ratio = result.accuracy_top1 / oracle.accuracy_top1 if oracle.accuracy_top1 > 0 else 0
            print(f"  Rank {rank:>4}: top1={result.accuracy_top1:.4f}, "
                  f"top5={result.accuracy_top5:.4f}, "
                  f"transfer_ratio={ratio:.3f}")

            results.append({
                "layer_a": la, "layer_b": lb, "method": f"low_rank_r{rank}",
                "rank": rank,
                "top1_accuracy": result.accuracy_top1,
                "top5_accuracy": result.accuracy_top5,
                "loss": result.loss,
            })

        # Transfer via ridge (full rank)
        ridge_file = align_dir / f"layer_{la}_to_{lb}_linear.npz"
        if ridge_file.exists():
            alignment = load_alignment(ridge_file)
            X_a_mapped = (X_a[test_idx] - alignment._X_mean) @ alignment.W + alignment._Y_mean
            result = probe_b.evaluate(X_a_mapped[valid_test_ab], y_test_a_b[valid_test_ab])
            ratio = result.accuracy_top1 / oracle.accuracy_top1 if oracle.accuracy_top1 > 0 else 0
            print(f"  Ridge:    top1={result.accuracy_top1:.4f}, "
                  f"top5={result.accuracy_top5:.4f}, "
                  f"transfer_ratio={ratio:.3f}")
            results.append({
                "layer_a": la, "layer_b": lb, "method": "ridge",
                "rank": "full",
                "top1_accuracy": result.accuracy_top1,
                "top5_accuracy": result.accuracy_top5,
                "loss": result.loss,
            })
        print(f"  Oracle (Model B native): top1={oracle.accuracy_top1:.4f}, "
              f"top5={oracle.accuracy_top5:.4f}")
        results.append({
            "layer_a": la, "layer_b": lb, "method": "oracle_model_b",
            "rank": "N/A",
            "top1_accuracy": oracle.accuracy_top1,
            "top5_accuracy": oracle.accuracy_top5,
            "loss": oracle.loss,
        })

    # Save results
    csv_path = labels_dir / "probe_transfer_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {csv_path}")

    # Plot
    plot_probe_transfer(results, labels_dir / "probe_transfer.png")


def plot_probe_transfer(results: list[dict], save_path: Path):
    """Plot rank vs transferred probe accuracy."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by layer pair
    layer_pairs = sorted(set((r["layer_a"], r["layer_b"]) for r in results))

    for la, lb in layer_pairs:
        pair_results = [r for r in results if r["layer_a"] == la and r["layer_b"] == lb]

        # Get baseline and oracle
        baseline = next((r for r in pair_results if r["method"] == "baseline_model_a"), None)
        oracle = next((r for r in pair_results if r["method"] == "oracle_model_b"), None)

        # Get rank results
        rank_results = [(r["rank"], r["top1_accuracy"]) for r in pair_results
                        if r["method"].startswith("low_rank")]
        rank_results.sort(key=lambda x: x[0])

        if rank_results:
            ranks, accs = zip(*rank_results)
            ax.plot(ranks, accs, marker="o", linewidth=2,
                    label=f"L{la}->L{lb} (transferred)")

        # Ridge
        ridge = next((r for r in pair_results if r["method"] == "ridge"), None)
        if ridge:
            ax.scatter([512], [ridge["top1_accuracy"]], marker="x", s=100,
                       zorder=5, label=f"L{la}->L{lb} ridge")

    # Add reference lines
    if baseline:
        ax.axhline(y=baseline["top1_accuracy"], color="blue", linestyle="--",
                    alpha=0.5, label=f"Model A baseline ({baseline['top1_accuracy']:.3f})")
    if oracle:
        ax.axhline(y=oracle["top1_accuracy"], color="green", linestyle="--",
                    alpha=0.5, label=f"Model B oracle ({oracle['top1_accuracy']:.3f})")

    ax.set_xlabel("Alignment Rank", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (next-token prediction)", fontsize=12)
    ax.set_title("Linear Probe Transfer: Accuracy vs Alignment Rank\n"
                 "Does the low-rank subspace carry task-relevant signal?", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
