#!/usr/bin/env python3
"""Binary classification probe transfer across model pairs.

Tests whether cross-model alignment preserves task-relevant signal for
BINARY tasks (much easier than 32k-class next-token prediction).

For each (task, model_pair, rank), we:
  1. Train logistic regression probe on source model activations
  2. Learn alignment: target -> source space at given rank
  3. Apply source probe to aligned target activations
  4. Compare: transfer_acc vs source_native vs target_native vs 50% chance

Key question: does cross-architecture transfer beat 50% on binary tasks,
even though it completely failed on next-token prediction?
"""

import time
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.procrustes_alignment import (
    low_rank_alignment, linear_projection_alignment, apply_mapping
)

OUTPUT_DIR = Path("outputs/binary_probe_transfer")
RESULTS_CSV = OUTPUT_DIR / "binary_probe_results.csv"

TASKS = ["sst2_sentiment", "toxigen_toxicity", "agnews_sports"]

# Model pairs: (source_alias, source_layer, target_alias, target_layer, pair_name)
PAIRS = [
    ("gemma_2b", 18, "qwen_1.5b", 23, "cross_arch_gemma_qwen"),
    ("llama_1b", 15, "llama_3b",  16, "within_family_llama"),
]

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEEDS = [42, 123, 456]  # 3 seeds for error bars
TRAIN_FRACTION = 0.8
REGULARIZATION = 1e-4


def load_acts(alias: str, task: str, layer: int) -> np.ndarray:
    """Load activations for a model/task/layer."""
    path = OUTPUT_DIR / f"acts_{alias}.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)
    key = f"{task}_L{layer}"
    return data[key].numpy()


def run_single(
    acts_src: np.ndarray,
    acts_tgt: np.ndarray,
    labels: np.ndarray,
    rank: int,
    seed: int,
) -> dict:
    """Run one probe transfer experiment. rank=-1 means full ridge."""
    rng = np.random.default_rng(seed)
    n = acts_src.shape[0]
    n_train = int(n * TRAIN_FRACTION)
    perm = rng.permutation(n)
    tr, te = perm[:n_train], perm[n_train:]

    # 1. Train probe on source
    probe_src = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    probe_src.fit(acts_src[tr], labels[tr])
    src_train_acc = accuracy_score(labels[tr], probe_src.predict(acts_src[tr]))
    src_test_acc = accuracy_score(labels[te], probe_src.predict(acts_src[te]))

    # 2. Train probe on target (ceiling)
    probe_tgt = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    probe_tgt.fit(acts_tgt[tr], labels[tr])
    tgt_test_acc = accuracy_score(labels[te], probe_tgt.predict(acts_tgt[te]))

    # 3. Learn alignment: target -> source space
    if rank == -1:
        alignment = linear_projection_alignment(
            X_train=acts_tgt[tr], Y_train=acts_src[tr],
            X_test=acts_tgt[te], Y_test=acts_src[te],
            regularization=REGULARIZATION,
        )
        method = "full_ridge"
    else:
        alignment = low_rank_alignment(
            X_train=acts_tgt[tr], Y_train=acts_src[tr],
            X_test=acts_tgt[te], Y_test=acts_src[te],
            rank=rank, regularization=REGULARIZATION,
        )
        method = f"low_rank_{rank}"

    # 4. Apply alignment, run source probe on mapped target
    mapped_tgt = apply_mapping(acts_tgt, alignment)
    transfer_acc = accuracy_score(labels[te], probe_src.predict(mapped_tgt[te]))

    # 5. Random baseline
    majority = max(np.mean(labels[te]), 1 - np.mean(labels[te]))

    return {
        "method": method,
        "rank": rank,
        "seed": seed,
        "src_train_acc": round(src_train_acc, 4),
        "src_test_acc": round(src_test_acc, 4),
        "tgt_direct_acc": round(tgt_test_acc, 4),
        "transfer_acc": round(transfer_acc, 4),
        "majority_baseline": round(majority, 4),
        "alignment_test_loss": round(alignment.test_loss, 4),
        "alignment_r2": round(alignment.explained_variance, 4),
    }


def main():
    overall_start = time.time()

    # Count total experiments
    total_exps = len(TASKS) * len(PAIRS) * (len(RANKS) + 1) * len(SEEDS)
    done = 0

    all_results = []

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = None

        for task in TASKS:
            labels = np.load(OUTPUT_DIR / f"labels_{task}.npy")
            print(f"\n{'='*70}")
            print(f"TASK: {task} | {len(labels)} samples | "
                  f"{np.mean(labels):.1%} positive")
            print(f"{'='*70}")

            for src_alias, src_layer, tgt_alias, tgt_layer, pair_name in PAIRS:
                print(f"\n  PAIR: {pair_name}")
                print(f"    Source: {src_alias} L{src_layer}")
                print(f"    Target: {tgt_alias} L{tgt_layer}")

                acts_src = load_acts(src_alias, task, src_layer)
                acts_tgt = load_acts(tgt_alias, task, tgt_layer)
                print(f"    Source shape: {acts_src.shape}, Target shape: {acts_tgt.shape}")

                for rank in RANKS + [-1]:
                    for seed in SEEDS:
                        t0 = time.time()
                        result = run_single(acts_src, acts_tgt, labels, rank, seed)
                        result["task"] = task
                        result["pair_name"] = pair_name
                        result["source_model"] = src_alias
                        result["source_layer"] = src_layer
                        result["target_model"] = tgt_alias
                        result["target_layer"] = tgt_layer
                        dt = time.time() - t0

                        all_results.append(result)
                        done += 1

                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=result.keys())
                            writer.writeheader()
                        writer.writerow(result)
                        f.flush()

                        total_elapsed = time.time() - overall_start
                        pct = done / total_exps * 100
                        rate = done / total_elapsed
                        eta = (total_exps - done) / rate if rate > 0 else 0

                        rank_str = f"rank={rank:>3}" if rank > 0 else "full_ridge"
                        print(f"    [{pct:5.1f}%] {task}/{pair_name} | "
                              f"{rank_str} | seed={seed} | "
                              f"transfer={result['transfer_acc']:.3f} | "
                              f"src={result['src_test_acc']:.3f} | "
                              f"tgt_direct={result['tgt_direct_acc']:.3f} | "
                              f"{dt:.1f}s | ETA: {eta:.0f}s")

    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS DONE in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Results: {RESULTS_CSV}")
    print(f"{'='*70}")

    # ── Plots ─────────────────────────────────────────────────────────────
    generate_plots(all_results)
    print_summary(all_results)


def generate_plots(all_results: list[dict]):
    """One subplot per (task, pair). 3 tasks x 2 pairs = 6 panels."""
    tasks = sorted(set(r["task"] for r in all_results))
    pairs = sorted(set(r["pair_name"] for r in all_results))

    fig, axes = plt.subplots(len(tasks), len(pairs),
                              figsize=(7 * len(pairs), 5 * len(tasks)),
                              squeeze=False)

    for row, task in enumerate(tasks):
        for col, pair_name in enumerate(pairs):
            ax = axes[row][col]
            subset = [r for r in all_results
                      if r["task"] == task and r["pair_name"] == pair_name]

            # Group by rank
            rank_accs = {}
            for r in subset:
                rank_accs.setdefault(r["rank"], []).append(r["transfer_acc"])

            low_rank = {k: v for k, v in rank_accs.items() if k > 0}
            ridge = rank_accs.get(-1, [])

            ranks_sorted = sorted(low_rank.keys())
            means = [np.mean(low_rank[r]) for r in ranks_sorted]
            stds = [np.std(low_rank[r]) for r in ranks_sorted]

            ax.errorbar(ranks_sorted, means, yerr=stds, marker='o',
                         linewidth=2, capsize=4, label='Low-rank transfer',
                         color='#2196F3', zorder=3)

            # Reference lines
            src_acc = np.mean([r["src_test_acc"] for r in subset])
            tgt_acc = np.mean([r["tgt_direct_acc"] for r in subset])
            majority = np.mean([r["majority_baseline"] for r in subset])

            ax.axhline(y=src_acc, color='green', linestyle='-', alpha=0.7,
                         label=f'Source native ({src_acc:.3f})')
            ax.axhline(y=tgt_acc, color='purple', linestyle='--', alpha=0.7,
                         label=f'Target native ({tgt_acc:.3f})')
            ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.7,
                         label='Chance (0.500)')
            ax.axhline(y=majority, color='orange', linestyle=':', alpha=0.5,
                         label=f'Majority ({majority:.3f})')

            if ridge:
                ridge_mean = np.mean(ridge)
                ax.axhline(y=ridge_mean, color='#FF9800', linestyle='-.',
                             alpha=0.7, label=f'Full ridge ({ridge_mean:.3f})')

            ax.set_xscale('log', base=2)
            ax.set_xlabel('Alignment Rank')
            ax.set_ylabel('Transfer Accuracy')
            ax.set_title(f'{task}\n{pair_name.replace("_", " ")}', fontsize=11)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(ranks_sorted)
            ax.set_xticklabels([str(r) for r in ranks_sorted], fontsize=8)
            ax.set_ylim(0.35, 1.0)

    plt.suptitle(
        'Binary Probe Transfer: Does Cross-Model Alignment Preserve Task Signal?\n'
        'Transfer accuracy vs alignment rank for 3 tasks x 2 model pairs',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "binary_probe_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {OUTPUT_DIR / 'binary_probe_results.png'}")


def print_summary(all_results: list[dict]):
    """Print a clean summary table."""
    tasks = sorted(set(r["task"] for r in all_results))
    pairs = sorted(set(r["pair_name"] for r in all_results))

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Task':<22} {'Pair':<28} {'Chance':>7} {'Src':>7} {'Tgt':>7} "
          f"{'Ridge':>7} {'Best LR':>7} {'@rank':>6}")
    print("-" * 80)

    for task in tasks:
        for pair_name in pairs:
            subset = [r for r in all_results
                      if r["task"] == task and r["pair_name"] == pair_name]

            chance = np.mean([r["majority_baseline"] for r in subset])
            src = np.mean([r["src_test_acc"] for r in subset])
            tgt = np.mean([r["tgt_direct_acc"] for r in subset])
            ridge = np.mean([r["transfer_acc"] for r in subset if r["rank"] == -1])

            # Best low-rank
            rank_means = {}
            for r in subset:
                if r["rank"] > 0:
                    rank_means.setdefault(r["rank"], []).append(r["transfer_acc"])
            best_rank = max(rank_means, key=lambda k: np.mean(rank_means[k]))
            best_lr = np.mean(rank_means[best_rank])

            print(f"{task:<22} {pair_name:<28} {chance:>7.3f} {src:>7.3f} "
                  f"{tgt:>7.3f} {ridge:>7.3f} {best_lr:>7.3f} {best_rank:>5}")

    print("=" * 80)

    # Key comparison
    print("\nKEY QUESTION: Does cross-arch transfer beat 50% chance?")
    for task in tasks:
        cross = [r for r in all_results
                 if r["task"] == task and r["pair_name"] == "cross_arch_gemma_qwen"]
        within = [r for r in all_results
                  if r["task"] == task and r["pair_name"] == "within_family_llama"]

        cross_best = max(np.mean([r["transfer_acc"] for r in cross if r["rank"] == rank])
                         for rank in RANKS + [-1]
                         if any(r["rank"] == rank for r in cross))
        within_best = max(np.mean([r["transfer_acc"] for r in within if r["rank"] == rank])
                          for rank in RANKS + [-1]
                          if any(r["rank"] == rank for r in within))

        above_chance = "YES" if cross_best > 0.55 else ("MARGINAL" if cross_best > 0.52 else "NO")
        print(f"  {task}: cross={cross_best:.3f} within={within_best:.3f} "
              f"-> Cross above chance? {above_chance}")


if __name__ == "__main__":
    main()
