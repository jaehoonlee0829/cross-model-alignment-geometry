#!/usr/bin/env python3
"""Dual-approach binary probe transfer: frozen general vs task-specific alignment.

Approach A (FROZEN): Load pile-10k alignment, apply to task activations.
  Tests: "does the general cross-model structure carry task signal?"
Approach B (TASK): Learn alignment on task data.
  Tests: "can you build task-specific bridges?"

The frozen alignment maps source→target (e.g., Gemma L18 → Qwen L23).
So we train the probe on TARGET model and map SOURCE activations into target space.
"""

import time, csv
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
    low_rank_alignment, linear_projection_alignment, apply_mapping,
    load_alignment, AlignmentResult,
)

OUTPUT_DIR = Path("outputs/dual_probe_transfer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BINARY_DIR = Path("outputs/binary_probe_transfer")

TASKS = ["sst2_sentiment", "toxigen_toxicity", "agnews_sports"]

# Frozen alignments go source→target. We train probe on TARGET, map SOURCE→TARGET.
PAIRS = [
    {
        "name": "cross_arch_gemma_qwen",
        "source_alias": "gemma_2b", "source_layer": 18,
        "target_alias": "qwen_1.5b", "target_layer": 23,
        "frozen_dir": Path("outputs/phase_b/alignment"),
        "frozen_prefix": "layer_18_to_23",
    },
    {
        "name": "within_family_llama",
        "source_alias": "llama_1b", "source_layer": 15,
        "target_alias": "llama_3b", "target_layer": 27,  # L27 = best CKA match
        "frozen_dir": Path("outputs/eval_c/alignment"),
        "frozen_prefix": "layer_15_to_27",
    },
]

RANKS = [4, 8, 16, 32, 64, 128, 256]
SEEDS = [42, 123, 456]
TRAIN_FRACTION = 0.8
REG = 1e-4


def load_acts(alias, task, layer):
    data = torch.load(BINARY_DIR / f"acts_{alias}.pt", map_location="cpu", weights_only=True)
    return data[f"{task}_L{layer}"].numpy()


def load_frozen(pair, rank):
    """Load frozen pile-10k alignment. Returns None if not found."""
    d = pair["frozen_dir"]
    prefix = pair["frozen_prefix"]
    if rank == -1:
        f = d / f"{prefix}_linear.npz"
    else:
        f = d / f"{prefix}_low_rank_rank{rank}.npz"
    if not f.exists():
        return None
    return load_alignment(f)


def run_single(acts_src, acts_tgt, labels, frozen_align, rank, seed):
    rng = np.random.default_rng(seed)
    n = min(acts_src.shape[0], acts_tgt.shape[0])
    n_train = int(n * TRAIN_FRACTION)
    perm = rng.permutation(n)
    tr, te = perm[:n_train], perm[n_train:]

    # Train probe on TARGET (since frozen maps source→target)
    probe_tgt = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    probe_tgt.fit(acts_tgt[tr], labels[tr])
    tgt_acc = accuracy_score(labels[te], probe_tgt.predict(acts_tgt[te]))

    # Source native probe (reference)
    probe_src = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    probe_src.fit(acts_src[tr], labels[tr])
    src_acc = accuracy_score(labels[te], probe_src.predict(acts_src[te]))

    majority = max(np.mean(labels[te]), 1 - np.mean(labels[te]))

    result = {
        "rank": rank, "seed": seed,
        "src_acc": round(src_acc, 4), "tgt_acc": round(tgt_acc, 4),
        "majority": round(majority, 4),
    }

    # APPROACH A: Frozen alignment (source→target), evaluate with target probe
    if frozen_align is not None:
        mapped_src = apply_mapping(acts_src, frozen_align)
        result["frozen_acc"] = round(accuracy_score(labels[te], probe_tgt.predict(mapped_src[te])), 4)
    else:
        result["frozen_acc"] = None

    # APPROACH B: Task-specific alignment (source→target on task data)
    if rank == -1:
        task_align = linear_projection_alignment(
            acts_src[tr], acts_tgt[tr], acts_src[te], acts_tgt[te], regularization=REG)
    else:
        task_align = low_rank_alignment(
            acts_src[tr], acts_tgt[tr], acts_src[te], acts_tgt[te], rank=rank, regularization=REG)
    mapped_src_task = apply_mapping(acts_src, task_align)
    result["task_acc"] = round(accuracy_score(labels[te], probe_tgt.predict(mapped_src_task[te])), 4)

    return result


def main():
    t0 = time.time()
    all_results = []
    total = len(TASKS) * len(PAIRS) * (len(RANKS) + 1) * len(SEEDS)
    done = 0

    with open(OUTPUT_DIR / "dual_probe_results.csv", "w", newline="") as f:
        writer = None
        for task in TASKS:
            labels = np.load(BINARY_DIR / f"labels_{task}.npy")
            print(f"\n{'='*60}\nTASK: {task} | {np.mean(labels):.1%} positive\n{'='*60}")

            for pair in PAIRS:
                print(f"\n  PAIR: {pair['name']}")
                acts_src = load_acts(pair["source_alias"], task, pair["source_layer"])
                acts_tgt = load_acts(pair["target_alias"], task, pair["target_layer"])
                print(f"    src={acts_src.shape}, tgt={acts_tgt.shape}")

                # Load frozen alignments
                frozen = {}
                for r in RANKS + [-1]:
                    frozen[r] = load_frozen(pair, r)
                found = sum(1 for v in frozen.values() if v is not None)
                print(f"    Frozen alignments found: {found}/{len(RANKS)+1}")

                for rank in RANKS + [-1]:
                    for seed in SEEDS:
                        result = run_single(acts_src, acts_tgt, labels, frozen[rank], rank, seed)
                        result["task"] = task
                        result["pair"] = pair["name"]
                        all_results.append(result)
                        done += 1

                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=result.keys())
                            writer.writeheader()
                        writer.writerow(result)
                        f.flush()

                        r_str = f"r={rank:>3}" if rank > 0 else "ridge"
                        frz = result.get("frozen_acc", "N/A")
                        print(f"    [{done*100/total:5.1f}%] {r_str} s={seed} | "
                              f"frozen={frz} task={result['task_acc']:.3f} "
                              f"src={result['src_acc']:.3f} tgt={result['tgt_acc']:.3f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}min")
    print_summary(all_results)
    plot_results(all_results)


def print_summary(results):
    print(f"\n{'='*90}")
    print(f"{'Task':<20} {'Pair':<26} {'Chance':>6} {'Src':>6} {'Tgt':>6} {'Frozen':>7} {'Task':>7}")
    print("-" * 90)
    for task in TASKS:
        for pair in PAIRS:
            sub = [r for r in results if r["task"] == task and r["pair"] == pair["name"]]
            chance = np.mean([r["majority"] for r in sub])
            src = np.mean([r["src_acc"] for r in sub])
            tgt = np.mean([r["tgt_acc"] for r in sub])
            frz_vals = [r["frozen_acc"] for r in sub if r["frozen_acc"] is not None]
            frz = np.mean(frz_vals) if frz_vals else float('nan')
            tsk = np.mean([r["task_acc"] for r in sub])
            print(f"{task:<20} {pair['name']:<26} {chance:>6.3f} {src:>6.3f} "
                  f"{tgt:>6.3f} {frz:>7.3f} {tsk:>7.3f}")
    print("=" * 90)
    print("Frozen = pile-10k alignment (frozen W) | Task = task-specific alignment")


def plot_results(results):
    tasks = sorted(set(r["task"] for r in results))
    pairs = [p["name"] for p in PAIRS]
    fig, axes = plt.subplots(len(tasks), len(pairs), figsize=(8*len(pairs), 5*len(tasks)), squeeze=False)

    for row, task in enumerate(tasks):
        for col, pair_name in enumerate(pairs):
            ax = axes[row][col]
            sub = [r for r in results if r["task"] == task and r["pair"] == pair_name]
            ranks_plot = sorted(set(r["rank"] for r in sub if r["rank"] > 0))

            for key, color, label in [
                ("frozen_acc", "#E53935", "Frozen (pile-10k)"),
                ("task_acc", "#2196F3", "Task-specific"),
            ]:
                means = [np.mean([r[key] for r in sub if r["rank"] == rank and r[key] is not None])
                         for rank in ranks_plot]
                stds = [np.std([r[key] for r in sub if r["rank"] == rank and r[key] is not None])
                        for rank in ranks_plot]
                if any(not np.isnan(m) for m in means):
                    ax.errorbar(ranks_plot, means, yerr=stds, marker='o', linewidth=2,
                                 capsize=4, label=label, color=color)

            src = np.mean([r["src_acc"] for r in sub])
            tgt = np.mean([r["tgt_acc"] for r in sub])
            chance = np.mean([r["majority"] for r in sub])
            ax.axhline(y=tgt, color='purple', linestyle='--', alpha=0.6, label=f'Target native ({tgt:.3f})')
            ax.axhline(y=src, color='green', linestyle='-', alpha=0.5, label=f'Source native ({src:.3f})')
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Rank'); ax.set_ylabel('Accuracy')
            ax.set_title(f'{task}\n{pair_name}', fontsize=11)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(0.35, 1.0)

    plt.suptitle('Frozen (pile-10k) vs Task-Specific Alignment', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dual_probe_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dual_probe_comparison.png'}")


if __name__ == "__main__":
    main()
