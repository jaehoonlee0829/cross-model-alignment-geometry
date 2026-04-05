"""
Plot POS tag transfer results: accuracy vs low-rank dimension for both pairs,
with reference lines for source native, target oracle, cross-model oracle, and ridge.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("outputs/pos_probe_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

pairs = [
    ("cross_arch_gemma_qwen", "Cross-Architecture: Gemma-2B → Qwen-1.5B"),
    ("within_family_llama", "Within-Family: Llama-1B → Llama-3B"),
]

for ax, (pair_name, title) in zip(axes, pairs):
    sub = df[df["pair"] == pair_name]

    # Reference lines
    source_native = sub[sub["method"] == "source_native"]["top1"].values[0]
    target_oracle = sub[sub["method"] == "target_oracle"]["top1"].values[0]
    cross_oracle = sub[sub["method"] == "cross_model_oracle"]["top1"].values[0]
    ridge_val = sub[sub["method"] == "ridge"]["top1"].values[0]

    # Low-rank results
    lr = sub[sub["method"].str.startswith("low_rank")].copy()
    lr["rank"] = lr["rank"].astype(int)
    lr = lr.sort_values("rank")

    # Plot low-rank curve
    ax.plot(lr["rank"], lr["top1"] * 100, "o-", color="#2196F3", linewidth=2.5,
            markersize=8, label="Low-rank bridge", zorder=5)

    # Reference lines
    ax.axhline(target_oracle * 100, color="#4CAF50", linestyle="--", linewidth=1.5,
               label=f"Target oracle ({target_oracle*100:.1f}%)")
    ax.axhline(source_native * 100, color="#FF9800", linestyle="--", linewidth=1.5,
               label=f"Source native ({source_native*100:.1f}%)")
    ax.axhline(cross_oracle * 100, color="#F44336", linestyle=":", linewidth=2,
               label=f"Cross-model oracle ({cross_oracle*100:.1f}%)")
    ax.axhline(ridge_val * 100, color="#9C27B0", linestyle="-.", linewidth=1.5,
               label=f"Ridge / full rank ({ridge_val*100:.1f}%)")

    # Annotate best rank
    best_idx = lr["top1"].idxmax()
    best_rank = lr.loc[best_idx, "rank"]
    best_val = lr.loc[best_idx, "top1"] * 100
    ax.annotate(f"Best: r{best_rank}\n{best_val:.1f}%",
                xy=(best_rank, best_val), xytext=(best_rank * 2, best_val + 3),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    ax.set_xscale("log", base=2)
    ax.set_xticks(lr["rank"].values)
    ax.set_xticklabels([str(r) for r in lr["rank"].values])
    ax.set_xlabel("Low-Rank Dimension", fontsize=12)
    ax.set_ylabel("POS Tag Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

fig.suptitle("POS Tag Probe Transfer: Accuracy vs Bridge Rank",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/plots/pos_transfer_rank_ablation.png", dpi=100, bbox_inches="tight")
print("Saved to outputs/plots/pos_transfer_rank_ablation.png")
