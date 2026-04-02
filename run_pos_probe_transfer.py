"""
run_pos_probe_transfer.py
Trains POS tag probes on Model A, transfers via bridge, evaluates on Model B.
17-way classification, completely tokenizer-independent.

TRANSFER METHOD: Forward mapping.
  X_mapped = (X_src - X_mean) @ W + Y_mean, then evaluate with target's oracle probe.

ALL COMPUTATION ON GPU.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(".")
from src.linear_probing import LinearProbe

DEVICE = "cuda"
assert torch.cuda.is_available(), "GPU required!"

N_POS_CLASSES = 17

# Model pairs to test
PAIRS = [
    {
        "name": "cross_arch_gemma_qwen",
        "model_a": "gemma", "layer_a": 18,
        "model_b": "qwen", "layer_b": 23,
        "acts_dir": "outputs/phase_b",
    },
    {
        "name": "within_family_llama",
        "model_a": "llama-1b", "layer_a": 15,
        "model_b": "llama-3b", "layer_b": 27,
        "acts_dir": "outputs/eval_c",
    },
]

all_results = []

for pair in PAIRS:
    print(f"\n{'='*60}")
    print(f"{pair['name']}: {pair['model_a']} L{pair['layer_a']} -> {pair['model_b']} L{pair['layer_b']}")
    print(f"{'='*60}")

    acts_dir = Path(pair["acts_dir"])
    acts_a_dict = torch.load(acts_dir / "activations" / f"activations_{pair['model_a']}.pt",
                             map_location="cpu", weights_only=False)
    acts_b_dict = torch.load(acts_dir / "activations" / f"activations_{pair['model_b']}.pt",
                             map_location="cpu", weights_only=False)

    la_key, lb_key = pair["layer_a"], pair["layer_b"]
    assert la_key in acts_a_dict, f"Layer {la_key} not in {pair['model_a']}. Available: {sorted(acts_a_dict.keys())}"
    assert lb_key in acts_b_dict, f"Layer {lb_key} not in {pair['model_b']}. Available: {sorted(acts_b_dict.keys())}"
    acts_a = acts_a_dict[la_key].numpy()
    acts_b = acts_b_dict[lb_key].numpy()
    del acts_a_dict, acts_b_dict
    print(f"Activations: A={acts_a.shape}, B={acts_b.shape}")

    # Load POS labels — alias matches activation file names
    pos_a = np.load(f"outputs/pos_labels_{pair['model_a']}.npy")
    pos_b = np.load(f"outputs/pos_labels_{pair['model_b']}.npy")

    assert len(pos_a) == len(acts_a), \
        f"POS label count ({len(pos_a)}) != activation count ({len(acts_a)}) for {pair['model_a']}"
    assert len(pos_b) == len(acts_b), \
        f"POS label count ({len(pos_b)}) != activation count ({len(acts_b)}) for {pair['model_b']}"

    # Train/test split
    n = len(acts_a)
    rng = np.random.default_rng(42)
    n_train = int(n * 0.8)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    y_train_a = pos_a[train_idx]
    y_test_a = pos_a[test_idx]
    y_test_b = pos_b[test_idx]
    y_train_b = pos_b[train_idx]

    valid_train_a = y_train_a >= 0
    valid_test_a = y_test_a >= 0
    valid_test_b = y_test_b >= 0
    valid_train_b = y_train_b >= 0

    print(f"Valid train A: {valid_train_a.sum()}, Valid test A: {valid_test_a.sum()}")
    print(f"Valid train B: {valid_train_b.sum()}, Valid test B: {valid_test_b.sum()}")

    # Train source probe
    print("Training source probe...")
    probe_a = LinearProbe(acts_a.shape[1], N_POS_CLASSES, device=DEVICE)
    probe_a.fit(acts_a[train_idx][valid_train_a], y_train_a[valid_train_a], epochs=30)
    baseline = probe_a.evaluate(acts_a[test_idx][valid_test_a], y_test_a[valid_test_a])
    print(f"Source native: top1={baseline.accuracy_top1:.4f}")

    # Train oracle
    print("Training oracle...")
    probe_b = LinearProbe(acts_b.shape[1], N_POS_CLASSES, device=DEVICE)
    probe_b.fit(acts_b[train_idx][valid_train_b], y_train_b[valid_train_b], epochs=30)
    oracle = probe_b.evaluate(acts_b[test_idx][valid_test_b], y_test_b[valid_test_b])
    print(f"Target oracle: top1={oracle.accuracy_top1:.4f}")

    all_results.append({"pair": pair["name"], "method": "source_native", "rank": "N/A", "top1": baseline.accuracy_top1})
    all_results.append({"pair": pair["name"], "method": "target_oracle", "rank": "N/A", "top1": oracle.accuracy_top1})

    # Transfer via alignments
    align_dir = acts_dir / "alignment"
    la, lb = pair["layer_a"], pair["layer_b"]

    for rank in [4, 8, 16, 32, 64, 128, 256]:
        align_path = align_dir / f"layer_{la}_to_{lb}_low_rank_rank{rank}.npz"
        if not align_path.exists():
            print(f"  Skipping rank {rank} — not found: {align_path}")
            continue

        data = np.load(align_path)
        assert "W_B" in data.files, f"Low-rank file missing W_B: {align_path}"
        W_full = data["W"] @ data["W_B"]
        X_mean = data["X_mean"]
        Y_mean = data["Y_mean"]

        X_a_mapped = (acts_a[test_idx] - X_mean) @ W_full + Y_mean

        result = probe_b.evaluate(X_a_mapped[valid_test_a], y_test_a[valid_test_a])
        ratio = result.accuracy_top1 / oracle.accuracy_top1 if oracle.accuracy_top1 > 0 else 0
        print(f"  Rank {rank}: top1={result.accuracy_top1:.4f}, ratio={ratio:.3f}")
        all_results.append({"pair": pair["name"], "method": f"low_rank_r{rank}", "rank": rank, "top1": result.accuracy_top1})

    # Ridge
    ridge_path = align_dir / f"layer_{la}_to_{lb}_linear.npz"
    if ridge_path.exists():
        data = np.load(ridge_path)
        W = data["W"]
        X_mean = data["X_mean"]
        Y_mean = data["Y_mean"]

        X_a_mapped = (acts_a[test_idx] - X_mean) @ W + Y_mean
        result = probe_b.evaluate(X_a_mapped[valid_test_a], y_test_a[valid_test_a])
        ratio = result.accuracy_top1 / oracle.accuracy_top1 if oracle.accuracy_top1 > 0 else 0
        print(f"  Ridge: top1={result.accuracy_top1:.4f}, ratio={ratio:.3f}")
        all_results.append({"pair": pair["name"], "method": "ridge", "rank": "full", "top1": result.accuracy_top1})

df = pd.DataFrame(all_results)
df.to_csv("outputs/pos_probe_results.csv", index=False)
print(f"\nAll results saved to outputs/pos_probe_results.csv")
print(df.to_string(index=False))
