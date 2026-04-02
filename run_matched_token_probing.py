"""
run_matched_token_probing.py
Runs the next-token probe transfer experiment with corrected (matched) token labels.
Same pipeline as run_probing.py but using shared vocabulary labels.

TRANSFER METHOD: Forward mapping.
  X_mapped = (X_src - X_mean) @ W + Y_mean
  Then evaluate with target model's oracle probe.

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
assert torch.cuda.is_available(), "This script requires GPU!"

PHASE_B_DIR = Path("outputs/phase_b")
PROBING_DIR = PHASE_B_DIR / "probing"
ALIGNMENT_DIR = PHASE_B_DIR / "alignment"

LAYER_A = 18  # Gemma
LAYER_B = 23  # Qwen

# Load shared labels
labels_a = np.load(PROBING_DIR / "labels_gemma_shared.npy")
labels_b = np.load(PROBING_DIR / "labels_qwen_shared.npy")
top_shared = np.load(PROBING_DIR / "top_shared_tokens.npy")

# Load activations — dict keyed by layer number
# NOTE: files are named activations_gemma.pt / activations_qwen.pt (not gemma-2-2b)
acts_a_dict = torch.load(PHASE_B_DIR / "activations" / "activations_gemma.pt",
                         map_location="cpu", weights_only=False)
acts_b_dict = torch.load(PHASE_B_DIR / "activations" / "activations_qwen.pt",
                         map_location="cpu", weights_only=False)

assert LAYER_A in acts_a_dict, f"Layer {LAYER_A} not in Gemma. Available: {sorted(acts_a_dict.keys())}"
assert LAYER_B in acts_b_dict, f"Layer {LAYER_B} not in Qwen. Available: {sorted(acts_b_dict.keys())}"
acts_a = acts_a_dict[LAYER_A].numpy()
acts_b = acts_b_dict[LAYER_B].numpy()
del acts_a_dict, acts_b_dict
print(f"Activations: A={acts_a.shape}, B={acts_b.shape}")
assert acts_a.shape[0] == len(labels_a), f"Mismatch: {acts_a.shape[0]} vs {len(labels_a)}"

# Remap to top-K classes
TOP_K = 500
token_to_class = {int(t): i for i, t in enumerate(sorted(top_shared))}
n_classes = len(token_to_class)

def remap(labels):
    return np.array([token_to_class.get(int(l), -1) for l in labels])

# Train/test split
n = len(acts_a)
rng = np.random.default_rng(42)
n_train = int(n * 0.8)
perm = rng.permutation(n)
train_idx, test_idx = perm[:n_train], perm[n_train:]

y_train_a = remap(labels_a[train_idx])
y_test_a = remap(labels_a[test_idx])
y_test_b = remap(labels_b[test_idx])
y_train_b = remap(labels_b[train_idx])

valid_train_a = y_train_a >= 0
valid_test_a = y_test_a >= 0
valid_test_b = y_test_b >= 0
valid_train_b = y_train_b >= 0

print(f"Shared vocab classes: {n_classes}")
print(f"Valid train (Gemma): {valid_train_a.sum()}")
print(f"Valid test (Gemma): {valid_test_a.sum()}")
print(f"Valid test (Qwen): {valid_test_b.sum()}")

# Train probe on Model A (Gemma)
print("\nTraining probe on Gemma (source)...")
probe_a = LinearProbe(acts_a.shape[1], n_classes, device=DEVICE)
probe_a.fit(acts_a[train_idx][valid_train_a], y_train_a[valid_train_a], epochs=30)
baseline = probe_a.evaluate(acts_a[test_idx][valid_test_a], y_test_a[valid_test_a])
print(f"Source native (Gemma): top1={baseline.accuracy_top1:.4f}, top5={baseline.accuracy_top5:.4f}")

# Train oracle on Model B (Qwen)
print("\nTraining oracle on Qwen (target)...")
probe_b = LinearProbe(acts_b.shape[1], n_classes, device=DEVICE)
probe_b.fit(acts_b[train_idx][valid_train_b], y_train_b[valid_train_b], epochs=30)
oracle = probe_b.evaluate(acts_b[test_idx][valid_test_b], y_test_b[valid_test_b])
print(f"Target oracle (Qwen): top1={oracle.accuracy_top1:.4f}, top5={oracle.accuracy_top5:.4f}")

results = []
results.append({
    "method": "source_native", "rank": "N/A",
    "top1": baseline.accuracy_top1, "top5": baseline.accuracy_top5
})
results.append({
    "method": "target_oracle", "rank": "N/A",
    "top1": oracle.accuracy_top1, "top5": oracle.accuracy_top5
})

# Cross-model oracle: B's probe on B's acts with A's labels
both_valid_test = valid_test_a & valid_test_b
cross_oracle = probe_b.evaluate(acts_b[test_idx][both_valid_test], y_test_a[both_valid_test])
print(f"Cross-model oracle (B probe, A labels, B acts): top1={cross_oracle.accuracy_top1:.4f}")
results.append({
    "method": "cross_model_oracle", "rank": "N/A",
    "top1": cross_oracle.accuracy_top1, "top5": cross_oracle.accuracy_top5
})

# Transfer via forward mapping
y_test_transfer = y_test_a
valid_test_transfer = valid_test_a

ranks = [4, 8, 16, 32, 64, 128, 256]
for rank in ranks:
    align_path = ALIGNMENT_DIR / f"layer_{LAYER_A}_to_{LAYER_B}_low_rank_rank{rank}.npz"
    if not align_path.exists():
        print(f"  Skipping rank {rank} — file not found: {align_path}")
        continue

    data = np.load(align_path)
    assert "W_B" in data.files, f"Low-rank file missing W_B: {align_path}"
    W_full = data["W"] @ data["W_B"]
    X_mean = data["X_mean"]
    Y_mean = data["Y_mean"]

    X_a_mapped = (acts_a[test_idx] - X_mean) @ W_full + Y_mean
    result = probe_b.evaluate(X_a_mapped[valid_test_transfer], y_test_transfer[valid_test_transfer])
    ratio = result.accuracy_top1 / oracle.accuracy_top1 if oracle.accuracy_top1 > 0 else 0
    print(f"  Rank {rank}: top1={result.accuracy_top1:.4f}, top5={result.accuracy_top5:.4f}, "
          f"transfer_ratio={ratio:.3f}")
    results.append({
        "method": f"low_rank_r{rank}", "rank": rank,
        "top1": result.accuracy_top1, "top5": result.accuracy_top5
    })

# Ridge
ridge_path = ALIGNMENT_DIR / f"layer_{LAYER_A}_to_{LAYER_B}_linear.npz"
if ridge_path.exists():
    data = np.load(ridge_path)
    W = data["W"]
    X_mean = data["X_mean"]
    Y_mean = data["Y_mean"]

    X_a_mapped = (acts_a[test_idx] - X_mean) @ W + Y_mean
    result = probe_b.evaluate(X_a_mapped[valid_test_transfer], y_test_transfer[valid_test_transfer])
    ratio = result.accuracy_top1 / oracle.accuracy_top1 if oracle.accuracy_top1 > 0 else 0
    print(f"  Ridge: top1={result.accuracy_top1:.4f}, top5={result.accuracy_top5:.4f}, "
          f"transfer_ratio={ratio:.3f}")
    results.append({
        "method": "ridge", "rank": "full",
        "top1": result.accuracy_top1, "top5": result.accuracy_top5
    })

df = pd.DataFrame(results)
df.to_csv(PROBING_DIR / "matched_token_probe_results.csv", index=False)
print(f"\nSaved to {PROBING_DIR / 'matched_token_probe_results.csv'}")
print(df.to_string(index=False))
