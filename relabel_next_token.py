"""
relabel_next_token.py
Remaps raw token IDs to shared class IDs using the vocabulary mapping.
Tokens not in the shared vocabulary are marked as -1 (excluded).
"""
import json
import numpy as np

with open("outputs/vocab_mapping.json") as f:
    mapping_dict = json.load(f)

gemma_qwen_map = mapping_dict["gemma_qwen"]
a_all_ids = mapping_dict["gemma_all_ids_per_class"]
b_all_ids = mapping_dict["qwen_all_ids_per_class"]

# Build lookup tables using ALL token ID variants
gemma_to_shared = {}
qwen_to_shared = {}

for class_id_str, token_ids in a_all_ids.items():
    class_id = int(class_id_str)
    for tid in token_ids:
        gemma_to_shared[tid] = class_id

for class_id_str, token_ids in b_all_ids.items():
    class_id = int(class_id_str)
    for tid in token_ids:
        qwen_to_shared[tid] = class_id

n_shared_classes = len(gemma_qwen_map)
print(f"Total shared classes: {n_shared_classes}")
print(f"Gemma token IDs covered: {len(gemma_to_shared)}")
print(f"Qwen token IDs covered: {len(qwen_to_shared)}")

# Load existing raw token ID labels
labels_a = np.load("outputs/phase_b/probing/labels_gemma.npy")
labels_b = np.load("outputs/phase_b/probing/labels_qwen.npy")

def remap_labels(labels, token_to_shared):
    remapped = np.full_like(labels, -1)
    for i, label in enumerate(labels):
        if int(label) in token_to_shared:
            remapped[i] = token_to_shared[int(label)]
    return remapped

labels_a_shared = remap_labels(labels_a, gemma_to_shared)
labels_b_shared = remap_labels(labels_b, qwen_to_shared)

valid_a = (labels_a_shared >= 0).sum()
valid_b = (labels_b_shared >= 0).sum()
print(f"Valid labels (Gemma): {valid_a}/{len(labels_a)} ({valid_a/len(labels_a)*100:.1f}%)")
print(f"Valid labels (Qwen): {valid_b}/{len(labels_b)} ({valid_b/len(labels_b)*100:.1f}%)")

if valid_a < 500 or valid_b < 500:
    raise ValueError(f"Too few valid labels! Gemma={valid_a}, Qwen={valid_b}.")

np.save("outputs/phase_b/probing/labels_gemma_shared.npy", labels_a_shared)
np.save("outputs/phase_b/probing/labels_qwen_shared.npy", labels_b_shared)

# Find top-K most frequent shared tokens
all_valid = np.concatenate([labels_a_shared[labels_a_shared >= 0],
                            labels_b_shared[labels_b_shared >= 0]])
token_counts = np.bincount(all_valid, minlength=n_shared_classes)
TOP_K = 500
top_shared = np.argsort(token_counts)[-TOP_K:]
print(f"\nTop-{TOP_K} shared tokens cover {token_counts[top_shared].sum()/len(all_valid)*100:.1f}% of valid data")

for idx in top_shared[-20:]:
    gid, qid, s = gemma_qwen_map[idx]
    print(f"  Class {idx}: '{s}' (gemma={gid}, qwen={qid}, count={token_counts[idx]})")

np.save("outputs/phase_b/probing/top_shared_tokens.npy", top_shared)
print("\nSaved shared labels and top tokens.")
