"""
build_vocab_mapping.py
Builds a mapping between Gemma-2B and Qwen-1.5B token vocabularies
by decoding all token IDs to strings, stripping tokenizer-specific prefixes,
and finding exact string matches.
"""
import json
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer

gemma_tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

def normalize_token(token_str):
    s = token_str
    s = s.replace('\u2581', ' ').replace('\u0120', ' ').replace('##', '')
    s = s.strip()
    return s

def build_vocab_map(tok_a, tok_b, name_a="gemma", name_b="qwen"):
    vocab_a_all = defaultdict(list)
    vocab_b_all = defaultdict(list)

    for token_id in range(tok_a.vocab_size):
        try:
            decoded = tok_a.decode([token_id])
            if decoded.startswith('<') and decoded.endswith('>'):
                continue
            if decoded.startswith('<0x'):
                continue
            normalized = normalize_token(decoded)
            if normalized and len(normalized) > 0:
                vocab_a_all[normalized].append(token_id)
        except:
            continue

    for token_id in range(tok_b.vocab_size):
        try:
            decoded = tok_b.decode([token_id])
            if decoded.startswith('<') and decoded.endswith('>'):
                continue
            if decoded.startswith('<0x'):
                continue
            normalized = normalize_token(decoded)
            if normalized and len(normalized) > 0:
                vocab_b_all[normalized].append(token_id)
        except:
            continue

    collisions_a = sum(1 for ids in vocab_a_all.values() if len(ids) > 1)
    collisions_b = sum(1 for ids in vocab_b_all.values() if len(ids) > 1)
    print(f"Collisions in {name_a}: {collisions_a} normalized strings map to multiple token IDs")
    print(f"Collisions in {name_b}: {collisions_b} normalized strings map to multiple token IDs")

    vocab_a = {s: min(ids) for s, ids in vocab_a_all.items()}
    vocab_b = {s: min(ids) for s, ids in vocab_b_all.items()}

    shared_strings = set(vocab_a.keys()) & set(vocab_b.keys())

    mapping = []
    a_all_ids_for_class = {}
    b_all_ids_for_class = {}

    for i, s in enumerate(sorted(shared_strings)):
        mapping.append((vocab_a[s], vocab_b[s], s))
        a_all_ids_for_class[i] = vocab_a_all[s]
        b_all_ids_for_class[i] = vocab_b_all[s]

    print(f"\nVocab A ({name_a}): {len(vocab_a)} unique normalized tokens")
    print(f"Vocab B ({name_b}): {len(vocab_b)} unique normalized tokens")
    print(f"Shared tokens: {len(mapping)} ({len(mapping)/min(len(vocab_a),len(vocab_b))*100:.1f}% of smaller vocab)")

    if len(mapping) == 0:
        raise ValueError(f"No shared tokens between {name_a} and {name_b}!")
    if len(mapping) < 1000:
        print(f"WARNING: Only {len(mapping)} shared tokens — experiment may be underpowered")

    print(f"\nExample shared tokens: {[m[2] for m in mapping[:20]]}")
    return mapping, a_all_ids_for_class, b_all_ids_for_class

print("=" * 60)
print("Gemma-2B <-> Qwen-1.5B")
print("=" * 60)
gemma_qwen_map, a_all_ids, b_all_ids = build_vocab_map(gemma_tok, qwen_tok, "gemma", "qwen")

mapping_dict = {
    "gemma_qwen": [(int(a), int(b), s) for a, b, s in gemma_qwen_map],
    "gemma_all_ids_per_class": {str(k): v for k, v in a_all_ids.items()},
    "qwen_all_ids_per_class": {str(k): v for k, v in b_all_ids.items()},
}
with open("outputs/vocab_mapping.json", "w") as f:
    json.dump(mapping_dict, f, indent=2)

print(f"\nSaved mapping to outputs/vocab_mapping.json")
