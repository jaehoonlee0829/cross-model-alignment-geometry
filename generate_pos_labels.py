"""
generate_pos_labels.py
Generates POS tag labels for all prompts, aligned to each model's token positions.
Uses spaCy for POS tagging, batched for speed.

MUST RUN ON GPU MACHINE.
"""
import spacy
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
import torch
from datasets import load_dataset
import time

assert torch.cuda.is_available(), "Must run on GPU machine!"

nlp = spacy.load("en_core_web_sm")

POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X"
]
POS_TO_ID = {pos: i for i, pos in enumerate(POS_TAGS)}
N_POS_CLASSES = len(POS_TAGS)
print(f"POS classes: {N_POS_CLASSES}", flush=True)

# Load same texts used in the experiment
print("Loading dataset NeelNanda/pile-10k...", flush=True)
ds = load_dataset("NeelNanda/pile-10k", split="train")
texts = list(ds["text"][:10000])
print(f"Loaded {len(texts)} prompts", flush=True)

# Pre-compute spaCy docs in batch (MUCH faster than one-by-one)
print("Running spaCy POS tagging on all texts (batched)...", flush=True)
t0 = time.time()
# Truncate texts for spaCy (only need first ~200 chars since tokenizer max_seq_len=128)
truncated = [t[:500] for t in texts]
docs = list(nlp.pipe(truncated, batch_size=256, n_process=1))
print(f"  spaCy done in {time.time()-t0:.1f}s", flush=True)

def get_pos_for_token_position(doc, text, tokenizer, max_seq_len=128):
    """
    For a given text + pre-computed spaCy doc, determine the POS tag
    of the NEXT word after the last tokenized position.
    """
    encoding = tokenizer(text, max_length=max_seq_len, truncation=True,
                         return_offsets_mapping=True)

    if 'offset_mapping' not in encoding or len(encoding['offset_mapping']) < 2:
        return -1

    offsets = encoding['offset_mapping']
    last_token_end = 0
    for start, end in reversed(offsets):
        if end > 0:
            last_token_end = end
            break

    if last_token_end == 0:
        return -1

    for token in doc:
        if token.idx >= last_token_end:
            return POS_TO_ID.get(token.pos_, -1)

    return -1

# Generate labels for each model's tokenizer
model_configs = [
    ("gemma", "google/gemma-2-2b"),
    ("qwen", "Qwen/Qwen2.5-1.5B"),
    ("llama-1b", "meta-llama/Llama-3.2-1B"),
    ("llama-3b", "meta-llama/Llama-3.2-3B"),
]

for model_alias, tok_name in model_configs:
    print(f"\nProcessing {model_alias} ({tok_name})...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    print(f"  Tokenizer loaded in {time.time()-t0:.1f}s", flush=True)

    pos_labels = np.full(len(texts), -1, dtype=np.int32)
    for i in range(len(texts)):
        pos_labels[i] = get_pos_for_token_position(docs[i], texts[i], tokenizer)
        if i % 2000 == 0:
            print(f"  {i}/{len(texts)}", flush=True)

    valid = (pos_labels >= 0).sum()
    print(f"  Valid: {valid}/{len(pos_labels)} ({valid/len(pos_labels)*100:.1f}%)", flush=True)
    if valid < 500:
        raise ValueError(f"Only {valid} valid POS labels for {model_alias}!")

    for pos, pid in POS_TO_ID.items():
        count = (pos_labels == pid).sum()
        if count > 0:
            print(f"    {pos}: {count} ({count/valid*100:.1f}%)", flush=True)

    out_path = Path(f"outputs/pos_labels_{model_alias}.npy")
    np.save(out_path, pos_labels)
    print(f"  Saved to {out_path}", flush=True)

print("\nDone generating POS labels for all models.", flush=True)
