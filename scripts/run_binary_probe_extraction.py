#!/usr/bin/env python3
"""Extract activations from 4 models on 3 binary classification datasets.

Datasets:
  1. SST-2 (sentiment): stanfordnlp/sst2
  2. ToxiGen (toxicity): skg/toxigen-data (subset "annotated")
  3. AG News (topic — Sports vs not): fancyzhx/ag_news

Models: Gemma-2B, Qwen-1.5B, Llama-1B, Llama-3B
Layers: max-CKA pairs from earlier experiments.
"""

import time
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

OUTPUT_DIR = Path("outputs/binary_probe_transfer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────
MODELS = [
    {"name": "google/gemma-2-2b",       "alias": "gemma_2b",   "layers": [18]},
    {"name": "Qwen/Qwen2.5-1.5B",       "alias": "qwen_1.5b",  "layers": [23]},
    {"name": "meta-llama/Llama-3.2-1B",  "alias": "llama_1b",   "layers": [15]},
    {"name": "meta-llama/Llama-3.2-3B",  "alias": "llama_3b",   "layers": [16]},
]

N_SAMPLES_PER_TASK = 5000  # 5k per task, 15k total
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
DEVICE = "cuda"


def load_sst2(n: int) -> tuple[list[str], np.ndarray]:
    """SST-2 sentiment: 0=negative, 1=positive."""
    ds = load_dataset("stanfordnlp/sst2", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return ds["sentence"], np.array(ds["label"])


def load_toxigen(n: int) -> tuple[list[str], np.ndarray]:
    """ToxiGen toxicity: 0=benign, 1=toxic."""
    ds = load_dataset("skg/toxigen-data", "annotated", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    texts = ds["text"]
    # toxigen_roberta label: 1=toxic, 0=benign (may vary by version)
    # Use the "toxicity_human" or "toxicity_ai" column if available,
    # otherwise fall back to binarizing the annotation
    if "toxicity_human" in ds.column_names:
        labels = np.array([1 if x >= 3.0 else 0 for x in ds["toxicity_human"]])
    elif "label" in ds.column_names:
        labels = np.array(ds["label"])
    else:
        # Fallback: use toxicity_ai column
        labels = np.array([1 if x >= 0.5 else 0 for x in ds["toxicity_ai"]])
    return texts, labels


def load_agnews_binary(n: int) -> tuple[list[str], np.ndarray]:
    """AG News Sports vs non-Sports: 0=non-sports, 1=sports.
    Original labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech.
    """
    ds = load_dataset("fancyzhx/ag_news", split="train")
    ds = ds.shuffle(seed=42)

    # Balance: take n/2 sports and n/2 non-sports
    sports = [i for i, label in enumerate(ds["label"]) if label == 1]
    non_sports = [i for i, label in enumerate(ds["label"]) if label != 1]

    n_each = n // 2
    selected_idx = sports[:n_each] + non_sports[:n_each]
    np.random.seed(42)
    np.random.shuffle(selected_idx)

    texts = [ds["text"][i] for i in selected_idx]
    labels = np.array([1 if ds["label"][i] == 1 else 0 for i in selected_idx])
    return texts, labels


TASKS = {
    "sst2_sentiment": load_sst2,
    "toxigen_toxicity": load_toxigen,
    "agnews_sports": load_agnews_binary,
}


def extract_model_on_task(
    model, tokenizer, model_cfg: dict,
    texts: list[str], task_name: str,
    layer_prefix: str
) -> dict[int, torch.Tensor]:
    """Extract activations from one model on one task's texts."""
    alias = model_cfg["alias"]
    target_layers = model_cfg["layers"]

    activations = {layer: [] for layer in target_layers}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[layer_idx].append(hidden.detach().float().cpu())
        return hook_fn

    hooks = []
    for layer_idx in target_layers:
        module_name = f"{layer_prefix}.{layer_idx}"
        parts = module_name.split(".")
        mod = model
        for p in parts:
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        hooks.append(mod.register_forward_hook(make_hook(layer_idx)))

    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE),
                      desc=f"[{alias}/{task_name}]", total=n_batches):
            batch_texts = texts[i:i+BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
            ).to(DEVICE)

            model(**inputs)

            lengths = inputs["attention_mask"].sum(dim=1).cpu() - 1
            for layer_idx in target_layers:
                hidden = activations[layer_idx][-1]
                selected = torch.stack([
                    hidden[b, lengths[b].item()] for b in range(hidden.shape[0])
                ])
                activations[layer_idx][-1] = selected

            # Progress every 20 batches
            done = min(i + BATCH_SIZE, len(texts))
            if (i // BATCH_SIZE) % 20 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(texts) - done) / rate if rate > 0 else 0
                pct = done / len(texts) * 100
                print(f"  [{alias}/{task_name}] {pct:.0f}% | {rate:.0f} samp/s | ETA: {eta:.0f}s")

    for h in hooks:
        h.remove()

    result = {}
    for layer_idx in target_layers:
        tensor = torch.cat(activations[layer_idx], dim=0)
        result[layer_idx] = tensor
        print(f"  [{alias}/{task_name}] Layer {layer_idx}: shape={tensor.shape}")

    return result


def detect_layer_prefix(model) -> str:
    """Detect the layer module path for a model."""
    for prefix in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        parts = prefix.split(".")
        mod = model
        try:
            for p in parts:
                mod = getattr(mod, p)
            return prefix
        except AttributeError:
            continue
    raise ValueError("Cannot detect layer prefix")


def main():
    overall_start = time.time()

    # ── Load all 3 datasets ──────────────────────────────────────────────
    print("Loading datasets...")
    task_data = {}
    for task_name, loader_fn in TASKS.items():
        texts, labels = loader_fn(N_SAMPLES_PER_TASK)
        task_data[task_name] = {"texts": texts, "labels": labels}
        pos_frac = np.mean(labels)
        print(f"  {task_name}: {len(texts)} samples, "
              f"{pos_frac:.1%} positive / {1-pos_frac:.1%} negative")
        # Save labels
        np.save(OUTPUT_DIR / f"labels_{task_name}.npy", labels)

    # ── Extract per model ────────────────────────────────────────────────
    for model_idx, model_cfg in enumerate(MODELS):
        alias = model_cfg["alias"]
        print(f"\n{'#'*60}")
        print(f"# MODEL {model_idx+1}/{len(MODELS)}: {model_cfg['name']} ({alias})")
        elapsed = time.time() - overall_start
        print(f"# Elapsed: {elapsed/60:.1f}min")
        if model_idx > 0:
            rate = elapsed / model_idx
            remaining = rate * (len(MODELS) - model_idx)
            print(f"# ETA for remaining models: {remaining/60:.1f}min")
        print(f"{'#'*60}")

        # Load model
        print(f"[{alias}] Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            torch_dtype=torch.float16,
            device_map=DEVICE,
            trust_remote_code=True,
        )
        model.eval()

        d_model = model.config.hidden_size
        print(f"[{alias}] d_model={d_model}, layers={model.config.num_hidden_layers}")

        layer_prefix = detect_layer_prefix(model)

        # Extract on each task
        all_acts = {}
        for task_name, data in task_data.items():
            acts = extract_model_on_task(
                model, tokenizer, model_cfg,
                data["texts"], task_name, layer_prefix
            )
            all_acts[task_name] = acts

        # Save: one file per model, containing all tasks and layers
        save_dict = {}
        for task_name, layer_dict in all_acts.items():
            for layer_idx, tensor in layer_dict.items():
                save_dict[f"{task_name}_L{layer_idx}"] = tensor
        save_path = OUTPUT_DIR / f"acts_{alias}.pt"
        torch.save(save_dict, save_path)
        print(f"[{alias}] Saved to {save_path} ({save_path.stat().st_size/1e6:.1f}MB)")

        # Free GPU
        del model
        torch.cuda.empty_cache()

    total = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"ALL EXTRACTIONS COMPLETE in {total/60:.1f}min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
