"""Extract residual stream activations from transformer models at specified layers.

Strategy:
- Use forward hooks to capture residual stream outputs at target layers
- Process prompts in batches to avoid OOM
- Extract at LAST token position by default (where next-token prediction happens)
- Store as (n_prompts, d_model) tensors per layer
"""

import math
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from rich.console import Console

from .config import Config, ModelConfig, ExtractionConfig

console = Console()


def get_layer_indices(model, layer_fractions: list[float]) -> list[int]:
    """Convert fractional layer depths to actual layer indices.

    For a model with N layers, fraction f maps to layer floor(f * (N-1)).
    This gives us: 0.0 -> layer 0, 0.5 -> middle layer, 1.0 -> last layer.
    """
    # Detect number of layers from model config
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None:
        raise ValueError("Cannot determine num_hidden_layers from model config")

    indices = []
    for frac in layer_fractions:
        idx = int(math.floor(frac * (n_layers - 1)))
        idx = max(0, min(idx, n_layers - 1))
        if idx not in indices:
            indices.append(idx)

    return sorted(indices)


def get_residual_stream_hook_points(model) -> list[str]:
    """Identify the module names for residual stream outputs at each layer.

    Different architectures name their layers differently:
    - Gemma/Llama: model.layers.{i}
    - Qwen: model.layers.{i}
    - GPT-2: transformer.h.{i}
    - Pythia: gpt_neox.layers.{i}

    We hook the output of each transformer block (post-residual connection).
    """
    # Try common patterns
    for pattern_prefix in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        parts = pattern_prefix.split(".")
        module = model
        try:
            for part in parts:
                module = getattr(module, part)
            # If we get here, this pattern exists
            n_layers = len(module)
            return [f"{pattern_prefix}.{i}" for i in range(n_layers)]
        except AttributeError:
            continue

    raise ValueError(
        f"Could not identify layer modules for {type(model).__name__}. "
        "Supported architectures: Gemma, Llama, Qwen, GPT-2, Pythia."
    )


def _get_module_by_name(model, name: str):
    """Get a submodule by dot-separated name."""
    parts = name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


class ActivationExtractor:
    """Extract residual stream activations from a model at specified layers."""

    def __init__(
        self,
        model_config: ModelConfig,
        extraction_config: ExtractionConfig,
        device: str = "cuda",
    ):
        self.model_config = model_config
        self.extraction_config = extraction_config
        self.device = device
        self.model = None
        self.tokenizer = None
        self._hooks = []
        self._activations = {}

    def load_model(self):
        """Load model and tokenizer."""
        console.print(f"[bold blue]Loading {self.model_config.name}...[/bold blue]")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.name,
            torch_dtype=torch.float16,  # Load in fp16 to save VRAM
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Auto-detect d_model
        d_model = getattr(self.model.config, "hidden_size", None)
        if d_model is None:
            raise ValueError("Cannot determine hidden_size from model config")
        self.model_config.d_model = d_model

        console.print(
            f"  Loaded: {self.model_config.name} | "
            f"d_model={d_model} | "
            f"n_layers={self.model.config.num_hidden_layers}"
        )

    def _register_hooks(self, layer_indices: list[int]):
        """Register forward hooks on target layers to capture activations."""
        self._clear_hooks()
        self._activations = {idx: [] for idx in layer_indices}

        hook_points = get_residual_stream_hook_points(self.model)

        for layer_idx in layer_indices:
            module = _get_module_by_name(self.model, hook_points[layer_idx])

            def make_hook(idx):
                def hook_fn(module, input, output):
                    # output is typically (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    # hidden: (batch, seq_len, d_model)
                    self._activations[idx].append(hidden.detach().float().cpu())
                return hook_fn

            handle = module.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

    def _clear_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def extract(
        self,
        texts: list[str],
        layer_fractions: Optional[list[float]] = None,
    ) -> dict[int, torch.Tensor]:
        """Extract activations for given texts at specified layer fractions.

        Returns:
            Dict mapping layer_index -> tensor of shape (n_texts, d_model)
        """
        if self.model is None:
            self.load_model()

        if layer_fractions is None:
            layer_fractions = self.extraction_config.layer_fractions

        layer_indices = get_layer_indices(self.model, layer_fractions)
        console.print(
            f"  Extracting at layers: {layer_indices} "
            f"(fractions: {layer_fractions})"
        )

        self._register_hooks(layer_indices)

        batch_size = self.extraction_config.batch_size
        token_position = self.extraction_config.token_position

        try:
            with torch.no_grad():
                for i in tqdm(
                    range(0, len(texts), batch_size),
                    desc=f"Extracting {self.model_config.alias}",
                ):
                    batch_texts = texts[i : i + batch_size]

                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.extraction_config.max_seq_len,
                    ).to(self.device)

                    # Forward pass triggers hooks
                    self.model(**inputs)

                    # Extract at desired token position
                    for idx in layer_indices:
                        hidden = self._activations[idx][-1]  # (batch, seq, d_model)

                        if token_position == "last":
                            # Get last non-padding token for each sequence
                            lengths = inputs["attention_mask"].sum(dim=1).cpu() - 1
                            selected = torch.stack([
                                hidden[b, lengths[b].item()]
                                for b in range(hidden.shape[0])
                            ])
                        elif token_position == "mean":
                            mask = inputs["attention_mask"].unsqueeze(-1).cpu().float()
                            selected = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                        elif isinstance(token_position, int):
                            selected = hidden[:, token_position]
                        else:
                            raise ValueError(f"Unknown token_position: {token_position}")

                        # Replace raw batch tensor with extracted positions
                        self._activations[idx][-1] = selected

        finally:
            self._clear_hooks()

        # Concatenate all batches: (n_texts, d_model)
        result = {}
        for idx in layer_indices:
            result[idx] = torch.cat(self._activations[idx], dim=0)
            console.print(
                f"  Layer {idx}: shape={result[idx].shape}, "
                f"mean={result[idx].mean():.4f}, std={result[idx].std():.4f}"
            )

        self._activations = {}
        return result

    def unload(self):
        """Free GPU memory."""
        self._clear_hooks()
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_prompts(config: ExtractionConfig) -> list[str]:
    """Load prompt texts from the configured dataset."""
    console.print(f"[bold]Loading dataset: {config.dataset}[/bold]")

    ds = load_dataset(config.dataset, split="train")

    # pile-10k has a "text" column
    if "text" in ds.column_names:
        texts = ds["text"]
    elif "content" in ds.column_names:
        texts = ds["content"]
    else:
        raise ValueError(f"Cannot find text column in dataset. Columns: {ds.column_names}")

    texts = texts[: config.n_prompts]
    console.print(f"  Loaded {len(texts)} prompts")
    return texts


def save_activations(
    activations: dict[int, torch.Tensor],
    model_alias: str,
    output_dir: Path,
):
    """Save extracted activations to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"activations_{model_alias}.pt"
    torch.save(activations, path)
    console.print(f"  Saved activations to {path}")


def load_activations(model_alias: str, output_dir: Path) -> dict[int, torch.Tensor]:
    """Load previously saved activations."""
    path = output_dir / f"activations_{model_alias}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)
