"""Test whether mapped activations from Model A are readable by Model B's oracle.

The activation oracle is a LoRA adapter trained on a base model that can
"read" the model's internal representations and answer questions about them.

The key experiment:
1. Extract activations from Model A on test prompts
2. Apply the learned alignment mapping (Procrustes or linear) to transform them
3. Inject the mapped activations into Model B at the corresponding layer
4. Run Model B's oracle on the injected activations
5. Compare oracle outputs to ground truth

If the oracle produces meaningful answers from mapped activations,
this demonstrates cross-model activation transfer.

NOTE: The oracle adapter and injection mechanism are architecture-specific.
This module provides the framework — you'll need to adapt the injection
method based on the specific oracle implementation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.table import Table

from .config import Config
from .procrustes_alignment import AlignmentResult, apply_mapping
from .activation_extraction import (
    ActivationExtractor,
    get_layer_indices,
    get_residual_stream_hook_points,
    _get_module_by_name,
)

console = Console()


@dataclass
class OracleTestResult:
    """Result of testing oracle on mapped activations."""
    prompt: str
    source_model: str
    target_model: str
    layer_source: int
    layer_target: int
    alignment_method: str
    oracle_output: str
    # Baseline: oracle output on target model's own activations
    baseline_output: Optional[str] = None
    # Cosine similarity between mapped and native activations
    activation_cosine_sim: Optional[float] = None


class OracleTransferTester:
    """Test activation oracle transfer between models."""

    def __init__(self, config: Config):
        self.config = config
        self.target_model = None
        self.target_tokenizer = None
        self.oracle_model = None

    def load_oracle(self):
        """Load the target model with oracle LoRA adapter."""
        cfg = self.config
        oracle_cfg = cfg.oracle

        if oracle_cfg.adapter_id is None:
            raise ValueError(
                "oracle.adapter_id not set in config. "
                "Set it to the HuggingFace path of the oracle LoRA adapter."
            )

        # Determine which model the oracle is built on
        if oracle_cfg.base_model == "model_b":
            base_name = cfg.model_b.name
        else:
            base_name = cfg.model_a.name

        console.print(f"[bold blue]Loading oracle base model: {base_name}[/bold blue]")

        self.target_tokenizer = AutoTokenizer.from_pretrained(
            base_name, trust_remote_code=True
        )
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        self.target_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            device_map=cfg.device,
            trust_remote_code=True,
        )

        console.print(f"[bold blue]Loading oracle adapter: {oracle_cfg.adapter_id}[/bold blue]")
        self.oracle_model = PeftModel.from_pretrained(
            self.target_model, oracle_cfg.adapter_id
        )
        self.oracle_model.eval()

    def _inject_activation_hook(
        self,
        layer_idx: int,
        activation: torch.Tensor,
        token_position: int = -1,
    ):
        """Create a forward hook that replaces a layer's output with injected activation.

        Args:
            layer_idx: Which layer to inject at
            activation: The activation to inject, shape (d_model,)
            token_position: Which token position to inject at (-1 = last)
        """
        hook_points = get_residual_stream_hook_points(self.oracle_model)
        module = _get_module_by_name(self.oracle_model, hook_points[layer_idx])

        def hook_fn(mod, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Inject at specified position
            pos = token_position if token_position >= 0 else hidden.shape[1] + token_position
            hidden[:, pos, :] = activation.to(hidden.device, hidden.dtype)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return module.register_forward_hook(hook_fn)

    def test_single(
        self,
        prompt: str,
        source_activation: np.ndarray,
        alignment: AlignmentResult,
        layer_source: int,
        layer_target: int,
        max_new_tokens: int = 50,
    ) -> OracleTestResult:
        """Test oracle on a single mapped activation.

        Args:
            prompt: The test prompt
            source_activation: Activation from Model A, shape (d_source,)
            alignment: The learned alignment mapping
            layer_source: Layer index in source model
            layer_target: Layer index in target model
            max_new_tokens: How many tokens to generate

        Returns:
            OracleTestResult with oracle's output
        """
        if self.oracle_model is None:
            self.load_oracle()

        # Apply alignment mapping
        mapped = apply_mapping(
            source_activation.reshape(1, -1), alignment
        ).squeeze(0)
        mapped_tensor = torch.from_numpy(mapped)

        # Inject mapped activation into target model and generate
        hook = self._inject_activation_hook(layer_target, mapped_tensor)

        try:
            inputs = self.target_tokenizer(prompt, return_tensors="pt").to(
                self.config.device
            )
            with torch.no_grad():
                outputs = self.oracle_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )

            generated = self.target_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
        finally:
            hook.remove()

        return OracleTestResult(
            prompt=prompt,
            source_model=self.config.model_a.alias,
            target_model=self.config.model_b.alias,
            layer_source=layer_source,
            layer_target=layer_target,
            alignment_method=alignment.method,
            oracle_output=generated.strip(),
        )

    def test_baseline(
        self,
        prompt: str,
        layer_target: int,
        max_new_tokens: int = 50,
    ) -> str:
        """Get oracle output on the target model's own activations (baseline).

        This runs the oracle normally without injection — the baseline
        to compare mapped-activation results against.
        """
        if self.oracle_model is None:
            self.load_oracle()

        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(
            self.config.device
        )
        with torch.no_grad():
            outputs = self.oracle_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = self.target_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated.strip()

    def run_transfer_experiment(
        self,
        source_acts: dict[int, torch.Tensor],
        alignments: dict[str, AlignmentResult],
        layer_pairs: list[tuple[int, int, float]],
        test_prompts: Optional[list[str]] = None,
    ) -> list[OracleTestResult]:
        """Run the full oracle transfer experiment.

        Args:
            source_acts: Activations from source model (from extraction)
            alignments: Dict of method -> AlignmentResult
            layer_pairs: List of (source_layer, target_layer, cka_score)
            test_prompts: Prompts to test with

        Returns:
            List of OracleTestResults
        """
        if test_prompts is None:
            test_prompts = self.config.oracle.test_prompts

        results = []

        for prompt_idx, prompt in enumerate(test_prompts):
            console.print(f"\n[bold]Prompt {prompt_idx + 1}: {prompt!r}[/bold]")

            for layer_s, layer_t, cka_score in layer_pairs:
                console.print(
                    f"  Layer pair: {layer_s} -> {layer_t} (CKA={cka_score:.3f})"
                )

                # Get source activation for this prompt at this layer
                if layer_s not in source_acts:
                    console.print(f"  [yellow]Layer {layer_s} not in source activations, skipping[/yellow]")
                    continue

                source_act = source_acts[layer_s][prompt_idx].numpy()

                for method_name, alignment in alignments.items():
                    result = self.test_single(
                        prompt=prompt,
                        source_activation=source_act,
                        alignment=alignment,
                        layer_source=layer_s,
                        layer_target=layer_t,
                    )

                    # Get baseline
                    result.baseline_output = self.test_baseline(prompt, layer_t)

                    console.print(f"    [{method_name}] Oracle says: {result.oracle_output!r}")
                    console.print(f"    [baseline] Oracle says: {result.baseline_output!r}")

                    results.append(result)

        return results

    def print_results(self, results: list[OracleTestResult]):
        """Print a summary table of transfer experiment results."""
        table = Table(title="Oracle Transfer Results")
        table.add_column("Prompt", max_width=30)
        table.add_column("Layers (S→T)")
        table.add_column("Method")
        table.add_column("Oracle Output", max_width=40)
        table.add_column("Baseline", max_width=40)

        for r in results:
            table.add_row(
                r.prompt[:30],
                f"{r.layer_source}→{r.layer_target}",
                r.alignment_method,
                r.oracle_output[:40],
                (r.baseline_output or "N/A")[:40],
            )

        console.print(table)

    def unload(self):
        """Free GPU memory."""
        if self.oracle_model is not None:
            del self.oracle_model
            self.oracle_model = None
        if self.target_model is not None:
            del self.target_model
            self.target_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
