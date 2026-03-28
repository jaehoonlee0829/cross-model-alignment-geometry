#!/usr/bin/env python3
"""Extract activations from both models on shared prompts.

Usage:
    python scripts/run_extraction.py --config configs/default.yaml
    python scripts/run_extraction.py --config configs/default.yaml --model-a-only
    python scripts/run_extraction.py --config configs/default.yaml --model-b-only
"""

import argparse
from pathlib import Path

import torch

from src.config import Config
from src.activation_extraction import (
    ActivationExtractor,
    load_prompts,
    save_activations,
)


def main():
    parser = argparse.ArgumentParser(description="Extract activations from models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-a-only", action="store_true")
    parser.add_argument("--model-b-only", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    torch.manual_seed(config.seed)

    output_dir = config.output_dir / "activations"

    # Load shared prompts
    prompts = load_prompts(config.extraction)

    if not args.model_b_only:
        # Extract Model A
        extractor_a = ActivationExtractor(
            config.model_a, config.extraction, config.device
        )
        acts_a = extractor_a.extract(prompts)
        save_activations(acts_a, config.model_a.alias, output_dir)
        extractor_a.unload()

    if not args.model_a_only:
        # Extract Model B
        extractor_b = ActivationExtractor(
            config.model_b, config.extraction, config.device
        )
        acts_b = extractor_b.extract(prompts)
        save_activations(acts_b, config.model_b.alias, output_dir)
        extractor_b.unload()

    print("\nDone! Activations saved to:", output_dir)


if __name__ == "__main__":
    main()
