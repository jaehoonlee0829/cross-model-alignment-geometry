#!/usr/bin/env python3
"""Learn alignment mapping and test oracle transfer.

Usage:
    # Just learn alignment (no oracle test)
    python scripts/run_alignment.py --config configs/default.yaml --alignment-only

    # Full pipeline: alignment + oracle transfer
    python scripts/run_alignment.py --config configs/default.yaml

Requires: run_extraction.py and run_cka.py have been run first.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.activation_extraction import load_activations
from src.procrustes_alignment import learn_alignment, save_alignment
from src.oracle_transfer_test import OracleTransferTester


def main():
    parser = argparse.ArgumentParser(description="Alignment and oracle transfer")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--alignment-only", action="store_true", help="Skip oracle test")
    parser.add_argument("--layer-a", type=int, default=None, help="Specific source layer")
    parser.add_argument("--layer-b", type=int, default=None, help="Specific target layer")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    torch.manual_seed(config.seed)

    act_dir = config.output_dir / "activations"
    cka_dir = config.output_dir / "cka"
    align_dir = config.output_dir / "alignment"
    align_dir.mkdir(parents=True, exist_ok=True)

    # Load activations
    acts_a = load_activations(config.model_a.alias, act_dir)
    acts_b = load_activations(config.model_b.alias, act_dir)

    # Determine which layer pairs to align
    if args.layer_a is not None and args.layer_b is not None:
        layer_pairs = [(args.layer_a, args.layer_b, 0.0)]
    else:
        # Load best pairs from CKA analysis
        pairs_data = np.load(cka_dir / "best_layer_pairs.npz")
        raw_pairs = pairs_data["pairs"]
        layer_pairs = [
            (int(row[0]), int(row[1]), float(row[2]))
            for row in raw_pairs
        ]

    print(f"Will align {len(layer_pairs)} layer pairs:")
    for la, lb, score in layer_pairs:
        print(f"  Layer {la} -> {lb} (CKA={score:.3f})")

    # Learn alignment for each pair
    all_alignments = {}
    for la, lb, score in layer_pairs:
        print(f"\n{'='*60}")
        print(f"Aligning: Layer {la} ({config.model_a.alias}) -> Layer {lb} ({config.model_b.alias})")
        print(f"CKA score: {score:.4f}")
        print(f"{'='*60}")

        X = acts_a[la].numpy()
        Y = acts_b[lb].numpy()

        alignments = learn_alignment(
            X, Y,
            method=config.alignment.method,
            train_fraction=config.alignment.train_fraction,
            regularization=config.alignment.regularization,
            seed=config.seed,
        )

        # Save alignments
        for method_name, result in alignments.items():
            key = f"layer_{la}_to_{lb}_{method_name}"
            all_alignments[key] = result
            save_alignment(result, align_dir / f"{key}.npz")

    if args.alignment_only:
        print("\n--alignment-only flag set, skipping oracle transfer test.")
        print(f"Alignments saved to {align_dir}")
        return

    # Oracle transfer test
    if config.oracle.adapter_id is None:
        print("\nNo oracle adapter_id configured. Set oracle.adapter_id in config to run transfer test.")
        print("Alignments saved — you can run oracle transfer later.")
        return

    print("\n" + "=" * 60)
    print("ORACLE TRANSFER TEST")
    print("=" * 60)

    tester = OracleTransferTester(config)

    # For each best layer pair, test with the first alignment method available
    for la, lb, score in layer_pairs[:3]:  # Test top 3 pairs
        key_prefix = f"layer_{la}_to_{lb}"
        pair_alignments = {
            k.replace(f"{key_prefix}_", ""): v
            for k, v in all_alignments.items()
            if k.startswith(key_prefix)
        }

        if pair_alignments:
            results = tester.run_transfer_experiment(
                source_acts=acts_a,
                alignments=pair_alignments,
                layer_pairs=[(la, lb, score)],
            )
            tester.print_results(results)

    tester.unload()
    print(f"\nAll results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
