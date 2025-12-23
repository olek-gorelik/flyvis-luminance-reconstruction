#!/usr/bin/env python3
"""Create a fixed train/test split for MultiTaskSintel sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyvis.datasets.sintel import MultiTaskSintel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed train/test split for MultiTaskSintel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (defaults to recon/splits/sintel_split_seed{seed}.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = MultiTaskSintel(
        tasks=["lum"],
        augment=False,
        random_temporal_crop=False,
        all_frames=False,
        vertical_splits=3,
        n_frames=19,
        _init_cache=False,
    )

    n_sequences = len(dataset)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_sequences)
    n_train = int(0.8 * n_sequences)
    train_indices = perm[:n_train].tolist()
    test_indices = perm[n_train:].tolist()

    output_path = (
        Path(args.output)
        if args.output is not None
        else Path(f"recon/splits/sintel_split_seed{args.seed}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"train_indices": train_indices, "test_indices": test_indices},
            handle,
            indent=2,
        )

    print(f"train sequences: {len(train_indices)}")
    print(f"test sequences: {len(test_indices)}")
    print(f"saved split to {output_path}")


if __name__ == "__main__":
    main()
