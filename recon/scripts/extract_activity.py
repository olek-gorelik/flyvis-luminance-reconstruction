#!/usr/bin/env python3
"""Extract aligned (activity, lum) pairs from FlyVis for reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flyvis import NetworkView
from flyvis.datasets.sintel import MultiTaskSintel


EARLY_RECEPTORS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
LAMINA_TYPES = ["L1", "L2", "L3", "L4", "L5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract (activity, lum) pairs from a pretrained FlyVis network."
    )
    parser.add_argument(
        "--network",
        default="flow/0000/000",
        help="Network path or name under flyvis results_dir.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint identifier to recover (e.g., 'best' or an index).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Number of samples to extract.",
    )
    parser.add_argument(
        "--include-lamina",
        action="store_true",
        help="Include L1-L5 alongside R1-R8.",
    )
    parser.add_argument(
        "--vertical-splits",
        type=int,
        default=3,
        help="Vertical splits used in MultiTaskSintel.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=19,
        help="Number of frames to sample per sequence.",
    )
    parser.add_argument(
        "--output-dir",
        default="recon/outputs",
        help="Directory to write extracted tensors.",
    )
    parser.add_argument(
        "--output-name",
        default="activity_lum_pairs.pt",
        help="Output filename for torch.save.",
    )
    return parser.parse_args()


def _as_feature_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(-1)
    if x.ndim == 1:
        return x.view(1, -1, 1)
    return x


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    network_view = NetworkView(args.network)
    network = network_view.init_network(checkpoint=args.checkpoint)

    dataset = MultiTaskSintel(
        tasks=["lum"],
        augment=False,
        random_temporal_crop=False,
        all_frames=False,
        vertical_splits=args.vertical_splits,
        n_frames=args.n_frames,
        _init_cache=True,
    )

    cell_types = list(EARLY_RECEPTORS)
    if args.include_lamina:
        cell_types.extend(LAMINA_TYPES)

    X_list = []
    Y_list = []

    n_samples = min(args.max_samples, len(dataset))
    for idx in range(n_samples):
        sample = dataset[idx]
        lum = sample["lum"]
        if lum.ndim != 3:
            raise ValueError(f"Expected lum with shape (frames, 1, hexals), got {lum.shape}")

        with torch.no_grad():
            activity = network.simulate(
                movie_input=lum[None, ...],
                dt=dataset.dt,
                as_layer_activity=True,
            )

        feats = []
        for cell_type in cell_types:
            act = getattr(activity.central, cell_type)
            feats.append(_as_feature_dim(act))

        X = torch.cat(feats, dim=-1).cpu()
        Y = lum.unsqueeze(0).cpu()

        print(f"sample {idx}: X {tuple(X.shape)} | Y {tuple(Y.shape)}")

        X_list.append(X)
        Y_list.append(Y)

    X_all = torch.cat(X_list, dim=0)
    Y_all = torch.cat(Y_list, dim=0)

    print(f"final: X {tuple(X_all.shape)} | Y {tuple(Y_all.shape)}")

    torch.save(
        {
            "X": X_all,
            "Y": Y_all,
            "cell_types": cell_types,
            "indices": list(range(n_samples)),
            "dt": dataset.dt,
        },
        output_dir / args.output_name,
    )


if __name__ == "__main__":
    main()
