#!/usr/bin/env python3
"""Visualize ridge-regularized reconstructions on FlyVis retinal representation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from flyvis.analysis.visualization.plots import quick_hex_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize train vs test reconstructions for ridge-regularized models."
    )
    parser.add_argument("--model", required=True, help="Path to ridge model file.")
    parser.add_argument(
        "--xnorm-stats",
        required=True,
        help="Path to normalization stats (mean/std/eps).",
    )
    parser.add_argument("--train-data", required=True, help="Path to train dataset.")
    parser.add_argument("--test-data", required=True, help="Path to test dataset.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of train and test samples to visualize.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def load_dataset(data_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = torch.load(data_path, map_location="cpu")
    X = data["X"].float()
    Y = data["Y"].float()
    if X.ndim != 3 or Y.ndim != 4:
        raise ValueError(f"Unexpected shapes: X {tuple(X.shape)} Y {tuple(Y.shape)}")
    return X, Y.squeeze(2)


def sample_sequence_frames(
    n_sequences: int, n_frames: int, n_select: int, seed: int
) -> list[tuple[int, int]]:
    if n_select > n_sequences:
        raise ValueError(
            f"Requested {n_select} sequences but only {n_sequences} available."
        )
    gen = torch.Generator()
    gen.manual_seed(seed)
    seq_indices = torch.randperm(n_sequences, generator=gen)[:n_select]
    frame_indices = torch.randint(0, n_frames, (n_select,), generator=gen)
    return list(zip(seq_indices.tolist(), frame_indices.tolist()))


def load_linear_decoder(
    n_features: int, n_hexals: int, model_path: Path
) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        model = torch.nn.Linear(n_features, n_hexals)
        model.load_state_dict(obj)
    else:
        raise TypeError(f"Unsupported model object: {type(obj)}")
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    stats_path = Path(args.xnorm_stats)
    train_path = Path(args.train_data)
    test_path = Path(args.test_data)

    print("RIDGE VISUALIZATION MODE")
    print(f"model path: {model_path}")
    print(f"stats path: {stats_path}")
    print(f"train data: {train_path}")
    print(f"test data: {test_path}")

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"]
    std = stats["std"]
    eps = stats.get("eps", 1e-6)

    X_train, Y_train = load_dataset(train_path)
    X_test, Y_test = load_dataset(test_path)

    n_features = X_train.shape[-1]
    n_hexals = Y_train.shape[-1]
    if mean.numel() != n_features or std.numel() != n_features:
        raise ValueError(
            f"Normalization stats mismatch: mean/std length {mean.numel()} vs F={n_features}"
        )

    model = load_linear_decoder(n_features, n_hexals, model_path)

    train_pairs = sample_sequence_frames(
        X_train.shape[0], X_train.shape[1], args.n_samples, args.seed
    )
    test_pairs = sample_sequence_frames(
        X_test.shape[0], X_test.shape[1], args.n_samples, args.seed + 1
    )

    samples = []
    with torch.no_grad():
        for seq_idx, frame_idx in train_pairs:
            x_t = X_train[seq_idx, frame_idx]
            y_t = Y_train[seq_idx, frame_idx]
            x_t = (x_t - mean) / (std + eps)
            y_pred = model(x_t.unsqueeze(0)).squeeze(0).cpu()
            samples.append(("train", seq_idx, frame_idx, y_t, y_pred, y_pred - y_t))
        for seq_idx, frame_idx in test_pairs:
            x_t = X_test[seq_idx, frame_idx]
            y_t = Y_test[seq_idx, frame_idx]
            x_t = (x_t - mean) / (std + eps)
            y_pred = model(x_t.unsqueeze(0)).squeeze(0).cpu()
            samples.append(("test", seq_idx, frame_idx, y_t, y_pred, y_pred - y_t))

    gt_vals = torch.stack([s[3] for s in samples])
    pred_vals = torch.stack([s[4] for s in samples])
    res_vals = torch.stack([s[5] for s in samples])

    gt_min, gt_max = float(gt_vals.min()), float(gt_vals.max())
    pred_min, pred_max = float(pred_vals.min()), float(pred_vals.max())
    res_abs = float(res_vals.abs().max())

    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 1.6 * n_rows), squeeze=False)

    axes[0, 0].set_title("True")
    axes[0, 1].set_title("Pred")
    axes[0, 2].set_title("Residual")

    for row, (split, seq_idx, frame_idx, y_true, y_pred, resid) in enumerate(samples):
        quick_hex_scatter(
            y_true,
            fig=fig,
            ax=axes[row, 0],
            cbar=False,
            vmin=gt_min,
            vmax=gt_max,
            cmap=plt.get_cmap("gray"),
            edgecolor="none",
        )
        quick_hex_scatter(
            y_pred,
            fig=fig,
            ax=axes[row, 1],
            cbar=False,
            vmin=pred_min,
            vmax=pred_max,
            cmap=plt.get_cmap("gray"),
            edgecolor="none",
        )
        quick_hex_scatter(
            resid,
            fig=fig,
            ax=axes[row, 2],
            cbar=False,
            vmin=-res_abs,
            vmax=res_abs,
            cmap=plt.get_cmap("seismic"),
            edgecolor="none",
        )
        axes[row, 0].set_ylabel(f"{split} s{seq_idx} t{frame_idx}", fontsize=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = Path("recon/outputs/train_vs_test_reconstruction_ridge.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved ridge visualization to {out_path}")


if __name__ == "__main__":
    main()
