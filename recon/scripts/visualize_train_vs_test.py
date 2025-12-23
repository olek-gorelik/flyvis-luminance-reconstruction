#!/usr/bin/env python3
"""Visualize random train/test reconstructions on FlyVis retinal representation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from flyvis.datasets.sintel import MultiTaskSintel
from flyvis.analysis.visualization.plots import quick_hex_scatter


TRAIN_DATA = Path("recon/outputs/activity_lum_train_fullfield.pt")
TEST_DATA = Path("recon/outputs/activity_lum_test_fullfield.pt")
MODEL_PATH = Path("recon/outputs/linear_decoder_fullfield_norm_split.pt")
STATS_PATH = Path("recon/outputs/linear_decoder_fullfield_norm_split_stats.pt")
OUT_PATH = Path("recon/outputs/train_vs_test_reconstruction.pdf")
N_SAMPLES = 10
SEED = 0


def load_linear_decoder(n_features: int, n_hexals: int, model_path: Path) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.nn.Linear(n_features, n_hexals)
    state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict for Linear, got {type(state)}")
    model.load_state_dict(state)
    model.eval()
    return model


def load_dataset(data_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = torch.load(data_path, map_location="cpu")
    X = data["X"].float()
    Y = data["Y"].float()
    if X.ndim != 3 or Y.ndim != 4:
        raise ValueError(f"Unexpected shapes: X {tuple(X.shape)} Y {tuple(Y.shape)}")
    return X, Y.squeeze(2)


def sample_frame_indices(n_samples: int, n_frames: int, n_select: int, seed: int) -> list[tuple[int, int]]:
    total = n_samples * n_frames
    if n_select > total:
        raise ValueError(f"Requested {n_select} frames but only {total} available.")
    gen = torch.Generator()
    gen.manual_seed(seed)
    flat_indices = torch.randperm(total, generator=gen)[:n_select]
    pairs = []
    for idx in flat_indices.tolist():
        pairs.append((idx // n_frames, idx % n_frames))
    return pairs


def main() -> None:
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Stats file not found: {STATS_PATH}")

    X_train, Y_train = load_dataset(TRAIN_DATA)
    X_test, Y_test = load_dataset(TEST_DATA)

    n_features = X_train.shape[-1]
    n_hexals = Y_train.shape[-1]

    stats = torch.load(STATS_PATH, map_location="cpu")
    mean = stats["mean"]
    std = stats["std"]
    eps = stats.get("eps", 1e-6)
    if mean.numel() != n_features or std.numel() != n_features:
        raise ValueError(
            f"Normalization stats mismatch: mean/std length {mean.numel()} vs F={n_features}"
        )

    model = load_linear_decoder(n_features, n_hexals, MODEL_PATH)

    x_train_n = (X_train - mean) / (std + eps)
    x_test_n = (X_test - mean) / (std + eps)

    train_pairs = sample_frame_indices(
        X_train.shape[0], X_train.shape[1], N_SAMPLES, SEED
    )
    test_pairs = sample_frame_indices(
        X_test.shape[0], X_test.shape[1], N_SAMPLES, SEED + 1
    )

    samples = []
    with torch.no_grad():
        for sample_idx, frame_idx in train_pairs:
            x_t = x_train_n[sample_idx, frame_idx]
            y_t = Y_train[sample_idx, frame_idx]
            y_pred = model(x_t.unsqueeze(0)).squeeze(0).cpu()
            samples.append(
                ("train", sample_idx, frame_idx, y_t, y_pred, y_pred - y_t)
            )
        for sample_idx, frame_idx in test_pairs:
            x_t = x_test_n[sample_idx, frame_idx]
            y_t = Y_test[sample_idx, frame_idx]
            y_pred = model(x_t.unsqueeze(0)).squeeze(0).cpu()
            samples.append(
                ("test", sample_idx, frame_idx, y_t, y_pred, y_pred - y_t)
            )

    dataset = MultiTaskSintel(
        tasks=["lum"],
        augment=False,
        random_temporal_crop=False,
        all_frames=False,
        _init_cache=False,
        unittest=True,
    )
    extent = dataset.boxfilter["extent"]
    expected_hexals = 1 + 3 * extent * (extent + 1)
    if expected_hexals != n_hexals:
        raise ValueError(
            f"Unexpected hexal count: expected {expected_hexals}, got {n_hexals}"
        )

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

    for row, (split, sample_idx, frame_idx, y_true, y_pred, resid) in enumerate(samples):
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
        axes[row, 0].set_ylabel(f"{split} s{sample_idx} t{frame_idx}", fontsize=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    print(f"saved visualization to {OUT_PATH}")


if __name__ == "__main__":
    main()
