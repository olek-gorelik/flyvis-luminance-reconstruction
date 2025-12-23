#!/usr/bin/env python3
"""Visualize central-only reconstructions on FlyVis retinal representation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from flyvis.connectome import ConnectomeFromAvgFilters
from flyvis.datasets.sintel import MultiTaskSintel
from flyvis.analysis.visualization.plots import quick_hex_scatter


DATA_PATH = Path("recon/outputs/activity_lum_pairs.pt")
MODEL_PATH = Path("recon/outputs/linear_decoder.pt")
SAMPLE_INDEX = 0
FRAME_INDICES = [0, 10, 20, 30]


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


def compute_feature_slices(cell_types: list[str], n_features: int) -> dict[str, slice]:
    connectome = ConnectomeFromAvgFilters()
    sizes = []
    for cell_type in cell_types:
        if cell_type not in connectome.nodes.layer_index:
            raise KeyError(f"Cell type {cell_type} not found in connectome layer_index")
        sizes.append(len(connectome.nodes.layer_index[cell_type]))
    if sum(sizes) != n_features:
        raise ValueError(
            f"Feature size mismatch: sum(cell_type sizes)={sum(sizes)} vs F={n_features}"
        )
    slices = {}
    offset = 0
    for cell_type, size in zip(cell_types, sizes):
        slices[cell_type] = slice(offset, offset + size)
        offset += size
    return slices


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    data = torch.load(DATA_PATH, map_location="cpu")
    X = data["X"].float()
    Y = data["Y"].float()
    cell_types = data.get("cell_types", None)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {tuple(X.shape)}")
    if Y.ndim != 4:
        raise ValueError(f"Expected Y with shape (N, T, 1, H), got {tuple(Y.shape)}")

    n_samples, n_frames, n_features = X.shape
    n_hexals = Y.shape[-1]

    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= n_samples:
        raise IndexError(f"SAMPLE_INDEX={SAMPLE_INDEX} out of range for N={n_samples}")

    frames = [f for f in FRAME_INDICES if 0 <= f < n_frames]
    if not frames:
        raise ValueError(f"No valid frames in {FRAME_INDICES} for T={n_frames}")

    model = load_linear_decoder(n_features, n_hexals, MODEL_PATH)

    x_sample = X[SAMPLE_INDEX]
    y_sample = Y[SAMPLE_INDEX].squeeze(1)

    with torch.no_grad():
        y_hat = model(x_sample).cpu()

    res = y_hat - y_sample

    feature_slices = None
    if isinstance(cell_types, list):
        try:
            feature_slices = compute_feature_slices(cell_types, n_features)
        except Exception as exc:
            print(f"feature map slicing disabled: {exc}")

    feature_keys = []
    if feature_slices is not None:
        for key in ("R1", "R6"):
            if key in feature_slices:
                feature_keys.append(key)

    # Instantiate dataset to access the same BoxEye configuration used for rendering.
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

    # Precompute scales per row for consistent color limits.
    gt_vals = torch.stack([y_sample[t] for t in frames])
    pred_vals = torch.stack([y_hat[t] for t in frames])
    res_vals = torch.stack([res[t] for t in frames])

    gt_min, gt_max = float(gt_vals.min()), float(gt_vals.max())
    pred_min, pred_max = float(pred_vals.min()), float(pred_vals.max())
    res_abs = float(res_vals.abs().max())

    feature_ranges = {}
    for key in feature_keys:
        sl = feature_slices[key]
        vals = torch.stack([x_sample[t, sl] for t in frames])
        feature_ranges[key] = (float(vals.min()), float(vals.max()))

    n_rows = 3 + len(feature_keys)
    n_cols = len(frames)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for col, t in enumerate(frames):
        quick_hex_scatter(
            y_sample[t],
            fig=fig,
            ax=axes[0, col],
            cbar=False,
            vmin=gt_min,
            vmax=gt_max,
            cmap=plt.get_cmap("gray"),
            edgecolor="none",
        )
        axes[0, col].set_title(f"True t={t}")

        quick_hex_scatter(
            y_hat[t],
            fig=fig,
            ax=axes[1, col],
            cbar=False,
            vmin=pred_min,
            vmax=pred_max,
            cmap=plt.get_cmap("gray"),
            edgecolor="none",
        )
        axes[1, col].set_title(f"Pred t={t}")

        quick_hex_scatter(
            res[t],
            fig=fig,
            ax=axes[2, col],
            cbar=False,
            vmin=-res_abs,
            vmax=res_abs,
            cmap=plt.get_cmap("seismic"),
            edgecolor="none",
        )
        axes[2, col].set_title(f"Residual t={t}")

        for row_offset, key in enumerate(feature_keys, start=3):
            sl = feature_slices[key]
            feat = x_sample[t, sl]
            if feat.numel() != n_hexals:
                continue
            vmin, vmax = feature_ranges[key]
            quick_hex_scatter(
                feat,
                fig=fig,
                ax=axes[row_offset, col],
                cbar=False,
                vmin=vmin,
                vmax=vmax,
                cmap=plt.get_cmap("viridis"),
                edgecolor="none",
            )
            axes[row_offset, col].set_title(f"{key} t={t}")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
