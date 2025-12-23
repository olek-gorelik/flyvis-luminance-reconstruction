#!/usr/bin/env python3
"""Visualize full-field reconstruction vs ground truth for selected frames."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flyvis.connectome import ConnectomeFromAvgFilters


DATA_PATH = Path("recon/outputs/activity_lum_pairs_fullfield.pt")
MODEL_PATH = Path("recon/outputs/linear_decoder_fullfield.pt")
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

    y_min = float(min(y_sample.min(), y_hat.min()))
    y_max = float(max(y_sample.max(), y_hat.max()))
    res = y_hat - y_sample
    res_abs = float(res.abs().max())

    feature_slices = None
    if isinstance(cell_types, list):
        try:
            feature_slices = compute_feature_slices(cell_types, n_features)
        except Exception as exc:
            print(f"feature map slicing disabled: {exc}")

    show_features = feature_slices is not None and "R1" in feature_slices and "R6" in feature_slices
    n_rows = 4 if show_features else 3
    n_cols = len(frames)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)

    for col, t in enumerate(frames):
        gt = y_sample[t].numpy()
        pred = y_hat[t].numpy()
        resid = res[t].numpy()

        axes[0, col].scatter(np.arange(n_hexals), gt, s=3, color="black")
        axes[0, col].set_title(f"GT t={t}")
        axes[0, col].set_ylim(y_min, y_max)

        axes[1, col].scatter(np.arange(n_hexals), pred, s=3, color="tab:blue")
        axes[1, col].set_title(f"Pred t={t}")
        axes[1, col].set_ylim(y_min, y_max)

        axes[2, col].scatter(np.arange(n_hexals), resid, s=3, color="tab:red")
        axes[2, col].set_title(f"Residual t={t}")
        axes[2, col].set_ylim(-res_abs, res_abs)

        if show_features:
            r1_slice = feature_slices["R1"]
            r6_slice = feature_slices["R6"]
            r1 = x_sample[t, r1_slice].numpy()
            r6 = x_sample[t, r6_slice].numpy()
            feat_min = float(min(r1.min(), r6.min()))
            feat_max = float(max(r1.max(), r6.max()))

            axes[3, col].scatter(np.arange(len(r1)), r1, s=3, color="tab:green", label="R1")
            axes[3, col].scatter(np.arange(len(r6)), r6, s=3, color="tab:orange", label="R6")
            axes[3, col].set_title(f"Features t={t}")
            axes[3, col].set_ylim(feat_min, feat_max)
            if col == 0:
                axes[3, col].legend(loc="upper right", fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("hex index")
        ax.set_ylabel("value")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
