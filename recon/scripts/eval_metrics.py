#!/usr/bin/env python3
"""Evaluate reconstruction metrics for lum predictions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NMSE, PSNR, and correlation for lum reconstruction."
    )
    parser.add_argument(
        "--data",
        default="recon/outputs/activity_lum_pairs.pt",
        help="Path to saved (X, Y) tensors.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a torch-saved model or linear state_dict.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--xnorm-stats",
        default=None,
        help="Path to feature normalization stats (mean/std/eps).",
    )
    parser.add_argument(
        "--no-xnorm",
        action="store_true",
        help="Disable feature normalization even if stats are provided.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--split-name",
        default=None,
        choices=["train", "test"],
        help="Optional split label for printing (train or test).",
    )
    return parser.parse_args()


def load_model(
    model_path: Path, n_features: int, n_hexals: int, device: torch.device
) -> torch.nn.Module:
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        if "state_dict" in obj:
            state = obj["state_dict"]
        else:
            state = obj
        model = torch.nn.Linear(n_features, n_hexals)
        model.load_state_dict(state)
    else:
        raise TypeError(f"Unsupported model object: {type(obj)}")
    return model.to(device)


def per_frame_corr(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_pred = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true = y_true - y_true.mean(dim=1, keepdim=True)
    num = (y_pred * y_true).sum(dim=1)
    denom = torch.sqrt((y_pred**2).sum(dim=1)) * torch.sqrt((y_true**2).sum(dim=1))
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return num / denom


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if args.split_name:
        print(f"{args.split_name.upper()} METRICS")

    data = torch.load(data_path, map_location="cpu")
    X = data["X"].float()
    Y = data["Y"].float()

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {tuple(X.shape)}")
    if Y.ndim != 4:
        raise ValueError(f"Expected Y with shape (N, T, 1, H), got {tuple(Y.shape)}")

    y_min = float(Y.min())
    y_max = float(Y.max())
    y_mean = float(Y.mean())
    y_var = float(Y.var(unbiased=False))
    data_range = y_max - y_min

    print(f"Y stats: min={y_min:.6f} max={y_max:.6f} mean={y_mean:.6f} var={y_var:.6f}")

    Y = Y.squeeze(2)
    n_samples, n_frames, n_features = X.shape
    n_hexals = Y.shape[-1]

    X_flat = X.reshape(-1, n_features)
    Y_flat = Y.reshape(-1, n_hexals)

    x_min = float(X_flat.min())
    x_max = float(X_flat.max())
    x_mean = float(X_flat.mean())
    x_std = float(X_flat.std(unbiased=False))
    print(
        f"X_flat stats (pre-norm): min={x_min:.6f} max={x_max:.6f} "
        f"mean={x_mean:.6f} std={x_std:.6f}"
    )

    norm_applied = False
    if args.xnorm_stats and not args.no_xnorm:
        stats = torch.load(Path(args.xnorm_stats), map_location="cpu")
        mean = stats["mean"]
        std = stats["std"]
        eps = stats.get("eps", 1e-6)
        if mean.numel() != n_features or std.numel() != n_features:
            raise ValueError(
                f"Normalization stats mismatch: mean/std length {mean.numel()} "
                f"vs F={n_features}"
            )
        X_flat = (X_flat - mean) / (std + eps)
        norm_applied = True

    x_min = float(X_flat.min())
    x_max = float(X_flat.max())
    x_mean = float(X_flat.mean())
    x_std = float(X_flat.std(unbiased=False))
    print(
        f"X_flat stats (post-norm): min={x_min:.6f} max={x_max:.6f} "
        f"mean={x_mean:.6f} std={x_std:.6f}"
    )
    print(f"feature normalization applied: {norm_applied}")

    device = torch.device(args.device)
    model = load_model(Path(args.model), n_features, n_hexals, device)
    model.eval()

    loader = DataLoader(
        TensorDataset(X_flat, Y_flat),
        batch_size=args.batch_size,
        shuffle=False,
    )

    mse_list = []
    mae_list = []
    corr_list = []
    psnr_list = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)

            mse = ((pred - yb) ** 2).mean(dim=1)
            mae = (pred - yb).abs().mean(dim=1)
            corr = per_frame_corr(pred, yb)

            mse_list.append(mse.cpu())
            mae_list.append(mae.cpu())
            corr_list.append(corr.cpu())

            mse_safe = torch.clamp(mse, min=1e-12)
            psnr = 20.0 * math.log10(max(data_range, 1e-12)) - 10.0 * torch.log10(
                mse_safe
            )
            psnr_list.append(psnr.cpu())

    mse_all = torch.cat(mse_list)
    mae_all = torch.cat(mae_list)
    corr_all = torch.cat(corr_list)
    psnr_all = torch.cat(psnr_list)
    nmse_all = mse_all / max(y_var, 1e-12)

    def report(name: str, values: torch.Tensor) -> None:
        mean = values.mean().item()
        std = values.std(unbiased=False).item()
        print(f"{name}: mean={mean:.6f} std={std:.6f}")

    report("MSE", mse_all)
    report("NMSE", nmse_all)
    report("PSNR", psnr_all)
    report("Corr", corr_all)
    report("MAE", mae_all)


if __name__ == "__main__":
    main()
