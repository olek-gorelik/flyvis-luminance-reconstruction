#!/usr/bin/env python3
"""Train a linear decoder on normalized full-field activity (train split only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear decoder on the train split with feature normalization."
    )
    parser.add_argument(
        "--data",
        default="recon/outputs/activity_lum_train_fullfield.pt",
        help="Path to train split (X, Y) tensors.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--out-model",
        default="recon/outputs/linear_decoder_fullfield_norm_split.pt",
        help="Path to save the trained decoder state_dict.",
    )
    parser.add_argument(
        "--out-stats",
        default="recon/outputs/linear_decoder_fullfield_norm_split_stats.pt",
        help="Path to save normalization statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = torch.load(data_path, map_location="cpu")
    X = data["X"].float()
    Y = data["Y"].float()

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {tuple(X.shape)}")
    if Y.ndim != 4:
        raise ValueError(f"Expected Y with shape (N, T, 1, H), got {tuple(Y.shape)}")

    Y = Y.squeeze(2)
    if Y.ndim != 3:
        raise ValueError(
            f"Expected Y after squeeze to be (N, T, H), got {tuple(Y.shape)}"
        )

    n_samples, n_frames, n_features = X.shape
    n_hexals = Y.shape[-1]

    X_flat = X.reshape(-1, n_features)
    Y_flat = Y.reshape(-1, n_hexals)

    eps = 1e-6
    mean = X_flat.mean(dim=0)
    std = X_flat.std(dim=0, unbiased=False)
    Xn = (X_flat - mean) / (std + eps)

    dataset = TensorDataset(Xn, Y_flat)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    model = torch.nn.Linear(n_features, n_hexals).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    print(f"model: Linear({n_features} -> {n_hexals}) on {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"epoch {epoch:03d} | mse {avg_loss:.6f}")

    out_model = Path(args.out_model)
    out_stats = Path(args.out_stats)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_stats.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_model)
    print(f"saved decoder state_dict to {out_model}")

    torch.save(
        {
            "mean": mean,
            "std": std,
            "eps": eps,
            "F": n_features,
            "H": n_hexals,
        },
        out_stats,
    )
    print(f"saved normalization stats to {out_stats}")


if __name__ == "__main__":
    main()
