#!/usr/bin/env python3
"""Simple per-frame linear decode test for (activity -> lum)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear decoder from activity to lum per frame."
    )
    parser.add_argument(
        "--data",
        default="recon/outputs/activity_lum_pairs.pt",
        help="Path to saved (X, Y) tensors.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--output",
        default="recon/outputs/linear_decoder_fullfield.pt",
        help="Path to save the trained decoder state_dict.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"loading data from {data_path.resolve()}")

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

    dataset = TensorDataset(X_flat, Y_flat)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    model = torch.nn.Linear(n_features, n_hexals).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    print(f"X_flat {tuple(X_flat.shape)} | Y_flat {tuple(Y_flat.shape)}")
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

    output_path = Path(args.output)
    torch.save(model.state_dict(), output_path)
    print(f"saved decoder state_dict to {output_path}")


if __name__ == "__main__":
    main()
