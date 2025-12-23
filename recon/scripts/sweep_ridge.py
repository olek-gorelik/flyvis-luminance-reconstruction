#!/usr/bin/env python3
"""Run a ridge sweep and evaluate on the test split."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


RIDGE_VALUES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

TRAIN_DATA = "recon/outputs/activity_lum_train_fullfield.pt"
TEST_DATA = "recon/outputs/activity_lum_test_fullfield.pt"

TRAIN_SCRIPT = "recon/scripts/train_linear_fullfield_norm_split_ridge.py"
EVAL_SCRIPT = "recon/scripts/eval_metrics.py"

OUTPUT_DIR = Path("recon/outputs")
RESULTS_CSV = OUTPUT_DIR / "ridge_sweep_results.csv"


def ridge_tag(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def run_command(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def parse_metrics(output: str) -> dict[str, float]:
    metrics = {}
    for line in output.splitlines():
        match = re.match(r"^(MSE|NMSE|PSNR|Corr|MAE): mean=([0-9eE.+-]+)", line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    best_corr = (-float("inf"), None)
    best_nmse = (float("inf"), None)

    for ridge in RIDGE_VALUES:
        tag = ridge_tag(ridge)
        out_model = OUTPUT_DIR / f"linear_decoder_fullfield_norm_split_ridge_{tag}.pt"
        out_stats = OUTPUT_DIR / f"linear_decoder_fullfield_norm_split_ridge_{tag}_stats.pt"
        out_log = OUTPUT_DIR / f"linear_decoder_fullfield_norm_split_ridge_{tag}_log.json"

        train_cmd = [
            sys.executable,
            TRAIN_SCRIPT,
            "--train-data",
            TRAIN_DATA,
            "--ridge",
            str(ridge),
            "--out-model",
            str(out_model),
            "--out-stats",
            str(out_stats),
            "--out-log",
            str(out_log),
        ]
        print(f"training ridge={ridge} -> {out_model.name}")
        run_command(train_cmd)

        eval_cmd = [
            sys.executable,
            EVAL_SCRIPT,
            "--model",
            str(out_model),
            "--data",
            TEST_DATA,
            "--xnorm-stats",
            str(out_stats),
            "--split-name",
            "test",
        ]
        print(f"evaluating ridge={ridge} on test")
        output = run_command(eval_cmd)
        metrics = parse_metrics(output)

        row = {
            "ridge": ridge,
            "MSE": metrics.get("MSE"),
            "NMSE": metrics.get("NMSE"),
            "PSNR": metrics.get("PSNR"),
            "Corr": metrics.get("Corr"),
            "MAE": metrics.get("MAE"),
        }
        rows.append(row)

        corr = row["Corr"] if row["Corr"] is not None else -float("inf")
        nmse = row["NMSE"] if row["NMSE"] is not None else float("inf")
        if corr > best_corr[0]:
            best_corr = (corr, ridge)
        if nmse < best_nmse[0]:
            best_nmse = (nmse, ridge)

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ridge", "MSE", "NMSE", "PSNR", "Corr", "MAE"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"saved sweep results to {RESULTS_CSV}")
    print(f"best ridge by TEST Corr: {best_corr[1]} (Corr={best_corr[0]:.6f})")
    print(f"best ridge by TEST NMSE: {best_nmse[1]} (NMSE={best_nmse[0]:.6f})")


if __name__ == "__main__":
    main()
