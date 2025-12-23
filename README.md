# FlyVis Luminance Reconstruction

This repository implements an inverse luminance reconstruction pipeline on top of FlyVis, a connectome-based model of the Drosophila visual system. Using only external scripts (no FlyVis core modifications), it tests how much visual information is preserved in early neural activity and how well luminance can be decoded with simple linear models.

The goal is not to extend FlyVis itself, but to treat FlyVis as a forward model and evaluate the inverse problem under clean, interpretable baselines.

## Background: FlyVis
FlyVis simulates neural activity from photoreceptors through downstream visual circuits. It is built for forward encoding tasks (e.g., optic flow), not inverse image reconstruction. This project uses FlyVis as a black-box forward model and builds the reconstruction pipeline entirely outside the FlyVis core.

Key points:
- FlyVis does not provide luminance reconstruction out of the box.
- Existing FlyVis decoders are task-specific, not inverse image models.
- Activity is simulated on a hexagonal retinal lattice, which is ideal for decoding tests.

## What This Repository Adds
This project adds a reconstruction pipeline that:
- Extracts neural activity from FlyVis in response to visual stimuli.
- Pairs activity with retinal luminance on the same hex lattice.
- Trains and evaluates linear decoders for luminance reconstruction.
- Compares central-column vs full-field activity.
- Applies feature normalization and ridge regularization for generalization.

No FlyVis core modules are modified.

## Pipeline Overview
1) Activity extraction
   - `recon/scripts/extract_activity.py` (central activity)
   - `recon/scripts/extract_activity_fullfield.py` (full-field activity)

2) Baseline decoding
   - `recon/scripts/test_invertibility.py`
   - `recon/scripts/test_invertibility_fullfield_norm.py`

3) Train/test split (sequence-level, no temporal leakage)
   - `recon/scripts/make_sintel_split.py`

4) Normalized training (train split only)
   - `recon/scripts/train_linear_fullfield_norm_split.py`

5) Ridge regularization and sweep
   - `recon/scripts/train_linear_fullfield_norm_split_ridge.py`
   - `recon/scripts/sweep_ridge.py`

6) Evaluation and visualization
   - `recon/scripts/eval_metrics.py`
   - `recon/scripts/visualize_recon.py`
   - `recon/scripts/visualize_recon_retinal*.py`
   - `recon/scripts/visualize_train_vs_test*.py`

## Key Results (Summary)
With full-field activity, proper feature normalization, and ridge regularization:
- Train reconstructions are sharp and structured.
- Test reconstructions generalize to held-out sequences.
- Residuals are localized and low-amplitude.

These results indicate early fly visual representations preserve enough information for accurate luminance reconstruction using linear models.

## Requirements
- A working FlyVis installation.
- Python environment compatible with FlyVis (e.g., the FlyVis conda env).

This repository does not include FlyVis or downloaded datasets. FlyVis must be installed separately.

## Reproducing the Full Split Pipeline
The following is a clean, leakage-free train/test split workflow.

1) Create a fixed split
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/make_sintel_split.py --seed 0
```

2) Extract train data
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/extract_activity_fullfield.py \
  --indices recon/splits/sintel_split_seed0.json \
  --indices-key train_indices \
  --output-name activity_lum_train_fullfield.pt
```

3) Extract test data
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/extract_activity_fullfield.py \
  --indices recon/splits/sintel_split_seed0.json \
  --indices-key test_indices \
  --output-name activity_lum_test_fullfield.pt
```

4) Train normalized decoder (train split only)
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/train_linear_fullfield_norm_split.py \
  --data recon/outputs/activity_lum_train_fullfield.pt \
  --out-model recon/outputs/linear_decoder_fullfield_norm_split.pt \
  --out-stats recon/outputs/linear_decoder_fullfield_norm_split_stats.pt
```

5) Ridge sweep and evaluation
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/sweep_ridge.py
```

6) Evaluate a chosen ridge model on train/test
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/eval_metrics.py \
  --model recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4.pt \
  --data recon/outputs/activity_lum_train_fullfield.pt \
  --xnorm-stats recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4_stats.pt \
  --split-name train

/opt/miniconda3/envs/flyvis/bin/python recon/scripts/eval_metrics.py \
  --model recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4.pt \
  --data recon/outputs/activity_lum_test_fullfield.pt \
  --xnorm-stats recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4_stats.pt \
  --split-name test
```

7) Visualize train vs test
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/visualize_train_vs_test_ridge.py \
  --model recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4.pt \
  --xnorm-stats recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4_stats.pt \
  --train-data recon/outputs/activity_lum_train_fullfield.pt \
  --test-data recon/outputs/activity_lum_test_fullfield.pt \
  --n-samples 10 \
  --seed 0
```

## Metrics
Evaluation uses scale-aware reconstruction metrics:
- MSE
- NMSE (MSE / Var(Y))
- PSNR (using data range)
- Per-frame Pearson correlation
- MAE

## Visualization
Two styles are provided:
- `visualize_recon.py`: quick 1D hex index scatter plots.
- `visualize_recon_retinal*.py`: true FlyVis hex-retinal layout via `quick_hex_scatter`.

Train/test comparisons are saved as PNGs under `recon/outputs/`.

## Scope and Limitations
- Focused on luminance reconstruction, not semantic image reconstruction.
- Linear decoders only; no deep or diffusion models.
- No temporal modeling beyond per-frame decoding.

## Relationship to FlyVis
- No forks or core modifications.
- FlyVis is used as an external forward model.
- Scripts should remain compatible with future FlyVis releases.

## Notes
Exact arguments and convenience run configs can be stored in `.vscode/launch.json`.

