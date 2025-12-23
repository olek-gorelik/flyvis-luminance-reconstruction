## FlyVis Luminance Reconstruction

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

## Script Inventory
All reconstruction logic lives in `recon/scripts/`. Use these scripts to recreate any part of the results:

Extraction
- `recon/scripts/extract_activity.py` (central activity)
- `recon/scripts/extract_activity_fullfield.py` (full-field activity, supports index lists)

Baseline decoding
- `recon/scripts/test_invertibility.py`
- `recon/scripts/test_invertibility_fullfield_norm.py`

Splits
- `recon/scripts/make_sintel_split.py`

Normalized training
- `recon/scripts/train_linear_fullfield_norm_split.py`

Ridge regularization
- `recon/scripts/train_linear_fullfield_norm_split_ridge.py`
- `recon/scripts/sweep_ridge.py`

Evaluation
- `recon/scripts/eval_metrics.py`

Visualization
- `recon/scripts/visualize_recon.py`
- `recon/scripts/visualize_recon_retinal.py`
- `recon/scripts/visualize_recon_retinal_central.py`
- `recon/scripts/visualize_recon_retinal_fullfield_norm.py`
- `recon/scripts/visualize_train_vs_test.py`
- `recon/scripts/visualize_train_vs_test_ridge.py`

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

## Best Working Workflow (Full-Field + Split + Normalization + Ridge)
This is the recommended pipeline for strong generalization on held-out sequences.

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

4) Ridge sweep and evaluation
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/sweep_ridge.py
```

5) Evaluate a chosen ridge model on train/test
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

6) Visualize train vs test (ridge)
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/visualize_train_vs_test_ridge.py \
  --model recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4.pt \
  --xnorm-stats recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4_stats.pt \
  --train-data recon/outputs/activity_lum_train_fullfield.pt \
  --test-data recon/outputs/activity_lum_test_fullfield.pt \
  --n-samples 10 \
  --seed 0
```

## Showcase of Best Workflow Performance
<img width="1377" height="8500" alt="FN Ridge Split Testing 20" src="https://github.com/user-attachments/assets/7b7f4660-2b3e-4eef-a73b-3b63007e1a7d" />

## Alternative Workflows (Ablations and Baselines)
Use these to isolate the effect of central vs full-field, normalization, and ridge.

### A) Central-only baseline (no normalization, no split)
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/extract_activity.py \
  --output-name activity_lum_pairs.pt

/opt/miniconda3/envs/flyvis/bin/python recon/scripts/test_invertibility.py \
  --data recon/outputs/activity_lum_pairs.pt \
  --output recon/outputs/linear_decoder.pt

/opt/miniconda3/envs/flyvis/bin/python recon/scripts/eval_metrics.py \
  --model recon/outputs/linear_decoder.pt \
  --data recon/outputs/activity_lum_pairs.pt
```

### B) Full-field baseline (no normalization, no split)
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/extract_activity_fullfield.py \
  --output-name activity_lum_pairs_fullfield.pt

/opt/miniconda3/envs/flyvis/bin/python recon/scripts/test_invertibility.py \
  --data recon/outputs/activity_lum_pairs_fullfield.pt \
  --output recon/outputs/linear_decoder_fullfield.pt

/opt/miniconda3/envs/flyvis/bin/python recon/scripts/eval_metrics.py \
  --model recon/outputs/linear_decoder_fullfield.pt \
  --data recon/outputs/activity_lum_pairs_fullfield.pt
```

### C) Full-field with normalization (no split)
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/test_invertibility_fullfield_norm.py

/opt/miniconda3/envs/flyvis/bin/python recon/scripts/eval_metrics.py \
  --model recon/outputs/linear_decoder_fullfield_norm.pt \
  --data recon/outputs/activity_lum_pairs_fullfield.pt \
  --xnorm-stats recon/outputs/linear_decoder_fullfield_norm_stats.pt
```

### D) Full-field with normalization (train/test split, no ridge)
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/train_linear_fullfield_norm_split.py \
  --data recon/outputs/activity_lum_train_fullfield.pt \
  --out-model recon/outputs/linear_decoder_fullfield_norm_split.pt \
  --out-stats recon/outputs/linear_decoder_fullfield_norm_split_stats.pt

/opt/miniconda3/envs/flyvis/bin/python recon/scripts/eval_metrics.py \
  --model recon/outputs/linear_decoder_fullfield_norm_split.pt \
  --data recon/outputs/activity_lum_test_fullfield.pt \
  --xnorm-stats recon/outputs/linear_decoder_fullfield_norm_split_stats.pt \
  --split-name test
```

### E) Ridge-regularized split (manual single ridge value)
```
/opt/miniconda3/envs/flyvis/bin/python recon/scripts/train_linear_fullfield_norm_split_ridge.py \
  --train-data recon/outputs/activity_lum_train_fullfield.pt \
  --ridge 1e-4 \
  --out-model recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4.pt \
  --out-stats recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4_stats.pt \
  --out-log recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4_log.json

/opt/miniconda3/envs/flyvis/bin/python recon/scripts/eval_metrics.py \
  --model recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4.pt \
  --data recon/outputs/activity_lum_test_fullfield.pt \
  --xnorm-stats recon/outputs/linear_decoder_fullfield_norm_split_ridge_1e-4_stats.pt \
  --split-name test
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



