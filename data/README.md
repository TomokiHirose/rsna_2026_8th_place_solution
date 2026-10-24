# data/

Small inputs that the configs read directly. Paths are kept as the configs reference them.

| Path | Content |
|---|---|
| `folds_v1.csv`, `folds_v1.json` | `StudyInstanceUID,fold` used by the team (scanner-grouped, 4 folds). Made by `tools/make_folds.py` |
| `depth_prior/v1.json` | Per-slot depth prior (Gaussian mixture over normalised slice position, per finding) used to sample training windows (`data.depth_prior`) |
| `depth_prior/teacher_folds.csv` | Folds of the teacher used to fit the depth prior |
| `label/ayfukushima/teacher_oof/v4/labels_teacher.csv` | Training labels of that teacher (round-1 `start`) |
| `label/ayfukushima/bilateral_knees/knee_series.csv` | Studies that contain both knees, split per knee (`data.subcases_csv`). Made by `tools/split_bilateral_knees.py` |
| `label/ayfukushima/blend3/pseudo_v6t275_r040/` | Soft labels, round 1 (see below) |
| `label/ayfukushima/blend3/pseudo_v6t275_r040_knees/` | The same labels, split per knee |
| `label/ayfukushima/blend3/pseudo2_v6_r040/` | Soft labels, round 2 |
| `label/ayfukushima/blend3/pseudo2_v6_r040_knees/` | The same labels, split per knee |
| `label/wadakoki/` | How the Qwen and Gemma report scores in the report-based label were generated (prompt, settings, code). The scores themselves are not included |

## Soft labels

Each folder holds `labels_soft.csv` (12 findings, 4,345 non-gold studies; the 58 gold studies come from `train.csv` at training time) and `manifest.json` (formula, sources and their sha256).

`label = (1 − r) × start + r × calibrated teacher OOF`, with `r = 0.4` for every finding.

- `start` blends three sources per finding with an image weight `w_f`: image-based labels × `w_f` + two report-based labels × `(1 − w_f) / 2` each.
- The teacher OOF is calibrated per finding by isotonic regression fitted on the other 3 folds.
- Round 1 (`pseudo_v6t275_r040`) uses a teacher trained on `start`; round 2 (`pseudo2_v6_r040`) uses a teacher with the same recipe trained on the round-1 labels, and is mixed with `start` again (not with round 1).

The source labels and the teachers are not part of this repository; the label files are provided as they were used. For the Qwen and Gemma report scores, the prompt, settings and code are in `label/wadakoki/`.

## Depth prior

`depth_prior/v1.json` is used as is. It was fitted on the window attention of a 4-fold teacher (CoAtNet-2 + spatial branch, trained on the `start` labels of round 1). To rebuild it:

1. Build the teacher's cache. It uses contiguous sampling of long series (the training caches use uniform sampling):

   ```bash
   uv run python tools/build_cache.py --data-dir input/rsna-knee-abnormality-detection --split train --slots native --caps 32,32,32,32,40 --span 0.06,0.94 --size 384 --crop-mm 140 --resample exact --long-series-sampling contiguous --series-layout all --memmap --workers 16 --out data/cache/cache_native_384_series_all_exact
   ```

2. Train the teacher, 4 folds (`split: folds` on `depth_prior/teacher_folds.csv`, the non-gold studies of `folds_v1.csv`; labels `label/ayfukushima/teacher_oof/v4/labels_teacher.csv`; 20 epochs):

   ```bash
   for i in 0 1 2 3; do uv run python -m rsna_knee.training --config configs/depth_prior_teacher/fold$i.json --fold $i; done
   ```

3. Dump the out-of-fold attention of every window (each fold model scores its held-out studies on the uniform-sampling cache):

   ```bash
   uv run python tools/window_attention.py --teacher output/exp242_coatnet_spatial_ema_teacher_start_v4 --cache data/cache/cache_native_384_series_all_exact_uniform_sub --out output/attention_prior
   ```

4. Fit the prior:

   ```bash
   uv run python tools/fit_depth_prior.py --attention output/attention_prior/window_attention.csv.gz --cache data/cache/cache_native_384_series_all_exact_uniform_sub --out data/depth_prior/v1.json
   ```

For each slot and finding it fits a 1-D Gaussian mixture (up to 4 components, chosen by BIC, σ ≥ 0.08) on the GAP-branch attention over the depth u in [−1, 1] (patient axes; medial positive on sagittal slices when the side is known), and stores the equal-weight mixture of the 12 findings.
