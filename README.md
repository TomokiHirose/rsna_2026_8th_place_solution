# RSNA 2026 Knee Abnormality Detection — 8th place solution

[![Kaggle](https://img.shields.io/badge/Kaggle-8th%20place-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![CI](https://github.com/TomokiHirose/rsna_2026_8th_place_solution/actions/workflows/ci.yml/badge.svg)](https://github.com/TomokiHirose/rsna_2026_8th_place_solution/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TomokiHirose/rsna_2026_8th_place_solution/graph/badge.svg)](https://codecov.io/gh/TomokiHirose/rsna_2026_8th_place_solution)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Code and configs to reproduce our two final submissions (A, B) and our best private-score submission (C) to the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition, from cache building to a late submission on Kaggle. The write-up is in [`docs/index.en.md`](docs/index.en.md).

| Submission | Models | Public | Private |
|---|---|---:|---:|
| A | 12 series_neck models: CoAtNet-2 140 mm × 4, CoAtNet-2 110 mm × 4, CAFormer-S36 140 mm × 4. Equal probability average | 0.968 | 0.957 |
| B | 4 parts at 0.25 each (probability blend): ctxenc × 4, slotctx × 4, CAFormer-S36 110 mm × 3, CAFormer-S36 140 mm × 4 | 0.967 | 0.957 |
| C (private best) | 16 models on the round-2 labels `pseudo2_v6_r040`, probability blend: CoAtNet-2 140 mm × 4 + 110 mm × 4 (0.6), ctxenc × 4 (0.2), slotctx × 4 (0.2) | 0.967 | 0.958 |

The CAFormer-S36 models are shared by A and B. Inference uses no TTA. C was not selected as a final submission; its notebook runs from the same model dataset, but the configs to retrain its 16 models are not included yet (the round-2-label versions of the CoAtNet-2 and ctxenc/slotctx sets).

The test set is hidden, so a submission can only be scored by running the notebook on Kaggle. The flow is: train → pick the weights → upload the code and weights as Kaggle datasets → push the notebook → submit that notebook version.

## Layout

```
rsna_knee/        model, data pipeline, training (python -m rsna_knee.training) and inference library
configs/<set>/<stage>/fold<i>.json
                  one directory per training set, one file per fold (seed, labels and initial weights differ per fold)
data/             folds, soft labels, depth prior, per-knee split (see data/README.md)
tests/            end-to-end and unit tests on synthetic DICOM studies (CPU only)
tools/            cache building, depth prior, weight selection, Kaggle operations
kernels/A, kernels/B, kernels/C
                  inference notebooks of the submissions (+ kernel-metadata.json); all read one model dataset
docs/             write-up
```

## Setup

```bash
uv sync --extra train
```

Tests (synthetic data only, CPU, a few minutes): `uv sync --extra train && uv run pytest`.

Kaggle credentials (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`) are needed for the data download and the Kaggle steps. Accept the competition rules first, then download the data to `input/rsna-knee-abnormality-detection/`:

```bash
uv run kaggle competitions download -c rsna-knee-abnormality-detection -p input/rsna-knee-abnormality-detection
unzip -q input/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip -d input/rsna-knee-abnormality-detection
```

We trained on RTX 4090 / RTX 5090 GPUs (24–32 GB). Each base model takes several hours per fold.

## 1. Build the caches

Two caches of 384 × 384 crops (5 slots of 32/32/32/32/40 slices per study, uniform sampling, all series, sub-series):

```bash
COMMON="--data-dir input/rsna-knee-abnormality-detection --split train --slots native --caps 32,32,32,32,40 --span 0.06,0.94 --size 384 --resample exact --long-series-sampling uniform --series-layout all --subseries --memmap --workers 16"
uv run python tools/build_cache.py $COMMON --crop-mm 140 --out data/cache/cache_native_384_series_all_exact_uniform_sub
uv run python tools/build_cache.py $COMMON --crop-mm 110 --anterior-shift-mm 10 --out data/cache/cache_native_384_series_all_exact_crop110_ant10_uniform_sub
```

Each cache is about 110 GB. `manifest.json` in the cache records the geometry and is copied into the model config at training time.

The depth prior `data/depth_prior/v1.json` is provided; `data/README.md` describes how to rebuild it (teacher training, attention dump, fit).

## 2. Train

Every model is trained in two stages: a base model (20 epochs, end-to-end), then a second stage (10 epochs) on a frozen encoder initialised from the base. Run one fold at a time:

```bash
uv run python -m rsna_knee.training --config configs/<set>/<stage>/fold<i>.json --fold <i>
```

Outputs go to `output/<experiment_name>/fold<i>/`, where `experiment_name` is written inside the config (the long names are kept because the notebooks refer to them).

| Set | Stages | Folds | Split | Cache | Used in |
|---|---|---|---|---|---|
| `coatnet2_140mm` | `base` → `neck` | 0–3 | gold | 140 mm | A |
| `coatnet2_110mm` | `base` → `neck` | 0–3 | gold | 110 mm | A |
| `caformer_s36_140mm` | `base` → `neck` | 0–3 | gold | 140 mm | A, B |
| `caformer_s36_110mm` | `base` → `neck` | 0–2 | gold | 110 mm | B |
| `ctxenc` | `base` → `stage2` | 0–3 | all | 140 mm | B |
| `slotctx` | `base` → `stage2` | 0–3 | all | 140 mm | B |

- **gold split**: the 58 gold studies are held out and used as validation; `--fold` only names the output directory. The `neck` stage starts from the base's `checkpoints/epoch19_ema.pt`.
- **all split**: all labelled studies are used for training. The `stage2` stage starts from the base's `model.pt`.
- In `ctxenc` and `slotctx`, `stage2` folds are not matched one-to-one with base folds, and folds 2–3 use the round-2 labels:

| `stage2` fold | seed | labels | base fold used as init |
|---:|---:|---|---:|
| 0 | 42 | `pseudo_v6t275_r040` | 0 |
| 1 | 65536 | `pseudo_v6t275_r040` | 3 |
| 2 | 120 | `pseudo2_v6_r040` | 1 |
| 3 | 223 | `pseudo2_v6_r040` | 2 |

(`slotctx` uses the `_knees` variants of the labels.) So train all base folds of a set before its second stage.

Example (`caformer_s36_140mm`):

```bash
for i in 0 1 2 3; do
  uv run python -m rsna_knee.training --config configs/caformer_s36_140mm/base/fold$i.json --fold $i
  uv run python -m rsna_knee.training --config configs/caformer_s36_140mm/neck/fold$i.json --fold $i
done
```

## 3. Pick the weights

The notebooks read `output/submit/<experiment_name>/` (`inference.json` + `fold<i>/model.pt` + `fold<i>/config.json`).

**gold-split sets (`neck`)**: per fold, take the epoch with the best gold `val/auc_macro` (EMA); if it is earlier than epoch 4, take the epoch-4 EMA instead. Inference uses the all-series bag capped at 384 windows.

```bash
uv run python tools/select_weights.py <neck experiment_name> [...] --union-cap 384
```

**all-split sets (`stage2`)**: there is no validation; the final `model.pt` is used.

```bash
for e in <ctxenc stage2 experiment_name> <slotctx stage2 experiment_name>; do
  mkdir -p output/submit/$e
  cp output/$e/inference.json output/submit/$e/
  for i in 0 1 2 3; do mkdir -p output/submit/$e/fold$i; cp output/$e/fold$i/model.pt output/$e/fold$i/config.json output/submit/$e/fold$i/; done
done
```

## 4. Submit on Kaggle

`tools/kaggle_ops.py` takes your Kaggle username with `--owner`. It uploads `rsna_knee/` and `output/submit/` as two datasets, pushes the notebook of a submission, and submits a notebook version as a late submission.

```bash
uv run python tools/kaggle_ops.py --owner <you> pkg -m "rsna_knee"
uv run python tools/kaggle_ops.py --owner <you> model -m "submission A weights"
uv run python tools/kaggle_ops.py --owner <you> kernel --target A --wait
uv run python tools/kaggle_ops.py --owner <you> submit --target A --version <N> --message "submission A"
```

The model dataset (`<you>/rsna-knee-8th-model`) is the whole `output/submit/`, so keep the stores of every submission there; the notebooks share it with the pkg dataset (`<you>/rsna-knee-8th-pkg`) and each reads only the experiments in its `Ensemble`. Kernels: A `rsna-knee-8th-inference-a`, B `rsna-knee-8th-inference-b`, C `rsna-knee-8th-private-best-ensemble`.

The submit step runs `kaggle competitions submit -c rsna-knee-abnormality-detection -f submission.csv -k <you>/rsna-knee-8th-inference-a -v <N> -m ...`.

The notebooks run offline on a T4 × 2 machine. The Python packages that are missing from the Kaggle image are attached through the Package Manager kernel listed in `kernel_sources` of `kernel-metadata.json`. Scoring submission A takes about 6 h 40 min (the limit is 9 h).

## License

See [LICENSE](LICENSE).
