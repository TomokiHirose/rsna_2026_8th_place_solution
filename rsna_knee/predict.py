from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch

from rsna_knee.config import Config
from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.data.dicom import PreprocessParams, resolve_slots
from rsna_knee.data.slots import make_slots, num_slices
from rsna_knee.experiments import load_experiment_config
from rsna_knee.inference import load_model
from rsna_knee.models import WindowModel
from rsna_knee.models.factory import build_model
from rsna_knee.models.window_model import compile_backbone
from rsna_knee.pipeline import DEFAULT_PREFETCH, DEFAULT_WORKERS, PhaseCounters, iter_studies, place_models, score_stream

__all__ = [
    "BACKBONES",
    "DEFAULT_TIME_BUDGET_S",
    "INFER_DTYPES",
    "PROB_HI",
    "PROB_LO",
    "Members",
    "group_by_preprocess",
    "load_members",
    "load_experiments",
    "load_models",
    "main",
    "member_weights",
    "predict_split",
    "preprocess_params_for",
    "read_split",
    "resolve_device",
    "resolve_devices",
    "submission_columns",
]

logger = logging.getLogger("rsna_knee.predict")

#: 採点側が求めるのは確率であり、sigmoid ヘッドが 0 や 1 を出す根拠はない。
PROB_LO, PROB_HI = 1e-4, 1.0 - 1e-4
#: このコンペの Kaggle notebook の制限時間。
DEFAULT_TIME_BUDGET_S = 8.0 * 3600.0
#: ``--infer-dtype`` の値 -> ``model.infer_dtype``。bf16 が無いのは T4 が持たないから。
INFER_DTYPES: dict[str, torch.dtype | None] = {"none": None, "float16": torch.float16}
#: ``--backbone`` の値。``torch`` は ``infer_dtype`` の autocast、``compile`` は ``torch.compile`` で包む。
BACKBONES = ("torch", "compile")


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` -> cuda が見えれば cuda、無ければ cpu。それ以外はその名前の device。"""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def resolve_devices(device: str | torch.device = "auto", devices: Sequence[str | torch.device] | None = None) -> list[torch.device]:
    """採点に使う device の列。

    ``devices`` を明示すればそのまま使う。``auto`` と ``cuda`` は見える CUDA device すべて、``cpu`` は 1 つ。
    """
    if devices:
        return [resolve_device(str(d)) for d in devices]
    name = str(device)
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        return [resolve_device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [resolve_device(name)]


def read_split(data_dir: str | Path, split: str) -> tuple[list[str], dict[str, list[dict]]]:
    """``{split}.csv`` の順に並んだ study id と、study ごとの ``{split}_series.csv`` の行。

    行の順序は保つ（どの series がどの Slot を埋めるかをそれが決める）。series の行が無い study も残す。
    """
    data_dir = Path(data_dir)
    series_path = data_dir / f"{split}_series.csv"
    if not series_path.exists():
        raise FileNotFoundError(f"missing {series_path}")
    series = pd.read_csv(series_path)

    study_csv = data_dir / f"{split}.csv"
    if study_csv.exists():
        studies = pd.read_csv(study_csv, usecols=[ID_COLUMN])[ID_COLUMN].astype(str).tolist()
    else:  # 途中までしか落ちていない split。id の出所は series 表しかない
        logger.warning("%s does not exist; taking study ids from %s", study_csv, series_path)
        studies = series[ID_COLUMN].astype(str).drop_duplicates().tolist()

    seen: set[str] = set()
    study_ids = [s for s in studies if not (s in seen or seen.add(s))]
    by_study = {str(uid): frame.to_dict(orient="records") for uid, frame in series.groupby(ID_COLUMN)}
    return study_ids, {s: by_study.get(s, []) for s in study_ids}


def submission_columns(data_dir: str | Path) -> list[str]:
    """``sample_submission.csv`` の列順。無い・読めない・既知の 12 ラベルと合わないときは正準の列順。"""
    canonical = [ID_COLUMN, *LABELS]
    path = Path(data_dir) / "sample_submission.csv"
    if not path.exists():
        return canonical
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:  # sample submission が読めなくても実行を止めない
        logger.warning("could not read %s (%s); using the canonical column order", path, exc)
        return canonical
    if columns and columns[0] == ID_COLUMN and set(columns[1:]) == set(LABELS):
        return columns
    logger.warning("%s columns %s do not match LABELS; using the canonical column order", path, columns)
    return canonical


def preprocess_params_for(model: WindowModel) -> PreprocessParams:
    """``model`` の学習時の画素を再現する DICOM -> tensor のレシピ。

    artifact 自身の ``model.preprocess`` から読む。無い artifact のレシピは既定値そのものである。
    """
    recipe = model.cfg.preprocess
    if recipe is None:
        return PreprocessParams()
    return PreprocessParams(
        crop_mm=recipe.crop_mm,
        size=recipe.size,
        span=tuple(recipe.span),
        percentiles=tuple(recipe.percentiles),
        default_pixel_spacing=recipe.default_pixel_spacing,
        caps=tuple(recipe.slot_table),
        native_spacing=recipe.native_spacing,
        resample=recipe.resample,
        long_series_sampling=recipe.long_series_sampling,
        crop_pad=recipe.crop_pad,
        histogram_standard=None if recipe.histogram_standard is None else tuple(tuple(row) for row in recipe.histogram_standard),
        # series の TTA はすべての series と sub-series を 1 度にデコードし、回ごとの Stack を推論側で組む
        # union bag も全 series と sub-series を 1 度にデコードして版を組む
        series_layout="all" if getattr(model, "tta_series", 1) > 1 or getattr(model, "union_cap", None) is not None else model.series_layout,
        anterior_shift_mm=recipe.anterior_shift_mm,
        # sub-series は uniform で採った series の 1 枚ずつずれた版なので、uniform のレシピだけに作る（contiguous の重みは候補の series だけを回す）
        subseries=(getattr(model, "tta_series", 1) > 1 or getattr(model, "union_cap", None) is not None) and recipe.long_series_sampling == "uniform",
    )


def check_series_tta_recipe(model: Any, experiment_dir: Path) -> None:
    """series の TTA は "all" でデコードした series を 5 Slot に写すので、レシピがあり、同じ撮像面の Slot の上限が等しいこと。"""
    if model.preprocess is None:
        raise ValueError(f"{experiment_dir}: tta_series needs model.preprocess (the training cache recipe)")
    caps: dict[str, set[int]] = {}
    for slot in make_slots(model.preprocess.slot_table):
        caps.setdefault(slot.plane, set()).add(slot.n_slices)
    unequal = {plane: sorted(values) for plane, values in caps.items() if len(values) > 1}
    if unequal:
        raise ValueError(f"{experiment_dir}: tta_series needs equal caps for the slots of a plane, got {unequal}")


def group_by_preprocess(models: Sequence[WindowModel]) -> list[tuple[PreprocessParams, list[WindowModel]]]:
    """fitted 時のレシピごとに束ねたメンバー。並びは初出順。

    study は束ごとに 1 度だけデコードし、束のメンバー全員がその画素を採点する。束の鍵は値である。
    """
    if not models:
        raise ValueError("no models given")
    groups: dict[PreprocessParams, list[WindowModel]] = {}
    for model in models:
        groups.setdefault(preprocess_params_for(model), []).append(model)
    return list(groups.items())


def load_models(model_dirs: Sequence[str | Path]) -> list[WindowModel]:
    """アンサンブルのメンバーを CPU の eval モードで読み、前処理レシピを確認する。"""
    models: list[WindowModel] = []
    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        model = load_model(model_dir, map_location="cpu")
        cfg = model.cfg
        params = preprocess_params_for(model)
        produced = num_slices(resolve_slots(params))
        logger.info("loaded %s (backbone=%s img_size=%d slices=%d k_eval=%s)", model_dir, cfg.backbone, cfg.img_size, produced, cfg.window.k_eval)
        models.append(model)
    if not models:
        raise ValueError("no --model-dir given")
    return models


def member_weights(model_dirs: Sequence[str | Path]) -> list[float]:
    """``<run>/<fold>`` と読んで run ごとに等しい重みを配る（合計 1）。``a/fold0 a/fold1 b/fold0`` なら ``0.25 0.25 0.5``。"""
    runs = [Path(model_dir).parts[-2] for model_dir in model_dirs]
    if not runs:
        raise ValueError("no model dirs given")
    folds = Counter(runs)
    return [1.0 / (len(folds) * folds[run]) for run in runs]


@dataclass
class Members:
    """読み込んだ member 一式と、前処理レシピごとの束 ``(レシピ, その束で採点する member の添字)``。"""

    models: list[WindowModel]
    dirs: list[Path]
    groups: list[tuple[PreprocessParams, list[int]]]
    parts: list[int]  # member ごとの part 番号。part の中は確率で平均し、part 同士は rank 平均で混ぜる

    @property
    def weights(self) -> list[float]:
        """member ごとの混合重み。run ごとに等分し、run の中では fold で等分する。合計は 1。"""
        return member_weights(self.dirs)

    @property
    def group_params(self) -> list[PreprocessParams]:
        """束ごとの前処理レシピ（:func:`rsna_knee.pipeline.iter_studies` に渡す）。"""
        return [params for params, _ in self.groups]

    @property
    def group_members(self) -> list[list[int]]:
        """束ごとの member の添字（:func:`rsna_knee.pipeline.score_stream` に渡す）。"""
        return [members for _, members in self.groups]

    def table(self) -> pd.DataFrame:
        """member ごとの backbone・画像サイズ・スライス数・採点窓数・束の番号。notebook の表示用。"""
        group_of = {m: g for g, (_, members) in enumerate(self.groups) for m in members}
        weights = self.weights
        rows = []
        for i, (model, model_dir) in enumerate(zip(self.models, self.dirs)):
            cfg = model.cfg
            rows.append(
                {
                    "run": "/".join(model_dir.parts[-2:]),
                    "backbone": cfg.backbone,
                    "img_size": cfg.img_size,
                    "slices": num_slices(resolve_slots(preprocess_params_for(model))),
                    "k_eval": cfg.window.k_eval,
                    "group": group_of[i],
                    "weight": round(weights[i], 4),
                }
            )
        return pd.DataFrame(rows)


def load_members(model_dirs: Sequence[str | Path], k_eval: Mapping[str, int] | None = None, encode_chunk: int | None = None) -> Members:
    """artifact を CPU に読み、run ごとの採点窓数を上書きし、前処理レシピごとに束ねる。

    CUDA には触らない（デコード worker の fork が先）。``k_eval`` は ``"<run>/<fold>"`` から採点窓数への表。
    """
    dirs = [Path(model_dir) for model_dir in model_dirs]
    models = load_models(dirs)
    for model, model_dir in zip(models, dirs):
        if encode_chunk is not None:
            model.encode_chunk = encode_chunk
        run = "/".join(model_dir.parts[-2:])
        if k_eval and run in k_eval:
            model.cfg.window.k_eval = k_eval[run]
    index_of = {id(model): i for i, model in enumerate(models)}
    groups = [(params, [index_of[id(model)] for model in members]) for params, members in group_by_preprocess(models)]
    return Members(models=models, dirs=dirs, groups=groups, parts=[0] * len(models))


def rank_average(probs: np.ndarray) -> np.ndarray:
    """``float[n_studies, n_parts, 12]`` を part・所見ごとに study 間の順位（同順位は平均、``(0, 1]``）へ直し、part で等しく平均する。"""
    ranks = [pd.DataFrame(probs[:, part]).rank(pct=True).to_numpy() for part in range(probs.shape[1])]
    return np.mean(ranks, axis=0)


def load_experiments(root: str | Path, ensemble: Sequence[Sequence[str]]) -> Members:
    """選択した experiment の共通設定と全 fold の重みを CPU に読み、前処理ごとに束ねる。

    ``ensemble`` の外側の 1 要素が rank 平均で混ぜる 1 part で、内側の experiment は確率で平均する。
    """
    if not ensemble or any(isinstance(part, str) or not part for part in ensemble):
        raise ValueError("Ensemble must be a non-empty list of non-empty lists of experiment names")
    names = [name for part in ensemble for name in part]
    if len(set(names)) != len(names):
        raise ValueError("Ensemble must contain distinct experiment names")
    models: list[WindowModel] = []
    dirs: list[Path] = []
    parts: list[int] = []
    for part, name in ((part, name) for part, group in enumerate(ensemble) for name in group):
        experiment_dir = Path(root) / name
        recipe = load_experiment_config(experiment_dir)
        if recipe.series_layout == "all" and recipe.model.kind == "vlm":
            raise ValueError(f"{experiment_dir}: series_layout='all' is for window models only")
        fold_dirs = sorted(path.parent for path in experiment_dir.glob("fold*/model.pt"))
        if not fold_dirs:
            raise FileNotFoundError(f"no completed folds in {experiment_dir}")
        for fold_dir in fold_dirs:
            config = Config(model=recipe.model.model_copy(deep=True))
            if config.model.kind == "vlm":
                config.model.vlm.source = fold_dir / config.model.vlm.source
            model = build_model(config, pretrained=False)
            model.load_state_dict(torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=True))
            model.eval()
            if recipe.model.kind != "vlm":
                model.encode_chunk = recipe.encode_chunk
                model.infer_dtype = {"float16": torch.float16, "float32": None}[recipe.infer_dtype]
                model.series_layout = recipe.series_layout
                model.tta_context_flip = recipe.tta_context_flip
                model.tta_series = recipe.tta_series
                model.union_cap = recipe.union_cap
                if recipe.tta_series > 1:
                    check_series_tta_recipe(recipe.model, experiment_dir)
            models.append(model)
            dirs.append(fold_dir)
            parts.append(part)
    index_of = {id(model): i for i, model in enumerate(models)}
    groups = [(params, [index_of[id(model)] for model in members]) for params, members in group_by_preprocess(models)]
    return Members(models=models, dirs=dirs, groups=groups, parts=parts)


def _progress(done: int, total: int, started: float) -> tuple[str, float]:
    """進捗の 1 行と、split 全体にかかる実時間の見積もり。"""
    elapsed = max(time.time() - started, 1e-6)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else float("inf")
    return f"[{done}/{total}] {rate:.3f} studies/s elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", elapsed + eta


def predict_split(
    data_dir: str | Path,
    split: str = "test",
    model_dirs: Sequence[str | Path] = (),
    out: str | Path = "submission.csv",
    device: str | torch.device = "auto",
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    limit: int | None = None,
    progress_every: int = 25,
    params: PreprocessParams | None = None,
    tta_hflip: bool = False,
    infer_dtype: torch.dtype | None = None,
    encode_chunk: int | None = None,
    cudnn_benchmark: bool = False,
    workers: int = DEFAULT_WORKERS,
    prefetch: int = DEFAULT_PREFETCH,
    devices: Sequence[str | torch.device] | None = None,
    backbone: str = "torch",
) -> pd.DataFrame:
    """split 1 つを端から端まで採点し、``out`` に提出を書く。

    デコードは worker process が先読みし (:func:`rsna_knee.pipeline.iter_studies`)、forward は device ごとの
    consumer thread が study 単位の round-robin で受け持つ (:func:`rsna_knee.pipeline.score_stream`)。デコー
    ドの失敗も forward の失敗もそのまま送出する（提出 notebook と同じ）。``params`` はレシピの明示的な
    上書き。
    """
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
    started = time.time()
    data_dir = Path(data_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    split_dir = data_dir / f"{split}_series"

    study_ids, series_rows = read_split(data_dir, split)
    columns = submission_columns(data_dir)
    labels = columns[1:]

    torch.backends.cudnn.benchmark = cudnn_benchmark
    models = load_models(model_dirs)
    groups = [(params, list(models))] if params is not None else group_by_preprocess(models)
    group_params = [group_params for group_params, _ in groups]
    index_of = {id(model): i for i, model in enumerate(models)}
    group_members = [[index_of[id(model)] for model in members] for _, members in groups]
    weights = member_weights(model_dirs)
    logger.info("member weights (run ごとに等分): %s", [round(w, 4) for w in weights])
    todo = study_ids[:limit] if limit else study_ids
    counters = PhaseCounters()
    # worker はここで fork する。CUDA を初期化した後の fork は子で CUDA が使えないので、device の解決と
    # モデルの配置はこの後に置く。
    stream = iter_studies([(split_dir / study_id, series_rows.get(study_id, [])) for study_id in todo], group_params, workers=workers, prefetch=prefetch, counters=counters)

    probabilities: dict[str, np.ndarray] = {}
    try:
        devices = resolve_devices(device, devices)
        banks = place_models(models, devices)
        for bank in banks:
            for model in bank.models:
                if infer_dtype is not None:
                    model.infer_dtype = infer_dtype
                if encode_chunk is not None:
                    model.encode_chunk = encode_chunk
        if backbone == "compile":
            for bank in banks:
                for model in bank.models:
                    compile_backbone(model)
        logger.info("devices=%s models=%d groups=%d studies=%d split_dir=%s workers=%d prefetch=%d", [str(d) for d in devices], len(models), len(groups), len(todo), split_dir, workers, prefetch)
        logger.info("tta_hflip=%s infer_dtype=%s encode_chunk=%s cudnn_benchmark=%s backbone=%s", tta_hflip, infer_dtype, encode_chunk, cudnn_benchmark, backbone)
        for params_of_group, group_models in groups:
            logger.info("preprocess (%d member(s)): %s", len(group_models), params_of_group)

        def decoded():
            """main thread 側: worker から届いた study を束ごとの (pixels, mask) にして consumer に渡す。"""
            for index, studies in stream:
                if isinstance(studies, Exception):
                    raise studies
                yield index, [(study.pixels, study.mask, study.series_headers, study.subseries) for study in studies]

        next_report = progress_every
        budget_warned = False
        for index, result in score_stream(decoded(), banks, group_members, tta_hflip=tta_hflip, counters=counters, weights=weights):
            if isinstance(result, Exception):
                raise result
            probabilities[todo[index]] = result
            done = len(probabilities)
            if progress_every and done >= next_report:
                next_report += progress_every
                line, projected = _progress(done, len(todo), started)
                logger.info("%s | %s", line, counters.summary(done))
                if projected > time_budget_s and not budget_warned:
                    budget_warned = True
                    logger.warning("projected wall-clock %.1fm exceeds the %.1fm budget", projected / 60, time_budget_s / 60)
    finally:
        stream.close()

    frame = pd.DataFrame({ID_COLUMN: todo})
    order = {name: i for i, name in enumerate(LABELS)}
    for name in labels:
        frame[name] = np.clip([probabilities[study_id][order[name]] for study_id in todo], PROB_LO, PROB_HI)
    frame = frame.loc[:, columns]
    frame.to_csv(out, index=False)

    elapsed = time.time() - started
    logger.info("wrote %s: %d rows, %.1fs total (%.2fs/study) | %s", out, len(frame), elapsed, elapsed / max(len(todo), 1), counters.summary(len(todo)))
    return frame


@click.command(help="Score a split's DICOM studies into a Kaggle submission.")
@click.option("--data-dir", type=click.Path(path_type=Path), required=True, help="Competition root holding {split}.csv and {split}_series/.")
@click.option("--split", default="test", type=click.Choice(["test", "train"]), help="Split to score (default: test).")
@click.option("--model-dir", "model_dirs", multiple=True, type=click.Path(path_type=Path), help="Artifact directory; repeat for an ensemble (mean of sigmoid probabilities).")
@click.option("--out", type=click.Path(path_type=Path), default=Path("submission.csv"), help="Submission path (default: submission.csv).")
@click.option("--device", default="auto", type=click.Choice(["auto", "cuda", "cpu"]), help="Device (default: auto). cuda/auto use every visible GPU.")
@click.option("--devices", default=None, help="Comma-separated devices overriding --device, e.g. cuda:0,cuda:1 (cuda:0,cuda:0 runs two consumers on one GPU).")
@click.option("--workers", type=int, default=DEFAULT_WORKERS, help=f"Decode worker processes; 0 decodes in the main process (default: {DEFAULT_WORKERS}).")
@click.option("--prefetch", type=int, default=DEFAULT_PREFETCH, help=f"Studies decoded ahead of the GPU (default: {DEFAULT_PREFETCH}).")
@click.option("--time-budget-s", type=float, default=DEFAULT_TIME_BUDGET_S, help="Expected wall-clock budget in seconds (default: 8h).")
@click.option("--limit", type=int, default=None, help="Score only the first N studies (debug); the submission then covers only those.")
@click.option("--progress-every", type=int, default=25, help="Progress line every N studies (default: 25).")
@click.option("--tta-hflip", is_flag=True, help="Average each study with its horizontally flipped self (two forward passes per model).")
@click.option("--infer-dtype", default="none", type=click.Choice(sorted(INFER_DTYPES)), help="Backbone dtype on CUDA: none = fp32 as fitted (default); float16 = fp16 autocast for the backbone, head stays fp32.")
@click.option("--encode-chunk", type=int, default=None, help="Windows per backbone pass at inference (default: the model's own default).")
@click.option("--cudnn-benchmark", is_flag=True, help="Set torch.backends.cudnn.benchmark (algorithm choice becomes timing-based, so runs differ by ~4e-5).")
@click.option("--backbone", default="torch", type=click.Choice(BACKBONES), help="torch = the PyTorch backbone (default); compile = torch.compile.")
def main(
    data_dir: Path,
    split: str,
    model_dirs: tuple[Path, ...],
    out: Path,
    device: str,
    devices: str | None,
    workers: int,
    prefetch: int,
    time_budget_s: float,
    limit: int | None,
    progress_every: int,
    tta_hflip: bool,
    infer_dtype: str,
    encode_chunk: int | None,
    cudnn_benchmark: bool,
    backbone: str,
) -> None:
    """Score ``--split`` with the ``--model-dir`` ensemble and write the submission CSV."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout, force=True)

    if not model_dirs:
        logger.error("at least one --model-dir is required")
        raise SystemExit(2)

    predict_split(
        data_dir=data_dir,
        split=split,
        model_dirs=list(model_dirs),
        out=out,
        device=device,
        time_budget_s=time_budget_s,
        limit=limit,
        progress_every=progress_every,
        tta_hflip=tta_hflip,
        infer_dtype=INFER_DTYPES[infer_dtype],
        encode_chunk=encode_chunk,
        cudnn_benchmark=cudnn_benchmark,
        workers=workers,
        prefetch=prefetch,
        devices=devices.split(",") if devices else None,
        backbone=backbone,
    )


if __name__ == "__main__":
    main()
