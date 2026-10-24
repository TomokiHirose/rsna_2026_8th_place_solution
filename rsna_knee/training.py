from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import click
import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger
from torch.utils.data import DataLoader, Dataset

from rsna_knee.config import Config, PreprocessConfig
from rsna_knee.constants import ID_COLUMN, LABELS, NUM_LABELS
from rsna_knee.data.cache_dataset import CACHE_IDS, CACHE_VOLS, build_uid_index, read_cache_manifest
from rsna_knee.experiments import export_experiment_config
from rsna_knee.inference import CONFIG_FILENAME, MODEL_FILENAME, predict_study
from rsna_knee.models import WindowModel
from rsna_knee.models.factory import build_model
from rsna_knee.models.multibranch import BRANCH_NAMES, MultiBranchModel
from rsna_knee.modules import KneeDataModule, collate_batch, model_inputs
from rsna_knee.modules.labels import binarize, gold_ids, load_labels, load_severity, load_weights
from rsna_knee.modules.lit_window import EMA_PREFIX, MODEL_PREFIX, LitWindowModel, export_weights_from_checkpoint
from rsna_knee.modules.losses import pos_weight_from_targets
from rsna_knee.modules.metrics import macro_auc
from rsna_knee.modules.window_dataset import WindowStudyDataset, load_subcase_sides, load_subcases
from rsna_knee.utils import get_logger

logger = get_logger(__name__)


def read_folds(folds_csv: Path) -> pd.DataFrame:
    """``tools/make_folds.py`` が書いた ``StudyInstanceUID,fold`` の表を読む。"""
    if not Path(folds_csv).exists():
        raise FileNotFoundError(f"{folds_csv} not found; build it with tools/make_folds.py")
    folds = pd.read_csv(folds_csv)
    missing = [c for c in (ID_COLUMN, "fold") if c not in folds.columns]
    if missing:
        raise ValueError(f"{folds_csv}: missing columns {missing}")
    return folds


def split_study_ids(folds: pd.DataFrame, fold: int, available: set[str], limit: int | None = None) -> tuple[list[str], list[str]]:
    """fold の表を ``(train_ids, val_ids)`` に分け、``available`` にある study だけ残す。``limit`` は両方を切り詰める。"""
    if fold not in set(folds["fold"]):
        raise ValueError(f"fold {fold} is not in {sorted(set(folds['fold']))}")
    keep = folds[folds[ID_COLUMN].isin(available)]
    train_ids = keep.loc[keep["fold"] != fold, ID_COLUMN].tolist()
    val_ids = keep.loc[keep["fold"] == fold, ID_COLUMN].tolist()
    if limit is not None:
        train_ids = train_ids[:limit]
        val_ids = val_ids[: max(limit // 4, 2)]
    return train_ids, val_ids


def build_datasets(cfg: Config, fold: int, limit: int | None = None) -> tuple[Dataset, Dataset]:
    """``cfg.data.cache_dir`` の上に立つ ``(train, val)`` の study データセット。

    どちらも :mod:`rsna_knee.modules.datamodule` のバッチ契約を返す。protocol は ``cfg.data.split`` が選ぶ。
    ``fold`` は検証に回す fold で、``split="gold"`` のときは無視される。``limit`` は split ごとの study 数の上限。
    """
    if cfg.data.cache_dir is None:
        raise ValueError("training needs data.cache_dir (a study cache built by tools/build_cache.py)")
    cache_dir = Path(cfg.data.cache_dir)
    manifest = read_cache_manifest(cache_dir)
    _adopt_cache_recipe(cfg, manifest, cache_dir)

    labels = load_labels(cfg.data.labels_csv, cfg.data.train_csv, unknown_to_nan=cfg.data.unknown_to_nan, unknown_value=cfg.data.unknown_value)
    available = set(build_uid_index([(cache_dir / CACHE_VOLS, cache_dir / CACHE_IDS)]))
    subcases = None if cfg.data.subcases_csv is None else load_subcases(cfg.data.subcases_csv)
    check_subcase_labels(labels, subcases, available)
    if subcases:
        # 親は子に置き換わる。親の行がラベルにあっても学習・検証には入らない。
        parents = {parent for parent, _ in subcases.values()}
        if parents & gold_ids(cfg.data.train_csv):
            raise ValueError("a gold study cannot be split into sub-cases")
        available = replace_parents(available, subcases)
        logger.info("sub-cases: %d parents replaced by %d knee-level studies", len(parents), len(subcases))
    if cfg.data.split == "all":
        train_ids, val_ids = all_split(labels, cfg.data.train_csv, available)
    elif cfg.data.split == "gold":
        train_ids, val_ids = gold_split(labels, cfg.data.train_csv, available, holdout=holdout_ids(cfg, available, subcases))
    else:
        train_ids, val_ids = split_study_ids(_folds_with_subcases(read_folds(cfg.data.folds_csv), subcases), fold, available)
    if limit is not None:
        train_ids, val_ids = train_ids[:limit], val_ids[: max(limit // 4, 2)]

    common = dict(labels=labels, window=cfg.model.window, img_size=cfg.model.img_size, manifest=manifest)
    if subcases:
        if cfg.model.kind == "vlm":
            raise ValueError("data.subcases_csv is not supported for the vlm model")
        common["subcases"] = subcases
        common["subcase_sides"] = load_subcase_sides(cfg.data.subcases_csv)
    aug = cfg.data.aug
    aug_kwargs = dict(brightness=aug.brightness_gain, translate_px=aug.translate_px, scale_range=aug.scale_range, scale_prob=aug.scale_prob, depth_flip=aug.depth_flip, window_aug=aug)
    dataset_class = WindowStudyDataset
    if cfg.model.kind == "vlm":
        from rsna_knee.modules.video_dataset import VideoStudyDataset

        dataset_class = VideoStudyDataset
        common["model"] = cfg.model
    if cfg.model.series_neck.enabled and cfg.data.union_bag is None:
        raise ValueError("model.series_neck needs data.union_bag (it reads the series and depth of every window)")
    if cfg.data.union_bag is not None:
        if cfg.model.kind != "window":
            raise ValueError(f"data.union_bag needs model.kind=window, got {cfg.model.kind}")
        union = dict(depth_prior=cfg.data.depth_prior, subseries=cfg.data.subseries, union_bag=cfg.data.union_bag, **common)
        # 検証は全版の全窓（eval_cap で頭打ち）で採る。
        train = WindowStudyDataset(cache_dir, train_ids, is_train=True, weights=cell_weights(cfg, train_ids), severity=study_severity(cfg, train_ids), **aug_kwargs, **union)
        return train, WindowStudyDataset(cache_dir, val_ids, is_train=False, severity=study_severity(cfg, val_ids), **union)
    sub_kwargs = dict(subseries=True) if cfg.data.subseries else {}
    train = dataset_class(cache_dir, train_ids, is_train=True, weights=cell_weights(cfg, train_ids), severity=study_severity(cfg, train_ids), depth_prior=cfg.data.depth_prior, **aug_kwargs, **sub_kwargs, **common)
    # 検証と OOF の重みは 1 のまま。学習 split の損失の形を、epoch を順位づける数字に持ち込まない。重症度は検証の補助損失と順位相関のために渡す。
    return train, dataset_class(cache_dir, val_ids, is_train=False, severity=study_severity(cfg, val_ids), **common)


def check_subcase_labels(labels: pd.DataFrame, subcases: dict[str, tuple[str, frozenset[str]]] | None, cache_ids: set[str]) -> None:
    """膝ごとの子 study のラベルと ``data.subcases_csv`` の食い違いを、黙って検査が消える前に止める。

    表があるのにラベルに子の行が無い（親のラベルを渡した）と、親は外れて子にはラベルが無く、その検査が学習から消える。
    逆に表が無いのにラベルに子の行がある（膝分割版のラベルを渡した）と、親の行が無く子は cache に無いので、やはり消える。
    """
    if subcases:
        absent = sorted(child for child, (parent, _) in subcases.items() if parent in cache_ids and child not in labels.index)
        if absent:
            raise ValueError(f"data.labels_csv has no rows for {len(absent)} sub-cases (e.g. {absent[:2]}); use a knee-level labels file (…_knees / v6_knees / v7_knees)")
    else:
        stray = sorted(uid for uid in labels.index.astype(str) if uid not in cache_ids and uid.rsplit("-", 1)[0] in cache_ids and uid.rsplit("-", 1)[-1].isdigit())
        if stray:
            raise ValueError(f"data.labels_csv has {len(stray)} knee-level ids (e.g. {stray[:2]}) that are not in the cache; set data.subcases_csv to the bilateral_knees/knee_series.csv")


def study_severity(cfg: Config, study_ids: list[str]) -> np.ndarray | None:
    """``data.severity_csv`` の重症度 ``[N, 12]``。表を指定していなければ ``None``。"""
    return None if cfg.data.severity_csv is None else load_severity(cfg.data.severity_csv, study_ids)


def cell_weights(cfg: Config, study_ids: list[str]) -> np.ndarray | None:
    """学習 split のセルごとの損失係数 ``[N, 12]``。``data.weights_csv`` の重みに、gold でない study の ``train.pseudo_weight`` を掛ける。

    ``data.weights_csv`` が無く ``train.pseudo_weight`` が 1 なら、重みそのものを持たない ``None`` を返す。
    """
    pseudo = float(cfg.train.pseudo_weight)
    if cfg.data.weights_csv is None and pseudo == 1.0:
        return None
    weights = np.ones((len(study_ids), NUM_LABELS), dtype=np.float32) if cfg.data.weights_csv is None else load_weights(cfg.data.weights_csv, study_ids)
    gold = gold_ids(cfg.data.train_csv)
    pseudo_rows = np.array([str(uid) not in gold for uid in study_ids], dtype=bool)
    weights[pseudo_rows] *= np.float32(pseudo)
    logger.info("cell weights: %d of %d training studies are not gold and take pseudo_weight=%.3g (mean weight %.3f)", int(pseudo_rows.sum()), len(study_ids), pseudo, float(weights.mean()))
    return weights


def _adopt_cache_recipe(cfg: Config, manifest: dict, cache_dir: Path) -> None:
    """cache 自身の DICOM -> 画素レシピを ``cfg.model.preprocess`` に写す。

    Stack の正本は cache であり、そのレシピは重みと一緒に運ばれなければならない。提出時に DICOM を
    この重みが期待する画素へ変える方法を notebook に教えられるものが他に無いからである。
    """
    table = manifest["slot_table"]
    n_slices = int(manifest["n_slices"])
    cfg.model.preprocess = PreprocessConfig(
        crop_mm=float(manifest["crop_mm"]),
        size=int(manifest["size"]),
        span=tuple(manifest["span"]),
        percentiles=tuple(manifest["percentiles"]),
        slot_table=list(table),
        native_spacing=bool(manifest.get("native_spacing", False)),
        resample=str(manifest.get("resample", "area")),
        long_series_sampling=manifest["long_series_sampling"],
        crop_pad=bool(manifest.get("crop_pad", False)),
        histogram_standard=manifest.get("histogram_standard"),
        anterior_shift_mm=float(manifest.get("anterior_shift_mm", 0.0)),
    )
    logger.info(
        "cache %s: slots=%s slot_table=%s n_slices=%d size=%s span=%s native_spacing=%s resample=%s histogram_standard=%s",
        cache_dir,
        manifest.get("slots"),
        table,
        n_slices,
        manifest.get("size"),
        manifest.get("span"),
        manifest.get("native_spacing"),
        cfg.model.preprocess.resample,
        cfg.model.preprocess.histogram_standard is not None,
    )


def all_split(labels: pd.DataFrame, train_csv: Path, available: set[str]) -> tuple[list[str], list[str]]:
    """検証を持たない split。ラベルのある study をすべて学習に回す。

    epoch は最後まで回して最後の重みを書き出す（best を選ぶ検証が存在しない）。返す val 側は artifact を
    書き出すためだけの gold 58 で、学習にも入っている。だから数字は ``insample.csv`` に書く。
    """
    if not Path(train_csv).exists():
        raise FileNotFoundError(f"{train_csv} not found; the all split needs the official train.csv for the artifact loader")
    gold = pd.read_csv(train_csv, usecols=lambda c: c in {ID_COLUMN, *LABELS})
    gold[ID_COLUMN] = gold[ID_COLUMN].astype(str)
    gold = gold[gold[LABELS].notna().all(axis=1)]
    train_ids = [uid for uid in labels.index.astype(str) if uid in available]
    val_ids = [uid for uid in gold[ID_COLUMN] if uid in available]
    logger.info("all split: train=%d val=%d (val は artifact を書くためだけの gold で、学習にも入っている)", len(train_ids), len(val_ids))
    if not train_ids or not val_ids:
        raise ValueError(f"empty all split: train={len(train_ids)} val={len(val_ids)}")
    return train_ids, val_ids


def gold_split(labels: pd.DataFrame, train_csv: Path, available: set[str], holdout: set[str] | None = None) -> tuple[list[str], list[str]]:
    """gold study（``train.csv`` が 0/1 ラベルを持つ ~58 件）が検証に回り、それ以外はすべて学習に回る split。

    この検証値は fold CV とは比較できない。fold の study はすべて学習側に入っているからである。
    """
    if not Path(train_csv).exists():
        raise FileNotFoundError(f"{train_csv} not found; the gold split needs the official train.csv")
    gold = pd.read_csv(train_csv, usecols=lambda c: c in {ID_COLUMN, *LABELS})
    gold[ID_COLUMN] = gold[ID_COLUMN].astype(str)
    gold = gold[gold[LABELS].notna().all(axis=1)]
    val_ids = [uid for uid in gold[ID_COLUMN] if uid in available]
    excluded = set(val_ids) | (holdout or set())
    train_ids = [uid for uid in labels.index.astype(str) if uid in available and uid not in excluded]
    logger.info("gold split: train=%d val=%d (gold rows=%d, held out=%d)", len(train_ids), len(val_ids), len(gold), len(holdout or ()))
    if not train_ids or not val_ids:
        raise ValueError(f"empty gold split: train={len(train_ids)} val={len(val_ids)}")
    return train_ids, val_ids


def _window_oof_loader(cfg: Config, val_loader: DataLoader) -> DataLoader:
    """検証と同じ study を、採点用の bag のサイズと batch 1 でもう一度。"""
    dataset = val_loader.dataset
    train_labels = load_labels(cfg.data.labels_csv, cfg.data.train_csv, unknown_to_nan=cfg.data.unknown_to_nan, unknown_value=cfg.data.unknown_value)
    if cfg.data.holdout_fold is None:
        study_ids, labels = dataset.study_ids, _eval_labels(cfg, train_labels)
    else:
        # fold を 1 つ外して持つ意味は、それを採点することにある。
        available = replace_parents(set(build_uid_index([(dataset.cache_dir / CACHE_VOLS, dataset.cache_dir / CACHE_IDS)])), dataset.subcases or None)
        study_ids = sorted(holdout_ids(cfg, available, dataset.subcases or None))
        labels = _eval_labels(cfg, train_labels)
        logger.info("OOF on the held-out fold: %d studies", len(study_ids))
    dataset_class = WindowStudyDataset
    extra = {"subcases": dataset.subcases} if getattr(dataset, "subcases", None) else {}
    if cfg.model.kind == "vlm":
        from rsna_knee.modules.video_dataset import VideoStudyDataset

        dataset_class = VideoStudyDataset
        extra["model"] = cfg.model
    if cfg.model.series_neck.enabled:
        # series_neck は全 series の Bag でしか採点できないので、採点も評価モードの union bag（eval_cap）で行う。
        extra = {"subcases": dataset.subcases or None, "subcase_sides": dataset.subcase_sides or None}
        scored = dataset_class(dataset.cache_dir, study_ids, **extra, labels=labels, window=cfg.model.window, img_size=cfg.model.img_size, manifest=dataset.manifest, depth_prior=cfg.data.depth_prior, union_bag=cfg.data.union_bag)
    else:
        scored = dataset_class(
            dataset.cache_dir, study_ids, **extra, labels=labels, window=cfg.model.window, img_size=cfg.model.img_size, is_train=False, k=cfg.model.window.k_eval, all_windows=cfg.model.window.k_eval is None, manifest=dataset.manifest
        )
    return DataLoader(scored, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=collate_batch)


def replace_parents(cache_ids: set[str], subcases: dict[str, tuple[str, frozenset[str]]] | None) -> set[str]:
    """cache の study ID のうち、表に出た親を外し、親が cache にある子を足す（学習・検証・OOF で同じ study の集合を使う）。"""
    if not subcases:
        return cache_ids
    parents = {parent for parent, _ in subcases.values()}
    return (cache_ids - parents) | {child for child, (parent, _) in subcases.items() if parent in cache_ids}


def _folds_with_subcases(folds: pd.DataFrame, subcases: dict[str, tuple[str, frozenset[str]]] | None) -> pd.DataFrame:
    """fold 表に子の study を足す。子は親と同じ fold に置く（同じ検査の左右が学習と検証に分かれないようにする）。"""
    if not subcases:
        return folds
    parent_fold = folds.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)["fold"]
    rows = [{ID_COLUMN: child, "fold": parent_fold[parent]} for child, (parent, _) in subcases.items() if parent in parent_fold.index]
    return pd.concat([folds, pd.DataFrame(rows)], ignore_index=True)


def holdout_ids(cfg: Config, available: set[str], subcases: dict[str, tuple[str, frozenset[str]]] | None = None) -> set[str]:
    """``data.holdout_fold`` が学習から外す study（未設定なら空）。"""
    if cfg.data.holdout_fold is None:
        return set()
    folds = _folds_with_subcases(read_folds(cfg.data.folds_csv), subcases)
    held = {uid for uid in folds.loc[folds["fold"] == cfg.data.holdout_fold, ID_COLUMN].astype(str) if uid in available}
    logger.info("holding fold %d out of training: %d studies", cfg.data.holdout_fold, len(held))
    return held


def _eval_labels(cfg: Config, fallback: pd.DataFrame) -> pd.DataFrame:
    """OOF を採点する相手のラベル。``data.eval_labels_csv`` があればそれで上書きする。"""
    if cfg.data.eval_labels_csv is None:
        return fallback
    logger.info("scoring the OOF against %s", cfg.data.eval_labels_csv)
    return load_labels(cfg.data.eval_labels_csv, cfg.data.train_csv, unknown_to_nan=cfg.data.unknown_to_nan)


@torch.inference_mode()
def _window_oof(model: WindowModel | MultiBranchModel, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Study ごとの予測確率と評価ラベルを OOF のフレームにまとめる。multibranch は混合後に加えて head ごとの確率も並べる。"""
    model = model.to(device).eval()
    multibranch = isinstance(model, MultiBranchModel)
    study_ids: list[str] = []
    probs: list[torch.Tensor] = []
    heads: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        inputs = model_inputs(batch, device)
        if multibranch:
            logits_all = model.forward_batch_branches(inputs)
            probs.append(torch.sigmoid(model.mix(logits_all).float()).cpu())
            heads.append(torch.sigmoid(logits_all.float()).cpu())
        else:
            probs.append(torch.sigmoid(model.forward_batch(inputs).float()).cpu())
        labels.append(batch["label"].float().cpu())
        study_ids.extend(batch["study_id"])
    head_probs = {name: torch.cat([h[i] for h in heads]) for i, name in enumerate(BRANCH_NAMES)} if multibranch else {}
    return _oof_frame(study_ids, torch.cat(probs), torch.cat(labels), head_probs)


def _oof_frame(study_ids: list[str], prob: torch.Tensor, label: torch.Tensor, head_probs: dict[str, torch.Tensor] | None = None) -> pd.DataFrame:
    """``study_id`` + ``pred_<label>`` + head ごとの ``pred_<head>_<label>`` + ``true_<label>`` の列を LABELS の順に並べる。"""
    oof = pd.DataFrame({"study_id": study_ids})
    for i, name in enumerate(LABELS):
        oof[f"pred_{name}"] = prob[:, i].numpy()
    for head, head_prob in (head_probs or {}).items():
        for i, name in enumerate(LABELS):
            oof[f"pred_{head}_{name}"] = head_prob[:, i].numpy()
    for i, name in enumerate(LABELS):
        oof[f"true_{name}"] = label[:, i].numpy()
    return oof


@torch.inference_mode()
def _all_series_oof(model: WindowModel, dataset: WindowStudyDataset, device: torch.device) -> pd.DataFrame:
    """全 series を平坦化した Stack を、提出と同じ ``predict_study(series_layout="all")`` で study ごとに採点する。"""
    model = model.to(device).eval()
    model.series_layout = "all"
    vols = np.load(dataset.vols_path, mmap_mode="r")
    probs: list[torch.Tensor] = []
    for uid in dataset.study_ids:
        _, row = dataset.uid_index[dataset.cache_uid[uid]]
        mask = np.asarray(dataset.masks[dataset.stack_offsets[row] : dataset.stack_offsets[row + 1]])
        # cache は実在スライスだけを詰めているので、Stack の行に戻してから推論の入口に渡す。
        stack = np.zeros((len(mask), dataset.img_size, dataset.img_size), dtype=np.uint8)
        stack[mask != 0] = vols[dataset.offsets[row] : dataset.offsets[row + 1]]
        # 提出の推論が series_headers で受け取る行範囲と撮像面・液体強調フラグを、cache の series 表から同じ形で渡す。
        units = dataset.units.get(row, [])
        if uid in dataset.subcase_units:
            # 子は自分の膝の series だけで採点する。"all" の推論は Stack 全体から窓を切るので、他の膝の行は実在から外す。
            units = dataset.subcase_units[uid]
            keep = np.zeros(len(mask), dtype=bool)
            for u in units:
                keep[u["stack_offset"] : u["stack_offset"] + u["cap"]] = True
            mask = np.where(keep, mask, 0)
        headers = [{"SeriesInstanceUID": u["SeriesInstanceUID"], "stack_start": u["stack_offset"], "stack_stop": u["stack_offset"] + u["cap"], "Anatomical_Plane": u["Anatomical_Plane"], "Fluid_Sensitive": u["Fluid_Sensitive"]} for u in units]
        probs.append(predict_study(model, stack, mask, device=device, series_headers=headers))
    model.series_layout = "slots"
    return _oof_frame(list(dataset.study_ids), torch.stack(probs), torch.from_numpy(dataset.label_array))


def _union_oof(cfg: Config, model: WindowModel, scored: WindowStudyDataset, device: torch.device) -> pd.DataFrame:
    """``scored`` と同じ study とラベルを、cache の画素から評価モードの union bag で採点する。"""
    dataset = WindowStudyDataset(
        scored.cache_dir,
        scored.study_ids,
        labels=pd.DataFrame(scored.label_array, index=scored.study_ids, columns=LABELS),
        window=cfg.model.window,
        img_size=cfg.model.img_size,
        manifest=scored.manifest,
        depth_prior=cfg.data.depth_prior,
        union_bag=cfg.data.union_bag,
        subcases=scored.subcases or None,
        subcase_sides=scored.subcase_sides or None,
    )
    return _window_oof(model, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=collate_batch), device)


def _log_oof_scores(name: str, frames: dict[str, pd.DataFrame]) -> None:
    """series layout ごとの予測を同じラベル（検証と同じく二値化）で採点し、macro と所見別の AUC を横に並べて出す。"""
    scores = {layout: macro_auc(binarize(frame[[f"true_{c}" for c in LABELS]].to_numpy()), frame[[f"pred_{c}" for c in LABELS]].to_numpy()) for layout, frame in frames.items()}
    layouts = list(scores)
    logger.info("%s macro AUC: %s", name, " | ".join(f"{layout} {scores[layout][0]:.4f}" for layout in layouts))
    for label in LABELS:
        logger.info("  %-17s %s", label, " | ".join(f"{layout} {scores[layout][1][label]:.4f}" for layout in layouts))


#: artifact の隣に書く検証予測と学習内予測のファイル名。``*_all`` は全 series を平坦化した推論（``series_layout="all"``）の予測。
OOF_FILENAME = "oof.csv"
OOF_ALL_FILENAME = "oof_all.csv"
INSAMPLE_FILENAME = "insample.csv"
INSAMPLE_ALL_FILENAME = "insample_all.csv"
#: union bag（全 series × 全版の窓を ``data.union_bag.eval_cap`` で頭打ち）の予測。
OOF_UNION_FILENAME = "oof_union.csv"
INSAMPLE_UNION_FILENAME = "insample_union.csv"
#: 最終 epoch の重みの全窓 OOF。export した best の ``oof.csv`` と並べて比べる。EMA を持つ run では EMA の重みで、``*_raw`` は学習中の重み。
LAST_OOF_FILENAME = "oof_last.csv"
LAST_RAW_OOF_FILENAME = "oof_last_raw.csv"


def export_artifact(cfg: Config, fold: int, state_dict: dict, val_loader: DataLoader, device: torch.device) -> Path:
    """重み・設定・検証予測を書く。``split="all"`` の予測は ``insample.csv`` に保存する。"""
    experiment_dir = cfg.output_dir / cfg.experiment_name
    saved_cfg = cfg.model_copy(deep=True)
    if cfg.model.kind == "vlm":
        saved_cfg.model.vlm.source = Path("qwen")
    export_experiment_config(experiment_dir, saved_cfg.model)
    artifact_dir = experiment_dir / f"fold{fold}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, pretrained=False)
    model.load_state_dict(state_dict)
    if cfg.model.kind == "vlm":
        model.save_assets(artifact_dir / "qwen")
    torch.save(model.state_dict(), artifact_dir / MODEL_FILENAME)
    (artifact_dir / CONFIG_FILENAME).write_text(json.dumps(saved_cfg.model_dump(mode="json"), indent=2, ensure_ascii=False))

    # k_eval の全部入りの bag で、forward 1 回につき study 1 件で採点する（提出 notebook と同じ形）。
    oof_loader = _window_oof_loader(cfg, val_loader)
    oof = _window_oof(model, oof_loader, device)
    # split="all" の予測は学習に入った study のものなので、OOF の名前で置いてはいけない。
    plain, plain_all = (INSAMPLE_FILENAME, INSAMPLE_ALL_FILENAME) if cfg.data.split == "all" else (OOF_FILENAME, OOF_ALL_FILENAME)
    oof.to_csv(artifact_dir / plain, index=False)
    neck = cfg.model.series_neck.enabled
    frames = {"union" if neck else "slots": oof}
    # series レイアウトの cache は全 series の Stack を持つので、series_layout="all" の推論でも同じ study を採点して並べる。
    # series_neck の採点はもともと union bag（上の oof）なので、series_layout="all" と union bag の採点は重ねない。
    if cfg.model.kind != "vlm" and oof_loader.dataset.series_layout == "all" and not neck:
        frames["all"] = _all_series_oof(model, oof_loader.dataset, device)
        frames["all"].to_csv(artifact_dir / plain_all, index=False)
    # union bag で学習した run は、cache の画素から評価モードの union bag（eval_cap）でも同じ study を採点して並べる。
    if cfg.data.union_bag is not None and not neck:
        frames["union"] = _union_oof(cfg, model, oof_loader.dataset, device)
        frames["union"].to_csv(artifact_dir / (INSAMPLE_UNION_FILENAME if cfg.data.split == "all" else OOF_UNION_FILENAME), index=False)
    _log_oof_scores(Path(plain).stem, frames)
    logger.info("Wrote inference artifact to %s", artifact_dir)
    return artifact_dir


class EpochWeights(Callback):
    """毎 epoch の終わりに ``model.load_state_dict`` にそのまま渡せる重みを ``epoch{NN}.pt``、EMA があればその重みを ``epoch{NN}_ema.pt`` として保存する。"""

    def __init__(self, dirpath: Path) -> None:
        self.dirpath = dirpath

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.export_state_dict(), self.dirpath / f"epoch{trainer.current_epoch:02d}.pt")
        if pl_module.ema is not None:
            torch.save(pl_module.export_state_dict(ema=True), self.dirpath / f"epoch{trainer.current_epoch:02d}_ema.pt")


class ProgressLog(Callback):
    """学習と検証の進み・速度・残り時間を ``every`` batch ごとにログへ 1 行書く。"""

    def __init__(self, every: int = 50) -> None:
        self.every = every

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.start = time.monotonic()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        done = batch_idx + 1
        total = trainer.num_training_batches
        if done % self.every and done != total:
            return
        elapsed = time.monotonic() - self.start
        per = elapsed / done
        epoch_left = per * (total - done)
        run_left = epoch_left + per * total * (trainer.max_epochs - trainer.current_epoch - 1)
        loss = trainer.callback_metrics.get("train/loss_step")
        loss_text = "nan" if loss is None else f"{float(loss):.4f}"
        grad_norm = trainer.callback_metrics.get("train/grad_norm")
        grad_text = "nan" if grad_norm is None else f"{float(grad_norm):.3f}"
        logger.info(
            "train ep %d/%d batch %d/%d (%.1f%%) elapsed %.1fm %.2fs/batch epoch_left %.1fm run_left(train) %.1fh loss %s grad_norm %s",
            trainer.current_epoch,
            trainer.max_epochs,
            done,
            total,
            100 * done / total,
            elapsed / 60,
            per,
            epoch_left / 60,
            run_left / 3600,
            loss_text,
            grad_text,
        )

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.val_start = time.monotonic()

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        done = batch_idx + 1
        total = trainer.num_val_batches[dataloader_idx]
        if done % self.every and done != total:
            return
        elapsed = time.monotonic() - self.val_start
        logger.info("val ep %d batch %d/%d (%.1f%%) elapsed %.1fm left %.1fm", trainer.current_epoch, done, total, 100 * done / total, elapsed / 60, elapsed / done * (total - done) / 60)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        auc = trainer.callback_metrics.get("val/auc_macro")
        logger.info("epoch %d done elapsed %.1fm val/auc_macro %s", trainer.current_epoch, (time.monotonic() - self.start) / 60, "nan" if auc is None else f"{float(auc):.4f}")


def build_loggers(cfg: Config, fold: int) -> list[Logger]:
    """メトリクスは常にディスク上の CSV に残し、W&B は ``wandb_mode`` が有効なときだけ足す。"""
    loggers: list[Logger] = [CSVLogger(save_dir=str(cfg.output_dir / cfg.experiment_name), name=f"fold{fold}")]
    if cfg.wandb_mode != "disabled":
        from lightning.pytorch.loggers import WandbLogger

        loggers.append(WandbLogger(project=cfg.wandb_project, name=f"{cfg.experiment_name}-fold{fold}", config=cfg.model_dump(mode="json")))
    return loggers


def run(cfg: Config, fold: int, fast_dev_run: bool = False, limit: int | None = None) -> None:
    """fold を 1 つ学習し、素の推論 artifact を export する。"""
    os.environ["WANDB_MODE"] = cfg.wandb_mode
    # 24 GB のカードで batch 4 × 24 窓を通すと断片化で OOM するので、allocator は常に expandable_segments で動かす（CUDA 初期化前に設定する）。
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # workers=True で DataLoader の worker ごとの乱数も seed から決める。
    L.seed_everything(cfg.seed, workers=True)
    logger.info("Experiment: %s | fold=%d | limit=%s", cfg.experiment_name, fold, limit)

    train_dataset, val_dataset = build_datasets(cfg, fold, limit=limit)
    datamodule = KneeDataModule(cfg, train_dataset, val_dataset)

    # 有病率は学習 split の性質なので、設定するのではなく split から計算する。
    targets = getattr(train_dataset, "label_array", None)
    pos_weight = None if targets is None else pos_weight_from_targets(targets)
    logger.info("pos_weight: %s", None if pos_weight is None else [round(float(v), 2) for v in pos_weight])
    lit_model = LitWindowModel(cfg, pos_weight=pos_weight)
    if cfg.train.init_weights is not None:
        loaded = torch.load(cfg.train.init_weights, map_location="cpu", weights_only=True)
        # Lightning checkpoint は ``state_dict`` に ``model.`` 接頭辞つきで、artifact の ``model.pt`` はモデルの state dict そのもの。
        weights = export_weights_from_checkpoint(loaded["state_dict"]) if "state_dict" in loaded else loaded
        if cfg.train.init_weights_skip:
            skipped = [key for key in weights if key.startswith(tuple(cfg.train.init_weights_skip))]
            if not skipped:
                raise ValueError(f"train.init_weights_skip {cfg.train.init_weights_skip} matches no weight in {cfg.train.init_weights}")
            weights = {key: value for key, value in weights.items() if key not in skipped}
            logger.info("Skipping %d weights of %s (kept at their initial values): %s", len(skipped), cfg.train.init_weights, skipped)
        result = lit_model.model.load_state_dict(weights, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"{cfg.train.init_weights}: {len(result.unexpected_keys)} weights are not in the model, e.g. {result.unexpected_keys[:5]}")
        if result.missing_keys:
            logger.warning("%s lacks %d weights, left at their initial values: %s", cfg.train.init_weights, len(result.missing_keys), result.missing_keys)
        # 文脈付き head が読み込み元に無ければ、局所 head の値（``init_weights_skip`` で飛ばしたなら初期値）から始める。
        if cfg.model.slot_context.enabled and "ctxW" in result.missing_keys:
            lit_model.model.copy_head_to_context()
            logger.info("Initialized ctxW / ctxb from clsW / clsb of %s", cfg.train.init_weights)
        logger.info("Initialized model weights from %s", cfg.train.init_weights)

    ckpt_dir = cfg.output_dir / cfg.experiment_name / f"fold{fold}" / "checkpoints"
    # split="all" は検証を持たない。best を選ぶ根拠が無いので checkpoint を置かず、最後の epoch の重みを書き出す。
    checkpoint = None if cfg.data.split == "all" else ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor="val/auc_macro", mode="max", save_top_k=1)
    # 最終 epoch の重みは monitor 無しの checkpoint で毎 epoch 上書きして残す（save_last は best を更新した epoch にしか書かない）。
    last = None if checkpoint is None else ModelCheckpoint(dirpath=ckpt_dir, filename="last", save_top_k=1, enable_version_counter=False)
    callbacks: list[Any] = [LearningRateMonitor(logging_interval="step"), ProgressLog()] + ([] if checkpoint is None else [checkpoint, last]) + ([EpochWeights(ckpt_dir)] if cfg.train.save_every_epoch else [])

    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=callbacks,
        logger=build_loggers(cfg, fold),
        # 同じ seed の run は同じ重みに着地させる。cuDNN の benchmark は切れ、cuBLAS は CUBLAS_WORKSPACE_CONFIG=:4096:8 で固定される。
        deterministic=True,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(lit_model, datamodule=datamodule)

    if checkpoint is not None and checkpoint.best_model_path:
        weights = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=True)["state_dict"]
        # EMA を持つ run は、検証と checkpoint の選択に使った EMA の重みを書き出す。
        state_dict = export_weights_from_checkpoint(weights, MODEL_PREFIX if cfg.train.ema_decay is None else EMA_PREFIX)
        # pos_weight は損失を形づくっただけでアーキテクチャではないので、モデルの state dict に混ぜない。
        state_dict.pop("pos_weight", None)
        logger.info("Best checkpoint: %s (val/auc_macro=%s)", checkpoint.best_model_path, checkpoint.best_model_score)
    else:  # split="all"、fast_dev_run、または検証が一度も走らなかった場合
        state_dict = lit_model.export_state_dict(ema=cfg.train.ema_decay is not None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = export_artifact(cfg, fold, state_dict, datamodule.val_dataloader(), device)
    scored = {"insample": INSAMPLE_FILENAME} if cfg.data.split == "all" else {"oof": OOF_FILENAME}
    # export_artifact が series_layout="all" の推論でも採点していれば、それも並べる。
    plain_all = INSAMPLE_ALL_FILENAME if cfg.data.split == "all" else OOF_ALL_FILENAME
    if (artifact_dir / plain_all).exists():
        scored[Path(plain_all).stem] = plain_all
    if checkpoint is not None and checkpoint.best_model_path:
        # 検証を持つ run は、best に加えて最終 epoch の重み（EMA があれば EMA）も同じ全窓で採点しておく。
        write_window_oof(cfg, lit_model.export_state_dict(ema=cfg.train.ema_decay is not None), datamodule.val_dataloader(), device, artifact_dir / LAST_OOF_FILENAME)
        scored["oof_last"] = LAST_OOF_FILENAME
        if cfg.train.ema_decay is not None:
            write_window_oof(cfg, lit_model.export_state_dict(), datamodule.val_dataloader(), device, artifact_dir / LAST_RAW_OOF_FILENAME)
            scored["oof_last_raw"] = LAST_RAW_OOF_FILENAME
    log_window_scores(trainer, artifact_dir, scored)


def write_window_oof(cfg: Config, state_dict: dict, val_loader: DataLoader, device: torch.device, path: Path) -> None:
    """``state_dict`` の重みで検証 split を全窓で採点し、OOF のフレームを ``path`` に書く。"""
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(state_dict)
    _window_oof(model, _window_oof_loader(cfg, val_loader), device).to_csv(path, index=False)


def log_window_scores(trainer: L.Trainer | None, artifact_dir: Path, scored: dict[str, str]) -> None:
    """全窓予測のファイル（``{接頭辞: ファイル名}``）ごとの macro / 所見別 AUC をログに出し、W&B が有効なら run の summary にも書く。"""
    from lightning.pytorch.loggers import WandbLogger

    runs = [] if trainer is None else [lg.experiment for lg in trainer.loggers if isinstance(lg, WandbLogger)]
    for prefix, filename in scored.items():
        frame = pd.read_csv(artifact_dir / filename)
        # 検証の val/auc_macro と同じく、soft な target は 2 値にしてから採点する。
        macro, per_label = macro_auc(binarize(frame[[f"true_{name}" for name in LABELS]].to_numpy()), frame[[f"pred_{name}" for name in LABELS]].to_numpy())
        logger.info("%s macro AUC (all windows): %.4f", prefix, macro)
        if runs:
            runs[0].summary.update({f"{prefix}/auc_macro": macro, **{f"{prefix}/auc/{name}": score for name, score in per_label.items()}})


def rescore_weights(cfg: Config, fold: int, weights: Path, limit: int | None = None) -> Path:
    """学習せずに ``weights`` の state dict で検証 split を全窓で採点し、``fold<k>/oof_<stem>.csv`` に書いて macro AUC をログに出す。"""
    if cfg.data.split == "all":
        raise ValueError("--rescore needs a validation split; data.split=all has none")
    L.seed_everything(cfg.seed, workers=True)
    train_dataset, val_dataset = build_datasets(cfg, fold, limit=limit)
    datamodule = KneeDataModule(cfg, train_dataset, val_dataset)
    artifact_dir = cfg.output_dir / cfg.experiment_name / f"fold{fold}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    filename = f"oof_{weights.stem}.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_window_oof(cfg, torch.load(weights, map_location="cpu", weights_only=True), datamodule.val_dataloader(), device, artifact_dir / filename)
    log_window_scores(None, artifact_dir, {Path(filename).stem: filename})
    return artifact_dir / filename


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None, help="Path to JSON config file.")
@click.option("--fold", type=int, default=0, show_default=True, help="Validation fold index.")
@click.option("--fast-dev-run", is_flag=True, help="Run a single train/val batch.")
@click.option("--limit", type=int, default=None, help="Use at most N training studies (and N/4 validation studies).")
@click.option("--epochs", type=int, default=None, help="Override train.epochs.")
@click.option("--rescore", type=click.Path(exists=True), default=None, help="Skip training and rewrite the all-window OOF from this state dict as oof_<stem>.csv.")
def main(config_path: str | None, fold: int, fast_dev_run: bool, limit: int | None, epochs: int | None, rescore: str | None) -> None:
    # Config は BaseSettings。キーワード引数が RSNA_KNEE_* 環境変数に勝ち、それが既定値に勝つ。
    cfg = Config(**json.loads(Path(config_path).read_text())) if config_path else Config()
    if rescore is not None:
        if fast_dev_run or epochs is not None:
            raise ValueError("--rescore does not train; drop --fast-dev-run / --epochs")
        rescore_weights(cfg, fold, Path(rescore), limit=limit)
        return
    if epochs is not None:
        cfg.train.epochs = epochs
    run(cfg, fold=fold, fast_dev_run=fast_dev_run, limit=limit)


if __name__ == "__main__":
    main()
