from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.utils import get_logger

logger = get_logger(__name__)

#: llm_labels_v4_blend.csv が「レポートがその所見に触れていない」を表すのに使う値。
#: v1/v2 の表を読むときは unknown_value=0.5 を渡す。
UNKNOWN_VALUE: float = 0.25

#: 重症度の表（long 形式）の列。1 行が (study, target) で、``severity_soft`` は ``[0, 1]`` の期待重症度。
SEVERITY_TARGET_COLUMN = "target"
SEVERITY_COLUMN = "severity_soft"
#: 重症度の表の ``target`` のキー -> ``LABELS`` の名前。
SEVERITY_KEYS: dict[str, str] = {
    "acl": "ACL",
    "mcl": "MCL",
    "medial_meniscus": "Medial Meniscus",
    "lateral_meniscus": "Lateral Meniscus",
    "medial_oa": "Medial OA",
    "lateral_oa": "Lateral OA",
    "pf_oa": "PF OA",
    "effusion": "Effusion",
    "synovitis": "Synovitis",
    "bakers": "Baker's",
    "contusion": "Contusion",
    "fracture": "Fracture",
}


def load_labels(labels_csv: str | Path, train_csv: str | Path | None = None, unknown_to_nan: bool = False, unknown_value: float = UNKNOWN_VALUE) -> pd.DataFrame:
    """LLM の soft label を読み、gold の Study を ``train.csv`` で上書きする。

    両者を ``StudyInstanceUID`` を index とする ``[n_studies, 12]`` の float32 1 枚に合流させ、target の
    契約（値は ``[0, 1]``、``NaN`` = unknown）に揃える。

    Args:
        labels_csv: ``StudyInstanceUID`` と 12 個のラベル列を持つ CSV。
        train_csv: 公式の ``train.csv``。12 列がすべて非 null の行を gold とみなして LLM の行を
            置き換える。ファイルが無いか ``None`` なら上書きしない。
        unknown_to_nan: ``unknown_value`` にちょうど等しいセルを ``NaN`` にして masked loss に無視させる
            （gold の行には触れない）。偽なら CSV の空セルを ``unknown_value`` で埋める。
        unknown_value: 「レポートが触れていない」の sentinel。:data:`UNKNOWN_VALUE` を見よ。
    """
    frame = pd.read_csv(labels_csv)
    missing = [c for c in [ID_COLUMN, *LABELS] if c not in frame.columns]
    if missing:
        raise ValueError(f"{labels_csv}: missing columns {missing}")
    labels = frame.set_index(ID_COLUMN)[LABELS].astype("float32")
    if labels.index.has_duplicates:
        raise ValueError(f"{labels_csv}: duplicate {ID_COLUMN} values")
    # CSV の空セルは「レポートが触れていない」。埋める前に数えてログに残す。
    n_empty = int(labels.isna().to_numpy().sum())

    gold_index = pd.Index([], name=ID_COLUMN)
    if train_csv is not None and Path(train_csv).exists():
        train = pd.read_csv(train_csv, usecols=lambda c: c in {ID_COLUMN, *LABELS})
        missing = [c for c in [ID_COLUMN, *LABELS] if c not in train.columns]
        if missing:
            raise ValueError(f"{train_csv}: missing columns {missing}")
        train = train.set_index(ID_COLUMN)[LABELS].astype("float32")
        gold = train[train.notna().all(axis=1)]
        # LLM 表に無い gold study も残す価値がある。
        labels = labels.reindex(labels.index.union(gold.index, sort=False))
        labels.loc[gold.index, LABELS] = gold[LABELS].to_numpy()
        gold_index = gold.index
    elif train_csv is not None:
        logger.warning("Gold labels not applied: %s does not exist", train_csv)

    if unknown_to_nan:
        sentinel = labels.to_numpy() == np.float32(unknown_value)
        sentinel[labels.index.isin(gold_index)] = False
        labels = labels.mask(pd.DataFrame(sentinel, index=labels.index, columns=LABELS))
    else:
        # 空セルも sentinel と同じ値に揃える。gold の行は上書き済みなので NaN は残っていない。
        labels = labels.fillna(np.float32(unknown_value))

    labels = labels.astype("float32")
    logger.info(
        "load_labels(%s): %d studies, %d gold, %d empty cells in CSV, %d NaN cells, %d cells == %.3g (unknown_to_nan=%s)",
        Path(labels_csv).name,
        len(labels),
        len(gold_index),
        n_empty,
        int(labels.isna().to_numpy().sum()),
        int((labels.to_numpy() == np.float32(unknown_value)).sum()),
        unknown_value,
        unknown_to_nan,
    )
    return labels


def gold_ids(train_csv: str | Path | None) -> set[str]:
    """``train.csv`` で 12 列すべてが埋まっている study の ``StudyInstanceUID``。

    :func:`load_labels` が LLM のラベルを上書きするのと同じ集合で、``train_csv`` が ``None`` か存在しなければ空。
    """
    if train_csv is None or not Path(train_csv).exists():
        return set()
    train = pd.read_csv(train_csv, usecols=lambda c: c in {ID_COLUMN, *LABELS})
    missing = [c for c in [ID_COLUMN, *LABELS] if c not in train.columns]
    if missing:
        raise ValueError(f"{train_csv}: missing columns {missing}")
    return set(train.loc[train[LABELS].notna().all(axis=1), ID_COLUMN].astype(str))


def load_weights(weights_csv: str | Path, study_ids: list[str]) -> np.ndarray:
    """セルごとの損失の重みを ``study_ids`` の順に並べた ``[N, 12] float32``。

    表に無い study の行と空セルは 1（重みを指定しないことと重み 1 は同じ意味である）。

    Args:
        weights_csv: ``StudyInstanceUID`` と 12 個のラベル列を持つ、セルごとの重みの CSV。
        study_ids: 並べる順。重複していてもよい。
    """
    frame = pd.read_csv(weights_csv)
    missing = [c for c in [ID_COLUMN, *LABELS] if c not in frame.columns]
    if missing:
        raise ValueError(f"{weights_csv}: missing columns {missing}")
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    weights = frame.set_index(ID_COLUMN)[LABELS].astype("float32")
    if weights.index.has_duplicates:
        raise ValueError(f"{weights_csv}: duplicate {ID_COLUMN} values")
    array = weights.reindex(pd.Index([str(uid) for uid in study_ids], name=ID_COLUMN)).to_numpy(dtype=np.float32)
    logger.info("load_weights(%s): %d studies, %d of %d cells default to 1", Path(weights_csv).name, len(study_ids), int(np.isnan(array).sum()), array.size)
    return np.where(np.isnan(array), np.float32(1.0), array)


def load_severity(severity_csv: str | Path, study_ids: list[str]) -> np.ndarray:
    """所見ごとの重症度 ``severity_soft`` を ``study_ids`` の順に並べた ``[N, 12] float32``。表に無い study の行と無い (study, target) のセルは ``NaN``。

    表は long 形式（1 行が (study, target)）で、``target`` のキーを :data:`SEVERITY_KEYS` で ``LABELS`` の順の列に写す。
    未知のキー、重複した (study, target)、``[0, 1]`` の外の値は例外。
    """
    frame = pd.read_csv(severity_csv)
    missing = [c for c in [ID_COLUMN, SEVERITY_TARGET_COLUMN, SEVERITY_COLUMN] if c not in frame.columns]
    if missing:
        raise ValueError(f"{severity_csv}: missing columns {missing}")
    unknown = sorted(set(frame[SEVERITY_TARGET_COLUMN].astype(str)) - set(SEVERITY_KEYS))
    if unknown:
        raise ValueError(f"{severity_csv}: unknown {SEVERITY_TARGET_COLUMN} keys {unknown}")
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    if frame.duplicated([ID_COLUMN, SEVERITY_TARGET_COLUMN]).any():
        raise ValueError(f"{severity_csv}: duplicate ({ID_COLUMN}, {SEVERITY_TARGET_COLUMN}) rows")
    values = frame[SEVERITY_COLUMN].astype("float64")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{severity_csv}: {SEVERITY_COLUMN} must lie in [0, 1]")
    wide = frame.pivot(index=ID_COLUMN, columns=SEVERITY_TARGET_COLUMN, values=SEVERITY_COLUMN).rename(columns=SEVERITY_KEYS).reindex(columns=LABELS)
    array = wide.reindex(pd.Index([str(uid) for uid in study_ids], name=ID_COLUMN)).to_numpy(dtype=np.float32)
    logger.info("load_severity(%s): %d studies in the table, %d of %d cells are NaN for the %d requested studies", Path(severity_csv).name, len(wide), int(np.isnan(array).sum()), array.size, len(study_ids))
    return array


def binarize(target: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """soft target を ROC AUC 用に硬くする: ``>t`` -> 1、``<t`` -> 0、``==t`` と ``NaN`` -> ``NaN``。"""
    target = np.asarray(target, dtype=np.float64)
    out = np.where(target > threshold, 1.0, 0.0)
    return np.where(np.isnan(target) | (target == threshold), np.nan, out)
