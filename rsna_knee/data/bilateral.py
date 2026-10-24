"""1 つの study に両膝の series が入っている検査を見つけ、series を右膝・左膝に振り分ける（膝ごとの子 study を作る前処理）。

DICOM の ``Laterality`` は空や矛盾が多く（train の両膝 8 検査では 77 series 中 40 本が空、片膝の検査にも R と L が混ざるものがある）、
左右の判定には使わない。代わりに ``series_meta.csv`` の幾何（``ImagePositionPatient``・``ImageOrientationPatient``・画素間隔・枚数）から
series の中心の患者座標を求め、左右方向（LPS の x、正が患者の左）の位置で分ける。

- 1 検査の series の中心の x の広がりが ``min_spread_mm`` を超えれば両膝の検査とみなす（train では両膝 8 検査が 188〜237 mm、
  それ以外の 4,399 検査は最大 58 mm）。
- 両膝の検査は、中心の x を並べた最大の隙間で 2 群に分け、x の小さい群を右膝（子 ``<親>-1``）、大きい群を左膝（``<親>-2``）にする。
- series が ``min_series`` 本に満たない膝は子を作らず、その series を ``unused`` にする（1 本では膝として評価できない）。

出力は ``data/label/ayfukushima/bilateral_knees/knee_series.csv`` と同じ列（``knee``・``child_study_id``）を持ち、そのまま
``data.subcases_csv`` に渡せる。train ではこの規則が手で付けた ``knee_series.csv`` の 77 series と一致する（``tools/split_bilateral_knees.py --compare``）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["series_center_x", "split_bilateral_studies"]

#: 両膝の検査とみなす、series の中心の左右方向の広がり（mm）。
MIN_SPREAD_MM = 100.0
#: 子 study を作るのに要る 1 膝あたりの series の本数。
MIN_SERIES = 2


def _floats(value: object) -> np.ndarray | None:
    try:
        out = np.array([float(x) for x in str(value).replace("\\", "|").split("|")], dtype=np.float64)
    except ValueError:
        return None
    return out if out.size else None


def series_center_x(meta: pd.DataFrame) -> pd.Series:
    """series ごとの中心の患者座標 x（mm、LPS で正が患者の左）。幾何が欠けた series は NaN。

    中心 = 先頭スライスの ``ImagePositionPatient`` + 面内の半分（行・列の方向余弦 x 画素間隔 x 画素数 / 2）+ スライス方向の半分
    （法線 x スライス間隔 x (枚数 - 1) / 2）。スライス間隔は ``SpacingBetweenSlices``、無ければ ``SliceThickness``。
    """
    out = []
    for rec in meta.itertuples(index=False):
        iop, ipp = _floats(rec.ImageOrientationPatient), _floats(rec.ImagePositionPatient)
        try:
            step = float(rec.SpacingBetweenSlices) if pd.notna(rec.SpacingBetweenSlices) else float(rec.SliceThickness)
            rows, cols, n = float(rec.Rows), float(rec.Columns), float(rec.n_files)
            ps_row, ps_col = float(rec.PixelSpacing_0), float(rec.PixelSpacing_1)
        except (TypeError, ValueError):
            out.append(np.nan)
            continue
        if iop is None or ipp is None or iop.size != 6 or ipp.size != 3 or not np.isfinite([step, rows, cols, n, ps_row, ps_col]).all():
            out.append(np.nan)
            continue
        row_dir, col_dir = iop[:3], iop[3:]
        center = ipp + row_dir * ps_col * cols / 2 + col_dir * ps_row * rows / 2 + np.cross(row_dir, col_dir) * step * (n - 1) / 2
        out.append(float(center[0]))
    return pd.Series(out, index=meta.index, dtype=np.float64)


def split_bilateral_studies(meta: pd.DataFrame, min_spread_mm: float = MIN_SPREAD_MM, min_series: int = MIN_SERIES) -> pd.DataFrame:
    """``series_meta.csv`` の表から両膝の検査を見つけ、その series を膝に振り分けた表を返す（両膝でない検査は含めない）。

    列: ``StudyInstanceUID``（親）・``SeriesInstanceUID``・``knee``（right / left / unused）・``child_study_id``（unused は空）・
    ``center_x_mm``。中心が求まらない series を含む両膝の検査はエラー（その series がどちらの膝か決められない）。
    """
    frame = meta[["StudyInstanceUID", "SeriesInstanceUID"]].astype(str).copy()
    frame["center_x_mm"] = series_center_x(meta).to_numpy()
    rows = []
    for study, group in frame.groupby("StudyInstanceUID", sort=True):
        known = group["center_x_mm"].dropna()
        if len(known) < 2 or known.max() - known.min() <= min_spread_mm:
            continue
        if group["center_x_mm"].isna().any():
            missing = group.loc[group["center_x_mm"].isna(), "SeriesInstanceUID"].tolist()
            raise ValueError(f"{study}: bilateral study with series of unknown position {missing}")
        ordered = group.sort_values("center_x_mm")
        cut = int(np.argmax(np.diff(ordered["center_x_mm"].to_numpy()))) + 1
        for knee, suffix, part in (("right", 1, ordered.iloc[:cut]), ("left", 2, ordered.iloc[cut:])):
            keep = len(part) >= min_series
            for rec in part.itertuples(index=False):
                rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": rec.SeriesInstanceUID, "knee": knee if keep else "unused", "child_study_id": f"{study}-{suffix}" if keep else None, "center_x_mm": round(float(rec.center_x_mm), 1)})
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID", "knee", "child_study_id", "center_x_mm"])
