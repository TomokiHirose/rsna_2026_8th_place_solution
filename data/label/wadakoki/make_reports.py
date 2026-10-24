"""``train.csv`` から 12 所見が全て空の検査（gold 以外の 4,349 件）のレポートを取り出し、``generate.py`` の入力の JSONL にする。

1 行は ``{"StudyInstanceUID": ..., "Report": ...}`` だけで、レポートは原文のまま（翻訳も整形もしない）。並びは ``train.csv`` の順。
問い合わせの seed は検査ごとに決まるので、並びはスコアを基本的に変えない（llama.cpp の並列実行で数値がわずかにぶれることはある）。wadakoki の v0.4 は別の順（seed 340918 で並べ替えた順）で回した。

    uv run python data/label/wadakoki/make_reports.py --train-csv input/rsna-knee-abnormality-detection/train.csv --out output/llm_labels/reports.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rsna_knee.constants import ID_COLUMN, LABELS  # noqa: E402


def unlabeled_reports(train_csv: Path) -> list[dict]:
    """12 所見が全て空の行の ``StudyInstanceUID`` と ``Report``。レポートが空の行があればエラー。"""
    train = pd.read_csv(train_csv, dtype={ID_COLUMN: str})
    rows = train.loc[train[LABELS].isna().all(axis=1), [ID_COLUMN, "Report"]]
    if rows["Report"].isna().any() or not rows["Report"].str.strip().all():
        raise ValueError("an unlabeled study has an empty Report")
    return [{"StudyInstanceUID": uid, "Report": report} for uid, report in zip(rows[ID_COLUMN], rows["Report"])]


@click.command(help="Write the reports of the train studies without official labels (4,349) as the JSONL input of generate.py.")
@click.option("--train-csv", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(train_csv: Path, out: Path) -> None:
    rows = unlabeled_reports(train_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(rows)} reports to {out}")


if __name__ == "__main__":
    main()
