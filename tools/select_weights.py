"""gold split の学習の出力から fold ごとに提出する重みを選び、提出用の保存先（``output/submit/<実験名>/``）に置く。

fold ごとに、gold の ``val/auc_macro``（EMA）が最大の epoch（同点は早い方。``best.ckpt`` と同じ選び方）を見る。
それが ``--min-epoch`` より前なら ``checkpoints/epoch{NN}_ema.pt``（``NN`` = ``--min-epoch``）を、そうでなければ gold best の EMA（``model.pt``）を置く。
gold は 58 例なので、学習の初めの数 epoch の best は揺れで選ばれやすい。最終提出 57482743・57523282 に入れた wadakoki の重みは、この規則で ``--min-epoch 4``。
``inference.json`` には ``--union-cap`` を書く（series_neck は全 series の Bag でしか採点できない）。選んだ結果は ``selection.json`` に残す。
置いた保存先は ``tools/kaggle_ops.py model -m ...`` で model dataset に上げる。

    uv run python tools/select_weights.py caformers36_spatial_planeaux01_s2online_union_seriesneck_headreset_ep10_aug_v6r2r040_knees_cos20_softlse_half_gold_seriesaug_ema_usub_dprior_a10 --union-cap 384
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import click
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rsna_knee.experiments import ExperimentConfig  # noqa: E402


def gold_auc_by_epoch(fold_dir: Path) -> dict[int, float]:
    """fold の最後の ``version_*/metrics.csv`` から epoch ごとの ``val/auc_macro``（EMA）。"""
    versions = sorted(fold_dir.glob("version_*/metrics.csv"), key=lambda path: int(path.parent.name.removeprefix("version_")))
    if not versions:
        raise FileNotFoundError(f"metrics.csv が無い: {fold_dir}")
    metrics = pd.read_csv(versions[-1]).dropna(subset=["val/auc_macro"])
    if metrics.empty:
        raise ValueError(f"val/auc_macro が無い: {versions[-1]}")
    return {int(epoch): float(auc) for epoch, auc in zip(metrics["epoch"], metrics["val/auc_macro"])}


def choose(auc: dict[int, float], min_epoch: int) -> tuple[int, int, str]:
    """``(best の epoch, 使う epoch, fold の中の重みのパス)``。best は最大の AUC のうち最も早い epoch。"""
    best = max(sorted(auc), key=lambda epoch: auc[epoch])
    if best < min_epoch:
        return best, min_epoch, f"checkpoints/epoch{min_epoch:02d}_ema.pt"
    return best, best, "model.pt"


def select(experiment: str, output_dir: Path, store: Path, min_epoch: int, union_cap: int | None) -> list[dict]:
    """1 実験の全 fold の重みを選んで ``store/<実験名>/`` に置き、選んだ結果の行を返す。"""
    src = output_dir / experiment
    folds = sorted(path for path in src.glob("fold*") if path.is_dir())
    if not folds:
        raise FileNotFoundError(f"fold が無い: {src}")
    config = ExperimentConfig.model_validate_json((src / "inference.json").read_text())
    config = ExperimentConfig.model_validate({**config.model_dump(mode="json"), "union_cap": union_cap})
    dest = store / experiment
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "inference.json").write_text(config.model_dump_json(indent=2) + "\n")
    rows = []
    for fold in folds:
        auc = gold_auc_by_epoch(fold)
        best, used, weight = choose(auc, min_epoch)
        if used not in auc:
            raise ValueError(f"epoch {used} の val/auc_macro が無い: {fold}")
        (dest / fold.name).mkdir()
        shutil.copy2(fold / weight, dest / fold.name / "model.pt")
        shutil.copy2(fold / "config.json", dest / fold.name / "config.json")
        rows.append({"fold": fold.name, "best_epoch": best, "best_auc": auc[best], "used_epoch": used, "used_auc": auc[used], "weight": weight})
    (dest / "selection.json").write_text(json.dumps({"min_epoch": min_epoch, "union_cap": union_cap, "folds": rows}, indent=2) + "\n")
    return rows


@click.command(help="Pick each fold's submitted weights from a gold-split training output and place them in the submission store (output/submit/<experiment>/).")
@click.argument("experiments", nargs=-1, required=True)
@click.option("--min-epoch", type=int, default=4, help="Use this epoch's EMA when the best epoch comes before it.")
@click.option("--union-cap", type=int, default=None, help="union_cap written to inference.json (384 for series_neck).")
@click.option("--output-dir", type=click.Path(path_type=Path), default=REPO_ROOT / "output", help="Training output directory.")
@click.option("--store", type=click.Path(path_type=Path), default=REPO_ROOT / "output" / "submit", help="Submission store directory.")
def main(experiments: tuple[str, ...], min_epoch: int, union_cap: int | None, output_dir: Path, store: Path) -> None:
    """Pick the submitted weight of each fold for every experiment under ``--output-dir`` and place it in ``--store``."""
    for experiment in experiments:
        for row in select(experiment, output_dir, store, min_epoch, union_cap):
            print(f"{experiment} {row['fold']}: best ep{row['best_epoch']} {row['best_auc']:.4f} -> ep{row['used_epoch']} {row['used_auc']:.4f} ({row['weight']})")


if __name__ == "__main__":
    main()
