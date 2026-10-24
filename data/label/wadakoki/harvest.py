"""``generate.py`` の chunk を検め、受理した検査のスコアを ``labels_accepted.csv`` に書き出す（受理できなかった検査は埋めない）。

全ての入力の検査が一度は問い合わされていること、受理した出力が生の応答の本文と一致することを確かめる。
受理できなかった検査は ``unresolved.json``、所見ごとの分布と source の内訳は ``target_summary.csv``、全体は ``summary.json`` に書く。

    uv run python data/label/wadakoki/harvest.py --root output/llm_labels/gemma_v04 --output output/llm_labels/gemma_v04/harvest
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import TARGETS, no_duplicates, validate_labels  # noqa: E402


def harvest(root: Path, output: Path) -> dict:
    """``root`` の全ての chunk から受理した出力を集め、``output`` に書いて要約を返す。"""
    if output.exists():
        raise FileExistsError(output)
    ids = [json.loads(line)["StudyInstanceUID"] for line in (root / "reports.jsonl").read_text().splitlines()]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate StudyInstanceUID in reports.jsonl")
    allowed = set(ids)
    accepted, seen, failures, retries = {}, set(), {}, 0
    chunks = sorted(root.glob("attempts/*/chunks/*.json"))
    for path in chunks:
        for row in json.loads(path.read_text()):
            uid = row["StudyInstanceUID"]
            if uid not in allowed:
                raise ValueError(f"unknown study in {path}")
            seen.add(uid)
            retries += len(row["attempts"]) - 1
            if row["accepted"]:
                if uid in accepted:
                    raise ValueError(f"study accepted twice: {path}")
                labels = validate_labels(row["labels"])
                choice = row["attempts"][-1]["raw_response"]["choices"][0]
                if choice["finish_reason"] != "stop" or json.loads(choice["message"]["content"], object_pairs_hook=no_duplicates) != labels:
                    raise ValueError(f"accepted labels differ from the raw response: {path}")
                accepted[uid] = labels
            else:
                failures[uid] = {"StudyInstanceUID": uid, "chunk": str(path.relative_to(root)), "errors": [a.get("error") for a in row["attempts"]]}
    if seen != allowed:
        raise ValueError(f"{len(allowed - seen)} input studies were never processed")
    unresolved = [failures[uid] for uid in ids if uid not in accepted]
    output.mkdir(parents=True)
    with (output / "labels_accepted.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["StudyInstanceUID", *TARGETS])
        writer.writeheader()
        for uid in ids:
            if uid in accepted:
                writer.writerow({"StudyInstanceUID": uid, **{t: accepted[uid][t]["score"] for t in TARGETS}})
    (output / "unresolved.json").write_text(json.dumps(unresolved, indent=2))
    sources = Counter(v["source"] for labels in accepted.values() for v in labels.values())
    fields = ["target", "count", "mean", "median", "min", "max", "score_ge_050_count", "direct", "default", "uncertain", "inferred", "conflicting"]
    with (output / "target_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for target in TARGETS:
            values = [labels[target]["score"] for labels in accepted.values()]
            counts = Counter(labels[target]["source"] for labels in accepted.values())
            writer.writerow(
                {"target": target, "count": len(values), "mean": round(mean(values), 6), "median": median(values), "min": min(values), "max": max(values), "score_ge_050_count": sum(x >= 0.5 for x in values), **{s: counts[s] for s in fields[7:]}}
            )
    summary = {"input_rows": len(ids), "accepted_rows": len(accepted), "unresolved_rows": len(unresolved), "chunks": len(chunks), "retries": retries, "source_counts": dict(sources)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@click.command(help="Check generate.py's chunks and export the accepted scores to labels_accepted.csv (abstentions are not filled).")
@click.option("--root", type=click.Path(exists=True, path_type=Path), required=True, help="--root of generate.py.")
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Directory to write (must not exist).")
def main(root: Path, output: Path) -> None:
    print(json.dumps(harvest(root, output)))


if __name__ == "__main__":
    main()
