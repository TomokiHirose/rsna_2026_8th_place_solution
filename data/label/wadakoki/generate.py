"""手元の llama.cpp サーバーに train のレポートを 1 件ずつ読ませ、12 所見の重症度スコアを JSON で返させて保存する（途中から続けられる）。

wadakoki の Qwen v0.4 / Gemma v0.4 のテキストラベル（チームのラベル L の材料。CSV はこのリポジトリに含めない）を作ったスクリプト。
1 件の問い合わせは system にプロンプト本文、user に ``"Report:\\n" + レポート本文`` だけを入れ、ほかの症例や既存のラベルは渡さない。
出力は JSON スキーマで固定し（所見ごとに score・source・evidence、または読めないレポートの ``{"error": "unreadable_report"}``）、
seed は検査ごとに ``sha256(StudyInstanceUID)`` の先頭 4 byte + ``generation.base_seed`` + やり直しの回数で決める。最大 ``generation.max_attempts`` 回まで。
結果は ``--root`` の下に 4 件ごとの chunk として書き、全件そろったら ``labels.csv`` を書く。取り出しは ``harvest.py``。

    uv run python data/label/wadakoki/launch_server.py --binary <llama-server> --model <gguf> --config data/label/wadakoki/gemma_v04.json --output output/llm_labels/gemma_v04/server
    uv run python data/label/wadakoki/generate.py --config data/label/wadakoki/gemma_v04.json --reports output/llm_labels/reports.jsonl --root output/llm_labels/gemma_v04 --attempt full
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent import futures
from datetime import UTC, datetime
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# レポートに記載が無いときの値（プロンプトの規則と同じ）。公開ラベルの付け方から逆に推定した尺度
DEFAULTS = dict(zip(TARGETS, [0.20, 0.15, 0.20, 0.20, 0.07, 0.08, 0.10, 0.20, 0.35, 0.05, 0.15, 0.10]))
SOURCES = ["direct", "uncertain", "conflicting", "inferred", "default"]
DEFAULT_EVIDENCE = "No mention; insufficient evidence for inference."


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: object) -> None:
    """一時ファイルに書いて fsync してから置き換える（途中で止まっても半端なファイルを残さない）。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    """``json.loads`` の ``object_pairs_hook``。同じキーが 2 度出る JSON は受け付けない。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def validate_labels(value: object) -> dict:
    """1 件の出力を検める。12 所見がこの順で、score は 0.05〜0.95 の小数 2 桁、``default`` は既定値と決まった evidence のときだけ。"""
    if not isinstance(value, dict) or list(value) != TARGETS:
        raise ValueError("target_keys_or_order")
    for target, entry in value.items():
        if not isinstance(entry, dict) or set(entry) != {"score", "source", "evidence"}:
            raise ValueError("entry_keys")
        score = entry["score"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0.05 <= score <= 0.95 or abs(score * 100 - round(score * 100)) > 1e-7:
            raise ValueError("score_range_or_precision")
        if entry["source"] not in SOURCES:
            raise ValueError("source_enum")
        if not isinstance(entry["evidence"], str) or not 1 <= len(entry["evidence"]) <= 120:
            raise ValueError("evidence_length")
        if entry["source"] == "default" and (score != DEFAULTS[target] or entry["evidence"] != DEFAULT_EVIDENCE):
            raise ValueError("default_contract")
        if entry["evidence"] == DEFAULT_EVIDENCE and entry["source"] != "default":
            raise ValueError("default_evidence_requires_default_source")
    return value


def output_schema() -> dict:
    """サーバーに渡す出力の JSON スキーマ。所見ごとに「根拠つきのスコア」か「既定値」、または読めないレポートのエラー。"""
    supported = {
        "type": "object",
        "properties": {"source": {"type": "string", "enum": [x for x in SOURCES if x != "default"]}, "evidence": {"type": "string", "minLength": 1, "maxLength": 120}, "score": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
        "required": ["source", "evidence", "score"],
        "additionalProperties": False,
    }
    entries = {}
    for target in TARGETS:
        default = {"type": "object", "properties": {"source": {"const": "default"}, "evidence": {"const": DEFAULT_EVIDENCE}, "score": {"const": DEFAULTS[target]}}, "required": ["source", "evidence", "score"], "additionalProperties": False}
        entries[target] = {"oneOf": [supported, default]}
    labels = {"type": "object", "properties": entries, "required": TARGETS, "additionalProperties": False}
    error = {"type": "object", "properties": {"error": {"const": "unreadable_report"}}, "required": ["error"], "additionalProperties": False}
    return {"oneOf": [labels, error]}


def request_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def generate_one(row: dict, prompt: str, cfg: dict, url: str) -> dict:
    """1 件のレポートを問い合わせ、受理できる出力が出るまで最大 ``max_attempts`` 回やり直す。全ての試行の生の応答を残す。"""
    generation = cfg["generation"]
    record = {"StudyInstanceUID": row["StudyInstanceUID"], "attempts": [], "accepted": False}
    seed_base = int.from_bytes(hashlib.sha256(row["StudyInstanceUID"].encode()).digest()[:4], "big")
    for retry in range(generation["max_attempts"]):
        started = time.monotonic()
        seed = (seed_base + generation["base_seed"] + retry) % (2**31 - 1)
        payload = {
            "model": cfg["engine"]["alias"],
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": "Report:\n" + row["Report"]}],
            **{k: generation[k] for k in ("temperature", "top_p", "top_k", "max_tokens")},
            "seed": seed,
            "stream": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": generation["enable_thinking"]},
            "response_format": {"type": "json_schema", "json_schema": {"name": "knee_severity", "strict": True, "schema": output_schema()}},
        }
        attempt = {"retry": retry, "seed": seed, "started_at": now()}
        try:
            response = request_json(url + "/v1/chat/completions", payload, cfg["runtime"]["request_timeout_seconds"])
            attempt["raw_response"] = response
            choice = response["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("non_stop_finish")
            parsed = json.loads(choice["message"]["content"], object_pairs_hook=no_duplicates)
            record["labels"] = validate_labels(parsed)
            record["accepted"] = True
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            attempt["error"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # レポート本文を含む payload や URL は出さず、例外の種類だけ残す
            attempt["error"] = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                attempt["http_status"] = exc.code
        attempt["seconds"] = time.monotonic() - started
        record["attempts"].append(attempt)
        if record["accepted"]:
            break
    return record


def load_accepted(root: Path, allowed_ids: set[str]) -> dict[str, dict]:
    """保存済みの chunk から受理した出力を集める。書きかけの ``.tmp`` は読まない。同じ検査を 2 度受理していたらエラー。"""
    accepted = {}
    for path in sorted(root.glob("attempts/*/chunks/*.json")):
        for row in json.loads(path.read_text()):
            uid = row["StudyInstanceUID"]
            if uid not in allowed_ids:
                raise ValueError("stored_unknown_id")
            if not row["accepted"]:
                continue
            if uid in accepted:
                raise ValueError("duplicate_accepted_id")
            accepted[uid] = validate_labels(row["labels"])
    return accepted


@click.command(help="Ask a local llama.cpp server to score the 12 findings of each train report (resumable).")
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="data/label/wadakoki/<gemma_v04|qwen_v04>.json")
@click.option("--reports", type=click.Path(exists=True, path_type=Path), required=True, help="StudyInstanceUID / Report JSONL written by make_reports.py.")
@click.option("--root", type=click.Path(path_type=Path), required=True, help="Output directory.")
@click.option("--attempt", required=True, help="Name of this run (written to attempts/<name>/).")
@click.option("--url", default=None, help="llama.cpp server URL. Defaults to 127.0.0.1 on the config's engine.port.")
@click.option("--limit", type=int, default=None, help="Only query the first N reports (trial run).")
@click.option("--resume", is_flag=True, help="Continue the previous run with the same config and input (accepted studies are skipped).")
def main(config: Path, reports: Path, root: Path, attempt: str, url: str | None, limit: int | None, resume: bool) -> None:
    """全件を受理できたら終了コード 0、足りなければ 2。"""
    raise SystemExit(run(config, reports, root, attempt, url, limit, resume))


def run(config: Path, reports: Path, root: Path, attempt: str, url: str | None, limit: int | None, resume: bool) -> int:
    cfg = json.loads(config.read_text())
    url = url or f"http://127.0.0.1:{cfg['engine']['port']}"
    prompt = (REPO_ROOT / cfg["prompt"]).read_text()
    rows = [json.loads(line) for line in reports.read_text().splitlines()]
    ids = {r["StudyInstanceUID"] for r in rows}
    if not len(ids) == len(rows) == cfg["input"]["expected_rows"]:
        raise ValueError(f"expected {cfg['input']['expected_rows']} distinct reports, got {len(rows)} rows / {len(ids)} ids")
    if not all(set(r) == {"StudyInstanceUID", "Report"} and r["Report"].strip() for r in rows):
        raise ValueError("each row must have StudyInstanceUID and a non-empty Report only")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "writer.lock"
    lock.mkdir()  # 前の実行の lock が残っていたら手で確かめて消す（自動では消さない）
    atomic_json(lock / "owner.json", {"pid": os.getpid(), "attempt": attempt, "started_at": now()})
    try:
        contract = {k: cfg[k] for k in ("prompt_version", "model", "generation")} | {"prompt_text": prompt}
        if (root / "contract.json").exists():
            if not resume:
                raise FileExistsError("existing_generation_requires_explicit_resume")
            if json.loads((root / "contract.json").read_text()) != contract:
                raise ValueError("generation_contract_changed")
            if (root / "reports.jsonl").read_bytes() != reports.read_bytes():
                raise ValueError("resume_input_changed")
        else:
            if resume:
                raise FileNotFoundError("resume_contract_missing")
            atomic_json(root / "contract.json", contract)
            shutil.copyfile(reports, root / "reports.jsonl")
            shutil.copyfile(config, root / "config.json")
        accepted = load_accepted(root, ids)
        selected = rows if limit is None else rows[:limit]
        pending = [r for r in selected if r["StudyInstanceUID"] not in accepted]
        attempt_dir = root / "attempts" / attempt
        attempt_dir.mkdir(parents=True, exist_ok=False)
        chunks = attempt_dir / "chunks"
        chunks.mkdir()
        atomic_json(attempt_dir / "runtime.json", {"started_at": now(), "pid": os.getpid(), "selected": len(selected), "already_accepted": len(accepted), "config": cfg})
        started = time.monotonic()
        last_save = started
        buffer, chunk_number = [], 0
        processed, rejected, retries = 0, 0, 0

        def flush() -> None:
            nonlocal buffer, chunk_number, last_save
            if buffer:
                path = chunks / f"{chunk_number:06d}.json"
                if path.exists():
                    raise FileExistsError("chunk_collision")
                atomic_json(path, buffer)
                # 書いたファイルを読み直して検めてから進んだことにする
                stored = json.loads(path.read_text())
                if len(stored) != len(buffer):
                    raise ValueError("chunk_readback_mismatch")
                for row in stored:
                    if row["accepted"]:
                        accepted[row["StudyInstanceUID"]] = validate_labels(row["labels"])
                buffer = []
                chunk_number += 1
            elapsed = time.monotonic() - started
            progress = {
                "updated_at": now(),
                "processed_this_attempt": processed,
                "accepted_total": len(accepted),
                "selected": len(selected),
                "expected_total": len(rows),
                "rejected_this_attempt": rejected,
                "retries": retries,
                "elapsed_seconds": elapsed,
                "rows_per_minute": processed / max(elapsed, 0.001) * 60,
                "durable_chunks": chunk_number,
            }
            atomic_json(attempt_dir / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            last_save = time.monotonic()

        remaining = iter(pending)
        deadline = started + cfg["runtime"]["maximum_hours"] * 3600
        with futures.ThreadPoolExecutor(max_workers=cfg["engine"]["parallel"]) as pool:
            active = set()

            def enqueue() -> None:
                if time.monotonic() >= deadline:
                    return
                row = next(remaining, None)
                if row is not None:
                    active.add(pool.submit(generate_one, row, prompt, cfg, url))

            for _ in range(cfg["engine"]["parallel"]):
                enqueue()
            while active:
                done, active = futures.wait(active, timeout=5, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    record = future.result()
                    buffer.append(record)
                    processed += 1
                    rejected += not record["accepted"]
                    retries += len(record["attempts"]) - 1
                    # 受理できない検査が 8 件たまったら新しい問い合わせを止める（設定かサーバーの異常を疑う）
                    if rejected < 8:
                        enqueue()
                if len(buffer) >= cfg["runtime"]["save_every_rows"] or time.monotonic() - last_save >= cfg["runtime"]["save_every_seconds"]:
                    flush()
        flush()
        complete = all(r["StudyInstanceUID"] in accepted for r in selected)
        summary = {
            "finished_at": now(),
            "selected_complete": complete,
            "accepted_total": len(accepted),
            "expected_total": len(rows),
            "processed_this_attempt": processed,
            "rejected_this_attempt": rejected,
            "elapsed_seconds": time.monotonic() - started,
        }
        if len(accepted) == len(rows):
            output = root / "labels.csv"
            if output.exists():
                raise FileExistsError("final_labels_already_exist")
            with output.with_suffix(".csv.tmp").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=["StudyInstanceUID", *TARGETS])
                writer.writeheader()
                for row in rows:
                    uid = row["StudyInstanceUID"]
                    writer.writerow({"StudyInstanceUID": uid, **{k: accepted[uid][k]["score"] for k in TARGETS}})
                f.flush()
                os.fsync(f.fileno())
            output.with_suffix(".csv.tmp").replace(output)
        summary["source_counts"] = dict(Counter(e["source"] for labels in accepted.values() for e in labels.values()))
        atomic_json(attempt_dir / "outcome.json", summary)
        print("SELECTED_COMPLETE" if complete else "INCOMPLETE_REQUIRES_REVIEW", flush=True)
        return 0 if complete else 2
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    main()
