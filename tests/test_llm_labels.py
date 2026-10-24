import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from rsna_knee.constants import LABELS

# Qwen / Gemma のテキストラベルのスクリプトは data/label/wadakoki/ に閉じているので、そのフォルダを import の先頭に足す
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "label" / "wadakoki"))

import generate as g  # noqa: E402
from harvest import harvest  # noqa: E402
from make_reports import unlabeled_reports  # noqa: E402


def _labels(score: float | None = None) -> dict:
    """全所見が既定値の出力。``score`` を与えれば ACL だけ根拠つきのそのスコアにする。"""
    out = {k: {"score": g.DEFAULTS[k], "source": "default", "evidence": g.DEFAULT_EVIDENCE} for k in g.TARGETS}
    if score is not None:
        out["ACL"] = {"score": score, "source": "direct", "evidence": "Complete ACL tear."}
    return out


def test_validate_rejects_bad_scores_and_broken_default_contracts() -> None:
    for score in (None, float("nan"), True, 0.999, 0.123, 0.10):
        data = _labels()
        data["ACL"]["score"] = score
        with pytest.raises(ValueError):
            g.validate_labels(data)
    data = _labels()
    data["ACL"]["source"] = "direct"
    with pytest.raises(ValueError, match="default_evidence_requires"):
        g.validate_labels(data)
    with pytest.raises(ValueError):
        json.loads('{"ACL":1,"ACL":2}', object_pairs_hook=g.no_duplicates)
    assert g.validate_labels(_labels(0.95))["ACL"]["score"] == 0.95


def test_load_accepted_skips_partial_writes_and_rejects_double_acceptance(tmp_path: Path) -> None:
    chunks = tmp_path / "attempts/a/chunks"
    chunks.mkdir(parents=True)
    g.atomic_json(chunks / "000000.json", [{"StudyInstanceUID": "a", "accepted": True, "labels": _labels()}, {"StudyInstanceUID": "b", "accepted": False}])
    (chunks / "000001.json.tmp").write_text("interrupted")
    assert set(g.load_accepted(tmp_path, {"a", "b"})) == {"a"}
    g.atomic_json(chunks / "000002.json", [{"StudyInstanceUID": "a", "accepted": True, "labels": _labels()}])
    with pytest.raises(ValueError, match="duplicate_accepted"):
        g.load_accepted(tmp_path, {"a", "b"})


def test_unlabeled_reports_keep_only_studies_without_official_labels(tmp_path: Path) -> None:
    train = pd.DataFrame({"StudyInstanceUID": ["gold", "u1", "u2"], "Report": ["g", "first report", "second"], **{c: [1.0, None, None] for c in LABELS}})
    train.to_csv(tmp_path / "train.csv", index=False)
    assert unlabeled_reports(tmp_path / "train.csv") == [{"StudyInstanceUID": "u1", "Report": "first report"}, {"StudyInstanceUID": "u2", "Report": "second"}]


def test_harvest_exports_accepted_scores_without_filling_abstentions(tmp_path: Path) -> None:
    """受理した検査のスコアだけを書き、受理できなかった検査は unresolved に回す。"""
    (tmp_path / "reports.jsonl").write_text("".join(json.dumps({"StudyInstanceUID": u, "Report": "r"}) + "\n" for u in ("a", "b")))
    chunks = tmp_path / "attempts/full/chunks"
    chunks.mkdir(parents=True)
    labels = _labels(0.8)
    raw = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(labels)}}]}
    g.atomic_json(chunks / "000000.json", [{"StudyInstanceUID": "a", "accepted": True, "labels": labels, "attempts": [{"retry": 0, "raw_response": raw}]}, {"StudyInstanceUID": "b", "accepted": False, "attempts": [{"error": "non_stop_finish"}] * 3}])
    summary = harvest(tmp_path, tmp_path / "out")
    assert summary["accepted_rows"] == 1 and summary["unresolved_rows"] == 1 and summary["retries"] == 2
    table = pd.read_csv(tmp_path / "out" / "labels_accepted.csv")
    assert list(table["StudyInstanceUID"]) == ["a"] and table.loc[0, "ACL"] == 0.8 and table.loc[0, "Fracture"] == g.DEFAULTS["Fracture"]
    assert json.loads((tmp_path / "out" / "unresolved.json").read_text())[0]["StudyInstanceUID"] == "b"
