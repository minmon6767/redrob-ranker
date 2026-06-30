"""
End-to-end smoke test: run rank.py against a small candidate sample and
check the output satisfies the submission format rules.

This intentionally runs against sample_candidates.json (50 records) rather
than the full 100K pool, since unit/CI runs should be fast; the full-pool
run is exercised manually and timed separately (see README.md "Reproducing
the submission").
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
SAMPLE_CANDIDATES = REPO_ROOT / "data" / "sample_candidates.json"


@pytest.fixture(scope="module")
def small_pool_jsonl(tmp_path_factory):
    """Build a >=100-row JSONL pool by repeating the 50 sample candidates
    with unique ids, since rank.py expects a top-100 ranking."""
    with open(SAMPLE_CANDIDATES, "r", encoding="utf-8") as f:
        base = json.load(f)

    pool = []
    for i in range(3):  # 50 * 3 = 150 rows, comfortably > 100
        for c in base:
            c2 = json.loads(json.dumps(c))  # deep copy
            c2["candidate_id"] = f"CAND_{9000000 + i * 1000 + int(c['candidate_id'].split('_')[1]):07d}"
            pool.append(c2)

    out_dir = tmp_path_factory.mktemp("data")
    out_path = out_dir / "pool.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in pool:
            f.write(json.dumps(c) + "\n")
    return out_path


def test_rank_script_produces_valid_submission(small_pool_jsonl, tmp_path):
    out_csv = tmp_path / "submission.csv"
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "rank.py"), "--candidates", str(small_pool_jsonl), "--out", str(out_csv)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if any(cell.strip() for cell in r)]

    assert header == ["candidate_id", "rank", "score", "reasoning"]
    assert len(rows) == 100

    ranks = [int(r[1]) for r in rows]
    assert sorted(ranks) == list(range(1, 101))

    ids = [r[0] for r in rows]
    assert len(set(ids)) == 100

    scores = [float(r[2]) for r in rows]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "scores must be non-increasing by rank"

    assert all(r[3].strip() for r in rows), "every row must have non-empty reasoning"
