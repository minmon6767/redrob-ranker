#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge
Main ranking entry point.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Produces a CSV with exactly 100 rows (candidate_id, rank, score, reasoning),
ranking the top-100 best-fit candidates for the released JD, per
submission_spec.md Sections 2-3.

Design goals enforced by this script:
  - CPU only, no GPU, no network calls of any kind during ranking.
  - Two cheap streaming passes over candidates.jsonl rather than holding
    100K full JSON records in memory: pass 1 extracts small scalar feature
    rows + text docs for every candidate (needed to score the whole pool);
    pass 2 re-reads the file just to pull full profile detail for the ~100
    candidates that actually made the cut (needed only for reasoning text).
    Peak memory for "things proportional to pool size" is therefore a list
    of small float dicts and a list of strings, not 100K nested objects.
  - Deterministic: same input -> same output, every time (no randomness
    anywhere in scoring; the only random_state is in the semantic SVD fit,
    which is pinned to 42).
  - Every score comes with a grounded, fact-checkable one-line reasoning
    string built directly from the candidate's own profile fields -- no
    LLM call, so it cannot hallucinate a skill or employer that isn't
    actually in the record.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from features import extract_features
from honeypot import honeypot_score
from reasoning import build_reasoning
from scoring import final_score
from semantic import SemanticScorer, candidate_doc

TOP_N = 100


def iter_candidates(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates against the Redrob JD.")
    parser.add_argument("--candidates", required=True, type=Path, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="Output submission CSV path")
    parser.add_argument(
        "--semantic-components", type=int, default=96,
        help="Number of LSA components for the semantic similarity model.",
    )
    parser.add_argument(
        "--max-features", type=int, default=30_000,
        help="Vocabulary cap for the TF-IDF vectorizer (memory control).",
    )
    args = parser.parse_args()

    t_start = time.time()

    # ---- Pass 1: stream through once, building only small, scalar feature
    #      rows and the text corpus needed to fit the semantic model. We do
    #      NOT retain the original JSON record for every candidate -- at
    #      100K candidates that would be the single biggest memory cost in
    #      the whole pipeline for no benefit, since we only need full profile
    #      detail for the ~100 candidates that end up in the output. ----
    feature_rows: list[dict] = []
    docs: list[str] = []
    honeypot_scores: list[int] = []

    n = 0
    for cand in iter_candidates(args.candidates):
        feature_rows.append(extract_features(cand))
        docs.append(candidate_doc(cand))
        honeypot_scores.append(honeypot_score(cand))
        n += 1

    print(f"[rank.py] pass 1: extracted features for {n} candidates in {time.time() - t_start:.1f}s", file=sys.stderr)

    # ---- Fit the semantic model on the full corpus, score every doc against
    #      the JD once. ----
    t0 = time.time()
    semantic = SemanticScorer(n_components=args.semantic_components, max_features=args.max_features)
    semantic.fit(docs)
    semantic_scores = semantic.score_against_jd(docs)
    print(f"[rank.py] semantic model fit+score in {time.time() - t0:.1f}s", file=sys.stderr)
    del docs  # free the raw text corpus; no longer needed

    # ---- Fuse into final scores. ----
    t0 = time.time()
    scored = []
    for feats, sem_score, hp_score in zip(feature_rows, semantic_scores, honeypot_scores):
        fs = final_score(feats, float(sem_score), hp_score)
        # Round to the same precision we write to the CSV (4 dp) *before*
        # sorting/tie-breaking. If we sorted on the unrounded float and two
        # candidates' raw scores both happen to round to the same displayed
        # value, the displayed scores would tie at ranks where our sort
        # order doesn't follow the required candidate_id-ascending tie-break
        # rule (submission_spec.md Section 3) -- this rounds first so the
        # value we sort on and the value we print are identical.
        scored.append((round(fs, 4), feats["candidate_id"], hp_score))
    print(f"[rank.py] scored {len(scored)} candidates in {time.time() - t0:.1f}s", file=sys.stderr)

    # ---- Rank: sort by score desc, tie-break by candidate_id ascending
    #      (matches submission_spec.md tie-break rule). ----
    scored.sort(key=lambda r: (-r[0], r[1]))
    top = scored[:TOP_N]
    top_ids = {cid for _, cid, _ in top}

    # Keep only the feature rows we'll actually need for reasoning text.
    feats_by_id = {f["candidate_id"]: f for f in feature_rows if f["candidate_id"] in top_ids}
    del feature_rows

    # ---- Pass 2: re-stream the file once more, this time just to pull the
    #      compact profile projection for the ~100 candidates that made the
    #      cut. This keeps peak memory bounded by TOP_N regardless of pool
    #      size, at the cost of one extra (cheap, I/O-bound) file read. ----
    t0 = time.time()
    raw_for_reasoning: dict[str, dict] = {}
    for cand in iter_candidates(args.candidates):
        if cand["candidate_id"] in top_ids:
            raw_for_reasoning[cand["candidate_id"]] = _compact_projection(cand)
            if len(raw_for_reasoning) == len(top_ids):
                break
    print(f"[rank.py] pass 2: collected profile detail for top {len(raw_for_reasoning)} in {time.time() - t0:.1f}s", file=sys.stderr)

    # ---- Write CSV. ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, hp_score) in enumerate(top, start=1):
            reasoning = build_reasoning(raw_for_reasoning[cid], feats_by_id[cid], hp_score, rank)
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    elapsed = time.time() - t_start
    print(f"[rank.py] wrote top {len(top)} candidates to {args.out} in {elapsed:.1f}s total", file=sys.stderr)


def _compact_projection(cand: dict) -> dict:
    """Keep only what reasoning.py needs, to bound memory on the full pool."""
    p = cand["profile"]
    return {
        "candidate_id": cand["candidate_id"],
        "current_title": p.get("current_title"),
        "current_company": p.get("current_company"),
        "years_of_experience": p.get("years_of_experience"),
        "location": p.get("location"),
        "country": p.get("country"),
        "top_skills": sorted(
            (cand.get("skills") or []),
            key=lambda s: -(s.get("endorsements", 0) or 0),
        )[:6],
        "redrob_signals": {
            k: cand.get("redrob_signals", {}).get(k)
            for k in ("recruiter_response_rate", "notice_period_days", "last_active_date", "open_to_work_flag")
        },
    }


if __name__ == "__main__":
    main()
