"""
Streamlit sandbox app for the Redrob Hackathon submission.

Per submission_spec.md Section 10.5, this sandbox accepts a small candidate
sample, runs the full ranking pipeline end-to-end, and produces a ranked CSV
-- all within the same compute budget (CPU only, no network, fast) as the
real ranking step, just over a smaller pool so it stays responsive in a
free-tier hosted environment.

Run locally with:
    streamlit run sandbox/app.py

Deploy on Streamlit Community Cloud by pointing it at this file with
requirements.txt at the repo root.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from features import extract_features  # noqa: E402
from honeypot import honeypot_score  # noqa: E402
from reasoning import build_reasoning  # noqa: E402
from scoring import final_score  # noqa: E402
from semantic import SemanticScorer, candidate_doc  # noqa: E402

st.set_page_config(page_title="Redrob Candidate Ranker — Sandbox", layout="wide")

st.title("🎯 Redrob Candidate Ranker — Sandbox")
st.caption(
    "Intelligent Candidate Discovery & Ranking Challenge — small-sample demo. "
    "CPU-only, no GPU, no network calls during ranking, matching the hackathon's compute constraints."
)

with st.expander("ℹ️ What is this?", expanded=False):
    st.markdown(
        """
This is the **reproducibility sandbox** for our hackathon submission, not the full
100K-candidate production run (see `src/rank.py` for that — same scoring code,
just over the full pool).

Upload a small `.jsonl` sample of candidates (matching `candidate_schema.json`),
or use the bundled sample, and this app will run the exact same ranking
pipeline used to produce our official submission CSV: structured JD-fit
features + a locally-fit TF-IDF/LSA semantic layer + a behavioral-signal
modifier + an honeypot/implausibility penalty, fused into one final score
with a grounded, fact-based reasoning string per candidate.
        """
    )

DEFAULT_SAMPLE_PATH = Path(__file__).parent.parent / "data" / "sample_candidates.json"

source = st.radio(
    "Candidate sample source",
    ["Use bundled sample_candidates.json", "Upload my own .json / .jsonl"],
    horizontal=True,
)

candidates: list[dict] = []

if source == "Use bundled sample_candidates.json":
    if DEFAULT_SAMPLE_PATH.exists():
        with open(DEFAULT_SAMPLE_PATH, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        st.success(f"Loaded {len(candidates)} bundled sample candidates.")
    else:
        st.error(
            "data/sample_candidates.json not found in this deployment. "
            "Upload a sample instead, or add the file to the repo's data/ folder."
        )
else:
    uploaded = st.file_uploader("Upload candidates (.json array or .jsonl)", type=["json", "jsonl"])
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        try:
            candidates = json.loads(raw)
            if isinstance(candidates, dict):
                candidates = [candidates]
        except json.JSONDecodeError:
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
        st.success(f"Loaded {len(candidates)} uploaded candidates.")

top_n = st.slider("How many top candidates to show", min_value=5, max_value=100, value=20)

if candidates and st.button("🚀 Run ranking", type="primary"):
    if len(candidates) > 100:
        st.info(
            f"Sandbox is scoped to small samples per Section 10.5 of the spec; "
            f"truncating {len(candidates)} candidates down to the first 100 for this demo run."
        )
        candidates = candidates[:100]

    t0 = time.time()
    progress = st.progress(0.0, text="Extracting features...")

    feature_rows = [extract_features(c) for c in candidates]
    progress.progress(0.4, text="Fitting local semantic model (TF-IDF + LSA)...")

    docs = [candidate_doc(c) for c in candidates]
    n_components = min(64, max(2, len(docs) - 1))
    semantic = SemanticScorer(n_components=n_components)
    semantic.fit(docs)
    sem_scores = semantic.score_against_jd(docs)
    progress.progress(0.7, text="Scoring and fusing signals...")

    hp_scores = [honeypot_score(c) for c in candidates]

    results = []
    for c, feats, sem_score, hp in zip(candidates, feature_rows, sem_scores, hp_scores):
        score = round(final_score(feats, float(sem_score), hp), 4)
        results.append((score, c, feats, hp))

    results.sort(key=lambda r: (-r[0], r[1]["candidate_id"]))
    progress.progress(1.0, text="Done.")
    elapsed = time.time() - t0

    st.success(f"Ranked {len(candidates)} candidates in {elapsed:.2f}s (CPU, no network).")

    rows = []
    for rank, (score, c, feats, hp) in enumerate(results[:top_n], start=1):
        compact_profile = {
            "candidate_id": c["candidate_id"],
            "current_title": c["profile"].get("current_title"),
            "current_company": c["profile"].get("current_company"),
            "years_of_experience": c["profile"].get("years_of_experience"),
            "top_skills": sorted(c.get("skills", []), key=lambda s: -(s.get("endorsements", 0) or 0))[:6],
            "redrob_signals": c.get("redrob_signals", {}),
        }
        reasoning = build_reasoning(compact_profile, feats, hp, rank)
        rows.append(
            {
                "rank": rank,
                "candidate_id": c["candidate_id"],
                "score": score,
                "title": c["profile"].get("current_title"),
                "company": c["profile"].get("current_company"),
                "location": c["profile"].get("location"),
                "yoe": c["profile"].get("years_of_experience"),
                "honeypot_score": hp,
                "reasoning": reasoning,
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download ranked CSV", data=csv_bytes, file_name="sandbox_ranking.csv", mime="text/csv")

elif not candidates:
    st.info("Load a candidate sample above, then click **Run ranking**.")
