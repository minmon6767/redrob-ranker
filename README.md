# Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge

**Team:** Pranjal (solo)
**Challenge:** Rank the top 100 best-fit candidates (out of 100,000) for Redrob's
"Senior AI Engineer — Founding Team" job description, beyond keyword matching.

This repo contains the full ranking system: feature engineering, a locally-fit
semantic similarity model, a behavioral-signal modifier, a honeypot/implausible-profile
detector, and a deterministic, grounded reasoning generator — all running CPU-only,
offline, in well under the hackathon's 5-minute / 16GB compute budget.

---

## TL;DR — reproduce the submission

```bash
pip install -r requirements.txt
python src/rank.py --candidates ./data/candidates.jsonl --out ./artifacts/submission.csv
python data/validate_submission.py ./artifacts/submission.csv  # organizer's validator
```

On a single CPU core with 4GB RAM (i.e. *less* than the 16GB/CPU-only budget the
challenge allows), this completes in **~105 seconds** end-to-end for the full
100K-candidate pool. No GPU, no network calls, no LLM API calls of any kind
happen during ranking.

## Getting the data

`data/candidates.jsonl` (the 100K-candidate pool, ~465MB) is **not** committed
to this repo — it's the organizer-provided hackathon dataset, not our own
artifact, and is well past reasonable repo size norms. Drop the file the
organizers released into `data/candidates.jsonl` before running `rank.py`.
Everything else needed to run the pipeline (the schema, sample candidates,
JD, signals doc) is included for reference under `data/`.

---

## Why this architecture

The JD is explicit that it does **not** want a keyword filter:

> "The 'right answer' to this JD is not 'find candidates whose skills section
> contains the most AI keywords.' ... Your ranking system should also weigh
> behavioral signals."

And the compute constraints (`submission_spec.md` §3) explicitly rule out
calling a hosted LLM per candidate:

> "Plan for a small ranker over precomputed features, indexes, or compact
> local models."

So the system is built as a **transparent, multi-signal hybrid ranker**, not a
single model and not an LLM pipeline:

```
                ┌─────────────────────┐
   candidate ─▶ │  structured features │ ── title/seniority, must-have skill
                │   (features.py)      │    coverage (trust-weighted against
                └─────────┬────────────┘    keyword stuffing), applied-ML career
                          │                 fraction, location/relocation fit,
                          │                 disqualifier flags (research-only,
                          │                 consulting-only, CV/speech/robotics,
                          │                 title-chasing, architecture-only)
                          │
                ┌─────────▼────────────┐
   candidate ─▶ │  semantic similarity  │ ── local TF-IDF + truncated SVD (LSA),
                │   (semantic.py)       │    fit on the candidate pool itself,
                └─────────┬────────────┘    cosine-similarity to the JD text
                          │
                ┌─────────▼────────────┐
   candidate ─▶ │  behavioral modifier  │ ── recency, recruiter response rate,
                │  (features.py)        │    notice period, availability —
                └─────────┬────────────┘    MULTIPLICATIVE, not additive
                          │
                ┌─────────▼────────────┐
   candidate ─▶ │  honeypot detector    │ ── profile-internal inconsistency
                │  (honeypot.py)        │    score (skill duration vs YOE,
                └─────────┬────────────┘    expert-with-no-time-invested,
                          │                 summary-vs-structured YOE mismatch)
                          ▼
                ┌──────────────────────┐
                │   score fusion        │ ── scoring.py: weighted blend +
                │   (scoring.py)        │    multiplicative penalties, bounded [0,1]
                └─────────┬────────────┘
                          ▼
                ┌──────────────────────┐
                │  reasoning generator  │ ── deterministic, fact-grounded,
                │  (reasoning.py)       │    1-2 sentence justification per
                └──────────────────────┘    candidate — no LLM, no hallucination
```

### Design decisions, and why

**No neural embeddings / no sentence-transformers.** The compute spec
requires CPU-only, no-GPU, no-network ranking within 5 minutes. Downloading a
pretrained checkpoint at ranking time would also violate "no network during
ranking" unless the weights are already vendored, and vendoring a multi-hundred-MB
checkpoint into a hackathon repo is its own liability. Instead, `semantic.py`
fits a TF-IDF + truncated-SVD (LSA) model **from scratch on the candidate pool
itself**. This is fully local, fits + scores 100K candidates in well under
90 seconds on one CPU core, and is fully inspectable (no opaque pretrained
weights to "trust" — the vocabulary and components are in the code path).
This is a deliberate, documented tradeoff, not a missing feature.

**Structured features are weighted more heavily than the semantic score.**
`scoring.py`'s blend gives ~78% weight to structured, auditable features and
~22% to semantic similarity. This is intentional: the Stage 4 manual review
specifically checks whether reasoning references *specific facts* from the
profile, and structured features are exactly the facts that produce that
kind of reasoning. The semantic layer acts as a softer "does the overall
narrative read right" check layered on top — and in practice it also catches
candidates who pattern-match structurally (right title, right years) but
whose actual career narrative doesn't read as retrieval/ranking work.

**Skill matching is trust-weighted, not boolean.** `features.skill_trust()`
combines proficiency, endorsement count, and duration so a candidate who
lists "RAG" as "expert" with 0 endorsements and 1 month of use scores very
differently from one with "advanced" + 30 endorsements + 24 months. This is
the direct counter to keyword-stuffing — the JD's own stated trap.

**Behavioral signals are a multiplier, not an additive score.** The JD frames
availability as a hiring-feasibility gate, not a competing axis of quality:
"a perfect-on-paper candidate who hasn't logged in for 6 months ... is, for
hiring purposes, not actually available." A multiplicative modifier can
meaningfully discount an unreachable candidate but can never by itself push
an unqualified candidate to the top — see `scoring.py` docstring for the
full rationale.

**Honeypot detection is a documented heuristic, not a guess.** See
`honeypot.py`'s docstring and `notebooks/honeypot_exploration.md` for the
empirical process: we checked education-date impossibilities and career-history
date overlaps and found the dataset clean on both. Three channels *do* show a
genuine, sharply-separated outlier population: (1) skill duration far
exceeding total years of experience, (2) multiple skills simultaneously
claimed at "expert" level with near-zero time invested, and (3) a candidate's
own free-text summary stating an explicit "N years of experience" that
disagrees with the structured field by more than 1.5 years (this last one
hits only 16 of 100,000 candidates — a clean, deliberate-looking signal, not
noise). All three combine into a composite score that *discounts* rather than
hard-zeros a candidate, since the detector is heuristic and we'd rather
under-penalize a false positive than silently disappear a real candidate.

**Disqualifiers are soft, multiplicative penalties, calibrated to the JD's
own language.** The JD says "we will not move forward" for pure-research-only
backgrounds (treated as a hard, heavy penalty) versus "we will probably not
move forward" for consulting-only, CV/speech/robotics-only, etc. (treated as
significant but recoverable penalties if other signals are strong). See
`jd_spec.py` and `scoring.DISQUALIFIER_PENALTIES`.

**Reasoning is template-resistant by construction, not by prompting an LLM.**
`reasoning.py` has no LLM call anywhere — every clause is string-formatted
from the candidate's own fields, so it's structurally impossible for it to
hallucinate a skill or employer that isn't in the record. Phrasing is picked
from small synonym pools keyed by a deterministic hash of `candidate_id`, so
wording varies across the 100 rows without introducing randomness (same input
→ same output, every run). Concerns (long notice period, inactivity, location
mismatch, honeypot flags, YOE outside the JD's stated band) are surfaced
explicitly rather than smoothed over, matching the "Honest concerns" check in
`submission_spec.md` §3.

---

## Repo layout

```
src/
  jd_spec.py     — structured JD requirements (the single source of truth for weights)
  features.py    — per-candidate structured feature extraction
  semantic.py    — local TF-IDF + LSA semantic similarity to the JD
  honeypot.py    — profile-internal-inconsistency / honeypot detector
  scoring.py     — score fusion (structured + semantic + behavioral + honeypot)
  reasoning.py   — deterministic, fact-grounded reasoning string generator
  rank.py        — main entry point; produces the submission CSV
tests/           — unit tests (honeypot, features, scoring) + an end-to-end format test
sandbox/         — Streamlit demo app (small-sample reproducibility sandbox)
data/            — schema, sample candidates, JD text, validator (candidates.jsonl excluded, see above)
artifacts/       — the submitted ranking CSV
submission_metadata.yaml — portal metadata mirror, per submission_spec.md §10.3
```

## Running the tests

```bash
pip install -r requirements.txt pytest
python -m pytest tests/ -v
```

26 tests covering the honeypot detector, feature extraction, score fusion,
and an end-to-end run of `rank.py` that checks the output against the actual
submission format rules (exactly 100 rows, ranks 1-100 each exactly once,
unique candidate_ids, non-increasing scores, non-empty reasoning).

## Running the sandbox demo

```bash
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```

Loads a small candidate sample (bundled `data/sample_candidates.json`, or
your own upload), runs the exact same scoring code as `rank.py`, and lets
you download a ranked CSV. This is the sandbox referenced in
`submission_metadata.yaml` for Stage 1/10.5 reproducibility.

## Compute environment this was developed and timed on

Single CPU core, ~4GB RAM container (Python 3.12, scikit-learn 1.8) — i.e.
*tighter* than the hackathon's stated 16GB/CPU-only budget. Full 100K-pool
run: **~105 seconds wall-clock**, peak memory well under 4GB. See
`submission_metadata.yaml` for the exact declared compute environment.
