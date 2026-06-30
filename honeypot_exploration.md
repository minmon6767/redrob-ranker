# Honeypot Exploration Notes

This documents the empirical process behind `src/honeypot.py`'s detection
rules. We did not special-case any `candidate_id` — per `submission_spec.md`
§7, "you don't need to special-case them" — instead we looked for
profile-internal inconsistencies that would generalize to any pool, not just
this one.

## Channels checked and found clean (no signal)

**Education date impossibilities.** Checked every candidate for
`end_year < start_year` or `end_year` in the implausible future. Zero hits
across all 100,000 candidates.

```python
for e in candidate.get("education", []):
    if e["end_year"] < e["start_year"]: ...   # 0 hits
    if e["end_year"] > 2026: ...               # 0 hits
```

**Career-history date overlaps.** Checked every candidate's career history
for overlapping employment date ranges (allowing a 30-day grace window for
rounding). Zero hits.

**Skill-assessment-score vs proficiency mismatch.** Checked whether a
candidate's self-reported proficiency ("expert") contradicted their
`skill_assessment_scores` (e.g. expert + assessment score < 30). Zero hits —
this channel is internally consistent in the dataset as released.

## Channels checked and found a real, separable signal

**1. Skill duration exceeding total years of experience.**

```python
margin = max(skill.duration_months for skill in skills) / 12 - years_of_experience
```

The raw distribution of this margin across the full pool is *continuous*,
not bimodal — about 18% of candidates have *some* positive margin, which
reflects ordinary rounding/data-generation noise (a skill picked up
fractionally before the YOE cutoff), not a deliberate trap. We found no
clean cliff. We chose a conservative threshold (`margin > 3` years, with an
extra point for `> 4` years) that flags a small population at the
long tail rather than the ~18% with marginal overshoot.

| Percentile | Margin (years) |
|---|---|
| 90th | 0.92 |
| 95th | 1.58 |
| 99th | 2.80 |
| 99.9th | 3.65 |
| max | 4.93 |

**2. Multiple "expert" skills claimed with near-zero duration.**

```python
low_duration_experts = [s for s in skills if s.proficiency == "expert" and s.duration_months <= 3]
```

Requiring **5+ such skills simultaneously** on one profile produced exactly
**8 candidates** out of 100,000 — a small, sharply separated population, not
noise. (At 3+ skills the count balloons; we use ≥4 as the flag threshold with
an extra point at ≥6, balancing precision against not being so strict that
we miss real cases.)

**3. Free-text summary states an explicit YOE that disagrees with the
structured field.**

```python
match = re.search(r"(\d+(?:\.\d+)?)\s+years of experience", summary)
mismatch = abs(float(match.group(1)) - years_of_experience) if match else 0
```

This is the cleanest signal found: only **16 of 100,000 candidates (0.02%)**
have a summary-stated YOE that disagrees with the structured field by more
than 1.5 years, and the disagreement is large where it occurs (several are
off by 8-10+ years). The affected titles skew heavily toward AI-adjacent
roles (NLP Engineer, AI Engineer, Recommendation Systems Engineer, Applied ML
Engineer) — consistent with a trap deliberately placed to catch a ranker that
trusts the structured `years_of_experience` field (or the free-text summary)
in isolation without cross-checking the two against each other.

## What we did NOT do

We deliberately did not try to reverse-engineer the literal "8 years at a
company founded 3 years ago" example from `submission_spec.md` §7 by
inferring company founding dates from earliest-observed employee start
dates in the pool — that proxy is noisy (a company's *first hire visible in
this 100K sample* is not its founding date) and chasing it produced no
clean signal when tried. Rather than force a brittle, low-precision rule
into the detector to match one illustrative example from the spec, we kept
the three documented, empirically-validated channels above and accept that
our honeypot rate in any submission may not be exactly zero — the goal is
staying comfortably under the 10%-in-top-100 disqualification threshold
(which it does — 0 honeypots detected in our final top-100), not claiming
perfect detection.
