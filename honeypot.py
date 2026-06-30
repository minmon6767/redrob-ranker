"""
Honeypot detection: catches "subtly impossible" candidate profiles.

The hackathon dataset embeds ~80 honeypot candidates (out of 100,000) with
profiles that are internally inconsistent in ways a careless keyword/embedding
ranker won't notice but a system that actually reads the profile will.

We do NOT special-case candidate_ids -- per the spec, "you don't need to
special-case them," and doing so would be a form of overfitting to this one
dataset rather than building a system that generalizes. Instead we compute a
composite implausibility score from profile-internal inconsistencies. This
score is then used as a *penalty* in the final ranking (heavily discounting,
not necessarily zeroing, since a few of these signals can fire on legitimately
unusual-but-real profiles too -- we'd rather be conservative than blackbox-drop
a real candidate).

Empirically (validated against the released 100K pool, see notebooks/
honeypot_exploration.md) the dataset does NOT contain education-date
impossibilities or career-history date overlaps -- those channels are clean.
The channels that DO show a genuine outlier population, cleanly separable
from natural data noise, are:

  1. A skill's claimed duration_months exceeds the candidate's total
     years_of_experience by an implausible margin. Small margins (a few
     months) occur naturally from rounding; we only flag large margins.
  2. Multiple skills are simultaneously claimed at "expert" proficiency with
     near-zero duration_months -- i.e., "expert" skill the candidate has
     supposedly used for under 3 months. One such skill could be a data
     artifact; several together is a strong inconsistency signal.
  3. The free-text summary states an explicit "N years of experience" that
     disagrees with the structured years_of_experience field by more than
     1.5 years. This affects only 16 of 100,000 candidates (0.02%) -- a
     sharp, clean outlier population, not noise -- and is weighted toward
     AI-adjacent titles, consistent with a deliberately placed trap aimed
     at exactly the candidates a naive AI-keyword ranker would otherwise
     surface near the top.
"""
from __future__ import annotations

import re
from typing import Any

_SUMMARY_YOE_RE = re.compile(r"(\d+(?:\.\d+)?)\s+years of experience")


def skill_duration_margin_years(candidate: dict[str, Any]) -> float:
    """Largest (skill claimed duration - total YOE), in years. Negative = consistent."""
    yoe = candidate["profile"].get("years_of_experience", 0) or 0
    skills = candidate.get("skills", []) or []
    if not skills:
        return -999.0
    max_dur_months = max((s.get("duration_months", 0) or 0) for s in skills)
    return (max_dur_months / 12.0) - yoe


def count_low_duration_experts(candidate: dict[str, Any], max_months: int = 3) -> int:
    """Number of skills claimed at 'expert' proficiency with <= max_months duration."""
    skills = candidate.get("skills", []) or []
    return sum(
        1
        for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months", 999) or 999) <= max_months
    )


def summary_yoe_mismatch(candidate: dict[str, Any]) -> float:
    """
    Absolute gap (years) between an explicit "N years of experience" claim
    in the free-text summary and the structured years_of_experience field.
    Returns 0.0 if no explicit claim is found in the summary text.
    """
    summary = candidate["profile"].get("summary", "") or ""
    match = _SUMMARY_YOE_RE.search(summary)
    if not match:
        return 0.0
    stated = float(match.group(1))
    structured = candidate["profile"].get("years_of_experience", stated) or stated
    return abs(stated - structured)


def honeypot_score(candidate: dict[str, Any]) -> int:
    """
    Composite implausibility score, roughly 0-5+.
    0-1: looks normal. 2+: flagged as likely honeypot / unreliable profile.

    Thresholds were chosen empirically against the released 100K pool to land
    on a tight, plausible-sized flagged set (see notebooks/honeypot_exploration.md)
    rather than catching the ~18% of candidates whose skill duration loosely
    overshoots YOE through ordinary rounding noise.
    """
    score = 0
    margin = skill_duration_margin_years(candidate)
    if margin > 3:
        score += 1
    if margin > 4:
        score += 1

    low_dur_experts = count_low_duration_experts(candidate)
    if low_dur_experts >= 4:
        score += 2
    if low_dur_experts >= 6:
        score += 1

    if summary_yoe_mismatch(candidate) > 1.5:
        score += 3  # this channel alone is a clean, high-precision signal

    return score


def is_likely_honeypot(candidate: dict[str, Any], threshold: int = 2) -> bool:
    return honeypot_score(candidate) >= threshold
