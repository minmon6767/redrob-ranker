"""
Final score fusion: combine structured JD-fit features, the local semantic
score, the honeypot/implausibility penalty, and the behavioral-signal
multiplier into one final ranking score.

Architecture rationale (this is the thing we defend at the Stage 5 interview):

  final_score = jd_fit_score * behavioral_modifier * honeypot_penalty

  - jd_fit_score is itself a weighted blend of structured features (title,
    YOE band, must-have skill coverage trust-weighted against keyword
    stuffing, applied-ML career fraction, location) and the semantic LSA
    similarity to the JD text. It captures "does this person's *career*
    look like the JD" independent of whether they're currently active.

  - behavioral_modifier (from features.behavioral_signal_features) is
    MULTIPLICATIVE, not additive, on purpose: the JD explicitly frames
    availability/engagement as a hiring-feasibility gate ("for hiring
    purposes, not actually available"), not as a competing dimension of
    quality. A great-on-paper but completely unreachable candidate should
    be discounted relative to an equally-great, reachable one -- but
    behavioral signals alone should never be able to push a fundamentally
    unqualified candidate to the top.

  - honeypot_penalty is a hard multiplicative discount (not a binary
    drop) on candidates whose profile shows internal inconsistencies. We
    discount rather than zero-out because the detector is heuristic and
    imperfect -- see honeypot.py for the documented reasoning -- and a
    false positive should hurt a candidate's rank, not silently vanish them.

  - disqualifier penalties (pure-research-only, consulting-only, etc.) are
    applied as multiplicative discounts on the structured component, scaled
    to match how strongly the JD states each one ("we will not move
    forward" vs "we will probably not move forward").
"""
from __future__ import annotations

from typing import Any

# Weights for the structured-feature blend that produces jd_fit_structured.
# These sum to 1.0 and were chosen to mirror the JD's own emphasis: the
# "Things you absolutely need" section names 4 must-have skill areas as the
# clear top priority, title/seniority and applied-ML fraction operationalize
# "How to read between the lines", and location/education are explicitly
# secondary per the JD's own text ("skills are teachable", location is
# "preferred but flexible").
STRUCTURED_WEIGHTS = {
    "must_have_coverage": 0.34,
    "nice_to_have_coverage": 0.06,
    "title_score": 0.16,
    "yoe_score": 0.10,
    "applied_ml_fraction": 0.16,
    "location_score": 0.10,
    "education_tier_score": 0.04,
    "semantic_score": 0.04,  # small explicit slice; semantic is mainly used as a tiebreaker / sanity check, see below
}

# Final blend between the structured score and the semantic-similarity score.
# Structured features are kept dominant because they are auditable and
# grounded in named facts (what the Stage 4 reasoning review wants); the
# semantic score is a softer "does the overall narrative sound right" signal
# layered on top, and also serves as a check against candidates who pattern
# match structurally but whose actual career narrative reads as off-topic
# (e.g. a "Senior AI Engineer" title with no ranking/retrieval substance).
SEMANTIC_BLEND_WEIGHT = 0.22
STRUCTURED_BLEND_WEIGHT = 1.0 - SEMANTIC_BLEND_WEIGHT

DISQUALIFIER_PENALTIES = {
    "pure_research_penalty": 0.35,        # JD states this in absolute terms -- heavy discount
    "consulting_only_penalty": 0.55,      # "probably not" -- significant but not crushing
    "cv_speech_robotics_penalty": 0.55,
    "title_chaser_penalty": 0.70,
    "architecture_only_penalty": 0.70,
}

HONEYPOT_PENALTY_BY_SCORE = {
    0: 1.0, 1: 1.0,
    2: 0.45, 3: 0.30,
    4: 0.15,
}


def structured_jd_fit(features: dict[str, Any]) -> float:
    score = sum(STRUCTURED_WEIGHTS[k] * features.get(k, 0.0) for k in STRUCTURED_WEIGHTS if k != "semantic_score")
    score += STRUCTURED_WEIGHTS["semantic_score"] * features.get("semantic_score", 0.0)
    return score


def disqualifier_multiplier(features: dict[str, Any]) -> float:
    mult = 1.0
    for key, penalty_if_triggered in DISQUALIFIER_PENALTIES.items():
        if features.get(key, 0.0) >= 1.0:
            mult *= penalty_if_triggered
    return mult


def honeypot_multiplier(honeypot_score: int) -> float:
    return HONEYPOT_PENALTY_BY_SCORE.get(honeypot_score, 0.1)


def final_score(features: dict[str, Any], semantic_score: float, honeypot_score: int) -> float:
    feats = dict(features)
    feats["semantic_score"] = semantic_score

    structured = structured_jd_fit(feats)
    blended = STRUCTURED_BLEND_WEIGHT * structured + SEMANTIC_BLEND_WEIGHT * semantic_score

    blended *= disqualifier_multiplier(feats)
    blended *= feats.get("behavioral_modifier", 1.0)
    blended *= honeypot_multiplier(honeypot_score)

    return max(0.0, min(1.0, blended))
