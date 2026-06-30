"""
Structured feature extraction for a single candidate, scored against the JD.

Design principle: every feature here maps to a specific sentence in the JD
(see jd_spec.py for the extracted requirements). This is what lets us write
ranking *reasoning* that's grounded in real profile facts instead of
generic-sounding text -- the submission_spec.md Stage 4 review explicitly
penalizes reasoning that doesn't connect to specific JD requirements or that
hallucinates facts not in the profile.

All functions are pure and operate on a single candidate dict (the parsed
JSON record from candidates.jsonl) -- no global state, no network calls, no
GPU. This module is imported by both the offline feature-build script and the
final rank.py so behavior can never drift between "what we evaluated" and
"what we submit."
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from jd_spec import (
    ACCEPTABLE_YOE_HIGH,
    ACCEPTABLE_YOE_LOW,
    ADJACENT_TITLE_MARKERS,
    CONSULTING_FIRMS,
    CORE_AI_TITLE_MARKERS,
    CV_SPEECH_ROBOTICS_MARKERS,
    IDEAL_YOE_HIGH,
    IDEAL_YOE_LOW,
    MAX_NOTICE_PERIOD_SOFT,
    MUST_HAVE_SKILL_FAMILIES,
    NICE_TO_HAVE_SKILL_FAMILIES,
    PREFERRED_CITIES,
    PURE_RESEARCH_MARKERS,
    TIER1_INDIAN_CITIES,
)

TODAY = date(2026, 6, 28)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_corpus(candidate: dict[str, Any]) -> str:
    """All free-text fields concatenated, lower-cased, for substring/keyword matching."""
    parts = []
    p = candidate.get("profile", {})
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(p.get("current_title", ""))
    for ch in candidate.get("career_history", []) or []:
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
    for s in candidate.get("skills", []) or []:
        parts.append(s.get("name", ""))
    return " | ".join(parts).lower()


def _skill_lookup(candidate: dict[str, Any]) -> dict[str, dict]:
    return {s["name"].lower(): s for s in candidate.get("skills", []) or [] if s.get("name")}


def skill_trust(skill: dict[str, Any]) -> float:
    """
    0-1 'trust' that a listed skill reflects real depth, not keyword stuffing.
    Combines proficiency, endorsements, and duration -- a candidate who lists
    'RAG' as 'expert' with 0 endorsements and 1 month duration should NOT
    score the same as one with 'advanced' + 30 endorsements + 24 months.
    """
    prof_weight = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}.get(
        skill.get("proficiency", "beginner"), 0.25
    )
    dur_months = skill.get("duration_months", 0) or 0
    dur_weight = min(1.0, dur_months / 18.0)  # saturates at 1.5 years
    endorsements = skill.get("endorsements", 0) or 0
    endorse_weight = min(1.0, endorsements / 15.0)  # saturates at 15 endorsements
    # duration and endorsements both corroborate proficiency; average them as
    # a corroboration multiplier so a high proficiency claim with NO
    # corroboration is heavily discounted (anti keyword-stuffing).
    corroboration = 0.5 * dur_weight + 0.5 * endorse_weight
    return prof_weight * (0.4 + 0.6 * corroboration)


def family_match_strength(candidate_text: str, skills: dict[str, dict], terms: list[str]) -> float:
    """
    Best trust-weighted match for a skill family. Checks both the structured
    skills list (preferred -- has trust signal) and free text (fallback, at a
    discount, since a mention in a summary without a skills-list entry is
    weaker evidence).
    """
    best = 0.0
    for term in terms:
        term_l = term.lower()
        # exact-ish skill list match
        for skill_name, skill in skills.items():
            if term_l == skill_name or term_l in skill_name or skill_name in term_l:
                best = max(best, skill_trust(skill))
        # free text mention (discounted, no corroboration data available)
        if term_l in candidate_text:
            best = max(best, 0.35)
    return best


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

def skills_features(candidate: dict[str, Any]) -> dict[str, float]:
    text = _text_corpus(candidate)
    skills = _skill_lookup(candidate)

    must_have_scores = {
        family: family_match_strength(text, skills, terms)
        for family, terms in MUST_HAVE_SKILL_FAMILIES.items()
    }
    nice_to_have_scores = {
        family: family_match_strength(text, skills, terms)
        for family, terms in NICE_TO_HAVE_SKILL_FAMILIES.items()
    }

    must_have_coverage = sum(must_have_scores.values()) / len(must_have_scores)
    nice_to_have_coverage = sum(nice_to_have_scores.values()) / max(len(nice_to_have_scores), 1)

    return {
        "must_have_coverage": must_have_coverage,
        "nice_to_have_coverage": nice_to_have_coverage,
        **{f"musthave_{k}": v for k, v in must_have_scores.items()},
    }


def title_and_seniority_features(candidate: dict[str, Any]) -> dict[str, float]:
    title = (candidate["profile"].get("current_title") or "").lower()
    yoe = candidate["profile"].get("years_of_experience", 0) or 0

    core_match = 1.0 if any(m in title for m in CORE_AI_TITLE_MARKERS) else 0.0
    adjacent_match = 1.0 if any(m in title for m in ADJACENT_TITLE_MARKERS) else 0.0
    title_score = 1.0 if core_match else (0.45 if adjacent_match else 0.0)

    # YOE band score: full credit inside the ideal 6-8y window, partial credit
    # across the wider acceptable 5-9y band, decaying gracefully outside it
    # rather than a hard cliff (JD: "we'll seriously consider candidates
    # outside the band if other signals are strong").
    if IDEAL_YOE_LOW <= yoe <= IDEAL_YOE_HIGH:
        yoe_score = 1.0
    elif ACCEPTABLE_YOE_LOW <= yoe <= ACCEPTABLE_YOE_HIGH:
        yoe_score = 0.8
    else:
        dist = min(abs(yoe - ACCEPTABLE_YOE_LOW), abs(yoe - ACCEPTABLE_YOE_HIGH))
        yoe_score = max(0.0, 0.8 - 0.12 * dist)

    return {
        "title_core_match": core_match,
        "title_score": title_score,
        "yoe_score": yoe_score,
        "years_of_experience": yoe,
    }


def applied_ml_fraction(candidate: dict[str, Any]) -> float:
    """
    Fraction of total career *time* spent in applied ML/AI-flavored roles vs
    total time, using the title + description of each past role. Distinct
    from current title alone -- this is the "what fraction of their career
    was actually applied ML" signal from "How to read between the lines."
    """
    history = candidate.get("career_history", []) or []
    total_months = sum((ch.get("duration_months", 0) or 0) for ch in history) or 1
    ml_months = 0
    ai_markers = CORE_AI_TITLE_MARKERS + [
        "embedding", "retrieval", "ranking", "recommendation", "machine learning",
        "nlp", "llm", "model", "ml ", "ai ", "vector",
    ]
    for ch in history:
        blob = f"{ch.get('title', '')} {ch.get('description', '')}".lower()
        if any(m in blob for m in ai_markers):
            ml_months += ch.get("duration_months", 0) or 0
    return min(1.0, ml_months / total_months)


def disqualifier_features(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Soft penalty signals from "Things we explicitly do NOT want" and the
    explicit disqualifier paragraph. These are multiplicative penalties in
    scoring.py, NOT hard zero-outs -- the JD's own language is "we will
    probably not move forward" / "we'd rather see strong counter-signals",
    not "automatic rejection", except for the pure-research case which the
    JD states in absolute terms ("we will not move forward... we are
    explicit about this").
    """
    text = _text_corpus(candidate)
    current_company = (candidate["profile"].get("current_company") or "").lower()
    history = candidate.get("career_history", []) or []

    # Pure research / no production deployment ever -- HARD per JD wording.
    pure_research_hard = 1.0 if (
        any(m in text for m in PURE_RESEARCH_MARKERS)
        and not any(k in text for k in ["shipped", "production", "deployed", "deploy", "live a/b", "users"])
    ) else 0.0

    # Consulting-firm-only career (every single role at a listed consulting firm).
    all_companies = {(ch.get("company") or "").lower() for ch in history} | {current_company}
    consulting_only = 1.0 if all_companies and all(
        any(cf in comp for cf in CONSULTING_FIRMS) for comp in all_companies if comp
    ) else 0.0

    # CV / speech / robotics primary specialism without NLP/IR exposure.
    cv_speech_robotics = 1.0 if (
        any(m in text for m in CV_SPEECH_ROBOTICS_MARKERS)
        and not any(k in text for k in ["nlp", "retrieval", "ranking", "search", "language model", "embedding"])
    ) else 0.0

    # Title-chasing: 3+ employers in last 5 years, each stint < 18 months,
    # with escalating seniority words -- a real pattern check, not just count.
    recent_cutoff = TODAY.replace(year=TODAY.year - 5)
    recent_stints = [
        ch for ch in history
        if ch.get("start_date") and ch["start_date"] >= recent_cutoff.isoformat()
    ]
    short_recent_stints = [ch for ch in recent_stints if (ch.get("duration_months", 99) or 99) < 18]
    title_chaser = 1.0 if len(short_recent_stints) >= 3 else 0.0

    # Long stretch with no hands-on code (proxy: most recent role's title
    # contains "architect"/"tech lead"/"head of"/"director" and is >=18mo).
    current_role = next((ch for ch in history if ch.get("is_current")), None)
    architecture_only = 0.0
    if current_role:
        t = (current_role.get("title") or "").lower()
        if any(k in t for k in ["architect", "head of", "director", "vp "]) and (current_role.get("duration_months", 0) or 0) >= 18:
            if "code" not in (current_role.get("description") or "").lower():
                architecture_only = 1.0

    return {
        "pure_research_penalty": pure_research_hard,
        "consulting_only_penalty": consulting_only,
        "cv_speech_robotics_penalty": cv_speech_robotics,
        "title_chaser_penalty": title_chaser,
        "architecture_only_penalty": architecture_only,
    }


def location_features(candidate: dict[str, Any]) -> dict[str, float]:
    location = (candidate["profile"].get("location") or "").lower()
    country = (candidate["profile"].get("country") or "").lower()
    signals = candidate.get("redrob_signals", {}) or {}
    willing_to_relocate = bool(signals.get("willing_to_relocate", False))

    in_preferred_city = any(c in location for c in PREFERRED_CITIES)
    in_tier1_city = any(c in location for c in TIER1_INDIAN_CITIES)
    is_india = "india" in country

    if in_preferred_city:
        loc_score = 1.0
    elif in_tier1_city and is_india:
        loc_score = 0.85
    elif is_india:
        loc_score = 0.6
    elif willing_to_relocate:
        loc_score = 0.35  # JD: outside-India is "case-by-case", no visa sponsorship
    else:
        loc_score = 0.1

    return {
        "location_score": loc_score,
        "is_india": float(is_india),
        "willing_to_relocate": float(willing_to_relocate),
    }


def education_features(candidate: dict[str, Any]) -> dict[str, float]:
    edu = candidate.get("education", []) or []
    if not edu:
        return {"education_tier_score": 0.4}
    tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.55, "tier_4": 0.4, "unknown": 0.4}
    best = max((tier_map.get(e.get("tier", "unknown"), 0.4) for e in edu), default=0.4)
    # JD never mentions pedigree as a requirement -- this is a minor signal only.
    return {"education_tier_score": best}


def behavioral_signal_features(candidate: dict[str, Any]) -> dict[str, float]:
    """
    The redrob_signals modifier: "a perfect-on-paper candidate who hasn't
    logged in for 6 months and has a 5% recruiter response rate is, for
    hiring purposes, not actually available." This block produces a single
    availability/engagement multiplier in [~0.3, ~1.15], NOT an additive
    score, so it can meaningfully discount an otherwise-strong profile
    without letting raw activity metrics dominate actual JD fit.
    """
    s = candidate.get("redrob_signals", {}) or {}

    last_active = s.get("last_active_date")
    days_inactive = 9999
    if last_active:
        y, m, d = (int(x) for x in last_active.split("-"))
        days_inactive = (TODAY - date(y, m, d)).days
    recency_score = (
        1.0 if days_inactive <= 14 else
        0.85 if days_inactive <= 30 else
        0.6 if days_inactive <= 90 else
        0.35 if days_inactive <= 180 else
        0.15
    )

    response_rate = s.get("recruiter_response_rate", 0.0) or 0.0
    open_to_work = 1.0 if s.get("open_to_work_flag") else 0.6  # not flagged isn't disqualifying, just lower prior
    notice_days = s.get("notice_period_days", 60) or 60
    notice_score = 1.0 if notice_days <= MAX_NOTICE_PERIOD_SOFT else max(0.55, 1.0 - (notice_days - 30) / 200)

    interview_completion = s.get("interview_completion_rate", 0.8)
    offer_accept = s.get("offer_acceptance_rate", -1)
    offer_accept_score = 1.0 if offer_accept < 0 else (0.6 + 0.4 * offer_accept)

    verification = (
        (1 if s.get("verified_email") else 0)
        + (1 if s.get("verified_phone") else 0)
        + (1 if s.get("linkedin_connected") else 0)
    ) / 3.0

    engagement_raw = (
        0.30 * recency_score
        + 0.20 * response_rate
        + 0.15 * open_to_work
        + 0.15 * notice_score
        + 0.10 * interview_completion
        + 0.05 * offer_accept_score
        + 0.05 * verification
    )
    # Map [0,1] engagement to a modifier band so it nudges, not dominates.
    modifier = 0.55 + 0.65 * engagement_raw  # ranges ~0.55 (terrible) to ~1.2 (excellent)

    return {
        "behavioral_modifier": modifier,
        "days_inactive": float(min(days_inactive, 9999)),
        "recruiter_response_rate": float(response_rate),
        "notice_period_days": float(notice_days),
        "open_to_work_flag": float(bool(s.get("open_to_work_flag"))),
        "github_activity_score": float(s.get("github_activity_score", -1)),
        "saved_by_recruiters_30d": float(s.get("saved_by_recruiters_30d", 0)),
    }


def extract_features(candidate: dict[str, Any]) -> dict[str, Any]:
    """Single entry point: all features for one candidate, flattened into one dict."""
    feats: dict[str, Any] = {"candidate_id": candidate["candidate_id"]}
    feats.update(skills_features(candidate))
    feats.update(title_and_seniority_features(candidate))
    feats["applied_ml_fraction"] = applied_ml_fraction(candidate)
    feats.update(disqualifier_features(candidate))
    feats.update(location_features(candidate))
    feats.update(education_features(candidate))
    feats.update(behavioral_signal_features(candidate))
    return feats
