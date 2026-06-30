"""Unit tests for honeypot.py — run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from honeypot import (
    count_low_duration_experts,
    honeypot_score,
    skill_duration_margin_years,
    summary_yoe_mismatch,
)


def _candidate(**overrides):
    base = {
        "candidate_id": "CAND_0000000",
        "profile": {
            "years_of_experience": 6.0,
            "summary": "Engineer with 6 years of experience building backend systems.",
        },
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 60},
        ],
    }
    base.update(overrides)
    return base


def test_normal_profile_scores_zero():
    c = _candidate()
    assert honeypot_score(c) <= 1


def test_skill_duration_far_exceeds_yoe_is_flagged():
    c = _candidate(
        profile={"years_of_experience": 1.0, "summary": "Recent grad."},
        skills=[{"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 96}],
    )
    margin = skill_duration_margin_years(c)
    assert margin > 4
    assert honeypot_score(c) >= 1


def test_small_skill_duration_margin_is_not_flagged():
    # 6 months of natural rounding noise should NOT trip the detector.
    c = _candidate(
        profile={"years_of_experience": 3.0, "summary": "no claim here"},
        skills=[{"name": "SQL", "proficiency": "advanced", "endorsements": 5, "duration_months": 42}],
    )
    assert honeypot_score(c) == 0


def test_many_low_duration_experts_is_flagged():
    skills = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 1}
        for i in range(5)
    ]
    c = _candidate(skills=skills)
    assert count_low_duration_experts(c) == 5
    assert honeypot_score(c) >= 2


def test_summary_yoe_mismatch_detected():
    c = _candidate(
        profile={
            "years_of_experience": 15.0,
            "summary": "Engineer with 4.0 years of experience in ML.",
        }
    )
    assert summary_yoe_mismatch(c) > 1.5
    assert honeypot_score(c) >= 3


def test_summary_yoe_consistent_not_flagged():
    c = _candidate(
        profile={
            "years_of_experience": 6.0,
            "summary": "Engineer with 6 years of experience in ML.",
        }
    )
    assert summary_yoe_mismatch(c) < 1.5


def test_no_skills_does_not_crash():
    c = _candidate(skills=[])
    assert honeypot_score(c) >= 0  # should not raise
