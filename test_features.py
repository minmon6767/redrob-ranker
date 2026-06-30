"""Unit tests for features.py — run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    applied_ml_fraction,
    disqualifier_features,
    extract_features,
    location_features,
    skill_trust,
    title_and_seniority_features,
)


def _minimal_candidate(**overrides):
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "headline": "Senior AI Engineer",
            "summary": "Senior AI engineer with 7 years building ranking and retrieval systems.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior AI Engineer",
            "current_company": "ExampleCo",
            "current_company_size": "201-500",
            "current_industry": "AI/ML",
        },
        "career_history": [
            {
                "company": "ExampleCo",
                "title": "Senior AI Engineer",
                "start_date": "2023-01-01",
                "end_date": None,
                "duration_months": 36,
                "is_current": True,
                "industry": "AI/ML",
                "company_size": "201-500",
                "description": "Built a hybrid retrieval and ranking system serving production traffic.",
            }
        ],
        "education": [
            {"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "CS", "start_year": 2013, "end_year": 2017, "tier": "tier_1"}
        ],
        "skills": [
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 36},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 15, "duration_months": 24},
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 50,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50,
            "search_appearance_30d": 20,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k].update(v)
        else:
            base[k] = v
    return base


def test_skill_trust_rewards_corroborated_claims():
    strong = {"proficiency": "expert", "endorsements": 30, "duration_months": 36}
    weak = {"proficiency": "expert", "endorsements": 0, "duration_months": 0}
    assert skill_trust(strong) > skill_trust(weak)


def test_title_score_full_credit_for_core_title():
    feats = title_and_seniority_features(_minimal_candidate())
    assert feats["title_core_match"] == 1.0
    assert feats["title_score"] == 1.0


def test_title_score_zero_for_unrelated_title():
    c = _minimal_candidate(profile={"current_title": "Mechanical Engineer"})
    feats = title_and_seniority_features(c)
    assert feats["title_score"] == 0.0


def test_yoe_score_peaks_in_ideal_band():
    c7 = _minimal_candidate(profile={"years_of_experience": 7.0})
    c20 = _minimal_candidate(profile={"years_of_experience": 20.0})
    f7 = title_and_seniority_features(c7)
    f20 = title_and_seniority_features(c20)
    assert f7["yoe_score"] > f20["yoe_score"]


def test_applied_ml_fraction_high_for_ml_career():
    feats_fraction = applied_ml_fraction(_minimal_candidate())
    assert feats_fraction > 0.8


def test_consulting_only_penalty_triggers_when_all_employers_are_consulting():
    c = _minimal_candidate(
        profile={"current_company": "TCS"},
        career_history=[
            {
                "company": "TCS", "title": "Engineer", "start_date": "2018-01-01", "end_date": None,
                "duration_months": 60, "is_current": True, "industry": "IT Services",
                "company_size": "10001+", "description": "General software work.",
            }
        ],
    )
    feats = disqualifier_features(c)
    assert feats["consulting_only_penalty"] == 1.0


def test_consulting_only_penalty_does_not_trigger_with_product_company_stint():
    c = _minimal_candidate(
        career_history=[
            {
                "company": "TCS", "title": "Engineer", "start_date": "2018-01-01", "end_date": "2021-01-01",
                "duration_months": 36, "is_current": False, "industry": "IT Services",
                "company_size": "10001+", "description": "General software work.",
            },
            {
                "company": "Razorpay", "title": "Senior AI Engineer", "start_date": "2021-01-01", "end_date": None,
                "duration_months": 36, "is_current": True, "industry": "AI/ML",
                "company_size": "201-500", "description": "Built ranking systems.",
            },
        ],
    )
    feats = disqualifier_features(c)
    assert feats["consulting_only_penalty"] == 0.0


def test_location_score_highest_for_preferred_city():
    c = _minimal_candidate(profile={"location": "Pune, Maharashtra", "country": "India"})
    feats = location_features(c)
    assert feats["location_score"] == 1.0


def test_location_score_lower_for_non_india_unwilling_to_relocate():
    c = _minimal_candidate(
        profile={"location": "London", "country": "UK"},
        redrob_signals={"willing_to_relocate": False},
    )
    feats = location_features(c)
    assert feats["location_score"] < 0.3


def test_extract_features_returns_all_expected_keys():
    feats = extract_features(_minimal_candidate())
    for key in ("must_have_coverage", "title_score", "yoe_score", "location_score", "behavioral_modifier"):
        assert key in feats
