"""Unit tests for scoring.py — run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoring import (
    disqualifier_multiplier,
    final_score,
    honeypot_multiplier,
    structured_jd_fit,
)


def _good_features():
    return {
        "must_have_coverage": 0.9,
        "nice_to_have_coverage": 0.5,
        "title_score": 1.0,
        "yoe_score": 1.0,
        "applied_ml_fraction": 0.9,
        "location_score": 1.0,
        "education_tier_score": 0.8,
        "pure_research_penalty": 0.0,
        "consulting_only_penalty": 0.0,
        "cv_speech_robotics_penalty": 0.0,
        "title_chaser_penalty": 0.0,
        "architecture_only_penalty": 0.0,
        "behavioral_modifier": 1.1,
    }


def test_structured_jd_fit_is_higher_for_better_features():
    good = structured_jd_fit(_good_features())
    bad_feats = dict(_good_features())
    bad_feats.update({"must_have_coverage": 0.1, "title_score": 0.0, "yoe_score": 0.1})
    bad = structured_jd_fit(bad_feats)
    assert good > bad


def test_disqualifier_multiplier_is_one_when_no_disqualifiers():
    assert disqualifier_multiplier(_good_features()) == 1.0


def test_disqualifier_multiplier_drops_for_pure_research():
    feats = dict(_good_features())
    feats["pure_research_penalty"] = 1.0
    assert disqualifier_multiplier(feats) < 1.0


def test_disqualifier_multiplier_compounds_for_multiple_flags():
    feats = dict(_good_features())
    feats["pure_research_penalty"] = 1.0
    feats["consulting_only_penalty"] = 1.0
    single = dict(_good_features())
    single["pure_research_penalty"] = 1.0
    assert disqualifier_multiplier(feats) < disqualifier_multiplier(single)


def test_honeypot_multiplier_decreases_with_score():
    assert honeypot_multiplier(0) == 1.0
    assert honeypot_multiplier(2) < honeypot_multiplier(0)
    assert honeypot_multiplier(4) < honeypot_multiplier(2)


def test_final_score_is_bounded_zero_to_one():
    feats = _good_features()
    s = final_score(feats, semantic_score=1.0, honeypot_score=0)
    assert 0.0 <= s <= 1.0


def test_final_score_penalizes_honeypot_heavily():
    feats = _good_features()
    clean = final_score(feats, semantic_score=0.8, honeypot_score=0)
    flagged = final_score(feats, semantic_score=0.8, honeypot_score=4)
    assert flagged < clean * 0.3


def test_final_score_rewards_behavioral_modifier():
    feats_engaged = dict(_good_features())
    feats_engaged["behavioral_modifier"] = 1.15
    feats_disengaged = dict(_good_features())
    feats_disengaged["behavioral_modifier"] = 0.55
    s_engaged = final_score(feats_engaged, semantic_score=0.7, honeypot_score=0)
    s_disengaged = final_score(feats_disengaged, semantic_score=0.7, honeypot_score=0)
    assert s_engaged > s_disengaged
