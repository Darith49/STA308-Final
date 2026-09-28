"""
Unit and integration tests for Course Recommendation System (STA308 Final Project).
Verifies all functional criteria outlined in Section 7 of the Project Plan.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.recommend import (
    ModelArtifacts,
    validate_grades,
    build_profile,
    predict_electives,
    recommend,
    explain,
    similar_student_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def artifacts():
    return ModelArtifacts.get_instance()


@pytest.fixture
def valid_grades():
    return {
        "Calculus": 85.0,
        "Statistics": 88.0,
        "Programming": 90.0,
        "English": 75.0,
        "Physics": 82.0,
        "Economics": 78.0
    }


def test_validate_grades_valid(valid_grades):
    """Test 1: validate_grades with valid input passes cleanly."""
    res = validate_grades(valid_grades)
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert len(res.cleaned_grades) == 6
    assert res.cleaned_grades["Calculus"] == 85.0


def test_validate_grades_out_of_range():
    """Test 2a: validate_grades flags out-of-range inputs (<0 or >100)."""
    bad_grades = {
        "Calculus": 105.0,
        "Statistics": -5.0,
        "Programming": 80.0,
        "English": 70.0,
        "Physics": 75.0,
        "Economics": 80.0
    }
    res = validate_grades(bad_grades)
    assert res.is_valid is False
    assert any("must be between 0 and 100" in err for err in res.errors)


def test_validate_grades_non_numeric():
    """Test 2b: validate_grades flags non-numeric values."""
    bad_grades = {
        "Calculus": "ninety",
        "Statistics": 85.0,
        "Programming": 80.0,
        "English": 70.0,
        "Physics": 75.0,
        "Economics": 80.0
    }
    res = validate_grades(bad_grades)
    assert res.is_valid is False
    assert any("valid numeric number" in err for err in res.errors)


def test_validate_grades_missing_imputation():
    """Test 2c: 1 or 2 missing grades are imputed and warned; >2 blanks are rejected."""
    # 1 missing: Calculus left None
    single_missing = {
        "Calculus": None,
        "Statistics": 85.0,
        "Programming": 85.0,
        "English": 80.0,
        "Physics": 80.0,
        "Economics": 80.0
    }
    res_single = validate_grades(single_missing)
    assert res_single.is_valid is True
    assert "Calculus" in res_single.imputed_subjects
    assert res_single.cleaned_grades["Calculus"] > 0.0
    assert any("estimated based on correlated core subjects" in w for w in res_single.warnings)

    # 3 missing: rejected
    too_many_missing = {
        "Calculus": None,
        "Statistics": None,
        "Programming": None,
        "English": 80.0,
        "Physics": 80.0,
        "Economics": 80.0
    }
    res_many = validate_grades(too_many_missing)
    assert res_many.is_valid is False
    assert any("at least 4 grades" in err for err in res_many.errors)


def test_build_profile_training_mean(artifacts):
    """Test 3: build_profile with cohort mean grades produces z-scores near 0."""
    mean_grades = {
        s: artifacts.scaler.mean_[i]
        for i, s in enumerate(artifacts.metadata["subject_list"])
    }
    profile = build_profile(mean_grades)
    for s, z in profile.z_scores.items():
        assert abs(z) < 1e-3, f"Subject {s} z-score should be ~0, got {z}"
    assert profile.cluster_id in [0, 1, 2, 3]
    assert profile.cluster_name != ""


def test_recommend_top_n_sorted(valid_grades):
    """Test 4: recommend returns exactly top_n, sorted descending by score."""
    rec = recommend(valid_grades, top_n=3)
    assert rec.is_valid is True
    assert len(rec.top_electives) == 3
    scores = rec.top_electives["combined_score"].tolist()
    assert scores == sorted(scores, reverse=True), "Top electives must be sorted descending"


def test_prediction_intervals(valid_grades):
    """Test 5: Prediction intervals satisfy lo <= pred <= hi and within [0, 100]."""
    profile = build_profile(valid_grades)
    pred_df = predict_electives(profile)
    for _, row in pred_df.iterrows():
        pred = row["pred_grade"]
        lo = row["interval_lo"]
        hi = row["interval_hi"]
        assert 0.0 <= lo <= pred <= hi <= 100.0, f"Violated interval constraints: {lo} <= {pred} <= {hi}"


def test_prerequisite_filter():
    """Test 6: Prerequisite filter accurately blocks ineligible courses."""
    # Student with failing Programming (< 50) and low Calculus (< 50)
    low_stem_grades = {
        "Calculus": 45.0,
        "Statistics": 70.0,
        "Programming": 40.0,
        "English": 88.0,
        "Physics": 50.0,
        "Economics": 85.0
    }
    rec = recommend(low_stem_grades, top_n=3)
    assert rec.is_valid is True
    
    # ELEC_DATAMIN requires Programming >= 65 and Statistics >= 60 -> should be blocked!
    blocked_ids = rec.blocked_electives["elective_id"].tolist()
    assert "ELEC_DATAMIN" in blocked_ids
    assert "ELEC_OPER" in blocked_ids  # Requires Calc >= 60 & Prog >= 50

    datamin_row = rec.blocked_electives[rec.blocked_electives["elective_id"] == "ELEC_DATAMIN"].iloc[0]
    assert "Requires Programming ≥ 65" in datamin_row["block_reason"]


def test_determinism(valid_grades):
    """Test 7: Determinism: identical input produces identical recommendations."""
    rec1 = recommend(valid_grades, top_n=3)
    rec2 = recommend(valid_grades, top_n=3)
    assert rec1.top_electives["elective_id"].tolist() == rec2.top_electives["elective_id"].tolist()
    assert rec1.top_electives["combined_score"].tolist() == rec2.top_electives["combined_score"].tolist()


def test_monotonicity():
    """Test 8: Monotonicity: raising a positively weighted subject never lowers predicted elective grade."""
    base_student = {
        "Calculus": 70.0,
        "Statistics": 70.0,
        "Programming": 70.0,
        "English": 70.0,
        "Physics": 70.0,
        "Economics": 70.0
    }
    boosted_student = dict(base_student)
    boosted_student["Calculus"] = 85.0  # Boost Calculus by 15 points

    pred_base = predict_electives(build_profile(base_student))
    pred_boost = predict_electives(build_profile(boosted_student))

    econ_base = pred_base[pred_base["elective_id"] == "ELEC_ECON"]["pred_grade"].values[0]
    econ_boost = pred_boost[pred_boost["elective_id"] == "ELEC_ECON"]["pred_grade"].values[0]

    assert econ_boost >= econ_base, f"Boosting Calculus should not lower Econometrics (Base: {econ_base}, Boosted: {econ_boost})"


def test_csv_integration_pipeline():
    """Test 9: End-to-end integration with CSV upload format."""
    csv_path = os.path.join(BASE_DIR, "data", "sample_student_template.csv")
    df = pd.read_csv(csv_path)
    row_dict = df.iloc[0].to_dict()
    
    rec = recommend(row_dict, top_n=3)
    assert rec.is_valid is True
    assert len(rec.top_electives) == 3
    assert rec.profile is not None
    assert rec.profile.cluster_name != ""
