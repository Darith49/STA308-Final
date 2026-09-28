"""
Core Recommender Engine for Course Recommendation System (STA308 Final Project).
Strictly adheres to Architectural Rule 1: No Streamlit imports.
Pure Python functions with full type annotations, deterministic behavior,
prerequisite gating, uncertainty intervals, and plain-language explanations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

# Path configurations relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ELECTIVE_MODELS_DIR = os.path.join(MODELS_DIR, "elective_models")

CORE_SUBJECTS = [
    "Calculus",
    "Statistics",
    "Programming",
    "English",
    "Physics",
    "Economics"
]


@dataclass
class ValidationResult:
    is_valid: bool
    cleaned_grades: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    imputed_subjects: List[str] = field(default_factory=list)


@dataclass
class Contribution:
    subject: str
    coefficient: float
    z_score: float
    point_impact: float
    direction: str
    plain_text: str


@dataclass
class Profile:
    raw_grades: Dict[str, float]
    z_scores: Dict[str, float]
    pca_scores: List[float]
    cluster_id: int
    cluster_name: str
    cluster_description: str
    cluster_centers_raw: Dict[str, float]
    imputed_subjects: List[str] = field(default_factory=list)


@dataclass
class Recommendation:
    is_valid: bool
    validation_result: ValidationResult
    profile: Optional[Profile]
    top_electives: pd.DataFrame
    all_eligible: pd.DataFrame
    blocked_electives: pd.DataFrame
    explanations: Dict[str, List[Contribution]]
    similar_student_stats: Dict[str, Dict[str, Any]]
    warnings: List[str]
    imputed_subjects: List[str]


class ModelArtifacts:
    """Singleton-style artifact cache for fast offline inference."""
    _instance = None

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        self.pca = joblib.load(os.path.join(models_dir, "pca.joblib"))
        self.kmeans_bundle = joblib.load(os.path.join(models_dir, "kmeans.joblib"))
        self.neighbors_bundle = joblib.load(os.path.join(models_dir, "neighbors.joblib"))
        
        imputation_path = os.path.join(models_dir, "imputation_models.joblib")
        if os.path.exists(imputation_path):
            self.imputation_models = joblib.load(imputation_path)
        else:
            self.imputation_models = None

        with open(os.path.join(models_dir, "prereqs.json"), "r") as f:
            self.prereqs = json.load(f)

        with open(os.path.join(models_dir, "residual_stats.json"), "r") as f:
            self.residual_stats = json.load(f)

        with open(os.path.join(models_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        with open(os.path.join(models_dir, "eda_summary.json"), "r") as f:
            self.eda_summary = json.load(f)

        self.elective_models = {}
        for elec_id in self.prereqs.keys():
            m_path = os.path.join(models_dir, "elective_models", f"{elec_id}.joblib")
            self.elective_models[elec_id] = joblib.load(m_path)

    @classmethod
    def get_instance(cls, models_dir: str = MODELS_DIR) -> "ModelArtifacts":
        if cls._instance is None:
            cls._instance = cls(models_dir)
        return cls._instance


def validate_grades(grades: Dict[str, Optional[float]], min_required: int = 4) -> ValidationResult:
    """
    Validate input core grades dictionary.
    Handles missing values (up to 2 allowed with imputation),
    validates numeric boundaries (0-100), and flags identical inputs.
    """
    errors = []
    warnings = []
    imputed_subjects = []
    cleaned: Dict[str, float] = {}

    artifacts = ModelArtifacts.get_instance()
    expected_subjects = artifacts.metadata.get("subject_list", CORE_SUBJECTS)

    # 1. Parse and validate individual subject values
    numeric_entered = {}
    missing_subjects = []

    for subj in expected_subjects:
        val = grades.get(subj)
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            missing_subjects.append(subj)
            continue
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            errors.append(f"Please enter a valid numeric number for {subj}.")
            continue

        if val_f < 0.0 or val_f > 100.0:
            errors.append(f"Grade for {subj} ({val_f}) must be between 0 and 100.")
        else:
            numeric_entered[subj] = round(val_f, 1)

    # If parsing errors already exist, return immediately
    if errors:
        return ValidationResult(is_valid=False, cleaned_grades={}, errors=errors, warnings=warnings)

    num_provided = len(numeric_entered)
    if num_provided < min_required:
        errors.append(f"Please enter at least {min_required} grades so the estimate is reliable. (Provided: {num_provided})")
        return ValidationResult(is_valid=False, cleaned_grades={}, errors=errors, warnings=warnings)

    # Check for identical grades warning
    if num_provided >= 3 and len(set(numeric_entered.values())) == 1:
        warnings.append("All entered grades are identical. Academic profile differentiation may be limited.")

    # 2. Impute missing subjects if 1 or 2 are missing
    cleaned = dict(numeric_entered)
    if missing_subjects:
        for m_subj in missing_subjects:
            imputed_val = _impute_single_grade(m_subj, cleaned, artifacts)
            cleaned[m_subj] = round(imputed_val, 1)
            imputed_subjects.append(m_subj)
        warnings.append(
            f"Note: {', '.join(imputed_subjects)} was not entered and was estimated based on correlated core subjects."
        )

    return ValidationResult(
        is_valid=True,
        cleaned_grades=cleaned,
        errors=[],
        warnings=warnings,
        imputed_subjects=imputed_subjects
    )


def _impute_single_grade(target_subject: str, available_grades: Dict[str, float], artifacts: ModelArtifacts) -> float:
    """Impute a missing subject grade using pre-trained regression or cohort mean fallback."""
    if artifacts.imputation_models and target_subject in artifacts.imputation_models:
        model_info = artifacts.imputation_models[target_subject]
        req_features = model_info["features"]
        reg = model_info["model"]
        
        # Check if all required features are present
        if all(f in available_grades for f in req_features):
            feat_df = pd.DataFrame([[available_grades[f] for f in req_features]], columns=req_features)
            pred_val = float(reg.predict(feat_df)[0])
            return float(np.clip(pred_val, 0.0, 100.0))

    # Fallback to mean of entered grades or cohort mean
    if available_grades:
        return float(np.mean(list(available_grades.values())))
    cohort_mean = artifacts.eda_summary.get("distributions", {}).get(target_subject, {}).get("mean", 72.0)
    return float(cohort_mean)


def build_profile(grades: Dict[str, float], imputed_subjects: Optional[List[str]] = None) -> Profile:
    """
    Standardize grades using training scaler, project onto PCA, and assign KMeans cluster.
    """
    artifacts = ModelArtifacts.get_instance()
    subj_list = artifacts.metadata["subject_list"]
    raw_vec = np.array([[grades[s] for s in subj_list]])

    # 1. Standardize (z-scores)
    scaled_vec = artifacts.scaler.transform(raw_vec)[0]
    z_scores = {s: float(scaled_vec[i]) for i, s in enumerate(subj_list)}

    # 2. PCA projection
    pca_coords = artifacts.pca.transform(raw_vec)[0].tolist()

    # 3. KMeans Cluster assignment
    kmeans_bundle = artifacts.kmeans_bundle
    cluster_id = int(kmeans_bundle["model"].predict([scaled_vec])[0])
    cluster_name = kmeans_bundle["cluster_labels"].get(cluster_id, f"Cluster {cluster_id}")
    cluster_desc = kmeans_bundle["cluster_descriptions"].get(cluster_id, "")
    cluster_raw_centers = kmeans_bundle["cluster_centers_raw"][cluster_id]
    cluster_centers_dict = {s: round(float(cluster_raw_centers[i]), 1) for i, s in enumerate(subj_list)}

    return Profile(
        raw_grades={s: float(grades[s]) for s in subj_list},
        z_scores=z_scores,
        pca_scores=[round(p, 3) for p in pca_coords],
        cluster_id=cluster_id,
        cluster_name=cluster_name,
        cluster_description=cluster_desc,
        cluster_centers_raw=cluster_centers_dict,
        imputed_subjects=imputed_subjects or []
    )


def predict_electives(profile: Profile) -> pd.DataFrame:
    """
    Predict expected grade and 95% prediction interval for all electives.
    Verifies prerequisite eligibility against student's raw grades.
    """
    artifacts = ModelArtifacts.get_instance()
    subj_list = artifacts.metadata["subject_list"]
    scaled_vec = np.array([[profile.z_scores[s] for s in subj_list]])

    rows = []
    for elec_id, model in artifacts.elective_models.items():
        meta = artifacts.eda_summary["electives_meta"].get(elec_id, {})
        res_info = artifacts.residual_stats.get(elec_id, {})
        res_sd = res_info.get("residual_sd", 3.8)

        # Predicted grade
        pred_val = float(model.predict(scaled_vec)[0])
        pred_val = float(np.clip(pred_val, 0.0, 100.0))

        # Approximate 95% prediction interval (pred ± 1.96 * residual SD)
        interval_margin = 1.96 * res_sd
        lo = float(np.clip(pred_val - interval_margin, 0.0, 100.0))
        hi = float(np.clip(pred_val + interval_margin, 0.0, 100.0))

        # Prerequisite check
        prereqs = artifacts.prereqs.get(elec_id, {})
        eligible = True
        block_reasons = []

        for req_subj, min_score in prereqs.items():
            actual_score = profile.raw_grades.get(req_subj, 0.0)
            if actual_score < min_score:
                eligible = False
                block_reasons.append(f"Requires {req_subj} ≥ {min_score:.0f} (You have: {actual_score:.1f})")

        reason_str = "; ".join(block_reasons) if block_reasons else None

        rows.append({
            "elective_id": elec_id,
            "name": meta.get("name", elec_id),
            "category": meta.get("category", "General"),
            "description": meta.get("description", ""),
            "pred_grade": round(pred_val, 1),
            "interval_lo": round(lo, 1),
            "interval_hi": round(hi, 1),
            "residual_sd": round(res_sd, 2),
            "eligible": eligible,
            "block_reason": reason_str
        })

    df = pd.DataFrame(rows)
    return df


def similar_student_score(profile: Profile, k: int = 15) -> Tuple[pd.Series, Dict[str, Dict[str, Any]]]:
    """
    Compute elective performance benchmarks among the top-k most similar students
    in the training cohort based on standardized core grade Euclidean distance.
    """
    artifacts = ModelArtifacts.get_instance()
    subj_list = artifacts.metadata["subject_list"]
    student_scaled = np.array([profile.z_scores[s] for s in subj_list])

    X_train_scaled = artifacts.neighbors_bundle["X_core_scaled"]
    Y_train_electives = artifacts.neighbors_bundle["Y_electives"]
    elective_ids = artifacts.neighbors_bundle["elective_ids"]

    # Euclidean distance in 6D standardized core space
    distances = np.linalg.norm(X_train_scaled - student_scaled, axis=1)
    k_nearest_indices = np.argsort(distances)[:k]

    # Elective stats for nearest neighbors
    neighbor_grades = Y_train_electives[k_nearest_indices]
    avg_scores = np.mean(neighbor_grades, axis=0)

    score_series = pd.Series(avg_scores, index=elective_ids)

    stats = {}
    for i, eid in enumerate(elective_ids):
        g_list = neighbor_grades[:, i]
        stats[eid] = {
            "mean": round(float(np.mean(g_list)), 1),
            "median": round(float(np.median(g_list)), 1),
            "p25": round(float(np.percentile(g_list, 25)), 1),
            "p75": round(float(np.percentile(g_list, 75)), 1),
            "k_neighbors": k
        }

    return score_series, stats


def explain(elective_id: str, grades: Dict[str, float]) -> List[Contribution]:
    """
    Explain regression predictions via feature contributions:
    contribution_i = coefficient_i * z_score_i.
    Returns top 3 contributing core subjects formatted in plain language.
    """
    artifacts = ModelArtifacts.get_instance()
    subj_list = artifacts.metadata["subject_list"]
    res_info = artifacts.residual_stats.get(elective_id, {})
    coefs = res_info.get("coefficients", {})

    # Calculate z-scores
    raw_vec = np.array([[grades[s] for s in subj_list]])
    scaled_vec = artifacts.scaler.transform(raw_vec)[0]

    contributions: List[Contribution] = []
    for i, subj in enumerate(subj_list):
        c = coefs.get(subj, 0.0)
        z = scaled_vec[i]
        impact = c * z  # Point impact relative to average
        direction = "positive" if impact >= 0 else "negative"

        sign_str = f"+{impact:.1f}" if impact >= 0 else f"{impact:.1f}"
        if abs(impact) >= 0.5:
            text = f"{'Strong' if abs(impact) > 2.0 else 'Solid'} {subj} performance contributed {sign_str} points"
        else:
            text = f"Neutral impact from {subj} ({sign_str} points)"

        contributions.append(Contribution(
            subject=subj,
            coefficient=round(c, 3),
            z_score=round(z, 2),
            point_impact=round(impact, 2),
            direction=direction,
            plain_text=text
        ))

    # Sort by absolute impact descending and take top 3
    contributions.sort(key=lambda x: abs(x.point_impact), reverse=True)
    return contributions[:3]


def recommend(grades: Dict[str, Optional[float]], top_n: int = 3, w1: float = 0.70) -> Recommendation:
    """
    End-to-end recommendation workflow:
    1. Validate inputs and optionally impute missing core grades.
    2. Build academic profile (z-scores, PCA, KMeans cluster).
    3. Predict performance for all electives with prediction intervals.
    4. Compute similar-student cohort scores.
    5. Rank eligible electives by combined score: w1 * predicted_grade + (1 - w1) * similar_cohort_avg.
    6. Generate plain-language explanations for top picks.
    """
    val_res = validate_grades(grades)
    if not val_res.is_valid:
        return Recommendation(
            is_valid=False,
            validation_result=val_res,
            profile=None,
            top_electives=pd.DataFrame(),
            all_eligible=pd.DataFrame(),
            blocked_electives=pd.DataFrame(),
            explanations={},
            similar_student_stats={},
            warnings=val_res.warnings,
            imputed_subjects=val_res.imputed_subjects
        )

    profile = build_profile(val_res.cleaned_grades, val_res.imputed_subjects)
    predictions_df = predict_electives(profile)
    sim_scores, sim_stats = similar_student_score(profile, k=15)

    # Attach similar student score and combined recommendation index
    predictions_df["similar_avg"] = predictions_df["elective_id"].map(sim_scores).round(1)
    predictions_df["combined_score"] = (
        w1 * predictions_df["pred_grade"] + (1.0 - w1) * predictions_df["similar_avg"]
    ).round(2)

    # Split into eligible and blocked
    eligible_mask = predictions_df["eligible"] == True
    all_eligible = predictions_df[eligible_mask].copy()
    all_eligible = all_eligible.sort_values(by="combined_score", ascending=False).reset_index(drop=True)
    all_eligible["rank"] = all_eligible.index + 1

    blocked_df = predictions_df[~eligible_mask].copy().reset_index(drop=True)

    # Top-N recommendations
    top_electives = all_eligible.head(top_n).copy()

    # Generate explanations for all eligible electives
    explanations = {}
    for eid in all_eligible["elective_id"]:
        explanations[eid] = explain(eid, val_res.cleaned_grades)

    return Recommendation(
        is_valid=True,
        validation_result=val_res,
        profile=profile,
        top_electives=top_electives,
        all_eligible=all_eligible,
        blocked_electives=blocked_df,
        explanations=explanations,
        similar_student_stats=sim_stats,
        warnings=val_res.warnings,
        imputed_subjects=val_res.imputed_subjects
    )
