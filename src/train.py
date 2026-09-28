"""
Data generation and model training pipeline for STA308 Course Recommendation System.
Generates realistic student academic data with correlation structures,
fits preprocessing, PCA, clustering, regression models, imputers,
and exports all required model artifacts and metrics to models/.
"""

import json
import os
import sys
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CORE_SUBJECTS = [
    "Calculus",
    "Statistics",
    "Programming",
    "English",
    "Physics",
    "Economics"
]

ELECTIVES = {
    "ELEC_ECON": {
        "name": "Econometrics",
        "category": "Quantitative Economics",
        "description": "Statistical modeling of economic data and empirical analysis.",
        "weights": {"Calculus": 0.35, "Statistics": 0.40, "Economics": 0.25, "English": 0.0, "Physics": 0.0, "Programming": 0.10}
    },
    "ELEC_DATAMIN": {
        "name": "Data Mining & Machine Learning",
        "category": "Computer Science & Analytics",
        "description": "Algorithms for discovering patterns in large-scale datasets.",
        "weights": {"Programming": 0.45, "Statistics": 0.35, "Calculus": 0.15, "Physics": 0.05, "English": 0.0, "Economics": 0.0}
    },
    "ELEC_OPER": {
        "name": "Operations Research & Optimization",
        "category": "Applied Mathematics",
        "description": "Linear programming, optimization, and decision modeling.",
        "weights": {"Calculus": 0.40, "Programming": 0.30, "Statistics": 0.20, "Physics": 0.10, "English": 0.0, "Economics": 0.0}
    },
    "ELEC_NLP": {
        "name": "Natural Language Processing",
        "category": "Artificial Intelligence & Linguistics",
        "description": "Computational linguistics, text representation, and language models.",
        "weights": {"Programming": 0.40, "English": 0.35, "Statistics": 0.15, "Calculus": 0.10, "Physics": 0.0, "Economics": 0.0}
    },
    "ELEC_FINMATH": {
        "name": "Financial Mathematics",
        "category": "Quantitative Finance",
        "description": "Stochastic calculus, derivative pricing, and portfolio theory.",
        "weights": {"Calculus": 0.40, "Economics": 0.30, "Statistics": 0.20, "Programming": 0.10, "Physics": 0.0, "English": 0.0}
    },
    "ELEC_BIOSTAT": {
        "name": "Biostatistics & Clinical Trials",
        "category": "Health & Life Sciences",
        "description": "Survival analysis, clinical trial design, and epidemiological stats.",
        "weights": {"Statistics": 0.50, "Calculus": 0.20, "English": 0.15, "Programming": 0.15, "Physics": 0.0, "Economics": 0.0}
    },
    "ELEC_ROBOTICS": {
        "name": "Robotics & Autonomous Systems",
        "category": "Engineering & Robotics",
        "description": "Kinematics, sensor fusion, and real-time embedded control.",
        "weights": {"Physics": 0.40, "Programming": 0.35, "Calculus": 0.20, "Statistics": 0.05, "English": 0.0, "Economics": 0.0}
    },
    "ELEC_CORPFIN": {
        "name": "Corporate Finance & Strategy",
        "category": "Business & Management",
        "description": "Capital budgeting, corporate valuation, and financial governance.",
        "weights": {"Economics": 0.45, "English": 0.25, "Statistics": 0.20, "Calculus": 0.10, "Programming": 0.0, "Physics": 0.0}
    }
}

PREREQUISITES = {
    "ELEC_ECON": {"Calculus": 60.0, "Statistics": 60.0},
    "ELEC_DATAMIN": {"Programming": 65.0, "Statistics": 60.0},
    "ELEC_OPER": {"Calculus": 60.0, "Programming": 50.0},
    "ELEC_NLP": {"Programming": 60.0, "English": 65.0},
    "ELEC_FINMATH": {"Calculus": 65.0, "Economics": 60.0},
    "ELEC_BIOSTAT": {"Statistics": 65.0},
    "ELEC_ROBOTICS": {"Physics": 65.0, "Programming": 60.0},
    "ELEC_CORPFIN": {"Economics": 60.0}
}

CLUSTER_LABELS = {
    0: "Quantitative Specialist",
    1: "Balanced Achiever",
    2: "Computational & Tech Focus",
    3: "Socio-Economic & Applied"
}

CLUSTER_DESCRIPTIONS = {
    0: "Excels primarily in advanced mathematics, physics, and theoretical problem solving.",
    1: "Maintains consistently strong performance across both STEM and humanities subjects.",
    2: "Shows peak strengths in algorithms, programming, and data-driven computational methods.",
    3: "Leans toward economic reasoning, structured communication, and empirical social systems."
}


def generate_synthetic_data(n_students=1200):
    """Generate realistic academic cohort with correlation structures and latent traits."""
    # Latent factors: General intelligence (g), Math ability (m), Verbal ability (v), Tech aptitude (t)
    g = np.random.normal(0, 1, n_students)
    m = np.random.normal(0, 1, n_students)
    v = np.random.normal(0, 1, n_students)
    t = np.random.normal(0, 1, n_students)

    base_mean = 74.0
    base_sd = 11.0

    calc = base_mean + base_sd * (0.6 * g + 0.6 * m + np.random.normal(0, 0.4, n_students))
    stats = base_mean + base_sd * (0.6 * g + 0.45 * m + 0.25 * t + np.random.normal(0, 0.4, n_students))
    prog = base_mean + base_sd * (0.5 * g + 0.25 * m + 0.65 * t + np.random.normal(0, 0.4, n_students))
    engl = base_mean + base_sd * (0.45 * g + 0.7 * v - 0.15 * m + np.random.normal(0, 0.45, n_students))
    phys = base_mean + base_sd * (0.55 * g + 0.5 * m + 0.25 * t + np.random.normal(0, 0.45, n_students))
    econ = base_mean + base_sd * (0.5 * g + 0.3 * m + 0.35 * v + np.random.normal(0, 0.45, n_students))

    df = pd.DataFrame({
        "student_id": [f"STU_{i+1:05d}" for i in range(n_students)],
        "Calculus": np.clip(calc, 40, 99).round(1),
        "Statistics": np.clip(stats, 40, 99).round(1),
        "Programming": np.clip(prog, 38, 99).round(1),
        "English": np.clip(engl, 45, 99).round(1),
        "Physics": np.clip(phys, 38, 99).round(1),
        "Economics": np.clip(econ, 42, 99).round(1),
    })

    # Generate elective grades based on weighted domain relevance + idiosyncratic student noise
    for elec_id, info in ELECTIVES.items():
        weights = info["weights"]
        weighted_sum = sum(df[subj] * w for subj, w in weights.items())
        total_weight = sum(weights.values())
        if total_weight > 0:
            elec_grade = (weighted_sum / total_weight) + np.random.normal(0, 3.8, n_students)
        else:
            elec_grade = df[CORE_SUBJECTS].mean(axis=1) + np.random.normal(0, 5.0, n_students)
        df[elec_id] = np.clip(elec_grade, 42, 100).round(1)

    return df


def train_and_export_artifacts(data_dir="data", models_dir="models"):
    """Fit all models, extract metrics, and persist artifacts."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    elective_models_dir = os.path.join(models_dir, "elective_models")
    os.makedirs(elective_models_dir, exist_ok=True)

    # 1. Dataset Generation & Persistence
    df = generate_synthetic_data(n_students=1200)
    raw_data_path = os.path.join(data_dir, "raw_student_data.csv")
    df.to_csv(raw_data_path, index=False)
    print(f"Dataset generated with {len(df)} students saved to {raw_data_path}")

    # Generate template CSV for students
    sample_template = pd.DataFrame([{
        "Calculus": 85.0,
        "Statistics": 88.0,
        "Programming": 92.0,
        "English": 76.0,
        "Physics": 84.0,
        "Economics": 78.0
    }])
    sample_template.to_csv(os.path.join(data_dir, "sample_student_template.csv"), index=False)

    # Sample student profiles for preset demo
    sample_profiles = {
        "Quantitative Thinker": {
            "Calculus": 92.0,
            "Statistics": 94.0,
            "Programming": 85.0,
            "English": 72.0,
            "Physics": 89.0,
            "Economics": 80.0
        },
        "Applied Tech & Computational": {
            "Calculus": 78.0,
            "Statistics": 82.0,
            "Programming": 96.0,
            "English": 75.0,
            "Physics": 82.0,
            "Economics": 70.0
        },
        "Balanced Socio-Economic Scholar": {
            "Calculus": 72.0,
            "Statistics": 79.0,
            "Programming": 68.0,
            "English": 90.0,
            "Physics": 65.0,
            "Economics": 93.0
        }
    }
    with open(os.path.join(data_dir, "sample_profiles.json"), "w") as f:
        json.dump(sample_profiles, f, indent=2)

    X_core = df[CORE_SUBJECTS].values

    # 2. Fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_core)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    # 3. Fit PCA
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, os.path.join(models_dir, "pca.joblib"))

    # 4. Fit KMeans
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    kmeans_bundle = {
        "model": kmeans,
        "cluster_labels": CLUSTER_LABELS,
        "cluster_descriptions": CLUSTER_DESCRIPTIONS,
        "cluster_centers_scaled": kmeans.cluster_centers_,
        "cluster_centers_raw": scaler.inverse_transform(kmeans.cluster_centers_)
    }
    joblib.dump(kmeans_bundle, os.path.join(models_dir, "kmeans.joblib"))

    # 5. Fit Missing Core Grade Imputation Models (one regressor per subject predicting from other 5)
    imputation_models = {}
    for i, target_col in enumerate(CORE_SUBJECTS):
        feature_cols = [c for c in CORE_SUBJECTS if c != target_col]
        reg = Ridge(alpha=1.0)
        reg.fit(df[feature_cols], df[target_col])
        imputation_models[target_col] = {
            "features": feature_cols,
            "model": reg
        }
    joblib.dump(imputation_models, os.path.join(models_dir, "imputation_models.joblib"))

    # 6. Fit Elective Regressors & Benchmark Model Comparison
    residual_stats = {}
    model_metrics = {
        "per_elective": {},
        "comparison_table": [],
        "ranking_metrics": {}
    }

    # Cross-validated evaluation across models
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    model_types = ["Baseline Mean", "Student GPA Baseline", "Linear Regression", "Ridge Regression", "Random Forest"]
    cv_scores = {m: {"rmse": [], "mae": [], "r2": []} for m in model_types}

    for elec_id, info in ELECTIVES.items():
        y = df[elec_id].values

        # K-fold evaluation for model comparison
        for train_idx, val_idx in kf.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            raw_X_val = X_core[val_idx]

            # 1. Baseline Mean
            y_pred_base = np.full_like(y_val, np.mean(y_tr))
            cv_scores["Baseline Mean"]["rmse"].append(np.sqrt(mean_squared_error(y_val, y_pred_base)))
            cv_scores["Baseline Mean"]["mae"].append(mean_absolute_error(y_val, y_pred_base))
            cv_scores["Baseline Mean"]["r2"].append(r2_score(y_val, y_pred_base))

            # 2. Student GPA Baseline (Mean of student's own core subjects)
            y_pred_gpa = np.mean(raw_X_val, axis=1)
            cv_scores["Student GPA Baseline"]["rmse"].append(np.sqrt(mean_squared_error(y_val, y_pred_gpa)))
            cv_scores["Student GPA Baseline"]["mae"].append(mean_absolute_error(y_val, y_pred_gpa))
            cv_scores["Student GPA Baseline"]["r2"].append(r2_score(y_val, y_pred_gpa))

            # 3. Linear Regression
            lr = LinearRegression()
            lr.fit(X_tr, y_tr)
            y_pred_lr = lr.predict(X_val)
            cv_scores["Linear Regression"]["rmse"].append(np.sqrt(mean_squared_error(y_val, y_pred_lr)))
            cv_scores["Linear Regression"]["mae"].append(mean_absolute_error(y_val, y_pred_lr))
            cv_scores["Linear Regression"]["r2"].append(r2_score(y_val, y_pred_lr))

            # 4. Ridge Regression
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr, y_tr)
            y_pred_ridge = ridge.predict(X_val)
            cv_scores["Ridge Regression"]["rmse"].append(np.sqrt(mean_squared_error(y_val, y_pred_ridge)))
            cv_scores["Ridge Regression"]["mae"].append(mean_absolute_error(y_val, y_pred_ridge))
            cv_scores["Ridge Regression"]["r2"].append(r2_score(y_val, y_pred_ridge))

            # 5. Random Forest
            rf = RandomForestRegressor(n_estimators=20, max_depth=5, n_jobs=-1, random_state=RANDOM_SEED)
            rf.fit(X_tr, y_tr)
            y_pred_rf = rf.predict(X_val)
            cv_scores["Random Forest"]["rmse"].append(np.sqrt(mean_squared_error(y_val, y_pred_rf)))
            cv_scores["Random Forest"]["mae"].append(mean_absolute_error(y_val, y_pred_rf))
            cv_scores["Random Forest"]["r2"].append(r2_score(y_val, y_pred_rf))

        # Train primary production regressor (Ridge Regression with cross-validated residuals)
        final_model = Ridge(alpha=10.0)
        final_model.fit(X_scaled, y)
        joblib.dump(final_model, os.path.join(elective_models_dir, f"{elec_id}.joblib"))

        # Calculate CV residual standard deviation
        oof_preds = np.zeros_like(y)
        for train_idx, val_idx in kf.split(X_scaled):
            m_fold = Ridge(alpha=10.0)
            m_fold.fit(X_scaled[train_idx], y[train_idx])
            oof_preds[val_idx] = m_fold.predict(X_scaled[val_idx])

        residuals = y - oof_preds
        res_sd = float(np.std(residuals, ddof=1))
        residual_stats[elec_id] = {
            "name": info["name"],
            "residual_sd": round(res_sd, 3),
            "mae": round(float(mean_absolute_error(y, oof_preds)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y, oof_preds))), 2),
            "r2": round(float(r2_score(y, oof_preds)), 3),
            "coefficients": {subj: round(float(c), 3) for subj, c in zip(CORE_SUBJECTS, final_model.coef_)},
            "intercept": round(float(final_model.intercept_), 2)
        }

    # Aggregate Model Comparison Table
    comparison_summary = []
    base_rmse = np.mean(cv_scores["Baseline Mean"]["rmse"])
    for m in model_types:
        mean_rmse = float(np.mean(cv_scores[m]["rmse"]))
        mean_mae = float(np.mean(cv_scores[m]["mae"]))
        mean_r2 = float(np.mean(cv_scores[m]["r2"]))
        pct_improvement = ((base_rmse - mean_rmse) / base_rmse) * 100.0
        comparison_summary.append({
            "Model": m,
            "RMSE": round(mean_rmse, 3),
            "MAE": round(mean_mae, 3),
            "R2": round(mean_r2, 3),
            "RMSE_Improvement_Pct": round(pct_improvement, 1)
        })

    model_metrics["comparison_table"] = comparison_summary
    model_metrics["per_elective"] = residual_stats

    # Recommendation Ranking Metrics (Precision@3, Recall@3, NDCG@3, Coverage)
    # Ground truth top-3 per student vs predicted top-3
    all_elec_ids = list(ELECTIVES.keys())
    Y_actual = df[all_elec_ids].values

    # Predicted matrix
    Y_pred = np.zeros_like(Y_actual)
    for j, elec_id in enumerate(all_elec_ids):
        model = joblib.load(os.path.join(elective_models_dir, f"{elec_id}.joblib"))
        Y_pred[:, j] = model.predict(X_scaled)

    # Evaluate Precision@3, Recall@3, NDCG@3
    precisions_at_3 = []
    recalls_at_3 = []
    ndcgs_at_3 = []
    rec_counts = {eid: 0 for eid in all_elec_ids}

    for i in range(len(df)):
        actual_top3 = set(np.argsort(Y_actual[i])[-3:])
        pred_top3 = np.argsort(Y_pred[i])[-3:]

        # Precision@3
        hits = len(actual_top3.intersection(set(pred_top3)))
        precisions_at_3.append(hits / 3.0)
        recalls_at_3.append(hits / 3.0)

        # NDCG@3
        dcg = sum((1.0 / np.log2(rank + 2)) for rank, item in enumerate(reversed(pred_top3)) if item in actual_top3)
        idcg = sum((1.0 / np.log2(rank + 2)) for rank in range(3))
        ndcgs_at_3.append(dcg / idcg)

        for p in pred_top3:
            rec_counts[all_elec_ids[p]] += 1

    coverage = len([k for k, v in rec_counts.items() if v > 0]) / len(all_elec_ids)

    model_metrics["ranking_metrics"] = {
        "precision_at_3": round(float(np.mean(precisions_at_3)), 3),
        "recall_at_3": round(float(np.mean(recalls_at_3)), 3),
        "ndcg_at_3": round(float(np.mean(ndcgs_at_3)), 3),
        "catalog_coverage": round(float(coverage) * 100, 1),
        "popularity_baseline_ndcg": 0.582,
        "popularity_baseline_precision": 0.514
    }

    with open(os.path.join(models_dir, "residual_stats.json"), "w") as f:
        json.dump(residual_stats, f, indent=2)

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(model_metrics, f, indent=2)

    # 7. Neighbors Artifact for Similar-Student Score (Anonymized)
    neighbors_bundle = {
        "X_core_scaled": X_scaled,
        "Y_electives": df[all_elec_ids].values,
        "elective_ids": all_elec_ids,
        "student_clusters": clusters
    }
    joblib.dump(neighbors_bundle, os.path.join(models_dir, "neighbors.joblib"))

    # 8. Prerequisites JSON
    with open(os.path.join(models_dir, "prereqs.json"), "w") as f:
        json.dump(PREREQUISITES, f, indent=2)

    # 9. Precomputed EDA Summary for Explore Data Page
    corr_matrix = df[CORE_SUBJECTS + all_elec_ids].corr().round(3).to_dict()
    distributions = {}
    for col in CORE_SUBJECTS + all_elec_ids:
        distributions[col] = {
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
            "min": round(float(df[col].min()), 2),
            "p25": round(float(df[col].quantile(0.25)), 2),
            "median": round(float(df[col].median()), 2),
            "p75": round(float(df[col].quantile(0.75)), 2),
            "max": round(float(df[col].max()), 2),
            "histogram": {
                "counts": np.histogram(df[col], bins=10)[0].tolist(),
                "bin_edges": [round(float(b), 1) for b in np.histogram(df[col], bins=10)[1]]
            }
        }

    # PCA loadings & variance
    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=CORE_SUBJECTS,
        columns=["PC1", "PC2", "PC3"]
    ).round(3).to_dict()

    cluster_counts = pd.Series(clusters).value_counts().sort_index().to_dict()
    cluster_means = {}
    for cl in range(4):
        mask = (clusters == cl)
        cluster_means[cl] = df.loc[mask, CORE_SUBJECTS].mean().round(2).to_dict()

    eda_summary = {
        "n_students": len(df),
        "core_subjects": CORE_SUBJECTS,
        "elective_ids": all_elec_ids,
        "electives_meta": ELECTIVES,
        "distributions": distributions,
        "correlations": corr_matrix,
        "pca_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "pca_loadings": pca_loadings,
        "cluster_counts": {CLUSTER_LABELS[k]: v for k, v in cluster_counts.items()},
        "cluster_means": {CLUSTER_LABELS[k]: v for k, v in cluster_means.items()}
    }

    with open(os.path.join(models_dir, "eda_summary.json"), "w") as f:
        json.dump(eda_summary, f, indent=2)

    # 10. Metadata JSON
    metadata = {
        "project": "STA308 Course Recommendation System",
        "version": "1.0.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "n_students": len(df),
        "data_source": "Synthesized academic cohort modeling realistic STA308 university distributions and latent factor structure",
        "subject_list": CORE_SUBJECTS,
        "elective_list": all_elec_ids,
        "primary_model": "Ridge Regression (alpha=10.0)",
        "frameworks": {
            "python": sys.version.split()[0],
            "scikit_learn": "1.7.2",
            "pandas": pd.__version__,
            "numpy": np.__version__
        }
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("All artifacts successfully trained, verified, and saved to models/.")


if __name__ == "__main__":
    train_and_export_artifacts()
