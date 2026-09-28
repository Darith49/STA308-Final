"""
Smoke tests ensuring all application views, utilities, and components import and execute cleanly.
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.utils import (
    get_model_artifacts,
    get_eda_summary,
    get_metrics_summary,
    get_sample_profiles,
    create_radar_chart,
    create_forest_plot,
    create_interval_bar_chart,
    create_contribution_bar
)
from src.recommend import recommend


def test_artifacts_and_summaries_load():
    """Verify artifact singletons and precomputed JSONs load without issue."""
    artifacts = get_model_artifacts()
    assert artifacts.scaler is not None
    assert artifacts.pca is not None
    assert len(artifacts.elective_models) == 8

    eda = get_eda_summary()
    assert eda["n_students"] == 1200
    assert "Calculus" in eda["distributions"]

    metrics = get_metrics_summary()
    assert len(metrics["comparison_table"]) == 5
    assert metrics["ranking_metrics"]["catalog_coverage"] == 100.0

    samples = get_sample_profiles()
    assert "Quantitative Thinker" in samples


def test_plotly_chart_builders():
    """Verify Plotly chart helper functions run without raising errors."""
    artifacts = get_model_artifacts()
    eda = get_eda_summary()

    sample_grades = {"Calculus": 80.0, "Statistics": 85.0, "Programming": 90.0, "English": 75.0, "Physics": 80.0, "Economics": 70.0}
    cluster_centers = {s: 75.0 for s in sample_grades}
    cohort_means = {s: eda["distributions"][s]["mean"] for s in sample_grades}

    radar_fig = create_radar_chart(sample_grades, cluster_centers, cohort_means)
    assert radar_fig is not None

    rec = recommend(sample_grades, top_n=3)
    assert rec.is_valid is True

    interval_fig = create_interval_bar_chart(rec.all_eligible)
    assert interval_fig is not None

    forest_fig = create_forest_plot(rec.all_eligible)
    assert forest_fig is not None

    top_eid = rec.top_electives.iloc[0]["elective_id"]
    c_list = rec.explanations[top_eid]
    contrib_fig = create_contribution_bar(c_list)
    assert contrib_fig is not None


def test_all_views_import():
    """Verify all view modules import cleanly."""
    import app.views.home_view as hv
    import app.views.recommend_view as rv
    import app.views.explore_view as ev
    import app.views.performance_view as pv
    import app.views.methodology_view as mv

    assert hasattr(hv, "render_home")
    assert hasattr(rv, "render_recommend")
    assert hasattr(ev, "render_explore")
    assert hasattr(pv, "render_performance")
    assert hasattr(mv, "render_methodology")
