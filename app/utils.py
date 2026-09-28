"""
UI and visualization helper utilities for the Course Recommendation Streamlit Application.
Provides styling, caching, Plotly chart generation, and data transformations.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.recommend import ModelArtifacts, CORE_SUBJECTS

# Color-blind safe curated palette (Okabe-Ito + Modern Slate)
OKABE_ITO = {
    "blue": "#0072B2",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "dark": "#1E293B",
    "gray": "#64748B",
    "light_gray": "#F1F5F9",
    "card_bg": "#FFFFFF",
    "border": "#E2E8F0"
}


@st.cache_resource
def get_model_artifacts() -> ModelArtifacts:
    """Load and cache all models and metadata."""
    return ModelArtifacts.get_instance()


@st.cache_data
def get_eda_summary() -> dict:
    """Load cached EDA metrics and distributions."""
    artifacts = get_model_artifacts()
    return artifacts.eda_summary


@st.cache_data
def get_metrics_summary() -> dict:
    """Load precomputed model evaluation metrics."""
    with open(os.path.join(BASE_DIR, "models", "metrics.json"), "r") as f:
        return json.load(f)


@st.cache_data
def get_sample_profiles() -> dict:
    """Load preset student demonstration profiles."""
    with open(os.path.join(BASE_DIR, "data", "sample_profiles.json"), "r") as f:
        return json.load(f)


def inject_custom_css():
    """Inject modern, clean, accessible CSS styling for cards, pills, and layout."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #2563EB 100%);
            padding: 2.2rem 2.2rem;
            border-radius: 16px;
            color: #FFFFFF;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        }

        .main-header h1 {
            color: #FFFFFF !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            margin: 0 0 0.5rem 0 !important;
            letter-spacing: -0.02em;
        }

        .main-header p {
            color: #93C5FD !important;
            font-size: 1.05rem !important;
            margin: 0 !important;
            font-weight: 500;
        }

        .badge-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
        }

        .badge-source {
            background-color: #DBEAFE;
            color: #1E40AF;
            border: 1px solid #BFDBFE;
        }

        .badge-category {
            background-color: #FEF3C7;
            color: #92400E;
            border: 1px solid #FDE68A;
        }

        .badge-cluster {
            background-color: #DCFCE7;
            color: #166534;
            border: 1px solid #BBF7D0;
        }

        .recommendation-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.3rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .recommendation-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
            border-color: #93C5FD;
        }

        .rank-circle {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: #2563EB;
            color: #FFFFFF;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.1rem;
            margin-right: 12px;
        }

        .stat-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }

        .stat-label {
            font-size: 0.78rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }

        .disclaimer-banner {
            background-color: #FFFBEB;
            border-left: 4px solid #F59E0B;
            padding: 1rem 1.2rem;
            border-radius: 0 8px 8px 0;
            margin: 1.5rem 0;
            font-size: 0.88rem;
            color: #78350F;
        }

        .how-step {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            height: 100%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        }

        .how-step-num {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: #EFF6FF;
            color: #2563EB;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.8rem;
            border: 2px solid #BFDBFE;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def create_radar_chart(student_grades: dict, cluster_centers: dict, cohort_means: dict) -> go.Figure:
    """Generate an interactive radar chart comparing student grades to cluster average and cohort."""
    subjects = list(student_grades.keys())
    # Close the radar loop
    r_student = [student_grades[s] for s in subjects] + [student_grades[subjects[0]]]
    r_cluster = [cluster_centers[s] for s in subjects] + [cluster_centers[subjects[0]]]
    r_cohort = [cohort_means[s] for s in subjects] + [cohort_means[subjects[0]]]
    theta = subjects + [subjects[0]]

    fig = go.Figure()

    # Cohort overall mean trace
    fig.add_trace(go.Scatterpolar(
        r=r_cohort,
        theta=theta,
        fill=None,
        name="Cohort Average",
        line=dict(color=OKABE_ITO["gray"], width=1.5, dash="dash"),
        hoverinfo="r+name"
    ))

    # Assigned cluster mean trace
    fig.add_trace(go.Scatterpolar(
        r=r_cluster,
        theta=theta,
        fill="toself",
        fillcolor="rgba(0, 158, 115, 0.12)",
        name="Cluster Benchmark",
        line=dict(color=OKABE_ITO["green"], width=2),
        hoverinfo="r+name"
    ))

    # Student actual grades trace
    fig.add_trace(go.Scatterpolar(
        r=r_student,
        theta=theta,
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.22)",
        name="Your Profile",
        line=dict(color=OKABE_ITO["blue"], width=3),
        hoverinfo="r+name"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[40, 100],
                tickfont=dict(size=10, color="#64748B"),
                gridcolor="#E2E8F0"
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#0F172A", family="Plus Jakarta Sans"),
                gridcolor="#E2E8F0"
            ),
            bgcolor="#F8FAFC"
        ),
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_interval_bar_chart(eligible_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart with 95% prediction intervals (error bars)."""
    # Sort ascending for clean top-down horizontal ranking
    plot_df = eligible_df.sort_values(by="combined_score", ascending=True).copy()

    # Calculate asymmetric or symmetric error lengths
    error_minus = plot_df["pred_grade"] - plot_df["interval_lo"]
    error_plus = plot_df["interval_hi"] - plot_df["pred_grade"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=plot_df["name"],
        x=plot_df["pred_grade"],
        orientation="h",
        name="Predicted Grade",
        marker=dict(
            color=plot_df["combined_score"],
            colorscale=[[0, "#93C5FD"], [0.5, "#3B82F6"], [1.0, "#1D4ED8"]],
            line=dict(color="#1E40AF", width=1.2),
            showscale=False
        ),
        error_x=dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color="#0F172A",
            thickness=2,
            width=6
        ),
        hovertemplate="<b>%{y}</b><br>Predicted Grade: %{x:.1f}<br>Rank Score: %{customdata[0]:.1f}<br>95% CI: [%{customdata[1]:.1f}, %{customdata[2]:.1f}]<extra></extra>",
        customdata=plot_df[["combined_score", "interval_lo", "interval_hi"]].values
    ))

    fig.update_layout(
        xaxis=dict(
            title="Predicted Grade (0 - 100 scale)",
            range=[40, 100],
            gridcolor="#E2E8F0",
            zeroline=False
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11, color="#0F172A")
        ),
        margin=dict(l=10, r=20, t=20, b=40),
        height=max(320, len(plot_df) * 44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_contribution_bar(contributions: list) -> go.Figure:
    """Create small horizontal bar chart of feature contributions."""
    subjects = [c.subject for c in reversed(contributions)]
    impacts = [c.point_impact for c in reversed(contributions)]
    colors = [OKABE_ITO["green"] if val >= 0 else OKABE_ITO["vermilion"] for val in impacts]

    fig = go.Figure(go.Bar(
        y=subjects,
        x=impacts,
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b>: %{x:+.2f} points<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(
            title="Impact on Predicted Score (Points)",
            zeroline=True,
            zerolinecolor="#94A3B8",
            zerolinewidth=1.5,
            gridcolor="#F1F5F9"
        ),
        yaxis=dict(title=""),
        margin=dict(l=10, r=10, t=10, b=30),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
