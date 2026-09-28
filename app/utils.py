"""
UI and visualization helper utilities for the Course Recommendation Streamlit Application.
Redesigned according to the Claude Design System (VoltAgent/awesome-design-md):
- Warm cream canvas (#faf9f5) with dark warm ink (#141413)
- Warm coral signature primary (#cc785c) and active coral (#a9583e)
- Editorial slab-serif display typography ("Newsreader", "Lora", serif)
- Warm card surfaces (#efe9de, #f5f0e8) and dark product surfaces (#181715)
- Delicate hairlines (#e6dfd8) and pill badges (rounded: 9999px)
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

# Claude Design System Tokens
CLAUDE = {
    "primary": "#cc785c",
    "primary_active": "#a9583e",
    "primary_disabled": "#e6dfd8",
    "ink": "#141413",
    "body": "#3d3d3a",
    "body_strong": "#252523",
    "muted": "#6c6a64",
    "muted_soft": "#8e8b82",
    "hairline": "#e6dfd8",
    "hairline_soft": "#ebe6df",
    "canvas": "#faf9f5",
    "surface_soft": "#f5f0e8",
    "surface_card": "#efe9de",
    "surface_cream_strong": "#e8e0d2",
    "surface_dark": "#181715",
    "surface_dark_elevated": "#252320",
    "surface_dark_soft": "#1f1e1b",
    "on_primary": "#ffffff",
    "on_dark": "#faf9f5",
    "on_dark_soft": "#a09d96",
    "accent_teal": "#5db8a6",
    "accent_amber": "#e8a55a",
    "success": "#5db872",
    "warning": "#d4a017",
    "error": "#c64545"
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
    """Inject Claude editorial design system CSS."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* App-wide canvas and typography */
        .stApp {{
            background-color: {CLAUDE['canvas']};
            color: {CLAUDE['body']};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* Editorial Display Headings */
        h1, h2, h3, .editorial-heading {{
            font-family: 'Newsreader', Georgia, serif !important;
            font-weight: 400 !important;
            color: {CLAUDE['ink']} !important;
            letter-spacing: -0.025em;
        }}

        h1 {{
            font-size: 2.5rem !important;
            line-height: 1.12 !important;
        }}

        h2 {{
            font-size: 1.85rem !important;
            line-height: 1.2 !important;
            margin-top: 1.6rem !important;
        }}

        h3 {{
            font-size: 1.35rem !important;
            line-height: 1.3 !important;
        }}

        /* Hero Header Band - Editorial Cream with Coral Accent */
        .claude-hero {{
            background-color: {CLAUDE['surface_soft']};
            border: 1px solid {CLAUDE['hairline']};
            border-radius: 16px;
            padding: 2.4rem 2.2rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }}

        .claude-hero::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 5px;
            height: 100%;
            background-color: {CLAUDE['primary']};
        }}

        .claude-hero h1 {{
            margin: 0 0 0.6rem 0 !important;
            font-size: 2.4rem !important;
            color: {CLAUDE['ink']} !important;
        }}

        .claude-hero p {{
            font-size: 1.1rem;
            color: {CLAUDE['muted']};
            margin: 0;
            line-height: 1.6;
            max-width: 820px;
        }}

        /* Badge Pills */
        .badge-pill {{
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 500;
            margin-right: 6px;
            margin-bottom: 6px;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.01em;
        }}

        .badge-coral {{
            background-color: {CLAUDE['primary']};
            color: {CLAUDE['on_primary']};
        }}

        .badge-cream {{
            background-color: {CLAUDE['surface_card']};
            color: {CLAUDE['body_strong']};
            border: 1px solid {CLAUDE['hairline']};
        }}

        .badge-dark {{
            background-color: {CLAUDE['surface_dark']};
            color: {CLAUDE['on_dark']};
            border: 1px solid {CLAUDE['surface_dark_elevated']};
        }}

        .badge-teal {{
            background-color: #E6F7F4;
            color: #0E6857;
            border: 1px solid #B8E8E0;
        }}

        /* Feature Card (Light Cream Surface) */
        .claude-card {{
            background-color: {CLAUDE['surface_card']};
            border: 1px solid {CLAUDE['hairline']};
            border-radius: 12px;
            padding: 1.4rem;
            margin-bottom: 1.2rem;
            transition: all 0.2s ease-in-out;
        }}

        .claude-card:hover {{
            background-color: {CLAUDE['surface_cream_strong']};
            border-color: {CLAUDE['primary']};
        }}

        /* Dark Product Chrome Card (for code, models, terminal) */
        .claude-dark-card {{
            background-color: {CLAUDE['surface_dark']};
            color: {CLAUDE['on_dark']};
            border: 1px solid {CLAUDE['surface_dark_elevated']};
            border-radius: 12px;
            padding: 1.4rem;
            margin-bottom: 1.2rem;
            font-family: 'JetBrains Mono', monospace;
        }}

        .claude-dark-card h3, .claude-dark-card h4 {{
            color: {CLAUDE['on_dark']} !important;
            font-family: 'Newsreader', serif !important;
        }}

        /* Stat Box */
        .claude-stat {{
            background-color: {CLAUDE['surface_card']};
            border: 1px solid {CLAUDE['hairline']};
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }}

        .claude-stat-value {{
            font-family: 'Newsreader', Georgia, serif;
            font-size: 2.1rem;
            font-weight: 500;
            color: {CLAUDE['ink']};
            line-height: 1.1;
        }}

        .claude-stat-label {{
            font-size: 0.76rem;
            color: {CLAUDE['muted']};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 5px;
        }}

        /* Recommendation Rank Card */
        .rank-circle-coral {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: {CLAUDE['primary']};
            color: {CLAUDE['on_primary']};
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 1rem;
            margin-right: 12px;
        }}

        /* Disclaimer Callout Band */
        .claude-callout {{
            background-color: {CLAUDE['surface_soft']};
            border-left: 3px solid {CLAUDE['primary']};
            border-radius: 0 8px 8px 0;
            padding: 1rem 1.2rem;
            margin: 1.5rem 0;
            font-size: 0.9rem;
            color: {CLAUDE['body']};
            line-height: 1.55;
        }}

        /* Button Styling Overrides */
        div.stButton > button[kind="primary"] {{
            background-color: {CLAUDE['primary']} !important;
            color: {CLAUDE['on_primary']} !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: 0.6rem 1.4rem !important;
            transition: background-color 0.15s ease !important;
        }}

        div.stButton > button[kind="primary"]:hover {{
            background-color: {CLAUDE['primary_active']} !important;
            box-shadow: 0 4px 12px rgba(204, 120, 92, 0.25) !important;
        }}

        div.stButton > button[kind="secondary"] {{
            background-color: {CLAUDE['canvas']} !important;
            color: {CLAUDE['ink']} !important;
            border: 1px solid {CLAUDE['hairline']} !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
        }}

        div.stButton > button[kind="secondary"]:hover {{
            border-color: {CLAUDE['primary']} !important;
            background-color: {CLAUDE['surface_soft']} !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {CLAUDE['surface_soft']};
            border-right: 1px solid {CLAUDE['hairline']};
        }}

        /* Input Elements */
        div[data-baseweb="input"] {{
            background-color: {CLAUDE['canvas']} !important;
            border-radius: 8px !important;
            border-color: {CLAUDE['hairline']} !important;
        }}

        /* Tabs */
        div[data-testid="stTabs"] button[role="tab"] {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            color: {CLAUDE['muted']} !important;
        }}

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            color: {CLAUDE['ink']} !important;
            border-bottom: 2px solid {CLAUDE['primary']} !important;
        }}

        /* Metrics */
        div[data-testid="stMetricValue"] {{
            font-family: 'Newsreader', serif !important;
            font-size: 1.85rem !important;
            color: {CLAUDE['ink']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def create_radar_chart(student_grades: dict, cluster_centers: dict, cohort_means: dict) -> go.Figure:
    """Generate Claude-styled interactive radar chart on warm cream canvas."""
    subjects = list(student_grades.keys())
    r_student = [student_grades[s] for s in subjects] + [student_grades[subjects[0]]]
    r_cluster = [cluster_centers[s] for s in subjects] + [cluster_centers[subjects[0]]]
    r_cohort = [cohort_means[s] for s in subjects] + [cohort_means[subjects[0]]]
    theta = subjects + [subjects[0]]

    fig = go.Figure()

    # Cohort benchmark (dark warm ink dash)
    fig.add_trace(go.Scatterpolar(
        r=r_cohort,
        theta=theta,
        fill=None,
        name="Cohort Average",
        line=dict(color=CLAUDE["muted"], width=1.5, dash="dash"),
        hoverinfo="r+name"
    ))

    # Cluster benchmark (accent teal fill)
    fig.add_trace(go.Scatterpolar(
        r=r_cluster,
        theta=theta,
        fill="toself",
        fillcolor="rgba(93, 184, 166, 0.14)",
        name="Archetype Benchmark",
        line=dict(color=CLAUDE["accent_teal"], width=2),
        hoverinfo="r+name"
    ))

    # Student actual grades (signature warm coral)
    fig.add_trace(go.Scatterpolar(
        r=r_student,
        theta=theta,
        fill="toself",
        fillcolor="rgba(204, 120, 92, 0.22)",
        name="Your Profile",
        line=dict(color=CLAUDE["primary"], width=2.8),
        hoverinfo="r+name"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[40, 100],
                tickfont=dict(size=10, color=CLAUDE["muted"]),
                gridcolor=CLAUDE["hairline"],
                linecolor=CLAUDE["hairline"]
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=CLAUDE["ink"], family="Inter"),
                gridcolor=CLAUDE["hairline"],
                linecolor=CLAUDE["hairline"]
            ),
            bgcolor=CLAUDE["surface_soft"]
        ),
        margin=dict(l=35, r=35, t=25, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11, family="Inter", color=CLAUDE["body"])
        ),
        height=370,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_interval_bar_chart(eligible_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart with 95% prediction intervals in Claude editorial aesthetic."""
    plot_df = eligible_df.sort_values(by="combined_score", ascending=True).copy()

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
            colorscale=[
                [0.0, CLAUDE["surface_cream_strong"]],
                [0.5, CLAUDE["accent_amber"]],
                [1.0, CLAUDE["primary"]]
            ],
            line=dict(color=CLAUDE["primary_active"], width=1.0),
            showscale=False
        ),
        error_x=dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color=CLAUDE["ink"],
            thickness=1.8,
            width=5
        ),
        hovertemplate="<b>%{y}</b><br>Predicted Grade: %{x:.1f}<br>Rank Score: %{customdata[0]:.1f}<br>95% Interval: [%{customdata[1]:.1f}, %{customdata[2]:.1f}]<extra></extra>",
        customdata=plot_df[["combined_score", "interval_lo", "interval_hi"]].values
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Predicted Grade (0 - 100 Scale)", font=dict(family="Inter", size=11, color=CLAUDE["muted"])),
            range=[40, 100],
            gridcolor=CLAUDE["hairline"],
            zeroline=False,
            tickfont=dict(color=CLAUDE["muted"])
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11, color=CLAUDE["ink"], family="Inter")
        ),
        margin=dict(l=10, r=20, t=15, b=35),
        height=max(320, len(plot_df) * 44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_contribution_bar(contributions: list) -> go.Figure:
    """Create feature contribution bar chart in Claude warm coral and muted ink palette."""
    subjects = [c.subject for c in reversed(contributions)]
    impacts = [c.point_impact for c in reversed(contributions)]
    colors = [CLAUDE["primary"] if val >= 0 else CLAUDE["muted_soft"] for val in impacts]

    fig = go.Figure(go.Bar(
        y=subjects,
        x=impacts,
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b>: %{x:+.2f} points<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Impact on Predicted Score (Points)", font=dict(family="Inter", size=11, color=CLAUDE["muted"])),
            zeroline=True,
            zerolinecolor=CLAUDE["hairline"],
            zerolinewidth=1.5,
            gridcolor=CLAUDE["hairline_soft"],
            tickfont=dict(color=CLAUDE["muted"])
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color=CLAUDE["ink"], family="Inter")
        ),
        margin=dict(l=10, r=10, t=10, b=30),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
