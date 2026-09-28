"""
UI, design system tokens, and visualization utilities for Course Recommendation System.
Engineered for executive, clean, and professional academic advising using the
Claude Warm Editorial Design System (Newsreader serif + Inter humanist sans + Warm Coral #cc785c).
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.recommend import ModelArtifacts, CORE_SUBJECTS

# Claude Warm Editorial Design Tokens
CLAUDE = {
    "primary": "#cc785c",
    "primary_hover": "#b86347",
    "primary_active": "#a9583e",
    "primary_light": "#faebe6",
    "ink": "#141413",
    "body": "#2c2c29",
    "body_muted": "#5e5c56",
    "muted_soft": "#8e8b82",
    "hairline": "#e6dfd8",
    "hairline_soft": "#f0ebe4",
    "canvas": "#faf9f5",
    "surface_soft": "#f5f0e8",
    "surface_card": "#efe9de",
    "surface_cream_strong": "#e8dfd1",
    "surface_dark": "#181715",
    "surface_dark_elevated": "#252320",
    "on_primary": "#ffffff",
    "on_dark": "#faf9f5",
    "on_dark_soft": "#a09d96",
    "accent_teal": "#0e7b6c",
    "accent_teal_light": "#e6f6f3",
    "accent_amber": "#b46914",
    "accent_amber_light": "#fef5e7",
    "success": "#2e7d43",
    "error": "#b93838",
    "error_light": "#fdf2f2"
}

# Standard chart configuration to remove clunky toolbars for professional look
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "staticPlot": False
}


@st.cache_resource
def get_model_artifacts() -> ModelArtifacts:
    """Load and cache model singletons."""
    return ModelArtifacts.get_instance()


@st.cache_data
def get_eda_summary() -> dict:
    """Load cached dataset summary statistics."""
    artifacts = get_model_artifacts()
    return artifacts.eda_summary


@st.cache_data
def get_metrics_summary() -> dict:
    """Load precomputed model evaluation metrics."""
    with open(os.path.join(BASE_DIR, "models", "metrics.json"), "r") as f:
        return json.load(f)


@st.cache_data
def get_sample_profiles() -> dict:
    """Load sample student benchmark profiles."""
    with open(os.path.join(BASE_DIR, "data", "sample_profiles.json"), "r") as f:
        return json.load(f)


def inject_custom_css():
    """Inject polished, modern Claude Warm Editorial CSS for professional presentation."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Clean app background and typography */
        .stApp {{
            background-color: {CLAUDE['canvas']};
            color: {CLAUDE['body']};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 15px;
            letter-spacing: -0.01em;
        }}

        /* Editorial Display Headings */
        h1, h2, h3, .editorial-heading {{
            font-family: 'Newsreader', Georgia, serif !important;
            font-weight: 400 !important;
            color: {CLAUDE['ink']} !important;
            letter-spacing: -0.03em;
        }}

        h1 {{
            font-size: 2.35rem !important;
            line-height: 1.15 !important;
            margin-bottom: 0.4rem !important;
        }}

        h2 {{
            font-size: 1.65rem !important;
            line-height: 1.2 !important;
            margin-top: 1.8rem !important;
            margin-bottom: 0.6rem !important;
        }}

        h3 {{
            font-size: 1.25rem !important;
            line-height: 1.3 !important;
            margin-top: 1.2rem !important;
            margin-bottom: 0.5rem !important;
        }}

        p, span, label {{
            color: {CLAUDE['body']};
        }}

        /* Executive Hero Banner */
        .claude-hero {{
            background: linear-gradient(135deg, {CLAUDE['surface_soft']} 0%, {CLAUDE['surface_card']} 100%);
            border: 1px solid {CLAUDE['hairline']};
            border-radius: 14px;
            padding: 2.2rem 2.2rem;
            margin-bottom: 1.8rem;
            position: relative;
            box-shadow: 0 2px 8px rgba(20, 20, 19, 0.03);
        }}

        .claude-hero::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background-color: {CLAUDE['primary']};
            border-top-left-radius: 14px;
            border-bottom-left-radius: 14px;
        }}

        .claude-hero h1 {{
            margin: 0 0 0.5rem 0 !important;
            font-size: 2.35rem !important;
        }}

        .claude-hero p {{
            font-size: 1.05rem;
            color: {CLAUDE['body_muted']};
            margin: 0;
            line-height: 1.6;
            max-width: 840px;
        }}

        /* Clean Badges & Tags */
        .badge-pill {{
            display: inline-flex;
            align-items: center;
            padding: 4px 11px;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 500;
            margin-right: 6px;
            margin-bottom: 6px;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.02em;
            line-height: 1.3;
        }}

        .badge-coral {{
            background-color: {CLAUDE['primary']};
            color: {CLAUDE['on_primary']};
        }}

        .badge-cream {{
            background-color: {CLAUDE['surface_card']};
            color: {CLAUDE['body']};
            border: 1px solid {CLAUDE['hairline']};
        }}

        .badge-teal {{
            background-color: {CLAUDE['accent_teal_light']};
            color: {CLAUDE['accent_teal']};
            border: 1px solid #c2ece4;
        }}

        .badge-amber {{
            background-color: {CLAUDE['accent_amber_light']};
            color: {CLAUDE['accent_amber']};
            border: 1px solid #f9e2ba;
        }}

        /* Cards */
        .claude-card {{
            background-color: {CLAUDE['surface_card']};
            border: 1px solid {CLAUDE['hairline']};
            border-radius: 12px;
            padding: 1.35rem 1.4rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 1px 3px rgba(20, 20, 19, 0.02);
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }}

        .claude-card:hover {{
            border-color: {CLAUDE['primary']};
            box-shadow: 0 3px 10px rgba(20, 20, 19, 0.04);
        }}

        /* Dark Product Surface Cards (Terminal, Architecture) */
        .claude-dark-card {{
            background-color: {CLAUDE['surface_dark']};
            color: {CLAUDE['on_dark']};
            border: 1px solid {CLAUDE['surface_dark_elevated']};
            border-radius: 12px;
            padding: 1.3rem 1.4rem;
            margin-bottom: 1.1rem;
            font-family: 'JetBrains Mono', monospace;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        }}

        /* Executive Scorecard Box */
        .claude-stat {{
            background-color: {CLAUDE['surface_card']};
            border: 1px solid {CLAUDE['hairline']};
            border-radius: 10px;
            padding: 1rem 0.8rem;
            text-align: center;
        }}

        .claude-stat-value {{
            font-family: 'Newsreader', Georgia, serif;
            font-size: 2.1rem;
            font-weight: 500;
            color: {CLAUDE['ink']};
            line-height: 1.05;
        }}

        .claude-stat-label {{
            font-size: 0.74rem;
            color: {CLAUDE['body_muted']};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-top: 6px;
        }}

        /* Rank Badge */
        .rank-circle-coral {{
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background-color: {CLAUDE['primary']};
            color: {CLAUDE['on_primary']};
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.92rem;
            margin-right: 12px;
            flex-shrink: 0;
        }}

        /* Advisory & Disclaimer Callouts */
        .claude-callout {{
            background-color: {CLAUDE['surface_soft']};
            border-left: 3px solid {CLAUDE['primary']};
            border-radius: 0 10px 10px 0;
            padding: 0.95rem 1.25rem;
            margin: 1.4rem 0;
            font-size: 0.9rem;
            color: {CLAUDE['body']};
            line-height: 1.55;
            border-top: 1px solid {CLAUDE['hairline']};
            border-right: 1px solid {CLAUDE['hairline']};
            border-bottom: 1px solid {CLAUDE['hairline']};
        }}

        /* Button Styling Overrides */
        div.stButton > button[kind="primary"] {{
            background-color: {CLAUDE['primary']} !important;
            color: {CLAUDE['on_primary']} !important;
            border: 1px solid {CLAUDE['primary']} !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 0.92rem !important;
            padding: 0.55rem 1.3rem !important;
            transition: all 0.15s ease !important;
        }}

        div.stButton > button[kind="primary"]:hover {{
            background-color: {CLAUDE['primary_hover']} !important;
            border-color: {CLAUDE['primary_hover']} !important;
            box-shadow: 0 4px 12px rgba(204, 120, 92, 0.22) !important;
        }}

        div.stButton > button[kind="secondary"] {{
            background-color: {CLAUDE['canvas']} !important;
            color: {CLAUDE['ink']} !important;
            border: 1px solid {CLAUDE['hairline']} !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 0.92rem !important;
            transition: all 0.15s ease !important;
        }}

        div.stButton > button[kind="secondary"]:hover {{
            border-color: {CLAUDE['primary']} !important;
            background-color: {CLAUDE['surface_soft']} !important;
            color: {CLAUDE['primary']} !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {CLAUDE['surface_soft']};
            border-right: 1px solid {CLAUDE['hairline']};
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
            padding: 7px 12px !important;
            border-radius: 8px !important;
            margin-bottom: 3px !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            color: {CLAUDE['ink']} !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
            background-color: {CLAUDE['surface_card']} !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
            background-color: {CLAUDE['surface_card']} !important;
            border: 1px solid {CLAUDE['hairline']} !important;
            font-weight: 600 !important;
        }}

        /* Modern Input Container Styling */
        div[data-baseweb="input"] {{
            background-color: {CLAUDE['canvas']} !important;
            border-radius: 8px !important;
            border: 1px solid {CLAUDE['hairline']} !important;
            transition: border-color 0.15s ease !important;
        }}

        div[data-baseweb="input"]:focus-within {{
            border-color: {CLAUDE['primary']} !important;
            box-shadow: 0 0 0 2px rgba(204, 120, 92, 0.15) !important;
        }}

        /* Tabs */
        div[data-testid="stTabs"] button[role="tab"] {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            color: {CLAUDE['body_muted']} !important;
            padding: 8px 16px !important;
        }}

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            color: {CLAUDE['ink']} !important;
            font-weight: 600 !important;
            border-bottom: 2px solid {CLAUDE['primary']} !important;
        }}

        /* Metrics */
        div[data-testid="stMetricValue"] {{
            font-family: 'Newsreader', serif !important;
            font-size: 1.85rem !important;
            font-weight: 500 !important;
            color: {CLAUDE['ink']} !important;
        }}

        div[data-testid="stMetricLabel"] {{
            color: {CLAUDE['body_muted']} !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}

        /* Clean Expanders */
        div[data-testid="stExpander"] {{
            border: 1px solid {CLAUDE['hairline']} !important;
            border-radius: 10px !important;
            background-color: {CLAUDE['surface_soft']} !important;
            margin-bottom: 0.8rem !important;
        }}

        /* Hide unwanted Plotly and Streamlit auto-margins */
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1180px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def create_radar_chart(student_grades: dict, cluster_centers: dict, cohort_means: dict) -> go.Figure:
    """Generate professional, clean radar chart on warm cream canvas."""
    subjects = list(student_grades.keys())
    r_student = [student_grades[s] for s in subjects] + [student_grades[subjects[0]]]
    r_cluster = [cluster_centers[s] for s in subjects] + [cluster_centers[subjects[0]]]
    r_cohort = [cohort_means[s] for s in subjects] + [cohort_means[subjects[0]]]
    theta = subjects + [subjects[0]]

    fig = go.Figure()

    # Cohort benchmark (subtle neutral line)
    fig.add_trace(go.Scatterpolar(
        r=r_cohort,
        theta=theta,
        fill=None,
        name="Cohort Average",
        line=dict(color=CLAUDE["muted_soft"], width=1.5, dash="dash"),
        hoverinfo="r+name"
    ))

    # Cluster benchmark (accent teal fill)
    fig.add_trace(go.Scatterpolar(
        r=r_cluster,
        theta=theta,
        fill="toself",
        fillcolor="rgba(14, 123, 108, 0.10)",
        name="Archetype Benchmark",
        line=dict(color=CLAUDE["accent_teal"], width=1.8),
        hoverinfo="r+name"
    ))

    # Student actual grades (signature warm coral)
    fig.add_trace(go.Scatterpolar(
        r=r_student,
        theta=theta,
        fill="toself",
        fillcolor="rgba(204, 120, 92, 0.25)",
        name="Your Grades",
        line=dict(color=CLAUDE["primary"], width=2.8),
        hoverinfo="r+name"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[40, 100],
                tickfont=dict(size=9, color=CLAUDE["body_muted"], family="Inter"),
                gridcolor=CLAUDE["hairline"],
                linecolor=CLAUDE["hairline"]
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=CLAUDE["ink"], family="Inter", weight=600),
                gridcolor=CLAUDE["hairline"],
                linecolor=CLAUDE["hairline"]
            ),
            bgcolor=CLAUDE["surface_soft"]
        ),
        margin=dict(l=30, r=30, t=20, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11, family="Inter", color=CLAUDE["body"])
        ),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_interval_bar_chart(eligible_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart with 95% prediction intervals and academic distinction tiers."""
    plot_df = eligible_df.sort_values(by="combined_score", ascending=True).copy()

    error_minus = plot_df["pred_grade"] - plot_df["interval_lo"]
    error_plus = plot_df["interval_hi"] - plot_df["pred_grade"]

    fig = go.Figure()

    # Reference guide line for Honours / Distinction threshold (80)
    fig.add_vline(
        x=80,
        line_width=1,
        line_dash="dot",
        line_color=CLAUDE["accent_amber"],
        annotation_text="High Distinction (80+)",
        annotation_position="top right",
        annotation_font=dict(size=10, color=CLAUDE["accent_amber"], family="Inter")
    )

    fig.add_trace(go.Bar(
        y=plot_df["name"],
        x=plot_df["pred_grade"],
        orientation="h",
        name="Predicted Grade",
        marker=dict(
            color=plot_df["combined_score"],
            colorscale=[
                [0.0, CLAUDE["surface_cream_strong"]],
                [0.45, CLAUDE["accent_amber"]],
                [1.0, CLAUDE["primary"]]
            ],
            line=dict(color=CLAUDE["primary_hover"], width=1.0),
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
        hovertemplate="<b>%{y}</b><br>Predicted Grade: <b>%{x:.1f}</b><br>Combined Index: %{customdata[0]:.1f}<br>95% CI: [%{customdata[1]:.1f}, %{customdata[2]:.1f}]<extra></extra>",
        customdata=plot_df[["combined_score", "interval_lo", "interval_hi"]].values
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Expected Grade (0 – 100)", font=dict(family="Inter", size=11, color=CLAUDE["body_muted"])),
            range=[40, 100],
            gridcolor=CLAUDE["hairline"],
            zeroline=False,
            tickfont=dict(color=CLAUDE["body_muted"], size=10)
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11, color=CLAUDE["ink"], family="Inter", weight=500)
        ),
        margin=dict(l=10, r=20, t=25, b=35),
        height=max(300, len(plot_df) * 40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_contribution_bar(contributions: list) -> go.Figure:
    """Create clean horizontal feature attribution bar chart."""
    subjects = [c.subject for c in reversed(contributions)]
    impacts = [c.point_impact for c in reversed(contributions)]
    colors = [CLAUDE["primary"] if val >= 0 else CLAUDE["muted_soft"] for val in impacts]

    fig = go.Figure(go.Bar(
        y=subjects,
        x=impacts,
        orientation="h",
        marker=dict(color=colors, line=dict(color=CLAUDE["ink"], width=0.5)),
        hovertemplate="<b>%{y}</b>: %{x:+.2f} points impact<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Score Impact (Points)", font=dict(family="Inter", size=10, color=CLAUDE["body_muted"])),
            zeroline=True,
            zerolinecolor=CLAUDE["hairline"],
            zerolinewidth=1.5,
            gridcolor=CLAUDE["hairline_soft"],
            tickfont=dict(color=CLAUDE["body_muted"], size=10)
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color=CLAUDE["ink"], family="Inter", size=11, weight=500)
        ),
        margin=dict(l=10, r=10, t=10, b=30),
        height=160,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
