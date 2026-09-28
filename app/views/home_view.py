"""
Home Page View for Course Recommendation System.
Engineered for executive clarity, professional presentation, and intuitive navigation.
"""

import streamlit as st
from app.utils import inject_custom_css, CLAUDE


def render_home(navigate_to):
    """Render the executive Home landing view in Claude Warm Editorial aesthetic."""
    inject_custom_css()

    st.markdown(
        f"""
        <div class="claude-hero">
            <div style="margin-bottom: 0.8rem;">
                <span class="badge-pill badge-coral">STA308 Final Project</span>
                <span class="badge-pill badge-cream">Intelligent Academic Advising</span>
                <span class="badge-pill badge-teal">Ridge Regression + PCA + 15-NN</span>
            </div>
            <h1>Course Recommendation System</h1>
            <p>An explainable, uncertainty-aware academic advising platform grounded in statistical machine learning. Built to assist university students and advisors with evidence-backed curriculum decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Executive Scorecard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="claude-stat">
                <div class="claude-stat-value">1,200</div>
                <div class="claude-stat-label">Cohort Size</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="claude-stat">
                <div class="claude-stat-value">&lt; 0.1s</div>
                <div class="claude-stat-label">Inference Latency</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="claude-stat">
                <div class="claude-stat-value">95%</div>
                <div class="claude-stat-label">Confidence Bounds</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="claude-stat">
                <div class="claude-stat-value">100%</div>
                <div class="claude-stat-label">Explainable Logic</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.4rem;'></div>", unsafe_allow_html=True)

    # 3-Step Systematic Pipeline Cards
    st.markdown("<h2>How the System Evaluates Elective Compatibility</h2>", unsafe_allow_html=True)
    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:
        st.markdown(
            f"""
            <div class="claude-card" style="height: 100%;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.6rem; color: {CLAUDE['primary']}; margin-bottom: 0.3rem;">01</div>
                <h4 style="margin: 0 0 0.5rem 0; font-family: 'Inter', sans-serif; font-weight: 600; color: {CLAUDE['ink']};">Foundation Analysis</h4>
                <p style="color: {CLAUDE['body_muted']}; font-size: 0.9rem; line-height: 1.55; margin: 0;">
                    Accepts 6 core disciplines (Calculus, Statistics, Programming, English, Physics, Economics) with automated validation and missing-grade imputation.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with step_col2:
        st.markdown(
            f"""
            <div class="claude-card" style="height: 100%;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.6rem; color: {CLAUDE['primary']}; margin-bottom: 0.3rem;">02</div>
                <h4 style="margin: 0 0 0.5rem 0; font-family: 'Inter', sans-serif; font-weight: 600; color: {CLAUDE['ink']};">Latent Dimensionality</h4>
                <p style="color: {CLAUDE['body_muted']}; font-size: 0.9rem; line-height: 1.55; margin: 0;">
                    Standardizes grades and projects onto 3 principal components (Aptitude, Verbal vs Math, Applied Tech) to identify your cohort archetype.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with step_col3:
        st.markdown(
            f"""
            <div class="claude-card" style="height: 100%;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.6rem; color: {CLAUDE['primary']}; margin-bottom: 0.3rem;">03</div>
                <h4 style="margin: 0 0 0.5rem 0; font-family: 'Inter', sans-serif; font-weight: 600; color: {CLAUDE['ink']};">Explainable Ranking</h4>
                <p style="color: {CLAUDE['body_muted']}; font-size: 0.9rem; line-height: 1.55; margin: 0;">
                    Ranks courses by projected grade aptitude and peer cohort outcomes, complete with 95% uncertainty intervals and prerequisite verification.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Primary Action CTAs
    btn_col1, btn_col2, _ = st.columns([1.3, 1.3, 2.0])
    with btn_col1:
        if st.button("✦ Get Recommendations", type="primary", use_container_width=True):
            navigate_to("Get Recommendations")
    with btn_col2:
        if st.button("📊 Explore Cohort Data", type="secondary", use_container_width=True):
            navigate_to("Explore Data")

    # Dark Product Surface Card: Architectural Contract Preview
    st.markdown(
        f"""
        <div class="claude-dark-card" style="margin-top: 1.8rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.7rem; border-bottom: 1px solid #252320; padding-bottom: 0.5rem;">
                <span style="color: {CLAUDE['on_dark_soft']}; font-size: 0.78rem; letter-spacing: 0.05em;">SYSTEM PIPELINE CONTRACT • PURE PYTHON ENGINE</span>
                <span style="color: {CLAUDE['primary']}; font-size: 0.78rem;">● OFFLINE ARTIFACT INFERENCE</span>
            </div>
            <pre style="margin: 0; color: {CLAUDE['on_dark']}; font-size: 0.84rem; line-height: 1.6; background: transparent; padding: 0;"><code># Fast, decoupled recommendation pipeline (zero UI imports)
from src.recommend import recommend

rec = recommend(grades={{"Calculus": 92.0, "Statistics": 94.0, "Programming": 85.0, ...}})
# Output:
# • Archetype: "Quantitative Specialist" (Cluster 1)
# • Top-1: Econometrics — Pred: 85.4, 95% CI: [78.2, 92.6]
# • Driver: Calculus (+4.2 pts), Statistics (+3.1 pts)</code></pre>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Privacy Assurance Callout
    st.markdown(
        """
        <div class="claude-callout">
            <strong>Data Privacy & Governance:</strong> User-entered grades are computed exclusively in volatile server RAM and discarded immediately upon session termination. No persistent databases, telemetry trackers, or profiling logs are retained.
        </div>
        """,
        unsafe_allow_html=True
    )
