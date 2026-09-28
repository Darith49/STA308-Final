"""
Home Page View for Course Recommendation System.
Redesigned with the Claude Design System:
- Editorial serif hero header with warm coral accents
- Warm cream card surfaces and delicate hairlines
- Stat cards with serif numerals
- Dark product surface card showcasing explainability architecture
- Humanist typography and clear editorial CTAs
"""

import streamlit as st
from app.utils import inject_custom_css, CLAUDE


def render_home(navigate_to):
    """Render the Home landing view in Claude aesthetic."""
    inject_custom_css()

    st.markdown(
        f"""
        <div class="claude-hero">
            <div style="margin-bottom: 0.8rem;">
                <span class="badge-pill badge-coral">STA308 Final Project</span>
                <span class="badge-pill badge-cream">Academic Advising AI</span>
                <span class="badge-pill badge-teal">Ridge + PCA + K-Means</span>
            </div>
            <h1>Course Recommendation System</h1>
            <p>An explainable, uncertainty-aware elective recommendation system grounded in statistical machine learning. Built to assist students and advisors with evidence-backed curriculum decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Stat boxes with serif numerals
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
                <div class="claude-stat-label">Inference Speed</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="claude-stat">
                <div class="claude-stat-value">95%</div>
                <div class="claude-stat-label">Uncertainty Bounds</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="claude-stat">
                <div class="claude-stat-value">100%</div>
                <div class="claude-stat-label">Attribution Clarity</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 3-Step Editorial Workflow
    st.markdown("<h2>How the Recommender Works</h2>", unsafe_allow_html=True)
    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:
        st.markdown(
            f"""
            <div class="claude-card" style="height: 100%;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.6rem; color: {CLAUDE['primary']}; margin-bottom: 0.4rem;">01</div>
                <h4 style="margin: 0 0 0.5rem 0; font-family: 'Inter', sans-serif; font-weight: 600; color: {CLAUDE['ink']};">Enter Core Grades</h4>
                <p style="color: {CLAUDE['muted']}; font-size: 0.9rem; line-height: 1.55; margin: 0;">
                    Input performance in 6 foundation subjects: Calculus, Statistics, Programming, English, Physics, and Economics.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with step_col2:
        st.markdown(
            f"""
            <div class="claude-card" style="height: 100%;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.6rem; color: {CLAUDE['primary']}; margin-bottom: 0.4rem;">02</div>
                <h4 style="margin: 0 0 0.5rem 0; font-family: 'Inter', sans-serif; font-weight: 600; color: {CLAUDE['ink']};">Profile & Benchmark</h4>
                <p style="color: {CLAUDE['muted']}; font-size: 0.9rem; line-height: 1.55; margin: 0;">
                    Grades are standardized and projected onto 3 principal components to assign your academic archetype and match peer cohorts.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with step_col3:
        st.markdown(
            f"""
            <div class="claude-card" style="height: 100%;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.6rem; color: {CLAUDE['primary']}; margin-bottom: 0.4rem;">03</div>
                <h4 style="margin: 0 0 0.5rem 0; font-family: 'Inter', sans-serif; font-weight: 600; color: {CLAUDE['ink']};">Rank & Explain</h4>
                <p style="color: {CLAUDE['muted']}; font-size: 0.9rem; line-height: 1.55; margin: 0;">
                    Receive your top-3 electives with 95% confidence intervals, prerequisite validation, and plain-language driver attributions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Action CTAs
    btn_col1, btn_col2, _ = st.columns([1.3, 1.3, 2.0])
    with btn_col1:
        if st.button("Get My Recommendations", type="primary", use_container_width=True):
            navigate_to("Get Recommendations")
    with btn_col2:
        if st.button("Explore Cohort Data", type="secondary", use_container_width=True):
            navigate_to("Explore Data")

    # Dark Product Surface Card: Architectural Contract Preview
    st.markdown(
        f"""
        <div class="claude-dark-card" style="margin-top: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; border-bottom: 1px solid #252320; padding-bottom: 0.6rem;">
                <span style="color: {CLAUDE['on_dark_soft']}; font-size: 0.8rem;">ARCHITECTURE • SYSTEM CONTRACT</span>
                <span style="color: {CLAUDE['accent_teal']}; font-size: 0.8rem;">● ACTIVE OFFLINE INFERENCE</span>
            </div>
            <pre style="margin: 0; color: {CLAUDE['on_dark']}; font-size: 0.85rem; line-height: 1.6; background: transparent; padding: 0;"><code># Zero-latency pure Python recommendation pipeline
from src.recommend import recommend

result = recommend(grades={{"Calculus": 88, "Statistics": 92, "Programming": 85, ...}})
# => Top-1: Econometrics (Predicted: 85.4, 95% CI: [78.2, 92.6])
# => Attribution: Calculus (+4.2 pts), Statistics (+3.1 pts) raise expected score.</code></pre>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Editorial Callout
    st.markdown(
        """
        <div class="claude-callout">
            <strong>Ethical & Privacy Assurance:</strong> User-entered grades are computed exclusively in-memory and are never stored, logged, or retained. Recommendations provide consultative decision support rather than paternalistic requirements.
        </div>
        """,
        unsafe_allow_html=True
    )
