"""
Home Page View for Course Recommendation System.
Section 5.2 of the project plan:
- Title, one-sentence purpose
- 3-step 'how it works' graphic
- Action buttons: 'Get my recommendations', 'See how it works'
- Data source badge and disclaimer
"""

import streamlit as st
from app.utils import inject_custom_css, OKABE_ITO


def render_home(navigate_to):
    """Render the Home landing view."""
    inject_custom_css()

    st.markdown(
        """
        <div class="main-header">
            <span class="badge-pill badge-source">STA308 Final Project</span>
            <span class="badge-pill badge-category">Intelligent Academic Advising</span>
            <span class="badge-pill badge-cluster">Python + Streamlit</span>
            <h1>Course Recommendation System</h1>
            <p>Empowering university students and advisors with instant, explainable, and uncertainty-aware elective recommendations grounded in statistical machine learning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Quick summary metric badges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-value">1,200</div>
                <div class="stat-label">Cohort Size</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-value">&lt; 0.1s</div>
                <div class="stat-label">Inference Latency</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-value">95%</div>
                <div class="stat-label">Prediction Interval</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-value">100%</div>
                <div class="stat-label">Explainable Logic</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 3-Step "How It Works" Section
    st.subheader("How It Works")
    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:
        st.markdown(
            """
            <div class="how-step">
                <div class="how-step-num">1</div>
                <h4 style="margin: 0 0 0.5rem 0; font-weight: 700;">Enter Core Grades</h4>
                <p style="color: #64748B; font-size: 0.9rem; margin: 0;">
                    Input your grades across 6 foundation subjects (Calculus, Statistics, Programming, English, Physics, Economics) or upload a CSV.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with step_col2:
        st.markdown(
            """
            <div class="how-step">
                <div class="how-step-num">2</div>
                <h4 style="margin: 0 0 0.5rem 0; font-weight: 700;">Academic Profiling</h4>
                <p style="color: #64748B; font-size: 0.9rem; margin: 0;">
                    Your grades are standardized, mapped onto latent PCA dimensions, and matched to an academic archetype and cohort peers.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with step_col3:
        st.markdown(
            """
            <div class="how-step">
                <div class="how-step-num">3</div>
                <h4 style="margin: 0 0 0.5rem 0; font-weight: 700;">Explainable Ranks</h4>
                <p style="color: #64748B; font-size: 0.9rem; margin: 0;">
                    Receive your top-3 electives with 95% uncertainty intervals, verified prerequisite checks, and feature contribution reasons.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Call-to-action buttons
    btn_col1, btn_col2, _ = st.columns([1.2, 1.2, 2.0])
    with btn_col1:
        if st.button("🎯 Get My Recommendations", type="primary", use_container_width=True):
            navigate_to("Get Recommendations")
    with btn_col2:
        if st.button("📊 Explore Cohort Data", type="secondary", use_container_width=True):
            navigate_to("Explore Data")

    # Disclaimer and Privacy Notice Banner
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>🔒 Privacy & Data Transparency:</strong> All grades entered are processed strictly in-memory and are never stored, logged, or transmitted.
            Predictions represent statistical estimates derived from a benchmark cohort of 1,200 students and serve as supportive decision guidance rather than academic guarantees.
        </div>
        """,
        unsafe_allow_html=True
    )
