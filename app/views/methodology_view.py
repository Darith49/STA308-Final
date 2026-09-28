"""
Methodology and Ethics View for Course Recommendation System.
Redesigned with the Claude Design System:
- Editorial serif typography and literary publication feel
- Pipeline flowchart presented in dark product chrome card (#181715)
- Mathematical derivations in clean expandable panels
- Ethical guarantees, privacy notices, and anti-paternalism principles
- Reproducibility metadata and model lineage
"""

import streamlit as st
from app.utils import get_model_artifacts, inject_custom_css, CLAUDE


def render_methodology():
    """Render the Methodology and Ethics page in Claude aesthetic."""
    inject_custom_css()
    artifacts = get_model_artifacts()
    meta = artifacts.metadata

    st.markdown(
        """
        <div style="margin-bottom: 1.6rem;">
            <h1 style="margin: 0; font-family: 'Newsreader', Georgia, serif; font-weight: 400; font-size: 2.3rem; color: #141413;">Methodology, Architecture & Ethics</h1>
            <p style="color: #6c6a64; margin-top: 6px; font-size: 1.05rem; line-height: 1.5;">
                Formal mathematical formulation, statistical assumptions, ethical guidelines, and system architecture.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. System Pipeline Architecture in Dark Product Chrome Card
    st.markdown("<h2>1. End-to-End System Pipeline</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="claude-dark-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; border-bottom: 1px solid #252320; padding-bottom: 0.6rem;">
                <span style="color: {CLAUDE['on_dark_soft']}; font-size: 0.8rem;">SYSTEM PIPELINE FLOW • REPRODUCIBLE REASONING</span>
                <span style="color: {CLAUDE['primary']}; font-size: 0.8rem;">● OFFLINE ARTIFACT CONTRACT</span>
            </div>
            <pre style="margin: 0; color: {CLAUDE['on_dark']}; font-size: 0.82rem; line-height: 1.55; background: transparent; padding: 0;"><code>┌─────────────────────────────────────────────────────────────────────────────┐
│                               STUDENT INPUT                                 │
│   Core Grades: Calculus, Statistics, Programming, English, Physics, Econ    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION & IMPUTATION                             │
│  • Range Check [0, 100]       • Max 2 Missing: Correlated Ridge Imputer     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ACADEMIC PROFILING PIPELINE                         │
│  • StandardScaler: z = (x - μ) / σ                                          │
│  • PCA: Project to PC1, PC2, PC3                                            │
│  • K-Means: Assign academic archetype cluster                               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│       REGRESSION INFERENCE    │             │      COHORT SIMILARITY        │
│  • Ridge Regression per course│             │  • 15-Nearest Neighbors in 6D │
│  • 95% Interval: y ± 1.96 * σ │             │  • Historical peer elective   │
│  • Prerequisite verification  │             │    grade distribution         │
└───────────────┬───────────────┘             └───────────────┬───────────────┘
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION & EXPLANATION                           │
│  • Combined Index = 0.70 * y_pred + 0.30 * y_peers                          │
│  • Filter to eligible courses (prerequisites satisfied)                     │
│  • Rank Top-3 picks                                                         │
│  • Explainability: Contribution = β_j * z_j                                 │
└─────────────────────────────────────────────────────────────────────────────┘</code></pre>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid #e6dfd8; margin: 1.8rem 0;'>", unsafe_allow_html=True)

    # 2. Mathematical Formulations
    st.markdown("<h2>2. Mathematical Formulations</h2>", unsafe_allow_html=True)

    with st.expander("A. Input Standardization & Preprocessing", expanded=True):
        st.latex(r"z_j = \frac{x_j - \mu_j}{\sigma_j}, \quad j \in \{\text{Calculus, Statistics, Programming, English, Physics, Economics}\}")
        st.write(
            "Where $\\mu_j$ and $\\sigma_j$ are computed strictly from the training cohort. "
            "If 1 or 2 core grades are missing, an offline multivariate ridge regression predicts the missing value from available subjects."
        )

    with st.expander("B. Prediction & Uncertainty Bounds"):
        st.latex(r"\hat{y}_k = \beta_{0, k} + \sum_{j=1}^6 \beta_{j, k} z_j")
        st.latex(r"\text{CI}_{95\%}(\hat{y}_k) = \left[ \hat{y}_k - 1.96 \cdot \hat{\sigma}_{\epsilon, k}, \; \hat{y}_k + 1.96 \cdot \hat{\sigma}_{\epsilon, k} \right]")
        st.write(
            "Where $\\hat{\\sigma}_{\\epsilon, k}$ is the out-of-fold cross-validated residual standard deviation for elective $k$. "
            "Interval bounds are clamped to the valid grade domain $[0, 100]$."
        )

    with st.expander("C. Combined Ranking Index & Feature Attribution"):
        st.latex(r"\text{Score}_k = w_1 \cdot \hat{y}_k + (1 - w_1) \cdot \bar{y}_{k, \text{peers}}, \quad w_1 = 0.70")
        st.latex(r"\text{Contribution}_{j, k} = \beta_{j, k} \cdot z_j")
        st.write(
            "Local feature attributions compute the exact point deviation attributable to each core subject grade. "
            "The top three positive or negative drivers are translated directly into plain-language guidance."
        )

    st.markdown("<hr style='border: 0; border-top: 1px solid #e6dfd8; margin: 1.8rem 0;'>", unsafe_allow_html=True)

    # 3. Limitations & Assumptions
    st.markdown("<h2>3. Limitations & Analytical Assumptions</h2>", unsafe_allow_html=True)
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown(
            f"""
            <div class="claude-card" style="height: 100%;">
                <h4 style="margin: 0 0 0.6rem 0; font-family: 'Newsreader', serif; font-size: 1.25rem; color: {CLAUDE['ink']};">Statistical Assumptions</h4>
                <ul style="color: {CLAUDE['body']}; font-size: 0.9rem; line-height: 1.6; margin: 0; padding-left: 1.1rem;">
                    <li><strong>Linearity in Core Aptitude:</strong> Foundation competencies demonstrate approximately linear transfer to advanced elective subjects.</li>
                    <li><strong>Stationarity:</strong> Grading scales and curriculum rigor are assumed consistent across consecutive academic semesters.</li>
                    <li><strong>Missing at Random (MAR):</strong> Missing 1-2 core grades is assumed random rather than an indication of academic disqualification.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_l2:
        st.markdown(
            f"""
            <div class="claude-card" style="height: 100%;">
                <h4 style="margin: 0 0 0.6rem 0; font-family: 'Newsreader', serif; font-size: 1.25rem; color: {CLAUDE['ink']};">Analytical Limitations</h4>
                <ul style="color: {CLAUDE['body']}; font-size: 0.9rem; line-height: 1.6; margin: 0; padding-left: 1.1rem;">
                    <li><strong>Aptitude Over Intrinsic Interest:</strong> The recommender models expected academic performance; it cannot capture student curiosity or career passion.</li>
                    <li><strong>Minimum Data Requirements:</strong> At least 4 of 6 core grades must be entered for dependable inference.</li>
                    <li><strong>Cohort Context:</strong> Calibrated against a structured benchmark cohort of 1,200 students reflecting university distributions.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border: 0; border-top: 1px solid #e6dfd8; margin: 1.8rem 0;'>", unsafe_allow_html=True)

    # 4. Ethics, Privacy & Anti-Paternalism
    st.markdown("<h2>4. Ethics, Privacy & Anti-Paternalism</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="claude-callout" style="border-left-color: {CLAUDE['primary']}; background-color: {CLAUDE['surface_soft']};">
            <h4 style="margin: 0 0 0.6rem 0; font-family: 'Newsreader', serif; font-size: 1.25rem; color: {CLAUDE['ink']};">Guiding Principles in AI Advising</h4>
            <p style="color: {CLAUDE['body']}; font-size: 0.92rem; line-height: 1.6; margin: 0 0 0.6rem 0;">
                • <strong>Strict Data Ephemerality:</strong> User-entered grades are processed exclusively in volatile RAM and discarded upon session close. No persistent databases, profiling cookies, or tracking logs exist.<br>
                • <strong>Anti-Paternalism (Support, Not Gatekeeping):</strong> The system is strictly advisory. It never enforces course selection or restricts student liberty.<br>
                • <strong>Explainability & Uncertainty:</strong> Every recommendation provides 95% confidence intervals and feature contribution breakdowns, removing black-box mystique.<br>
                • <strong>Algorithmic Fairness:</strong> Audits confirm error parity across academic specializations without bias against humanities or STEM disciplines.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid #e6dfd8; margin: 1.8rem 0;'>", unsafe_allow_html=True)

    # 5. Metadata & Model Lineage
    st.markdown("<h2>5. Model Lineage & Metadata</h2>", unsafe_allow_html=True)
    m_c1, m_c2 = st.columns(2)
    with m_c1:
        st.markdown(f"**Project:** {meta.get('project')}")
        st.markdown(f"**Release Version:** `v{meta.get('version')}`")
        st.markdown(f"**Training Timestamp:** `{meta.get('training_date')}`")
        st.markdown(f"**Random Seed:** `{meta.get('random_seed')}`")
    with m_c2:
        st.markdown(f"**Primary Architecture:** `{meta.get('primary_model')}`")
        st.markdown(f"**Training Cohort:** `{meta.get('n_students')} student records`")
        st.markdown(f"**Core Stack:** Python {meta.get('frameworks', {}).get('python')}, Scikit-Learn {meta.get('frameworks', {}).get('scikit_learn')}")
