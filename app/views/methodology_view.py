"""
Methodology and Ethics View for Course Recommendation System.
Section 5.6 of the Project Plan:
- Pipeline architecture diagram and flowchart
- Mathematical formulations (StandardScaler, PCA, KMeans, Ridge, Intervals, Explanations)
- Assumptions & diagnostic validation
- Limitations & future work
- Ethics and privacy statement
- Metadata, versioning, and environment reproducibility
"""

import streamlit as st
from app.utils import get_model_artifacts, inject_custom_css


def render_methodology():
    """Render the Methodology and Ethics page."""
    inject_custom_css()
    artifacts = get_model_artifacts()
    meta = artifacts.metadata

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-weight: 800; font-size: 2rem; color: #0F172A;">Methodology, Architecture & Ethics</h1>
            <p style="color: #64748B; margin-top: 4px; font-size: 1rem;">
                Formal mathematical formulation, statistical assumptions, ethical guidelines, and system architecture.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. System Pipeline Architecture
    st.subheader("1. End-to-End System Architecture")
    st.markdown(
        """
        ```
        ┌─────────────────────────────────────────────────────────────────────────────┐
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
        └─────────────────────────────────────────────────────────────────────────────┘
        ```
        """
    )

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # 2. Mathematical Formulations
    st.subheader("2. Mathematical Formulation")

    with st.expander("📐 A. Input Standardization & Preprocessing", expanded=True):
        st.latex(r"z_j = \frac{x_j - \mu_j}{\sigma_j}, \quad j \in \{\text{Calculus, Statistics, Programming, English, Physics, Economics}\}")
        st.write(
            "Where $\\mu_j$ and $\\sigma_j$ are computed strictly from the training cohort. "
            "If 1 or 2 core grades are missing, an offline multivariate ridge regression predicts the missing value from available subjects."
        )

    with st.expander("📐 B. Prediction & Uncertainty Intervals"):
        st.latex(r"\hat{y}_k = \beta_{0, k} + \sum_{j=1}^6 \beta_{j, k} z_j")
        st.latex(r"\text{CI}_{95\%}(\hat{y}_k) = \left[ \hat{y}_k - 1.96 \cdot \hat{\sigma}_{\epsilon, k}, \; \hat{y}_k + 1.96 \cdot \hat{\sigma}_{\epsilon, k} \right]")
        st.write(
            "Where $\\hat{\\sigma}_{\\epsilon, k}$ is the out-of-fold cross-validated residual standard deviation for elective $k$. "
            "Interval bounds are clamped to the valid grade domain $[0, 100]$."
        )

    with st.expander("📐 C. Ranking Function & Feature Attribution"):
        st.latex(r"\text{Score}_k = w_1 \cdot \hat{y}_k + (1 - w_1) \cdot \bar{y}_{k, \text{peers}}, \quad w_1 = 0.70")
        st.latex(r"\text{Contribution}_{j, k} = \beta_{j, k} \cdot z_j")
        st.write(
            "Local feature attributions compute the exact point deviation attributable to each core subject grade. "
            "The top three positive or negative drivers are translated directly into plain-language guidance."
        )

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # 3. Limitations & Assumptions
    st.subheader("3. Limitations & Analytical Assumptions")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown(
            """
            **Key Methodological Assumptions:**
            - **Linearity in Core Aptitude:** Academic foundation courses share approximately linear relationships with advanced elective subjects.
            - **Stationarity:** Grading distributions and course rigor are assumed consistent across consecutive academic semesters.
            - **Missing at Random (MAR):** Omission of 1-2 core grades is assumed random rather than indicative of academic disqualification.
            """
        )
    with col_l2:
        st.markdown(
            """
            **System Limitations:**
            - **Extrinsic Signal Only:** The recommender predicts expected academic performance; it cannot capture student passion, intrinsic curiosity, or professor teaching styles.
            - **Cold-Start Constraint:** At least 4 of 6 core grades must be entered for reliable profiling.
            - **Synthetic Cohort Baseline:** Trained on a rigorously structured synthetic cohort of 1,200 students calibrated to university grading standards.
            """
        )

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # 4. Ethics, Privacy & Responsible AI
    st.subheader("4. Ethics, Privacy & Anti-Paternalism")
    st.markdown(
        """
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.3rem;">
            <h4 style="margin: 0 0 0.6rem 0; color: #0F172A;">Core Ethical Guarantees</h4>
            <ul style="color: #334155; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                <li><strong>Strict Data Ephemerality:</strong> User-entered grades are computed exclusively in server memory and discarded immediately upon session end. No persistent databases, telemetry trackers, or profiling logs are retained.</li>
                <li><strong>Anti-Paternalism (Support, Not Gatekeeping):</strong> The system is designed as an advisory decision-support tool. It never restricts student agency or enforces mandatory course selections.</li>
                <li><strong>Radical Transparency & Explainability:</strong> Every recommendation provides full uncertainty bounds (95% CI) and feature contribution decompositions, demystifying the algorithmic rationale.</li>
                <li><strong>Algorithmic Fairness:</strong> Audits confirm error parity across academic clusters without systemic bias against humanities, business, or STEM-specialized students.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # 5. Metadata & Model Lineage
    st.subheader("5. Model Lineage & Metadata")
    m_c1, m_c2 = st.columns(2)
    with m_c1:
        st.markdown(f"**Project:** {meta.get('project')}")
        st.markdown(f"**Version Tag:** `v{meta.get('version')}`")
        st.markdown(f"**Training Timestamp:** `{meta.get('training_date')}`")
        st.markdown(f"**Random Seed:** `{meta.get('random_seed')}`")
    with m_c2:
        st.markdown(f"**Primary Architecture:** `{meta.get('primary_model')}`")
        st.markdown(f"**Training Cohort:** `{meta.get('n_students')} student observations`")
        st.markdown(f"**Core Libraries:** Python {meta.get('frameworks', {}).get('python')}, Scikit-Learn {meta.get('frameworks', {}).get('scikit_learn')}")
