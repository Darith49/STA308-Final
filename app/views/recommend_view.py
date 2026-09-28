"""
Main Recommendation View for Course Recommendation System.
Section 5.3 of the Project Plan:
- 3-column input grid with tooltips and range validation
- Direct input, CSV upload, and preset sample profiles
- Profile card with interactive radar chart (Student vs Cluster vs Cohort)
- Top-3 cards with uncertainty intervals, plain-language explanations, and similar-student scores
- Prediction interval bar chart for all eligible electives
- Full ranked table, prerequisite-blocked electives, and imputed values notices
- CSV download and Start Over functionality
"""

import io
import pandas as pd
import streamlit as st

from app.utils import (
    create_contribution_bar,
    create_interval_bar_chart,
    create_radar_chart,
    get_eda_summary,
    get_sample_profiles,
    inject_custom_css,
    OKABE_ITO,
)
from src.recommend import CORE_SUBJECTS, recommend, validate_grades


def render_recommend():
    """Render the Get Recommendations page."""
    inject_custom_css()
    eda_summary = get_eda_summary()
    sample_profiles = get_sample_profiles()

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-weight: 800; font-size: 2rem; color: #0F172A;">Get Course Recommendations</h1>
            <p style="color: #64748B; margin-top: 4px; font-size: 1rem;">
                Enter your foundation subject grades to receive personalized elective rankings, uncertainty bounds, and explainable insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state keys for grades if not present
    if "input_grades" not in st.session_state:
        st.session_state.input_grades = {
            "Calculus": 85.0,
            "Statistics": 88.0,
            "Programming": 92.0,
            "English": 76.0,
            "Physics": 84.0,
            "Economics": 78.0
        }
    if "recommendation_result" not in st.session_state:
        st.session_state.recommendation_result = None

    # Input method selection tabs
    input_mode = st.radio(
        "Select input method:",
        options=["Type Grades", "Load Preset Student Profile", "Upload CSV File"],
        horizontal=True
    )

    # Handle preset profiles
    if input_mode == "Load Preset Student Profile":
        preset_choice = st.selectbox(
            "Choose a demo student archetype:",
            options=list(sample_profiles.keys()),
            help="Pre-configured student grade profiles representing distinct academic trajectories."
        )
        if st.button("Apply Preset Profile", type="secondary"):
            st.session_state.input_grades = dict(sample_profiles[preset_choice])
            st.success(f"Applied profile: {preset_choice}")
            st.rerun()

    # Handle CSV Upload
    elif input_mode == "Upload CSV File":
        st.info("Upload a CSV file containing core grades columns: Calculus, Statistics, Programming, English, Physics, Economics.")
        
        # Download template button
        sample_df = pd.DataFrame([sample_profiles["Quantitative Thinker"]])
        csv_buffer = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_buffer,
            file_name="student_grades_template.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Upload student CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                missing_cols = [c for c in CORE_SUBJECTS if c not in uploaded_df.columns]
                if missing_cols:
                    st.error(f"CSV is missing required core subject columns: {', '.join(missing_cols)}")
                else:
                    first_row = uploaded_df.iloc[0].to_dict()
                    for subj in CORE_SUBJECTS:
                        val = first_row.get(subj)
                        st.session_state.input_grades[subj] = float(val) if pd.notna(val) else None
                    st.success("Successfully loaded grades from CSV!")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    # 3-Column Input Form
    st.markdown("<h4 style='margin-top: 1rem; margin-bottom: 0.5rem;'>Foundation Subject Grades (0 - 100)</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    curr_grades = st.session_state.input_grades

    with col1:
        calc_val = st.number_input(
            "Calculus",
            min_value=0.0,
            max_value=100.0,
            value=float(curr_grades.get("Calculus", 85.0)) if curr_grades.get("Calculus") is not None else None,
            step=0.5,
            help="Single-variable and multi-variable differential & integral calculus."
        )
        stats_val = st.number_input(
            "Statistics",
            min_value=0.0,
            max_value=100.0,
            value=float(curr_grades.get("Statistics", 88.0)) if curr_grades.get("Statistics") is not None else None,
            step=0.5,
            help="Probability theory, estimation, hypothesis testing, and regression foundations."
        )

    with col2:
        prog_val = st.number_input(
            "Programming",
            min_value=0.0,
            max_value=100.0,
            value=float(curr_grades.get("Programming", 92.0)) if curr_grades.get("Programming") is not None else None,
            step=0.5,
            help="Data structures, procedural algorithms, and object-oriented programming."
        )
        engl_val = st.number_input(
            "English",
            min_value=0.0,
            max_value=100.0,
            value=float(curr_grades.get("English", 76.0)) if curr_grades.get("English") is not None else None,
            step=0.5,
            help="Academic research writing, critical discourse, and structured composition."
        )

    with col3:
        phys_val = st.number_input(
            "Physics",
            min_value=0.0,
            max_value=100.0,
            value=float(curr_grades.get("Physics", 84.0)) if curr_grades.get("Physics") is not None else None,
            step=0.5,
            help="Classical mechanics, electromagnetism, and mathematical modeling of physical systems."
        )
        econ_val = st.number_input(
            "Economics",
            min_value=0.0,
            max_value=100.0,
            value=float(curr_grades.get("Economics", 78.0)) if curr_grades.get("Economics") is not None else None,
            step=0.5,
            help="Microeconomic behavior, macroeconomic systems, and quantitative incentives."
        )

    # Update session state with current inputs
    input_dict = {
        "Calculus": calc_val,
        "Statistics": stats_val,
        "Programming": prog_val,
        "English": engl_val,
        "Physics": phys_val,
        "Economics": econ_val
    }
    st.session_state.input_grades = input_dict

    # Action Buttons: Submit & Reset
    btn_c1, btn_c2, _ = st.columns([1.5, 1.0, 3.5])
    with btn_c1:
        submit_btn = st.button("🚀 Generate Recommendations", type="primary", use_container_width=True)
    with btn_c2:
        if st.button("↺ Start Over", type="secondary", use_container_width=True):
            st.session_state.input_grades = {s: 75.0 for s in CORE_SUBJECTS}
            st.session_state.recommendation_result = None
            st.rerun()

    if submit_btn:
        with st.spinner("Analyzing academic profile and computing recommendation matrix..."):
            rec_result = recommend(input_dict, top_n=3, w1=0.70)
            st.session_state.recommendation_result = rec_result

    # Display Results if Available
    rec = st.session_state.recommendation_result

    if rec is not None:
        if not rec.is_valid:
            for err in rec.validation_result.errors:
                st.error(f"❌ {err}")
            return

        # Display warning banners (imputed values, identical grades, etc.)
        for w in rec.warnings:
            st.warning(f"⚠️ {w}")

        st.markdown("<hr style='margin: 1.8rem 0;'>", unsafe_allow_html=True)

        # 1. Profile Section
        profile = rec.profile
        st.subheader("1. Your Academic Profile Archetype")
        
        prof_c1, prof_c2 = st.columns([1.2, 1.8])
        with prof_c1:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.4rem; height: 100%;">
                    <span class="badge-pill badge-cluster">Cluster {profile.cluster_id + 1}</span>
                    <h3 style="margin: 0.5rem 0; color: #0F172A; font-weight: 800;">{profile.cluster_name}</h3>
                    <p style="color: #475569; font-size: 0.95rem; line-height: 1.5;">{profile.cluster_description}</p>
                    <div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid #F1F5F9;">
                        <span style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Latent Dimensions</span>
                        <div style="display: flex; gap: 8px; margin-top: 6px;">
                            <span class="badge-pill badge-source">PC1: {profile.pca_scores[0]:+.2f}</span>
                            <span class="badge-pill badge-source">PC2: {profile.pca_scores[1]:+.2f}</span>
                            <span class="badge-pill badge-source">PC3: {profile.pca_scores[2]:+.2f}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with prof_c2:
            cohort_means = {s: eda_summary["distributions"][s]["mean"] for s in CORE_SUBJECTS}
            radar_fig = create_radar_chart(profile.raw_grades, profile.cluster_centers_raw, cohort_means)
            st.plotly_chart(radar_fig, use_container_width=True)

        # 2. Top-3 Recommendations Cards
        st.subheader("2. Top-3 Recommended Electives")
        st.caption("Ranked by combined predicted grade aptitude (70%) and similar-student historical success (30%).")

        top_df = rec.top_electives

        for idx, row in top_df.iterrows():
            rank_num = idx + 1
            elec_id = row["elective_id"]
            name = row["name"]
            category = row["category"]
            pred = row["pred_grade"]
            lo = row["interval_lo"]
            hi = row["interval_hi"]
            combined = row["combined_score"]
            sim_avg = row["similar_avg"]
            desc = row["description"]

            # Card container
            card_html = f"""
            <div class="recommendation-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center;">
                        <div class="rank-circle">#{rank_num}</div>
                        <div>
                            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #0F172A;">{name}</h3>
                            <span class="badge-pill badge-category" style="margin-top: 4px;">{category}</span>
                        </div>
                    </div>
                    <div style="text-align: right; margin-top: 4px;">
                        <span style="font-size: 1.45rem; font-weight: 800; color: #1E40AF;">{pred:.1f}</span>
                        <span style="font-size: 0.88rem; color: #64748B;">/ 100</span>
                        <div style="font-size: 0.8rem; font-weight: 600; color: #059669;">
                            95% CI: [{lo:.1f}, {hi:.1f}]
                        </div>
                    </div>
                </div>
                <p style="color: #475569; font-size: 0.9rem; margin: 0.8rem 0 0.4rem 0;">{desc}</p>
                <div style="display: flex; gap: 15px; font-size: 0.82rem; color: #64748B; background: #F8FAFC; padding: 8px 12px; border-radius: 8px;">
                    <div>🎯 <strong>Rank Index:</strong> {combined:.1f}</div>
                    <div>👥 <strong>Similar Students Avg:</strong> {sim_avg:.1f}</div>
                    <div>📊 <strong>Residual SD:</strong> ±{row['residual_sd']:.1f}</div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Expandable "Why was this recommended?"
            with st.expander(f"🔍 Why was {name} recommended for you?"):
                c_list = rec.explanations.get(elec_id, [])
                st.write("**Key Contributing Factors:**")
                for c in c_list:
                    icon = "📈" if c.direction == "positive" else "📉"
                    st.markdown(f"- {icon} **{c.subject}:** {c.plain_text} (Weight: {c.coefficient:+.2f})")

                # Bar chart of contributions
                c_fig = create_contribution_bar(c_list)
                st.plotly_chart(c_fig, use_container_width=True)

                st.info(
                    f"👥 **Cohort Benchmark Insight:** Among the 15 students in the historical cohort with academic profiles most similar to yours, the average grade achieved in {name} was **{sim_avg:.1f}**."
                )

        # 3. Prediction Intervals Bar Chart
        st.subheader("3. Expected Performance Across All Eligible Electives")
        st.caption("Horizontal bars represent predicted grade with 95% confidence intervals.")
        interval_chart = create_interval_bar_chart(rec.all_eligible)
        st.plotly_chart(interval_chart, use_container_width=True)

        # 4. Expanders: Full Table, Blocked Electives, Download
        st.subheader("4. Detailed Breakdown & Prerequisite Gating")

        with st.expander("📋 Full Ranked Table of Eligible Electives"):
            display_cols = [
                "rank", "name", "category", "pred_grade", "interval_lo", "interval_hi", "similar_avg", "combined_score"
            ]
            renamed_cols = {
                "rank": "Rank",
                "name": "Elective Course",
                "category": "Discipline",
                "pred_grade": "Predicted Grade",
                "interval_lo": "95% Low",
                "interval_hi": "95% High",
                "similar_avg": "Peer Mean",
                "combined_score": "Combined Index"
            }
            clean_table = rec.all_eligible[display_cols].rename(columns=renamed_cols)
            st.dataframe(clean_table, use_container_width=True, hide_index=True)

            # Download CSV Button
            csv_data = clean_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Recommendations as CSV",
                data=csv_data,
                file_name="course_recommendations.csv",
                mime="text/csv"
            )

        with st.expander("🚫 Ineligible / Blocked Electives (Prerequisites Unmet)"):
            if not rec.blocked_electives.empty:
                st.warning("The following electives require higher grades in prerequisite foundation subjects:")
                blocked_display = rec.blocked_electives[["name", "category", "block_reason"]].rename(
                    columns={"name": "Course", "category": "Category", "block_reason": "Prerequisite Requirement"}
                )
                st.dataframe(blocked_display, use_container_width=True, hide_index=True)
            else:
                st.success("Great job! You satisfy all academic prerequisites across every elective course in the catalog.")

        # Persistent Advisory Disclaimer
        st.markdown(
            """
            <div class="disclaimer-banner">
                <strong>⚠️ Academic Disclaimer:</strong> Predictions and intervals are generated using linear Ridge models trained on historical student performance.
                They reflect expected academic fit but do not account for intrinsic curiosity, career passions, syllabus updates, or professor changes. Always consult with your academic advisor before finalizing course registration.
            </div>
            """,
            unsafe_allow_html=True
        )
