"""
Main Recommendation View for Course Recommendation System.
Engineered for executive clarity, professional academic advising, and frictionless UX:
- 1-Click preset archetype selectors for instant demo
- Clean 3-column input grid with real-time Core Average metric
- Interactive profile breakdown with high-contrast radar chart
- Top-3 recommendation cards with visual grade meters, 95% CI bounds, and feature drivers
- Executive catalog table with column progress bars and sorting
- Clean prerequisite gating and CSV export
"""

import io
import textwrap
import numpy as np
import pandas as pd
import streamlit as st

from app.utils import (
    create_contribution_bar,
    create_interval_bar_chart,
    create_radar_chart,
    get_eda_summary,
    get_sample_profiles,
    inject_custom_css,
    CLAUDE,
    PLOTLY_CONFIG
)
from src.recommend import CORE_SUBJECTS, recommend, validate_grades


def render_recommend():
    """Render the optimized professional recommendation interface."""
    inject_custom_css()
    eda_summary = get_eda_summary()
    sample_profiles = get_sample_profiles()

    st.markdown(
        """
        <div style="margin-bottom: 1.4rem;">
            <h1 style="margin: 0; font-family: 'Newsreader', Georgia, serif; font-size: 2.25rem; color: #141413;">Course Recommendations</h1>
            <p style="color: #5e5c56; margin-top: 4px; font-size: 1rem; line-height: 1.5;">
                Enter foundation grades to generate personalized elective compatibility scores, 95% uncertainty intervals, and explainable feature contributions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state keys
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
        # Pre-compute initial recommendation on load for instant professional presentation
        st.session_state.recommendation_result = recommend(st.session_state.input_grades, top_n=3, w1=0.70)

    # 1-Click Preset Archetype Selector Band
    st.markdown("<span style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #8e8b82; letter-spacing: 0.06em;'>Quick Demo Profiles (1-Click Load)</span>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4 = st.columns([1.2, 1.2, 1.2, 0.8])

    with p_col1:
        if st.button("🔬 Quantitative Specialist", use_container_width=True, help="High math, statistics, and physics"):
            st.session_state.input_grades = dict(sample_profiles["Quantitative Thinker"])
            st.session_state.recommendation_result = recommend(st.session_state.input_grades, top_n=3, w1=0.70)
            st.rerun()

    with p_col2:
        if st.button("💻 Applied Tech & Computing", use_container_width=True, help="Peak programming and algorithms"):
            st.session_state.input_grades = dict(sample_profiles["Applied Tech & Computational"])
            st.session_state.recommendation_result = recommend(st.session_state.input_grades, top_n=3, w1=0.70)
            st.rerun()

    with p_col3:
        if st.button("⚖️ Socio-Economic Scholar", use_container_width=True, help="High economics, writing, and empirical analysis"):
            st.session_state.input_grades = dict(sample_profiles["Balanced Socio-Economic Scholar"])
            st.session_state.recommendation_result = recommend(st.session_state.input_grades, top_n=3, w1=0.70)
            st.rerun()

    with p_col4:
        if st.button("↺ Reset (75)", use_container_width=True, help="Reset all subjects to cohort average (75)"):
            st.session_state.input_grades = {s: 75.0 for s in CORE_SUBJECTS}
            st.session_state.recommendation_result = recommend(st.session_state.input_grades, top_n=3, w1=0.70)
            st.rerun()

    # Expandable CSV Upload
    with st.expander("📁 Or Upload Grades via CSV / Batch Template"):
        st.write("Upload a CSV with columns: `Calculus, Statistics, Programming, English, Physics, Economics`")
        u_col1, u_col2 = st.columns([2, 1])
        with u_col1:
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    missing_cols = [c for c in CORE_SUBJECTS if c not in uploaded_df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        first_row = uploaded_df.iloc[0].to_dict()
                        for subj in CORE_SUBJECTS:
                            val = first_row.get(subj)
                            st.session_state.input_grades[subj] = float(val) if pd.notna(val) else None
                        st.session_state.recommendation_result = recommend(st.session_state.input_grades, top_n=3, w1=0.70)
                        st.success("Successfully loaded and generated recommendations from CSV!")
                        st.rerun()
                except Exception as e:
                    st.error(f"CSV Parse Error: {str(e)}")
        with u_col2:
            sample_df = pd.DataFrame([sample_profiles["Quantitative Thinker"]])
            csv_buffer = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Template",
                data=csv_buffer,
                file_name="grades_template.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Core Grades Input Card Container
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background: {CLAUDE['surface_card']}; border: 1px solid {CLAUDE['hairline']}; border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1.2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; border-bottom: 1px solid {CLAUDE['hairline']}; padding-bottom: 0.5rem;">
                <span style="font-weight: 600; color: {CLAUDE['ink']}; font-size: 0.95rem;">Foundation Subject Grades</span>
                <span style="font-size: 0.8rem; color: {CLAUDE['body_muted']};">Valid scale: 0.0 – 100.0 (blank allows up to 2 for imputation)</span>
            </div>
        """,
        unsafe_allow_html=True
    )

    curr_grades = st.session_state.input_grades
    col1, col2, col3 = st.columns(3)

    with col1:
        calc_val = st.number_input(
            "Calculus", min_value=0.0, max_value=100.0,
            value=float(curr_grades.get("Calculus", 85.0)) if curr_grades.get("Calculus") is not None else None,
            step=0.5, help="Differential & integral calculus"
        )
        stats_val = st.number_input(
            "Statistics", min_value=0.0, max_value=100.0,
            value=float(curr_grades.get("Statistics", 88.0)) if curr_grades.get("Statistics") is not None else None,
            step=0.5, help="Probability theory & statistical inference"
        )

    with col2:
        prog_val = st.number_input(
            "Programming", min_value=0.0, max_value=100.0,
            value=float(curr_grades.get("Programming", 92.0)) if curr_grades.get("Programming") is not None else None,
            step=0.5, help="Algorithms & object-oriented programming"
        )
        engl_val = st.number_input(
            "English", min_value=0.0, max_value=100.0,
            value=float(curr_grades.get("English", 76.0)) if curr_grades.get("English") is not None else None,
            step=0.5, help="Academic discourse & technical writing"
        )

    with col3:
        phys_val = st.number_input(
            "Physics", min_value=0.0, max_value=100.0,
            value=float(curr_grades.get("Physics", 84.0)) if curr_grades.get("Physics") is not None else None,
            step=0.5, help="Classical mechanics & physical modeling"
        )
        econ_val = st.number_input(
            "Economics", min_value=0.0, max_value=100.0,
            value=float(curr_grades.get("Economics", 78.0)) if curr_grades.get("Economics") is not None else None,
            step=0.5, help="Micro & macroeconomic analysis"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    input_dict = {
        "Calculus": calc_val,
        "Statistics": stats_val,
        "Programming": prog_val,
        "English": engl_val,
        "Physics": phys_val,
        "Economics": econ_val
    }
    st.session_state.input_grades = input_dict

    # Core GPA calculation
    valid_nums = [v for v in input_dict.values() if v is not None]
    core_gpa = np.mean(valid_nums) if valid_nums else 0.0

    action_c1, action_c2, action_c3 = st.columns([1.8, 1.2, 2.0])
    with action_c1:
        submit_btn = st.button("✦ Recalculate Recommendations", type="primary", use_container_width=True)
    with action_c2:
        st.markdown(
            f"""
            <div style="background: {CLAUDE['surface_soft']}; border: 1px solid {CLAUDE['hairline']}; border-radius: 8px; padding: 6px 12px; text-align: center; height: 38px; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 0.8rem; color: {CLAUDE['body_muted']};">Foundation Avg: <strong style="color: {CLAUDE['ink']}; font-size: 0.95rem;">{core_gpa:.1f}</strong></span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if submit_btn:
        with st.spinner("Analyzing profile & computing uncertainty bounds..."):
            st.session_state.recommendation_result = recommend(input_dict, top_n=3, w1=0.70)

    # Render Recommendation Results
    rec = st.session_state.recommendation_result

    if rec is not None:
        if not rec.is_valid:
            for err in rec.validation_result.errors:
                st.error(f"❌ {err}")
            return

        for w in rec.warnings:
            st.warning(f"⚠️ {w}")

        st.markdown("<hr style='border: 0; border-top: 1px solid #e6dfd8; margin: 1.8rem 0;'>", unsafe_allow_html=True)

        # 1. Profile Archetype & Radar Section
        profile = rec.profile
        st.markdown("<h2>1. Academic Trajectory & Trait Archetype</h2>", unsafe_allow_html=True)

        prof_c1, prof_c2 = st.columns([1.1, 1.4])
        with prof_c1:
            st.markdown(
                f"""
                <div class="claude-card" style="height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span class="badge-pill badge-teal">Cluster {profile.cluster_id + 1}</span>
                        <span style="font-size: 0.8rem; color: {CLAUDE['body_muted']}; font-weight: 500;">PCA Projections</span>
                    </div>
                    <h3 style="margin: 0.2rem 0 0.5rem 0; font-family: 'Newsreader', serif; font-size: 1.7rem; color: {CLAUDE['ink']};">{profile.cluster_name}</h3>
                    <p style="color: {CLAUDE['body']}; font-size: 0.93rem; line-height: 1.6; margin-bottom: 1rem;">
                        {profile.cluster_description}
                    </p>
                    <div style="border-top: 1px solid {CLAUDE['hairline']}; padding-top: 0.8rem;">
                        <div style="font-size: 0.76rem; color: {CLAUDE['body_muted']}; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 6px;">Latent Academic Coordinates</div>
                        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                            <span class="badge-pill badge-cream">PC1 (General): <strong>{profile.pca_scores[0]:+.2f}</strong></span>
                            <span class="badge-pill badge-cream">PC2 (Verbal vs Math): <strong>{profile.pca_scores[1]:+.2f}</strong></span>
                            <span class="badge-pill badge-cream">PC3 (Applied Tech): <strong>{profile.pca_scores[2]:+.2f}</strong></span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with prof_c2:
            cohort_means = {s: eda_summary["distributions"][s]["mean"] for s in CORE_SUBJECTS}
            radar_fig = create_radar_chart(profile.raw_grades, profile.cluster_centers_raw, cohort_means)
            st.plotly_chart(radar_fig, use_container_width=True, config=PLOTLY_CONFIG)

        # 2. Top-3 Recommendations Section
        st.markdown("<h2>2. Top-3 Recommended Electives</h2>", unsafe_allow_html=True)
        st.caption("Ranked by combined predicted aptitude (70%) and peer cohort success (30%).")

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

            # Visual progress bar width (normalized 40-100 to 0-100%)
            progress_pct = max(0, min(100, (pred - 40) / 60 * 100))

            card_html = textwrap.dedent(f"""
            <div class="claude-card" style="padding: 1.4rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center;">
                        <div class="rank-circle-coral">#{rank_num}</div>
                        <div>
                            <h3 style="margin: 0; font-family: 'Newsreader', serif; font-size: 1.4rem; color: {CLAUDE['ink']}; font-weight: 500;">{name}</h3>
                            <span class="badge-pill badge-cream" style="margin-top: 4px;">{category}</span>
                        </div>
                    </div>
                    <div style="text-align: right; margin-top: 2px;">
                        <div style="display: flex; align-items: baseline; justify-content: flex-end; gap: 4px;">
                            <span style="font-family: 'Newsreader', serif; font-size: 1.85rem; font-weight: 500; color: {CLAUDE['ink']}; line-height: 1;">{pred:.1f}</span>
                            <span style="font-size: 0.85rem; color: {CLAUDE['body_muted']};">/ 100</span>
                        </div>
                        <div style="margin-top: 4px;">
                            <span class="badge-pill badge-teal" style="margin: 0;">95% CI: [{lo:.1f}, {hi:.1f}]</span>
                        </div>
                    </div>
                </div>
                <div style="margin: 0.8rem 0 0.5rem 0;">
                    <div style="height: 6px; background-color: {CLAUDE['hairline']}; border-radius: 9999px; overflow: hidden;">
                        <div style="height: 100%; width: {progress_pct}%; background-color: {CLAUDE['primary']}; border-radius: 9999px;"></div>
                    </div>
                </div>
                <p style="color: {CLAUDE['body']}; font-size: 0.91rem; line-height: 1.55; margin: 0.6rem 0 0.6rem 0;">{desc}</p>
                <div style="display: flex; gap: 14px; font-size: 0.82rem; color: {CLAUDE['body_muted']}; background: {CLAUDE['surface_soft']}; padding: 7px 12px; border-radius: 8px; border: 1px solid {CLAUDE['hairline']}; flex-wrap: wrap;">
                    <div>Rank Score: <strong style="color: {CLAUDE['ink']};">{combined:.1f}</strong></div>
                    <div>Peer Average (15 Nearest): <strong style="color: {CLAUDE['ink']};">{sim_avg:.1f}</strong></div>
                    <div>Model Precision (SD): <strong style="color: {CLAUDE['ink']};">±{row['residual_sd']:.1f} pts</strong></div>
                </div>
            </div>
            """).strip()
            st.markdown(card_html, unsafe_allow_html=True)

            with st.expander(f"✦ Explainability Analysis: Why {name} was recommended"):
                c_list = rec.explanations.get(elec_id, [])
                st.write("**Key Contributing Foundation Drivers:**")
                for c in c_list:
                    icon = "▲" if c.direction == "positive" else "▼"
                    color_style = CLAUDE['primary'] if c.direction == "positive" else CLAUDE['muted_soft']
                    st.markdown(f"- <span style='color: {color_style}; font-weight: 700;'>{icon}</span> **{c.subject}:** {c.plain_text} *(Standardized Weight: `{c.coefficient:+.2f}`)*", unsafe_allow_html=True)

                c_fig = create_contribution_bar(c_list)
                st.plotly_chart(c_fig, use_container_width=True, config=PLOTLY_CONFIG)

                st.info(
                    f"**Peer Cohort Evidence:** Among the 15 historical students with academic profiles most similar to yours, the average grade achieved in {name} was **{sim_avg:.1f}**."
                )

        # 3. Prediction Intervals Bar Chart
        st.markdown("<h2>3. Expected Performance Across All Eligible Electives</h2>", unsafe_allow_html=True)
        st.caption("Point estimates with 95% confidence intervals and 80+ High Distinction benchmark.")
        interval_chart = create_interval_bar_chart(rec.all_eligible)
        st.plotly_chart(interval_chart, use_container_width=True, config=PLOTLY_CONFIG)

        # 4. Interactive Full Catalog Dashboard Table
        st.markdown("<h2>4. Comprehensive Elective Comparison & Gating</h2>", unsafe_allow_html=True)

        with st.expander("📋 Full Ranked Catalog of Eligible Electives", expanded=True):
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

            # Modern Streamlit column configuration
            st.dataframe(
                clean_table,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="#%d", width="small"),
                    "Elective Course": st.column_config.TextColumn("Elective Course", width="medium"),
                    "Discipline": st.column_config.TextColumn("Discipline", width="small"),
                    "Predicted Grade": st.column_config.ProgressColumn(
                        "Predicted Grade", min_value=0, max_value=100, format="%.1f"
                    ),
                    "95% Low": st.column_config.NumberColumn("95% Low", format="%.1f"),
                    "95% High": st.column_config.NumberColumn("95% High", format="%.1f"),
                    "Peer Mean": st.column_config.NumberColumn("Peer Mean", format="%.1f"),
                    "Combined Index": st.column_config.NumberColumn("Rank Index", format="%.2f")
                },
                use_container_width=True,
                hide_index=True
            )

            # Clean CSV Download
            csv_data = clean_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Full Recommendations as CSV",
                data=csv_data,
                file_name="course_recommendations.csv",
                mime="text/csv"
            )

        # Blocked Electives with Prerequisites Check
        with st.expander("🚫 Ineligible / Blocked Electives (Prerequisites Unmet)"):
            if not rec.blocked_electives.empty:
                st.warning("The following courses require higher grades in prerequisite foundation courses:")
                blocked_display = rec.blocked_electives[["name", "category", "block_reason"]].rename(
                    columns={"name": "Course", "category": "Category", "block_reason": "Prerequisite Rule"}
                )
                st.dataframe(blocked_display, use_container_width=True, hide_index=True)
            else:
                st.success("All departmental prerequisites are satisfied across every elective in the catalog.")

        # Advisory Disclaimer Callout
        st.markdown(
            """
            <div class="claude-callout">
                <strong>Advisory Consultation Note:</strong> Predictions reflect statistical transfer from foundation coursework. They do not account for intrinsic curiosity, personal interests, or syllabus updates. Always review selections with an academic advisor.
            </div>
            """,
            unsafe_allow_html=True
        )
