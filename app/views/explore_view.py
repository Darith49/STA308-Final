"""
Explore Data View for Course Recommendation System.
Section 5.4 of the Project Plan:
- Precomputed from models/eda_summary.json (fast, no runtime training)
- Tabs:
  1. Distributions: Histograms, summary statistics, interactive subject selector
  2. Correlations: Full interactive correlation heatmap with hover annotations
  3. Profiles: Scree plot (PCA variance), PCA loadings table, cluster distribution
  4. Electives: Grade distributions, enrolment breakdown, and course details
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import get_eda_summary, inject_custom_css, OKABE_ITO
from src.recommend import CORE_SUBJECTS


def render_explore():
    """Render the Explore Cohort Data page."""
    inject_custom_css()
    eda = get_eda_summary()

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-weight: 800; font-size: 2rem; color: #0F172A;">Explore Cohort Dataset</h1>
            <p style="color: #64748B; margin-top: 4px; font-size: 1rem;">
                Investigate distributions, correlation structures, principal component loadings, and clustering profiles across 1,200 students.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distributions & Summary",
        "🔥 Correlation Heatmap",
        "🧭 Academic Profiles & PCA",
        "📚 Elective Catalog"
    ])

    # TAB 1: DISTRIBUTIONS
    with tab1:
        st.subheader("Grade Distributions by Subject")
        all_subjects = CORE_SUBJECTS + list(eda["electives_meta"].keys())
        
        selected_subject = st.selectbox(
            "Select a course to inspect distribution:",
            options=all_subjects,
            format_func=lambda x: f"{x} ({eda['electives_meta'][x]['name']})" if x in eda["electives_meta"] else x
        )

        dist_info = eda["distributions"][selected_subject]

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Mean Grade", f"{dist_info['mean']:.1f}")
        with m_col2:
            st.metric("Std Deviation", f"{dist_info['std']:.1f}")
        with m_col3:
            st.metric("Median", f"{dist_info['median']:.1f}")
        with m_col4:
            st.metric("Min Grade", f"{dist_info['min']:.1f}")
        with m_col5:
            st.metric("Max Grade", f"{dist_info['max']:.1f}")

        # Histogram chart
        hist_data = dist_info["histogram"]
        bin_labels = [
            f"{hist_data['bin_edges'][i]:.0f}-{hist_data['bin_edges'][i+1]:.0f}"
            for i in range(len(hist_data['counts']))
        ]
        
        fig_hist = go.Figure(go.Bar(
            x=bin_labels,
            y=hist_data["counts"],
            marker_color=OKABE_ITO["blue"],
            hovertemplate="Score range %{x}: %{y} students<extra></extra>"
        ))
        fig_hist.update_layout(
            title=f"Histogram of Student Performance: {selected_subject}",
            xaxis_title="Grade Bins (0 - 100)",
            yaxis_title="Student Count",
            margin=dict(l=20, r=20, t=40, b=40),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Summary statistics table for all core subjects
        st.markdown("#### Summary Statistics (Core Subjects)")
        summary_rows = []
        for s in CORE_SUBJECTS:
            d = eda["distributions"][s]
            summary_rows.append({
                "Subject": s,
                "Mean": d["mean"],
                "Std Dev": d["std"],
                "Q1 (25%)": d["p25"],
                "Median": d["median"],
                "Q3 (75%)": d["p75"],
                "IQR": round(d["p75"] - d["p25"], 1),
                "Min": d["min"],
                "Max": d["max"]
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # TAB 2: CORRELATIONS
    with tab2:
        st.subheader("Bivariate Correlation Matrix")
        st.caption("Pearson correlation coefficients across foundation courses and elective performance.")

        corr_dict = eda["correlations"]
        corr_cols = CORE_SUBJECTS + [e for e in eda["electives_meta"].keys()]
        corr_matrix = pd.DataFrame(corr_dict).loc[corr_cols, corr_cols]

        # Rename for clean readability
        display_names = [eda["electives_meta"][c]["name"] if c in eda["electives_meta"] else c for c in corr_cols]

        fig_corr = px.imshow(
            corr_matrix.values,
            x=display_names,
            y=display_names,
            color_continuous_scale="Blues",
            zmin=0.0,
            zmax=1.0,
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(
            height=540,
            margin=dict(l=20, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown(
            """
            **Key Correlation Takeaways:**
            - **Quantitative Coupling:** Calculus, Statistics, and Physics exhibit strong pairwise positive correlations ($r \approx 0.65 - 0.72$), reflecting shared mathematical foundations.
            - **Domain Predictability:** Econometrics strongly tracks Statistics and Calculus, while Data Mining strongly tracks Programming and Statistics.
            - **Orthogonal Verbal Signal:** English exhibits distinct variance ($r \approx 0.35 - 0.45$ against STEM subjects), providing crucial discriminative power for NLP and Corporate Finance.
            """
        )

    # TAB 3: PROFILES & PCA
    with tab3:
        st.subheader("Principal Component Analysis & Latent Dimensions")
        
        pca_c1, pca_c2 = st.columns(2)
        with pca_c1:
            # Scree Plot
            var_ratios = eda["pca_variance_ratio"]
            pc_labels = [f"PC{i+1}" for i in range(len(var_ratios))]
            cum_var = [sum(var_ratios[:i+1]) * 100 for i in range(len(var_ratios))]

            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(
                x=pc_labels,
                y=[v * 100 for v in var_ratios],
                name="Individual Explained Variance (%)",
                marker_color=OKABE_ITO["sky_blue"]
            ))
            fig_scree.add_trace(go.Scatter(
                x=pc_labels,
                y=cum_var,
                name="Cumulative Variance (%)",
                mode="lines+markers",
                line=dict(color=OKABE_ITO["vermilion"], width=2.5)
            ))
            fig_scree.update_layout(
                title="Scree Plot: Explained Variance by Component",
                yaxis_title="Variance Explained (%)",
                legend=dict(orientation="h", y=-0.2),
                height=340,
                margin=dict(l=20, r=20, t=40, b=40)
            )
            st.plotly_chart(fig_scree, use_container_width=True)

        with pca_c2:
            st.markdown("#### Component Factor Loadings")
            loadings_df = pd.DataFrame(eda["pca_loadings"])
            st.dataframe(loadings_df, use_container_width=True)
            st.caption(
                "**PC1 (Aptitude):** General academic performance across all courses. "
                "**PC2 (Verbal vs Math):** Strong contrast between English/Economics and Physics/Calculus. "
                "**PC3 (Applied Tech):** High positive loading on Programming."
            )

        st.markdown("<hr style='margin: 1.2rem 0;'>", unsafe_allow_html=True)

        # Cluster Archetypes Breakdown
        st.subheader("Cluster Archetypes (K-Means)")
        cl_c1, cl_c2 = st.columns([1.2, 1.8])
        with cl_c1:
            cl_counts = eda["cluster_counts"]
            fig_pie = px.pie(
                names=list(cl_counts.keys()),
                values=list(cl_counts.values()),
                color_discrete_sequence=[OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["purple"]],
                hole=0.45
            )
            fig_pie.update_layout(
                title="Student Distribution Across Archetypes",
                height=340,
                margin=dict(l=10, r=10, t=40, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with cl_c2:
            st.markdown("#### Mean Grades by Academic Archetype")
            cl_means_df = pd.DataFrame(eda["cluster_means"]).T
            st.dataframe(cl_means_df, use_container_width=True)

    # TAB 4: ELECTIVES
    with tab4:
        st.subheader("Elective Course Offerings")
        
        elec_rows = []
        for eid, info in eda["electives_meta"].items():
            d = eda["distributions"].get(eid, {})
            elec_rows.append({
                "Code": eid,
                "Course Name": info["name"],
                "Discipline": info["category"],
                "Mean Grade": d.get("mean", "-"),
                "Std Dev": d.get("std", "-"),
                "Description": info["description"]
            })
        st.dataframe(pd.DataFrame(elec_rows), use_container_width=True, hide_index=True)
