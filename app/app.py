"""
Main Application Entry Point for Course Recommendation System (STA308 Final Project).
Manages layout, sidebar navigation, theme injection, and page state preservation.
"""

import os
import sys
import streamlit as st

# Ensure project root is in python search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.utils import inject_custom_css
from app.views.explore_view import render_explore
from app.views.home_view import render_home
from app.views.methodology_view import render_methodology
from app.views.performance_view import render_performance
from app.views.recommend_view import render_recommend

# Configure Streamlit page layout
st.set_page_config(
    page_title="Course Recommendation System | STA308",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom global design system CSS
inject_custom_css()

# Navigation pages list
PAGES = [
    "Home",
    "Get Recommendations",
    "Explore Data",
    "Model Performance",
    "Methodology & Ethics"
]

# Initialize navigation state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"


def navigate_to(page_name: str):
    """Programmatic page transition helper."""
    st.session_state.current_page = page_name
    st.rerun()


# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem;">
            <span style="font-size: 2rem;">🎓</span>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #0F172A;">CourseRec AI</h3>
                <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">STA308 Final Project</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page_icons = {
        "Home": "🏠",
        "Get Recommendations": "🎯",
        "Explore Data": "📊",
        "Model Performance": "📈",
        "Methodology & Ethics": "📖"
    }

    selected = st.radio(
        "Navigation",
        options=PAGES,
        index=PAGES.index(st.session_state.current_page),
        format_func=lambda p: f"{page_icons.get(p, '•')}  {p}",
        label_visibility="collapsed"
    )

    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.rerun()

    st.markdown("<hr style='margin: 1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">
            <strong>System Status:</strong><br>
            • Artifacts: <span style="color: #059669; font-weight: 600;">Cached & Active</span><br>
            • Cohort: 1,200 students<br>
            • Model: Ridge (λ=10.0)<br>
            • Latency: &lt; 0.1s
        </div>
        <div style="margin-top: 1.5rem; font-size: 0.75rem; color: #94A3B8;">
            Version 1.0.0 • Python 3.13<br>
            MIT License • Confidential
        </div>
        """,
        unsafe_allow_html=True
    )

# Page Router
current = st.session_state.current_page

if current == "Home":
    render_home(navigate_to)
elif current == "Get Recommendations":
    render_recommend()
elif current == "Explore Data":
    render_explore()
elif current == "Model Performance":
    render_performance()
elif current == "Methodology & Ethics":
    render_methodology()
