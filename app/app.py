"""
Main Application Entry Point for Course Recommendation System (STA308 Final Project).
Redesigned with the Claude Design System:
- Editorial slab-serif display typography ("Newsreader", Georgia)
- Warm tinted cream canvas (#faf9f5) with dark warm ink (#141413)
- Warm coral signature accent (#cc785c)
- Clean sidebar navigation with state persistence
"""

import os
import sys
import streamlit as st

# Ensure project root is in python search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.utils import inject_custom_css, CLAUDE
from app.views.explore_view import render_explore
from app.views.home_view import render_home
from app.views.methodology_view import render_methodology
from app.views.performance_view import render_performance
from app.views.recommend_view import render_recommend

# Configure Streamlit page layout
st.set_page_config(
    page_title="Course Recommendation System | STA308",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom Claude design system CSS
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

if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = st.session_state.current_page


def on_nav_change():
    """Single-pass transition callback preventing double reruns."""
    st.session_state.current_page = st.session_state.nav_radio


def navigate_to(page_name: str):
    """Programmatic page transition helper."""
    st.session_state.current_page = page_name
    st.session_state.nav_radio = page_name
    st.rerun()


# Sidebar Navigation with Claude Warm Editorial Theme
with st.sidebar:
    st.markdown(
        f"""
        <div style="margin-bottom: 1.4rem; padding-bottom: 0.8rem; border-bottom: 1px solid {CLAUDE['hairline']};">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: {CLAUDE['primary']}; font-size: 1.4rem; font-weight: 700;">✦</span>
                <span style="font-family: 'Newsreader', Georgia, serif; font-size: 1.35rem; font-weight: 500; color: {CLAUDE['ink']}; letter-spacing: -0.01em;">CourseRec</span>
            </div>
            <div style="font-size: 0.76rem; color: {CLAUDE['muted']}; font-weight: 500; margin-top: 2px;">
                STA308 Final Project • Anthropic Claude Aesthetic
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page_labels = {
        "Home": "Overview",
        "Get Recommendations": "Recommendations",
        "Explore Data": "Cohort Dataset",
        "Model Performance": "Model Evaluation",
        "Methodology & Ethics": "Methodology & Ethics"
    }

    st.radio(
        "Navigation",
        options=PAGES,
        index=PAGES.index(st.session_state.current_page),
        format_func=lambda p: f"✦  {page_labels.get(p, p)}",
        key="nav_radio",
        on_change=on_nav_change,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid #e6dfd8; margin: 1.6rem 0 1rem 0;'>", unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <div style="background-color: {CLAUDE['surface_card']}; border: 1px solid {CLAUDE['hairline']}; border-radius: 8px; padding: 10px 12px; font-size: 0.8rem; line-height: 1.55; color: {CLAUDE['body']};">
            <span style="font-weight: 600; color: {CLAUDE['ink']};">System Status</span><br>
            • Artifacts: <span style="color: {CLAUDE['accent_teal']}; font-weight: 600;">Active & Cached</span><br>
            • Cohort: 1,200 observations<br>
            • Primary Model: Ridge (λ=10.0)<br>
            • Latency: &lt; 0.1s
        </div>
        <div style="margin-top: 1.2rem; font-size: 0.74rem; color: {CLAUDE['muted']}; line-height: 1.4;">
            Design System: <span style="color: {CLAUDE['primary']}; font-weight: 500;">Claude Warm Editorial</span><br>
            Version 1.0.0 • Python 3.13
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
