import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import importlib.util
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker/server environments

# Configure Streamlit page for mobile-friendly display
st.set_page_config(
    page_title="Chemlara Suite",

    layout="wide",
    initial_sidebar_state="collapsed"
)

# Deployment configuration
if 'RENDER' in os.environ:
    # Running on Render.com - removed deprecated options
    pass

# Apple-style iOS interface CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    /* Global iOS-like styling */
    .stApp {
        background: #fefcf7;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        color: #1d1d1f;
        min-height: 100vh;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        max-width: 100% !important;
        padding: 0.5rem 0.8rem;
        background: transparent;
        margin: 0 auto;
    }
    
    /* Remove default Streamlit spacing */
    .element-container {
        margin-bottom: 0 !important;
    }
    
    /* Remove gap between containers */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    
    /* Ensure no extra spacing in columns */
    .row-widget.stHorizontal {
        gap: 0 !important;
    }
    
    /* Header Navigation Bar */
    .nav-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 12px 20px;
        margin: 4px 0 0 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    /* Horizontal Navigation Bar */
    .horizontal-nav {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 12px 16px;
        margin: 4px 0 16px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .nav-buttons {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
    }
    
    /* Navigation Button */
    .nav-button {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 12px;
        padding: 8px 16px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: 1.5px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        min-width: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-decoration: none;
        color: #2c3e50;
        font-weight: 600;
        font-size: 0.85rem;
        height: 48px;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.25);
        background: linear-gradient(145deg, #ffffff, #f0f4ff);
        border-color: rgba(102, 126, 234, 0.4);
        color: #667eea;
    }
    
    .nav-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 12px 12px 0 0;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .nav-button:hover::before {
        opacity: 1;
    }
    
    /* Navigation Button Title */
    .nav-button-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
        letter-spacing: -0.01em;
        line-height: 1.1;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .nav-button:hover .nav-button-title {
        color: #667eea;
    }
    
    /* Main content area */
    .main-content {
        max-width: 100% !important;
        margin: 0 auto;
        padding: 4px 8px;
        min-height: auto;
    }
    
    /* Home page specific styling */
    .home-content {
        max-width: 100% !important;
        margin: 0 auto;
        padding: 4px 8px 8px 8px;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        margin-bottom: 16px;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 20px 32px;
        margin: 0 auto 12px auto;
        max-width: 100% !important;
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        pointer-events: none;
    }
    
    /* Logo Container */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
    }
    
    .main-logo {
        font-size: 3rem;
        margin-right: 16px;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
        animation: logoFloat 3s ease-in-out infinite;
    }
    
    @keyframes logoFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: white !important;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #fff 0%, #f0f8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subtitle Container */
    .subtitle-container {
        position: relative;
        z-index: 1;
        margin-bottom: 8px;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 400;
        margin-bottom: 12px;
        line-height: 1.4;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .ai-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3); }
        to { box-shadow: 0 4px 25px rgba(255, 107, 107, 0.5); }
    }
    
    .subtitle-text {
        font-weight: 500;
    }
    
    .tagline {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 400;
        margin: 0;
        line-height: 1.2;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .tagline-icon {
        font-size: 1.1rem;
        animation: rocket 2s ease-in-out infinite;
    }
    
    @keyframes rocket {
        0%, 100% { transform: translateX(0px) rotate(0deg); }
        25% { transform: translateX(2px) rotate(5deg); }
        75% { transform: translateX(-2px) rotate(-5deg); }
    }
    
    /* Gradient Line */
    .gradient-line {
        width: 120px;
        height: 4px;
        background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%);
        margin: 0 auto;
        border-radius: 2px;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.7; transform: scaleX(1); }
        50% { opacity: 1; transform: scaleX(1.1); }
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 6px;
        height: 6px;
        background: #30d158;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 8px;
        padding: 8px;
        color: rgba(29, 29, 31, 0.6);
        font-size: 0.7rem;
        line-height: 1.2;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .nav-buttons {
            gap: 8px;
        }
        
        .nav-button {
            min-width: 120px;
            font-size: 0.75rem;
            padding: 6px 12px;
            height: 42px;
        }
        
        .main-content {
            padding: 16px;
        }
        
        .hero-section {
            padding: 24px 16px;
            margin: 0 auto 12px auto;
            border-radius: 16px;
        }
        
        .main-title {
            font-size: 2.2rem;
        }
        
        .main-logo {
            font-size: 2.2rem;
            margin-right: 12px;
        }
        
        .main-subtitle {
            font-size: 1rem;
            flex-direction: column;
            gap: 6px;
        }
        
        .tagline {
            font-size: 0.9rem;
        }
        
        .logo-container {
            margin-bottom: 16px;
            flex-direction: column;
            gap: 8px;
        }
        
        .nav-button {
            min-width: 100px;
            font-size: 0.7rem;
            padding: 5px 10px;
            height: 38px;
        }
        
        .nav-container {
            padding: 10px 14px;
        }
    }
    
    @media (max-width: 480px) {
        .hero-section {
            padding: 20px 12px;
        }
        
        .main-title {
            font-size: 1.8rem;
        }
        
        .main-logo {
            font-size: 1.8rem;
            margin-right: 0;
            margin-bottom: 8px;
        }
        
        .logo-container {
            flex-direction: column;
            gap: 4px;
        }
        
        .main-subtitle {
            font-size: 0.9rem;
        }
        
        .tagline {
            font-size: 0.8rem;
        }
            margin: 6px 0 12px 0;
        }
        
        .horizontal-nav {
            padding: 8px 12px;
            margin: 6px 0 16px 0;
        }
        
        .main .block-container {
            padding: 0.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.4rem;
        }
        
        .nav-button {
            min-width: 90px;
            font-size: 0.65rem;
            padding: 4px 8px;
            height: 36px;
        }
        
        .nav-buttons {
            gap: 6px;
        }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3),
                    0 2px 8px rgba(102, 126, 234, 0.15);
        width: 100%;
        margin-top: 8px;
        height: 48px;
        letter-spacing: 0.3px;
        position: relative;
        overflow: hidden;
        z-index: 10;
        opacity: 1;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4),
                    0 4px 12px rgba(102, 126, 234, 0.25);
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Back button specific styling */
    .stButton[data-testid="back_btn"] > button {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 500;
        width: auto;
        min-width: 60px;
        height: 32px;
        margin-top: 0;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .stButton[data-testid="back_btn"] > button:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Elegant spacing and typography */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(29, 29, 31, 0.1);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(29, 29, 31, 0.3);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(29, 29, 31, 0.4);
    }
    
    /* iOS-style Navigation Bar */
    .nav-bar-container {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        margin: 0 -1rem 0 -1rem;
        padding: 1rem;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.05);
    }
    
    /* Apple-style buttons - iOS navigation */
    .stButton > button {
        background: rgba(255, 255, 255, 0.7) !important;
        color: #333 !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 14px 20px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        height: auto !important;
        min-width: auto !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 40px rgba(0, 122, 255, 0.2) !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 122, 255, 0.3) !important;
    }
    
    /* Fallback for buttons without backdrop-filter support */
    @supports not (backdrop-filter: blur(10px)) {
        .stButton > button {
            background: rgba(255, 255, 255, 0.95) !important;
        }
    }
    
    /* Navigation columns layout */
    .nav-columns {
        gap: 12px;
    }
    
    /* Navigation columns layout */
    .nav-columns {
        gap: 12px;
    }
    
    /* App content container */
    .app-content {
        min-height: 400px;
        padding: 2rem 0 1rem 0;
        margin-top: -1rem;
    }
    
    /* Main content areas */
    .home-content {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Individual app containers */
    .individual-app-container {
        background: transparent;
        backdrop-filter: none;
        border-radius: 0;
        padding: 0;
        margin: 0;
        box-shadow: none;
        border: none;
        min-height: auto;
    }
    
    /* Enhanced styling for content */
    .app-content h1, .app-content h2, .app-content h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .app-content p {
        color: #5a6c7d;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Mobile responsive styles */
    @media (max-width: 768px) {
        .main .block-container {
            max-width: 100%;
            padding: 0.4rem 0.6rem;
        }
        
        .main-content {
            max-width: 100%;
            padding: 4px 6px;
        }
        
        .home-content {
            max-width: 100%;
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        
        .individual-app-container {
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        
        .hero-section {
            max-width: 750px;
            padding: 16px 24px;
        }
        
        .nav-bar-container {
            padding: 0.75rem;
        }
        
        .stButton > button {
            font-size: 0.8rem !important;
            padding: 10px 16px !important;
            min-width: 100px !important;
            height: 42px !important;
        }
    }
    
    /* Extra small mobile screens */
    @media (max-width: 480px) {
        .main .block-container {
            max-width: 100%;
            padding: 0.3rem 0.5rem;
        }
        
        .main-content {
            max-width: 100%;
            padding: 2px 4px;
        }
        
        .home-content {
            max-width: 100%;
            padding: 1rem;
        }
        
        .individual-app-container {
            padding: 0;
            margin: 0;
            background: transparent;
        }
        
        .hero-section {
            max-width: 650px;
            padding: 14px 20px;
        }
        
        .nav-bar-container {
            padding: 0.5rem;
        }
        
        .stButton > button {
            font-size: 0.75rem !important;
            padding: 8px 12px !important;
            min-width: 85px !important;
            height: 38px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_app' not in st.session_state:
    st.session_state.current_app = 'home'

# App configurations with navigation info
apps_config = {
    'home': {
        'title': 'Home',
        'description': 'Home',
        'nav_label': '🏠 Home',
        'file': None
    },
    'classification': {
        'title': 'AutoML Activity Prediction',
        'description': 'AutoML Classification',
        'nav_label': '🧬 Classification',
        'file': 'app_classification.py'
    },
    'classification_multiple': {
        'title': 'AutoML Multi-Class Prediction',
        'description': 'Multi-Class AutoML',
        'nav_label': '🎨 Multi-Class',
        'file': 'app_classification_multiple.py'
    },
    'regression': {
        'title': 'AutoML Potency Prediction',
        'description': 'AutoML Regression',
        'nav_label': '💊 Regression',
        'file': 'app_regression.py'
    },
    'graph_classification': {
        'title': 'Graph Convolution Activity Prediction',
        'description': 'Graph Classification',
        'nav_label': '🔗 Graph Class',
        'file': 'app_graph_classification.py'
    },
    'graph_multiclass': {
        'title': 'Graph Multi-Class Prediction',
        'description': 'Graph Multi-Class',
        'nav_label': '🔥 Graph Multi',
        'file': 'app_graph_multiclass.py'
    },
    'graph_regression': {
        'title': 'Graph Convolution Potency Prediction',
        'description': 'Graph Regression',
        'nav_label': '⚗️ Graph Reg',
        'file': 'app_graph_regression.py'
    }
}

def load_app_module(app_file):
    """Dynamically load an app module"""
    try:
        spec = importlib.util.spec_from_file_location("app_module", app_file)
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        return app_module
    except Exception as e:
        st.error(f"Error loading app: {e}")
        return None

def render_navigation_bar():
    """Render iOS-style horizontal navigation bar"""
    
    # Add custom CSS for active state
    st.markdown("""
    <style>
        /* Active navigation button */
        div[data-testid="column"] button[kind="primary"] {
            background: linear-gradient(135deg, #007AFF, #5856D6) !important;
            color: white !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 122, 255, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
        }
        
        div[data-testid="column"] button[kind="primary"]:hover {
            transform: translateY(-4px) !important;
            box-shadow: 0 12px 35px rgba(0, 122, 255, 0.5) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 12px;
        margin: 10px 0 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    ">
    """, unsafe_allow_html=True)
    
    # Create navigation items
    cols = st.columns(len(apps_config))
    
    for i, (app_key, app_info) in enumerate(apps_config.items()):
        with cols[i]:
            # Check if this is the current app
            is_active = st.session_state.current_app == app_key
            
            # Create button with appropriate styling
            if st.button(
                app_info['nav_label'], 
                key=f"nav_{app_key}",
                help=app_info['title'],
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_app = app_key
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_main_interface():
    """Render the main interface with navigation and content"""
    
    # Header
    st.markdown("""
    <div class="nav-container">
        <div class="main-header">
            <div class="hero-section">
                <div class="logo-container">
                    <div class="main-logo">⚛️</div>
                    <h1 class="main-title">Chemlara Suite</h1>
                </div>
                <div class="subtitle-container">
                    <p class="main-subtitle">
                        <span class="ai-badge">AI</span>
                        <span class="subtitle-text">Based Activity and Potency Prediction</span>
                    </p>
                    <p class="tagline">
                        <span class="tagline-icon">🚀</span>
                        Modeling and Deployment
                    </p>
                </div>
                <div class="gradient-line"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation bar
    render_navigation_bar()
    
    # Content area with better spacing
    current_app = st.session_state.current_app
    
    if current_app == 'home':
        render_home_content()
    else:
        # Wrap individual apps in a styled container
        st.markdown('<div class="individual-app-container">', unsafe_allow_html=True)
        run_individual_app(current_app)
        st.markdown('</div>', unsafe_allow_html=True)

def render_home_content():
    """Render the home page content"""
    # Home content area with beautiful styling
    st.markdown('<div class="home-content">', unsafe_allow_html=True)
    
    # Welcome content
    st.markdown("""
    ### 🎯 Welcome to Chemlara Suite
    
    Advanced machine learning applications for chemical research and drug discovery. 
    Use the navigation bar above to access different prediction models:
    
    **🧬 Classification Models:**
    - **AutoML Classification**: Binary activity prediction using automated machine learning
    - **Multi-Class Classification**: Categorize compounds into multiple activity classes
    
    **💊 Regression Models:**
    - **AutoML Regression**: Predict potency values and continuous properties
    
    **🔗 Graph Neural Networks:**
    - **Graph Classification**: Molecular activity prediction using graph convolution
    - **Graph Multi-Class**: Multi-category prediction with graph neural networks  
    - **Graph Regression**: Potency prediction using molecular graph structures
    
    ---
    
    **🚀 Features:**
    - Upload CSV files with SMILES and activity data
    - Automated model training and optimization
    - Real-time predictions on new compounds
    - Batch processing capabilities
    - Model performance metrics and validation
    
    All models are powered by advanced algorithms including **TPOT AutoML**, **DeepChem**, and **RDKit**.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with reduced spacing
    st.markdown("""
    <div class="footer" style="margin-top: 1rem; text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.5); border-radius: 15px;">
        <p style="margin: 0; color: #667eea; font-weight: 500;">Built with Streamlit • Powered by RDKit, DeepChem & TPOT</p>
        <p style="margin: 0; color: #8e9aaf; font-size: 0.9rem;">© 2025 Chemlara Suite - Advanced Chemical Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

def render_app_header(app_info):
    """Render app header with back button"""
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        if st.button("← Back", key="back_btn", help="Return to main menu"):
            st.session_state.current_app = 'home'
            st.rerun()
    
    with col2:
        st.markdown(f"""
        <div style="
            text-align: center;
            padding: 8px 0;
        ">
            <h1 style="
                color: #2c3e50;
                margin: 0;
                font-size: 1.5rem;
                font-weight: 600;
            ">
                🧬 {app_info['title']}
            </h1>
            <p style="
                color: #667eea;
                margin: 4px 0 0 0;
                font-size: 0.9rem;
                font-weight: 500;
            ">
                {app_info['description']} Application
            </p>
        </div>
        """, unsafe_allow_html=True)

def run_individual_app(app_key):
    """Run an individual app"""
    app_info = apps_config[app_key]
    app_file = app_info['file']
    
    # Check if file exists
    if not os.path.exists(app_file):
        st.error(f"❌ App file '{app_file}' not found!")
        st.info("📂 Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.py'):
                st.write(f"  • {file}")
        return
    
    # Load and execute the app
    try:
        # Read the app file content
        with open(app_file, 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        # Remove conflicting st.set_page_config calls
        lines = app_content.split('\n')
        filtered_lines = []
        skip_config = False
        
        for line in lines:
            if 'st.set_page_config' in line:
                skip_config = True
                continue
            elif skip_config and (line.strip().endswith(')') or line.strip() == ''):
                skip_config = False
                if line.strip().endswith(')'):
                    continue
            elif skip_config:
                continue
            else:
                filtered_lines.append(line)
        
        modified_content = '\n'.join(filtered_lines)
        
        # Create a clean namespace for the app
        app_globals = {
            '__name__': '__main__',
            'st': st,
            'pd': pd,
            'os': os,
            'sys': sys,
            'Path': Path,
            'matplotlib': matplotlib
        }
        
        # Execute the app content
        exec(modified_content, app_globals)
        
    except ImportError as e:
        st.error(f"❌ Import Error: {str(e)}")
        st.info("💡 This might be due to missing dependencies in the Docker container.")
        st.code(f"pip install {str(e).split()[-1]}", language="bash")
        
    except Exception as e:
        st.error(f"❌ Error running {app_info['title']}: {str(e)}")
        
        # Show error details in expandable section
        with st.expander("🔍 View Error Details"):
            st.code(str(e), language="python")
            import traceback
            st.code(traceback.format_exc(), language="python")

# Main app logic
def main():
    render_main_interface()

if __name__ == "__main__":
    main()
