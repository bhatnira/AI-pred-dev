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
    page_title="ChemML Suite",
    page_icon="🧬",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        max-width: 1200px;
        padding: 0.5rem 1rem;
        background: transparent;
        margin: 0 auto;
    }
    
    /* iOS Navigation Bar */
    .nav-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 16px 24px;
        margin: 12px 0 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* App Grid */
    .app-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 24px;
        margin: 24px 0;
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
        padding: 0 8px;
    }
    
    /* iOS App Card */
    .app-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 24px;
        padding: 32px 24px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: 1.5px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 
                    0 2px 8px rgba(0, 0, 0, 0.05);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 8px;
    }
    
    .app-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15), 
                    0 8px 24px rgba(0, 0, 0, 0.08),
                    inset 0 1px 0 rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.98);
        border-color: rgba(255, 255, 255, 0.6);
    }
    
    .app-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 24px 24px 0 0;
        opacity: 0.8;
    }
    
    .app-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.03) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    
    .app-card:hover::after {
        opacity: 1;
    }
    
    /* App Title */
    .app-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1c1c1e;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
        line-height: 1.1;
        transition: all 0.3s ease;
        position: relative;
        z-index: 2;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
    }
    
    .app-card:hover .app-title {
        color: #007AFF;
        transform: translateY(-1px);
        text-shadow: 0 2px 6px rgba(0, 122, 255, 0.4);
    }
    
    /* App Description */
    .app-description {
        font-size: 0.95rem;
        color: #3a3a3c !important;
        line-height: 1.3;
        font-weight: 800;
        margin: 0;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        transition: all 0.3s ease;
        position: relative;
        z-index: 2;
        background: rgba(58, 58, 60, 0.15);
        padding: 8px 16px;
        border-radius: 16px;
        border: 2px solid rgba(58, 58, 60, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
    }
    
    .app-card:hover .app-description {
        color: #ffffff !important;
        background: rgba(0, 122, 255, 0.95);
        border-color: rgba(0, 122, 255, 1);
        transform: translateY(-2px);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        margin-bottom: 16px;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1d1d1f !important;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
        line-height: 1.1;
        text-shadow: none;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: #1d1d1f !important;
        font-weight: 500;
        margin-bottom: 16px;
        line-height: 1.3;
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
        margin-top: 32px;
        padding: 16px;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.8rem;
        line-height: 1.4;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .app-grid {
            grid-template-columns: 1fr;
            gap: 12px;
            margin: 12px 0;
        }
        
        .main-title {
            font-size: 2rem;
        }
        
        .main-subtitle {
            font-size: 1rem;
        }
        
        .app-card {
            padding: 24px 18px;
            height: 140px;
            margin-bottom: 6px;
        }
        
        .app-title {
            font-size: 1.4rem;
            font-weight: 800;
        }
        
        .app-description {
            font-size: 0.9rem;
            padding: 7px 14px;
            font-weight: 800;
        }
        
        .nav-container {
            padding: 12px 16px;
            margin: 8px 0 16px 0;
        }
        
        .main .block-container {
            padding: 0.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.8rem;
        }
        
        .app-card {
            height: 120px;
        }
        
        .app-title {
            font-size: 1.3rem;
            font-weight: 800;
        }
        
        .app-description {
            font-size: 0.85rem;
            color: #3a3a3c !important;
            font-weight: 800;
            padding: 6px 12px;
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
        margin-top: 12px;
        height: 42px;
        letter-spacing: 0.3px;
        position: relative;
        overflow: hidden;
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
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_app' not in st.session_state:
    st.session_state.current_app = 'home'

# App configurations
apps_config = {
    'classification': {
        'title': 'AutoML Activity Prediction',
        'description': 'Classification',
        'file': 'app_classification.py'
    },
    'regression': {
        'title': 'AutoML Potency Prediction',
        'description': 'Regression',
        'file': 'app_regression.py'
    },
    'graph_classification': {
        'title': 'Graph Convolution Activity Prediction',
        'description': 'Classification',
        'file': 'app_graph_classification.py'
    },
    'graph_regression': {
        'title': 'Graph Convolution Potency Prediction',
        'description': 'Regression',
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

def render_home():
    """Render the home page with app selection"""
    
    # Header
    st.markdown("""
    <div class="nav-container">
        <div class="main-header">
            <h1 class="main-title">ChemML Suite</h1>
            <p class="main-subtitle">
                <span class="status-indicator"></span>
                AI Based Activity and Potency Prediction
            </p>
            <p style="
                font-size: 0.95rem;
                color: #1d1d1f !important;
                font-weight: 400;
                margin: 0;
                line-height: 1.3;
            ">
                Modeling and Deployment
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # App grid
    st.markdown('<div class="app-grid">', unsafe_allow_html=True)
    
    # Create columns for responsive grid
    cols = st.columns(2)
    
    for i, (app_key, app_info) in enumerate(apps_config.items()):
        col = cols[i % 2]
        
        with col:
            st.markdown(f"""
            <div class="app-card">
                <h3 class="app-title">{app_info['title']}</h3>
                <p class="app-description">{app_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Launch {app_info['title']}", key=f"btn_{app_key}"):
                st.session_state.current_app = app_key
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with Streamlit • Powered by RDKit, DeepChem & TPOT</p>
        <p>© 2025 ChemML Suite - Advanced Chemical Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

def render_app_header(app_info):
    """Render app header with back button"""
    st.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 12px 20px;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div style="flex: 1;"></div>
        <div style="
            text-align: center;
            flex: 2;
        ">
            <h1 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            ">
                {app_info['title']}
            </h1>
        </div>
        <div style="flex: 1;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        if st.button("← Back", key="back_btn"):
            st.session_state.current_app = 'home'
            st.rerun()

def run_individual_app(app_key):
    """Run an individual app"""
    app_info = apps_config[app_key]
    app_file = app_info['file']
    
    # Check if file exists
    if not os.path.exists(app_file):
        st.error(f"App file '{app_file}' not found!")
        return
    
    # Render header
    render_app_header(app_info)
    
    # Reduce spacing
    st.markdown('<div style="margin-top: -8px;"></div>', unsafe_allow_html=True)
    
    # Load and run the app
    try:
        # Read the app file content
        with open(app_file, 'r') as f:
            app_content = f.read()
        
        # Remove the st.set_page_config call to avoid conflicts
        lines = app_content.split('\n')
        filtered_lines = []
        skip_next = False
        
        for line in lines:
            if 'st.set_page_config' in line:
                skip_next = True
                continue
            elif skip_next and line.strip().startswith(')'):
                skip_next = False
                continue
            elif skip_next and ('=' in line or line.strip().startswith('"')):
                continue
            else:
                skip_next = False
                filtered_lines.append(line)
        
        modified_content = '\n'.join(filtered_lines)
        
        # Execute the modified app content
        exec(modified_content, globals())
        
    except Exception as e:
        st.error(f"Error running app: {str(e)}")
        st.info("Click 'Back' to return to the main menu")

# Main app logic
def main():
    if st.session_state.current_app == 'home':
        render_home()
    else:
        run_individual_app(st.session_state.current_app)

if __name__ == "__main__":
    main()
