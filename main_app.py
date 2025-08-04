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
        border-radius: 16px;
        padding: 8px 16px;
        margin: 4px 0 8px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* App Grid */
    .app-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 12px;
        margin: 8px 0;
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
        padding: 0 8px;
    }
    
    /* iOS App Card */
    .app-card {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 20px;
        padding: 16px 18px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: 1.5px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15), 
                    0 1px 8px rgba(102, 126, 234, 0.08);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 12px;
    }
    
    .app-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.25), 
                    0 6px 16px rgba(102, 126, 234, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.8);
        background: linear-gradient(145deg, #ffffff, #f0f4ff);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .app-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe);
        border-radius: 20px 20px 0 0;
        opacity: 0.9;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.9; }
        50% { opacity: 1; }
    }
    
    .app-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.08) 0%, rgba(240, 244, 255, 0.05) 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    
    .app-card:hover::after {
        opacity: 1;
    }
    
    /* App Title */
    .app-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
        line-height: 1.1;
        transition: all 0.3s ease;
        position: relative;
        z-index: 2;
        text-shadow: 0 1px 3px rgba(102, 126, 234, 0.1);
    }
    
    .app-card:hover .app-title {
        color: #667eea;
        transform: translateY(-1px);
        text-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* App Description */
    .app-description {
        font-size: 0.75rem;
        color: #5a6c7d !important;
        line-height: 1.3;
        font-weight: 700;
        margin: 0;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 1;
        -webkit-box-orient: vertical;
        transition: all 0.3s ease;
        position: relative;
        z-index: 2;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        padding: 4px 8px;
        border-radius: 12px;
        border: 1.5px solid rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        text-shadow: 0 1px 2px rgba(102, 126, 234, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .app-card:hover .app-description {
        color: #ffffff !important;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-color: #667eea;
        transform: translateY(-2px);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        margin-bottom: 8px;
    }
    
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1d1d1f !important;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
        line-height: 1.1;
        text-shadow: none;
    }
    
    .main-subtitle {
        font-size: 0.9rem;
        color: #1d1d1f !important;
        font-weight: 500;
        margin-bottom: 8px;
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
        margin-top: 8px;
        padding: 8px;
        color: rgba(29, 29, 31, 0.6);
        font-size: 0.7rem;
        line-height: 1.2;
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
                font-size: 0.8rem;
                color: #1d1d1f !important;
                font-weight: 400;
                margin: 0;
                line-height: 1.2;
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
            # Create a clickable card container
            st.markdown(f"""
            <div class="app-card" style="margin-bottom: 20px;">
                <h3 class="app-title">{app_info['title']}</h3>
                <p class="app-description">{app_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Place the button below the card with proper spacing
            if st.button(f"🚀 Launch {app_info['description']}", key=f"card_{app_key}", 
                        help=f"Click to launch {app_info['title']} application"):
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
    
    # Render header with back navigation
    render_app_header(app_info)
    
    # Load and execute the app
    try:
        st.info(f"🔄 Loading {app_info['title']}...")
        
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
        
        st.success(f"✅ {app_info['title']} loaded successfully!")
        
    except ImportError as e:
        st.error(f"❌ Import Error: {str(e)}")
        st.info("💡 This might be due to missing dependencies in the Docker container.")
        st.code(f"pip install {str(e).split()[-1]}", language="bash")
        
    except Exception as e:
        st.error(f"❌ Error running {app_info['title']}: {str(e)}")
        st.info("🔙 Click 'Back' to return to the main menu")
        
        # Show error details in expandable section
        with st.expander("🔍 View Error Details"):
            st.code(str(e), language="python")
            import traceback
            st.code(traceback.format_exc(), language="python")

# Main app logic
def main():
    if st.session_state.current_app == 'home':
        render_home()
    else:
        run_individual_app(st.session_state.current_app)

if __name__ == "__main__":
    main()
