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
        max-width: 1200px;
        padding: 0.5rem 1rem;
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
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        min-height: 60vh;
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
        
        .main-title {
            font-size: 1.6rem;
        }
        
        .main-subtitle {
            font-size: 0.85rem;
        }
        
        .nav-button {
            min-width: 100px;
            font-size: 0.7rem;
            padding: 5px 10px;
            height: 38px;
        }
        
        .nav-container {
            padding: 10px 14px;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_app' not in st.session_state:
    st.session_state.current_app = 'home'

# App configurations
apps_config = {
    'home': {
        'title': 'Home',
        'description': 'Home',
        'file': None
    },
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
        'description': 'Graph Classification',
        'file': 'app_graph_classification.py'
    },
    'graph_regression': {
        'title': 'Graph Convolution Potency Prediction',
        'description': 'Graph Regression',
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

def render_main_interface():
    """Render the main interface with tabbed navigation"""
    
    # Header
    st.markdown("""
    <div class="nav-container">
        <div class="main-header">
            <h1 class="main-title">Chemlara Suite</h1>
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
    
    # Create tabs
    tab_labels = [app_info['description'] for app_info in apps_config.values()]
    tabs = st.tabs(tab_labels)
    
    # Home tab
    with tabs[0]:
        render_home_content()
    
    # Classification tab
    with tabs[1]:
        run_individual_app('classification')
    
    # Regression tab  
    with tabs[2]:
        run_individual_app('regression')
    
    # Graph Classification tab
    with tabs[3]:
        run_individual_app('graph_classification')
    
    # Graph Regression tab
    with tabs[4]:
        run_individual_app('graph_regression')

def render_home_content():
    """Render the home page content"""
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Welcome content
    st.markdown("""
    ### 🎯 Welcome to Chemlara Suite
    
    Select an application from the navigation tabs above to get started:
    
    - **🧬 AutoML Activity Prediction**: Classification models for molecular activity
    - **💊 AutoML Potency Prediction**: Regression models for potency estimation  
    - **🔗 Graph Activity Prediction**: Graph neural networks for activity classification
    - **⚗️ Graph Potency Prediction**: Graph neural networks for potency regression
    
    All models are powered by advanced machine learning algorithms including TPOT AutoML and DeepChem.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with Streamlit • Powered by RDKit, DeepChem & TPOT</p>
        <p>© 2025 Chemlara Suite - Advanced Chemical Machine Learning</p>
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
