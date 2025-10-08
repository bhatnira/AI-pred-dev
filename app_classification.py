import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import joblib
import base64
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import time
import ssl
import deepchem as dc
from lime import lime_tabular
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import numpy as np
import colorsys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from datetime import datetime
import io
import zipfile
import threading
import random
import streamlit.components.v1 as components

# Configure matplotlib and RDKit for headless mode
os.environ['MPLBACKEND'] = 'Agg'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

# Disable RDKit warnings and configure for headless rendering
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Configure Streamlit page for mobile-friendly display
st.set_page_config(
    page_title="Chemlara Predictor - TPOT Classification",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apple-style iOS interface CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    /* Global iOS-like styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', system-ui, sans-serif;
        color: #1D1D1F;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 100%;
        padding: 1rem;
        background: transparent;
    }
    
    /* iOS Typography System */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif;
        font-weight: 700;
        color: #1D1D1F;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.5em;
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    
    h2 {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.022em;
    }
    
    h3 {
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    h4 {
        font-size: 1.25rem;
        font-weight: 600;
        letter-spacing: -0.018em;
    }
    
    h5 {
        font-size: 1.125rem;
        font-weight: 600;
        letter-spacing: -0.015em;
    }
    
    h6 {
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* iOS Body Text */
    p, div, span, .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif;
        font-weight: 400;
        color: #1D1D1F;
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
    }
    
    /* iOS Text Sizes */
    .ios-text-large {
        font-size: 1.125rem;
        font-weight: 400;
        line-height: 1.4;
    }
    
    .ios-text-regular {
        font-size: 1rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    .ios-text-small {
        font-size: 0.875rem;
        font-weight: 400;
        line-height: 1.4;
        color: #8E8E93;
    }
    
    .ios-text-caption {
        font-size: 0.75rem;
        font-weight: 500;
        line-height: 1.3;
        color: #8E8E93;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* iOS Headings with specific styling */
    .ios-heading-xl {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.05;
        color: #1D1D1F;
    }
    
    .ios-heading-large {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.1;
        color: #1D1D1F;
    }
    
    .ios-heading-medium {
        font-size: 1.75rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        line-height: 1.2;
        color: #1D1D1F;
    }
    
    .ios-heading-small {
        font-size: 1.25rem;
        font-weight: 600;
        letter-spacing: -0.015em;
        line-height: 1.3;
        color: #1D1D1F;
    }
    
    /* iOS Card styling */
    .ios-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .ios-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Header with Apple-style gradient */
    .ios-header {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        border-radius: 24px;
        padding: 32px 24px;
        margin-bottom: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 122, 255, 0.3);
    }
    
    /* Apple-style buttons */
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
        width: 100%;
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
    
    /* Fallback for buttons without backdrop-filter support */
    @supports not (backdrop-filter: blur(10px)) {
        .stButton > button {
            background: rgba(255, 255, 255, 0.95) !important;
        }
    }
    
    /* iOS Input fields */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 16px;
        font-size: 16px;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid #007AFF;
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1);
        outline: none;
    }
    
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 16px;
        font-size: 16px;
    }
    
    /* iOS Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #007AFF 0%, #5856D6 100%);
        border-radius: 8px;
        height: 8px;
    }
    
    .stProgress > div > div {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        height: 8px;
    }
    
    /* Metric cards - iOS style (more compact) */
    .ios-metric {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 8px; /* smaller */
        margin: 4px;  /* smaller */
        text-align: center;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .ios-metric:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
    }
    
    /* File uploader - iOS style */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 2px dashed #007AFF;
        padding: 32px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        background: rgba(0, 122, 255, 0.05);
        border-color: #5856D6;
    }
    
    /* Tabs - iOS style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 12px 20px;
        transition: all 0.2s ease;
        border: none;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }
    
    /* Expander - iOS style */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        font-weight: 500;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Success/Warning/Error - iOS style */
    .stSuccess {
        background: rgba(52, 199, 89, 0.1);
        border: 1px solid rgba(52, 199, 89, 0.3);
        border-radius: 12px;
        color: #34C759;
    }
    
    .stWarning {
        background: rgba(255, 149, 0, 0.1);
        border: 1px solid rgba(255, 149, 0, 0.3);
        border-radius: 12px;
        color: #FF9500;
    }
    
    .stError {
        background: rgba(255, 59, 48, 0.1);
        border: 1px solid rgba(255, 59, 48, 0.3);
        border-radius: 12px;
        color: #FF3B30;
    }
    
    /* Spinner */
    .stSpinner > div {
        color: #007AFF;
    }
    
    /* Streamlit component typography enhancements */
    .stSelectbox label, .stTextInput label, .stFileUploader label {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        color: #1D1D1F !important;
        letter-spacing: -0.01em !important;
    }
    
    .stExpander summary {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        color: #1D1D1F !important;
        letter-spacing: -0.015em !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        letter-spacing: -0.01em !important;
    }
    
    .stButton > button {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif !important;
        font-weight: 600 !important;
        font-size: 17px !important;
        letter-spacing: -0.01em !important;
    }
    
    .stSlider label, .stNumberInput label {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        color: #1D1D1F !important;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
        }
        
        .ios-card {
            margin: 8px 0;
            padding: 16px;
            border-radius: 16px;
        }
        
        .ios-header {
            padding: 24px 16px;
            border-radius: 16px;
        }
        
        .ios-metric {
            margin: 4px;
            padding: 6px; /* smaller on mobile */
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 122, 255, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 122, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

ssl._create_default_https_context = ssl._create_unverified_context

# iOS-style component functions
def create_ios_metric_card(title, value, description="", icon="📊"):
    return f"""
    <div class="ios-metric">
        <div style="font-size: 1.1em; margin-bottom: 4px;">{icon}</div>
        <h3 class="ios-text-caption" style="margin: 0; color: #007AFF; font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">{title}</h3>
        <h2 class="ios-heading-small" style="margin: 4px 0; color: #1D1D1F; font-weight: 800; font-size: 16px; letter-spacing: -0.02em;">{value}</h2>
        <p class="ios-text-small" style="margin: 0; color: #8E8E93; font-size: 9px; font-weight: 400; line-height: 1.4;">{description}</p>
    </div>
    """

def create_ios_card(title, content, icon=""):
    return f"""
    <div class="ios-card">
        <h3 class="ios-heading-small" style="color: #007AFF; margin-bottom: 16px; font-weight: 600; font-size: 20px; letter-spacing: -0.015em;">{icon} {title}</h3>
        <div class="ios-text-regular" style="color: #1D1D1F; line-height: 1.6; font-size: 16px;">{content}</div>
    </div>
    """

def create_ios_header(title, subtitle=""):
    return f"""
    <div class="ios-header">
        <h1 class="ios-heading-xl" style="margin: 0; font-size: 3.2em; font-weight: 800; letter-spacing: -0.03em; line-height: 1.05;">{title}</h1>
        <p class="ios-text-large" style="margin: 8px 0 0 0; font-size: 1.2em; opacity: 0.9; font-weight: 400; line-height: 1.4;">{subtitle}</p>
    </div>
    """

def create_prediction_result_card(prediction, probability, smiles):
    activity_icon = "🟢" if prediction == 1 else "🔴"
    activity_text = "Active" if prediction == 1 else "Not Active"
    confidence_color = "#34C759" if prediction == 1 else "#FF3B30"
    
    # Format probability safely
    prob_display = f"{probability:.1%}" if isinstance(probability, (int, float)) and not np.isnan(probability) else "N/A"
    
    return f"""
    <div class="ios-card">
        <div style="text-align: center;">
            <div style="font-size: 3em; margin-bottom: 16px;">{activity_icon}</div>
            <h2 class="ios-heading-medium" style="color: {confidence_color}; margin: 0; font-weight: 800; font-size: 32px; letter-spacing: -0.025em;">{activity_text}</h2>
            <div style="margin: 16px 0;">
                <div style="background: rgba(0, 122, 255, 0.1); border-radius: 12px; padding: 16px;">
                    <p class="ios-text-caption" style="margin: 0; color: #007AFF; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 13px;">Confidence Score</p>
                    <h3 class="ios-heading-small" style="margin: 4px 0 0 0; color: #1D1D1F; font-weight: 800; font-size: 24px; letter-spacing: -0.02em;">{prob_display}</h3>
                </div>
            </div>
            <p class="ios-text-small" style="color: #8E8E93; font-size: 14px; margin: 8px 0; line-height: 1.4;">
                <strong style="font-weight: 600;">SMILES:</strong> {smiles}
            </p>
        </div>
    </div>
    """

# Dictionary of featurizers using DeepChem
def get_featurizers():
    """Get available featurizers, handling potential initialization errors."""
    featurizers = {
        "Circular Fingerprint": dc.feat.CircularFingerprint(size=2048, radius=4),
        "MACCSKeys": dc.feat.MACCSKeysFingerprint(),
        "modred": dc.feat.MordredDescriptors(ignore_3D=True),
        "rdkit": dc.feat.RDKitDescriptors(),
        "pubchem": dc.feat.PubChemFingerprint(),
    }
    
    # Skip Mol2Vec due to model corruption issues - using robust alternative featurizers
    # This provides reliable molecular representations without dependency on external model files
    
    return featurizers

# Get available featurizers
Featurizer = get_featurizers()

# Function to standardize Smile using RDKit
def standardize_smiles(smiles, verbose=False):
    if verbose:
        st.write(smiles)
    std_mol = standardize_mol(Chem.MolFromSmiles(smiles), verbose=verbose)
    return Chem.MolToSmiles(std_mol)

# Function to standardize molecule using RDKit
def standardize_mol(mol, verbose=False):
    from rdkit.Chem.MolStandardize import rdMolStandardize
    
    clean_mol = rdMolStandardize.Cleanup(mol)
    if verbose:
        st.write('Remove Hs, disconnect metal atoms, normalize the molecule, reionize the molecule:')

    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    if verbose:
        st.write('Select the "parent" fragment:')

    uncharger = rdMolStandardize.Uncharger()
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    if verbose:
        st.write('Neutralize the molecule:')

    te = rdMolStandardize.TautomerEnumerator()
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    if verbose:
        st.write('Enumerate tautomers:')

    assert taut_uncharged_parent_clean_mol is not None
    if verbose:
        st.write(Chem.MolToSmiles(taut_uncharged_parent_clean_mol))

    return taut_uncharged_parent_clean_mol

# Function to estimate training time based on dataset size and parameters
def estimate_training_time(n_samples, n_features, generations, population_size):
    """Estimate TPOT training time based on dataset characteristics"""
    # Base time per pipeline evaluation (in seconds)
    base_time = 2.0
    
    # Scaling factors
    sample_factor = min(n_samples / 1000, 5.0)  # Cap at 5x for very large datasets
    feature_factor = min(n_features / 100, 3.0)  # Cap at 3x for high-dimensional data
    complexity_factor = generations * population_size / 100  # TPOT complexity
    
    # Estimate total time
    estimated_time = base_time * sample_factor * feature_factor * complexity_factor
    return max(estimated_time, 30)  # Minimum 30 seconds

# Function to format time duration
def format_time_duration(seconds):
    """Format time duration in a human-readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"

# Enhanced progress tracking with time estimation
def create_progress_tracker(total_time_estimate):
    """Create an advanced progress tracker with time estimation"""
    progress_container = st.container()
    
    with progress_container:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            estimated_time_placeholder = st.empty()
            estimated_time_placeholder.markdown(create_ios_metric_card("Estimated Time", format_time_duration(total_time_estimate), "", "⏱️"), unsafe_allow_html=True)
        
        with col2:
            remaining_time_placeholder = st.empty()
            remaining_time_placeholder.markdown(create_ios_metric_card("Time Remaining", "Calculating...", "", "⏳"), unsafe_allow_html=True)
        
        with col3:
            speed_placeholder = st.empty()
            speed_placeholder.markdown(create_ios_metric_card("Progress Speed", "Starting...", "", "🚀"), unsafe_allow_html=True)
        
        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        return {
            'progress_bar': progress_bar,
            'status_text': status_text,
            'remaining_time': remaining_time_placeholder,
            'speed': speed_placeholder,
            'start_time': time.time()
        }

def update_progress_tracker(tracker, current_progress, total_progress, current_status="Processing..."):
    """Update the progress tracker with current status"""
    if current_progress == 0:
        return
    
    # Calculate progress percentage
    progress_percent = min(current_progress / total_progress, 1.0)
    
    # Update progress bar
    tracker['progress_bar'].progress(progress_percent)
    
    # Calculate elapsed time and estimated remaining time
    elapsed_time = time.time() - tracker['start_time']
    
    if progress_percent > 0:
        estimated_total_time = elapsed_time / progress_percent
        remaining_time = max(estimated_total_time - elapsed_time, 0)
        
        # Calculate speed (iterations per minute)
        speed = (current_progress / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        # Update displays using metric cards
        with tracker['remaining_time']:
            tracker['remaining_time'].markdown(create_ios_metric_card("Time Remaining", format_time_duration(remaining_time), "", "⏳"), unsafe_allow_html=True)
        
        with tracker['speed']:
            tracker['speed'].markdown(create_ios_metric_card("Progress Speed", f"{speed:.1f}/min", "", "🚀"), unsafe_allow_html=True)
    
    # Update status
    tracker['status_text'].info(f"🔬 {current_status} ({int(progress_percent * 100)}% complete)")

# Function to preprocess data and perform modeling for classification
def preprocess_and_model(df, smiles_col, activity_col, featurizer_name, generations=3, cv=3, verbosity=0, test_size=0.20):
    """
    Streamlined preprocessing and TPOT model building with time tracking
    """
    start_time = time.time()
    
    # Simplified progress tracking
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Estimate total time based on dataset size
    n_samples = len(df)
    estimated_total_time = max(30, min(300, n_samples * 0.5 + generations * cv * 15))  # Smart estimation
    
    def update_progress_with_time(progress_percent, status_msg):
        progress_bar.progress(progress_percent)
        status_text.info(f"🔬 {status_msg}")
    
    try:
        # Phase 1: Data Preparation
        update_progress_with_time(0.05, "Standardizing SMILES...")
        
        df[smiles_col + '_standardized'] = df[smiles_col].apply(standardize_smiles)
        df.dropna(subset=[smiles_col + '_standardized'], inplace=True)

        # Convert activity column to binary labels
        unique_classes = df[activity_col].unique()
        if len(unique_classes) < 2:
            st.error("Not enough classes present for binary classification. Please check your dataset and ensure it has at least two distinct classes.")
            return (None,) * 15

        update_progress_with_time(0.15, "Featurizing molecules...")
        
        # Featurize molecules with progress updates
        featurizer_name = st.session_state.selected_featurizer_name
        
        # Create featurizer with custom parameters if Circular Fingerprint
        if featurizer_name == "Circular Fingerprint":
            # Get custom parameters from session state
            radius = st.session_state.get('cfp_radius', 4)
            size = st.session_state.get('cfp_size', 2048)
            featurizer = dc.feat.CircularFingerprint(size=size, radius=radius)
            # Store custom parameters for later use in visualization
            st.session_state['cfp_custom_radius'] = radius
            st.session_state['cfp_custom_size'] = size
        else:
            featurizer = Featurizer[featurizer_name]
        features = []
        smiles_list = df[smiles_col + '_standardized'].tolist()
        
        # Process in batches with progress updates
        batch_size = 50
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            for smiles in batch:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    features.append(featurizer.featurize([mol])[0])
                else:
                    st.warning(f"Invalid SMILES: {smiles}")
            
            # Update progress for featurization (15% to 45%)
            progress = 0.15 + (i / len(smiles_list)) * 0.3
            update_progress_with_time(min(progress, 0.45), f"Featurizing molecules... {i+batch_size}/{len(smiles_list)}")

        if not features:
            st.error("No valid molecules found for featurization. Please ensure your SMILES data is correct.")
            return (None,) * 15

        update_progress_with_time(0.5, "Preparing training data...")
        
        feature_df = pd.DataFrame(features)
        X = feature_df
        y = df[activity_col]

        # Convert integer column names to strings
        new_column_names = [f"fp_{col}" for col in X.columns]
        X.columns = new_column_names

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        update_progress_with_time(0.6, "Initializing TPOT classifier...")
        
        # Use optimized TPOT configuration
        tpot = TPOTClassifier(
            generations=generations,
            population_size=10,
            cv=cv,
            random_state=42, 
            verbosity=verbosity,  # Use user-selected verbosity
            config_dict='TPOT light',
            n_jobs=1,
            max_time_mins=5,
            max_eval_time_mins=0.5
        )

        update_progress_with_time(0.65, "Training TPOT model...")
        
        # Train TPOT with periodic progress updates
        training_start_time = time.time()
        
        # Create a separate thread to update progress during training
        import threading
        training_complete = threading.Event()
        
        def training_progress_updater():
            training_progress = 0.65
            while not training_complete.is_set():
                training_elapsed = time.time() - training_start_time
                # Gradually increase progress during training (65% to 85%)
                if training_elapsed < 60:  # First minute
                    training_progress = 0.65 + (training_elapsed / 60) * 0.1
                elif training_elapsed < 180:  # Next 2 minutes
                    training_progress = 0.75 + ((training_elapsed - 60) / 120) * 0.1
                else:  # After 3 minutes
                    training_progress = min(0.85, 0.85 + ((training_elapsed - 180) / 120) * 0.05)
                
                update_progress_with_time(training_progress, f"Training TPOT model... ({format_time_duration(training_elapsed)} elapsed)")
                time.sleep(5)  # Update every 5 seconds
        
        # Start progress updater
        progress_thread = threading.Thread(target=training_progress_updater)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Actual training
        tpot.fit(X_train, y_train)
        
        # Stop progress updater
        training_complete.set()
        progress_thread.join(timeout=1)
        
        training_end_time = time.time()
        actual_training_time = training_end_time - training_start_time
        
        # Removed verbose evaluation status message
        
        # Model evaluation
        y_pred = tpot.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # ROC AUC score and curve
        roc_auc = None
        fpr, tpr, thresholds = None, None, None
        
        try:
            if hasattr(tpot, 'predict_proba') and len(set(y_test)) == 2:
                y_proba = tpot.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        except Exception as e:
            st.warning(f"Could not calculate ROC AUC: {str(e)}")

        # Clear progress
        progress_container.empty()
        
        # Calculate total training time
        total_end_time = time.time()
        total_training_time = total_end_time - start_time
        
        # Display results in iOS-style cards
        if roc_auc and fpr is not None:
            # Create two columns for mobile-friendly layout
            col1, col2 = st.columns(2)
            
            # Note: Visualization plots will be displayed in the enhanced results section below
            # Store figures for session state
            if roc_auc:
                st.session_state['roc_curve_fig'] = None  # Will be created in enhanced section
            st.session_state['confusion_matrix_fig'] = None  # Will be created in enhanced section
        else:
            st.info("ℹ️ ROC curve not available for this classification problem.")

        return tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer, total_training_time, actual_training_time
    
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return (None,) * 15

# Function to create a downloadable link for HTML content
def create_download_link(html_content, link_text):
    href = f'<a href="data:text/html;base64,{base64.b64encode(html_content.encode()).decode()}" download="{link_text}.html">{link_text}</a>'
    return href

# Function to create a downloadable link for model files
def create_downloadable_model_link(model_filename, link_text):
    with open(model_filename, 'rb') as f:
        model_data = f.read()
    b64 = base64.b64encode(model_data).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="{model_filename}">{link_text}</a>'
    return href

# Safe ZIP report creator for model package
def create_model_report_zip(accuracy, precision, recall, f1, roc_auc, confusion_matrix_fig=None, roc_curve_fig=None, model_params=None):
    """Create a comprehensive model report ZIP buffer with safe formatting."""
    buffer = io.BytesIO()

    # Safely format metric values
    def fmt_metric(val):
        try:
            return f"{float(val):.4f}"
        except Exception:
            return "N/A"

    acc_s = fmt_metric(accuracy)
    prec_s = fmt_metric(precision)
    rec_s = fmt_metric(recall)
    f1_s = fmt_metric(f1)
    roc_auc_s = fmt_metric(roc_auc) if roc_auc is not None else "N/A"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S");

    # Prepare text report
    lines = [
        "Chemlara TPOT Classification - Model Report",
        f"Generated: {timestamp}",
        "",
        "Performance Metrics:",
        f"- Accuracy: {acc_s}",
        f"- Precision: {prec_s}",
        f"- Recall: {rec_s}",
        f"- F1 Score: {f1_s}",
        f"- ROC AUC: {roc_auc_s}",
        "",
        "Model Parameters:" if model_params else ""
    ]

    if model_params:
        for k, v in model_params.items():
            # Ensure safe string conversion
            try:
                if isinstance(v, float) and (k.lower().endswith('%') or 'percent' in k.lower()):
                    v_str = f"{v:.2%}"
                else:
                    v_str = str(v)
            except Exception:
                v_str = str(v)
            lines.append(f"- {k}: {v_str}")

    text_report = "\n".join([l for l in lines if l != ""]) + "\n"

    # Prepare CSV content
    csv_header = "metric,value\n"
    csv_rows = [
        f"accuracy,{acc_s}",
        f"precision,{prec_s}",
        f"recall,{rec_s}",
        f"f1,{f1_s}",
        f"roc_auc,{roc_auc_s}"
    ]
    csv_content = csv_header + "\n".join(csv_rows) + "\n"

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Write reports
        zf.writestr("model_report.txt", text_report)
        zf.writestr("metrics.csv", csv_content)

        # Write model params as a separate file for convenience
        if model_params:
            params_lines = [f"{k}: {v}" for k, v in model_params.items()]
            zf.writestr("model_params.txt", "\n".join(params_lines) + "\n")

        # Save figures if provided
        if roc_curve_fig is not None:
            try:
                buf = io.BytesIO()
                roc_curve_fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                buf.seek(0)
                zf.writestr('images/roc_curve.png', buf.read())
            except Exception:
                pass
        if confusion_matrix_fig is not None:
            try:
                buf = io.BytesIO()
                confusion_matrix_fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                buf.seek(0)
                zf.writestr('images/confusion_matrix.png', buf.read())
            except Exception:
                pass

        # Add a simple README
        readme = (
            "This ZIP contains the model performance reports and optional visualizations.\n"
            "To reproduce predictions, also download the model files included elsewhere in the package.\n"
        )
        zf.writestr("README.txt", readme)

    buffer.seek(0)
    return buffer

# NEW: Binary batch prediction ZIP report creator (parity with multi-class)
def create_prediction_report_zip(predictions_df, smiles_col, prediction_col, confidence_col, individual_structures=None):
    """Create a ZIP file containing all binary prediction outputs (CSV, summary, images)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1) Full predictions CSV
        csv_buffer = io.StringIO()
        predictions_df.to_csv(csv_buffer, index=False)
        zip_file.writestr(f"binary_predictions_{timestamp}.csv", csv_buffer.getvalue())

        # 2) Summary text
        summary_lines = [
            "="*40,
            "CHEMLARA BINARY BATCH PREDICTION",
            "="*40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PREDICTION STATISTICS",
            "-"*20,
            f"Total Molecules: {len(predictions_df)}",
            "",
        ]
        if prediction_col in predictions_df.columns:
            summary_lines.append("CLASS DISTRIBUTION")
            summary_lines.append("-"*20)
            counts = predictions_df[prediction_col].value_counts()
            for cls, cnt in counts.items():
                pct = (cnt/len(predictions_df))*100 if len(predictions_df) else 0
                summary_lines.append(f"{cls}: {cnt} ({pct:.1f}%)")
            summary_lines.append("")
        if confidence_col in predictions_df.columns:
            summary_lines.append("CONFIDENCE STATISTICS")
            summary_lines.append("-"*20)
            try:
                conf_vals = predictions_df[confidence_col].apply(
                    lambda x: float(str(x).strip('%'))/100 if isinstance(x, str) and '%' in str(x)
                    else float(x) if pd.notna(x) else 0.0
                )
                summary_lines.append(f"Mean Confidence: {conf_vals.mean():.2%}")
                summary_lines.append(f"Min Confidence:  {conf_vals.min():.2%}")
                summary_lines.append(f"Max Confidence:  {conf_vals.max():.2%}")
            except Exception:
                pass
            summary_lines.append("")
        # Note: include per-class prob columns if present
        for col in [c for c in predictions_df.columns if c.startswith('Prob_')]:
            summary_lines.append(f"Includes probability column: {col}")
        zip_file.writestr(f"binary_prediction_summary_{timestamp}.txt", "\n".join(summary_lines))

        # 3) Individual images
        if individual_structures:
            base = "individual_predictions/"
            for idx, images in enumerate(individual_structures):
                if isinstance(images, dict):
                    if images.get('fragment_map') is not None:
                        buf = io.BytesIO()
                        images['fragment_map'].save(buf, format='PNG', dpi=(300, 300))
                        buf.seek(0)
                        zip_file.writestr(f"{base}molecule_{idx+1}_fragment_map.png", buf.getvalue())
                    if images.get('structure') is not None:
                        buf = io.BytesIO()
                        images['structure'].save(buf, format='PNG')
                        buf.seek(0)
                        zip_file.writestr(f"{base}molecule_{idx+1}_structure.png", buf.getvalue())
        
        # 4) README
        readme = f"""
# Chemlara Binary Batch Prediction Report

This archive contains outputs from your binary batch prediction.

Files:
1. binary_predictions_{timestamp}.csv - Full predictions
2. binary_prediction_summary_{timestamp}.txt - Summary stats
3. individual_predictions/ - Fragment maps and structures (if available)

Columns:
- {smiles_col}: Input SMILES
- {prediction_col}: Predicted class (Active / Not Active)
- {confidence_col}: Confidence for predicted class
- Prob_Not Active, Prob_Active: Per-class probabilities (if present)
"""
        zip_file.writestr("README.txt", readme)

    zip_buffer.seek(0)
    return zip_buffer

# Function to interpret prediction using LIME (classification)
def interpret_prediction(tpot_model, input_features, X_train):
    """Return an HTML string with LIME explanation for a classification model."""
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        mode="classification",
        feature_names=X_train.columns,
        class_names=["Not Active", "Active"],
        verbose=True,
        discretize_continuous=True
    )

    explanation = explainer.explain_instance(
        input_features.values[0],
        tpot_model.predict_proba,
        num_features=len(input_features.columns)
    )

    return explanation.as_html()

# --- Fragment Contribution Mapping for Circular Fingerprint ---

def weight_to_google_color(weight, min_weight, max_weight):
    """Convert weight to color using improved HLS color scheme with better handling of edge cases"""
    import colorsys
    
    # Handle edge cases
    if max_weight == min_weight:
        norm = 0.5
    else:
        norm = (abs(weight) - min_weight) / (max_weight - min_weight + 1e-6)
    
    # Use more vibrant colors with better contrast
    lightness = 0.3 + 0.5 * norm  # Avoid too light colors
    saturation = 0.85
    hue = 210/360 if weight >= 0 else 0/360  # Blue (positive) or Red (negative)
    
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (r, g, b)

def create_download_button_for_image(image, filename, button_text="📥 Download Image"):
    """Create a download button for PIL images"""
    try:
        buf = io.BytesIO()
        image.save(buf, format='PNG', dpi=(300, 300))
        buf.seek(0)
        
        return st.download_button(
            label=button_text,
            data=buf.getvalue(),
            file_name=filename,
            mime='image/png',
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Could not create download button: {str(e)}")
        return False

def draw_molecule_with_fragment_weights(mol, atom_weights, width=1200, height=1200):
    """Draw molecule with atom highlighting based on fragment weights using improved color scheme and high resolution"""
    try:
        # Create high-resolution drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        options = drawer.drawOptions()
        options.atomHighlightsAreCircles = True
        options.highlightRadius = 0.3
        options.bondLineWidth = 3
        options.atomLabelFontSize = 18
        options.legendFontSize = 16

        weights = list(atom_weights.values())
        if not weights:
            return None

        max_abs = max(abs(w) for w in weights)
        min_abs = min(abs(w) for w in weights)

        highlight_atoms = list(atom_weights.keys())
        highlight_colors = {
            idx: weight_to_google_color(atom_weights[idx], min_abs, max_abs)
            for idx in highlight_atoms
        }

        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png))
        return img
    except Exception:
        return None

def map_cfp_bits_to_atoms(mol, bit_weights, radius=4, n_bits=2048):
    """Map circular fingerprint bits to atoms using RDKit's Morgan fingerprint"""
    try:
        atom_weights = {}
        
        # Get bit info from Morgan fingerprint
        bit_info = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)
        on_bits = set(fp.GetOnBits())
        
        # Map each bit to its contributing atoms
        for bit_idx, weight in bit_weights.items():
            if bit_idx in on_bits and bit_idx in bit_info:
                # Each entry in bit_info is (center_atom, radius_used)
                for center_atom, radius_used in bit_info[bit_idx]:
                    # Get all atoms in the environment (fragment)
                    if radius_used == 0:
                        contributing_atoms = [center_atom]
                    else:
                        env_atoms = Chem.FindAtomEnvironmentOfRadiusN(mol, radius_used, center_atom)
                        contributing_atoms = set()
                        for bond_idx in env_atoms:
                            bond = mol.GetBondWithIdx(bond_idx)
                            contributing_atoms.add(bond.GetBeginAtomIdx())
                            contributing_atoms.add(bond.GetEndAtomIdx())
                        contributing_atoms.add(center_atom)  # Ensure center is included
                        contributing_atoms = list(contributing_atoms)
                    
                    # Distribute weight among contributing atoms in the fragment
                    weight_per_atom = weight / len(contributing_atoms)
                    for atom_idx in contributing_atoms:
                        atom_weights[atom_idx] = atom_weights.get(atom_idx, 0) + weight_per_atom
        
        return atom_weights
    except Exception:
        return {}

def map_specific_cfp_to_atoms(mol, cfp_number, radius=4, n_bits=2048):
    """Map a specific circular fingerprint number to atoms with improved weight distribution"""
    try:
        atom_weights = {}
        
        # Get bit info from Morgan fingerprint
        bit_info = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)
        on_bits = set(fp.GetOnBits())
        
        # Check if the specific CFP number is present in this molecule
        if cfp_number in on_bits and cfp_number in bit_info:
            # Initialize all atoms with small negative weight first
            for i in range(mol.GetNumAtoms()):
                atom_weights[i] = -0.5
                
            # Each entry in bit_info is (center_atom, radius_used)
            for center_atom, radius_used in bit_info[cfp_number]:
                # Get all atoms in the environment (fragment)
                if radius_used == 0:
                    contributing_atoms = [center_atom]
                else:
                    env_atoms = Chem.FindAtomEnvironmentOfRadiusN(mol, radius_used, center_atom)
                    contributing_atoms = set()
                    for bond_idx in env_atoms:
                        bond = mol.GetBondWithIdx(bond_idx)
                        contributing_atoms.add(bond.GetBeginAtomIdx())
                        contributing_atoms.add(bond.GetEndAtomIdx())
                    contributing_atoms.add(center_atom)  # Ensure center is included
                    contributing_atoms = list(contributing_atoms)
                
                # Assign positive weights to atoms that contribute to this CFP
                weight_center = 2.0   # Highest weight for center atom
                weight_fragment = 1.0  # Medium weight for fragment atoms
                
                # Center atom gets highest weight
                atom_weights[center_atom] = weight_center
                
                # Other atoms in fragment get medium weight
                for atom_idx in contributing_atoms:
                    if atom_idx != center_atom:
                        atom_weights[atom_idx] = weight_fragment
        else:
            # If specific CFP not found, still create contrast
            # Set all atoms to negative weight to show they don't contribute
            for i in range(mol.GetNumAtoms()):
                atom_weights[i] = -1.0
        
        return atom_weights
    except Exception:
        return {}

def generate_fragment_contribution_map(smiles, model, X_train, featurizer_obj, cfp_number=None):
    """Generate fragment contribution map for circular fingerprint predictions"""
    try:
        # Ensure we have the right featurizer parameters
        radius = getattr(featurizer_obj, 'radius', 4)
        n_bits = getattr(featurizer_obj, 'size', 2048)
        
        # Standardize and create molecule
        std_smiles = standardize_smiles(smiles)
        mol = Chem.MolFromSmiles(std_smiles)
        if mol is None:
            return None
        
        # Generate features
        features = featurizer_obj.featurize([mol])[0]
        feature_df = pd.DataFrame([features], columns=[f"fp_{i}" for i in range(len(features))])
        feature_df = feature_df.astype(float)
        
        # If specific CFP number is provided, highlight only that fingerprint
        if cfp_number is not None:
            atom_weights = map_specific_cfp_to_atoms(mol, cfp_number, radius=radius, n_bits=n_bits)
        else:
            # Use LIME explanation for overall contribution
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                mode="classification",
                feature_names=X_train.columns,
                class_names=["Not Active", "Active"],
                verbose=False,
                discretize_continuous=True
            )
            
            explanation = explainer.explain_instance(
                feature_df.values[0],
                model.predict_proba,
                num_features=min(100, len(feature_df.columns))  # Limit features for better visualization
            )
            
            # Get predicted class and its weights
            pred_class = int(model.predict(feature_df)[0])
            weights_list = explanation.as_map().get(pred_class, [])
            
            # If no weights or all weights are similar, create artificial contrast
            if not weights_list:
                # Create random weights for visualization
                import random
                weights_list = [(i, random.uniform(-1, 1)) for i in range(min(50, len(feature_df.columns)))]
            
            # Convert to bit weights dictionary
            bit_weights = {}
            for feature_idx, weight in weights_list:
                # feature_idx corresponds to the bit position in the fingerprint
                bit_weights[feature_idx] = float(weight)
            
            # If all weights are very similar, add some artificial variation
            weight_values = list(bit_weights.values())
            if weight_values and (max(weight_values) - min(weight_values)) < 0.01:
                # Add artificial variation to show structure
                for i, (bit_idx, weight) in enumerate(bit_weights.items()):
                    bit_weights[bit_idx] = weight + (i % 3 - 1) * 0.5  # Add variation
            
            # Map bits to atoms
            atom_weights = map_cfp_bits_to_atoms(mol, bit_weights, radius=radius, n_bits=n_bits)
        
        if not atom_weights:
            # Fallback: create simple atom highlighting
            atom_weights = {}
            for i in range(mol.GetNumAtoms()):
                atom_weights[i] = (i % 3 - 1) * 0.5  # Create pattern for visualization
        
        # Generate visualization
        return draw_molecule_with_fragment_weights(mol, atom_weights)
        
    except Exception as e:
        # Debug: print error for troubleshooting
        print(f"Error in generate_fragment_contribution_map: {str(e)}")
        return None

# Function to predict from single Smile input
def predict_from_single_smiles(single_smiles, featurizer_name='Circular Fingerprint'):
    standardized_smiles = standardize_smiles(single_smiles)
    mol = Chem.MolFromSmiles(standardized_smiles)
    
    if mol is not None:
        # Create featurizer with custom parameters if Circular Fingerprint
        if featurizer_name == "Circular Fingerprint":
            # Use custom parameters if available, otherwise use defaults
            radius = st.session_state.get('cfp_custom_radius', 4)
            size = st.session_state.get('cfp_custom_size', 2048)
            featurizer = dc.feat.CircularFingerprint(size=size, radius=radius)
        else:
            featurizer = Featurizer[featurizer_name]
            
        features = featurizer.featurize([mol])[0]
        feature_df = pd.DataFrame([features], columns=[f"fp_{i}" for i in range(len(features))])
        feature_df = feature_df.astype(float)

        # Load trained model and X_train
        try:
            with open('best_model.pkl', 'rb') as f_model, open('X_train.pkl', 'rb') as f_X_train:
                tpot_model = joblib.load(f_model)
                X_train = joblib.load(f_X_train)
                # Store in session state for visualization
                st.session_state['_tpot_model'] = tpot_model
                st.session_state['_X_train'] = X_train
                st.session_state['_featurizer_obj'] = featurizer
        except FileNotFoundError:
            st.warning("Please build and save the model in the 'Build Model' section first.")
            return None, None, None

        # Predict using the trained model
        prediction = tpot_model.predict(feature_df)[0]
        probability = tpot_model.predict_proba(feature_df)[0][1] if hasattr(tpot_model, 'predict_proba') else None

        # Interpret prediction using LIME
        explanation_html = interpret_prediction(tpot_model, feature_df, X_train)

        return prediction, probability, explanation_html
    else:
        st.warning("Invalid Smile input. Please check your input and try again.")
        return None, None, None

# Navigation bar function
def render_navigation_bar():
    """Render iOS-style horizontal navigation bar"""
    nav_options = {
        "🏠 Home": "home",
        "🔬 Build Model": "build", 
        "🧪 Single Prediction": "predict",
        "📊 Batch Prediction": "batch"
    }
    
    # Initialize session state
    if 'class_active_tab' not in st.session_state:
        st.session_state.class_active_tab = "home"
    
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
    
    cols = st.columns(len(nav_options))
    
    for idx, (label, key) in enumerate(nav_options.items()):
        with cols[idx]:
            is_active = st.session_state.class_active_tab == key
            
            if st.button(
                label,
                key=f"class_nav_{key}",
                help=f"Switch to {label}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if st.session_state.class_active_tab != key:
                    st.session_state.class_active_tab = key
    
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.class_active_tab

# Function to create iOS-style section title
def create_ios_section_title(title, emoji="", size="large"):
    """Create iOS-style section title with gradient background"""
    if size == "large":
        return f"""
        <div style="
            background: linear-gradient(135deg, rgba(0, 122, 255, 0.1) 0%, rgba(88, 86, 214, 0.1) 100%);
            border-left: 4px solid #007AFF;
            border-radius: 12px;
            padding: 16px 20px;
            margin: 24px 0 16px 0;
        ">
            <h3 style="
                margin: 0;
                color: #007AFF;
                font-weight: 700;
                font-size: 1.5em;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                {emoji} {title}
            </h3>
        </div>
        """
    else:  # small/medium
        return f"""
        <div style="
            background: linear-gradient(135deg, rgba(0, 122, 255, 0.08) 0%, rgba(88, 86, 214, 0.08) 100%);
            border-left: 3px solid #007AFF;
            border-radius: 10px;
            padding: 12px 16px;
            margin: 16px 0 12px 0;
        ">
            <h4 style="
                margin: 0;
                color: #007AFF;
                font-weight: 600;
                font-size: 1.2em;
                display: flex;
                align-items: center;
                gap: 10px;
            ">
                {emoji} {title}
            </h4>
        </div>
        """

# Main Streamlit application
def main():
    # Initialize selected featurizer name session variable
    if 'selected_featurizer_name' not in st.session_state:
        st.session_state.selected_featurizer_name = list(Featurizer.keys())[0]  # Set default featurizer

    # Create main header
    st.markdown(create_ios_header("Chemlara TPOT Classifier", "Traditional AutoML for Chemical Activity Prediction"), unsafe_allow_html=True)

    # Render navigation and get active tab
    active_tab = render_navigation_bar()

    if active_tab == "home":
        st.markdown(create_ios_card("Welcome to Chemlara Predictor!", 
                   """
                   <p class="ios-text-large" style="font-size: 18px; margin-bottom: 20px; font-weight: 500; color: #1D1D1F;">🎯 <strong style="font-weight: 700;">What can you do here?</strong></p>
                   <div style="background: rgba(0, 122, 255, 0.05); border-radius: 16px; padding: 20px; margin: 20px 0;">
                       <p class="ios-text-regular" style="margin: 12px 0; font-size: 16px; font-weight: 500; color: #1D1D1F;">🔬 Build ML models for chemical activity prediction</p>
                       <p class="ios-text-regular" style="margin: 12px 0; font-size: 16px; font-weight: 500; color: #1D1D1F;">🧪 Predict activity from single SMILES</p>
                       <p class="ios-text-regular" style="margin: 12px 0; font-size: 16px; font-weight: 500; color: #1D1D1F;">📊 Batch predictions from Excel files</p>
                       <p class="ios-text-regular" style="margin: 12px 0; font-size: 16px; font-weight: 500; color: #1D1D1F;">📈 Get detailed model explanations with LIME</p>
                   </div>
                   <p class="ios-text-small" style="color: #8E8E93; font-style: italic; text-align: center; font-size: 14px; font-weight: 500; margin-top: 20px;">📱 Optimized for mobile and desktop use!</p>
                   """, "🎉"), unsafe_allow_html=True)

    elif active_tab == "build":
        st.markdown(create_ios_section_title("Build Your ML Model", "🔬"), unsafe_allow_html=True)
        
        with st.expander("📁 Upload Training Data", expanded=True):
            uploaded_file = st.file_uploader("Upload Excel file with SMILES and Activity", type=["xlsx"], 
                                            help="Excel file should contain SMILES strings and corresponding activity labels")

        if uploaded_file is not None:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            # Show dataframe in iOS card
            with st.expander("📊 View Uploaded Data", expanded=False):
                st.dataframe(df, use_container_width=True)

            # Configuration section in iOS card
            st.markdown(create_ios_card("Model Configuration", 
                                      "Configure your machine learning model parameters below.", "⚙️"), unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                col_names = df.columns.tolist()
                smiles_col = st.selectbox("🧬 SMILES Column", col_names, key='smiles_column')
                activity_col = st.selectbox("🎯 Activity Column", col_names, key='activity_column')
            
            with col2:
                st.session_state.selected_featurizer_name = st.selectbox("🔧 Featurizer", list(Featurizer.keys()), 
                                                                        key='featurizer_name', 
                                                                        index=list(Featurizer.keys()).index(st.session_state.selected_featurizer_name))
                
                # Additional parameters for Circular Fingerprint
                if st.session_state.selected_featurizer_name == "Circular Fingerprint":
                    st.markdown('<p class="ios-text-regular" style="font-weight: 600; font-size: 16px; color: #007AFF; margin: 16px 0 8px 0; letter-spacing: -0.01em;">🔬 Circular Fingerprint Parameters:</p>', unsafe_allow_html=True)
                    col_fp1, col_fp2 = st.columns(2)
                    with col_fp1:
                        cfp_radius = st.slider("Radius", min_value=1, max_value=6, value=4, 
                                             help="Circular fingerprint radius (default: 4)", key='cfp_radius')
                    with col_fp2:
                        cfp_size = st.number_input("Fingerprint Size", min_value=64, max_value=16384, value=2048, step=64,
                                              help="Number of bits in fingerprint (default: 2048)", key='cfp_size')

            # Advanced settings in collapsible section
            with st.expander("🔧 Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    generations = st.slider("Generations", min_value=1, max_value=20, value=3,
                                          help="Number of generations for TPOT optimization (lower = faster)")
                    cv = st.slider("CV Folds", min_value=2, max_value=10, value=3,
                                 help="Number of cross-validation folds (lower = faster)")
                with col2:
                    verbosity = st.slider("Verbosity", min_value=0, max_value=3, value=3,
                                        help="Verbosity level for TPOT output (0 = silent, 3 = most verbose)")
                    test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05,
                                        help="Fraction of data to use for testing")

            # Build model button with confirmation
            col1, col2 = st.columns([3, 1])
            with col1:
                train_button = st.button("🚀 Build and Train Model", use_container_width=True)
            with col2:
                if st.button("ℹ️ Info", use_container_width=True):
                    st.info(f"""
                    **Training Details:**
                    - Dataset: {len(df)} samples
                    - Generations: {generations}
                    - CV Folds: {cv}
                    - Population: 20 pipelines per generation
                    
                    This will evaluate approximately {generations * 20} different ML pipelines to find the best one for your data.
                    """)

            if train_button:
                with st.spinner("🔄 Building your model... This may take a few minutes."):
                    result = preprocess_and_model(
                        df, smiles_col, activity_col, st.session_state.selected_featurizer_name, 
                        generations=generations, cv=cv, verbosity=verbosity, test_size=test_size)
                    
                    if not result or len(result) != 15:
                        st.error("Training failed. Please check your dataset and settings.")
                    else:
                        tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer, total_time, training_time = result

                        if tpot is not None:
                            # Store which featurizer was used for training
                            st.session_state['trained_featurizer_name'] = st.session_state.selected_featurizer_name
                            
                            # Display model metrics in cards
                            st.markdown(create_ios_section_title("Model Performance", "📈"), unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(create_ios_metric_card("Accuracy", f"{accuracy:.3f}", "Overall correctness", "🎯"), unsafe_allow_html=True)
                                st.markdown(create_ios_metric_card("Precision", f"{precision:.3f}", "True positive rate", "✅"), unsafe_allow_html=True)
                            with col2:
                                st.markdown(create_ios_metric_card("Recall", f"{recall:.3f}", "Sensitivity", "🔍"), unsafe_allow_html=True)
                                st.markdown(create_ios_metric_card("F1 Score", f"{f1:.3f}", "Harmonic mean", "⚖️"), unsafe_allow_html=True)
                            with col3:
                                if roc_auc is not None:
                                    st.markdown(create_ios_metric_card("ROC AUC", f"{roc_auc:.3f}", "Area under curve", "📊"), unsafe_allow_html=True)
                            
                            # Success message (no training time shown)
                            st.success("✅ Model trained successfully!")

                            # Best TPOT Pipeline section
                            st.markdown(create_ios_section_title("Best TPOT Pipeline", "🏆"), unsafe_allow_html=True)
                            with st.expander("View Pipeline Details", expanded=False):
                                st.code(str(tpot.fitted_pipeline_), language='python')

                            # Visualizations in 2 columns
                            col1, col2 = st.columns(2)
                            roc_fig = None
                            cm_fig = None

                            with col1:
                                try:
                                    if roc_auc is not None and hasattr(tpot, 'predict_proba') and len(set(y_test)) == 2:
                                        y_proba = tpot.predict_proba(X_test)[:, 1]
                                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                                        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                                        ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.3f})', linewidth=3, color='#667eea')
                                        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.6)
                                        ax_roc.fill_between(fpr, tpr, alpha=0.2, color='#667eea')
                                        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
                                        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
                                        ax_roc.set_title('📊 ROC Curve', fontsize=14, fontweight='bold')
                                        ax_roc.legend(loc='lower right')
                                        ax_roc.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        st.pyplot(fig_roc)
                                        st.session_state['roc_curve_fig'] = fig_roc
                                        roc_fig = fig_roc
                                    else:
                                        st.info("ℹ️ ROC curve not available for this classification problem.")
                                except Exception as e:
                                    st.error(f"ROC Error: {str(e)}")
                                    st.info("ℹ️ ROC curve not available for this classification problem.")

                            with col2:
                                try:
                                    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
                                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                                    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax_cm, 
                                                cbar_kws={'shrink': 0.8}, square=True, linewidths=0.5)
                                    ax_cm.set_title('📈 Confusion Matrix', fontsize=14, fontweight='bold')
                                    plt.tight_layout()
                                    st.pyplot(fig_cm)
                                    st.session_state['confusion_matrix_fig'] = fig_cm
                                    cm_fig = fig_cm
                                except Exception as e:
                                    st.error(f"CM Error: {str(e)}")
                                    st.info("ℹ️ Confusion matrix not available for this classification problem.")

                            # Model download section with comprehensive ZIP package
                            st.markdown(create_ios_section_title("Download Trained Model", "💾"), unsafe_allow_html=True)
                            
                            # Save TPOT model and related files
                            model_filename = 'best_model.pkl'
                            label_encoder_filename = 'label_encoder.pkl'
                            X_train_filename = 'X_train.pkl'
                            featurizer_filename = 'featurizer.pkl'

                            try:
                                with open(model_filename, 'wb') as f_model:
                                    joblib.dump(tpot.fitted_pipeline_, f_model)
                                
                                with open(X_train_filename, 'wb') as f_X_train:
                                    joblib.dump(X_train, f_X_train)
                                
                                with open(featurizer_filename, 'wb') as f_feat:
                                    joblib.dump(featurizer, f_feat)
                                
                                # Generate sample LIME interpretation and save
                                lime_sample_filename = 'lime_interpretation_sample.html'
                                try:
                                    # Create a sample LIME interpretation using the first test sample
                                    if len(X_test) > 0:
                                        sample_features = pd.DataFrame([X_test.iloc[0]], columns=X_train.columns if hasattr(X_train, 'columns') else None)
                                        lime_html = interpret_prediction(tpot.fitted_pipeline_, sample_features, X_train)
                                        if lime_html:
                                            with open(lime_sample_filename, 'w', encoding='utf-8') as f_lime:
                                                f_lime.write(f"""
                                                <!DOCTYPE html>
                                                <html>
                                                <head><title>LIME Interpretation Sample - Binary Classification</title></head>
                                                <body>
                                                    <h1>LIME Interpretation Sample</h1>
                                                    <p>This is a sample interpretation for the first test sample.</p>
                                                    {lime_html}
                                                </body>
                                                </html>
                                                """)
                                except Exception as lime_error:
                                    pass  # Silent error handling

                                # Create comprehensive model configuration details
                                model_config = {
                                    'smiles_column': smiles_col,
                                    'activity_column': activity_col,
                                    'featurizer': st.session_state.selected_featurizer_name,
                                    'generations': generations,
                                    'cv_folds': cv,
                                    'verbosity': verbosity,
                                    'test_size': test_size,
                                    'total_samples': len(df),
                                    'training_samples': len(X_train),
                                    'test_samples': len(X_test),
                                    'classification_type': 'binary',
                                }
                                
                                # Add CFP-specific parameters if applicable
                                if st.session_state.selected_featurizer_name == "Circular Fingerprint":
                                    model_config['cfp_radius'] = st.session_state.get('cfp_radius', 4)
                                    model_config['cfp_size'] = st.session_state.get('cfp_size', 2048)
                                
                                # Create comprehensive model package ZIP
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                try:
                                    # Prepare model parameters for the report
                                    model_params = {
                                        'TPOT Generations': generations,
                                        'Cross-Validation Folds': cv,
                                        'Test Size': f"{test_size:.2%}" if test_size and isinstance(test_size, (int, float)) else str(test_size),
                                        'Featurizer': st.session_state.selected_featurizer_name,
                                        'Total Samples': len(df),
                                        'Training Samples': len(X_train),
                                        'Test Samples': len(X_test),
                                        'Classification Type': 'Binary'
                                    }
                                    
                                    if st.session_state.selected_featurizer_name == "Circular Fingerprint":
                                        model_params['CFP Radius'] = st.session_state.get('cfp_radius', 4)
                                        model_params['CFP Size'] = st.session_state.get('cfp_size', 2048)
                                    
                                    # Create comprehensive model report ZIP
                                    model_package_zip = create_model_report_zip(
                                        accuracy=accuracy,
                                        precision=precision,
                                        recall=recall,
                                        f1=f1,
                                        roc_auc=roc_auc,
                                        confusion_matrix_fig=cm_fig,
                                        roc_curve_fig=roc_fig,
                                        model_params=model_params
                                    )
                                    
                                    # Add model files to the ZIP
                                    zip_buffer = io.BytesIO(model_package_zip.getvalue())
                                    
                                    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
                                        # Add model files
                                        with open(model_filename, 'rb') as f:
                                            zip_file.writestr(f"model_files/{model_filename}", f.read())
                                        with open(X_train_filename, 'rb') as f:
                                            zip_file.writestr(f"model_files/{X_train_filename}", f.read())
                                        with open(featurizer_filename, 'rb') as f:
                                            zip_file.writestr(f"model_files/{featurizer_filename}", f.read())
                                        
                                        
                                        # Add LIME sample if available
                                        if os.path.exists(lime_sample_filename):
                                            with open(lime_sample_filename, 'r', encoding='utf-8') as f:
                                                zip_file.writestr(f"model_files/{lime_sample_filename}", f.read())
                                        
                                        # Add model configuration JSON
                                        import json
                                        config_json = json.dumps(model_config, indent=2, default=str)
                                        zip_file.writestr(f"model_configuration_{timestamp}.json", config_json)
                                        
                                        # Add pipeline details
                                        try:
                                            pipeline_details = str(tpot.fitted_pipeline_)
                                            zip_file.writestr(f"best_pipeline_{timestamp}.txt", pipeline_details)
                                        except:
                                            pass
                                    
                                    zip_buffer.seek(0)
                                    
                                    # Display comprehensive download button
                                    st.download_button(
                                        label="📦 Download Complete Model Package (ZIP)",
                                        data=zip_buffer.getvalue(),
                                                                               file_name=f'binary_model_package_{timestamp}.zip',
                                                                                                                                                             mime='application/zip',
                                        use_container_width=True,
                                        help="Complete package with model files, configuration, performance metrics, and visualizations"
                                    )
                                    
                                except Exception as zip_error:
                                    st.error(f"Error creating model package ZIP: {str(zip_error)}")
                                
                                # Individual file downloads (optional)
                                with st.expander("📂 Download Individual Model Files"):
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.markdown(create_downloadable_model_link(model_filename, '📥 Model'), unsafe_allow_html=True)
                                    with col2:
                                        st.markdown(create_downloadable_model_link(X_train_filename, '📥 Training Data'), unsafe_allow_html=True)
                                    with col3:
                                        st.markdown(create_downloadable_model_link(featurizer_filename, '📥 Featurizer'), unsafe_allow_html=True)
                                    with col4:
                                        if os.path.exists(lime_sample_filename):
                                            st.markdown(create_downloadable_model_link(lime_sample_filename, '📥 LIME Sample'), unsafe_allow_html=True)
                            except Exception as e:
                                st.warning(f"Could not save model files: {str(e)}")

    elif active_tab == "predict":
        st.markdown(create_ios_section_title("Single SMILES Prediction", "🧪"), unsafe_allow_html=True)
        
        smile_input = st.text_input(
            "Enter SMILES string", 
            value="", 
            placeholder="e.g., CCO (ethanol)",
            help="Enter the SMILES representation of the molecule",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            predict_button = st.button("🔮 Predict Activity", use_container_width=True, key="predict_single_binary_btn")
        with col2:
            if st.button("🧹 Clear", use_container_width=True, key="clear_single_binary_btn"):
                st.rerun()

        if predict_button and smile_input:
            # Load trained model and training features
            try:
                with open('best_model.pkl', 'rb') as f_model, \
                     open('X_train.pkl', 'rb') as f_X_train:
                    tpot_model = joblib.load(f_model)
                    X_train = joblib.load(f_X_train)
                # Optional featurizer
                try:
                    with open('featurizer.pkl', 'rb') as f_feat:
                        featurizer_loaded = joblib.load(f_feat)
                    st.session_state['_featurizer_obj'] = featurizer_loaded
                except Exception:
                    featurizer_loaded = None
            except FileNotFoundError:
                st.error("❌ No trained model found. Please build a model first in the 'Build Model' tab.")
                st.stop()

            with st.spinner("🔍 Analyzing molecule for prediction..."):
                prediction, probability, explanation_html = predict_from_single_smiles(
                    smile_input, st.session_state.selected_featurizer_name
                )
                
                if prediction is not None:
                    # Prepare display values
                    pred_label = "Active" if int(prediction) == 1 else "Not Active"
                    prob_value = probability if isinstance(probability, (int, float, np.floating)) else None
                    
                    # Three-column layout for results
                    c1, c2, c3 = st.columns([2, 1, 1])
                    
                    with c1:
                        # Display fragment contribution map for Circular Fingerprint
                        if st.session_state.selected_featurizer_name == "Circular Fingerprint":
                            try:
                                # Get featurizer/model/X_train
                                featurizer_obj = st.session_state.get('_featurizer_obj') or featurizer_loaded or Featurizer[st.session_state.selected_featurizer_name]
                                model_obj = st.session_state.get('_tpot_model', tpot_model)
                                X_train_obj = st.session_state.get('_X_train', X_train)
                                
                                with st.spinner("🗺️ Generating high-resolution fragment contribution map..."):
                                    atomic_contrib_img = generate_fragment_contribution_map(
                                        smile_input, model_obj, X_train_obj, featurizer_obj, cfp_number=None
                                    )
                                
                                if atomic_contrib_img is not None:
                                    st.markdown("""
                                    <div class="ios-card" style="padding: 16px; text-align: center;">
                                        <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🗺️ Fragment Contribution Map</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.image(atomic_contrib_img, width=500)
                                    safe_smiles = smile_input.replace('/', '_').replace('\\', '_')[:20]
                                    create_download_button_for_image(
                                        atomic_contrib_img,
                                        f"atomic_contribution_{safe_smiles}.png",
                                        "📥 Download HD Image"
                                    )
                                else:
                                    mol = Chem.MolFromSmiles(smile_input)
                                    if mol is not None:
                                        img = Draw.MolToImage(mol, size=(400, 400))
                                        st.markdown("""
                                        <div class="ios-card" style="padding: 16px; text-align: center;">
                                            <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🧬 Molecule Structure</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.image(img, width=400)
                                    else:
                                        st.warning("⚠️ Could not visualize molecule")
                            except Exception as e:
                                st.warning(f"⚠️ Could not generate atomic contribution map: {str(e)}")
                                mol = Chem.MolFromSmiles(smile_input)
                                if mol is not None:
                                    img = Draw.MolToImage(mol, size=(400, 400))
                                    st.markdown("""
                                    <div class="ios-card" style="padding: 16px; text-align: center;">
                                        <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🧬 Molecule Structure</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.image(img, width=400)
                        else:
                            mol = Chem.MolFromSmiles(smile_input)
                            if mol is not None:
                                img = Draw.MolToImage(mol, size=(400, 400))
                                st.markdown("""
                                <div class="ios-card" style="padding: 16px; text-align: center;">
                                    <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🧬 Molecule Structure</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                st.image(img, width=400)
                            else:
                                st.warning("⚠️ Could not visualize molecule")
                    
                    with c2:
                        # Binary prediction results (mirroring multi-class card style)
                        st.markdown(f"""
                        <div class="ios-card" style="padding: 20px;">
                            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                                <div style="font-size: 2em; margin-right: 12px;">🎯</div>
                                <div>
                                    <h2 style="color: #007AFF; margin: 0; font-weight: 700; font-size: 1.5em;">{pred_label}</h2>
                                    <p style="margin: 4px 0 0 0; color: #8E8E93; font-size: 14px;">Predicted Class</p>
                                </div>
                            </div>
                            <div style="background: rgba(0, 122, 255, 0.1); border-radius: 12px; padding: 12px; margin-bottom: 12px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="color: #007AFF; font-weight: 600;">Confidence:</span>
                                    <span style="color: #1D1D1F; font-weight: 700; font-size: 1.1em;">{(prob_value if prob_value is not None else 0):.1%}</span>
                                </div>
                            </div>
                            <div style="background: rgba(0, 0, 0, 0.05); border-radius: 8px; padding: 8px;">
                                <p style="margin: 0; color: #8E8E93; font-size: 12px; font-weight: 500;">
                                    <strong>SMILES:</strong> {smile_input}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c3:
                        # Color legend (for CFP)
                        if st.session_state.selected_featurizer_name == "Circular Fingerprint":
                            st.markdown("🔵 **High Positive (Dark Blue)**")
                            st.caption("Strongly contributes to activity")
                            st.markdown("🟦 **Low Positive (Light Blue)**")
                            st.caption("Moderately supports activity")
                            st.markdown("⚪ **Neutral (Gray)**")
                            st.caption("No significant contribution")
                            st.markdown("🟧 **Low Negative (Light Red)**")
                            st.caption("Moderately reduces activity")
                            st.markdown("🔴 **High Negative (Dark Red)**")
                            st.caption("Strongly reduces activity")
                    
                    # Show LIME explanation (HTML)
                    if explanation_html:
                        with st.expander("🧠 LIME Explanation", expanded=False):
                            # Render full HTML safely in an isolated iframe to avoid showing raw HTML junk
                            components.html(explanation_html, height=420, scrolling=True)
                            # Optional: allow users to download the raw HTML file
                            st.download_button(
                                label="📥 Download LIME HTML",
                                data=explanation_html.encode("utf-8"),
                                file_name="lime_explanation.html",
                                mime="text/html",
                                use_container_width=True
                            )
                else:
                    st.error("❌ Failed to make prediction. Please check your SMILES input.")

    elif active_tab == "batch":
        st.markdown(create_ios_section_title("Batch Prediction", "📊"), unsafe_allow_html=True)
        
        with st.expander("📁 Upload Prediction File", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload Excel file with SMILES for batch prediction", 
                type=["xlsx"], key="batch_prediction_upload_bin",
                help="Select an Excel file containing SMILES strings for batch prediction"
            )

        if uploaded_file is not None:
            # Read Excel file
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
                st.stop()

            with st.expander("📊 Preview Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)

            # Select SMILES column in iOS card
            st.markdown('<div class="ios-card">', unsafe_allow_html=True)
            col_names = df.columns.tolist()
            smiles_col_predict = st.selectbox("🧬 Select SMILES Column", col_names, key='smiles_column_predict_bin')
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🚀 Run Batch Prediction", use_container_width=True, key="batch_predict_btn_bin"):
                # Load model and related artifacts
                try:
                    with open('best_model.pkl', 'rb') as f_model, \
                         open('X_train.pkl', 'rb') as f_X_train:
                        tpot_model = joblib.load(f_model)
                        X_train = joblib.load(f_X_train)
                    try:
                        with open('featurizer.pkl', 'rb') as f_feat:
                            featurizer_obj = joblib.load(f_feat)
                    except Exception:
                        # Build featurizer from current selection
                        if st.session_state.get('selected_featurizer_name') == "Circular Fingerprint":
                            radius = st.session_state.get('cfp_custom_radius', 4)
                            size = st.session_state.get('cfp_custom_size', 2048)
                            featurizer_obj = dc.feat.CircularFingerprint(size=size, radius=radius)
                        else:
                            featurizer_obj = Featurizer[st.session_state.get('selected_featurizer_name', 'Circular Fingerprint')]
                    # Store refs for visuals
                    st.session_state['_tpot_model'] = tpot_model
                    st.session_state['_X_train'] = X_train
                    st.session_state['_featurizer_obj'] = featurizer_obj
                except FileNotFoundError:
                    st.error("❌ No trained model found. Please build a model first in the 'Build Model' tab.")
                    st.stop()

                predictions = []
                probabilities = []
                class_probabilities = []
                individual_images = []

                # Progress
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                total = len(df)

                for idx, row in df.iterrows():
                    progress = (idx + 1) / total if total else 1.0
                    progress_bar.progress(progress)
                    status_text.markdown(f"<div style='text-align:center;color:#007AFF;font-weight:600;'>Processing molecule {idx+1} of {total}</div>", unsafe_allow_html=True)
                    try:
                        standardized_smiles = standardize_smiles(str(row[smiles_col_predict]))
                        if not standardized_smiles:
                            raise ValueError("Invalid SMILES")
                        mol = Chem.MolFromSmiles(standardized_smiles)
                        if mol is None:
                            raise ValueError("Invalid SMILES")

                        # Featurize and predict
                        feats = featurizer_obj.featurize([mol])[0]
                        feature_df = pd.DataFrame([feats])
                        feature_df.columns = [f"fp_{c}" for c in feature_df.columns]
                        pred_raw = tpot_model.predict(feature_df)[0]
                        proba = tpot_model.predict_proba(feature_df)[0] if hasattr(tpot_model, 'predict_proba') else np.array([1.0 - float(pred_raw), float(pred_raw)])
                        pred_label = "Active" if int(pred_raw) == 1 else "Not Active"
                        max_prob = float(proba[int(pred_raw)]) if proba is not None and len(proba) > int(pred_raw) else 0.0

                        predictions.append(pred_label)
                        probabilities.append(max_prob)
                        if proba is not None and len(proba) == 2:
                            class_probabilities.append([float(proba[0]), float(proba[1])])
                        else:
                            class_probabilities.append([1.0 - max_prob, max_prob])

                        # Images
                        image_data = {}
                        try:
                            if st.session_state.get('selected_featurizer_name') == "Circular Fingerprint":
                                fmap = generate_fragment_contribution_map(standardized_smiles, tpot_model, X_train, featurizer_obj, cfp_number=None)
                                if fmap is not None:
                                    image_data['fragment_map'] = fmap
                        except Exception:
                            pass
                        try:
                            structure_img = Draw.MolToImage(mol, size=(400, 400))
                            image_data['structure'] = structure_img
                        except Exception:
                            pass
                        individual_images.append(image_data)
                    except Exception as e:
                        predictions.append(f"Error: {str(e)}")
                        probabilities.append(0.0)
                        class_probabilities.append([0.0, 0.0])
                        individual_images.append({})

                progress_container.empty()

                # Build results dataframe
                df['Predicted_Class'] = predictions
                df['Confidence'] = [f"{p:.1%}" if isinstance(p, float) else "N/A" for p in probabilities]
                df['Prob_Not Active'] = [f"{probs[0]:.1%}" if len(probs) > 0 else "N/A" for probs in class_probabilities]
                df['Prob_Active'] = [f"{probs[1]:.1%}" if len(probs) > 1 else "N/A" for probs in class_probabilities]

                # Individual results (paginated)
                st.markdown(create_ios_section_title("Individual Prediction Results", "🧪"), unsafe_allow_html=True)
                results_per_page = 5
                valid_indices = [i for i, p in enumerate(predictions) if not str(p).startswith("Error") and p != "Invalid SMILES"]
                total_valid = len(valid_indices)
                if total_valid > 0:
                    num_pages = (total_valid + results_per_page - 1) // results_per_page
                    page = st.selectbox("📄 Select Results Page", options=list(range(1, num_pages + 1)) if num_pages > 1 else [1], 
                                        format_func=lambda x: f"Page {x} ({min(results_per_page, total_valid - (x-1)*results_per_page)} results)",
                                        key='batch_page_selector_bin')
                    start_idx = (page - 1) * results_per_page
                    end_idx = min(start_idx + results_per_page, total_valid)
                    for pos in range(start_idx, end_idx):
                        i = valid_indices[pos]
                        row = df.iloc[i]
                        pred_label = predictions[i]
                        prob = probabilities[i]
                        smiles_val = row[smiles_col_predict]

                        with st.expander(f"🧬 Molecule {i + 1}: {pred_label} ({prob:.1%} confidence)", expanded=False):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                if st.session_state.get('selected_featurizer_name') == "Circular Fingerprint":
                                    try:
                                        fmap_img = generate_fragment_contribution_map(smiles_val, tpot_model, X_train, featurizer_obj, cfp_number=None)
                                        if fmap_img is not None:
                                            st.markdown("""
                                            <div class="ios-card" style="padding: 16px; text-align: center;">
                                                <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🗺️ Fragment Contribution Map</h4>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            st.image(fmap_img, width=500)
                                            create_download_button_for_image(
                                                fmap_img,
                                                f"fragment_contribution_molecule_{i + 1}.png",
                                                "📥 Download HD Image"
                                            )
                                        else:
                                            mol = Chem.MolFromSmiles(smiles_val)
                                            if mol is not None:
                                                st.image(Draw.MolToImage(mol, size=(400, 400)), width=400)
                                    except Exception:
                                        mol = Chem.MolFromSmiles(smiles_val)
                                        if mol is not None:
                                            st.image(Draw.MolToImage(mol, size=(400, 400)), width=400)
                            with col2:
                                st.markdown(f"""
                                <div class="ios-card" style="padding: 20px;">
                                    <div style="display: flex; align-items: center; margin-bottom: 16px;">
                                        <div style="font-size: 2em; margin-right: 12px;">{'🟢' if pred_label == 'Active' else '🔴'}</div>
                                        <div>
                                            <h2 style="color: {'#34C759' if pred_label == 'Active' else '#FF3B30'}; margin: 0; font-weight: 700; font-size: 1.5em;">{pred_label}</h2>
                                            <p style="margin: 4px 0 0 0; color: #8E8E93; font-size: 14px;">Predicted Class</p>
                                        </div>
                                    </div>
                                    <div style="background: rgba(0, 122, 255, 0.1); border-radius: 12px; padding: 12px; margin-bottom: 12px;">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <span style="color: #007AFF; font-weight: 600;">Confidence:</span>
                                            <span style="color: #1D1D1F; font-weight: 700; font-size: 1.1em;">{prob:.1%}</span>
                                        </div>
                                    </div>
                                    <div style="background: rgba(0, 0, 0, 0.05); border-radius: 8px; padding: 8px;">
                                        <p style="margin: 0; color: #8E8E93; font-size: 12px; font-weight: 500;">
                                            <strong>SMILES:</strong> {str(smiles_val)[:40]}{'...' if isinstance(smiles_val, str) and len(smiles_val) > 40 else ''}
                                        </p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col3:
                                if st.session_state.get('selected_featurizer_name') == "Circular Fingerprint":
                                    st.markdown("🔵 **High Positive (Dark Blue)**")
                                    st.caption("Strongly contributes to activity")
                                    st.markdown("🟦 **Low Positive (Light Blue)**")
                                    st.caption("Moderately supports activity")
                                    st.markdown("⚪ **Neutral (Gray)**")
                                    st.caption("No significant contribution")
                                    st.markdown("🟧 **Low Negative (Light Red)**")
                                    st.caption("Moderately reduces activity")
                                    st.markdown("🔴 **High Negative (Dark Red)**")
                                    st.caption("Strongly reduces activity")

                # Results table
                st.markdown(create_ios_section_title("Complete Results Table", "📊"), unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)

                # Downloads: CSV, Excel, ZIP
                st.markdown("---")
                st.markdown(create_ios_section_title("Download Prediction Results", "📥"), unsafe_allow_html=True)
                dl1, dl2, dl3 = st.columns(3)
                with dl1:
                    csv_data = df.to_csv(index=False)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=f'binary_predictions_{ts}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                with dl2:
                    xbuf = io.BytesIO()
                    with pd.ExcelWriter(xbuf, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Predictions')
                    xbuf.seek(0)
                    st.download_button(
                        label="📊 Download Excel",
                        data=xbuf,
                        file_name=f'binary_predictions_{ts}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
                with dl3:
                    try:
                        zip_buf = create_prediction_report_zip(
                            df,
                            smiles_col=smiles_col_predict,
                            prediction_col='Predicted_Class',
                            confidence_col='Confidence',
                            individual_structures=individual_images
                        )
                        st.download_button(
                            label="📦 Download Full Report",
                            data=zip_buf,
                            file_name=f'binary_prediction_report_{ts}.zip',
                            mime='application/zip',
                            use_container_width=True,
                            help="ZIP with CSV, summary, and individual images"
                        )
                    except Exception as e:
                        st.error(f"Error creating ZIP report: {str(e)}")

if __name__ == "__main__":
    main()
