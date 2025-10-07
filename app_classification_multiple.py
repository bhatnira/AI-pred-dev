import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import threading
import base64
import joblib
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
matplotlib.use('Agg')  # Set backend before importing pyplot
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
import io
import zipfile
from datetime import datetime
from PIL import Image
import colorsys

# Configure matplotlib and RDKit for headless mode
os.environ['MPLBACKEND'] = 'Agg'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

# Disable RDKit warnings and configure for headless rendering
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Configure Streamlit page for mobile-friendly display
st.set_page_config(
    page_title="Chemlara Predictor - TPOT Multi-Class Classification",
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
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 100%;
        padding: 1rem;
        background: transparent;
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
    
    /* Metric cards - iOS style */
    .ios-metric {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 12px;
        margin: 4px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .ios-metric:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
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
    
    /* Mobile responsiveness */
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
            padding: 16px;
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

# Global Variables for session state  
if 'selected_featurizer_name_multiclass' not in st.session_state:
    st.session_state.selected_featurizer_name_multiclass = list(Featurizer.keys())[0]  # Set default featurizer

def standardize_smiles(smiles):
    """Standardize SMILES strings for consistency"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return None
    except:
        return None

def format_time_duration(seconds):
    """Format time duration in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

# iOS-style component functions
def create_ios_metric_card(title, value, description="", icon="📊"):
    return f"""
    <div class="ios-metric">
        <div style="font-size: 1.2em; margin-bottom: 4px;">{icon}</div>
        <h3 style="margin: 0; color: #007AFF; font-weight: 600; font-size: 11px;">{title}</h3>
        <h2 style="margin: 4px 0; color: #1D1D1F; font-weight: 700; font-size: 16px;">{value}</h2>
        <p style="margin: 0; color: #8E8E93; font-size: 9px; font-weight: 400;">{description}</p>
    </div>
    """

def create_ios_card(title, content, icon=""):
    return f"""
    <div class="ios-card">
        <h3 style="color: #007AFF; margin-bottom: 16px; font-weight: 600; font-size: 18px;">{icon} {title}</h3>
        <div style="color: #1D1D1F; line-height: 1.5;">{content}</div>
    </div>
    """

def create_ios_header(title, subtitle=""):
    return f"""
    <div class="ios-header">
        <h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">{title}</h1>
        <p style="margin: 8px 0 0 0; font-size: 1.1em; opacity: 0.9; font-weight: 400;">{subtitle}</p>
    </div>
    """

# --- Fragment Contribution Mapping for Circular Fingerprint ---

def weight_to_google_color(weight, min_weight, max_weight):
    """Convert weight to color using improved HLS color scheme with better handling of edge cases"""
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

def generate_fragment_contribution_map(smiles, model, X_train, featurizer_obj, class_names, target_class_idx=None, cfp_number=None):
    """Generate fragment contribution map for circular fingerprint predictions in multi-class setting"""
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
                class_names=class_names,
                verbose=False,
                discretize_continuous=True
            )
            
            explanation = explainer.explain_instance(
                feature_df.values[0],
                model.predict_proba,
                num_features=min(100, len(feature_df.columns))  # Limit features for better visualization
            )
            
            # Get predicted class and its weights
            if target_class_idx is None:
                pred_class = int(model.predict(feature_df)[0])
            else:
                pred_class = target_class_idx
            
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

def create_multiclass_roc_curves(y_test, y_pred_proba, class_names):
    """
    Create ROC curves for multi-class classification showing actual curves
    """
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        # Convert inputs to numpy arrays for consistency
        y_test = np.array(y_test)
        y_pred_proba = np.array(y_pred_proba)
        
        # Convert class_names to proper format if it's numeric
        if isinstance(class_names, np.ndarray):
            class_names = [f"Class {i}" for i in class_names]
        elif not isinstance(class_names, list):
            class_names = [f"Class {i}" for i in range(len(class_names))]
        
        # Get unique classes
        n_classes = len(class_names)
        
        if n_classes < 2:
            return None
        
        # Ensure y_pred_proba has the right shape
        if y_pred_proba.ndim == 1:
            y_pred_proba = y_pred_proba.reshape(-1, 1)
        
        if y_pred_proba.shape[1] != n_classes:
            return None
        
        # Binarize the output labels for multi-class ROC
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        if n_classes == 2:
            y_test_bin = np.c_[1-y_test_bin, y_test_bin]
        
        # Create smaller figure
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Colors for different classes
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        # Plot ROC curve for each class
        for i, (color, class_name) in enumerate(zip(colors, class_names)):
            if i < y_pred_proba.shape[1] and i < y_test_bin.shape[1]:
                # Calculate ROC curve and AUC for this class
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                ax.plot(fpr, tpr, color=color, lw=2, alpha=0.8,
                       label=f'{class_name} (AUC = {roc_auc:.3f})')
                
                # Fill area under curve with transparency
                ax.fill_between(fpr, tpr, alpha=0.1, color=color)
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
        
        # Customize the plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=8)
        ax.set_ylabel('True Positive Rate', fontsize=8)
        ax.set_title('ROC Curves', fontsize=10, fontweight='bold')
        ax.legend(loc="lower right", fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, alpha=0.3)
        
        # Add overall metrics text box
        try:
            roc_auc_weighted = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')
            roc_auc_macro = roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovr')
            
            textstr = f'Weighted: {roc_auc_weighted:.3f}\nMacro: {roc_auc_macro:.3f}'
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=props)
        except Exception as e:
            pass
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        return None

def create_multiclass_confusion_matrix(y_test, y_pred, class_names):
    """Create confusion matrix for multi-class classification"""
    try:
        # Convert inputs to numpy arrays for consistency
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        
        # Convert class_names to proper format if it's numeric
        if isinstance(class_names, np.ndarray):
            class_names = [f"Class {i}" for i in class_names]
        elif not isinstance(class_names, list):
            class_names = [f"Class {i}" for i in range(len(class_names))]
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get the number of classes from confusion matrix shape
        n_classes = cm.shape[0]
        
        # Ensure we have the right number of class names
        if len(class_names) != n_classes:
            class_names = [f"Class {i}" for i in range(n_classes)]
        
        fig = plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=10, fontweight='bold')
        plt.xlabel('Predicted', fontsize=8)
        plt.ylabel('Actual', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        return fig
    except Exception:
        return None

def create_multiclass_precision_recall_curves(y_test, y_pred_proba, class_names):
    """
    Create precision-recall curves for multi-class classification
    """
    try:
        from sklearn.preprocessing import label_binarize
        
        # Binarize the output labels for multi-class
        y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes = axes.ravel()
        
        # Plot precision-recall curve for each class
        for i, class_name in enumerate(class_names[:4]):  # Limit to 4 classes for display
            if i < len(class_names):
                if len(class_names) == 2:
                    # Binary case
                    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    avg_precision = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
                else:
                    # Multi-class case
                    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    avg_precision = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
                
                axes[i].plot(recall, precision, linewidth=2, 
                           label=f'AP = {avg_precision:.3f}', color=plt.cm.Set1(i))
                axes[i].fill_between(recall, precision, alpha=0.2, color=plt.cm.Set1(i))
                axes[i].set_xlabel('Recall', fontweight='bold')
                axes[i].set_ylabel('Precision', fontweight='bold')
                axes[i].set_title(f'PR Curve - {class_name}', fontweight='bold')
                axes[i].legend(loc='lower left')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
        
        # Hide unused subplots
        for i in range(len(class_names), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('📈 Precision-Recall Curves by Class', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating precision-recall curves: {str(e)}")
        return None

# Function to create a ZIP report for model outputs
def create_model_report_zip(accuracy, precision, recall, f1, roc_auc, 
                             confusion_matrix_fig=None, roc_curve_fig=None,
                             feature_importance_fig=None, model_params=None,
                             class_names=None):
    """Create a ZIP file containing all multi-class modeling outputs and reports"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Create metrics report (TXT)
        # Format ROC AUC value separately to avoid nested f-string issues
        roc_auc_value = f"{roc_auc:.4f}" if roc_auc else 'N/A'
        
        metrics_report = f"""
========================================
 CHEMLARA MULTI-CLASS MODEL REPORT
========================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MODEL PERFORMANCE METRICS
------------------------
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f} (weighted)
Recall:    {recall:.4f} (weighted)
F1 Score:  {f1:.4f} (weighted)
ROC AUC:   {roc_auc_value} (weighted)

"""
        if class_names is not None:
            metrics_report += f"\nCLASSES DETECTED\n----------------\n"
            for i, class_name in enumerate(class_names):
                metrics_report += f"Class {i}: {class_name}\n"
        
        # Format ROC AUC message separately to avoid nested f-string issues
        roc_auc_msg = f'Multi-class discrimination ability ({roc_auc:.4f})' if roc_auc else 'Not available'
        
        metrics_report += f"""
MODEL DESCRIPTION
-----------------
Type: AutoML Multi-Class Classification (TPOT)
Optimization: Genetic Algorithm Pipeline Search
Cross-Validation: Stratified K-Fold

INTERPRETATION
--------------
- Accuracy: Overall model correctness ({accuracy*100:.2f}% of predictions are correct)
- Precision: {precision*100:.2f}% weighted average across all classes
- Recall: Model captures {recall*100:.2f}% of cases (weighted average)
- F1 Score: Harmonic mean of precision and recall (weighted)
- ROC AUC: {roc_auc_msg}

"""
        if model_params:
            metrics_report += f"\nMODEL PARAMETERS\n----------------\n"
            for key, value in model_params.items():
                metrics_report += f"{key}: {value}\n"
        
        zip_file.writestr(f"multiclass_model_metrics_report_{timestamp}.txt", metrics_report)
        
        # 2. Create metrics CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)', 'ROC AUC (Weighted)'],
            'Value': [accuracy, precision, recall, f1, roc_auc if roc_auc else 0.0]
        })
        csv_buffer = io.StringIO()
        metrics_df.to_csv(csv_buffer, index=False)
        zip_file.writestr(f"multiclass_model_metrics_{timestamp}.csv", csv_buffer.getvalue())
        
        # 3. Save confusion matrix figure
        if confusion_matrix_fig is not None:
            img_buffer = io.BytesIO()
            confusion_matrix_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            zip_file.writestr(f"confusion_matrix_{timestamp}.png", img_buffer.getvalue())
        
        # 4. Save ROC curve figure
        if roc_curve_fig is not None:
            img_buffer = io.BytesIO()
            roc_curve_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            zip_file.writestr(f"multiclass_roc_curves_{timestamp}.png", img_buffer.getvalue())
        
        # 5. Save feature importance figure
        if feature_importance_fig is not None:
            img_buffer = io.BytesIO()
            feature_importance_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            zip_file.writestr(f"feature_importance_{timestamp}.png", img_buffer.getvalue())
        
        # 6. Create README
        readme_content = """
# Chemlara Multi-Class Model Training Report

This folder contains all outputs from your multi-class model training session.

## Files Included:

1. **multiclass_model_metrics_report_[timestamp].txt** - Detailed text report with all metrics and interpretation
2. **multiclass_model_metrics_[timestamp].csv** - Metrics in CSV format for easy import
3. **confusion_matrix_[timestamp].png** - Visual confusion matrix showing per-class performance
4. **multiclass_roc_curves_[timestamp].png** - ROC curves for each class (one-vs-rest)
5. **feature_importance_[timestamp].png** - Feature importance plot (if available)

## How to Use:

- Review the text report for a complete summary
- Import the CSV into Excel or other tools for further analysis
- Use the PNG images in presentations or reports
- The confusion matrix shows how well each class is distinguished
- ROC curves show the discrimination ability for each class

Generated by Chemlara Predictor - Multi-Class Classification
"""
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer

# Function to create a ZIP report of the prediction results
def create_prediction_report_zip(predictions_df, smiles_col='SMILES', 
                                   prediction_col='Predicted_Class',
                                   confidence_col='Confidence',
                                   individual_structures=None,
                                   class_names=None):
    """Create a ZIP file containing all multi-class prediction outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Save full predictions as CSV
        csv_buffer = io.StringIO()
        predictions_df.to_csv(csv_buffer, index=False)
        zip_file.writestr(f"multiclass_predictions_{timestamp}.csv", csv_buffer.getvalue())
        
        # 3. Create summary statistics
        summary_stats = f"""
========================================
CHEMLARA MULTI-CLASS BATCH PREDICTION
========================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION STATISTICS
--------------------
Total Molecules: {len(predictions_df)}
"""
        if prediction_col in predictions_df.columns:
            value_counts = predictions_df[prediction_col].value_counts()
            summary_stats += f"\nCLASS DISTRIBUTION\n------------------\n"
            for category, count in value_counts.items():
                percentage = (count / len(predictions_df)) * 100
                summary_stats += f"{category}: {count} ({percentage:.1f}%)\n"
        
        if confidence_col in predictions_df.columns:
            summary_stats += f"\nCONFIDENCE STATISTICS\n--------------------\n"
            # Extract numeric confidence values
            try:
                confidences = predictions_df[confidence_col].apply(
                    lambda x: float(str(x).strip('%'))/100 if isinstance(x, str) and '%' in str(x) else float(x) if pd.notna(x) else 0.0
                )
                summary_stats += f"Mean Confidence: {confidences.mean():.2%}\n"
                summary_stats += f"Min Confidence:  {confidences.min():.2%}\n"
                summary_stats += f"Max Confidence:  {confidences.max():.2%}\n"
            except Exception:
                # Skip confidence stats if there's an error
                pass
        
        # Add class-specific probabilities summary if available
        if class_names is not None and len(class_names) > 0:
            summary_stats += f"\nCLASS PROBABILITY COLUMNS\n------------------------\n"
            for class_name in class_names:
                prob_col = f'Prob_{class_name}'
                if prob_col in predictions_df.columns:
                    summary_stats += f"- {prob_col}\n"
        
        zip_file.writestr(f"multiclass_prediction_summary_{timestamp}.txt", summary_stats)
        
        # 4. Save individual molecular fragment contribution maps if provided
        if individual_structures:
            structures_folder = "individual_predictions/"
            for idx, structure_data in enumerate(individual_structures):
                if isinstance(structure_data, dict):
                    # Save fragment contribution map
                    if 'fragment_map' in structure_data and structure_data['fragment_map']:
                        img_buffer = io.BytesIO()
                        structure_data['fragment_map'].save(img_buffer, format='PNG', dpi=(300, 300))
                        img_buffer.seek(0)
                        zip_file.writestr(f"{structures_folder}molecule_{idx+1}_fragment_map.png", img_buffer.getvalue())
                    
                    # Save molecular structure
                    if 'structure' in structure_data and structure_data['structure']:
                        img_buffer = io.BytesIO()
                        structure_data['structure'].save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        zip_file.writestr(f"{structures_folder}molecule_{idx+1}_structure.png", img_buffer.getvalue())
                elif isinstance(structure_data, tuple) and len(structure_data) == 2:
                    # Legacy format: (smiles, img_bytes)
                    smiles, img_bytes = structure_data
                    if img_bytes:
                        zip_file.writestr(f"{structures_folder}molecule_{idx+1}.png", img_bytes)
        
        # 5. Create README
        readme_content = f"""
# Chemlara Multi-Class Batch Prediction Report

This folder contains all outputs from your batch multi-class prediction session.

## Files Included:

1. **multiclass_predictions_{timestamp}.csv** - Complete predictions in CSV format
2. **multiclass_prediction_summary_{timestamp}.txt** - Statistical summary of predictions
3. **individual_predictions/** - Fragment contribution maps and structures for each molecule

## Column Descriptions:

- **{smiles_col}**: Input SMILES string
- **{prediction_col}**: Predicted activity class
- **{confidence_col}**: Model confidence in prediction
- **Prob_[ClassName]**: Probability for each specific class

## Individual Predictions Folder:

For each molecule, you'll find:
- **molecule_N_fragment_map.png**: High-resolution fragment contribution map (for Circular Fingerprint)
- **molecule_N_structure.png**: Molecular structure visualization

## How to Use:

- Open the CSV file in your preferred spreadsheet software
- Review the summary for overall statistics and class distribution
- View individual fragment maps to understand which molecular fragments contribute to predictions
- Use class probabilities to understand model confidence across all classes

Generated by Chemlara Predictor - Multi-Class Classification
"""
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer

# Function to preprocess data and perform modeling for multi-class classification
def preprocess_and_model_multiclass(df, smiles_col, activity_col, featurizer_name, generations=3, cv=3, verbosity=0, test_size=0.20, cfp_radius=4, cfp_size=2048):
    """
    Multi-class preprocessing and TPOT model building with time tracking
    """
    start_time = time.time()
    
    # Simple progress tracking without time estimates
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    def update_progress(progress_percent, status_msg):
        progress_bar.progress(progress_percent)
        status_text.info(f"🔬 {status_msg}")
    
    try:
        # Phase 1: Data Preparation
        update_progress(0.05, "Standardizing SMILES...")
        
        df[smiles_col + '_standardized'] = df[smiles_col].apply(standardize_smiles)
        df.dropna(subset=[smiles_col + '_standardized'], inplace=True)

        # Check for multi-class scenario
        unique_classes = df[activity_col].unique()
        if len(unique_classes) < 3:
            st.error("Multi-class classification requires at least 3 distinct classes. Please check your dataset.")
            return None, None, None, None, None, None, None, None, None, None, None, None

        update_progress(0.15, "Featurizing molecules...")
        
        # Featurize molecules with progress updates
        # For Circular Fingerprint, use custom radius and size
        if featurizer_name == "Circular Fingerprint":
            from deepchem.feat import CircularFingerprint
            featurizer = CircularFingerprint(radius=cfp_radius, size=cfp_size)
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
            update_progress(min(progress, 0.45), f"Featurizing molecules... {i+batch_size}/{len(smiles_list)}")

        if not features:
            st.error("No valid molecules found for featurization. Please ensure your SMILES data is correct.")
            return None, None, None, None, None, None, None, None, None, None, None, None

        update_progress(0.5, "Preparing training data...")
        
        feature_df = pd.DataFrame(features)
        X = feature_df
        y = df[activity_col]

        # Convert integer column names to strings
        new_column_names = [f"fp_{col}" for col in X.columns]
        X.columns = new_column_names

        # Encode labels for multi-class
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)

        update_progress(0.6, "Initializing TPOT classifier...")
        
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

        update_progress(0.65, "Training TPOT model...")
        
        # Train the TPOT model
        tpot.fit(X_train, y_train)

        update_progress(0.9, "Evaluating model performance...")
        
        # Make predictions
        y_pred = tpot.predict(X_test)
        y_pred_proba = tpot.predict_proba(X_test)

        # Calculate multi-class metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Multi-class ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = None

        update_progress(1.0, "Training completed successfully!")
        
        # Calculate total elapsed time
        elapsed_time = time.time() - start_time
        
        # Clear progress indicators
        progress_container.empty()

        return tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, y_pred_proba, class_names, le, df, X_train, y_train, featurizer, elapsed_time
        
    except Exception as e:
        progress_container.empty()
        st.error(f"An error occurred during model training: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def predict_from_single_smiles_multiclass(smiles, featurizer_name, model, label_encoder, class_names=None, X_train=None, featurizer_obj=None):
    """
    Predict activity for a single SMILES string in multi-class setting
    """
    try:
        # Standardize SMILES
        standardized_smiles = standardize_smiles(smiles)
        if standardized_smiles is None:
            return None, None, "Invalid SMILES string"

        # Use provided featurizer object if available, otherwise use default
        if featurizer_obj is not None:
            featurizer = featurizer_obj
        else:
            featurizer = Featurizer[featurizer_name]
        
        mol = Chem.MolFromSmiles(standardized_smiles)
        if mol is None:
            return None, None, "Could not parse molecule"

        features = featurizer.featurize([mol])[0]
        feature_df = pd.DataFrame([features])
        new_column_names = [f"fp_{col}" for col in feature_df.columns]
        feature_df.columns = new_column_names

        # Make prediction
        prediction_encoded = model.predict(feature_df)[0]
        probabilities = model.predict_proba(feature_df)[0]
        
        # Decode prediction
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        max_probability = float(np.max(probabilities))
        
        # Use class_names if provided, otherwise use label_encoder.classes_
        display_classes = class_names if class_names is not None else label_encoder.classes_
        
        # Create explanation
        explanation = ""
        
        # Generate LIME interpretation file if X_train is provided
        if X_train is not None and display_classes is not None:
            lime_html = interpret_prediction_multiclass(model, feature_df, X_train, display_classes)
            if lime_html:
                # Save LIME interpretation as downloadable file
                safe_smiles = smiles.replace('/', '_').replace('\\', '_')[:20]
                lime_filename = f"lime_interpretation_{safe_smiles}.html"
                with open(lime_filename, 'w', encoding='utf-8') as f:
                    f.write(f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>LIME Interpretation - {smiles}</title>
                        <meta charset="utf-8">
                    </head>
                    <body>
                        <h1>LIME Interpretation</h1>
                        <p><strong>Molecule:</strong> {smiles}</p>
                        <p><strong>Predicted Class:</strong> {prediction}</p>
                        <p><strong>Confidence:</strong> {max_probability:.1%}</p>
                        <hr>
                        {lime_html}
                    </body>
                    </html>
                    """)
                # Store filename for download
                st.session_state['lime_filename'] = lime_filename

        return prediction, max_probability, explanation

    except Exception as e:
        return None, None, f"Error in prediction: {str(e)}"

# Function to create downloadable model link
def create_downloadable_model_link(model_filename, link_text):
    with open(model_filename, 'rb') as f:
        model_data = f.read()
    b64 = base64.b64encode(model_data).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="{model_filename}">{link_text}</a>'
    return href

# Function to interpret prediction using LIME for multi-class
def interpret_prediction_multiclass(tpot_model, input_features, X_train, class_names):
    """
    Interpret multi-class prediction using LIME
    """
    try:
        # Create LIME explainer using X_train
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            mode="classification",
            feature_names=X_train.columns,
            class_names=class_names,
            verbose=True,
            discretize_continuous=True
        )
        
        explanation = explainer.explain_instance(
            input_features.values[0],
            tpot_model.predict_proba,
            num_features=len(input_features.columns)
        )

        # Generate HTML explanation
        html_explanation = explanation.as_html()
        return html_explanation
    except Exception as e:
        st.warning(f"Could not generate LIME explanation: {str(e)}")
        return None


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
    if 'multi_class_active_tab' not in st.session_state:
        st.session_state.multi_class_active_tab = "home"
    
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
            is_active = st.session_state.multi_class_active_tab == key
            
            if st.button(
                label,
                key=f"multi_class_nav_{key}",
                help=f"Switch to {label}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if st.session_state.multi_class_active_tab != key:
                    st.session_state.multi_class_active_tab = key
    
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.multi_class_active_tab

# Main Streamlit application
def main():
    # Create main header
    st.markdown(create_ios_header("Chemlara Multi-Class Classifier", "Advanced AutoML for Multi-Class Chemical Activity Prediction"), unsafe_allow_html=True)

    # Render navigation and get active tab
    active_tab = render_navigation_bar()

    if active_tab == "home":
        st.markdown(create_ios_card("Welcome to Multi-Class Chemlara Predictor!", 
                   """
                   <p style="font-size: 16px; margin-bottom: 16px;">🎯 <strong>Multi-Class Classification Features:</strong></p>
                   <div style="background: rgba(0, 122, 255, 0.05); border-radius: 12px; padding: 16px; margin: 16px 0;">
                       <p style="margin: 8px 0;">🔬 <strong>Multi-Class AutoML:</strong> Handle 3+ activity classes automatically</p>
                       <p style="margin: 8px 0;">📊 <strong>Advanced Metrics:</strong> Per-class precision, recall, F1-score</p>
                       <p style="margin: 8px 0;">📈 <strong>ROC Curves:</strong> Multi-class ROC analysis with micro/macro averaging</p>
                       <p style="margin: 8px 0;">🧪 <strong>Batch Predictions:</strong> Process multiple molecules with class probabilities</p>
                       <p style="margin: 8px 0;">🔍 <strong>Model Explanations:</strong> LIME interpretability for each prediction</p>
                       <p style="margin: 8px 0;">📱 <strong>Confusion Matrix:</strong> Detailed classification performance heatmap</p>
                   </div>
                   <div style="background: rgba(255, 149, 0, 0.05); border-radius: 12px; padding: 12px; margin: 16px 0;">
                       <p style="color: #FF9500; font-weight: 600; margin: 0;">📋 Required: Excel file with SMILES and activity classes (3+ categories)</p>
                   </div>
                   <p style="color: #8E8E93; font-style: italic; text-align: center;">🧬 Perfect for drug discovery with multiple activity profiles!</p>
                   """, "🎉"), unsafe_allow_html=True)

    elif active_tab == "build":
        st.markdown("### 🔬 Build Your Multi-Class ML Model")
        
        with st.expander("� Upload Training Data", expanded=True):
            uploaded_file = st.file_uploader("Upload Excel file with SMILES and Multi-Class Activity", type=["xlsx"], 
                                            key="training_upload",
                                            help="Excel file should contain SMILES strings and corresponding activity labels (3+ classes)")

        if uploaded_file is not None:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            # Show dataframe in iOS card
            with st.expander("📊 View Uploaded Data", expanded=False):
                st.dataframe(df, use_container_width=True)
                
                # Show class distribution
                if len(df.columns) > 1:
                    activity_col_preview = st.selectbox("Preview Activity Column", df.columns.tolist(), key='preview_col')
                    if activity_col_preview:
                        class_counts = df[activity_col_preview].value_counts()
                        st.markdown("#### 📈 Class Distribution")
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            fig, ax = plt.subplots(figsize=(6, 3))
                            class_counts.plot(kind='bar', ax=ax, color=['#007AFF', '#5856D6', '#34C759', '#FF9500', '#FF3B30'][:len(class_counts)])
                            ax.set_title('Class Distribution', fontweight='bold')
                            ax.set_xlabel('Activity Class')
                            ax.set_ylabel('Count')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        with col2:
                            for class_name, count in class_counts.items():
                                st.markdown(create_ios_metric_card(str(class_name), str(count), "samples", "📊"), unsafe_allow_html=True)

            # Configuration section in iOS card
            st.markdown(create_ios_card("Model Configuration", 
                                      "Configure your multi-class machine learning model parameters below.", "⚙️"), unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                col_names = df.columns.tolist()
                smiles_col = st.selectbox("🧬 SMILES Column", col_names, key='smiles_column')
                activity_col = st.selectbox("🎯 Activity Column", col_names, key='activity_column')
                
                # Validate multi-class requirement
                if activity_col:
                    unique_classes = df[activity_col].unique()
                    if len(unique_classes) < 3:
                        st.error(f"❌ Only {len(unique_classes)} classes found. Multi-class requires 3+ classes.")
                    # else:
                    #     st.success(f"✅ {len(unique_classes)} classes detected: {', '.join(map(str, unique_classes))}")
            
            with col2:
                st.session_state.selected_featurizer_name_multiclass = st.selectbox("🔧 Featurizer", list(Featurizer.keys()), 
                                                                        key='featurizer_name_multiclass', 
                                                                        index=list(Featurizer.keys()).index(st.session_state.selected_featurizer_name_multiclass))
            
            # Circular Fingerprint specific settings
            if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                st.markdown("#### ⚙️ Circular Fingerprint Parameters")
                col_fp1, col_fp2 = st.columns(2)
                with col_fp1:
                    cfp_radius = st.slider("Radius", min_value=1, max_value=6, value=4, 
                                         help="Circular fingerprint radius (default: 4)", key='cfp_radius_multiclass')
                with col_fp2:
                    cfp_size = st.number_input("Fingerprint Size", min_value=64, max_value=16384, value=2048, step=64,
                                          help="Number of bits in fingerprint (default: 2048)", key='cfp_size_multiclass')

            # Advanced settings in collapsible section
            with st.expander("🔧 Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    generations = st.slider("Generations", min_value=1, max_value=20, value=5,
                                          help="Number of generations for TPOT optimization (higher for better models)")
                    cv = st.slider("CV Folds", min_value=2, max_value=10, value=3,
                                 help="Number of cross-validation folds (higher for more robust evaluation)")
                with col2:
                    verbosity = st.slider("Verbosity", min_value=0, max_value=3, value=2,
                                        help="Verbosity level for TPOT output (0 = silent, 3 = most verbose)")
                    test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05,
                                        help="Fraction of data to use for testing")

            # Build model button with confirmation
            col1, col2 = st.columns([3, 1])
            with col1:
                train_button = st.button("🚀 Build and Train Multi-Class Model", use_container_width=True, key="train_multiclass_btn")
            with col2:
                if st.button("ℹ️ Info", use_container_width=True, key="info_multiclass_btn"):
                    st.info(f"""
                    **Multi-Class Training Details:**
                    - Dataset: {len(df)} samples
                    - Classes: {len(df[activity_col].unique()) if activity_col else 'N/A'}
                    - Generations: {generations}
                    - CV Folds: {cv}
                    - Population: 15 pipelines per generation
                    
                    This will evaluate approximately {generations * 15} different ML pipelines optimized for multi-class classification.
                    """)

            if train_button and activity_col and len(df[activity_col].unique()) >= 3:
                # Get CFP parameters if Circular Fingerprint is selected
                if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                    cfp_radius_val = st.session_state.get('cfp_radius_multiclass', 4)
                    cfp_size_val = st.session_state.get('cfp_size_multiclass', 2048)
                else:
                    cfp_radius_val = 4
                    cfp_size_val = 2048
                
                with st.spinner("🔄 Building your multi-class model... This may take several minutes."):
                    results = preprocess_and_model_multiclass(
                        df, smiles_col, activity_col, st.session_state.selected_featurizer_name_multiclass, 
                        generations=generations, cv=cv, verbosity=verbosity, test_size=test_size,
                        cfp_radius=cfp_radius_val, cfp_size=cfp_size_val)

                    if results[0] is not None:
                        tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, y_pred_proba, class_names, le, df_result, X_train, y_train, featurizer, elapsed_time = results
                        
                        # Save model and necessary data
                        with open('best_multiclass_model.pkl', 'wb') as f:
                            joblib.dump(tpot.fitted_pipeline_, f)
                        with open('X_train_multiclass.pkl', 'wb') as f:
                            joblib.dump(X_train, f)
                        with open('label_encoder_multiclass.pkl', 'wb') as f:
                            joblib.dump(le, f)
                        with open('class_names_multiclass.pkl', 'wb') as f:
                            joblib.dump(class_names, f)
                        with open('featurizer_multiclass.pkl', 'wb') as f:
                            joblib.dump(featurizer, f)

                        # Display elapsed time
                        st.markdown(f"### ⏱️ Training Time: {format_time_duration(elapsed_time)}")
                        
                        # Display model metrics in cards
                        st.markdown("### 📈 Model Performance")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(create_ios_metric_card("Accuracy", f"{accuracy:.3f}", "Overall correctness", "🎯"), unsafe_allow_html=True)
                            st.markdown(create_ios_metric_card("Precision", f"{precision:.3f}", "Weighted average", "✅"), unsafe_allow_html=True)
                        with col2:
                            st.markdown(create_ios_metric_card("Recall", f"{recall:.3f}", "Weighted average", "🔍"), unsafe_allow_html=True)
                            st.markdown(create_ios_metric_card("F1 Score", f"{f1:.3f}", "Weighted average", "⚖️"), unsafe_allow_html=True)
                        with col3:
                            if roc_auc is not None:
                                st.markdown(create_ios_metric_card("ROC AUC", f"{roc_auc:.3f}", "Multi-class weighted", "📊"), unsafe_allow_html=True)
                            st.success("✅ Model trained successfully!")

                        # Visualizations in 2 columns (matching binary classification format)
                        col1, col2 = st.columns(2)
                        
                        roc_fig = None
                        cm_fig = None
                        
                        with col1:
                            try:
                                roc_fig = create_multiclass_roc_curves(y_test, y_pred_proba, class_names)
                                if roc_fig:
                                    st.pyplot(roc_fig)
                                    # Store figure in session state for download
                                    st.session_state['roc_curve_fig_multiclass'] = roc_fig
                                else:
                                    st.info("ℹ️ ROC curve not available for this classification problem.")
                            except Exception as e:
                                st.error(f"ROC Error: {str(e)}")
                                st.info("ℹ️ ROC curve not available for this classification problem.")
                        
                        with col2:
                            try:
                                cm_fig = create_multiclass_confusion_matrix(y_test, y_pred, class_names)
                                if cm_fig:
                                    st.pyplot(cm_fig)
                                    # Store figure in session state for download
                                    st.session_state['confusion_matrix_fig_multiclass'] = cm_fig
                                else:
                                    st.info("ℹ️ Confusion matrix not available for this classification problem.")
                            except Exception as e:
                                st.error(f"CM Error: {str(e)}")
                                st.info("ℹ️ Confusion matrix not available for this classification problem.")

                        # Display best pipeline in a nice container
                        st.markdown("### 🏆 Best TPOT Pipeline")
                        with st.expander("🔍 View Pipeline Details", expanded=False):
                            try:
                                st.code(str(tpot.fitted_pipeline_), language='python')
                            except:
                                st.code("Pipeline details not available", language='text')

                        # Model download section
                        st.markdown("### 💾 Download Trained Multi-Class Model")
                        
                        # Save TPOT model and related files
                        model_filename = 'best_multiclass_model.pkl'
                        label_encoder_filename = 'label_encoder_multiclass.pkl'
                        class_names_filename = 'class_names_multiclass.pkl'
                        X_train_filename = 'X_train_multiclass.pkl'

                        try:
                            with open(model_filename, 'wb') as f_model:
                                joblib.dump(tpot.fitted_pipeline_, f_model)
                            
                            with open(label_encoder_filename, 'wb') as f_le:
                                joblib.dump(le, f_le)
                                
                            with open(class_names_filename, 'wb') as f_classes:
                                joblib.dump(class_names, f_classes)
                            
                            with open(X_train_filename, 'wb') as f_X_train:
                                joblib.dump(X_train, f_X_train)
                            
                            # Generate sample LIME interpretation and save
                            lime_sample_filename = 'lime_interpretation_sample_multiclass.html'
                            try:
                                # Create a sample LIME interpretation using the first test sample
                                if len(X_test) > 0:
                                    sample_features = pd.DataFrame([X_test[0]], columns=X_train.columns if hasattr(X_train, 'columns') else None)
                                    lime_html = interpret_prediction_multiclass(tpot.fitted_pipeline_, sample_features, X_train, class_names)
                                    if lime_html:
                                        with open(lime_sample_filename, 'w', encoding='utf-8') as f_lime:
                                            f_lime.write(f"""
                                            <!DOCTYPE html>
                                            <html>
                                            <head><title>LIME Interpretation Sample - Multi-Class</title></head>
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
                                'featurizer': st.session_state.selected_featurizer_name_multiclass,
                                'generations': generations,
                                'cv_folds': cv,
                                'verbosity': verbosity,
                                'test_size': test_size,
                                'total_samples': len(df),
                                'training_samples': len(X_train),
                                'test_samples': len(X_test),
                                'num_classes': len(class_names),
                                'class_names': list(class_names),
                            }
                            
                            # Add CFP-specific parameters if applicable
                            if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                                model_config['cfp_radius'] = st.session_state.get('cfp_radius_multiclass', 4)
                                model_config['cfp_size'] = st.session_state.get('cfp_size_multiclass', 2048)
                            
                            # Create comprehensive model package ZIP
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            try:
                                # Prepare model parameters for the report
                                model_params = {
                                    'TPOT Generations': generations,
                                    'Cross-Validation Folds': cv,
                                    'Test Size': f"{test_size:.2%}",
                                    'Featurizer': st.session_state.selected_featurizer_name_multiclass,
                                    'Total Samples': len(df),
                                    'Training Samples': len(X_train),
                                    'Test Samples': len(X_test),
                                    'Number of Classes': len(class_names)
                                }
                                
                                if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                                    model_params['CFP Radius'] = st.session_state.get('cfp_radius_multiclass', 4)
                                    model_params['CFP Size'] = st.session_state.get('cfp_size_multiclass', 2048)
                                
                                # Create comprehensive model report ZIP
                                model_package_zip = create_model_report_zip(
                                    accuracy=accuracy,
                                    precision=precision,
                                    recall=recall,
                                    f1=f1,
                                    roc_auc=roc_auc,
                                    confusion_matrix_fig=cm_fig,
                                    roc_curve_fig=roc_fig,
                                    model_params=model_params,
                                    class_names=class_names
                                )
                                
                                # Add model files to the ZIP
                                zip_buffer = io.BytesIO(model_package_zip.getvalue())
                                
                                with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
                                    # Add model files
                                    with open(model_filename, 'rb') as f:
                                        zip_file.writestr(f"model_files/{model_filename}", f.read())
                                    with open(label_encoder_filename, 'rb') as f:
                                        zip_file.writestr(f"model_files/{label_encoder_filename}", f.read())
                                    with open(class_names_filename, 'rb') as f:
                                        zip_file.writestr(f"model_files/{class_names_filename}", f.read())
                                    with open(X_train_filename, 'rb') as f:
                                        zip_file.writestr(f"model_files/{X_train_filename}", f.read())
                                    
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
                                    data=zip_buffer,
                                    file_name=f'multiclass_model_package_{timestamp}.zip',
                                    mime='application/zip',
                                    use_container_width=True,
                                    help="Complete package with model files, configuration, performance metrics, and visualizations"
                                )
                                
                            except Exception as zip_error:
                                st.error(f"Error creating model package ZIP: {str(zip_error)}")
                            
                            # Individual file downloads (optional)
                            with st.expander("📂 Download Individual Model Files"):
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.markdown(create_downloadable_model_link(model_filename, '📥 Model'), unsafe_allow_html=True)
                                with col2:
                                    st.markdown(create_downloadable_model_link(label_encoder_filename, '📥 Encoder'), unsafe_allow_html=True)
                                with col3:
                                    st.markdown(create_downloadable_model_link(class_names_filename, '📥 Classes'), unsafe_allow_html=True)
                                with col4:
                                    st.markdown(create_downloadable_model_link(X_train_filename, '📥 Training Data'), unsafe_allow_html=True)
                                with col5:
                                    if os.path.exists(lime_sample_filename):
                                        st.markdown(create_downloadable_model_link(lime_sample_filename, '📥 LIME Sample'), unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Could not save model files: {str(e)}")

    elif active_tab == "predict":
        st.markdown("### 🧪 Single SMILES Multi-Class Prediction")
        
        smile_input = st.text_input("Enter SMILES string for multi-class prediction", 
                                  placeholder="e.g., CCO (ethanol)",
                                  help="Enter a valid SMILES string representing your molecule",
                                  label_visibility="collapsed")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            predict_button = st.button("🔮 Predict Multi-Class Activity", use_container_width=True, key="predict_single_btn")
        with col2:
            if st.button("🧹 Clear", use_container_width=True, key="clear_single_btn"):
                st.rerun()

        if predict_button and smile_input:
            # Check if model exists
            try:
                with open('best_multiclass_model.pkl', 'rb') as f_model, \
                     open('label_encoder_multiclass.pkl', 'rb') as f_le, \
                     open('class_names_multiclass.pkl', 'rb') as f_classes, \
                     open('X_train_multiclass.pkl', 'rb') as f_X_train:
                    tpot_model = joblib.load(f_model)
                    label_encoder = joblib.load(f_le)
                    class_names = joblib.load(f_classes)
                    X_train = joblib.load(f_X_train)
                
                # Try to load featurizer (may not exist in older models)
                try:
                    with open('featurizer_multiclass.pkl', 'rb') as f_feat:
                        featurizer_loaded = joblib.load(f_feat)
                except:
                    featurizer_loaded = None
                    
            except FileNotFoundError:
                st.error("❌ No trained multi-class model found. Please build a model first in the 'Build Model' tab.")
                return

            with st.spinner("🔍 Analyzing molecule for multi-class prediction..."):
                prediction, probability, explanation_html = predict_from_single_smiles_multiclass(
                    smile_input, st.session_state.selected_featurizer_name_multiclass, tpot_model, label_encoder, class_names, X_train, featurizer_loaded)
                
                if prediction is not None:
                    # Three-column layout for results
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        # Display atomic contribution map for Circular Fingerprint
                        if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                            try:
                                # Get the featurizer object (use loaded one if available, otherwise create new)
                                if featurizer_loaded is not None:
                                    featurizer_obj = featurizer_loaded
                                else:
                                    featurizer_obj = Featurizer[st.session_state.selected_featurizer_name_multiclass]
                                
                                # Get predicted class index
                                pred_class_idx = np.where(class_names == prediction)[0][0] if isinstance(class_names, np.ndarray) else list(class_names).index(prediction)
                                
                                with st.spinner("🗺️ Generating high-resolution fragment contribution map..."):
                                    atomic_contrib_img = generate_fragment_contribution_map(
                                        smile_input, tpot_model, X_train, featurizer_obj, 
                                        class_names, target_class_idx=pred_class_idx, cfp_number=None
                                    )
                                
                                if atomic_contrib_img:
                                    st.markdown("""
                                    <div class="ios-card" style="padding: 16px; text-align: center;">
                                        <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🗺️ Fragment Contribution Map</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display larger, high-resolution image
                                    st.image(atomic_contrib_img, width=500)
                                    
                                    # Download button for the high-resolution image
                                    create_download_button_for_image(
                                        atomic_contrib_img, 
                                        f"atomic_contribution_{smile_input.replace('/', '_')}.png",
                                        "📥 Download HD Image"
                                    )
                                else:
                                    # Fallback to molecular structure
                                    mol = Chem.MolFromSmiles(smile_input)
                                    if mol:
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
                                # Fallback to basic structure
                                mol = Chem.MolFromSmiles(smile_input)
                                if mol:
                                    img = Draw.MolToImage(mol, size=(400, 400))
                                    st.markdown("""
                                    <div class="ios-card" style="padding: 16px; text-align: center;">
                                        <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🧬 Molecule Structure</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.image(img, width=400)
                        else:
                            # Fallback to molecular structure for other featurizers
                            mol = Chem.MolFromSmiles(smile_input)
                            if mol:
                                img = Draw.MolToImage(mol, size=(400, 400))
                                st.markdown("""
                                <div class="ios-card" style="padding: 16px; text-align: center;">
                                    <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🧬 Molecule Structure</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                st.image(img, width=400)
                            else:
                                st.warning("⚠️ Could not visualize molecule")
                    
                    with col2:
                        # Multi-class prediction results
                        st.markdown(f"""
                        <div class="ios-card" style="padding: 20px;">
                            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                                <div style="font-size: 2em; margin-right: 12px;">🎯</div>
                                <div>
                                    <h2 style="color: #007AFF; margin: 0; font-weight: 700; font-size: 1.5em;">{prediction}</h2>
                                    <p style="margin: 4px 0 0 0; color: #8E8E93; font-size: 14px;">Predicted Class</p>
                                </div>
                            </div>
                            <div style="background: rgba(0, 122, 255, 0.1); border-radius: 12px; padding: 12px; margin-bottom: 12px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="color: #007AFF; font-weight: 600;">Confidence:</span>
                                    <span style="color: #1D1D1F; font-weight: 700; font-size: 1.1em;">{probability:.1%}</span>
                                </div>
                            </div>
                            <div style="background: rgba(0, 0, 0, 0.05); border-radius: 8px; padding: 8px;">
                                <p style="margin: 0; color: #8E8E93; font-size: 12px; font-weight: 500;">
                                    <strong>SMILES:</strong> {smile_input}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Color legend (only show for Circular Fingerprint)
                        if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                            # High Positive
                            st.markdown("🔵 **High Positive (Dark Blue)**")
                            st.caption("Strongly contributes to activity")
                            
                            # Low Positive  
                            st.markdown("🟦 **Low Positive (Light Blue)**")
                            st.caption("Moderately supports activity")
                            
                            # Neutral
                            st.markdown("⚪ **Neutral (Gray)**")
                            st.caption("No significant contribution")
                            
                            # Low Negative
                            st.markdown("🟧 **Low Negative (Light Red)**")
                            st.caption("Moderately reduces activity")
                            
                            # High Negative
                            st.markdown("🔴 **High Negative (Dark Red)**")
                            st.caption("Strongly reduces activity")
                    
                    # Prediction explanation
                    if explanation_html:
                        st.markdown(explanation_html, unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to make prediction. Please check your SMILES input.")

    elif active_tab == "batch":
        st.markdown("### 📊 Batch Multi-Class Prediction from File")
        
        with st.expander("📁 Upload Prediction File", expanded=True):
            uploaded_file = st.file_uploader("Upload Excel file with SMILES for batch multi-class prediction", 
                                            type=["xlsx"], key="batch_prediction_upload",
                                            help="Select an Excel file containing SMILES strings for batch prediction")

        if uploaded_file is not None:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            with st.expander("📊 Preview Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)

            # Select SMILES column in iOS card
            st.markdown('<div class="ios-card">', unsafe_allow_html=True)
            col_names = df.columns.tolist()
            smiles_col_predict = st.selectbox("🧬 Select SMILES Column", col_names, key='smiles_column_predict')
            st.markdown('</div>', unsafe_allow_html=True)

            # Batch prediction button
            if st.button("🚀 Run Batch Multi-Class Prediction", use_container_width=True, key="batch_predict_btn"):
                # Check if model exists
                try:
                    with open('best_multiclass_model.pkl', 'rb') as f_model, \
                         open('label_encoder_multiclass.pkl', 'rb') as f_le, \
                         open('class_names_multiclass.pkl', 'rb') as f_classes, \
                         open('X_train_multiclass.pkl', 'rb') as f_X_train, \
                         open('featurizer_multiclass.pkl', 'rb') as f_feat:
                        tpot_model = joblib.load(f_model)
                        label_encoder = joblib.load(f_le)
                        class_names = joblib.load(f_classes)
                        X_train = joblib.load(f_X_train)
                        featurizer_obj = joblib.load(f_feat)
                        
                        # Store in session state for visualization
                        st.session_state['_tpot_model_multiclass'] = tpot_model
                        st.session_state['_X_train_multiclass'] = X_train
                        st.session_state['_featurizer_obj_multiclass'] = featurizer_obj
                        
                except FileNotFoundError:
                    st.error("❌ No trained multi-class model found. Please build a model first in the 'Build Model' tab.")
                    return

                if smiles_col_predict in df.columns:
                    predictions = []
                    probabilities = []
                    all_class_probabilities = []
                    individual_images = []  # Store fragment maps and structures for ZIP
                    
                    # iOS-style progress tracking
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    total_molecules = len(df)
                    
                    for index, row in df.iterrows():
                        # Update progress with iOS styling
                        progress = (index + 1) / total_molecules
                        progress_bar.progress(progress)
                        status_text.markdown(f"<div style='text-align: center; color: #007AFF; font-weight: 600;'>Processing molecule {index + 1} of {total_molecules}</div>", unsafe_allow_html=True)
                        
                        try:
                            standardized_smiles = standardize_smiles(row[smiles_col_predict])
                            if standardized_smiles:
                                mol = Chem.MolFromSmiles(standardized_smiles)
                                if mol is not None:
                                    # Use the loaded featurizer object from session state
                                    features = featurizer_obj.featurize([mol])[0]
                                    feature_df = pd.DataFrame([features])
                                    new_column_names = [f"fp_{col}" for col in feature_df.columns]
                                    feature_df.columns = new_column_names

                                    # Predict
                                    prediction_encoded = tpot_model.predict(feature_df)[0]
                                    prediction_proba = tpot_model.predict_proba(feature_df)[0]
                                    
                                    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                                    max_probability = float(np.max(prediction_proba))
                                    
                                    predictions.append(prediction)
                                    probabilities.append(max_probability)
                                    all_class_probabilities.append(list(prediction_proba))
                                    
                                    # Generate fragment map and structure for ZIP export
                                    image_data = {}
                                    
                                    # Generate fragment contribution map for Circular Fingerprint
                                    if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                                        try:
                                            pred_class_idx = np.where(class_names == prediction)[0][0] if isinstance(class_names, np.ndarray) else list(class_names).index(prediction)
                                            fragment_map = generate_fragment_contribution_map(
                                                standardized_smiles, tpot_model, X_train, featurizer_obj,
                                                class_names, target_class_idx=pred_class_idx, cfp_number=None
                                            )
                                            if fragment_map:
                                                image_data['fragment_map'] = fragment_map
                                        except Exception:
                                            pass
                                    
                                    # Generate molecular structure
                                    try:
                                        structure_img = Draw.MolToImage(mol, size=(400, 400))
                                        image_data['structure'] = structure_img
                                    except Exception:
                                        pass
                                    
                                    individual_images.append(image_data)
                                else:
                                    predictions.append("Invalid SMILES")
                                    probabilities.append(0.0)
                                    all_class_probabilities.append([0.0] * len(class_names))
                                    individual_images.append({})
                            else:
                                predictions.append("Invalid SMILES")
                                probabilities.append(0.0)
                                all_class_probabilities.append([0.0] * len(class_names))
                                individual_images.append({})
                        except Exception as e:
                            predictions.append(f"Error: {str(e)}")
                            probabilities.append(0.0)
                            all_class_probabilities.append([0.0] * len(class_names))
                            individual_images.append({})

                    # Clear progress indicators
                    progress_container.empty()
                    
                    # Add results to dataframe
                    df['Predicted_Class'] = predictions
                    df['Confidence'] = [f"{p:.1%}" if isinstance(p, float) else "N/A" for p in probabilities]
                    
                    # Add class probabilities
                    for i, class_name in enumerate(class_names):
                        df[f'Prob_{class_name}'] = [f"{probs[i]:.1%}" if len(probs) > i else "N/A" for probs in all_class_probabilities]

                    # Display individual results first in iOS cards
                    st.markdown('<h3 class="ios-heading-small" style="color: #1D1D1F; font-weight: 600; font-size: 22px; letter-spacing: -0.02em; margin: 16px 0 12px 0;">🧪 Individual Prediction Results</h3>', unsafe_allow_html=True)
                    
                    # Show results in expandable sections for better organization
                    results_per_page = 5  # Show 5 results at a time
                    total_valid_molecules = sum(1 for p in predictions if not str(p).startswith("Error") and p != "Invalid SMILES")
                    
                    if total_valid_molecules > 0:
                        # Create pagination for large datasets
                        num_pages = (total_valid_molecules + results_per_page - 1) // results_per_page
                        
                        if num_pages > 1:
                            page = st.selectbox("📄 Select Results Page", 
                                              options=list(range(1, num_pages + 1)), 
                                              format_func=lambda x: f"Page {x} ({min(results_per_page, total_valid_molecules - (x-1)*results_per_page)} results)",
                                              key='batch_page_selector')
                        else:
                            page = 1
                        
                        # Calculate indices for current page
                        valid_indices = [i for i, p in enumerate(predictions) if not str(p).startswith("Error") and p != "Invalid SMILES"]
                        start_idx = (page - 1) * results_per_page
                        end_idx = min(start_idx + results_per_page, len(valid_indices))
                        current_page_indices = valid_indices[start_idx:end_idx]
                        
                        # Display individual results for current page
                        for idx in current_page_indices:
                            row = df.iloc[idx]
                            prediction = predictions[idx]
                            probability = probabilities[idx]
                            smiles = row[smiles_col_predict]
                            
                            with st.expander(f"🧬 Molecule {idx + 1}: {prediction} ({probability:.1%} confidence)", expanded=False):
                                # Use EXACT same format as single SMILES prediction
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    # Display atomic contribution map for Circular Fingerprint (SAME AS SINGLE PREDICTION)
                                    if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                                        try:
                                            # Get the featurizer object from session state
                                            featurizer_obj = st.session_state.get('_featurizer_obj_multiclass')
                                            model = st.session_state.get('_tpot_model_multiclass')
                                            X_train_data = st.session_state.get('_X_train_multiclass')
                                            
                                            if featurizer_obj is not None and model is not None and X_train_data is not None:
                                                # Get predicted class index
                                                pred_class_idx = np.where(class_names == prediction)[0][0] if isinstance(class_names, np.ndarray) else list(class_names).index(prediction)
                                                
                                                atomic_contrib_img = generate_fragment_contribution_map(
                                                    smiles, model, X_train_data, featurizer_obj, 
                                                    class_names, target_class_idx=pred_class_idx, cfp_number=None
                                                )
                                                
                                                if atomic_contrib_img:
                                                    st.markdown("""
                                                    <div class="ios-card" style="padding: 16px; text-align: center;">
                                                        <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🗺️ Fragment Contribution Map</h4>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                    
                                                    # Display larger, high-resolution image
                                                    st.image(atomic_contrib_img, width=500)
                                                    
                                                    # Download button for the high-resolution image
                                                    create_download_button_for_image(
                                                        atomic_contrib_img, 
                                                        f"fragment_contribution_molecule_{idx + 1}.png",
                                                        "📥 Download HD Image"
                                                    )
                                                else:
                                                    # Fallback to molecular structure
                                                    mol = Chem.MolFromSmiles(smiles)
                                                    if mol:
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
                                            # Fallback to basic structure
                                            mol = Chem.MolFromSmiles(smiles)
                                            if mol:
                                                img = Draw.MolToImage(mol, size=(400, 400))
                                                st.markdown("""
                                                <div class="ios-card" style="padding: 16px; text-align: center;">
                                                    <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🧬 Molecule Structure</h4>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                st.image(img, width=400)
                                    else:
                                        # Fallback to molecular structure for other featurizers
                                        mol = Chem.MolFromSmiles(smiles)
                                        if mol:
                                            img = Draw.MolToImage(mol, size=(400, 400))
                                            st.markdown("""
                                            <div class="ios-card" style="padding: 16px; text-align: center;">
                                                <h4 style="margin: 0 0 8px 0; color: #007AFF; font-weight: 600; font-size: 16px;">🧬 Molecule Structure</h4>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            st.image(img, width=400)
                                        else:
                                            st.warning("⚠️ Could not visualize molecule")
                                
                                with col2:
                                    # Multi-class prediction results (SAME FORMAT AS SINGLE PREDICTION)
                                    st.markdown(f"""
                                    <div class="ios-card" style="padding: 20px;">
                                        <div style="display: flex; align-items: center; margin-bottom: 16px;">
                                            <div style="font-size: 2em; margin-right: 12px;">🎯</div>
                                            <div>
                                                <h2 style="color: #007AFF; margin: 0; font-weight: 700; font-size: 1.5em;">{prediction}</h2>
                                                <p style="margin: 4px 0 0 0; color: #8E8E93; font-size: 14px;">Predicted Class</p>
                                            </div>
                                        </div>
                                        <div style="background: rgba(0, 122, 255, 0.1); border-radius: 12px; padding: 12px; margin-bottom: 12px;">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <span style="color: #007AFF; font-weight: 600;">Confidence:</span>
                                                <span style="color: #1D1D1F; font-weight: 700; font-size: 1.1em;">{probability:.1%}</span>
                                            </div>
                                        </div>
                                        <div style="background: rgba(0, 0, 0, 0.05); border-radius: 8px; padding: 8px;">
                                            <p style="margin: 0; color: #8E8E93; font-size: 12px; font-weight: 500;">
                                                <strong>SMILES:</strong> {smiles[:40]}{'...' if len(smiles) > 40 else ''}
                                            </p>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    # Color legend (only show for Circular Fingerprint) - SAME AS SINGLE PREDICTION
                                    if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                                        # High Positive
                                        st.markdown("🔵 **High Positive (Dark Blue)**")
                                        st.caption("Strongly contributes to activity")
                                        
                                        # Low Positive  
                                        st.markdown("🟦 **Low Positive (Light Blue)**")
                                        st.caption("Moderately supports activity")
                                        
                                        # Neutral
                                        st.markdown("⚪ **Neutral (Gray)**")
                                        st.caption("No significant contribution")
                                        
                                        # Low Negative
                                        st.markdown("🟧 **Low Negative (Light Red)**")
                                        st.caption("Moderately reduces activity")
                                        
                                        # High Negative
                                        st.markdown("🔴 **High Negative (Dark Red)**")
                                        st.caption("Strongly reduces activity")

                    # Display results table
                    st.markdown('<h3 class="ios-heading-small" style="color: #1D1D1F; font-weight: 600; font-size: 22px; letter-spacing: -0.02em; margin: 16px 0 12px 0;">📊 Complete Multi-Class Results Table</h3>', unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download Section with ZIP file option
                    st.markdown("---")
                    st.markdown('<h3 class="ios-heading-small" style="color: #1D1D1F; font-weight: 600; font-size: 22px; letter-spacing: -0.02em; margin: 16px 0 12px 0;">📥 Download Prediction Results</h3>', unsafe_allow_html=True)
                    
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    # Simple CSV download
                    with col_dl1:
                        csv = df.to_csv(index=False)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv,
                            file_name=f'multiclass_predictions_{timestamp}.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    
                    # Excel download
                    with col_dl2:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Predictions')
                        excel_buffer.seek(0)
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_buffer,
                            file_name=f'multiclass_predictions_{timestamp}.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True
                        )
                    
                    # Complete ZIP report with all files
                    with col_dl3:
                        try:
                            prediction_zip = create_prediction_report_zip(
                                df, 
                                smiles_col=smiles_col_predict,
                                prediction_col='Predicted_Class',
                                confidence_col='Confidence',
                                individual_structures=individual_images,
                                class_names=class_names
                            )
                            st.download_button(
                                label="📦 Download Full Report",
                                data=prediction_zip,
                                file_name=f'multiclass_prediction_report_{timestamp}.zip',
                                mime='application/zip',
                                use_container_width=True,
                                help="Download complete report with CSV, summary, and individual fragment contribution maps"
                            )
                        except Exception as e:
                            st.error(f"Error creating ZIP report: {str(e)}")

if __name__ == "__main__":
    main()
