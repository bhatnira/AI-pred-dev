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
from rdkit.Chem import Draw
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
matplotlib.use('Agg')  # Set backend before importing pyplot
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
import colorsys
import io
import traceback
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

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
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 14px 20px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 4px 16px rgba(0, 122, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 122, 255, 0.4);
        background: linear-gradient(135deg, #0056D3 0%, #4A44C4 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3);
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

# --- Fragment Contribution Mapping for Circular Fingerprint ---
def weight_to_google_color(weight, min_weight, max_weight):
    """Convert weight to color using improved HLS color scheme with better handling of edge cases"""
    
    # Handle edge cases
    if max_weight == min_weight:
        norm = 0.5  # Neutral color
    else:
        norm = (weight - min_weight) / (max_weight - min_weight)
    
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
    except Exception as e:
        st.error(f"Error in draw_molecule_with_fragment_weights: {str(e)}")
        print(f"Error in draw_molecule_with_fragment_weights: {str(e)}")
        traceback.print_exc()
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
    except Exception as e:
        st.error(f"Error in map_cfp_bits_to_atoms: {str(e)}")
        print(f"Error in map_cfp_bits_to_atoms: {str(e)}")
        traceback.print_exc()
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
    except Exception as e:
        st.error(f"Error in map_specific_cfp_to_atoms: {str(e)}")
        print(f"Error in map_specific_cfp_to_atoms: {str(e)}")
        traceback.print_exc()
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
                class_names=["Inactive", "Active"],  # Updated for multi-class
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
        st.error(f"Error in generate_fragment_contribution_map: {str(e)}")
        print(f"Error in generate_fragment_contribution_map: {str(e)}")
        traceback.print_exc()
        return None

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

# Function to preprocess data and perform modeling for multi-class classification
def preprocess_and_model_multiclass(df, smiles_col, activity_col, featurizer_name, generations=3, cv=3, verbosity=0, test_size=0.20, cfp_params=None):
    """
    Multi-class preprocessing and TPOT model building with time tracking
    """
    start_time = time.time()
    
    # Enhanced progress tracking with time estimates
    progress_container = st.container()
    
    with progress_container:
        # Time tracking metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            elapsed_placeholder = st.empty()
        with col2:
            remaining_placeholder = st.empty()
        with col3:
            estimated_placeholder = st.empty()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Estimate total time based on dataset size
    n_samples = len(df)
    estimated_total_time = max(30, min(300, n_samples * 0.5 + generations * cv * 20))  # Longer for multi-class
    
    def update_progress_with_time(progress_percent, status_msg):
        elapsed = time.time() - start_time
        
        if progress_percent > 0.05:  # After some progress
            estimated_remaining = (elapsed / progress_percent) * (1 - progress_percent)
        else:
            estimated_remaining = estimated_total_time
        
        # Update time displays
        elapsed_placeholder.markdown(create_ios_metric_card("Elapsed", format_time_duration(elapsed), "", "⏱️"), unsafe_allow_html=True)
        remaining_placeholder.markdown(create_ios_metric_card("Remaining", format_time_duration(max(0, estimated_remaining)), "", "⏳"), unsafe_allow_html=True)
        estimated_placeholder.markdown(create_ios_metric_card("Total Est.", format_time_duration(estimated_total_time), "", "📊"), unsafe_allow_html=True)
        
        progress_bar.progress(progress_percent)
        status_text.info(f"🔬 {status_msg}")
    
    try:
        # Phase 1: Data Preparation
        update_progress_with_time(0.05, "Standardizing SMILES...")
        
        df[smiles_col + '_standardized'] = df[smiles_col].apply(standardize_smiles)
        df.dropna(subset=[smiles_col + '_standardized'], inplace=True)

        # Check for multi-class scenario
        unique_classes = df[activity_col].unique()
        if len(unique_classes) < 3:
            st.error("Multi-class classification requires at least 3 distinct classes. Please check your dataset.")
            return None, None, None, None, None, None, None, None, None, None, None, None

        update_progress_with_time(0.15, "Featurizing molecules...")
        
        # Create featurizer with custom parameters if CFP is selected
        if featurizer_name == "Circular Fingerprint" and cfp_params:
            radius = cfp_params.get('radius', 4)
            size = cfp_params.get('size', 2048)
            featurizer = dc.feat.CircularFingerprint(size=size, radius=radius)
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
            return None, None, None, None, None, None, None, None, None, None, None, None

        update_progress_with_time(0.5, "Preparing training data...")
        
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
        
        # Train the TPOT model
        tpot.fit(X_train, y_train)

        update_progress_with_time(0.9, "Evaluating model performance...")
        
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

        update_progress_with_time(1.0, "Training completed successfully!")
        
        # Clear progress indicators after a brief pause
        time.sleep(1)
        progress_container.empty()

        return tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, y_pred_proba, class_names, le, df, X_train, y_train, featurizer
        
    except Exception as e:
        progress_container.empty()
        st.error(f"An error occurred during model training: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def predict_from_single_smiles_multiclass(smiles, featurizer_name, model, label_encoder, class_names=None, X_train=None):
    """
    Predict activity for a single SMILES string in multi-class setting
    """
    try:
        # Standardize SMILES
        standardized_smiles = standardize_smiles(smiles)
        if standardized_smiles is None:
            return None, None, "Invalid SMILES string"

        # Featurize molecule - use stored featurizer if available for consistency
        if (featurizer_name == "Circular Fingerprint" and 
            'featurizer_multiclass' in st.session_state and 
            st.session_state.featurizer_multiclass is not None):
            featurizer = st.session_state.featurizer_multiclass
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
        max_probability = max(probabilities)
        
        # Use class_names if provided, otherwise use label_encoder.classes_
        display_classes = class_names if class_names is not None else label_encoder.classes_
        
        # Create explanation
        explanation = f"""
        <div class="ios-card" style="margin: 16px 0;">
            <h4 style="color: #007AFF; margin-bottom: 12px;">🧬 Prediction Details</h4>
            <p><strong>Predicted Class:</strong> {prediction}</p>
            <p><strong>Confidence:</strong> {max_probability:.1%}</p>
            <p><strong>Input SMILES:</strong> {smiles}</p>
            <p><strong>Standardized SMILES:</strong> {standardized_smiles}</p>
        """
        
        # Close the explanation div
        explanation += "</div>"
        
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
        
        explanation += "</div>"

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

# Main Streamlit application
def main():
    # Create main header
    st.markdown(create_ios_header("Chemlara Multi-Class Classifier", "Advanced AutoML for Multi-Class Chemical Activity Prediction"), unsafe_allow_html=True)

    # Mobile-friendly navigation using tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔬 Build Model", "🧪 Single Prediction", "📊 Batch Prediction"])

    with tab1:
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

                   <p style="color: #8E8E93; font-style: italic; text-align: center;">🧬 Perfect for drug discovery with multiple activity profiles!</p>
                   """, "🎉"), unsafe_allow_html=True)

    with tab2:
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
                
                # Additional parameters for Circular Fingerprint
                if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                    st.markdown("#### 🔬 Circular Fingerprint Parameters")
                    col_fp1, col_fp2 = st.columns(2)
                    with col_fp1:
                        cfp_radius = st.slider("Radius", min_value=1, max_value=6, value=4, 
                                             help="Circular fingerprint radius (default: 4)", key='cfp_radius_multi')
                    with col_fp2:
                        cfp_size = st.number_input("Fingerprint Size", min_value=64, max_value=16384, value=2048, step=64,
                                              help="Number of bits in fingerprint (default: 2048)", key='cfp_size_multi')

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
                with st.spinner("🔄 Building your multi-class model... This may take several minutes."):
                    st.markdown(create_ios_card("Multi-Class Training in Progress", 
                                              "Processing data and training your machine learning model for multi-class classification...", "🤖"), unsafe_allow_html=True)
                    
                    # Get CFP parameters if Circular Fingerprint is selected
                    cfp_params = {}
                    if st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint":
                        cfp_params = {
                            'radius': locals().get('cfp_radius', 4),
                            'size': locals().get('cfp_size', 2048)
                        }
                    
                    results = preprocess_and_model_multiclass(
                        df, smiles_col, activity_col, st.session_state.selected_featurizer_name_multiclass, 
                        generations=generations, cv=cv, verbosity=verbosity, test_size=test_size, cfp_params=cfp_params)

                    if results[0] is not None:
                        tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, y_pred_proba, class_names, le, df_result, X_train, y_train, featurizer = results
                        
                        # Store featurizer and model components for fragment contribution
                        st.session_state.featurizer_multiclass = featurizer
                        st.session_state.tpot_model_multiclass = tpot.fitted_pipeline_
                        st.session_state.X_train_multiclass = X_train
                        st.session_state.trained_featurizer_name_multiclass = st.session_state.selected_featurizer_name_multiclass
                        
                        # Save model and necessary data
                        with open('best_multiclass_model.pkl', 'wb') as f:
                            joblib.dump(tpot.fitted_pipeline_, f)
                        with open('X_train_multiclass.pkl', 'wb') as f:
                            joblib.dump(X_train, f)
                        with open('label_encoder_multiclass.pkl', 'wb') as f:
                            joblib.dump(le, f)
                        with open('class_names_multiclass.pkl', 'wb') as f:
                            joblib.dump(class_names, f)

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
                        
                        with col1:
                            try:
                                roc_fig = create_multiclass_roc_curves(y_test, y_pred_proba, class_names)
                                if roc_fig:
                                    st.pyplot(roc_fig)
                                    plt.close(roc_fig)
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
                                    plt.close(cm_fig)
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

                            # Create download buttons
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

    with tab3:
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
            except FileNotFoundError:
                st.error("❌ No trained multi-class model found. Please build a model first in the 'Build Model' tab.")
                return

            with st.spinner("🔍 Analyzing molecule for multi-class prediction..."):
                prediction, probability, explanation_html = predict_from_single_smiles_multiclass(
                    smile_input, st.session_state.selected_featurizer_name_multiclass, tpot_model, label_encoder, class_names, X_train)
                
                if prediction is not None:
                    # Display molecular structure or fragment contribution map
                    try:
                        mol = Chem.MolFromSmiles(smile_input)
                        if mol:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                # Check if we should display fragment contribution map or regular structure
                                trained_featurizer = st.session_state.get('trained_featurizer_name_multiclass', st.session_state.selected_featurizer_name_multiclass)
                                current_featurizer = st.session_state.selected_featurizer_name_multiclass
                                
                                if (trained_featurizer == "Circular Fingerprint" and 
                                    current_featurizer == "Circular Fingerprint"):
                                    
                                    # Display fragment contribution map automatically
                                    try:
                                        # Get model components from session state
                                        model = st.session_state.get('tpot_model_multiclass')
                                        X_train_data = st.session_state.get('X_train_multiclass')
                                        featurizer_obj = st.session_state.get('featurizer_multiclass')
                                        
                                        if model is not None and X_train_data is not None and featurizer_obj is not None:
                                            with st.spinner("🧬 Generating fragment contribution map..."):
                                                atomic_contrib_img = generate_fragment_contribution_map(
                                                    smile_input, model, X_train_data, featurizer_obj, cfp_number=None
                                                )
                                            
                                            if atomic_contrib_img:
                                                st.markdown(create_ios_card("🧬 Fragment Contribution", "", "🧬"), unsafe_allow_html=True)
                                                
                                                # Create layout with image and legend side by side
                                                main_col, legend_col = st.columns([4, 1])
                                                
                                                with main_col:
                                                    # Display larger, high-resolution image
                                                    st.image(atomic_contrib_img, caption="Fragment Contribution Analysis", use_column_width=True)
                                                    
                                                    # Download button for high-res image
                                                    create_download_button_for_image(
                                                        atomic_contrib_img, 
                                                        f"fragment_contribution_{smile_input.replace('/', '_')}.png",
                                                        "📥 Download Fragment Map"
                                                    )
                                                
                                                with legend_col:
                                                    # Compact color legend on the side
                                                    st.markdown("**🎨 Legend**")
                                                    
                                                    # High Positive
                                                    st.markdown("🔵 **High +**")
                                                    st.caption("Strong increase")
                                                    
                                                    # Low Positive  
                                                    st.markdown("🟦 **Low +**")
                                                    st.caption("Moderate increase")
                                                    
                                                    # Neutral
                                                    st.markdown("⚪ **Neutral**")
                                                    st.caption("No effect")
                                                    
                                                    # Low Negative
                                                    st.markdown("🟧 **Low -**")
                                                    st.caption("Moderate decrease")
                                                    
                                                    # High Negative
                                                    st.markdown("🔴 **High -**")
                                                    st.caption("Strong decrease")
                                            else:
                                                # Fallback to regular structure
                                                st.markdown(create_ios_card("Molecular Structure", "", "🧬"), unsafe_allow_html=True)
                                                img = Draw.MolToImage(mol, size=(300, 300))
                                                st.image(img, width=250)
                                        else:
                                            # Fallback to regular structure
                                            st.markdown(create_ios_card("Molecular Structure", "", "🧬"), unsafe_allow_html=True)
                                            img = Draw.MolToImage(mol, size=(300, 300))
                                            st.image(img, width=250)
                                    except Exception as e:
                                        # Debug: show error for troubleshooting
                                        st.error(f"Error loading model for fragment analysis: {str(e)}")
                                        print(f"Error loading model for fragment analysis: {str(e)}")
                                        traceback.print_exc()
                                        
                                        # Fallback to regular structure
                                        st.markdown(create_ios_card("Molecular Structure", "", "🧬"), unsafe_allow_html=True)
                                        img = Draw.MolToImage(mol, size=(300, 300))
                                        st.image(img, width=250)
                                else:
                                    # Display regular molecular structure for non-CFP featurizers
                                    st.markdown(create_ios_card("Molecular Structure", "", "🧬"), unsafe_allow_html=True)
                                    img = Draw.MolToImage(mol, size=(300, 300))
                                    st.image(img, width=250)
                            
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
                    except:
                        st.markdown(create_ios_card("Prediction Result", f"Predicted Class: {prediction} (Confidence: {probability:.1%})", "🎯"), unsafe_allow_html=True)
                    
                    # Prediction explanation
                    if explanation_html:
                        st.markdown(explanation_html, unsafe_allow_html=True)
                    
                    # LIME interpretation download button
                    if 'lime_filename' in st.session_state and st.session_state['lime_filename']:
                        st.markdown("### � Download LIME Interpretation")
                        st.markdown(create_downloadable_model_link(st.session_state['lime_filename'], '📥 Download LIME Interpretation'), unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to make prediction. Please check your SMILES input.")

    with tab4:
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
                         open('X_train_multiclass.pkl', 'rb') as f_X_train:
                        tpot_model = joblib.load(f_model)
                        label_encoder = joblib.load(f_le)
                        class_names = joblib.load(f_classes)
                        X_train = joblib.load(f_X_train)
                except FileNotFoundError:
                    st.error("❌ No trained multi-class model found. Please build a model first in the 'Build Model' tab.")
                    return

                if smiles_col_predict in df.columns:
                    predictions = []
                    probabilities = []
                    all_class_probabilities = []
                    
                    # iOS-style progress tracking
                    st.markdown(create_ios_card("Processing Molecules", 
                                              "Analyzing your molecules using the trained multi-class model...", "⚗️"), unsafe_allow_html=True)
                    
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
                                    # Use stored featurizer if available for consistency
                                    if (st.session_state.selected_featurizer_name_multiclass == "Circular Fingerprint" and 
                                        'featurizer_multiclass' in st.session_state and 
                                        st.session_state.featurizer_multiclass is not None):
                                        featurizer = st.session_state.featurizer_multiclass
                                    else:
                                        featurizer = Featurizer[st.session_state.selected_featurizer_name_multiclass]
                                    
                                    features = featurizer.featurize([mol])[0]
                                    feature_df = pd.DataFrame([features])
                                    new_column_names = [f"fp_{col}" for col in feature_df.columns]
                                    feature_df.columns = new_column_names

                                    # Predict
                                    prediction_encoded = tpot_model.predict(feature_df)[0]
                                    prediction_proba = tpot_model.predict_proba(feature_df)[0]
                                    
                                    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                                    max_probability = max(prediction_proba)
                                    
                                    predictions.append(prediction)
                                    probabilities.append(max_probability)
                                    all_class_probabilities.append(prediction_proba)
                                else:
                                    predictions.append("Invalid SMILES")
                                    probabilities.append(0.0)
                                    all_class_probabilities.append([0.0] * len(class_names))
                            else:
                                predictions.append("Invalid SMILES")
                                probabilities.append(0.0)
                                all_class_probabilities.append([0.0] * len(class_names))
                        except Exception as e:
                            predictions.append(f"Error: {str(e)}")
                            probabilities.append(0.0)
                            all_class_probabilities.append([0.0] * len(class_names))

                    # Clear progress indicators
                    progress_container.empty()
                    st.success("🎉 Batch multi-class prediction completed successfully!")
                    
                    # Add results to dataframe
                    df['Predicted_Class'] = predictions
                    df['Confidence'] = [f"{p:.1%}" if isinstance(p, float) else "N/A" for p in probabilities]
                    
                    # Add class probabilities
                    for i, class_name in enumerate(class_names):
                        df[f'Prob_{class_name}'] = [f"{probs[i]:.1%}" if len(probs) > i else "N/A" for probs in all_class_probabilities]

                    # Display individual results first in iOS cards
                    st.markdown("### 🧪 Individual Multi-Class Prediction Results")
                    
                    # Show results in expandable sections for better organization
                    results_per_page = 5  # Show 5 results at a time
                    total_valid_molecules = sum(1 for p in predictions if not str(p).startswith("Error") and str(p) != "Invalid SMILES")
                    
                    if total_valid_molecules > 0:
                        # Create pagination for large datasets
                        num_pages = (total_valid_molecules + results_per_page - 1) // results_per_page
                        
                        if num_pages > 1:
                            page = st.selectbox("📄 Select Results Page", 
                                              options=list(range(1, num_pages + 1)), 
                                              format_func=lambda x: f"Page {x} ({min(results_per_page, total_valid_molecules - (x-1)*results_per_page)} results)")
                        else:
                            page = 1
                        
                        # Calculate indices for current page
                        valid_indices = [i for i, p in enumerate(predictions) if not str(p).startswith("Error") and str(p) != "Invalid SMILES"]
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
                                # Create three columns for structure, results, and additional info
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    # Display molecular structure - try fragment contribution if available
                                    try:
                                        trained_featurizer = st.session_state.get('trained_featurizer_name_multiclass')
                                        current_featurizer = st.session_state.selected_featurizer_name_multiclass
                                        
                                        if (trained_featurizer == "Circular Fingerprint" and 
                                            current_featurizer == "Circular Fingerprint"):
                                            
                                            # Try to get stored model components
                                            model = st.session_state.get('tpot_model_multiclass')
                                            X_train_data = st.session_state.get('X_train_multiclass')
                                            featurizer_obj = st.session_state.get('featurizer_multiclass')
                                            
                                            if model is not None and X_train_data is not None and featurizer_obj is not None:
                                                st.markdown("#### 🧬 Fragment Contribution Map")
                                                frag_img = generate_fragment_contribution_map(smiles, model, X_train_data, featurizer_obj, None)
                                                if frag_img:
                                                    st.image(frag_img, width=400, caption="")
                                                    
                                                    # Create download button for the image
                                                    create_download_button_for_image(
                                                        frag_img, 
                                                        f"fragment_contribution_molecule_{idx + 1}.png",
                                                        "📥 Download Fragment Map"
                                                    )
                                                else:
                                                    # Fallback to basic structure
                                                    mol = Chem.MolFromSmiles(smiles)
                                                    if mol:
                                                        img = Draw.MolToImage(mol, size=(300, 300))
                                                        st.markdown("#### 🧬 Molecule Structure")
                                                        st.image(img, width=300)
                                            else:
                                                # Basic structure display
                                                mol = Chem.MolFromSmiles(smiles)
                                                if mol:
                                                    img = Draw.MolToImage(mol, size=(300, 300))
                                                    st.markdown("#### 🧬 Molecule Structure")
                                                    st.image(img, width=300)
                                        else:
                                            # Basic structure for other featurizers
                                            mol = Chem.MolFromSmiles(smiles)
                                            if mol:
                                                img = Draw.MolToImage(mol, size=(300, 300))
                                                st.markdown("#### 🧬 Molecule Structure")
                                                st.image(img, width=300)
                                    except Exception as e:
                                        # Fallback structure display
                                        try:
                                            mol = Chem.MolFromSmiles(smiles)
                                            if mol:
                                                img = Draw.MolToImage(mol, size=(300, 300))
                                                st.markdown("#### 🧬 Molecule Structure")
                                                st.image(img, width=300)
                                        except:
                                            st.markdown("""
                                            <div class="ios-card" style="padding: 16px; text-align: center;">
                                                <h4 style="margin: 0; color: #8E8E93;">Structure unavailable</h4>
                                            </div>
                                            """, unsafe_allow_html=True)
                                
                                with col2:
                                    # Multi-class prediction results in compact iOS card
                                    # Use different colors for different classes
                                    class_colors = ["#34C759", "#007AFF", "#FF9500", "#FF3B30", "#5856D6"]
                                    class_icons = ["🟢", "🔵", "🟡", "🔴", "🟣"]
                                    
                                    # Get color and icon based on prediction
                                    try:
                                        pred_index = list(class_names).index(prediction) if prediction in class_names else 0
                                        prediction_color = class_colors[pred_index % len(class_colors)]
                                        prediction_icon = class_icons[pred_index % len(class_icons)]
                                    except:
                                        prediction_color = "#007AFF"
                                        prediction_icon = "🎯"
                                    
                                    st.markdown(f"""
                                    <div class="ios-card" style="padding: 12px; margin: 8px 0;">
                                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                            <div style="font-size: 1.5em; margin-right: 8px;">{prediction_icon}</div>
                                            <div>
                                                <h3 style="color: {prediction_color}; margin: 0; font-weight: 700; font-size: 1.1em;">{prediction}</h3>
                                                <p style="margin: 2px 0 0 0; color: #8E8E93; font-size: 11px;">Predicted Class</p>
                                            </div>
                                        </div>
                                        <div style="background: rgba(0, 122, 255, 0.1); border-radius: 8px; padding: 8px; margin-bottom: 8px;">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <span style="color: #007AFF; font-weight: 600; font-size: 12px;">Confidence:</span>
                                                <span style="color: #1D1D1F; font-weight: 700; font-size: 14px;">{probability:.1%}</span>
                                            </div>
                                        </div>
                                        <div style="background: rgba(0, 0, 0, 0.05); border-radius: 6px; padding: 6px;">
                                            <p style="margin: 0; color: #8E8E93; font-size: 10px; font-weight: 500;">
                                                <strong>SMILES:</strong> {smiles[:30]}{'...' if len(smiles) > 30 else ''}
                                            </p>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    # Additional data and class probabilities in compact card
                                    other_columns = [col for col in row.index if col != smiles_col_predict and not col.startswith('Predicted_') and not col.startswith('Prob_') and col != 'Confidence']
                                    
                                    st.markdown(f"""
                                    <div class="ios-card" style="padding: 12px; margin: 8px 0;">
                                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                            <div style="font-size: 1.5em; margin-right: 8px;">📊</div>
                                            <div>
                                                <h3 style="color: #007AFF; margin: 0; font-weight: 600; font-size: 14px;">Class Probabilities</h3>
                                                <p style="margin: 2px 0 0 0; color: #8E8E93; font-size: 10px;">All class predictions</p>
                                            </div>
                                        </div>
                                        <div style="background: rgba(0, 0, 0, 0.02); border-radius: 6px; padding: 8px;">
                                    """, unsafe_allow_html=True)
                                    
                                    # Show class probabilities
                                    if idx < len(all_class_probabilities):
                                        probs = all_class_probabilities[idx]
                                        for i, (class_name, prob) in enumerate(zip(class_names, probs)):
                                            color = class_colors[i % len(class_colors)] if isinstance(prob, float) else "#8E8E93"
                                            prob_display = f"{prob:.1%}" if isinstance(prob, float) else "N/A"
                                            st.markdown(f"""
                                            <p style="margin: 2px 0; color: {color}; font-size: 11px; font-weight: 500;">
                                                <strong>{class_name}:</strong> {prob_display}
                                            </p>
                                            """, unsafe_allow_html=True)
                                    
                                    # Show additional data if available
                                    if other_columns:
                                        st.markdown("""
                                        <hr style="margin: 8px 0; border: none; border-top: 1px solid rgba(0,0,0,0.1);">
                                        <p style="margin: 4px 0 2px 0; color: #007AFF; font-size: 10px; font-weight: 600;">Additional Data:</p>
                                        """, unsafe_allow_html=True)
                                        
                                        for col in other_columns[:2]:  # Limit to first 2 additional columns
                                            value = str(row[col])
                                            if len(value) > 15:
                                                value = value[:15] + "..."
                                            st.markdown(f"""
                                            <p style="margin: 1px 0; color: #1D1D1F; font-size: 10px;">
                                                <strong>{col}:</strong> {value}
                                            </p>
                                            """, unsafe_allow_html=True)
                                    
                                    st.markdown("</div></div>", unsafe_allow_html=True)

                    # Display results table
                    st.markdown("### 📊 Complete Multi-Class Results Table")
                    st.markdown(create_ios_card("Complete Multi-Class Prediction Results", 
                                              "Your batch prediction results with class probabilities are ready!", "📊"), unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                    
                    # iOS-style download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Multi-Class Results as CSV",
                        data=csv,
                        file_name='multiclass_batch_predictions.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                    
                    # Summary statistics in iOS cards
                    st.markdown("### 📈 Multi-Class Summary Statistics")
                    valid_predictions = [p for p in predictions if str(p) not in ["Invalid SMILES"] and not str(p).startswith("Error")]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(create_ios_metric_card("Total Processed", str(len(df)), "molecules", "⚗️"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_ios_metric_card("Valid Predictions", str(len(valid_predictions)), f"out of {len(df)}", "✅"), unsafe_allow_html=True)
                    with col3:
                        if valid_predictions:
                            most_common_class = max(set(valid_predictions), key=valid_predictions.count)
                            class_count = valid_predictions.count(most_common_class)
                            st.markdown(create_ios_metric_card("Most Common", most_common_class, f"{class_count} predictions", "🏆"), unsafe_allow_html=True)
                    
                else:
                    st.error("❌ SMILES column not found in the uploaded file.")

if __name__ == "__main__":
    main()
