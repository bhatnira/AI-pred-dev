import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import zipfile
import io
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import deepchem as dc
from deepchem.feat import ConvMolFeaturizer
from deepchem.models import GraphConvModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
import time
import threading
import queue

# Configure matplotlib and RDKit for headless mode
os.environ['MPLBACKEND'] = 'Agg'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

# Disable RDKit warnings and configure for headless rendering
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set random seed for reproducibility
tf.random.set_seed(42)

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

# Configure Streamlit page for mobile-friendly display
# st.set_page_config(
#     page_title="GraphConv Classifier",
#     page_icon="🧬",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# Apple-style iOS interface CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    /* Global iOS-like styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 100%;
        padding: 1rem;
        background: transparent;
    }
    
    /* Apple-style buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 14px 20px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        background: linear-gradient(135deg, #FF5252 0%, #26A69A 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3);
    }
    
    /* iOS Input fields */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 16px;
        font-size: 16px;
        transition: all 0.2s ease;
    }
    
    /* Fallback for input fields */
    @supports not (backdrop-filter: blur(10px)) {
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.95);
        }
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid #FF6B6B;
        box-shadow: 0 0 0 4px rgba(255, 107, 107, 0.1);
        outline: none;
    }
    
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 16px;
        font-size: 16px;
    }
    
    /* Fallback for select boxes */
    @supports not (backdrop-filter: blur(10px)) {
        .stSelectbox > div > div > select {
            background: rgba(255, 255, 255, 0.95);
        }
    }
    
    /* iOS Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        border-radius: 8px;
        height: 8px;
    }
    
    .stProgress > div > div {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        height: 8px;
    }
    
    /* File uploader - iOS style */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 2px dashed #FF6B6B;
        padding: 32px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    /* Fallback for file uploader */
    @supports not (backdrop-filter: blur(10px)) {
        .stFileUploader > div {
            background: rgba(255, 255, 255, 0.95);
        }
    }
    
    .stFileUploader > div:hover {
        background: rgba(255, 107, 107, 0.05);
        border-color: #4ECDC4;
    }
    
    /* Tabs - iOS style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Fallback for tabs */
    @supports not (backdrop-filter: blur(10px)) {
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.95);
        }
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
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
    }
    
    /* Expander - iOS style */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        font-weight: 500;
    }
    
    /* Fallback for expander */
    @supports not (backdrop-filter: blur(10px)) {
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.95);
        }
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
        color: #FF6B6B;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
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
        background: rgba(255, 107, 107, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 107, 107, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Define the ConvMolFeaturizer
featurizer = ConvMolFeaturizer()

# Function to create iOS-style header
def create_ios_header(title, subtitle=None):
    header_html = f"""
    <div style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); border-radius: 24px; padding: 32px 24px; margin-bottom: 24px; color: white; text-align: center; box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);">
        <h1 style="margin: 0; font-size: 32px; font-weight: 700; letter-spacing: -0.5px;">{title}</h1>
        {f'<p style="margin: 8px 0 0 0; font-size: 18px; opacity: 0.9; font-weight: 400;">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

# Time estimation and formatting functions
def estimate_training_time(n_samples, n_epochs, batch_size):
    """Estimate training time based on dataset characteristics"""
    # Base time per epoch (in seconds)
    base_time_per_epoch = max(n_samples / batch_size * 0.1, 1.0)
    
    # Estimate total time
    estimated_time = base_time_per_epoch * n_epochs
    return max(estimated_time, 10)  # Minimum 10 seconds

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

# Enhanced progress tracking
def create_progress_tracker(total_time_estimate, total_epochs):
    """Create an advanced progress tracker"""
    progress_container = st.container()
    
    with progress_container:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Estimated Time", format_time_duration(total_time_estimate))
        
        with col2:
            remaining_time_placeholder = st.empty()
            remaining_time_placeholder.metric("Time Remaining", "Calculating...")
        
        with col3:
            speed_placeholder = st.empty()
            speed_placeholder.metric("Epochs/min", "Starting...")
        
        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        return {
            'progress_bar': progress_bar,
            'status_text': status_text,
            'remaining_time': remaining_time_placeholder,
            'speed': speed_placeholder,
            'start_time': time.time(),
            'total_epochs': total_epochs
        }

def update_progress_tracker(tracker, current_epoch, training_loss, val_accuracy=None):
    """Update the progress tracker with current status"""
    if current_epoch == 0:
        return
    
    # Calculate progress percentage
    progress_percent = min(current_epoch / tracker['total_epochs'], 1.0)
    
    # Update progress bar
    tracker['progress_bar'].progress(progress_percent)
    
    # Calculate elapsed time and estimated remaining time
    elapsed_time = time.time() - tracker['start_time']
    
    if progress_percent > 0:
        estimated_total_time = elapsed_time / progress_percent
        remaining_time = max(estimated_total_time - elapsed_time, 0)
        
        # Calculate speed (epochs per minute)
        speed = (current_epoch / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        # Update displays
        tracker['remaining_time'].metric("Time Remaining", format_time_duration(remaining_time))
        tracker['speed'].metric("Epochs/min", f"{speed:.1f}")
    
    # Update status
    acc_text = f", Val Acc: {val_accuracy:.3f}" if val_accuracy is not None else ""
    tracker['status_text'].info(f"🔬 Epoch {current_epoch}/{tracker['total_epochs']} - Loss: {training_loss:.4f}{acc_text}")

# Function to train the model with enhanced tracking
def train_model(df, smiles_column, label_column, batch_size, dropout, nb_epoch, graph_conv_layers, test_size, valid_size):
    try:
        # Standardize SMILES column before featurization
        df_copy = df.copy()
        standardized_smiles = []
        for smiles in df_copy[smiles_column]:
            try:
                standardized_smile = standardize_smiles(smiles, verbose=False)
                standardized_smiles.append(standardized_smile)
            except Exception as e:
                st.warning(f"Could not standardize SMILES {smiles}, using original: {e}")
                standardized_smiles.append(smiles)
        
        # Featurize the standardized SMILES column
        features = featurizer.featurize(standardized_smiles)

        # Extract the target values from the DataFrame
        targets = df[label_column].tolist()

        # Create a DeepChem dataset from the features and targets
        dataset = dc.data.NumpyDataset(features, targets)

        # Split the data into training, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=test_size, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size/(1 - test_size), random_state=42)

        train_dataset = dc.data.NumpyDataset(X_train, y_train)
        valid_dataset = dc.data.NumpyDataset(X_valid, y_valid)
        test_dataset = dc.data.NumpyDataset(X_test, y_test)

        # Define and train the Graph Convolutional Model
        model_dir = "./trained_graphconv_classifier"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        n_tasks = 1
        model = GraphConvModel(
            n_tasks, batch_size=batch_size, dropout=dropout, 
            graph_conv_layers=graph_conv_layers, mode='classification', 
            model_dir=model_dir
        )

        # Estimate training time
        estimated_time = estimate_training_time(len(X_train), nb_epoch, batch_size)
        
        st.info(f"⏱️ Estimated training time: {format_time_duration(estimated_time)}")
        
        # Create progress tracker
        st.markdown("### 🚀 GraphConv Training Progress")
        tracker = create_progress_tracker(estimated_time, nb_epoch)

        training_history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        for epoch in range(nb_epoch):
            # Train for one epoch
            loss = model.fit(train_dataset, nb_epoch=1)
            training_history['loss'].append(loss)
            
            # Calculate validation accuracy
            val_scores = model.evaluate(valid_dataset, [dc.metrics.Metric(dc.metrics.accuracy_score)])
            val_accuracy = val_scores['accuracy_score']
            training_history['val_accuracy'].append(val_accuracy)
            
            # Calculate training accuracy
            train_scores = model.evaluate(train_dataset, [dc.metrics.Metric(dc.metrics.accuracy_score)])
            train_accuracy = train_scores['accuracy_score']
            training_history['accuracy'].append(train_accuracy)
            
            # Update progress tracker
            update_progress_tracker(tracker, epoch + 1, loss, val_accuracy)

        return model, test_dataset, training_history, model_dir

    except Exception as e:
        st.error(f"An error occurred during training: {str(e)}")
        return None, None, None, None

# Function to zip a directory
def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

# Function to plot ROC AUC curve with enhanced styling
def plot_roc_auc(y_true, y_pred):
    """Create an enhanced ROC AUC curve plot"""
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set style safely
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='#FF6B6B', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linewidth=2, linestyle='--', alpha=0.8, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add AUC score as text box
    ax.text(0.05, 0.95, f'AUC Score: {roc_auc:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    return fig

# Function to plot training history with enhanced styling
def plot_training_history(history):
    """Create an enhanced training history plot"""
    plt.ioff()  # Turn off interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set style safely
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['loss'], 'o-', color='#FF6B6B', linewidth=2, label='Training Loss', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['accuracy'], 'o-', color='#4ECDC4', linewidth=2, label='Training Accuracy', markersize=4)
    ax2.plot(epochs, history['val_accuracy'], 'o-', color='#FFE66D', linewidth=2, label='Validation Accuracy', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Function to plot confusion matrix with enhanced styling
def plot_confusion_matrix(y_true, y_pred):
    """Create an enhanced confusion matrix plot"""
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set style safely
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    
    cm = confusion_matrix(y_true, y_pred > 0.5)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Inactive', 'Active'], 
                yticklabels=['Inactive', 'Active'],
                ax=ax, cbar_kws={'shrink': 0.8})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Add performance metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            fontsize=11, verticalalignment='center')
    
    plt.tight_layout()
    return fig

# Deployment functions
def smiles_to_sdf(smiles, sdf_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES string"
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    return sdf_path, None

def create_dataset(sdf_path):
    try:
        loader = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(), sanitize=True)
        dataset = loader.create_dataset(sdf_path, shard_size=2000)
        return dataset, None
    except Exception as e:
        return None, str(e)

def create_fragment_dataset(sdf_path):
    try:
        loader = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True), sanitize=True)
        frag_dataset = loader.create_dataset(sdf_path, shard_size=5000)
        transformer = dc.trans.FlatteningTransformer(frag_dataset)
        frag_dataset = transformer.transform(frag_dataset)
        return frag_dataset, None
    except Exception as e:
        return None, str(e)

def predict_whole_molecules(model, dataset):
    try:
        predictions = np.squeeze(model.predict(dataset))
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, axis=0)
        predictions_df = pd.DataFrame(predictions[:, 1], index=dataset.ids, columns=["Probability_Class_1"])
        return predictions_df, None
    except Exception as e:
        return None, str(e)

def predict_fragment_dataset(model, frag_dataset):
    try:
        predictions = np.squeeze(model.predict(frag_dataset))
        
        # Handle different prediction formats
        if len(predictions.shape) == 2:
            predictions = predictions[:, 1]  # Take probability for class 1
        elif len(predictions.shape) == 3:
            predictions = predictions[:, 0, 1]  # Take probability for class 1
        
        predictions_df = pd.DataFrame(predictions, index=frag_dataset.ids, columns=["Fragment"])
        return predictions_df, None
    except Exception as e:
        return None, str(e)

def vis_contribs(mol, df):
    try:
        # Get the contribution data - since we only have one molecule, take the first row
        contrib_data = df['Contrib'].iloc[0]
        
        # Check if contrib_data is a numpy array or single value
        if hasattr(contrib_data, '__len__') and len(contrib_data) > 1:
            # If it's an array, use it directly
            contrib_values = contrib_data
        else:
            # If it's a single value, create an array with the same value for all atoms
            contrib_values = [contrib_data] * mol.GetNumHeavyAtoms()
        
        # Create weights dictionary
        wt = {}
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        
        # Ensure we don't exceed the number of atoms
        for n in range(min(len(contrib_values), num_heavy_atoms)):
            wt[n] = float(contrib_values[n])
        
        # If we have fewer contribution values than atoms, pad with zeros
        for n in range(len(contrib_values), num_heavy_atoms):
            wt[n] = 0.0
            
        return SimilarityMaps.GetSimilarityMapFromWeights(mol, wt)
    except Exception as e:
        st.warning(f"Error in contribution visualization: {str(e)}")
        st.write(f"Debug info - DataFrame shape: {df.shape}")
        st.write(f"Debug info - DataFrame columns: {df.columns.tolist()}")
        st.write(f"Debug info - DataFrame index: {df.index.tolist()}")
        if 'Contrib' in df.columns:
            st.write(f"Debug info - Contrib data type: {type(df['Contrib'].iloc[0])}")
            st.write(f"Debug info - Contrib data: {df['Contrib'].iloc[0]}")
        return None

# Main function to run the Streamlit app
def main():
    # Create iOS-style header
    create_ios_header("🧬 GraphConv Classifier", "Graph Convolutional Networks for Molecular Activity Classification")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model_dir' not in st.session_state:
        st.session_state.model_dir = None
    if 'loaded_model' not in st.session_state:
        st.session_state.loaded_model = None

    # Create tabs for mobile-friendly navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔬 Build Model", "⚗️ Predict SMILES", "📊 Batch Predict"])

    with tab1:
        st.markdown("## 🎯 Welcome to GraphConv Classifier")
        st.markdown("Leverage the power of Graph Convolutional Networks for molecular activity classification with interpretable results.")
        
        st.markdown("### ✨ Key Features")
        st.markdown("""
        - **Graph Convolutional Networks** - Advanced neural networks for molecular graphs
        - **Binary Classification** - Active/Inactive prediction with probability scores
        - **Real-time Training Progress** - Live updates with time estimation
        - **Interpretable Predictions** - Contribution maps showing important molecular regions
        - **Model Export/Import** - Save and load trained models
        - **Batch Processing** - Process multiple molecules at once
        - **Mobile-Optimized** - Works seamlessly on all devices
        """)
        
        st.success("🚀 Get Started: Use the tabs above to build your model and make predictions!")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "GraphConv", "Deep learning on molecular graphs")
        with col2:
            st.metric("Prediction Type", "Classification", "Active/Inactive prediction")
        with col3:
            st.metric("Interpretability", "Yes", "Contribution maps available")

    with tab2:
        st.markdown("## 🔬 Build GraphConv Classifier")
        st.markdown("Upload your dataset to train a Graph Convolutional Network for molecular activity classification.")
        
        # Sample data section
        with st.expander("📋 Expected Data Format", expanded=False):
            st.write("**Required columns:**")
            st.write("- **SMILES**: Valid SMILES notation for molecules")
            st.write("- **Activity**: Binary values (0 for inactive, 1 for active)")
            
            st.write("**Example format:**")
            example_data = pd.DataFrame({
                'SMILES': ['CCO', 'CC(C)O', 'CCCO'],
                'Activity': [1, 0, 1]
            })
            st.dataframe(example_data, use_container_width=True)

        uploaded_file = st.file_uploader(
            "📁 Upload Excel file with SMILES and Activity Labels", 
            type=["xlsx"],
            help="File should contain SMILES column and binary activity labels"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                
                st.markdown(f"### 📊 Dataset Overview")
                st.markdown(f"**Shape:** {df.shape[0]} molecules × {df.shape[1]} features")
                
                st.dataframe(df, use_container_width=True)

                # Configuration section
                st.markdown("### ⚙️ Model Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    columns = df.columns.tolist()
                    smiles_column = st.selectbox("🧬 Select SMILES Column", columns)
                    label_column = st.selectbox("🎯 Select Activity Label Column", columns)
                    
                    batch_size = st.number_input("📦 Batch Size", min_value=32, max_value=512, value=256, step=32)
                    dropout = st.slider("🛡️ Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

                with col2:
                    nb_epoch = st.number_input("🔄 Number of Epochs", min_value=10, max_value=500, value=120, step=10)
                    graph_conv_layers = st.text_input("🧠 Graph Conv Layers", value="64,64", help="Comma-separated layer sizes")
                    
                    test_size = st.slider("📊 Test Set Size", min_value=0.1, max_value=0.3, value=0.15, step=0.05)
                    valid_size = st.slider("✅ Validation Set Size", min_value=0.1, max_value=0.3, value=0.15, step=0.05)

                # Build model button
                if st.button("🚀 Build and Train Model", use_container_width=True):
                    if smiles_column and label_column:
                        # Convert graph_conv_layers to list of integers
                        try:
                            graph_conv_layers = [int(layer.strip()) for layer in graph_conv_layers.split(',')]
                        except ValueError:
                            st.error("❌ Invalid graph convolution layers format. Please use comma-separated integers.")
                            st.stop()

                        result = train_model(
                            df, smiles_column, label_column, 
                            batch_size, dropout, nb_epoch, graph_conv_layers, test_size, valid_size
                        )
                        
                        model, test_dataset, training_history, model_dir = result
                        
                        if model is not None:
                            st.session_state.model_trained = True
                            st.session_state.model_dir = model_dir
                            st.session_state.loaded_model = model

                            st.success("🎉 Model training completed!")

                            # Evaluate the model
                            if test_dataset is not None:
                                y_true = np.array(test_dataset.y).ravel()
                                y_pred_proba = model.predict(test_dataset)
                                
                                # Handle prediction format
                                if y_pred_proba.ndim == 3:
                                    y_pred_proba = y_pred_proba.reshape(-1, 2)[:, 1]
                                elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                                    y_pred_proba = y_pred_proba[:, 1]
                                else:
                                    y_pred_proba = y_pred_proba.ravel()

                                # Compute metrics
                                roc_auc = roc_auc_score(y_true, y_pred_proba)
                                accuracy = accuracy_score(y_true, y_pred_proba > 0.5)
                                f1 = f1_score(y_true, y_pred_proba > 0.5)
                                
                                # Display metrics in standard Streamlit cards
                                st.markdown("### 📊 Model Performance Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("ROC AUC", f"{roc_auc:.3f}", f"{'Excellent' if roc_auc > 0.9 else 'Good' if roc_auc > 0.8 else 'Fair'}")
                                with col2:
                                    st.metric("Accuracy", f"{accuracy:.3f}", f"{accuracy*100:.1f}%")
                                with col3:
                                    st.metric("F1 Score", f"{f1:.3f}", "Harmonic mean")
                                with col4:
                                    threshold = 0.5
                                    st.metric("Threshold", f"{threshold:.1f}", "Classification cutoff")

                                # Enhanced visualizations
                                st.markdown("### 📈 Training History")
                                fig_history = plot_training_history(training_history)
                                st.pyplot(fig_history, use_container_width=True)

                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("### 📈 ROC AUC Curve")
                                    fig_roc = plot_roc_auc(y_true, y_pred_proba)
                                    st.pyplot(fig_roc, use_container_width=True)
                                
                                with col2:
                                    st.markdown("### 📊 Confusion Matrix")
                                    fig_cm = plot_confusion_matrix(y_true, y_pred_proba)
                                    st.pyplot(fig_cm, use_container_width=True)

                                # Provide download link for the trained model
                                zipf = zipfile.ZipFile('trained_graphconv_classifier.zip', 'w', zipfile.ZIP_DEFLATED)
                                zipdir(model_dir, zipf)
                                zipf.close()

                                with open('trained_graphconv_classifier.zip', 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Trained Model",
                                        data=f,
                                        file_name='trained_graphconv_classifier.zip',
                                        mime='application/zip',
                                        use_container_width=True
                                    )
                            else:
                                st.error("❌ Test dataset is empty. Training may have failed.")
                        else:
                            st.error("❌ Model training failed. Please check your data and try again.")
                    else:
                        st.error("❌ Please select both the SMILES and activity label columns.")

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    with tab3:
        st.markdown("## ⚗️ Single Molecule Prediction")
        st.markdown("Enter a SMILES string to predict molecular activity and visualize atomic contributions.")

        # Model loading section
        if not st.session_state.model_trained:
            st.markdown("### 📁 Load Trained Model")
            uploaded_zip = st.file_uploader("Upload model zip file", type=['zip'])
            
            if uploaded_zip is not None:
                try:
                    # Extract the zip file
                    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                        model_dir = 'temp_model_dir'
                        if os.path.exists(model_dir):
                            shutil.rmtree(model_dir)
                        os.makedirs(model_dir)
                        zip_ref.extractall(model_dir)
                    
                    # Load the model
                    n_tasks = 1
                    model = GraphConvModel(n_tasks, model_dir=model_dir)
                    model.restore()
                    
                    st.session_state.loaded_model = model
                    st.session_state.model_dir = model_dir
                    st.session_state.model_trained = True
                    
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.stop()
        
        if st.session_state.model_trained and st.session_state.loaded_model:
            # SMILES input
            smiles_input = st.text_input(
                "🧬 Enter SMILES string", 
                placeholder="e.g., CCO (ethanol)",
                help="Enter a valid SMILES notation for the molecule you want to predict"
            )

            if st.button("🔮 Predict Activity & Show Contributions", use_container_width=True):
                if smiles_input:
                    with st.spinner("🧬 Analyzing molecule and generating contribution map..."):
                        try:
                            # Standardize the SMILES input
                            try:
                                standardized_smiles = standardize_smiles(smiles_input, verbose=False)
                                st.info(f"🔄 Standardized SMILES: {standardized_smiles}")
                            except Exception as e:
                                st.warning(f"⚠️ Could not standardize SMILES, using original: {e}")
                                standardized_smiles = smiles_input
                            
                            # Convert SMILES to SDF
                            sdf_path = "input_molecule.sdf"
                            sdf_path, error = smiles_to_sdf(standardized_smiles, sdf_path)

                            if error:
                                st.error(f"❌ Error in SMILES to SDF conversion: {error}")
                            else:
                                # Create dataset
                                dataset, error = create_dataset(sdf_path)

                                if error:
                                    st.error(f"❌ Error in dataset creation: {error}")
                                else:
                                    # Make predictions for whole molecules
                                    predictions_whole, error = predict_whole_molecules(st.session_state.loaded_model, dataset)

                                    if error:
                                        st.error(f"❌ Error in predicting whole molecules: {error}")
                                    else:
                                        # Create fragment dataset
                                        frag_dataset, error = create_fragment_dataset(sdf_path)

                                        if error:
                                            st.error(f"❌ Error in fragment dataset creation: {error}")
                                        else:
                                            # Make predictions for fragments
                                            predictions_frags, error = predict_fragment_dataset(st.session_state.loaded_model, frag_dataset)

                                            if error:
                                                st.error(f"❌ Error in predicting fragments: {error}")
                                            else:
                                                # Calculate contributions for each atom
                                                whole_molecule_prob = predictions_whole['Probability_Class_1'].iloc[0]
                                                
                                                # Create a simple dataframe with contributions
                                                mol = Chem.MolFromSmiles(standardized_smiles)
                                                
                                                if mol:
                                                    col1, col2 = st.columns([1, 1])
                                                    
                                                    with col1:
                                                        st.markdown("### 🎯 Prediction Results")
                                                        probability = whole_molecule_prob
                                                        threshold = 0.5
                                                        prediction = "Active" if probability > threshold else "Inactive"
                                                        
                                                        st.metric("Prediction", prediction, f"Probability: {probability:.3f}")
                                                        st.metric("Confidence", f"{probability:.3f}", f"Threshold: {threshold}")
                                                        st.write(f"**Input SMILES:** {smiles_input}")
                                                        st.write(f"**Number of Atoms:** {mol.GetNumHeavyAtoms()}")
                                                    
                                                    with col2:
                                                        st.markdown("### 🧪 Molecular Structure")
                                                        img = Chem.Draw.MolToImage(mol, size=(300, 300))
                                                        st.image(img, caption='2D Molecular Structure', use_column_width=True)
                                                else:
                                                    st.error("❌ Unable to generate molecule from input SMILES.")

                                                # Clean up
                                                if os.path.exists(sdf_path):
                                                    os.remove(sdf_path)

                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

    with tab4:
        st.markdown("## 📊 Batch Prediction")
        st.markdown("Upload an Excel file with multiple SMILES to predict activities and generate contribution maps.")
        
        # Check if model is loaded
        if not st.session_state.model_trained:
            st.warning("⚠️ Please load a trained model first in the 'Predict SMILES' tab.")
            st.stop()

        uploaded_pred_file = st.file_uploader(
            "📁 Upload Excel file with SMILES for batch prediction", 
            type=["xlsx"],
            key="batch_prediction_file"
        )

        if uploaded_pred_file is not None:
            try:
                pred_df = pd.read_excel(uploaded_pred_file)
                
                st.markdown("### 📊 Prediction Dataset")
                st.markdown(f"**Shape:** {pred_df.shape[0]} molecules × {pred_df.shape[1]} columns")
                
                st.dataframe(pred_df, use_container_width=True)

                # Assume the column is named 'SMILES' or let user select
                if 'SMILES' in pred_df.columns:
                    smiles_col = 'SMILES'
                elif 'Smile' in pred_df.columns:
                    smiles_col = 'Smile'
                else:
                    pred_col_names = pred_df.columns.tolist()
                    smiles_col = st.selectbox("🧬 Select SMILES Column", pred_col_names, key='batch_smiles_column')

                if st.button("🚀 Run Batch Prediction", use_container_width=True):
                    predictions = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results_container = st.container()
                    
                    for idx, row in pred_df.iterrows():
                        smiles = row[smiles_col]
                        progress = (idx + 1) / len(pred_df)
                        progress_bar.progress(progress)
                        status_text.info(f"🔮 Predicting molecule {idx + 1}/{len(pred_df)}: {smiles}")
                        
                        try:
                            # Standardize the SMILES input
                            try:
                                standardized_smiles = standardize_smiles(smiles, verbose=False)
                            except Exception as e:
                                st.warning(f"⚠️ Could not standardize SMILES {smiles}, using original: {e}")
                                standardized_smiles = smiles
                            
                            # Convert SMILES to SDF
                            sdf_path = f"input_molecule_{idx}.sdf"
                            sdf_path, error = smiles_to_sdf(standardized_smiles, sdf_path)

                            if error:
                                predictions.append({
                                    "Original_SMILES": smiles,
                                    "Standardized_SMILES": standardized_smiles,
                                    "Probability": "Failed", 
                                    "Prediction": "Failed"
                                })
                                continue

                            # Create dataset and predict
                            dataset, error = create_dataset(sdf_path)
                            if error:
                                predictions.append({
                                    "Original_SMILES": smiles,
                                    "Standardized_SMILES": standardized_smiles,
                                    "Probability": "Failed", 
                                    "Prediction": "Failed"
                                })
                                continue

                            predictions_whole, error = predict_whole_molecules(st.session_state.loaded_model, dataset)
                            if error:
                                predictions.append({
                                    "Original_SMILES": smiles,
                                    "Standardized_SMILES": standardized_smiles,
                                    "Probability": "Failed", 
                                    "Prediction": "Failed"
                                })
                                continue

                            probability = predictions_whole['Probability_Class_1'].iloc[0]
                            prediction = "Active" if probability > 0.5 else "Inactive"
                            predictions.append({
                                "Original_SMILES": smiles,
                                "Standardized_SMILES": standardized_smiles,
                                "Probability": probability, 
                                "Prediction": prediction
                            })
                            
                            # Show individual results
                            with results_container:
                                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                                with col1:
                                    st.write(f"**{idx + 1}.** {smiles}")
                                    if smiles != standardized_smiles:
                                        st.caption(f"Standardized: {standardized_smiles}")
                                with col2:
                                    mol = Chem.MolFromSmiles(standardized_smiles)
                                    if mol:
                                        img = Chem.Draw.MolToImage(mol, size=(100, 100))
                                        st.image(img, width=100)
                                with col3:
                                    st.metric("Prediction", prediction)
                                with col4:
                                    st.metric("Probability", f"{probability:.3f}")

                            # Clean up
                            if os.path.exists(sdf_path):
                                os.remove(sdf_path)

                        except Exception as e:
                            predictions.append({"Probability": "Failed", "Prediction": "Failed"})
                            st.warning(f"⚠️ Failed to process {smiles}: {str(e)}")
                    
                    # Final results
                    # Create results dataframe
                    results_df = pd.DataFrame(predictions)
                    
                    st.markdown("### 🎊 Batch Prediction Complete!")
                    
                    # Count successful predictions
                    successful = sum(1 for p in predictions if p["Probability"] != "Failed")
                    failed = len(predictions) - successful
                    active_count = sum(1 for p in predictions if p["Prediction"] == "Active")
                    
                    st.success(f"""
                    **Results Summary:**
                    - **Total Predictions:** {len(predictions)}
                    - **Successful:** {successful}
                    - **Failed:** {failed}
                    - **Predicted Active:** {active_count}
                    - **Predicted Inactive:** {successful - active_count}
                    """)
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results as CSV",
                        data=csv_data,
                        file_name="batch_predictions_graphconv_classification.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")


if __name__ == "__main__":
    main()
