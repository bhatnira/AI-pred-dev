import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import base64
import matplotlib.pyplot as plt
import time
import ssl
import deepchem as dc
from lime import lime_tabular
from sklearn.model_selection import train_test_split
import seaborn as sns

# Configure Streamlit page for mobile-friendly display
st.set_page_config(
    page_title="ChemML Predictor",
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
        border-radius: 16px;
        padding: 20px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
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

# iOS-style component functions
def create_ios_metric_card(title, value, description="", icon="📊"):
    return f"""
    <div class="ios-metric">
        <div style="font-size: 2em; margin-bottom: 8px;">{icon}</div>
        <h3 style="margin: 0; color: #007AFF; font-weight: 600; font-size: 14px;">{title}</h3>
        <h2 style="margin: 8px 0; color: #1D1D1F; font-weight: 700; font-size: 24px;">{value}</h2>
        <p style="margin: 0; color: #8E8E93; font-size: 12px; font-weight: 400;">{description}</p>
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

def create_prediction_result_card(prediction, probability, smiles):
    activity_icon = "🟢" if prediction == 1 else "🔴"
    activity_text = "Active" if prediction == 1 else "Not Active"
    confidence_color = "#34C759" if prediction == 1 else "#FF3B30"
    
    return f"""
    <div class="ios-card">
        <div style="text-align: center;">
            <div style="font-size: 3em; margin-bottom: 16px;">{activity_icon}</div>
            <h2 style="color: {confidence_color}; margin: 0; font-weight: 700;">{activity_text}</h2>
            <div style="margin: 16px 0;">
                <div style="background: rgba(0, 122, 255, 0.1); border-radius: 12px; padding: 16px;">
                    <p style="margin: 0; color: #007AFF; font-weight: 600;">Confidence Score</p>
                    <h3 style="margin: 4px 0 0 0; color: #1D1D1F; font-weight: 700;">{probability:.1%}</h3>
                </div>
            </div>
            <p style="color: #8E8E93; font-size: 14px; margin: 8px 0;">
                <strong>SMILES:</strong> {smiles}
            </p>
        </div>
    </div>
    """

# Dictionary of featurizers using DeepChem
Featurizer = {
    "Circular Fingerprint": dc.feat.CircularFingerprint(size=2048, radius=4),
    "MACCSKeys": dc.feat.MACCSKeysFingerprint(),
    "modred": dc.feat.MordredDescriptors(ignore_3D=True),
    "rdkit": dc.feat.RDKitDescriptors(),
    "pubchem":dc.feat.PubChemFingerprint(),
    "mol2vec":dc.feat.Mol2VecFingerprint()
}

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
    estimated_total_time = max(30, min(300, n_samples * 0.5 + generations * cv * 15))  # Smart estimation
    
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

        # Convert activity column to binary labels
        unique_classes = df[activity_col].unique()
        if len(unique_classes) < 2:
            st.error("Not enough classes present for binary classification. Please check your dataset and ensure it has at least two distinct classes.")
            return None, None, None, None, None, None, None, None, None, None

        update_progress_with_time(0.15, "Featurizing molecules...")
        
        # Featurize molecules with progress updates
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
            return None, None, None, None, None, None, None, None, None, None

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
        
        update_progress_with_time(0.9, "Evaluating model performance...")
        
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

        update_progress_with_time(1.0, "Training completed successfully!")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Clear progress and show completion
        time.sleep(2)
        progress_container.empty()
        st.success(f"🎉 Training completed in {format_time_duration(total_time)}!")
        
        # Display results in iOS-style cards
        if roc_auc and fpr is not None:
            # Create two columns for mobile-friendly layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot ROC curve with better styling
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
            
            with col2:
                # Confusion Matrix Heatmap with better styling
                try:
                    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax_cm, 
                               cbar_kws={'shrink': 0.8}, square=True, linewidths=0.5)
                    ax_cm.set_title('📈 Confusion Matrix', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_cm)
                except Exception as e:
                    st.warning(f"Could not generate confusion matrix: {str(e)}")
        else:
            st.info("ℹ️ ROC curve not available for this classification problem.")

        # Display best pipeline in a nice container
        st.markdown("### 🏆 Best TPOT Pipeline")
        with st.expander("🔍 View Pipeline Details", expanded=False):
            try:
                st.code(str(tpot.fitted_pipeline_), language='python')
            except:
                st.code("Pipeline details not available", language='text')

        # Model download section
        st.markdown("### 💾 Download Trained Model")
        
        # Save TPOT model and X_train separately
        model_filename = 'best_model.pkl'
        X_train_filename = 'X_train.pkl'

        try:
            with open(model_filename, 'wb') as f_model:
                joblib.dump(tpot.fitted_pipeline_, f_model)
            
            with open(X_train_filename, 'wb') as f_X_train:
                joblib.dump(X_train, f_X_train)

            # Create download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(create_downloadable_model_link(model_filename, '📥 Download Model'), unsafe_allow_html=True)
            with col2:
                st.markdown(create_downloadable_model_link(X_train_filename, '📥 Download Training Data'), unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not save model files: {str(e)}")

        # Get feature names used in modeling
        feature_names = list(X_train.columns)

        return tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer
    
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None, None, None, None

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

# Function to interpret prediction using LIME
def interpret_prediction(tpot_model, input_features, X_train):
    # Create LIME explainer using X_train
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

    # Generate HTML explanation
    html_explanation = explanation.as_html()
    return html_explanation

# Function to predict from single Smile input
def predict_from_single_smiles(single_smiles, featurizer_name='Circular Fingerprint'):
    standardized_smiles = standardize_smiles(single_smiles)
    mol = Chem.MolFromSmiles(standardized_smiles)
    
    if mol is not None:
        featurizer = Featurizer[featurizer_name]
        features = featurizer.featurize([mol])[0]
        feature_df = pd.DataFrame([features], columns=[f"fp_{i}" for i in range(len(features))])
        feature_df = feature_df.astype(float)

        # Load trained model and X_train
        try:
            with open('best_model.pkl', 'rb') as f_model, open('X_train.pkl', 'rb') as f_X_train:
                tpot_model = joblib.load(f_model)
                X_train = joblib.load(f_X_train)
        except FileNotFoundError:
            st.warning("Please build and save the model in the 'Build Model' section first.")
            return None, None

        # Predict using the trained model
        prediction = tpot_model.predict(feature_df)[0]
        probability = tpot_model.predict_proba(feature_df)[0][1] if hasattr(tpot_model, 'predict_proba') else None

        # Interpret prediction using LIME
        explanation_html = interpret_prediction(tpot_model, feature_df, X_train)

        return prediction, probability, explanation_html
    else:
        st.warning("Invalid Smile input. Please check your input and try again.")
        return None, None, None

# Main Streamlit application
def main():
    # Initialize selected featurizer name session variable
    if 'selected_featurizer_name' not in st.session_state:
        st.session_state.selected_featurizer_name = list(Featurizer.keys())[0]  # Set default featurizer

    # Create main header
    st.markdown(create_ios_header("ChemML Predictor", "AI-Powered Chemical Activity Prediction"), unsafe_allow_html=True)

    # Mobile-friendly navigation using tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔬 Build Model", "🧪 Single Prediction", "📊 Batch Prediction"])

    with tab1:
        st.markdown(create_ios_card("Welcome to ChemML Predictor!", 
                   """
                   <p style="font-size: 16px; margin-bottom: 16px;">🎯 <strong>What can you do here?</strong></p>
                   <div style="background: rgba(0, 122, 255, 0.05); border-radius: 12px; padding: 16px; margin: 16px 0;">
                       <p style="margin: 8px 0;">🔬 Build ML models for chemical activity prediction</p>
                       <p style="margin: 8px 0;">🧪 Predict activity from single SMILES</p>
                       <p style="margin: 8px 0;">📊 Batch predictions from Excel files</p>
                       <p style="margin: 8px 0;">📈 Get detailed model explanations with LIME</p>
                   </div>
                   <p style="color: #8E8E93; font-style: italic; text-align: center;">📱 Optimized for mobile and desktop use!</p>
                   """, "🎉"), unsafe_allow_html=True)
        
        # Quick stats or features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(create_ios_metric_card("Featurizers", "6", "Available options", "🔧"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_ios_metric_card("AutoML", "TPOT", "Powered by", "🤖"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_ios_metric_card("Explainable", "LIME", "AI interpretability", "🔍"), unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🔬 Build Your ML Model")
        
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
                    st.markdown(create_ios_card("Training in Progress", 
                                              "Processing data and training your machine learning model...", "🤖"), unsafe_allow_html=True)
                    
                    tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer = preprocess_and_model(
                        df, smiles_col, activity_col, st.session_state.selected_featurizer_name, 
                        generations=generations, cv=cv, verbosity=verbosity, test_size=test_size)

                    if tpot is not None:
                        # Display model metrics in cards
                        st.markdown("### 📈 Model Performance")
                        
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
                            st.success("✅ Model trained successfully!")

    with tab3:
        st.markdown("### 🧪 Single SMILES Prediction")
        
        smile_input = st.text_input("Enter SMILES string for prediction", 
                                  placeholder="e.g., CCO (ethanol)",
                                  help="Enter a valid SMILES string representing your molecule",
                                  label_visibility="collapsed")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            predict_button = st.button("🔮 Predict Activity", use_container_width=True)
        with col2:
            if st.button("🧹 Clear", use_container_width=True):
                st.rerun()

        if predict_button and smile_input:
            with st.spinner("🔍 Analyzing molecule..."):
                prediction, probability, explanation_html = predict_from_single_smiles(smile_input, st.session_state.selected_featurizer_name)
                
                if prediction is not None:
                    # Display results in iOS-style layout
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Display molecular structure
                        try:
                            mol = Chem.MolFromSmiles(smile_input)
                            if mol:
                                img = Draw.MolToImage(mol, size=(300, 300))
                                st.markdown('<div class="ios-card">', unsafe_allow_html=True)
                                st.image(img, use_column_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        except:
                            pass
                    
                    with col2:
                        # Prediction results in iOS card
                        st.markdown(create_prediction_result_card(prediction, probability, smile_input), unsafe_allow_html=True)
                    
                    # LIME explanation
                    if explanation_html:
                        st.markdown(create_ios_card("Model Explanation", 
                                                  f'<a href="data:text/html;base64,{base64.b64encode(explanation_html.encode()).decode()}" download="LIME_Explanation.html" style="color: #007AFF; text-decoration: none; font-weight: 600;">📥 Download Detailed Explanation</a>',
                                                  "�"), unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to make prediction. Please check your SMILES input.")

    with tab4:
        st.markdown("### 📊 Batch Prediction from File")
        
        with st.expander("📁 Upload Prediction File", expanded=True):
            uploaded_file = st.file_uploader("Upload Excel file with SMILES for batch prediction", 
                                            type=["xlsx"], key="batch_upload",
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
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                # Check if model exists
                try:
                    with open('best_model.pkl', 'rb') as f_model, open('X_train.pkl', 'rb') as f_X_train:
                        tpot_model = joblib.load(f_model)
                        X_train = joblib.load(f_X_train)
                except FileNotFoundError:
                    st.error("❌ No trained model found. Please build a model first in the 'Build Model' tab.")
                    return

                if smiles_col_predict in df.columns:
                    predictions = []
                    probabilities = []
                    
                    # iOS-style progress tracking
                    st.markdown(create_ios_card("Processing Molecules", 
                                              "Analyzing your molecules using the trained model...", "⚗️"), unsafe_allow_html=True)
                    
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
                                    featurizer = Featurizer[st.session_state.selected_featurizer_name]
                                    features = featurizer.featurize([mol])[0]
                                    feature_df = pd.DataFrame([features], columns=[f"fp_{i}" for i in range(len(features))])
                                    feature_df = feature_df.astype(float)

                                    # Predict
                                    prediction = tpot_model.predict(feature_df)[0]
                                    probability = tpot_model.predict_proba(feature_df)[0][1] if hasattr(tpot_model, 'predict_proba') else None

                                    predictions.append("Active" if prediction == 1 else "Not Active")
                                    probabilities.append(probability if probability is not None else 0.0)
                                else:
                                    predictions.append("Invalid SMILES")
                                    probabilities.append(0.0)
                            else:
                                predictions.append("Invalid SMILES")
                                probabilities.append(0.0)
                        except Exception as e:
                            predictions.append(f"Error: {str(e)}")
                            probabilities.append(0.0)

                    # Clear progress indicators
                    progress_container.empty()
                    st.success("🎉 Batch prediction completed successfully!")
                    
                    # Add results to dataframe
                    df['Predicted_Activity'] = predictions
                    df['Confidence'] = [f"{p:.1%}" if isinstance(p, float) else "N/A" for p in probabilities]

                    # Display results in iOS card
                    st.markdown(create_ios_card("Prediction Results", 
                                              "Your batch prediction results are ready!", "📊"), unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                    
                    # iOS-style download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='batch_predictions.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                    
                    # Summary statistics in iOS cards
                    st.markdown("### 📈 Summary Statistics")
                    active_count = sum(1 for p in predictions if p == "Active")
                    total_valid = sum(1 for p in predictions if p in ["Active", "Not Active"])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(create_ios_metric_card("Total Processed", str(len(df)), "molecules", "⚗️"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_ios_metric_card("Active Compounds", str(active_count), f"out of {total_valid}", "✅"), unsafe_allow_html=True)
                    with col3:
                        if total_valid > 0:
                            active_rate = (active_count / total_valid) * 100
                            st.markdown(create_ios_metric_card("Activity Rate", f"{active_rate:.1f}%", "predicted active", "📊"), unsafe_allow_html=True)
                else:
                    st.error("❌ SMILES column not found in the uploaded file.")

if __name__ == "__main__":
    main()
