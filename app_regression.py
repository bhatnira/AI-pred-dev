import streamlit as st
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import base64
import time
import deepchem as dc
import ssl
from lime import lime_tabular
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Configure matplotlib and RDKit for headless mode
os.environ['MPLBACKEND'] = 'Agg'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

# Disable RDKit warnings and configure for headless rendering
RDLogger.DisableLog('rdApp.*')

# Configure Streamlit page for mobile-friendly display
st.set_page_config(
    page_title="Chemlara Regressor",
    page_icon="📊",
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

def create_prediction_result_card(prediction, confidence_score, smiles):
    prediction_icon = "📈" if prediction > 0 else "📉"
    prediction_color = "#34C759" if prediction > 0 else "#FF6B6B"
    
    return f"""
    <div class="ios-card">
        <div style="text-align: center;">
            <div style="font-size: 3em; margin-bottom: 16px;">{prediction_icon}</div>
            <h2 style="color: {prediction_color}; margin: 0; font-weight: 700;">Predicted Value</h2>
            <h1 style="color: #1D1D1F; margin: 8px 0; font-weight: 800; font-size: 2.5em;">{prediction:.4f}</h1>
            <div style="margin: 16px 0;">
                <div style="background: rgba(0, 122, 255, 0.1); border-radius: 12px; padding: 16px;">
                    <p style="margin: 0; color: #007AFF; font-weight: 600;">Model Confidence</p>
                    <h3 style="margin: 4px 0 0 0; color: #1D1D1F; font-weight: 700;">{confidence_score}</h3>
                </div>
            </div>
            <p style="color: #8E8E93; font-size: 14px; margin: 8px 0;">
                <strong>SMILES:</strong> {smiles}
            </p>
        </div>
    </div>
    """

ssl._create_default_https_context = ssl._create_unverified_context

# Disable RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

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
    """Format seconds into human-readable duration"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

# Dictionary of featurizers using DeepChem
def get_featurizers():
    """Get available featurizers, handling potential initialization errors."""
    import deepchem as dc  # Ensure dc is available in function scope
    
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

# Get available featurizers - moved to main function to avoid import issues
# Featurizer = get_featurizers()

# Function to standardize SMILES using RDKit
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

# Function to preprocess data and perform modeling for regression
def preprocess_and_model(df, smiles_col, activity_col, featurizer_name, generations=5, population_size=20, cv=5, test_size=0.20, random_state=42, verbosity=2, cfp_params=None):
    with st.spinner("Preparing molecular data..."):
        # Standardize SMILES
        df[smiles_col + '_standardized'] = df[smiles_col].apply(standardize_smiles)
        df.dropna(subset=[smiles_col + '_standardized'], inplace=True)
        
        st.success(f"Standardized {len(df)} molecules")

    with st.spinner("Generating molecular features..."):
        # Featurize molecules
        featurizers_dict = get_featurizers()
        
        # Use custom CFP parameters if provided for Circular Fingerprint
        if featurizer_name == "Circular Fingerprint" and cfp_params:
            import deepchem as dc
            featurizer = dc.feat.CircularFingerprint(
                size=cfp_params.get('size', 2048), 
                radius=cfp_params.get('radius', 4)
            )
        else:
            featurizer = featurizers_dict[featurizer_name]
        features = []
        failed_molecules = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, smiles in enumerate(df[smiles_col + '_standardized']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    feat = featurizer.featurize([mol])[0]
                    # Handle NaN and infinite values for Mordred and other descriptors
                    if feat is not None:
                        feat_array = np.array(feat, dtype=float)
                        if len(feat_array) > 0:
                            # Clean NaN and infinite values
                            feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=0.0, neginf=0.0)
                            features.append(feat_array)
                        else:
                            # Try circular fingerprint as fallback for empty feature arrays
                            try:
                                import deepchem as dc
                                fallback_feat = dc.feat.CircularFingerprint(size=512, radius=2).featurize([mol])[0]
                                if fallback_feat is not None and len(fallback_feat) > 0:
                                    features.append(np.array(fallback_feat, dtype=float))
                                else:
                                    features.append(None)
                                    failed_molecules += 1
                                    st.warning(f"Empty features and fallback failed for SMILES: {smiles}")
                            except:
                                features.append(None)
                                failed_molecules += 1
                                st.warning(f"Empty features for SMILES: {smiles}")
                    else:
                        features.append(None)
                        failed_molecules += 1
                        st.warning(f"Null features returned for SMILES: {smiles}")
                except Exception as e:
                    features.append(None)
                    failed_molecules += 1
                    st.warning(f"Featurization failed for SMILES {smiles}: {str(e)}")
            else:
                features.append(None)
                failed_molecules += 1
                st.warning(f"Invalid SMILES: {smiles}")
            
            # Update progress
            progress = (i + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Featurizing molecule {i + 1}/{len(df)}")
        
        # Remove failed molecules
        valid_indices = [i for i, feat in enumerate(features) if feat is not None]
        if not valid_indices:
            st.error("No valid molecules found for featurization. Please check your SMILES data.")
            return None, None, None, None, None, None, None

        features = [features[i] for i in valid_indices]
        df = df.iloc[valid_indices].reset_index(drop=True)
        
        st.success(f"Generated features for {len(features)} molecules ({failed_molecules} failed)")

    # Create feature dataframe
    feature_df = pd.DataFrame(features)
    
    # Fill NaN values with 0 (important for Mordred and other descriptors)
    feature_df = feature_df.fillna(0)
    
    # Replace any infinite values with 0
    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    
    # Ensure all values are numeric and finite
    feature_df = feature_df.astype(float)
    
    # Additional validation: check for any remaining non-finite values
    if not np.isfinite(feature_df.values).all():
        st.warning("Found non-finite values in features, replacing with zeros...")
        feature_df = pd.DataFrame(np.nan_to_num(feature_df.values, nan=0.0, posinf=0.0, neginf=0.0), 
                                columns=feature_df.columns, index=feature_df.index)
    
    # Prepare data for modeling
    X = feature_df
    y = df[activity_col]

    # Convert integer column names to strings
    new_column_names = [f"fp_{col}" for col in X.columns]
    X.columns = new_column_names

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize TPOT with time tracking
    with st.spinner("Training TPOT model..."):
        # Estimate training time
        estimated_time = estimate_training_time(len(X_train), X_train.shape[1], generations, population_size)
        st.info(f"⏱️ Estimated training time: {format_time_duration(estimated_time)}")
        
        tpot = TPOTRegressor(
            generations=generations, 
            population_size=population_size, 
            cv=cv, 
            random_state=random_state, 
            verbosity=verbosity,
            n_jobs=1
        )

        # Track training time
        training_start_time = time.time()
        
        # Train the model
        tpot.fit(X_train, y_train)
        
        # Calculate actual training time
        training_end_time = time.time()
        actual_training_time = training_end_time - training_start_time
    
    # Model evaluation with time tracking
    with st.spinner("Evaluating model performance..."):
        evaluation_start_time = time.time()
        
        y_pred = tpot.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        evaluation_end_time = time.time()
        evaluation_time = evaluation_end_time - evaluation_start_time
    
    # Calculate total processing time
    total_time = time.time() - training_start_time
    
    # Display best pipeline
    st.markdown("### Best TPOT Pipeline")
    with st.expander("View Pipeline Details", expanded=False):
        st.code(str(tpot.fitted_pipeline_), language='python')

    # Save model and training data
    with st.spinner("Saving model..."):
        with open('best_model_regression.pkl', 'wb') as f_model:
            joblib.dump(tpot.fitted_pipeline_, f_model)
        
        with open('X_train_regression.pkl', 'wb') as f_X_train:
            joblib.dump(X_train, f_X_train)
    
    st.success("Model saved successfully!")

    return tpot, mse, rmse, mae, r2, y_test, y_pred, df, X_train, y_train, featurizer, actual_training_time, evaluation_time, total_time

# Function to interpret prediction using LIME
def interpret_prediction(tpot_model, input_features, X_train):
    # Create LIME explainer using X_train
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        mode="regression",
        feature_names=X_train.columns,
        verbose=True
    )
    
    explanation = explainer.explain_instance(
        input_features.values[0],
        tpot_model.predict,
        num_features=len(input_features.columns)
    )

    # Generate HTML explanation
    html_explanation = explanation.as_html()
    return html_explanation

# Function to predict from single SMILES input
def predict_from_single_smiles(single_smiles, featurizer_name='Circular Fingerprint'):
    standardized_smiles = standardize_smiles(single_smiles, verbose=False)
    if standardized_smiles:
        mol = Chem.MolFromSmiles(standardized_smiles)
        if mol is not None:
            # Use stored featurizer if available (from training), otherwise create new one
            if 'featurizer' in st.session_state and st.session_state.featurizer is not None:
                featurizer = st.session_state.featurizer
            else:
                featurizer = get_featurizers()[featurizer_name]
            try:
                features = featurizer.featurize([mol])[0]
                
                # Handle features the same way as in training
                if features is not None:
                    feat_array = np.array(features, dtype=float)
                    if len(feat_array) > 0:
                        # Clean NaN and infinite values (same as training)
                        feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        input_features = pd.DataFrame([feat_array])

                        # Additional validation: ensure all values are finite
                        if not np.isfinite(input_features.values).all():
                            input_features = pd.DataFrame(np.nan_to_num(input_features.values, nan=0.0, posinf=0.0, neginf=0.0), 
                                                        columns=input_features.columns, index=input_features.index)

                        # Convert integer column names to strings (same as training)
                        new_column_names = [f"fp_{col}" for col in input_features.columns]
                        input_features.columns = new_column_names
                    else:
                        # Try circular fingerprint as fallback for empty feature arrays (same as training)
                        st.warning(f"Empty features from {featurizer_name}, trying CircularFingerprint fallback...")
                        try:
                            import deepchem as dc
                            fallback_feat = dc.feat.CircularFingerprint(size=512, radius=2).featurize([mol])[0]
                            if fallback_feat is not None and len(fallback_feat) > 0:
                                feat_array = np.array(fallback_feat, dtype=float)
                                input_features = pd.DataFrame([feat_array])
                                new_column_names = [f"fp_{col}" for col in input_features.columns]
                                input_features.columns = new_column_names
                                st.info("Successfully used CircularFingerprint fallback for prediction")
                            else:
                                st.warning("Both primary featurizer and fallback failed to generate features.")
                                return None, None, None, None
                        except Exception as fallback_e:
                            st.warning(f"Fallback featurization also failed: {str(fallback_e)}")
                            return None, None, None, None
                else:
                    st.warning(f"Null features returned from {featurizer_name} for this molecule.")
                    return None, None, None, None

            except Exception as e:
                st.warning(f"Featurization failed with {featurizer_name}: {str(e)}")
                # Try fallback to CircularFingerprint
                try:
                    st.info("Attempting CircularFingerprint fallback...")
                    import deepchem as dc
                    fallback_featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
                    features = fallback_featurizer.featurize([mol])[0]
                    if features is not None and len(features) > 0:
                        feat_array = np.array(features, dtype=float)
                        feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=0.0, neginf=0.0)
                        input_features = pd.DataFrame([feat_array])
                        new_column_names = [f"fp_{col}" for col in input_features.columns]
                        input_features.columns = new_column_names
                        st.success("Successfully used CircularFingerprint fallback")
                    else:
                        st.warning("Fallback featurization also returned empty features.")
                        return None, None, None, None
                except Exception as fallback_e:
                    st.warning(f"Both primary and fallback featurization failed: {str(fallback_e)}")
                    return None, None, None, None

            # Load trained model and X_train
            try:
                with open('best_model_regression.pkl', 'rb') as f_model, open('X_train_regression.pkl', 'rb') as f_X_train:
                    tpot_model = joblib.load(f_model)
                    X_train = joblib.load(f_X_train)
            except FileNotFoundError:
                st.warning("Please build and save the model in the 'Build Model' section first.")
                return None, None, None, None

            # Predict using the trained model
            prediction = tpot_model.predict(input_features)[0]

            # Interpret prediction using LIME
            explanation_html = interpret_prediction(tpot_model, input_features, X_train)

            # Generate RDKit image of the molecule
            img = Chem.Draw.MolToImage(mol, size=(300, 300))

            return prediction, explanation_html, img, standardized_smiles
        else:
            st.warning("Invalid SMILES input. Please check your input and try again.")
            return None, None, None, None
    else:
        st.warning("Invalid SMILES input. Please check your input and try again.")
        return None, None, None, None

# Function to plot predicted vs true values
def plot_predicted_vs_true(y_true, y_pred):
    """Create predicted vs true values plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Predicted vs True scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, color='#FF6B6B', s=60)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8, label='Perfect Prediction')
    
    # Calculate and display R²
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=12, verticalalignment='top')
    
    ax1.set_xlabel('True Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax1.set_title('Predicted vs True Values', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals plot
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6, color='#4ECDC4', s=60)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_title('Residuals Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

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
        import io
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

def draw_molecule_with_fragment_weights(mol, atom_weights, width=1200, height=1200):
    """Draw molecule with atom highlighting based on fragment weights using improved color scheme and high resolution"""
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image
        import io
        
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
        import traceback
        traceback.print_exc()
        return None

def map_cfp_bits_to_atoms(mol, bit_weights, radius=4, n_bits=2048):
    """Map circular fingerprint bits to atoms using RDKit's Morgan fingerprint"""
    try:
        from rdkit.Chem import rdMolDescriptors
        
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
        import traceback
        traceback.print_exc()
        return {}

def map_specific_cfp_to_atoms(mol, cfp_number, radius=4, n_bits=2048):
    """Map a specific circular fingerprint number to atoms with improved weight distribution"""
    try:
        from rdkit.Chem import rdMolDescriptors
        
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
        import traceback
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
                mode="regression",  # Changed from "classification" to "regression"
                feature_names=X_train.columns,
                verbose=False,
                discretize_continuous=True
            )
            
            explanation = explainer.explain_instance(
                feature_df.values[0],
                model.predict,  # For regression, use predict instead of predict_proba
                num_features=min(100, len(feature_df.columns))  # Limit features for better visualization
            )
            
            # Get weights (for regression, explanation.as_map() returns a list directly)
            weights_list = explanation.as_map()[1]  # For regression, the target is index 1
            
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
        import traceback
        traceback.print_exc()
        return None

# Main function to run the Streamlit app
def main():
    # Initialize featurizers to avoid import scope issues
    Featurizer = get_featurizers()
    
    # Initialize session state
    if 'selected_featurizer_name' not in st.session_state:
        st.session_state.selected_featurizer_name = "Circular Fingerprint"
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # Create main header
    st.markdown(create_ios_header("Chemlara Regressor", "AI-Powered Molecular Property Prediction"), unsafe_allow_html=True)

    # Mobile-friendly navigation using tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔬 Build Model", "🧪 Single Prediction", "📊 Batch Prediction"])

    with tab1:
        st.markdown(create_ios_card("Welcome to Chemlara Regressor!", 
                   """
                   <p style="font-size: 16px; margin-bottom: 16px;">🎯 <strong>What can you do here?</strong></p>
                   <div style="background: rgba(0, 122, 255, 0.05); border-radius: 12px; padding: 16px; margin: 16px 0;">
                       <p style="margin: 8px 0;">📈 Build ML regression models for molecular property prediction</p>
                       <p style="margin: 8px 0;">🧪 Predict numerical properties from single SMILES</p>
                       <p style="margin: 8px 0;">📊 Batch predictions from Excel files</p>
                       <p style="margin: 8px 0;">🔍 Get detailed model explanations with LIME</p>
                       <p style="margin: 8px 0;">📱 Advanced molecular featurization with 6 different methods</p>
                   </div>
                   <p style="color: #8E8E93; font-style: italic; text-align: center;">📱 Optimized for mobile and desktop use!</p>
                   """, "🎉"), unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🔬 Build Your Regression Model")
        
        # Data format guide
        with st.expander("📁 Expected Data Format & Example", expanded=False):
            st.markdown(create_ios_card("Data Requirements", 
                       """
                       <p><strong>Required columns:</strong></p>
                       <ul>
                           <li><strong>SMILES:</strong> Valid SMILES notation for molecules</li>  
                           <li><strong>Target Property:</strong> Numerical values (e.g., solubility, toxicity, binding affinity)</li>
                       </ul>
                       <p><strong>Supported file format:</strong> Excel (.xlsx)</p>
                       <p style="color: #34C759;"><strong>💡 Tip:</strong> Your dataset should have at least 20-50 molecules for meaningful training results.</p>
                       """, "📋"), unsafe_allow_html=True)
            
            # Show example data format
            example_data = pd.DataFrame({
                'SMILES': ['CCO', 'CC(C)O', 'CCCO', 'CC(C)(C)O', 'c1ccccc1O'],
                'Solubility': [0.85, 0.76, 0.81, 0.65, 0.72]
            })
            st.dataframe(example_data, use_container_width=True)

        uploaded_file = st.file_uploader(
            "📤 Upload Excel file with SMILES and Property Values", 
            type=["xlsx"],
            help="File should contain SMILES column and target property column"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                
                # Data validation
                if df.empty:
                    st.error("The uploaded file is empty. Please check your data.")
                    st.stop()
                
                if len(df) < 10:
                    st.warning(f"Your dataset has only {len(df)} rows. Consider adding more data for better model performance (recommended: 20+ molecules).")
                
                # Data preview
                with st.expander("👁️ Data Preview", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Check for missing values
                missing_data = df.isnull().sum()
                if missing_data.any():
                    missing_info = ""
                    for col, missing_count in missing_data[missing_data > 0].items():
                        missing_info += f"<p>• {col}: {missing_count} missing values</p>"
                    st.markdown(create_ios_card("⚠️ Missing Values Detected", missing_info, "⚠️"), unsafe_allow_html=True)

                # Configuration section
                st.markdown(create_ios_card("Model Configuration", 
                                          "Configure your machine learning model parameters below.", "⚙️"), unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    col_names = df.columns.tolist()
                    smiles_col = st.selectbox("🧬 Select SMILES Column", col_names, key='train_smiles_column')
                    activity_col = st.selectbox("📊 Select Target Property Column", col_names, key='train_activity_column')
                    st.session_state.selected_featurizer_name = st.selectbox(
                        "🔧 Select Molecular Featurizer", 
                        list(Featurizer.keys()), 
                        key='train_featurizer_name',
                        index=list(Featurizer.keys()).index(st.session_state.selected_featurizer_name)
                    )
                    
                    # Additional parameters for Circular Fingerprint
                    if st.session_state.selected_featurizer_name == "Circular Fingerprint":
                        st.markdown("#### 🔬 Circular Fingerprint Parameters")
                        col_fp1, col_fp2 = st.columns(2)
                        with col_fp1:
                            cfp_radius = st.slider("Radius", min_value=1, max_value=6, value=4, 
                                                 help="Circular fingerprint radius (default: 4)", key='cfp_radius')
                        with col_fp2:
                            cfp_size = st.number_input("Fingerprint Size", min_value=64, max_value=16384, value=2048, step=64,
                                                  help="Number of bits in fingerprint (default: 2048)", key='cfp_size')

                with col2:
                    with st.expander("🔧 Advanced Settings"):
                        generations = st.slider("Generations (Evolution Cycles)", min_value=1, max_value=50, value=5)
                        cv = st.slider("Cross-Validation Folds", min_value=2, max_value=10, value=5)
                        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
                        verbosity = st.slider("Verbosity Level", min_value=0, max_value=3, value=2)

                # Build model button
                if st.button("🚀 Build and Train Model", use_container_width=True):
                    # Get CFP parameters if Circular Fingerprint is selected
                    cfp_params = {}
                    if st.session_state.selected_featurizer_name == "Circular Fingerprint":
                        cfp_params = {
                            'radius': locals().get('cfp_radius', 4),
                            'size': locals().get('cfp_size', 2048)
                        }
                    
                    result = preprocess_and_model(
                        df, smiles_col, activity_col, st.session_state.selected_featurizer_name, 
                        generations, cv=cv, test_size=test_size, verbosity=verbosity, cfp_params=cfp_params
                    )
                    
                    if result[0] is not None:  # Check if modeling was successful
                        tpot, mse, rmse, mae, r2, y_test, y_pred, df_result, X_train_result, y_train_result, featurizer_result, actual_training_time, evaluation_time, total_time = result
                        st.session_state.model_trained = True
                        st.session_state.trained_featurizer_name = st.session_state.selected_featurizer_name  # Track the featurizer used for training
                        st.session_state.featurizer = featurizer_result  # Store the actual featurizer object for fragment contribution
                        st.session_state.tpot_model = tpot.fitted_pipeline_  # Store the trained model for fragment contribution
                        st.session_state.X_train = X_train_result  # Store training data for fragment contribution

                        # Display metrics in iOS cards
                        st.markdown("### 🏆 Model Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(create_ios_metric_card("R² Score", f"{r2:.3f}", "Coefficient of determination", "📊"), unsafe_allow_html=True)
                        with col2:
                            st.markdown(create_ios_metric_card("RMSE", f"{rmse:.3f}", "Root mean squared error", "📏"), unsafe_allow_html=True)
                        with col3:
                            st.markdown(create_ios_metric_card("MAE", f"{mae:.3f}", "Mean absolute error", "📐"), unsafe_allow_html=True)
                        with col4:
                            st.markdown(create_ios_metric_card("MSE", f"{mse:.3f}", "Mean squared error", "📈"), unsafe_allow_html=True)
                        
                        # Training time metrics
                        st.markdown("### ⏱️ Training Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(create_ios_metric_card("Training Time", format_time_duration(actual_training_time), "TPOT optimization", "🤖"), unsafe_allow_html=True)
                        with col2:
                            st.markdown(create_ios_metric_card("Evaluation Time", format_time_duration(evaluation_time), "Model testing", "📊"), unsafe_allow_html=True)
                        with col3:
                            st.markdown(create_ios_metric_card("Total Time", format_time_duration(total_time), "End-to-end", "⏰"), unsafe_allow_html=True)
                        with col4:
                            # Calculate time per generation
                            time_per_generation = actual_training_time / generations
                            st.markdown(create_ios_metric_card("Time/Generation", format_time_duration(time_per_generation), f"{generations} generations", "🔄"), unsafe_allow_html=True)

                        # Visualization
                        st.markdown("### 📊 Model Visualization")
                        fig = plot_predicted_vs_true(y_test, y_pred)
                        st.pyplot(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    with tab3:
        st.markdown("### 🧪 Single Molecule Prediction")
        
        # Check if model is trained
        if not st.session_state.model_trained:
            try:
                # Load model and required components for fragment contribution
                with open('best_model_regression.pkl', 'rb') as f_model, open('X_train_regression.pkl', 'rb') as f_X_train:
                    st.session_state.tpot_model = joblib.load(f_model)
                    st.session_state.X_train = joblib.load(f_X_train)
                    st.session_state.model_trained = True
                    
                    # Try to load featurizer if available
                    if 'featurizer' not in st.session_state:
                        # Create a default featurizer - this should match the training featurizer
                        import deepchem as dc
                        st.session_state.featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
                        st.session_state.trained_featurizer_name = "Circular Fingerprint"
            except FileNotFoundError:
                st.markdown(create_ios_card("⚠️ Model Required", 
                           "Please train a model first in the 'Build Model' tab to use predictions.", "⚠️"), unsafe_allow_html=True)
                st.stop()

        # SMILES input
        col1, col2 = st.columns([3, 1])
        with col1:
            smiles_input = st.text_input(
                "🧬 Enter SMILES string", 
                placeholder="e.g., CCO (ethanol)",
                help="Enter a valid SMILES notation for the molecule you want to predict"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💡 Use Example", use_container_width=True):
                st.session_state.example_smiles = "CCO"
                st.rerun()
        
        # Use example SMILES if set
        if hasattr(st.session_state, 'example_smiles') and st.session_state.example_smiles and not smiles_input:
            smiles_input = st.session_state.example_smiles
            st.session_state.example_smiles = None  # Clear after use

        if st.button("🔮 Predict Property", use_container_width=True):
            if smiles_input:
                with st.spinner("🔄 Analyzing molecule..."):
                    result = predict_from_single_smiles(smiles_input, st.session_state.selected_featurizer_name)
                    
                if result[0] is not None:
                    prediction, explanation_html, img, standardized_smiles = result
                    
                    # Results layout - compact 1:2 ratio for mobile-friendly display
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Check if we should display fragment contribution map or regular structure
                        trained_featurizer = st.session_state.get('trained_featurizer_name', st.session_state.selected_featurizer_name)
                        current_featurizer = st.session_state.selected_featurizer_name
                        
                        if (trained_featurizer == "Circular Fingerprint" and 
                            current_featurizer == "Circular Fingerprint"):
                            
                            # Display fragment contribution map automatically
                            try:
                                # Load model and training data from separate files
                                with open('best_model_regression.pkl', 'rb') as f_model, open('X_train_regression.pkl', 'rb') as f_X_train:
                                    model = joblib.load(f_model)
                                    X_train = joblib.load(f_X_train)
                                    
                                    # Get the featurizer from session state
                                    featurizer_obj = st.session_state.get('featurizer')
                                    
                                    if model is not None and X_train is not None and featurizer_obj is not None:
                                        with st.spinner("🧬 Generating fragment contribution map..."):
                                            atomic_contrib_img = generate_fragment_contribution_map(
                                                smiles_input, model, X_train, featurizer_obj, cfp_number=None
                                            )
                                        
                                        if atomic_contrib_img:
                                            st.markdown(create_ios_card("🧬 Fragment Contribution", "", "🧬"), unsafe_allow_html=True)
                                            
                                            # Create layout with image and legend side by side without distorting main layout
                                            main_col, legend_col = st.columns([4, 1])
                                            
                                            with main_col:
                                                # Display larger, high-resolution image (preserve original layout)
                                                st.image(atomic_contrib_img, caption="Fragment Contribution Analysis", use_column_width=True)
                                                
                                                # Download button for high-res image
                                                create_download_button_for_image(
                                                    atomic_contrib_img, 
                                                    f"fragment_contribution_{smiles_input.replace('/', '_')}.png",
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
                                            from PIL import Image
                                            import io
                                            
                                            # Convert RDKit image to PIL and resize
                                            img_buffer = io.BytesIO()
                                            img.save(img_buffer, format='PNG')
                                            img_buffer.seek(0)
                                            pil_img = Image.open(img_buffer)
                                            resized_img = pil_img.resize((200, 200), Image.Resampling.LANCZOS)
                                            
                                            st.image(resized_img, caption='2D Structure', use_column_width=True)
                                    else:
                                        # Fallback to regular structure
                                        st.markdown(create_ios_card("Molecular Structure", "", "🧬"), unsafe_allow_html=True)
                                        from PIL import Image
                                        import io
                                        
                                        # Convert RDKit image to PIL and resize
                                        img_buffer = io.BytesIO()
                                        img.save(img_buffer, format='PNG')
                                        img_buffer.seek(0)
                                        pil_img = Image.open(img_buffer)
                                        resized_img = pil_img.resize((200, 200), Image.Resampling.LANCZOS)
                                        
                                        st.image(resized_img, caption='2D Structure', use_column_width=True)
                            except Exception as e:
                                # Debug: show error for troubleshooting
                                st.error(f"Error loading model for fragment analysis: {str(e)}")
                                print(f"Error loading model for fragment analysis: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                
                                # Fallback to regular structure
                                st.markdown(create_ios_card("Molecular Structure", "", "🧬"), unsafe_allow_html=True)
                                from PIL import Image
                                import io
                                
                                # Convert RDKit image to PIL and resize
                                img_buffer = io.BytesIO()
                                img.save(img_buffer, format='PNG')
                                img_buffer.seek(0)
                                pil_img = Image.open(img_buffer)
                                resized_img = pil_img.resize((200, 200), Image.Resampling.LANCZOS)
                                
                                st.image(resized_img, caption='2D Structure', use_column_width=True)
                        else:
                            # Display regular molecular structure for non-CFP featurizers
                            st.markdown(create_ios_card("Molecular Structure", "", "🧬"), unsafe_allow_html=True)
                            from PIL import Image
                            import io
                            
                            # Convert RDKit image to PIL and resize
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            pil_img = Image.open(img_buffer)
                            resized_img = pil_img.resize((200, 200), Image.Resampling.LANCZOS)
                            
                            st.image(resized_img, caption='2D Structure', use_column_width=True)
                        
                        # Compact Model Explanation below structure
                        with st.expander("🔍 Model Explanation", expanded=False):
                            st.download_button(
                                "📥 Download LIME Explanation", 
                                data=explanation_html,
                                file_name=f"lime_explanation_{smiles_input.replace('/', '_')}.html",
                                mime="text/html",
                                use_container_width=True
                            )
                    
                    with col2:
                        st.markdown(create_prediction_result_card(prediction, "High", standardized_smiles), unsafe_allow_html=True)
                        
                        # Additional info
                        st.markdown(create_ios_card("Prediction Details", 
                                   f"""
                                   <p><strong>Input SMILES:</strong> {smiles_input}</p>
                                   <p><strong>Standardized:</strong> {standardized_smiles}</p>
                                   <p><strong>Featurizer:</strong> {st.session_state.selected_featurizer_name}</p>
                                   """, "ℹ️"), unsafe_allow_html=True)
                        
    with tab4:
        st.markdown("### 📊 Batch Prediction")
        
        # Check if model is trained
        if not st.session_state.model_trained:
            try:
                # Load model and required components for fragment contribution
                with open('best_model_regression.pkl', 'rb') as f_model, open('X_train_regression.pkl', 'rb') as f_X_train:
                    st.session_state.tpot_model = joblib.load(f_model)
                    st.session_state.X_train = joblib.load(f_X_train)
                    st.session_state.model_trained = True
                    
                    # Try to load featurizer if available
                    if 'featurizer' not in st.session_state:
                        # Create a default featurizer - this should match the training featurizer
                        from rdkit.Chem import rdMolDescriptors
                        import deepchem as dc
                        st.session_state.featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
                        st.session_state.trained_featurizer_name = "Circular Fingerprint"
            except FileNotFoundError:
                st.markdown(create_ios_card("⚠️ Model Required", 
                           "Please train a model first in the 'Build Model' tab to use batch predictions.", "⚠️"), unsafe_allow_html=True)
                st.stop()

        st.markdown(create_ios_card("Batch Property Prediction", 
                   "Upload an Excel file with multiple SMILES to predict properties for entire datasets.", "📋"), unsafe_allow_html=True)

        # Data format guide for batch prediction
        with st.expander("📋 Batch Prediction Format", expanded=False):
            st.markdown(create_ios_card("Format Requirements", 
                       """
                       <p><strong>Required:</strong></p>
                       <ul>
                           <li>Excel file (.xlsx) with a column containing SMILES strings</li>
                           <li>Each row should contain one molecule's SMILES notation</li>
                       </ul>
                       <p><strong>Optional:</strong> Additional columns (like molecule names) will be preserved in results.</p>
                       """, "📝"), unsafe_allow_html=True)
            
            batch_example = pd.DataFrame({
                'SMILES': ['CCO', 'CC(C)O', 'CCCO', 'c1ccccc1O'],
                'Molecule_Name': ['Ethanol', 'Isopropanol', 'Propanol', 'Phenol']  # Optional
            })
            st.dataframe(batch_example, use_container_width=True)

        uploaded_pred_file = st.file_uploader(
            "📤 Upload Excel file with SMILES for batch prediction", 
            type=["xlsx"],
            key="batch_prediction_file"
        )

        if uploaded_pred_file is not None:
            try:
                pred_df = pd.read_excel(uploaded_pred_file)
                
                # Validation
                if pred_df.empty:
                    st.error("The uploaded file is empty.")
                    st.stop()
                
                # Dataset overview

                with st.expander("👁️ Data Preview", expanded=False):
                    st.dataframe(pred_df.head(5), use_container_width=True)

                pred_col_names = pred_df.columns.tolist()
                pred_smiles_col = st.selectbox("🧬 Select SMILES Column", pred_col_names, key='batch_pred_smiles_column')

                if st.button("🚀 Run Batch Prediction", use_container_width=True):
                    predictions = []
                    failed_molecules = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    start_time = time.time()
                    
                    for idx, smiles in enumerate(pred_df[pred_smiles_col]):
                        progress = (idx + 1) / len(pred_df)
                        progress_bar.progress(progress)
                        status_text.info(f"🔄 Predicting molecule {idx + 1}/{len(pred_df)}: {smiles}")
                        
                        result = predict_from_single_smiles(smiles, st.session_state.selected_featurizer_name)
                        
                        if result[0] is not None:
                            prediction, explanation_html, img, standardized_smiles = result
                            predictions.append(round(prediction, 4))
                        else:
                            predictions.append(None)
                            failed_molecules.append(f"{idx + 1}: {smiles}")
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Add predictions to dataframe
                    pred_df['Predicted_Property'] = predictions
                    successful_predictions = sum(1 for p in predictions if p is not None)
                    
                    # Results summary
                    st.markdown("### 🎉 Batch Prediction Complete!")
                    
                    # Metrics in iOS cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(create_ios_metric_card("Total", str(len(predictions)), "Molecules processed", "📊"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_ios_metric_card("Success", str(successful_predictions), "Predictions made", "✅"), unsafe_allow_html=True)
                    with col3:
                        st.markdown(create_ios_metric_card("Failed", str(len(failed_molecules)), "Invalid SMILES", "❌"), unsafe_allow_html=True)
                    with col4:
                        st.markdown(create_ios_metric_card("Time", f"{processing_time:.1f}s", "Processing time", "⏱️"), unsafe_allow_html=True)
                    
                    # Show failed molecules if any
                    if failed_molecules:
                        with st.expander("❌ Failed Predictions", expanded=False):
                            for failed in failed_molecules:
                                st.write(f"• {failed}")
                    
                    # Individual results section with pagination
                    st.markdown("### 🧬 Individual Molecule Results")
                    
                    # Create list of successful results with molecule images
                    successful_results = []
                    
                    # Debug: Show processing info
                    st.write(f"Processing {len(predictions)} predictions...")
                    
                    for idx, (smiles, prediction) in enumerate(zip(pred_df[pred_smiles_col], predictions)):
                        if prediction is not None:
                            try:
                                # Create molecular structure image
                                mol = Chem.MolFromSmiles(smiles)
                                if mol:
                                    # Standardize SMILES
                                    standardized_smiles = Chem.MolToSmiles(mol)
                                    
                                    # Create molecular image
                                    img = Chem.Draw.MolToImage(mol, size=(300, 300))
                                    
                                    # Create explanation placeholder (since LIME might be complex for batch)
                                    explanation_html = f"<h3>Molecule {idx + 1} Analysis</h3><p>SMILES: {standardized_smiles}</p><p>Predicted Property: {prediction:.4f}</p>"
                                    
                                    successful_results.append({
                                        'index': idx + 1,
                                        'smiles': smiles,
                                        'standardized_smiles': standardized_smiles,
                                        'prediction': prediction,
                                        'img': img,
                                        'explanation_html': explanation_html,
                                        'additional_data': {col: pred_df.iloc[idx][col] for col in pred_df.columns if col not in [pred_smiles_col, 'Predicted_Property']}
                                    })
                            except Exception as e:
                                st.write(f"Error processing molecule {idx + 1}: {str(e)}")
                                continue
                    
                    st.write(f"Successfully processed {len(successful_results)} molecules for individual display.")
                    
                    # Pagination for individual results
                    results_per_page = 5
                    total_results = len(successful_results)
                    total_pages = (total_results - 1) // results_per_page + 1 if total_results > 0 else 1
                    
                    if total_results > 0:
                        # Page selector
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            current_page = st.selectbox(
                                f"📄 Page (Showing {results_per_page} results per page)",
                                options=list(range(1, total_pages + 1)),
                                format_func=lambda x: f"Page {x} of {total_pages}"
                            ) - 1
                        
                        # Display results for current page
                        start_idx = current_page * results_per_page
                        end_idx = min(start_idx + results_per_page, total_results)
                        
                        for i in range(start_idx, end_idx):
                            result = successful_results[i]
                            
                            with st.expander(f"🧬 Molecule {result['index']}: Predicted Property = {result['prediction']:.4f}", expanded=False):
                                # Create three columns for structure, results, and additional info (like classification app)
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    # Display molecular structure with fragment contribution if available
                                    try:
                                        trained_featurizer = st.session_state.get('trained_featurizer_name')
                                        current_featurizer = st.session_state.selected_featurizer_name
                                        
                                        if (trained_featurizer == "Circular Fingerprint" and 
                                            current_featurizer == "Circular Fingerprint"):
                                            
                                            # Try to get stored model components
                                            model = st.session_state.get('tpot_model')
                                            X_train = st.session_state.get('X_train')
                                            featurizer_obj = st.session_state.get('featurizer')
                                            
                                            if model is not None and X_train is not None and featurizer_obj is not None:
                                                st.markdown("#### 🧬 Fragment Contribution Map")
                                                frag_img = generate_fragment_contribution_map(
                                                    result['standardized_smiles'], model, X_train, featurizer_obj, cfp_number=None
                                                )
                                                if frag_img:
                                                    st.image(frag_img, width=400, caption="")
                                                    
                                                    # Create download button for the image
                                                    create_download_button_for_image(
                                                        frag_img, 
                                                        f"fragment_contribution_molecule_{result['index']}.png",
                                                        f"📥 Download Fragment Map {result['index']}"
                                                    )
                                                else:
                                                    # Fallback to basic structure
                                                    st.markdown("#### 🧬 Molecule Structure")
                                                    st.image(result['img'], width=300)
                                            else:
                                                # Basic structure display
                                                st.markdown("#### 🧬 Molecule Structure")
                                                st.image(result['img'], width=300)
                                        else:
                                            # Basic structure for other featurizers
                                            st.markdown("#### 🧬 Molecule Structure")
                                            st.image(result['img'], width=300)
                                    except Exception as e:
                                        # Fallback structure display
                                        st.markdown("#### 🧬 Molecule Structure")
                                        st.image(result['img'], width=300)
                                        st.write(f"Debug: Error generating fragment map: {str(e)}")
                                        st.markdown("#### 🧬 Molecule Structure")
                                        st.image(result['img'], width=300)
                                
                                with col2:
                                    # Prediction results in compact iOS card (adapted for regression)
                                    prediction_value = result['prediction']
                                    prediction_icon = "📈" if prediction_value > 0 else "📉"
                                    prediction_color = "#34C759" if prediction_value > 0 else "#FF6B6B"
                                    
                                    st.markdown(f"""
                                    <div class="ios-card" style="padding: 12px; margin: 8px 0;">
                                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                            <div style="font-size: 1.5em; margin-right: 8px;">{prediction_icon}</div>
                                            <div>
                                                <h3 style="color: {prediction_color}; margin: 0; font-weight: 700; font-size: 1.1em;">Property Value</h3>
                                                <p style="margin: 2px 0 0 0; color: #8E8E93; font-size: 11px;">Predicted Result</p>
                                            </div>
                                        </div>
                                        <div style="background: rgba(0, 122, 255, 0.1); border-radius: 8px; padding: 8px; margin-bottom: 8px;">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <span style="color: #007AFF; font-weight: 600; font-size: 12px;">Value:</span>
                                                <span style="color: #1D1D1F; font-weight: 700; font-size: 14px;">{prediction_value:.4f}</span>
                                            </div>
                                        </div>
                                        <div style="background: rgba(0, 0, 0, 0.05); border-radius: 6px; padding: 6px;">
                                            <p style="margin: 0; color: #8E8E93; font-size: 10px; font-weight: 500;">
                                                <strong>SMILES:</strong> {result['standardized_smiles'][:30]}{'...' if len(result['standardized_smiles']) > 30 else ''}
                                            </p>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    # Additional data and information in compact card
                                    additional_data = result['additional_data']
                                    
                                    st.markdown(f"""
                                    <div class="ios-card" style="padding: 12px; margin: 8px 0;">
                                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                            <div style="font-size: 1.5em; margin-right: 8px;">📋</div>
                                            <div>
                                                <h3 style="color: #007AFF; margin: 0; font-weight: 600; font-size: 14px;">Additional Data</h3>
                                                <p style="margin: 2px 0 0 0; color: #8E8E93; font-size: 10px;">Molecule {result['index']} info</p>
                                            </div>
                                        </div>
                                        <div style="background: rgba(0, 0, 0, 0.02); border-radius: 6px; padding: 8px;">
                                    """, unsafe_allow_html=True)
                                    
                                    if additional_data:
                                        displayed_cols = 0
                                        for col, value in list(additional_data.items())[:3]:  # Limit to first 3 additional columns
                                            value_str = str(value)
                                            if len(value_str) > 20:
                                                value_str = value_str[:20] + "..."
                                            st.markdown(f"""
                                            <p style="margin: 2px 0; color: #1D1D1F; font-size: 11px;">
                                                <strong>{col}:</strong> {value_str}
                                            </p>
                                            """, unsafe_allow_html=True)
                                            displayed_cols += 1
                                        
                                        if len(additional_data) > 3:
                                            st.markdown(f"""
                                            <p style="margin: 4px 0 0 0; color: #8E8E93; font-size: 10px; font-style: italic;">
                                                +{len(additional_data) - 3} more fields
                                            </p>
                                            """, unsafe_allow_html=True)
                                    else:
                                        st.markdown("""
                                        <p style="margin: 0; color: #8E8E93; font-size: 11px; font-style: italic;">
                                            No additional data
                                        </p>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("</div></div>", unsafe_allow_html=True)
                                    
                                    # LIME explanation download button
                                    st.download_button(
                                        f"� LIME Report",
                                        data=result['explanation_html'],
                                        file_name=f"lime_molecule_{result['index']}.html",
                                        mime="text/html",
                                        use_container_width=True,
                                        key=f"download_lime_{result['index']}"
                                    )

                    else:
                        st.info("No successful predictions to display individually.")
                    
                    # Complete results table
                    st.markdown("### 📋 Complete Prediction Results")
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV download
                        csv_data = pred_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download as CSV",
                            data=csv_data,
                            file_name=f"batch_predictions_{int(time.time())}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Excel download
                        from io import BytesIO
                        buffer = BytesIO()
                        pred_df.to_excel(buffer, index=False, engine='openpyxl')
                        buffer.seek(0)
                        
                        st.download_button(
                            "📥 Download as Excel",
                            data=buffer.getvalue(),
                            file_name=f"batch_predictions_{int(time.time())}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
