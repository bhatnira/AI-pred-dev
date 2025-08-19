#!/usr/bin/env python3
"""
Optimized startup script for instant app launching
Pre-loads all models, featurizers, and dependencies
"""

import sys
import time
import warnings
import os

# Suppress warnings for faster startup
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🚀 Starting Chemlara Suite - Optimized for instant launching...")

start_time = time.time()

# Pre-import all heavy libraries
print("📦 Loading core libraries...")
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
import deepchem as dc
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Pre-initialize DeepChem featurizers
print("🧪 Pre-loading molecular featurizers...")
try:
    # Initialize all featurizers to cache them
    featurizers = {
        "Circular Fingerprint": dc.feat.CircularFingerprint(size=2048, radius=4),
        "MACCSKeys": dc.feat.MACCSKeysFingerprint(),
        "modred": dc.feat.MordredDescriptors(ignore_3D=True),
        "rdkit": dc.feat.RDKitDescriptors(),
        "pubchem": dc.feat.PubChemFingerprint()
    }
    print(f"✅ Loaded {len(featurizers)} featurizers successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not pre-load some featurizers: {e}")

# Pre-load TPOT if available
print("🤖 Checking for pre-trained models...")
try:
    if os.path.exists('best_model.pkl') and os.path.exists('X_train.pkl'):
        with open('best_model.pkl', 'rb') as f_model:
            model = joblib.load(f_model)
        with open('X_train.pkl', 'rb') as f_X_train:
            X_train = joblib.load(f_X_train)
        print("✅ Pre-trained model loaded successfully")
    else:
        print("ℹ️ No pre-trained model found (will be created when needed)")
except Exception as e:
    print(f"⚠️ Warning: Could not pre-load model: {e}")

# Set up environment for optimal performance
print("⚙️ Optimizing environment...")
os.environ['MPLBACKEND'] = 'Agg'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

# Test molecular processing to warm up caches
print("🔥 Warming up molecular processing...")
try:
    test_smiles = "CCO"  # Simple ethanol molecule
    mol = Chem.MolFromSmiles(test_smiles)
    if mol:
        # Test featurization
        fp = featurizers["Circular Fingerprint"].featurize([mol])
        print("✅ Molecular processing warmed up")
except Exception as e:
    print(f"⚠️ Warning: Could not warm up processing: {e}")

elapsed_time = time.time() - start_time
print(f"🎉 Startup optimization completed in {elapsed_time:.2f} seconds")
print("🌟 Chemlara Suite is ready for instant launching!")

if __name__ == "__main__":
    print("Starting Streamlit application...")
    import subprocess
    
    # Start Streamlit with optimized settings
    cmd = [
        "streamlit", "run", "main_app.py",
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]
    
    subprocess.run(cmd)
