#!/bin/bash

# Optimized start script for instant app launching
echo "🚀 Starting Chemlara Suite with instant launch optimization..."

# Set environment variables for optimal performance
export STREAMLIT_SERVER_HEADLESS=true
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

# Pre-warm Python imports and caches
echo "🔥 Pre-warming application..."
python3 -c "
import warnings
warnings.filterwarnings('ignore')

# Pre-import heavy libraries
import streamlit as st
import deepchem as dc
import pandas as pd
import numpy as np
from rdkit import Chem
import joblib

# Pre-initialize featurizers
try:
    featurizers = {
        'Circular Fingerprint': dc.feat.CircularFingerprint(size=2048, radius=4),
        'MACCSKeys': dc.feat.MACCSKeysFingerprint(),
        'modred': dc.feat.MordredDescriptors(ignore_3D=True),
        'rdkit': dc.feat.RDKitDescriptors(),
        'pubchem': dc.feat.PubChemFingerprint()
    }
    print('✅ Featurizers pre-loaded')
except Exception as e:
    print(f'⚠️ Featurizer warning: {e}')

print('🎉 Pre-warming completed!')
"

# Start Streamlit with optimized settings for instant response
echo "🌟 Launching Streamlit application..."
exec streamlit run main_app.py \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.maxUploadSize=200 \
    --server.enableWebsocketCompression=true \
    --runner.magicEnabled=false \
    --logger.level=warning
