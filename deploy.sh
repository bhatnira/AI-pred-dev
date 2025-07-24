#!/bin/bash
# Deployment script for ChemML Suite on Render.com

echo "🚀 ChemML Suite Deployment Script"
echo "=================================="

# Check Python version
echo "📋 Python version:"
python --version

# Install dependencies
echo "📦 Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Verify critical packages
echo "🔍 Verifying installation..."
python -c "import streamlit; print(f'✅ Streamlit {streamlit.__version__}')"
python -c "import rdkit; print(f'✅ RDKit {rdkit.__version__}')"
python -c "import deepchem; print(f'✅ DeepChem {deepchem.__version__}')"
python -c "import tpot; print(f'✅ TPOT {tpot.__version__}')"
python -c "import sklearn; print(f'✅ Scikit-learn {sklearn.__version__}')"

echo "✅ Deployment preparation complete!"
echo "🌐 Starting ChemML Suite..."
