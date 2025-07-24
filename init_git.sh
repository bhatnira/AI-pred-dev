#!/bin/bash

# Git initialization script for ChemML Suite
echo "🔧 Initializing Git repository for ChemML Suite..."

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "📋 Initializing new Git repository..."
    git init
    
    # Set up initial commit
    git add .
    git commit -m "Initial commit: ChemML Suite - iOS-style multipage ML app

Features:
- AutoML Activity and Potency Prediction
- Graph Convolution Models  
- Multiple Featurizers (RDKit, DeepChem)
- iOS-style responsive interface
- LIME model interpretability
- Batch processing capabilities
- Render.com deployment ready

Technology Stack:
- Streamlit + Custom CSS
- TPOT AutoML
- RDKit + DeepChem
- scikit-learn
- Matplotlib + Seaborn"

    echo "✅ Initial commit created"
else
    echo "📁 Git repository already exists"
fi

# Display current status
echo "📊 Current repository status:"
git status --short

echo ""
echo "🚀 Next steps for Render.com deployment:"
echo "1. Create a new repository on GitHub"
echo "2. Add remote: git remote add origin <your-repo-url>"  
echo "3. Push code: git push -u origin main"
echo "4. Connect repository to Render.com"
echo "5. Deploy as Web Service"
echo ""
echo "🔗 Render.com will use these files:"
echo "   - render.yaml (service configuration)"
echo "   - requirements.txt (dependencies)"
echo "   - Procfile (start command)"
echo "   - runtime.txt (Python version)"
echo ""
echo "✨ Your ChemML Suite is ready for deployment!"
