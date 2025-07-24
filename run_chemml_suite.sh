#!/bin/bash

# ChemML Suite Launcher
# This script activates the virtual environment and starts the main app

echo "🧬 Starting ChemML Suite..."
echo "================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv .venv"
    echo "Then run: source .venv/bin/activate"
    echo "And install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
echo "📦 Checking dependencies..."
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed!"
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Start the main app
echo "🚀 Launching ChemML Suite..."
echo "📱 Open your browser to view the iOS-style interface"
echo "🔗 URL will be displayed below:"
echo "================================"

streamlit run main_app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
