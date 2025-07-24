#!/bin/bash

# Render.com build script for ChemML Suite
echo "Starting ChemML Suite build process..."

# Check Python version
echo "Current Python version:"
python --version
python3 --version

# Update pip
echo "Updating pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
python -m pip install -r requirements.txt

# Create necessary directories
mkdir -p /opt/render/project/src/.streamlit

# Set up Streamlit configuration
echo "Setting up Streamlit configuration..."

echo "Build completed successfully!"
