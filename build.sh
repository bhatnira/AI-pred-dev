#!/bin/bash

# Render.com build script for ChemML Suite
echo "Starting ChemML Suite build process..."

# Update pip
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p /opt/render/project/src/.streamlit

# Set up Streamlit configuration
echo "Setting up Streamlit configuration..."

echo "Build completed successfully!"
