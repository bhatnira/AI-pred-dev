#!/bin/bash
# Script to activate the Python 3.10 virtual environment
# Usage: source activate_env.sh

source venv/bin/activate
echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "Virtual environment path: $VIRTUAL_ENV"
