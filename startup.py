#!/usr/bin/env python3
"""
Startup script for ChemML Suite
Handles initialization and environment setup for deployment
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and configurations"""
    
    # Set environment variables for Streamlit
    os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'false')
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION', 'false')
    
    # Create necessary directories
    directories = ['.streamlit', 'temp', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Directory created/verified: {directory}")

def verify_packages():
    """Verify that all required packages are available"""
    
    required_packages = [
        'streamlit',
        'rdkit',
        'deepchem', 
        'tpot',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    logger.info("🔍 Verifying package installations...")
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            logger.info(f"✅ {package}: Available")
        except ImportError:
            logger.error(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
    
    logger.info("✅ All packages verified successfully")
    return True

def main():
    """Main startup function"""
    
    logger.info("🚀 Starting ChemML Suite initialization...")
    
    # Setup environment
    setup_environment()
    
    # Verify packages
    if not verify_packages():
        logger.error("❌ Package verification failed")
        sys.exit(1)
    
    logger.info("✅ ChemML Suite initialization completed successfully")
    
    # Start the main application
    import subprocess
    port = os.environ.get('PORT', '8501')
    
    cmd = [
        'streamlit', 'run', 'main_app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    logger.info(f"🌐 Starting Streamlit on port {port}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
