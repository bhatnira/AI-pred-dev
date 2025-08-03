#!/usr/bin/env python3
"""
ChemML Suite Setup Verification Script
=====================================

This script verifies that all essential components are properly installed
and configured for the ChemML Suite application.
"""

import sys
import importlib
import subprocess
import platform

def check_python_version():
    """Check if Python version is 3.10.x"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 10:
        print("✅ Python 3.10.x detected - Perfect!")
        return True
    else:
        print("⚠️  Recommended Python version is 3.10.x")
        return False

def check_essential_packages():
    """Check if essential packages are available"""
    essential_packages = [
        ('streamlit', 'Streamlit Web Framework'),
        ('rdkit', 'RDKit Chemistry Library'),
        ('tensorflow', 'TensorFlow Deep Learning'),
        ('sklearn', 'scikit-learn ML Library'),
        ('pandas', 'Pandas Data Analysis'),
        ('numpy', 'NumPy Numerical Computing'),
        ('matplotlib', 'Matplotlib Plotting'),
        ('plotly', 'Plotly Interactive Plots')
    ]
    
    print("\n📦 Checking Essential Packages:")
    print("-" * 40)
    
    success_count = 0
    for package, description in essential_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package:<12} - {description}")
            success_count += 1
        except ImportError:
            print(f"❌ {package:<12} - {description} (MISSING)")
    
    print(f"\n📊 Package Status: {success_count}/{len(essential_packages)} packages available")
    return success_count == len(essential_packages)

def check_docker_status():
    """Check if Docker is available and container is running"""
    print("\n🐳 Docker Status:")
    print("-" * 20)
    
    try:
        # Check if docker command is available
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
        else:
            print("❌ Docker: Not available")
            return False
            
        # Check if container is running
        result = subprocess.run(['docker', 'ps', '--filter', 'name=chemml-suite'], 
                              capture_output=True, text=True, timeout=5)
        if 'chemml-suite' in result.stdout:
            print("✅ ChemML Suite container: Running")
            return True
        else:
            print("⚠️  ChemML Suite container: Not running")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Docker: Not available or not responding")
        return False

def check_streamlit_config():
    """Check Streamlit configuration"""
    print("\n📱 Streamlit Configuration:")
    print("-" * 30)
    
    try:
        import streamlit as st
        print("✅ Streamlit: Available")
        
        # Check if config file exists
        import os
        config_path = ".streamlit/config.toml"
        if os.path.exists(config_path):
            print("✅ Streamlit config: Found")
            return True
        else:
            print("⚠️  Streamlit config: Missing")
            return False
            
    except ImportError:
        print("❌ Streamlit: Not available")
        return False

def main():
    """Main verification function"""
    print("🧪 ChemML Suite Setup Verification")
    print("=" * 40)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print()
    
    checks = []
    checks.append(check_python_version())
    checks.append(check_essential_packages())
    checks.append(check_docker_status())
    checks.append(check_streamlit_config())
    
    print("\n" + "=" * 40)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 40)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED! ChemML Suite is ready to use.")
        print("🚀 Access your application at: http://localhost:8501")
    else:
        print(f"⚠️  {passed}/{total} checks passed. Please address the issues above.")
        
    print("\n💡 Quick Start Commands:")
    print("   docker-compose up --build  # Start the application")
    print("   docker-compose down        # Stop the application")
    print("   python verify_setup.py     # Run this verification again")

if __name__ == "__main__":
    main()
