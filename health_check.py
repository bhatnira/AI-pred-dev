import streamlit as st
import sys
import platform
from datetime import datetime

def health_check():
    """Simple health check for deployment verification"""
    
    st.title("🏥 ChemML Suite Health Check")
    
    # System information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 System Info")
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Platform:** {platform.platform()}")
        st.write(f"**Architecture:** {platform.architecture()[0]}")
        st.write(f"**Processor:** {platform.processor()}")
    
    with col2:
        st.subheader("⏰ Status")
        st.write(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Status:** 🟢 Healthy")
        st.write(f"**Uptime:** Available")
    
    # Package verification
    st.subheader("📦 Package Verification")
    packages = {
        'streamlit': 'streamlit',
        'rdkit': 'rdkit',
        'deepchem': 'deepchem',
        'tpot': 'tpot',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    for package_name, import_name in packages.items():
        try:
            imported_package = __import__(import_name)
            version = getattr(imported_package, '__version__', 'Unknown')
            st.success(f"✅ {package_name}: {version}")
        except ImportError:
            st.error(f"❌ {package_name}: Not installed")
        except Exception as e:
            st.warning(f"⚠️ {package_name}: Error - {str(e)}")
    
    # Navigation link
    st.markdown("---")
    st.info("🏠 [Return to Main Application](/) ")

if __name__ == "__main__":
    health_check()
