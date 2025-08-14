# 🧪 Chemlara Suite - AI-Powered Chemistry Activity Prediction

[![Python 3.10.12](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/streamlit-web_app-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15.0-orange.svg)](https://tensorflow.org/)
[![RDKit](https://img.shields.io/badge/rdkit-chemistry-purple.svg)](https://www.rdkit.org/)

A comprehensive machine learning suite for chemistry activity prediction with a beautiful iOS-style interface. This application provides AutoML capabilities, Graph Convolution Neural Networks, and advanced molecular analysis tools.

## 🐳 **Docker-First Approach**

**Chemlara Suite is designed to run with Docker** - no complex Python setup required!

- ✅ **One-command setup** - Works on Windows, macOS, and Linux
- ✅ **No Python installation needed** - Docker handles everything
- ✅ **All 116 packages included** - RDKit, TensorFlow, scikit-learn, and more
- ✅ **Consistent environment** - Same experience for all users
- ✅ **Production-ready** - Container can be deployed anywhere

**🚀 Get started in 30 seconds:**
```bash
git clone https://github.com/bhatnira/AI-Activity-Prediction.git
cd AI-Activity-Prediction
docker-compose up --build
# Open http://localhost:8501 🎉
```

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop**: [Download & Install](https://www.docker.com/products/docker-desktop/) ⭐ **Required**
- **Git**: [Install Git](https://git-scm.com/downloads) (optional - can download ZIP instead)

> 💡 **That's it!** No Python installation needed. Docker handles everything.

### 📥 Clone & Run (Recommended)

**One-line setup:**
```bash
git clone https://github.com/bhatnira/AI-Activity-Prediction.git && cd AI-Activity-Prediction && docker-compose up --build
```

**Step-by-step:**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhatnira/AI-Activity-Prediction.git
   cd AI-Activity-Prediction
   ```

2. **Build and start the application:**
   ```bash
   docker-compose up --build
   ```
   > 🕐 First run takes 2-3 minutes to build. Subsequent runs are instant!

3. **Access the application:**
   - Open your browser and go to: **http://localhost:8501**
   - The iOS-style Chemlara Suite interface will load automatically

### 📦 Alternative: Download ZIP

If you don't have Git:
1. Download ZIP: [https://github.com/bhatnira/AI-Activity-Prediction/archive/refs/heads/main.zip](https://github.com/bhatnira/AI-Activity-Prediction/archive/refs/heads/main.zip)
2. Extract and navigate to folder
3. Run: `docker-compose up --build`

### 🛑 Stop the Application

```bash
# Stop the container
docker-compose down

# Stop and remove all data
docker-compose down -v
```

## 🏗️ Architecture

### **Technology Stack**
- **🐍 Python 3.10.12** - Core runtime environment
- **🧪 RDKit** - Chemistry informatics and molecular analysis
- **🤖 TensorFlow 2.15.0** - Deep learning and neural networks
- **📊 scikit-learn** - Traditional machine learning algorithms
- **⚡ XGBoost** - Gradient boosting framework
- **📱 Streamlit** - Modern web interface
- **🐳 Docker** - Containerization and deployment

### **Key Features**
- **📱 iOS-Style Interface** - Beautiful purple-blue gradient design
- **🔬 AutoML Classification** - Automated machine learning for classification tasks
- **📈 AutoML Regression** - Automated regression analysis
- **🧬 Graph Convolution Networks** - Advanced molecular graph analysis
- **⚛️ Chemistry-Specific Tools** - Molecular descriptors, fingerprints, and analysis
- **📊 Interactive Visualizations** - Real-time plotting and data exploration

## 📦 Included Packages

The Chemlara Suite includes **116 carefully selected packages** for comprehensive chemistry and ML analysis:

### **Core Chemistry & ML Libraries**
- **RDKit** - Molecular informatics
- **TensorFlow** - Deep learning
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib/seaborn** - Visualization
- **TPOT** - Automated ML

### **Advanced Libraries**
- **DGL (Deep Graph Library)** - Graph neural networks
- **Optuna** - Hyperparameter optimization
- **LightGBM/CatBoost** - Additional boosting algorithms
- **NetworkX** - Graph analysis
- **Plotly** - Interactive visualizations

## 🎯 Application Modules

### 1. **📱 Main Launcher** (`main_app.py`)
- iOS-style interface with gradient design
- Navigation to all ML modules
- Responsive card-based layout

### 2. **🤖 AutoML Classification** (`app_classification.py`)
- Automated feature engineering
- Model selection and optimization
- Classification performance metrics

### 3. **📊 AutoML Regression** (`app_regression.py`)
- Automated regression pipeline
- Feature importance analysis
- Regression performance evaluation

### 4. **🧬 Graph Classification** (`app_graph_classification.py`)
- Molecular graph neural networks
- Graph convolution layers
- Chemical property prediction

### 5. **📈 Graph Regression** (`app_graph_regression.py`)
- Graph-based regression models
- Molecular property prediction
- Advanced graph analytics

## 🔧 Development Setup

### **🐳 Docker Development** (Recommended)

Docker is the **preferred way** to run Chemlara Suite locally. It ensures consistent environment across all platforms and handles all dependencies automatically.

#### **Quick Start with Docker:**
```bash
# Clone and start in one command
git clone https://github.com/bhatnira/AI-Activity-Prediction.git
cd AI-Activity-Prediction
docker-compose up --build
```

#### **Development Commands:**
```bash
# Build the image manually
docker build -t chemlara-suite .

# Run with custom settings and live reload
docker run -p 8501:8501 -v $(pwd):/app chemlara-suite

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

#### **Advantages of Docker Approach:**
- ✅ **No Python setup required** - Works on any machine with Docker
- ✅ **Consistent environment** - Same Python 3.10.12 + all 116 packages
- ✅ **No dependency conflicts** - Isolated container environment
- ✅ **Easy cleanup** - Remove everything with one command
- ✅ **Production-ready** - Same environment for development and deployment

### **💻 Local Development** (Alternative - Advanced Users)

⚠️ **Note**: Local setup requires Python 3.10.12 and may have dependency conflicts. Docker is strongly recommended.

1. **Create Python environment:**
   ```bash
   python3.10 -m venv chemlara_env
   source chemlara_env/bin/activate  # On Windows: chemlara_env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run main_app.py
   ```

#### **Local Setup Challenges:**
- ❌ Requires exact Python 3.10.12 version
- ❌ Complex chemistry dependencies (RDKit, TensorFlow)
- ❌ Platform-specific build issues
- ❌ Dependency version conflicts
- ❌ Time-consuming setup process

## 📁 Project Structure

```
AI-Activity-Prediction/
├── 📱 main_app.py                 # iOS-style launcher interface
├── 🤖 app_classification.py       # AutoML classification module
├── 📊 app_regression.py           # AutoML regression module
├── 🧬 app_graph_classification.py # Graph neural networks (classification)
├── 📈 app_graph_regression.py     # Graph neural networks (regression)
├── 🐳 Dockerfile                  # Container configuration
├── 🐳 docker-compose.yml          # Service orchestration
├── 📦 requirements.txt            # Python dependencies (116 packages)
├── ⚙️ .streamlit/config.toml      # Streamlit configuration
├── 🚀 startup.py                  # Application startup script
├── 🏥 health_check.py             # Container health monitoring
└── 📚 README.md                   # This documentation
```

## 🎨 Interface Preview

The Chemlara Suite features a modern iOS-style interface with:
- **Purple-blue gradient backgrounds**
- **Rounded card designs**
- **Smooth animations and transitions**
- **Responsive layout for all screen sizes**
- **Intuitive navigation between modules**

## 🔍 Health Monitoring

The application includes built-in health checks:
- **Container health status**
- **Application responsiveness**
- **Port availability**
- **Service connectivity**

Check health status:
```bash
docker ps  # Look for "healthy" status
```

## 🚨 Troubleshooting

### **Docker Issues** (Most Common)

1. **Docker not installed:**
   ```bash
   # Install Docker Desktop from https://www.docker.com/products/docker-desktop/
   # Restart your computer after installation
   ```

2. **Port 8501 already in use:**
   ```bash
   # Kill process using port 8501
   lsof -ti:8501 | xargs kill -9
   # Or use a different port
   docker-compose up --build -p 8502:8501
   ```

3. **Docker build fails:**
   ```bash
   # Clean Docker cache and rebuild
   docker system prune -f
   docker-compose build --no-cache
   docker-compose up
   ```

4. **Container won't start:**
   ```bash
   # Check logs for errors
   docker-compose logs chemlara-suite
   
   # Restart Docker Desktop and try again
   docker-compose down
   docker-compose up --build
   ```

5. **Permission issues (Linux/macOS):**
   ```bash
   # Fix permissions
   sudo chmod +x startup.py health_check.py
   sudo chown -R $USER:$USER .
   ```

### **Advanced Troubleshooting**

- **Out of disk space**: Docker images are large (~2GB). Free up space with `docker system prune -a`
- **Memory issues**: Ensure Docker has at least 4GB RAM allocated
- **Network issues**: Check if port 8501 is blocked by firewall
- **Slow performance**: Allocate more CPU/memory to Docker Desktop

### **Verification Commands**

```bash
# Check Docker installation
docker --version
docker-compose --version

# Check if container is running
docker ps

# View application logs
docker-compose logs -f

# Test container health
docker inspect --format='{{.State.Health.Status}}' $(docker ps -q)
```

### **Performance Optimization**

- **Recommended**: 8GB+ RAM for optimal performance
- **Minimum**: 4GB RAM for basic functionality
- **CPU**: Multi-core recommended for ML training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RDKit Community** - Chemistry informatics tools
- **TensorFlow Team** - Deep learning framework
- **Streamlit Team** - Amazing web app framework
- **scikit-learn Community** - Machine learning library
- **Docker Team** - Containerization platform

## 📞 Support

- **Repository**: [GitHub Issues](https://github.com/bhatnira/AI-Activity-Prediction/issues)
- **Documentation**: This README file
- **Community**: GitHub Discussions

## 🐳 Why Docker?

### **For End Users:**
- **🎯 Zero Setup** - No Python, no dependencies, no conflicts
- **🔄 Consistent** - Same experience on Windows, macOS, Linux
- **🚀 Fast** - One command to start everything
- **🧹 Clean** - Easy to remove completely
- **🔒 Isolated** - Doesn't affect your system

### **For Developers:**
- **📦 Reproducible** - Same environment everywhere
- **🚢 Deployable** - Container runs anywhere
- **🔧 Maintainable** - Single Dockerfile for all dependencies
- **🧪 Testable** - Consistent testing environment
- **⚡ Efficient** - Cached layers for fast rebuilds

### **System Requirements:**
- **Minimum**: 4GB RAM, 5GB disk space
- **Recommended**: 8GB RAM, 10GB disk space
- **Platforms**: Windows 10+, macOS 10.14+, Linux (most distributions)

---

**🚀 Ready to explore chemistry with AI? One command to start:**

```bash
git clone https://github.com/bhatnira/AI-Activity-Prediction.git
cd AI-Activity-Prediction
docker-compose up --build
# Open http://localhost:8501 🎉
```

**🔗 Quick Links:**
- 🌐 **Live Demo**: http://localhost:8501 (after running)
- 📖 **Documentation**: This README
- 🐛 **Issues**: [GitHub Issues](https://github.com/bhatnira/AI-Activity-Prediction/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/bhatnira/AI-Activity-Prediction/discussions)