# ChemML Suite

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

A comprehensive iOS-style multipage Streamlit application for chemical machine learning, featuring AutoML activity and potency prediction using TPOT, RDKit, and DeepChem.

## � Quick Start with Docker (Recommended)

### Prerequisites
- Docker installed on your system
- Docker Compose (optional, for easier management)

### Run with Docker
```bash
# Clone the repository
git clone https://github.com/bhatnira/AI-Activity-Prediction.git
cd AI-Activity-Prediction

# Build and run with Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t chemml-suite .
docker run -p 8501:8501 chemml-suite

# Access at http://localhost:8501
```

## �🚀 Features

- **AutoML Activity Prediction**: Binary classification for chemical activity
- **AutoML Potency Prediction**: Regression modeling for chemical potency
- **Graph Convolution Models**: Advanced graph neural networks for molecular prediction
- **Multiple Featurizers**: Support for various molecular descriptors (Circular Fingerprints, MACCS Keys, Mordred, RDKit, PubChem, Mol2Vec)
- **Model Interpretability**: LIME explanations for prediction insights
- **iOS-Style Interface**: Modern, mobile-responsive design
- **Batch Processing**: Support for bulk predictions via Excel uploads

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom iOS-style CSS
- **Machine Learning**: TPOT (AutoML), scikit-learn
- **Chemistry**: RDKit, DeepChem
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker, Render.com

## 🐳 Docker Deployment

### Docker Commands
```bash
# Build the image
docker build -t chemml-suite .

# Run container (foreground)
docker run -p 8501:8501 chemml-suite

# Run container (background)
docker run -d -p 8501:8501 --name chemml chemml-suite

# View logs
docker logs chemml

# Stop and remove
docker stop chemml && docker rm chemml
```

### Docker Compose (Recommended)
```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down
```

### Production Deployment (Render.com)
The application is configured for automatic Docker deployment on Render.com:
1. Fork this repository
2. Connect to Render.com
3. The `render.yaml` file configures Docker deployment automatically
4. Uses Python 3.10 in a controlled Docker environment

## 📦 Installation

### Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd chemml-suite
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run main_app.py
```

## 🔧 Configuration Files

- `render.yaml`: Render.com service configuration
- `.streamlit/config.toml`: Streamlit configuration
- `Procfile`: Process file for deployment
- `runtime.txt`: Python version specification
- `requirements.txt`: Python dependencies

## 📱 Usage

### Main Navigation
1. **Home**: Overview and feature highlights
2. **AutoML Activity Prediction**: Build classification models
3. **AutoML Potency Prediction**: Build regression models
4. **Graph Convolution Models**: Advanced graph neural networks

### Within Each App
1. **Build Model**: Upload training data and configure AutoML
2. **Single Prediction**: Predict individual molecules
3. **Batch Prediction**: Process multiple molecules from Excel files

## 📊 Supported File Formats

- **Training Data**: Excel files (.xlsx) with SMILES and target columns
- **Prediction Data**: Excel files (.xlsx) with SMILES column
- **Output**: CSV files with predictions and confidence scores

## 🧪 Example SMILES

- Ethanol: `CCO`
- Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
- Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`

## 🎯 Model Features

- **Featurizers**: Circular Fingerprints, MACCS Keys, Mordred Descriptors, RDKit Descriptors, PubChem Fingerprints, Mol2Vec
- **AutoML**: TPOT optimization with customizable generations and cross-validation
- **Interpretability**: LIME explanations for model predictions
- **Visualization**: ROC curves, confusion matrices, performance metrics

## 🔍 Model Interpretability

Each prediction includes:
- Confidence scores
- LIME explanations (downloadable HTML)
- Feature importance analysis
- Model performance metrics

## 📈 Performance Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Regression**: R², MAE, MSE, RMSE
- **Visualizations**: ROC curves, confusion matrices, prediction plots

## 🛡️ Security & Privacy

- No data persistence between sessions
- Temporary file processing
- Local model training and prediction
- No external API calls for sensitive data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the GitHub Issues page
2. Create a new issue with detailed description
3. Include error messages and environment details

## 🔄 Updates

The application automatically updates when you push changes to your connected GitHub repository.

---

**Built with ❤️ using Streamlit, RDKit, and DeepChem**
