# ChemML Suite 🧬

An iOS-style multipage Streamlit application for chemical machine learning, featuring four specialized apps for molecular analysis and prediction.

## 🚀 Features

### 📱 iOS-Style Interface
- Clean, modern design inspired by iOS
- Smooth animations and transitions
- Mobile-responsive layout
- Glassmorphism effects with backdrop blur

### 🧪 Included Applications

1. **Classification** - Predict molecular properties using TPOT AutoML classification
2. **Regression** - Continuous molecular property prediction with regression models  
3. **Graph Classification** - Deep learning classification using molecular graph neural networks
4. **Graph Regression** - Deep learning regression with graph neural networks

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10
- All dependencies from `requirements.txt`

### Quick Start

1. **Install Dependencies** (if not already done):
   ```bash
   source .venv/bin/activate  # or activate your Python 3.10 environment
   pip install -r requirements.txt
   ```

2. **Launch the Suite**:
   ```bash
   # Option 1: Use the launcher script
   ./run_chemml_suite.sh
   
   # Option 2: Direct Streamlit command
   streamlit run main_app.py
   ```

3. **Open Your Browser**:
   - Navigate to the URL shown in terminal (typically `http://localhost:8501`)
   - Enjoy the iOS-style interface!

## 📱 Usage

### Home Screen
- View all available applications in an iOS-style grid
- Each app card shows an icon, title, and description
- Tap/click any app to launch it

### Navigation
- Use the "← Back" button to return to the home screen
- Each app runs independently with its own interface
- Seamless switching between applications

### Individual Apps
Each app maintains its original functionality:
- **Classification**: Upload datasets, train models, make predictions
- **Regression**: Continuous value prediction with visualization
- **Graph Classification**: Molecular graph-based classification
- **Graph Regression**: Graph neural network regression

## 🎨 Design Features

- **Glassmorphism**: Semi-transparent cards with backdrop blur
- **Gradient Backgrounds**: Beautiful color gradients
- **Smooth Animations**: Hover effects and transitions
- **Mobile Responsive**: Works on phones, tablets, and desktops
- **Typography**: Uses SF Pro Display font (iOS system font)

## 📂 File Structure

```
├── main_app.py              # Main iOS-style launcher app
├── app_classification.py    # Classification application
├── app_regression.py        # Regression application
├── app_graph_classification.py  # Graph classification
├── app_graph_regression.py     # Graph regression
├── run_chemml_suite.sh      # Launch script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Customization

### Adding New Apps
1. Create your new app file (e.g., `app_new_feature.py`)
2. Add it to the `apps_config` dictionary in `main_app.py`
3. Specify title, icon, description, and file path

### Styling
- Modify the CSS in `main_app.py` to customize colors, fonts, or layout
- Icons use emoji - replace with custom SVGs if needed
- Gradients and colors can be adjusted in the CSS section

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed in your Python 3.10 environment
2. **Port Conflicts**: If port 8501 is busy, Streamlit will automatically use the next available port
3. **File Not Found**: Ensure all app files are in the same directory as `main_app.py`

### Performance Tips
- For large datasets, consider implementing data caching
- Use Streamlit's `@st.cache_data` for expensive computations
- Monitor memory usage with multiple apps running

## 📊 Requirements

See `requirements.txt` for the complete list of dependencies including:
- Streamlit 1.37.0
- RDKit 2024.3.3
- DeepChem 2.8.0
- TensorFlow 2.15.0
- TPOT 0.12.2
- And many more...

## 🤝 Contributing

Feel free to contribute by:
- Adding new machine learning applications
- Improving the iOS-style interface
- Optimizing performance
- Adding new features

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using Streamlit, RDKit, DeepChem, and TPOT**
