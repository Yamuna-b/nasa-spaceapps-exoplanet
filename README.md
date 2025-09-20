# 🪐 NASA Space Apps Exoplanet Detection AI

**Universal AI-Powered Exoplanet Detection & Analysis Platform**

*Upload Any Astronomical Dataset - We'll Find the Planets*

[![NASA Space Apps 2025](https://img.shields.io/badge/NASA%20Space%20Apps-2025-blue)](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Open_Source-green)](LICENSE)

***

## Challenge Overview

This project addresses the **NASA Space Apps 2025 Challenge: "A World Away: Hunting for Exoplanets with AI"** by creating the first **universal exoplanet analysis platform** that can automatically detect, classify, and analyze exoplanets from any astronomical dataset.

**Revolutionary Capability**: Scientists can upload **any exoplanet-related dataset** (Kepler, TESS, K2, ground-based observations, or custom data) and our AI will automatically:
- **Detect data format** and map columns intelligently
- **Classify exoplanet candidates** with confidence scores
- **Analyze confirmed planets** for habitability and characteristics  
- **Generate scientific insights** with publication-ready visualizations

**Perfect for**: Professional astronomers, graduate students, citizen scientists, educators, and space agencies who need quick, reliable exoplanet analysis without manual preprocessing.

***

## Features Completed

### **Universal Dataset Analysis**
- **Smart Data Detection**: Automatically recognizes any exoplanet dataset format
- **Intelligent Column Mapping**: Maps your columns to standard exoplanet features
- **Multi-Mission Support**: Kepler, TESS, K2, ground-based, and custom observations
- **Real-time Analysis**: Upload any CSV → Get instant exoplanet insights
- **Flexible Analysis Modes**: Candidate classification, habitability analysis, population studies

### **Advanced Visualizations & Analytics**
- **Model Comparison Dashboard**: Side-by-side performance evaluation
- **Interactive Feature Analysis**: Feature importance with scientific descriptions
- **Confidence Scoring**: Prediction reliability assessment
- **Export Functionality**: Download predictions as timestamped CSV files
- **Cross-Validation Metrics**: Robust model evaluation with statistical confidence

### **Professional Tools**
- **Hyperparameter Tuning Interface**: In-app model optimization
- **Enhanced Training Pipeline**: Advanced metrics collection for research-grade analysis
- **Comprehensive Documentation**: Scientific references, methodology, and usage guides
- **Responsive UI**: Professional Streamlit interface with intuitive navigation

### **Cross-Challenge Integration**
- **Biology Knowledge Engine**: NASA space biology research dashboard
- **Air Quality Forecaster**: Earth protection through ML-based air quality prediction
- **Research Integration**: Comprehensive scientific literature references

***

## Quick Start Guide

### **Step 1: Install Dependencies**
```bash
# Navigate to project directory
cd c:\Users\yamun\Desktop\Projects\Exoplanets

# Install required packages (Python 3.8+ required)
pip install -r requirements.txt
```

### **Step 2: Initialize Application**
```bash
# Run setup script (creates database, directories, .env file)
python setup.py
```

### **Step 3: Configure (Optional)**
```bash
# Edit .env file for real email notifications
notepad .env

# Add your email settings:
# SMTP_USERNAME=your-email@gmail.com
# SMTP_PASSWORD=your-app-password
```

### **Step 4: Launch Application**
```bash
# Start the Streamlit application
streamlit run app.py
```

**Access:** Open http://localhost:8501 in your browser

### **Features Available:**
- **Authentication System** - Sign up/Sign in with profile management
- **Email Verification** - Mock emails (configure SMTP for real emails)
- **Accessibility Features** - High contrast, large text, screen reader support
- **Help System** - Interactive tutorials and FAQ
- **Error Reporting** - Built-in issue reporting system
- **API Documentation** - Complete reference with examples

### **Access the Application**
- **Local**: http://localhost:8501
- **Features Available**: All 8 navigation tabs including cross-challenge integrations
- **Sample Data**: Built-in sample data for immediate testing

***

## Application Navigation

### **Home & Prediction**
- Upload CSV files with exoplanet candidate data
- Real-time predictions with confidence scores
- Built-in sample data for testing
- **Export functionality**: Download results as timestamped CSV
- **Explainability**: SHAP and LIME model interpretations

### **Model Performance** 
- Advanced metrics: Accuracy, Precision, Recall, Cross-validation
- **Confusion Matrix**: Detailed classification breakdown
- **ROC Curves**: Multi-class performance visualization
- **Classification Reports**: Per-class performance statistics

### **Feature Analysis**
- Interactive feature importance visualization
- Scientific descriptions of exoplanet parameters
- Top 10 most predictive features analysis

### **Model Comparison**
- Side-by-side Random Forest vs Logistic Regression comparison
- Performance metrics visualization
- Model selection guidance

### **Hyperparameter Tuning**
- Interactive parameter optimization interface
- Real-time performance impact simulation
- Best parameter recommendations

### **Biology Knowledge Engine** *(Cross-Challenge)*
- NASA space biology research dashboard
- Searchable publication database
- Key findings from decades of space experiments

### **Air Quality Forecaster** *(Cross-Challenge)*
- ML-based air quality predictions
- Multi-city coverage with health recommendations
- Real-time environmental monitoring

### **About & References**
- Comprehensive project documentation
- Scientific methodology and references
- Technical specifications and usage guides

***

## Data Sources & Methodology

### **Datasets Supported**
| Mission | Dataset | Features | Size | Status |
|---------|---------|----------|------|--------|
| **Kepler** | [KOI Catalog](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) | 8 key features | ~9,000 candidates | ✅ Active |
| **TESS** | [TOI Catalog](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI) | 8 key features | ~6,000 candidates | ✅ Active |
| **K2** | [K2 Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc) | 8 key features | ~1,000 candidates | ✅ Active |

### **Machine Learning Pipeline**
1. **Data Preprocessing**: Automated cleaning, normalization, feature engineering
2. **Feature Selection**: Orbital period, planet radius, transit depth, duration, etc.
3. **Model Training**: Random Forest (ensemble) & Logistic Regression (linear)
4. **Evaluation**: Cross-validation, confusion matrices, ROC analysis
5. **Explainability**: SHAP values, LIME explanations, feature importance
6. **Deployment**: Interactive Streamlit interface with real-time predictions

### **Target Classifications**
- **CONFIRMED**: Verified exoplanets
- **CANDIDATE**: Potential exoplanets requiring further analysis  
- **FALSE POSITIVE**: Non-planetary signals (eclipsing binaries, etc.)

***

## 🔬 Scientific References

### **Key Research Papers**
- **Malik et al. (2021)**: "Machine Learning for Exoplanet Detection in Kepler Data" - *MNRAS* [[Link]](https://academic.oup.com/mnras/article/513/4/5505/6472249)
- **Luz et al. (2022)**: "Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification" - *Electronics* [[Link]](https://www.mdpi.com/2079-9292/13/19/3950)
- **NASA Exoplanet Archive**: Official documentation and data processing handbooks [[Link]](https://exoplanetarchive.ipac.caltech.edu/)

### **Technical Stack**
- **Backend**: Python 3.8+, scikit-learn, pandas, numpy
- **Frontend**: Streamlit, Plotly, matplotlib
- **ML Explainability**: SHAP, LIME
- **Data Processing**: joblib, tqdm
- **Deployment**: Local/cloud-ready Streamlit application

***

## 🏆 NASA Space Apps Challenge Compliance

### **Primary Challenge: "A World Away: Hunting for Exoplanets with AI"**
✅ **Requirement**: Create AI/ML model trained on NASA open-source datasets  
✅ **Requirement**: Automatically analyze new data to identify exoplanets  
✅ **Requirement**: Use multiple datasets (Kepler, TESS, K2)  
✅ **Requirement**: Provide accurate predictions with confidence metrics  
✅ **Requirement**: Enable user interaction and data upload  

### **Bonus: Cross-Challenge Integration**
✅ **Biology Knowledge Engine**: Space biology research dashboard  
✅ **Air Quality Forecaster**: Earth protection through environmental monitoring  
✅ **Educational Impact**: Comprehensive documentation for students and researchers  

***

## 📄 License & Attribution

**License**: MIT License - Open source for educational and research purposes

**Data Attribution**: All datasets courtesy of NASA Exoplanet Archive

**Developed for**: NASA Space Apps Challenge 2025

**Contact**: Built with ❤️ for space exploration and scientific discovery

***
