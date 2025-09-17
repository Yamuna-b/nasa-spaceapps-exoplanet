# 🪐 Exoplanet Detection with AI/ML
Here is a **ready-to-copy-paste professional README.md** (no emojis, with all standard sections) for your GitHub repository:

***

# nasa-spaceapps-exoplanet

Streamlit app for exoplanet classification using NASA public datasets

***

## Overview

This project is an interactive web application for detecting and classifying exoplanets using machine learning and NASA data. Built for the [NASA Space Apps Challenge 2025](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/), it enables students, researchers, and space enthusiasts to explore exoplanet predictions, analyze large datasets from space missions, and experiment with AI/ML—all in the browser.

***

## Features

- Supports Kepler, TESS, and K2 datasets (easy to add more)
- Clean UI built with Streamlit
- ML model training and inference (Random Forest / Logistic Regression)
- Dataset selector for multi-mission support
- Easy CSV upload – preview and map columns if needed
- Results with class prediction and confidence scores
- Model performance metrics (accuracy, cross-validation, feature importance, plots)
- Hyperparameter tuning and in-app retraining (coming soon)
- Built-in help, about, and documentation tabs
- Powered by Python, pandas, scikit-learn, plotly, joblib

***

## Quickstart

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Yamuna-b/nasa-spaceapps-exoplanet.git
   cd nasa-spaceapps-exoplanet
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. **Prepare data (e.g., for Kepler):**
   ```bash
   python data_preprocessing.py --dataset kepler --input data/kepler_koi.csv --output data/kepler_koi_cleaned.csv
   ```

4. **Train model:**
   ```bash
   python train_model.py --dataset kepler --input data/kepler_koi_cleaned.csv
   ```

5. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

***

## Data Sources

- [Kepler Objects of Interest (KOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [TESS Objects of Interest (TOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)
- [K2 Planets and Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)

***

## Methodology

- **Preprocessing:** Data cleaning, normalization, feature engineering, handling different column naming conventions.
- **Modeling:** Train Random Forest or Logistic Regression to classify candidates as Confirmed, Candidate, or False Positive.
- **Evaluation:** Accuracy, cross-validation, confusion matrix, ROC curve, feature importance.
- **UI:** Streamlit web app for dataset selection, upload, prediction, and visualization.
- **Explainability:** Results interpreted with feature plots; roadmap for SHAP/LIME explanations.

***

## References & Further Reading

- Malik et al., "Exoplanet detection using machine learning", MNRAS 513 (2022) [link](https://academic.oup.com/mnras/article/513/4/5505/6472249)
- Luz et al., "Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification", Electronics 2024 [link](https://www.mdpi.com/2079-9292/13/19/3950)
- NASA Exoplanet Archive [main portal](https://exoplanetarchive.ipac.caltech.edu/)

***
