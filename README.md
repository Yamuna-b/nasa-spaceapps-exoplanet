# 🪐 Exoplanet Detection with AI/ML

## Overview
A Streamlit web application that predicts exoplanet candidates using NASA mission datasets (Kepler first, extensible to TESS and K2) and machine learning.

## Event & Problem Statement
- **Event**: NASA Space Apps Challenge 2025 — the world’s largest annual global hackathon hosted by NASA and partner space agencies.
- **Challenge**: **A World Away: Hunting for Exoplanets with AI**
- **Source**: See the challenge page (Resources tab) on the NASA Space Apps site: [NASA Space Apps](https://www.spaceappschallenge.org/)
- **Goal**: Build a modular AI/ML app that can ingest public exoplanet mission data, preprocess features, train a classifier, and provide interactive predictions, metrics, and explainability.

## Datasets & Resources (use from the official challenge resources)
- **Kepler Objects of Interest (KOI)** — NASA Exoplanet Archive: [Kepler KOI Overview](https://exoplanetarchive.ipac.caltech.edu/docs/KOI_kepcandidate.html)
- **TESS Objects of Interest (TOI)** — NASA Exoplanet Archive: [TESS TOI Overview](https://exoplanetarchive.ipac.caltech.edu/docs/TOI.html)
- **K2 Planets & Candidates** — NASA Exoplanet Archive: [K2 Catalogs](https://exoplanetarchive.ipac.caltech.edu/docs/K2Catalogs.html)
- **Partner/related**:
  - Canadian NEOSSat data (astronomical images) — see Space Apps Resources
  - JWST overview — [JWST Mission](https://jwst.nasa.gov/)
- **Background reading** (examples):
  - Exoplanet detection using machine learning — see Space Apps Resources
  - Assessment of ensemble-based ML algorithms for exoplanet identification — see Space Apps Resources

> Note: For judging/completeness, cite the specific dataset files and the Space Apps challenge page Resources tab used for download.

## Features
- Interactive web UI (Streamlit) for predictions and data exploration
- Dataset selector (Kepler now; TESS/K2 ready)
- File upload for candidate CSVs
- Model metrics: accuracy, confusion matrix, feature importances (and ROC as optional)
- Explainability-ready: feature importance; optional SHAP/LIME
- Optional in-app hyperparameter tuning and retraining

## Project Structure
```
Exoplanets/
  data/
    raw/                # Place raw CSVs here (Kepler/TESS/K2)
    processed/          # Saved cleaned/processed per-dataset CSVs
  app.py                # Streamlit app
  data_preprocessing.py # Cleaning/feature engineering per-dataset
  train_model.py        # Training, evaluation, model persistence per-dataset
  requirements.txt
  README.md
  venv/ (optional local virtual environment)
```

## Master Checklist (Roadmap)
1) Project Setup
- Create folders: `data/raw`, `data/processed`
- Scripts: `data_preprocessing.py`, `train_model.py`, `app.py` (optionally `utils/`)
- Keep `requirements.txt` and `README.md` updated

2) Data Resources to Include
- Kepler KOI, TESS TOI, K2 Candidates (from Space Apps Resources / NASA Exoplanet Archive)
- Partner datasets (e.g., NEOSSat) and relevant research articles for references

3) Data Preparation/Cleaning (per dataset)
- Remove header comments/metadata lines (e.g., starting with `#`)
- Drop irrelevant columns (IDs, names) and non-predictive text
- Handle missing values: numeric (median/mean), categorical (mode/encoding)
- Outlier handling (IQR method for numeric features)
- Select relevant features. For KOI, examples include:
  - `koi_period`, `koi_prad`, `koi_depth`, `koi_duration`, `koi_ingress`, `koi_dror`, `koi_count`, `koi_num_transits`
  - Target: `koi_disposition` (encode classes: CONFIRMED, CANDIDATE, FALSE POSITIVE)
- Scale numeric features with `StandardScaler` (persist scaler)
- Save processed CSV per dataset to `data/processed/`

4) Model Training & Saving (per dataset)
- Split train/test (e.g., 80/20)
- Train baseline classifier (RandomForestClassifier); optional Logistic Regression
- Evaluate: accuracy, precision, recall, F1; save confusion matrix
- Persist artifacts:
  - Trained model
  - Fitted scaler and label encoder
  - Feature importance (and optional CV results)

5) Streamlit Web App
- Core:
  - Dataset selector (Kepler, optionally TESS/K2 when available)
  - CSV uploader for candidate prediction
  - Preview uploaded data
  - Predict button → returns class label and confidence per row
- Metrics & Visuals:
  - Show test metrics, confusion matrix plot
  - Feature importance bar chart; optional ROC curve
- Advanced:
  - Hyperparameter sliders (e.g., `n_estimators`, `max_depth`)
  - Retrain with uploaded labeled data; allow downloading model/results
- Explainability:
  - Built-in feature importance
  - Optional SHAP/LIME explanations per prediction
- Help/About:
  - Dataset meanings, how to use app, references to Space Apps Resources and papers

6) Modular, Extensible Architecture
- Separate processing and model artifacts per dataset (Kepler/TESS/K2)
- Add a new dataset by adding processing + training + registering in app selector

7) Documentation & Submission
- Clear run instructions (local/deploy), list dependencies, and how to add datasets
- Cite NASA/partner datasets and challenge page
- Prepare a short demo video showing multi-dataset workflow

## Installation
Prereqs: Python 3.10+ (Windows, macOS, Linux)

```bash
# From project root
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# macOS/Linux
# source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Getting Data
- Download datasets from the Space Apps challenge page (Resources tab) or NASA Exoplanet Archive links above
- Place raw CSV files under `data/raw/` (e.g., `data/raw/kepler_koi.csv`)

## Preprocess & Train
```bash
# Example: preprocess and train Kepler dataset
python data_preprocessing.py --dataset kepler --input data/raw/kepler_koi.csv --output data/processed/kepler_processed.csv
python train_model.py --dataset kepler --input data/processed/kepler_processed.csv --model artifacts/kepler_model.pkl --scaler artifacts/kepler_scaler.pkl --encoder artifacts/kepler_encoder.pkl --metrics artifacts/kepler_metrics.json
```

> Adjust flags to match the script interfaces in this repo. For additional datasets, repeat with `--dataset tess` or `--dataset k2` once supported.

## Run the App
```bash
streamlit run app.py
```

- Use the sidebar to select the dataset (Kepler initially)
- Upload a CSV of candidates to classify
- View predictions, confidence scores, metrics, and plots

## Explainability (Optional)
If you enable SHAP/LIME in the app:
- Ensure `shap`/`lime` are installed (already in `requirements.txt`)
- Compute and cache explanations for faster UI

## Deployment (Optional)
- Streamlit Community Cloud, Render, or similar PaaS
- Provide environment setup and a small demo dataset to run easily

## Attribution
- This project responds to the NASA Space Apps Challenge 2025 challenge: **A World Away: Hunting for Exoplanets with AI**.
- Data and documentation links reference NASA/partner missions and the NASA Exoplanet Archive. Always review license/usage notes on the respective pages.

## License
Specify your preferred open-source license (e.g., MIT) if applicable.
