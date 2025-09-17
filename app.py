import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
except ImportError:
    shap = None
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

# Configure page
st.set_page_config(
    page_title="Exoplanet Detection AI",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f4e79;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models_and_data(dataset_key: str, model_type: str = "rf"):
    """Load trained model and preprocessors for selected dataset"""
    try:
        artifacts_dir = 'artifacts'
        model_path = os.path.join(artifacts_dir, f"{dataset_key}_{model_type}_model.joblib")
        label_path = os.path.join(artifacts_dir, f"{dataset_key}_label_encoder.joblib")
        scaler_path = os.path.join(artifacts_dir, f"{dataset_key}_scaler.joblib")
        metrics_path = os.path.join(artifacts_dir, f"{dataset_key}_metrics.joblib")

        model = joblib.load(model_path)
        label_encoder = joblib.load(label_path)
        feature_scaler = joblib.load(scaler_path)
        metrics = joblib.load(metrics_path)
        return model, label_encoder, feature_scaler, metrics
    except FileNotFoundError as e:
        st.error(f"Artifacts not found for dataset '{dataset_key}': {e}")
        st.info("Run preprocessing and training first. See README instructions.")
        return None, None, None, None

def preprocess_user_data(data, feature_scaler, expected_features):
    """Preprocess user uploaded data"""
    # Ensure all expected features are present
    for feature in expected_features:
        if feature not in data.columns:
            data[feature] = data.get(feature, 0)  # Default to 0 if missing
    
    # Select only the features used in training
    data_features = data[expected_features]
    
    # Handle missing values
    data_features = data_features.fillna(data_features.median())
    
    # Scale features
    data_scaled = feature_scaler.transform(data_features)
    
    return data_scaled, data_features

def main():
    # Header
    st.markdown('<h1 class="main-header">🪐 Exoplanet Detection with AI/ML</h1>', unsafe_allow_html=True)
    st.markdown("**Predict exoplanet candidates using NASA's Kepler Objects of Interest dataset**")
    
    # Sidebar dataset/model selection
    st.sidebar.header("Configuration")
    dataset_key = st.sidebar.selectbox("Dataset", ["kepler", "tess", "k2"], index=0)
    model_type = st.sidebar.selectbox("Model", ["rf", "logreg"], index=0)

    # Load artifacts
    model, label_encoder, feature_scaler, metrics = load_models_and_data(dataset_key, model_type)
    
    if model is None:
        return
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home & Prediction",
        "📊 Model Performance",
        "🔍 Feature Analysis",
        "ℹ️ About"
    ])
    
    if page == "🏠 Home & Prediction":
        show_prediction_page(model, label_encoder, feature_scaler, metrics)
    elif page == "📊 Model Performance":
        show_performance_page(metrics)
    elif page == "🔍 Feature Analysis":
        show_feature_analysis_page(metrics)
    else:
        show_about_page()

def show_prediction_page(model, label_encoder, feature_scaler, metrics):
    st.header("Exoplanet Prediction")
    
    # Model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", f"{metrics['accuracy']:.1%}")
    with col2:
        st.metric("Cross-Validation", f"{metrics['cv_mean']:.1%}")
    with col3:
        st.metric("Features Used", len(metrics['feature_importance']))
    
    # File upload
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with exoplanet candidate data",
        type="csv",
        help="Upload a CSV file containing the same features used in training"
    )
    
    # Sample data for testing
    if st.button("Load Sample Data"):
        sample_data = pd.DataFrame({
            'koi_period': [10.5, 365.25, 88.0],
            'koi_prad': [1.2, 0.8, 2.1],
            'koi_depth': [100, 500, 200],
            'koi_duration': [2.5, 6.0, 1.8],
            'koi_ingress': [0.5, 1.2, 0.3],
            'koi_dror': [0.01, 0.005, 0.02],
            'koi_count': [1, 2, 1],
            'koi_num_transits': [50, 20, 100]
        })
        st.session_state.sample_data = sample_data
        uploaded_file = "sample"
    
    if uploaded_file is not None:
        if uploaded_file == "sample":
            data = st.session_state.sample_data
        else:
            data = pd.read_csv(uploaded_file)
        
        st.write("Data Preview:")
        st.dataframe(data.head())
        
        # Get expected features
        expected_features = [item['feature'] for item in metrics['feature_importance']]
        
        # Preprocess data
        try:
            data_scaled, data_features = preprocess_user_data(data, feature_scaler, expected_features)
            
            # Make predictions
            predictions = model.predict(data_scaled)
            prediction_proba = model.predict_proba(data_scaled)
            
            # Decode predictions
            prediction_labels = label_encoder.inverse_transform(predictions)
            
            # Create results dataframe
            results = data.copy()
            results['Prediction'] = prediction_labels
            results['Confidence'] = np.max(prediction_proba, axis=1)
            
            # Display results
            st.subheader("Prediction Results")
            
            # Summary
            prediction_counts = pd.Series(prediction_labels).value_counts()
            col1, col2, col3 = st.columns(3)
            
            for i, (pred, count) in enumerate(prediction_counts.items()):
                with [col1, col2, col3][i % 3]:
                    st.metric(pred, count)
            
            # Detailed results
            st.write("Detailed Predictions:")
            st.dataframe(results[['Prediction', 'Confidence']])
            
            # Confidence distribution
            fig = px.histogram(
                results, x='Confidence', color='Prediction',
                title="Prediction Confidence Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Explainability (optional)")
            explain_choice = st.selectbox("Method", ["None", "SHAP", "LIME"], index=0)
            if explain_choice == "SHAP":
                if shap is None:
                    st.warning("SHAP not installed. Add it in requirements and reinstall.")
                else:
                    try:
                        # Use TreeExplainer when available
                        explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') else shap.Explainer(model)
                        shap_values = explainer(data_scaled)
                        st.write("Global feature importance (SHAP):")
                        shap.summary_plot(shap_values, data_features, plot_type="bar", show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except Exception as e:
                        st.info(f"SHAP explanation not available: {e}")
            elif explain_choice == "LIME":
                if LimeTabularExplainer is None:
                    st.warning("LIME not installed. Add it in requirements and reinstall.")
                else:
                    try:
                        explainer = LimeTabularExplainer(
                            training_data=data_features.values,
                            feature_names=list(data_features.columns),
                            class_names=list(label_encoder.classes_),
                            mode='classification'
                        )
                        # Explain first row as an example
                        exp = explainer.explain_instance(
                            data_features.values[0],
                            model.predict_proba
                        )
                        st.write("LIME explanation for first row:")
                        st.components.v1.html(exp.as_html(), height=400, scrolling=True)
                    except Exception as e:
                        st.info(f"LIME explanation not available: {e}")
            
        except Exception as e:
            st.error(f"Error processing data: {e}")

def show_performance_page(metrics):
    st.header("Model Performance")
    
    # Metrics overview
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cross-Validation Mean", f"{metrics['cv_mean']:.1%} ± {metrics['cv_std']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance explanation
    st.subheader("Model Evaluation")
    st.write("""
    - **Accuracy**: Overall percentage of correct predictions
    - **Cross-Validation**: Model performance across different data splits
    - **Random Forest**: Ensemble method combining multiple decision trees
    """)

def show_feature_analysis_page(metrics):
    st.header("Feature Importance Analysis")
    
    # Feature importance plot
    importance_df = pd.DataFrame(metrics['feature_importance'])
    
    fig = px.bar(
        importance_df.head(10), 
        x='importance', 
        y='feature',
        orientation='h',
        title="Top 10 Most Important Features"
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature descriptions
    st.subheader("Feature Descriptions")
    feature_descriptions = {
        'koi_period': 'Orbital period of the planet (days)',
        'koi_prad': 'Planet radius (Earth radii)',
        'koi_depth': 'Transit depth (parts per million)',
        'koi_duration': 'Transit duration (hours)',
        'koi_ingress': 'Ingress duration (hours)',
        'koi_dror': 'Planet-star radius ratio',
        'koi_count': 'Number of planets in the system',
        'koi_num_transits': 'Number of observed transits'
    }
    
    for feature, description in feature_descriptions.items():
        st.write(f"**{feature}**: {description}")

def show_about_page():
    st.header("About This Project")
    
    st.write("""
    ## 🪐 Exoplanet Detection with AI/ML
    
    This application uses machine learning to predict exoplanet candidates from NASA's Kepler Objects of Interest dataset.
    
    ### 🔬 Methodology
    1. **Data Preprocessing**: Clean and normalize the Kepler KOI dataset
    2. **Feature Selection**: Use key transit and orbital parameters
    3. **Model Training**: Random Forest classifier with cross-validation
    4. **Prediction**: Classify candidates as confirmed, candidate, or false positive
    
    ### 📊 Key Features
    - **Interactive Prediction**: Upload your own candidate data
    - **Model Performance**: View accuracy metrics and validation results
    - **Feature Analysis**: Understand which parameters are most important
    
    ### 🎯 Target Classifications
    - **CONFIRMED**: Verified exoplanet
    - **CANDIDATE**: Potential exoplanet requiring further study
    - **FALSE POSITIVE**: Not an exoplanet (stellar activity, etc.)
    
    ### 🚀 Built For
    Researchers, students, and space enthusiasts interested in exoplanet discovery and data science.
    """)

if __name__ == "__main__":
    main()
