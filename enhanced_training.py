"""
Enhanced Training Script for NASA Space Apps Exoplanet Challenge
Generates advanced metrics including confusion matrices, ROC curves, and detailed classification reports
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, precision_score, recall_score
)
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(dataset_key):
    """Load and preprocess dataset based on key"""
    data_path = f"data/{dataset_key}_candidates.csv" if dataset_key != "kepler" else "data/kepler_koi.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Dataset-specific preprocessing
    if dataset_key == "kepler":
        # Kepler preprocessing
        target_col = 'koi_disposition'
        feature_cols = [
            'koi_period', 'koi_prad', 'koi_depth', 'koi_duration',
            'koi_ingress', 'koi_dror', 'koi_count', 'koi_num_transits'
        ]
    elif dataset_key == "tess":
        # TESS preprocessing
        target_col = 'tfopwg_disp'
        feature_cols = [
            'pl_orbper', 'pl_rade', 'tran_depth', 'tran_dur14',
            'tran_ingress', 'pl_ratror', 'sy_pnum', 'tran_flag'
        ]
    elif dataset_key == "k2":
        # K2 preprocessing
        target_col = 'k2c_disp'
        feature_cols = [
            'pl_orbper', 'pl_rade', 'tran_depth', 'tran_dur14',
            'tran_ingress', 'pl_ratror', 'sy_pnum', 'tran_flag'
        ]
    
    # Filter for available columns
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 3:
        print(f"Warning: Only {len(available_features)} features available for {dataset_key}")
    
    # Clean data
    df_clean = df[available_features + [target_col]].dropna()
    
    # Prepare features and target
    X = df_clean[available_features]
    y = df_clean[target_col]
    
    return X, y, available_features

def calculate_multiclass_roc(y_true, y_pred_proba, classes):
    """Calculate ROC curves for multiclass classification"""
    roc_data = {}
    
    # Binarize the output
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=classes)
    
    # Handle binary classification case
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_data[classes[1]] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': roc_auc
        }
    else:
        # Multiclass case
        for i, class_name in enumerate(classes):
            if i < y_pred_proba.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                roc_data[class_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc
                }
    
    return roc_data

def train_enhanced_model(dataset_key, model_type='rf'):
    """Train model with enhanced metrics collection"""
    print(f"Training {model_type.upper()} model for {dataset_key.upper()} dataset...")
    
    # Load data
    X, y, feature_names = load_and_preprocess_data(dataset_key)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'logreg':
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='ovr'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Precision and Recall (macro average)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification Report
    class_report = classification_report(y_test, y_pred, 
                                       target_names=label_encoder.classes_,
                                       output_dict=True, zero_division=0)
    
    # ROC Curves
    try:
        roc_data = calculate_multiclass_roc(y_test, y_pred_proba, label_encoder.classes_)
    except Exception as e:
        print(f"ROC calculation failed: {e}")
        roc_data = {}
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = [
            {'feature': name, 'importance': importance}
            for name, importance in zip(feature_names, model.feature_importances_)
        ]
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    else:
        # For logistic regression, use coefficient magnitudes
        if len(label_encoder.classes_) == 2:
            coef_importance = np.abs(model.coef_[0])
        else:
            coef_importance = np.abs(model.coef_).mean(axis=0)
        
        feature_importance = [
            {'feature': name, 'importance': importance}
            for name, importance in zip(feature_names, coef_importance)
        ]
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    # Compile metrics
    metrics = {
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist(),
        'class_labels': label_encoder.classes_.tolist(),
        'classification_report': class_report,
        'roc_data': roc_data,
        'feature_importance': feature_importance,
        'model_type': model_type,
        'dataset': dataset_key,
        'n_samples': len(X),
        'n_features': len(feature_names)
    }
    
    # Save artifacts
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save model and preprocessors
    joblib.dump(model, os.path.join(artifacts_dir, f"{dataset_key}_{model_type}_model.joblib"))
    joblib.dump(label_encoder, os.path.join(artifacts_dir, f"{dataset_key}_label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(artifacts_dir, f"{dataset_key}_scaler.joblib"))
    joblib.dump(metrics, os.path.join(artifacts_dir, f"{dataset_key}_metrics.joblib"))
    
    # Save feature importance as CSV
    importance_df = pd.DataFrame(feature_importance)
    importance_df.to_csv(os.path.join(artifacts_dir, f"{dataset_key}_feature_importance.csv"), index=False)
    
    print(f"✅ Training complete for {dataset_key} {model_type}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   CV Score: {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    
    return metrics

def main():
    """Train all models with enhanced metrics"""
    datasets = ['kepler', 'tess', 'k2']
    models = ['rf', 'logreg']
    
    print("🚀 Starting Enhanced Training for NASA Space Apps Exoplanet Challenge")
    print("=" * 60)
    
    for dataset in datasets:
        print(f"\n📊 Processing {dataset.upper()} dataset...")
        
        for model_type in models:
            try:
                metrics = train_enhanced_model(dataset, model_type)
                print(f"   ✅ {model_type.upper()} model trained successfully")
            except Exception as e:
                print(f"   ❌ Failed to train {model_type.upper()} model: {e}")
    
    print("\n🎉 Enhanced training complete!")
    print("All models now include:")
    print("  • Confusion matrices")
    print("  • ROC curves")
    print("  • Precision/Recall metrics")
    print("  • Detailed classification reports")
    print("  • Enhanced feature importance")

if __name__ == "__main__":
    main()
