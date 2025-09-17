import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_exoplanet_model(input_csv: str, artifacts_dir: str, dataset: str, model_type: str = "rf"):
    """Train and evaluate exoplanet classification model (dataset-aware)."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load preprocessed data
    print("Loading preprocessed data...")
    data = pd.read_csv(input_csv)

    # Prepare features and target
    # Determine target column (created by preprocessing)
    inferred_target = 'target_encoded' if 'target_encoded' in data.columns else 'koi_disposition_encoded'
    # Exclude all non-numeric columns and target columns from features
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    target_cols = [col for col in data.columns if 'disposition' in col.lower() or col == inferred_target]
    feature_cols = [col for col in numeric_cols if col not in target_cols]
    X = data[feature_cols]
    y = data[inferred_target]

    print(f"Training with {len(feature_cols)} features: {feature_cols}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Choose model
    if model_type == "rf":
        print("\nTraining Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == "logreg":
        print("\nTraining Logistic Regression model...")
        model = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None)
    else:
        raise ValueError("Unsupported model type. Use 'rf' or 'logreg'.")

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance (for models that support it)
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.nan
        })

    # Save artifacts
    model_path = os.path.join(artifacts_dir, f"{dataset}_{model_type}_model.joblib")
    metrics_path = os.path.join(artifacts_dir, f"{dataset}_metrics.joblib")
    fi_csv_path = os.path.join(artifacts_dir, f"{dataset}_feature_importance.csv")
    joblib.dump(model, model_path)
    feature_importance.to_csv(fi_csv_path, index=False)

    metrics = {
        'dataset': dataset,
        'model_type': model_type,
        'accuracy': float(accuracy),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'classification_report': report,
        'feature_importance': feature_importance.to_dict('records'),
        'features': feature_cols,
    }
    joblib.dump(metrics, metrics_path)

    print("\nModel training completed successfully!")
    print("Files created:")
    print(f"- {model_path}")
    print(f"- {fi_csv_path}")
    print(f"- {metrics_path}")

    return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Train exoplanet classifier (dataset-aware)")
    parser.add_argument("--dataset", choices=["kepler", "tess", "k2"], required=True)
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for saved models/metrics")
    parser.add_argument("--model-type", choices=["rf", "logreg"], default="rf")
    args = parser.parse_args()

    train_exoplanet_model(
        input_csv=args.input,
        artifacts_dir=args.artifacts_dir,
        dataset=args.dataset,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
