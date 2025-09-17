import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from io import StringIO

def load_csv_skipping_comments(filename: str) -> pd.DataFrame:
    """Load CSV and skip header comment lines starting with '#'"""
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    return pd.read_csv(StringIO("".join(lines)))

DATASET_REGISTRY = {
    # Kepler KOI
    "kepler": {
        "target": ["koi_disposition"],
        "feature_aliases": {
            "koi_period": ["koi_period", "period", "orbital_period"],
            "koi_prad": ["koi_prad", "planet_radius", "prad"],
            "koi_depth": ["koi_depth", "transit_depth", "depth"],
            "koi_duration": ["koi_duration", "transit_duration", "duration"],
            "koi_ingress": ["koi_ingress", "ingress_time", "ingress"],
            "koi_dror": ["koi_dror", "radius_ratio", "rp_rs"],
            "koi_count": ["koi_count", "planet_count", "n_planets"],
            "koi_num_transits": ["koi_num_transits", "num_transits", "transit_count"],
        },
    },
    # TESS TOI (column names vary; provide common aliases)
    "tess": {
        "target": ["disposition", "toi_disposition", "tfopwg_disp"],
        "feature_aliases": {
            "orbital_period": ["orbital_period", "pl_orbper", "period", "toi_period"],
            "planet_radius": ["planet_radius", "pl_rade", "radius", "toi_prad"],
            "transit_depth": ["transit_depth", "depth", "toi_depth"],
            "transit_duration": ["transit_duration", "duration", "toi_duration"],
            "ingress_time": ["ingress_time", "ingress", "toi_ingress"],
            "radius_ratio": ["radius_ratio", "rp_rs", "toi_rp_rs"],
            "planet_count": ["planet_count", "n_planets", "toi_count"],
            "transit_count": ["transit_count", "num_transits", "toi_num_transits"],
        },
    },
    # K2
    "k2": {
        "target": ["disposition", "k2_disposition"],
        "feature_aliases": {
            "orbital_period": ["orbital_period", "pl_orbper", "period"],
            "planet_radius": ["planet_radius", "pl_rade", "radius"],
            "transit_depth": ["transit_depth", "depth"],
            "transit_duration": ["transit_duration", "duration"],
            "ingress_time": ["ingress_time", "ingress"],
            "radius_ratio": ["radius_ratio", "rp_rs"],
            "planet_count": ["planet_count", "n_planets"],
            "transit_count": ["transit_count", "num_transits"],
        },
    },
}


def resolve_columns_by_alias(df: pd.DataFrame, aliases: dict) -> dict:
    """Return a mapping from canonical feature name to existing column in df using alias lists."""
    resolved = {}
    for canonical, options in aliases.items():
        for candidate in options:
            if candidate in df.columns:
                resolved[canonical] = candidate
                break
    return resolved


def preprocess_dataset(filename: str, dataset_key: str):
    """Preprocess any supported dataset using the registry and return scaled data, encoder, scaler."""
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError("Unsupported dataset")

    print(f"Loading '{dataset_key}' dataset...")
    data = load_csv_skipping_comments(filename)
    print(f"Initial data shape: {data.shape}")
    
    # Resolve features and target using aliases
    registry = DATASET_REGISTRY[dataset_key]
    target_candidates = registry["target"]
    feature_aliases = registry["feature_aliases"]
    resolved = resolve_columns_by_alias(data, feature_aliases)

    # Determine target column in data
    target_col = next((t for t in target_candidates if t in data.columns), None)
    
    # Build list of selected columns that exist (features only)
    feature_cols = list(resolved.values())
    # Add target column if found
    if target_col is not None:
        selected_cols = feature_cols + [target_col]
    else:
        selected_cols = feature_cols
    data = data[selected_cols]
    print(f"Selected columns: {selected_cols}")
    
    # Remove rows with missing target variable
    if target_col is not None:
        data = data.dropna(subset=[target_col])
    
    # Handle missing values in features
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if target_col is None or col != target_col:
            data[col] = data[col].fillna(data[col].median())
    
    # Remove outliers using IQR method
    for col in numeric_columns:
        if target_col is None or col != target_col:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
    # Encode target variable
    le = LabelEncoder()
    if target_col is not None:
        original_labels = data[target_col].astype(str).unique()
        data['target_encoded'] = le.fit_transform(data[target_col].astype(str))
        print(f"Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Save label encoder for later use
        # Saved by caller with dataset-aware path
    
    # Feature scaling
    # Exclude target columns from features
    target_cols = [target_col, 'target_encoded'] if target_col else ['target_encoded']
    feature_cols = [col for col in data.columns if col not in target_cols]
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    # Saved by caller with dataset-aware path
    
    print(f"Final preprocessed data shape: {data_scaled.shape}")
    
    return data_scaled, le, scaler

def main():
    parser = argparse.ArgumentParser(description="Dataset-aware preprocessing for exoplanet datasets")
    parser.add_argument("--dataset", choices=["kepler", "tess", "k2"], required=True, help="Dataset key")
    parser.add_argument("--input", required=True, help="Path to raw input CSV")
    parser.add_argument("--output", required=True, help="Path to save processed CSV")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory to save encoders/scalers")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    processed_data, label_encoder, feature_scaler = preprocess_dataset(args.input, args.dataset)

    # Save artifacts with dataset prefix
    label_path = os.path.join(args.artifacts_dir, f"{args.dataset}_label_encoder.joblib")
    scaler_path = os.path.join(args.artifacts_dir, f"{args.dataset}_scaler.joblib")
    joblib.dump(label_encoder, label_path)
    joblib.dump(feature_scaler, scaler_path)

    # Save processed CSV
    processed_data.to_csv(args.output, index=False)

    print("Data preprocessing completed successfully!")
    print("Files created:")
    print(f"- {args.output}")
    print(f"- {label_path}")
    print(f"- {scaler_path}")


if __name__ == "__main__":
    main()
