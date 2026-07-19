import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

import config

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def feature_engineering(df):
    print("Engineering domain-specific features...")
    # 1. Habitability Index (Goldilocks Zone proxy)
    # teq between 273K and 373K is roughly 0 to 100 Celsius
    if 'koi_teq' in df.columns:
        df['goldilocks_flag'] = ((df['koi_teq'] > 273) & (df['koi_teq'] < 373)).astype(int)
    else:
        df['goldilocks_flag'] = 0
        
    # 2. Planet Size Classification Proxy
    if 'koi_prad' in df.columns:
        # 1: Earth-size (<1.25), 2: Super-Earth (1.25-2.0), 3: Mini-Neptune (2.0-4.0), 4: Giant (>4.0)
        conditions = [
            (df['koi_prad'] < 1.25),
            (df['koi_prad'] >= 1.25) & (df['koi_prad'] < 2.0),
            (df['koi_prad'] >= 2.0) & (df['koi_prad'] < 4.0),
            (df['koi_prad'] >= 4.0)
        ]
        choices = [1, 2, 3, 4]
        df['planet_size_class'] = np.select(conditions, choices, default=0)
    else:
        df['planet_size_class'] = 0

    return df

def clean_data(df):
    print("Cleaning data...")
    # Drop rows where target is missing
    df = df.dropna(subset=[config.TARGET_COL])
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Add new engineered features to the list of features to keep
    extended_features = config.FEATURES + ['goldilocks_flag', 'planet_size_class']
    
    # Filter only required features + target
    available_features = [col for col in extended_features if col in df.columns]
    df_filtered = df[available_features + [config.TARGET_COL]].copy()
    
    # Convert disposition to numerical labels
    le = LabelEncoder()
    df_filtered['target'] = le.fit_transform(df_filtered[config.TARGET_COL])
    
    # Save label encoder for later interpretation
    joblib.dump(le, config.MODEL_DIR / 'label_encoder.joblib')
    
    X = df_filtered.drop([config.TARGET_COL, 'target'], axis=1)
    y = df_filtered['target']
    
    return X, y, le.classes_

def preprocess_data(X_train, X_test):
    print("Imputing missing values and scaling features...")
    # Handle missing values - using median to be robust to outliers
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Save preprocessors
    joblib.dump(imputer, config.MODEL_DIR / 'imputer.joblib')
    joblib.dump(scaler, config.MODEL_DIR / 'scaler.joblib')
    
    return X_train_scaled, X_test_scaled

def main():
    df = load_data(config.DATA_PATH)
    X, y, classes = clean_data(df)
    
    print(f"Classes found: {classes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    # Save processed data for training and evaluation
    np.save(config.MODEL_DIR / 'X_train.npy', X_train_scaled)
    np.save(config.MODEL_DIR / 'X_test.npy', X_test_scaled)
    np.save(config.MODEL_DIR / 'y_train.npy', y_train)
    np.save(config.MODEL_DIR / 'y_test.npy', y_test)
    
    # Save feature names for SHAP plots later
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, config.MODEL_DIR / 'feature_names.joblib')
    
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
