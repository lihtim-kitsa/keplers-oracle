import numpy as np
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import time

import config

def load_processed_data():
    X_train = np.load(config.MODEL_DIR / 'X_train.npy')
    y_train = np.load(config.MODEL_DIR / 'y_train.npy')
    return X_train, y_train

def handle_imbalance(X_train, y_train):
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled

def train_models(X_train, y_train):
    num_classes = len(np.unique(y_train))
    
    # 1. XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        objective='multi:softprob' if num_classes > 2 else 'binary:logistic',
        num_class=num_classes if num_classes > 2 else None,
        eval_metric='mlogloss' if num_classes > 2 else 'logloss',
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, config.MODEL_DIR / 'xgb_model.joblib')
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, config.MODEL_DIR / 'rf_model.joblib')
    
    # 3. LightGBM
    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        objective='multiclass' if num_classes > 2 else 'binary',
        random_state=config.RANDOM_STATE,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    joblib.dump(lgbm, config.MODEL_DIR / 'lgbm_model.joblib')
    
    # 4. Voting Classifier (Soft Voting for probabilities)
    print("Training Voting Classifier ensemble...")
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('lgbm', lgbm)
        ],
        voting='soft',
        n_jobs=-1
    )
    voting_clf.fit(X_train, y_train)
    joblib.dump(voting_clf, config.MODEL_DIR / 'best_xgb_model.joblib') # Overwriting the old name so app.py still works without changes
    print("Voting Classifier saved as main model.")
    
    return voting_clf

def main():
    X_train, y_train = load_processed_data()
    X_train_resampled, y_train_resampled = handle_imbalance(X_train, y_train)
    train_models(X_train_resampled, y_train_resampled)
    print("Training complete!")

if __name__ == "__main__":
    main()
