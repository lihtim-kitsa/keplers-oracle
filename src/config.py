import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "KOI_Cumulative_clean.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset configuration
TARGET_COL = "koi_disposition"

# Features to use for modeling
# We exclude IDs, designations, string flags, and the pre-computed 'koi_score' if it exists.
# We also focus mainly on the primary estimate values rather than just the error bounds to avoid too much noise,
# though incorporating errors can be a more advanced step later.
FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq", 
    "koi_insol", "koi_impact", "koi_eccen", "koi_srho", "koi_sma", "koi_incl",
    "koi_steff", "koi_slogg", "koi_srad", "koi_smass", "koi_kepmag"
]

# Random seed for reproducibility
RANDOM_STATE = 42

# XGBoost hyperparameters for RandomizedSearchCV
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}
