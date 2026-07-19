import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import shap

import config

def load_data_and_models():
    X_test = np.load(config.MODEL_DIR / 'X_test.npy')
    y_test = np.load(config.MODEL_DIR / 'y_test.npy')
    # Load main ensemble model
    model = joblib.load(config.MODEL_DIR / 'best_xgb_model.joblib')
    # Load individual models for comparison/explainability
    xgb_model = joblib.load(config.MODEL_DIR / 'xgb_model.joblib')
    
    le = joblib.load(config.MODEL_DIR / 'label_encoder.joblib')
    feature_names = joblib.load(config.MODEL_DIR / 'feature_names.joblib')
    return X_test, y_test, model, xgb_model, le, feature_names

def evaluate_model(model, X_test, y_test, target_names):
    print("Evaluating Ensemble Model performance...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Ensemble Model)')
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'confusion_matrix.png')
    print(f"Saved confusion matrix to {config.RESULTS_DIR / 'confusion_matrix.png'}")
    
    # ROC Curve Plot (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green']
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC curve of class {target_names[i]} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Multi-class)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'roc_curve.png')
    print(f"Saved ROC curve to {config.RESULTS_DIR / 'roc_curve.png'}")

def explain_model(xgb_model, X_test, feature_names):
    print("Generating SHAP feature importance plot (using XGBoost base model)...")
    # Use TreeExplainer for the XGBoost component of the ensemble
    explainer = shap.TreeExplainer(xgb_model)
    sample_size = min(1000, X_test.shape[0])
    np.random.seed(config.RANDOM_STATE)
    indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
    X_test_sampled = X_test[indices]
    
    shap_values = explainer.shap_values(X_test_sampled)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sampled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'shap_summary.png')
    print(f"Saved SHAP summary to {config.RESULTS_DIR / 'shap_summary.png'}")

def main():
    X_test, y_test, model, xgb_model, le, feature_names = load_data_and_models()
    evaluate_model(model, X_test, y_test, le.classes_)
    explain_model(xgb_model, X_test, feature_names)
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
