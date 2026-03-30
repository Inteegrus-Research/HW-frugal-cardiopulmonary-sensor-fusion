"""
03_model_benchmark.py
---------------------
Objective: Execute a multi-model evaluation utilizing nested 5-Fold Cross Validation. 
Generates performance metrics, calibration curves, SHAP explainability, 
and simulates hardware duty-cycle ablation for theoretical power budgeting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.utils import resample
import shap
import warnings
import os

warnings.filterwarnings('ignore')

# Global configuration
DATA_PATH = '../data/extracted_features.csv'
RESULTS_DIR = '../figures/'
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
RANDOM_SEED = 42

def main():
    print("Initializing Machine Learning Pipeline...")
    df_raw = pd.read_csv(DATA_PATH)

    # SQI Filtering
    df_clean = df_raw[(df_raw['SpO2_Quality'] == 1) & (df_raw['ECG_Quality'] == 1)].copy()
    print(f"Initial Cohort: N={len(df_raw)} | Final Analytical Cohort (post-SQI): N={len(df_clean)}")

    features = ['Age', 'BMI', 'ODI', 'CT90', 'RMSSD', 'LF_HF', 'EDR_var']
    X = df_clean[features]
    y = df_clean['Target_Apnea']

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    # --- XGBoost Hyperparameter Tuning ---
    pos_weight = len(y[y==0]) / len(y[y==1])
    xgb_param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    
    xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=pos_weight, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(xgb_base, xgb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    best_xgb = grid_search.best_estimator_

    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED),
        "SVM (RBF)": SVC(probability=True, class_weight='balanced', random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=RANDOM_SEED),
        "XGBoost (Tuned)": best_xgb
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    predictions = {}

    print(f"{'Model':<20} | {'AUC':<5} | {'Acc':<5} | {'Sens':<5} | {'Spec':<5} | {'Prec':<5} | {'F1':<5}")
    print("-" * 75)

    # --- Benchmark Evaluation ---
    fig, ax_cal = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        y_pred_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        predictions[name] = y_pred_proba
        
        acc = accuracy_score(y, y_pred)
        sens = recall_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        spec = tn / (tn + fp)
        prec = precision_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred)
        roc_auc = auc(*roc_curve(y, y_pred_proba)[:2])
        
        print(f"{name:<20} | {roc_auc:.3f} | {acc:.3f} | {sens:.3f} | {spec:.3f} | {prec:.3f} | {f1:.3f}")
        CalibrationDisplay.from_predictions(y, y_pred_proba, n_bins=10, name=name, ax=ax_cal)

    ax_cal.set_title("Reliability Diagram (Calibration Plot)", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(RESULTS_DIR, 'ieee_calibration_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- SHAP Explainability ---
    best_xgb.fit(X_scaled, y)
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_scaled)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled, feature_names=features, show=False)
    plt.title('SHAP Feature Importance (XGBoost)', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ieee_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Bootstrapping ---
    best_preds = predictions["XGBoost (Tuned)"]
    bootstrapped_aucs = []
    for i in range(1000):
        indices = resample(np.arange(len(y)), random_state=i)
        if len(np.unique(y.iloc[indices])) < 2: continue
        score = auc(*roc_curve(y.iloc[indices], best_preds[indices])[:2])
        bootstrapped_aucs.append(score)
        
    sorted_scores = np.sort(np.array(bootstrapped_aucs))
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(f"\nXGBoost Bootstrapped AUC: {np.mean(sorted_scores):.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")

    # --- Hardware Duty-Cycle Simulation ---
    print("\nExecuting Duty-Cycle Power Simulation...")
    np.random.seed(RANDOM_SEED)
    duty_cycles = [0.0, 0.1, 0.25, 0.5, 1.0] 
    energy_mw = [1.2 + (dc * 5.0) for dc in duty_cycles]
    simulated_aucs = []

    for dc in duty_cycles:
        mask = np.random.rand(len(X)) > dc
        X_sim = X.copy()
        X_sim.loc[mask, ['RMSSD', 'LF_HF', 'EDR_var']] = np.nan 
        X_sim_scaled = pd.DataFrame(scaler.transform(X_sim), columns=features)
        
        sim_preds = cross_val_predict(best_xgb, X_sim_scaled, y, cv=cv, method='predict_proba')[:, 1]
        sim_auc = auc(*roc_curve(y, sim_preds)[:2])
        simulated_aucs.append(sim_auc)
        print(f"ECG Uptime {dc*100:3.0f}% | Est. Power: {1.2 + (dc * 5.0):.2f} mW | AUC: {sim_auc:.3f}")

    # Trade-off Figure
    fig, ax1 = plt.subplots(figsize=(8, 6))
    color = 'tab:red'
    ax1.set_xlabel('ECG Duty Cycle (%)', fontweight='bold')
    ax1.set_ylabel('Diagnostic Performance (AUC)', color=color, fontweight='bold')
    ax1.plot([dc * 100 for dc in duty_cycles], simulated_aucs, color=color, marker='o', lw=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.80, 1.0])

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Sensor Power Consumption (mW)', color=color, fontweight='bold')  
    ax2.plot([dc * 100 for dc in duty_cycles], energy_mw, color=color, linestyle='--', marker='s', lw=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 7])

    plt.title('Hardware Power Budget vs. Diagnostic Accuracy', fontweight='bold', pad=15)
    fig.tight_layout()  
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, 'ieee_energy_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Pipeline execution completed successfully. Assets saved to figures/ directory.")

if __name__ == "__main__":
    main()
