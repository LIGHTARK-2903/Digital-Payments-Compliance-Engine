# src/shap_explain.py
"""
SHAP explainability script for ML Upgrade model.
Outputs:
 - data/shap_explanations.csv  (per-tx mean abs SHAP per feature)
 - plots/shap_summary.png
Requires: shap, joblib, pandas, numpy, matplotlib
Run:
  python .\src\shap_explain.py
"""
import os, joblib, json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

ROOT = os.getcwd()
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
PLOTS = os.path.join(ROOT, "plots")
OUT_SHAP_CSV = os.path.join(DATA, "shap_explanations.csv")
OUT_SHAP_PLOT = os.path.join(PLOTS, "shap_summary.png")

# helper to load artifact and feature matrix (reuse feature engineering)
from src.ml_upgrade import prepare_features, load_audit_csv, load_str_jsonl

def main():
    print("Loading model artifact...")
    mp = joblib.load(os.path.join(MODELS, "ml_upgrade_model.joblib"))
    model = mp['model']
    le = mp.get('label_encoder', None)
    print("Model loaded. Loading data and features...")
    df = load_audit_csv()
    # features as used in ml_upgrade
    X_all = prepare_features(df)
    X_features = X_all.drop(columns=['tx_id']).reset_index(drop=True)
    # load ensemble preds (if present) to pick top flagged transactions
    ensemble_path = os.path.join(DATA, "ensemble_preds.csv")
    if os.path.exists(ensemble_path):
        ens = pd.read_csv(ensemble_path)
        top_tx = ens.sort_values('ensemble_prob', ascending=False).head(200)['tx_id'].tolist()
    else:
        # fallback: use ml_upgrade_preds ordering by ml_prob_STR
        mlp = pd.read_csv(os.path.join(DATA, "ml_upgrade_preds.csv"))
        top_tx = mlp.sort_values('ml_prob_STR', ascending=False).head(200)['tx_id'].tolist()
    # select rows for top_tx and preserve index mapping
    mask = X_all['tx_id'].isin(top_tx)
    X_top = X_features[mask].reset_index(drop=True)
    if X_top.shape[0] == 0:
        print("No top transactions found; aborting.")
        return

    print(f"Running SHAP on {len(X_top)} transactions (may take time)...")
    # Get underlying model object for shap
    try:
        # if pipeline, obtain the final estimator
        if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
            clf = model.named_steps['clf']
            # and we must pass transformed data to explainer if scaler present
            # generate explainer on pipeline: use shap.Explainer with the full pipeline
            explainer = shap.Explainer(model, X_top, feature_names=X_top.columns.tolist())
            shap_values = explainer(X_top)
        else:
            clf = model
            explainer = shap.Explainer(clf, X_top)
            shap_values = explainer(X_top)
    except Exception as e:
        print("SHAP explainer creation failed:", e)
        return

    # Summarize mean absolute SHAP per transaction
    try:
        # shap_values.values shape: (n_samples, n_features) or list for multi-output
        vals = shap_values.values
        mean_abs = np.mean(np.abs(vals), axis=1)
        # create per-tx per-feature mean abs (for top transactions)
        # shap_values.data: original data matrix; shap_values.values same shape
        shap_arr = np.abs(vals)
        df_shap = pd.DataFrame(shap_arr, columns=X_top.columns, index=X_top.index)
        df_shap['tx_id'] = X_all.loc[mask, 'tx_id'].values
        # for each tx compute top feature contribution
        df_shap['top_feature'] = df_shap.drop(columns=['tx_id']).idxmax(axis=1)
        df_shap['mean_abs_shap'] = df_shap.drop(columns=['tx_id']).abs().mean(axis=1)
        df_shap.to_csv(OUT_SHAP_CSV, index=False)
        print("Wrote SHAP explanations to", OUT_SHAP_CSV)
        # summary plot
        shap.summary_plot(shap_values, X_top, show=False)
        plt.savefig(OUT_SHAP_PLOT, bbox_inches='tight')
        plt.close()
        print("Wrote SHAP summary plot to", OUT_SHAP_PLOT)
    except Exception as e:
        print("Failed to convert SHAP output:", e)

if __name__ == "__main__":
    main()