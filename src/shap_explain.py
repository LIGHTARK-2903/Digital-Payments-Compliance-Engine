# src/shap_explain.py
"""
SHAP explainability for ML Upgrade model (clean standalone version).
- Extracts scaler + classifier (ignores SMOTEENN)
- Applies scaler to features
- Handles multi-class XGBoost SHAP (reduces to STR class or summed abs)
- Writes:
    data/shap_explanations.csv
    plots/shap_summary.png
Run:
    python -m src.shap_explain
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.ml_upgrade import prepare_features, load_audit_csv

ROOT = os.getcwd()
DATA = os.path.join(ROOT, "data")
PLOTS = os.path.join(ROOT, "plots")
MODELS = os.path.join(ROOT, "models")

OUT_SHAP_CSV = os.path.join(DATA, "shap_explanations.csv")
OUT_SHAP_PLOT = os.path.join(PLOTS, "shap_summary.png")

MODEL_FILE = os.path.join(MODELS, "ml_upgrade_model.joblib")


# ---------------------------------------------------------
#  Extract scaler + classifier from the saved artifact
# ---------------------------------------------------------
def extract_scaler_and_clf(artifact):
    model = artifact["model"]

    scaler = None
    clf = None

    if hasattr(model, "named_steps"):
        steps = model.named_steps
        scaler = steps.get("scaler", None)
        clf = steps.get("clf", list(steps.values())[-1])
    else:
        clf = model

    return scaler, clf


# ---------------------------------------------------------
#  Select STR class index for SHAP (if available)
# ---------------------------------------------------------
def get_str_class_index(artifact):
    if "labels" not in artifact:
        return None
    labels = artifact["labels"]
    if "STR_review" in labels:
        return labels.index("STR_review")
    return None


# ---------------------------------------------------------
#  Main
# ---------------------------------------------------------
def main():
    print("Loading model artifact...")
    if not os.path.exists(MODEL_FILE):
        print("Model artifact missing:", MODEL_FILE)
        return

    artifact = joblib.load(MODEL_FILE)

    scaler, clf = extract_scaler_and_clf(artifact)
    print("Scaler:", type(scaler))
    print("Classifier:", type(clf))

    print("Loading data + preparing features...")
    df = load_audit_csv()
    X_all = prepare_features(df)
    X = X_all.drop(columns=["tx_id"]).reset_index(drop=True)

    # Get top 200 risky transactions
    ens_file = os.path.join(DATA, "ensemble_preds.csv")
    if os.path.exists(ens_file):
        ens = pd.read_csv(ens_file)
        top_ids = ens.sort_values("ensemble_prob", ascending=False).head(200)["tx_id"].tolist()
    else:
        mlp = pd.read_csv(os.path.join(DATA, "ml_upgrade_preds.csv"))
        top_ids = mlp.sort_values("ml_prob_STR", ascending=False).head(200)["tx_id"].tolist()

    mask = X_all["tx_id"].isin(top_ids)
    X_top = X[mask].reset_index(drop=True)

    print(f"Selected {len(X_top)} transactions for SHAP.")

    # scale features if scaler exists
    if scaler is not None:
        X_scaled = scaler.transform(X_top)
        X_scaled = pd.DataFrame(X_scaled, columns=X_top.columns)
    else:
        X_scaled = X_top.copy()

    print("Running SHAP… (this may take time)")

    # Build SHAP explainer
    try:
        try:
            explainer = shap.TreeExplainer(clf)
        except Exception:
            explainer = shap.Explainer(clf, X_scaled)

        shap_values = explainer(X_scaled)
    except Exception as e:
        print("SHAP error:", e)
        return

    # Convert SHAP → 2D
    vals = shap_values.values

    # Multi-class case → reduce to STR class
    str_idx = get_str_class_index(artifact)
    if vals.ndim == 3:
        if str_idx is not None:
            print("Reducing multi-class SHAP to STR class...")
            vals2d = vals[:, :, str_idx]
        else:
            print("Summing |SHAP| across classes...")
            vals2d = np.sum(np.abs(vals), axis=2)
    else:
        vals2d = vals

    # Build CSV
    shap_abs = np.abs(vals2d)                # numpy array (n_samples, n_features)
    # create DataFrame of absolute SHAP values (features only)
    df_shap_feats = pd.DataFrame(shap_abs, columns=X_scaled.columns, index=X_scaled.index)
    # compute numeric mean_abs_shap directly from numpy (robust)
    mean_abs = shap_abs.mean(axis=1)         # shape (n_samples,)
    # assemble final DF with tx_id and top feature
    df_shap = df_shap_feats.copy()
    df_shap['tx_id'] = X_all.loc[mask, 'tx_id'].values
    # top feature by mean absolute SHAP per row (feature names come from df_shap_feats)
    df_shap['top_feature'] = df_shap_feats.idxmax(axis=1)
    df_shap['mean_abs_shap'] = mean_abs
    df_shap.to_csv(OUT_SHAP_CSV, index=False)
    print("Wrote SHAP explanations to", OUT_SHAP_CSV)


    # Summary plot
    try:
        shap.summary_plot(shap_values, X_scaled, show=False)
        plt.tight_layout()
        plt.savefig(OUT_SHAP_PLOT, dpi=300)
        plt.close()
        print("Wrote plot:", OUT_SHAP_PLOT)
    except Exception as e:
        print("SHAP plot failed:", e)


if __name__ == "__main__":
    main()