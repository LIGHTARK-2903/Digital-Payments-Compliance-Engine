# src/ml_upgrade.py
"""
ML Upgrade (all-in-one) for Digital-Payments-Compliance-Engine

Features:
 - temporal splitting (train on older, validate on recent)
 - robust feature engineering (handles missing cols)
 - SMOTEENN oversampling + pipeline
 - RandomizedSearchCV hyperparam tuning for RandomForest and XGBoost (if available)
 - Label encoding for XGBoost compatibility
 - Threshold tuning for STR class (maximize F1 for STR)
 - IsolationForest anomaly scoring and ensemble combination
 - Outputs:
     - models/ml_upgrade_model.joblib  (dict: {'model', 'labels', 'label_encoder', 'threshold'})
     - data/ml_upgrade_preds.csv
     - data/ensemble_preds.csv
     - reports/ml_upgrade_report.txt
     - plots/ml_upgrade_pr_curve.png, ml_upgrade_roc_curve.png
Usage:
    (venv) python .\src\ml_upgrade.py
"""
import os
import json
import warnings
from collections import Counter, defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, roc_auc_score
from sklearn.utils.multiclass import unique_labels

# imbalanced-learn
try:
    from imblearn.combine import SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    SMOTEENN = None
    ImbPipeline = None

# xgboost optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False

# joblib
import joblib

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Silence benign warnings
warnings.filterwarnings("ignore")

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
PLOTS_DIR = os.path.join(ROOT, "plots")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CSV_IN = os.path.join(DATA_DIR, "audit_summary.csv")
STR_JSONL = os.path.join(DATA_DIR, "str_audit.jsonl")
OUT_PRED_CSV = os.path.join(DATA_DIR, "ml_upgrade_preds.csv")
OUT_ENSEMBLE_CSV = os.path.join(DATA_DIR, "ensemble_preds.csv")
MODEL_OUT = os.path.join(MODEL_DIR, "ml_upgrade_model.joblib")
REPORT_OUT = os.path.join(REPORT_DIR, "ml_upgrade_report.txt")
PR_PLOT = os.path.join(PLOTS_DIR, "ml_upgrade_pr_curve.png")
ROC_PLOT = os.path.join(PLOTS_DIR, "ml_upgrade_roc_curve.png")

RANDOM_STATE = 42

# ----------------- helpers -----------------
def load_audit_csv(path=CSV_IN):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the simulation to generate audit_summary.csv first.")
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df

def load_str_jsonl(path=STR_JSONL):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def is_round_number(amount, multiple=1000):
    try:
        return (abs(amount - round(amount)) < 1e-6) and (float(amount) % multiple == 0)
    except Exception:
        return False

def prepare_features(df):
    """
    Robust feature engineering:
     - amount, log_amount, hour, is_kyc
     - payment system dummies
     - is_round_1000
     - days_dormant (handles missing)
     - velocity_10m and reversal_count (naive sliding windows)
     - det_score if present
    Returns DataFrame with tx_id included.
    """
    out = pd.DataFrame()
    out['tx_id'] = df.get('tx_id', pd.Series(range(len(df))))
    out['amount'] = df['amount'].astype(float)
    out['log_amount'] = np.log1p(out['amount'])
    out['hour'] = pd.to_datetime(df['timestamp']).dt.hour.fillna(0).astype(int)
    out['is_kyc'] = df.get('kyc_verified', pd.Series([False]*len(df))).fillna(False).astype(int)

    # payment system dummies (robust)
    ps = df.get('payment_system', pd.Series(['UPI'] * len(df))).fillna('UPI').astype(str)
    ps_dummies = pd.get_dummies(ps, prefix='ps')
    out = pd.concat([out.reset_index(drop=True), ps_dummies.reset_index(drop=True)], axis=1)

    out['is_round_1000'] = out['amount'].apply(lambda a: 1 if is_round_number(a, 1000) else 0).astype(int)

    # days dormant
    if 'last_active' in df.columns:
        last_active = pd.to_datetime(df['last_active'], errors='coerce')
        out['days_dormant'] = (pd.Timestamp.utcnow() - last_active).dt.days.fillna(999).astype(int)
    else:
        out['days_dormant'] = 999

    # velocity and reversals (naive but effective)
    # compute per-payer sliding window counts
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    window_mins = 10
    velocity_map = defaultdict(deque)
    reversal_map = defaultdict(int)
    vel_map = {}
    rev_map = {}
    for _, row in df_sorted.iterrows():
        payer = row.get('payer_id', None)
        ts = pd.to_datetime(row['timestamp'])
        dq = velocity_map[payer]
        # pop older
        while dq and (ts - dq[0]).total_seconds() > window_mins * 60:
            dq.popleft()
        dq.append(ts)
        vel_map[row.get('tx_id')] = len(dq)
        reversal_map[payer] += 1 if str(row.get('status','')).lower() == 'reversed' else 0
        rev_map[row.get('tx_id')] = reversal_map[payer]

    out['velocity_10m'] = out['tx_id'].apply(lambda t: vel_map.get(t, 0)).astype(int)
    out['reversal_count'] = out['tx_id'].apply(lambda t: rev_map.get(t, 0)).astype(int)

    # deterministic rule score if present
    out['det_score'] = df.get('score', pd.Series([0]*len(df))).fillna(0).astype(float)

    return out

# ----------------- label mapping -----------------
def build_labels(df, df_str=None):
    """
    Map flags into three labels:
    - STR -> 'STR_review'
    - High-Risk/Medium -> 'Suspicious'
    - Low/others -> 'Normal'
    If df_str provided (from str_audit.jsonl), use it; else fallback to df['flag'].
    """
    if df_str is not None:
        flag_map = {r['tx_id']: r['flag'] for _, r in df_str.iterrows()}
        def map_flag(txid, fallback):
            f = flag_map.get(txid, fallback)
            if f == 'STR':
                return 'STR_review'
            if f in ('High-Risk', 'Medium'):
                return 'Suspicious'
            return 'Normal'
        y = df.apply(lambda row: map_flag(row.get('tx_id'), row.get('flag', 'Low')), axis=1)
    else:
        if 'flag' in df.columns:
            def fallback_map(f):
                if f == 'STR': return 'STR_review'
                if f in ('High-Risk','Medium'): return 'Suspicious'
                return 'Normal'
            y = df['flag'].fillna('Low').apply(fallback_map)
        else:
            y = pd.Series(['Normal'] * len(df))
    return y.astype(str)

# ----------------- training helpers -----------------
def find_best_threshold(y_true_bin, probs_for_class):
    precisions, recalls, thresholds = precision_recall_curve(y_true_bin, probs_for_class)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    if len(f1s) == 0:
        return 0.5, 0.0, 0.0, 0.0
    best_idx = int(np.nanargmax(f1s))
    # thresholds array length = len(f1s) - 1 sometimes; guard
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_thr, float(precisions[best_idx]), float(recalls[best_idx]), float(f1s[best_idx])

def encode_labels(y_series):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series)
    return y_enc, le

# ----------------- model training -----------------
def run_randomized_search_rf(X_train, y_train_enc):
    # pipeline: SMOTEENN -> scaler -> RF (if SMOTEENN not available fall back to SMOTE)
    if SMOTEENN is not None and ImbPipeline is not None:
        sampler = SMOTEENN(random_state=RANDOM_STATE)
        pipe = ImbPipeline([
            ('smoteenn', sampler),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1, random_state=RANDOM_STATE))
        ])
    else:
        # fallback: simple pipeline without SMOTEENN
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1, random_state=RANDOM_STATE))
        ])
    param_dist = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [6, 8, 12, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=cv, scoring='f1_macro', random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    print("Starting RandomForest RandomizedSearchCV...")
    rs.fit(X_train, y_train_enc)
    print("RF best params:", rs.best_params_)
    return rs.best_estimator_

def run_randomized_search_xgb(X_train, y_train_enc):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost not available in environment.")
    if SMOTEENN is not None and ImbPipeline is not None:
        sampler = SMOTEENN(random_state=RANDOM_STATE)
        pipe = ImbPipeline([
            ('smoteenn', sampler),
            ('scaler', StandardScaler()),
            ('clf', xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1, random_state=RANDOM_STATE))
        ])
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1, random_state=RANDOM_STATE))
        ])
    param_dist = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__subsample': [0.6, 0.8, 1.0],
        'clf__colsample_bytree': [0.6, 0.8, 1.0]
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=cv, scoring='f1_macro', random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    print("Starting XGBoost RandomizedSearchCV...")
    rs.fit(X_train, y_train_enc)
    print("XGB best params:", rs.best_params_)
    return rs.best_estimator_

# ----------------- main flow -----------------
def main():
    print("Loading data...")
    df = load_audit_csv()
    df_str = load_str_jsonl()
    print(f"Rows in audit_summary: {len(df)}")
    if df_str is not None:
        print(f"STR records loaded: {len(df_str)}")
    else:
        print("No str_audit.jsonl found; using flags in audit_summary if present.")

    # temporal split: sort by timestamp (old -> new), use first 80% as train, last 20% as test
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df_sorted)
    split_idx = int(0.8 * n) if n > 10 else int(0.7 * n)
    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
    print(f"Temporal split: train rows={len(train_df)}, test rows={len(test_df)} (split index {split_idx})")

    # build features
    X_all = prepare_features(df_sorted)
    # build labels using df_str if present (full DF)
    y_all = build_labels(df_sorted, df_str)
    labels = sorted(y_all.unique())
    print("Label distribution (overall):", Counter(y_all))

    # split feature matrices accordingly
    X_train = X_all.iloc[:split_idx].drop(columns=['tx_id']).reset_index(drop=True)
    X_test = X_all.iloc[split_idx:].drop(columns=['tx_id']).reset_index(drop=True)
    y_train = y_all.iloc[:split_idx].reset_index(drop=True)
    y_test = y_all.iloc[split_idx:].reset_index(drop=True)

    # encode labels to integers for model training (needed for xgboost)
    le = LabelEncoder()
    le.fit(labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Train RandomForest
    rf_est = run_randomized_search_rf(X_train, y_train_enc)

    # Try XGBoost
    xgb_est = None
    if XGBOOST_AVAILABLE:
        try:
            xgb_est = run_randomized_search_xgb(X_train, y_train_enc)
        except Exception as e:
            print("XGBoost training failed, continuing with RF only. Error:", e)
            xgb_est = None

    # Candidate models dictionary: name->estimator (pipelines that predict encoded labels)
    candidates = {'rf': rf_est}
    if xgb_est is not None:
        candidates['xgb'] = xgb_est

    final_reports = {}
    # evaluate candidates on test set: compute probabilities for STR (mapped by label encoder)
    for name, est in candidates.items():
        print("Evaluating candidate:", name)
        # predict_proba expects numeric columns; est pipelines return numeric-coded proba for integer classes
        try:
            probs = est.predict_proba(X_test)
        except Exception as e:
            print(f"Model {name} predict_proba failed: {e}")
            # fallback: try predict_proba on scaler+clf if pipeline structure different
            probs = np.zeros((len(X_test), len(labels)))

        # determine index of STR_review in label encoding
        if 'STR_review' in le.classes_:
            str_idx = int(np.where(le.classes_ == 'STR_review')[0][0])
        elif 'Suspicious' in le.classes_:
            str_idx = int(np.where(le.classes_ == 'Suspicious')[0][0])
        else:
            str_idx = int(np.argmax(probs.mean(axis=0)))

        probs_str = probs[:, str_idx]
        # binary ground truth for STR
        y_test_bin = np.array([1 if y == 'STR_review' else 0 for y in y_test])
        best_thr, best_prec, best_rec, best_f1 = find_best_threshold(y_test_bin, probs_str)
        print(f"{name} best_thr={best_thr:.3f} prec={best_prec:.3f} rec={best_rec:.3f} f1={best_f1:.3f}")

        # prepare predicted multi-class labels (inverse_transform of argmax)
        try:
            pred_idx = np.argmax(probs, axis=1)
            pred_labels = le.inverse_transform(pred_idx)
        except Exception:
            # fallback: use estimator.predict and convert if numeric
            pred_raw = est.predict(X_test)
            try:
                # if numeric, inverse transform
                if np.issubdtype(type(pred_raw[0]), np.integer):
                    pred_labels = le.inverse_transform(pred_raw)
                else:
                    pred_labels = np.array(pred_raw, dtype=str)
            except Exception:
                pred_labels = np.array(['Normal'] * len(X_test))

        report_text = classification_report(y_test, pred_labels, zero_division=0)
        cm = confusion_matrix(y_test, pred_labels, labels=labels)
        try:
            auc_ovr = roc_auc_score(pd.get_dummies(y_test), probs, multi_class='ovr')
        except Exception:
            auc_ovr = None

        final_reports[name] = {
            'estimator': est,
            'probs_str': probs_str,
            'best_threshold': best_thr,
            'best_f1': best_f1,
            'precision': best_prec,
            'recall': best_rec,
            'report_text': report_text,
            'confusion_matrix': cm.tolist(),
            'auc_ovr': float(auc_ovr) if auc_ovr is not None else None
        }

    # Select best candidate by best_f1
    chosen_name = max(final_reports.items(), key=lambda kv: kv[1]['best_f1'])[0]
    chosen_entry = final_reports[chosen_name]
    chosen_est = chosen_entry['estimator']
    chosen_thr = float(chosen_entry['best_threshold'])
    print("Selected best model:", chosen_name, "with threshold:", chosen_thr)

    # Save chosen model artifact (include label encoder and labels and threshold)
    model_artifact = {
        'model': chosen_est,
        'labels': list(le.classes_),
        'label_encoder': le,
        'threshold': chosen_thr
    }
    joblib.dump(model_artifact, MODEL_OUT)
    print("Saved model artifact to", MODEL_OUT)

    # Predictions across full dataset
    X_full = X_all.drop(columns=['tx_id']).reset_index(drop=True)
    try:
        probs_full = chosen_est.predict_proba(X_full)
    except Exception:
        probs_full = np.zeros((len(X_full), len(le.classes_)))
    if 'STR_review' in le.classes_:
        str_idx_full = int(np.where(le.classes_ == 'STR_review')[0][0])
    elif 'Suspicious' in le.classes_:
        str_idx_full = int(np.where(le.classes_ == 'Suspicious')[0][0])
    else:
        str_idx_full = int(np.argmax(probs_full.mean(axis=0)))
    probs_str_full = probs_full[:, str_idx_full]
    # multi-label preds via threshold override
    preds_bin_full = (probs_str_full >= chosen_thr).astype(int)
    est_argmax = np.argmax(probs_full, axis=1)
    est_labels = le.inverse_transform(est_argmax)
    final_multi = [ 'STR_review' if b==1 else lab for b, lab in zip(preds_bin_full, est_labels) ]

    out_df = pd.DataFrame({
        'tx_id': X_all['tx_id'],
        'ml_prob_STR': probs_str_full,
        'ml_pred_by_threshold': final_multi,
        'det_score': X_all['det_score']
    })
    out_df.to_csv(OUT_PRED_CSV, index=False)
    print("Wrote ML predictions to", OUT_PRED_CSV)

    # Ensemble: IsolationForest anomaly score + model prob (normalize & combine)
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=RANDOM_STATE)
    try:
        iso.fit(X_full)
        anom_score = -iso.decision_function(X_full)  # bigger -> more anomalous
        anom_norm = (anom_score - anom_score.min()) / (anom_score.max() - anom_score.min() + 1e-12)
    except Exception:
        anom_norm = np.zeros(len(X_full))

    # normalize model prob 0-1 (already 0-1)
    model_prob_norm = probs_str_full
    ensemble_prob = 0.6 * model_prob_norm + 0.4 * anom_norm
    ensemble_pred = ['STR_review' if p >= 0.55 else ('Suspicious' if p >= 0.35 else 'Normal') for p in ensemble_prob]
    ensemble_df = pd.DataFrame({
        'tx_id': X_all['tx_id'],
        'model_prob': model_prob_norm,
        'anom_score': anom_norm,
        'ensemble_prob': ensemble_prob,
        'ensemble_pred': ensemble_pred
    })
    ensemble_df.to_csv(OUT_ENSEMBLE_CSV, index=False)
    print("Wrote ensemble predictions to", OUT_ENSEMBLE_CSV)

    # Save a short textual report
    with open(REPORT_OUT, 'w', encoding='utf-8') as rf:
        rf.write("ML Upgrade Report\n")
        rf.write(f"Selected model: {chosen_name}\n")
        rf.write(f"Threshold (STR): {chosen_thr}\n\n")
        for name, entry in final_reports.items():
            rf.write(f"----- Model: {name} -----\n")
            rf.write(f"best_f1: {entry['best_f1']}\n")
            rf.write(f"precision: {entry['precision']}, recall: {entry['recall']}\n")
            rf.write("classification_report:\n")
            rf.write(entry['report_text'] + "\n")
            rf.write("confusion_matrix:\n")
            rf.write(json.dumps(entry['confusion_matrix']) + "\n")
            rf.write(f"auc_ovr: {entry['auc_ovr']}\n\n")
        # top feature importances if available
        rf_clf = None
        try:
            # try to dig into pipeline to get classifier
            if hasattr(chosen_est, 'named_steps') and 'clf' in chosen_est.named_steps:
                rf_clf = chosen_est.named_steps['clf']
            else:
                rf_clf = chosen_est
            if hasattr(rf_clf, 'feature_importances_'):
                feat_names = X_full.columns.tolist()
                importances = rf_clf.feature_importances_
                fi = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:20]
                rf.write("Top feature importances:\n")
                for f,v in fi:
                    rf.write(f"{f}: {v}\n")
        except Exception:
            rf.write("Could not extract feature importances.\n")
    print("Wrote textual report to", REPORT_OUT)

    # PR curve and ROC summary for STR on test set
    y_test_bin = np.array([1 if y == 'STR_review' else 0 for y in y_test])
    probs_test = final_reports[chosen_name]['probs_str']
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_test_bin, probs_test)
        plt.figure(figsize=(6,4))
        plt.plot(recalls, precisions, label='PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve (STR)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PR_PLOT)
        plt.close()
        print("Wrote PR curve to", PR_PLOT)
    except Exception as e:
        print("PR curve generation failed:", e)

    # ROC (OVR) summary
    try:
        y_test_bin_multi = pd.get_dummies(y_test)
        if probs.shape[1] == y_test_bin_multi.shape[1]:
            roc = roc_auc_score(y_test_bin_multi, probs, multi_class='ovr')
            plt.figure(figsize=(6,4))
            plt.text(0.1,0.6,f"ROC AUC (ovr): {roc:.3f}", fontsize=12)
            plt.title('ROC AUC summary')
            plt.axis('off')
            plt.savefig(ROC_PLOT)
            plt.close()
            print("Wrote ROC summary to", ROC_PLOT)
    except Exception:
        pass

    print("ML upgrade flow COMPLETE.")
    print("Artifacts:")
    print(" - model:", MODEL_OUT)
    print(" - ml preds:", OUT_PRED_CSV)
    print(" - ensemble preds:", OUT_ENSEMBLE_CSV)
    print(" - report:", REPORT_OUT)
    print(" - plots:", PR_PLOT, ROC_PLOT)

if __name__ == "__main__":
    main()