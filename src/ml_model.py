# src/ml_model.py
"""
ML add-on for Digital-Payments-Compliance-Engine
- Reads data/audit_summary.csv and data/str_audit.jsonl (STR labels)
- Builds features, creates synthetic labels (Normal / Suspicious / STR_review)
- Trains RandomForest classifier, evaluates, saves model
- Produces data/ml_predictions.csv and data/combined_risk.csv
Run:
    (venv) python .\src\ml_model.py
Requirements:
    pip install scikit-learn pandas numpy joblib
"""
import os
import json
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_IN = os.path.join(DATA_DIR, "audit_summary.csv")
STR_JSONL = os.path.join(DATA_DIR, "str_audit.jsonl")
CSV_ML_OUT = os.path.join(DATA_DIR, "ml_predictions.csv")
CSV_COMBINED = os.path.join(DATA_DIR, "combined_risk.csv")
MODEL_PKL = os.path.join(MODEL_DIR, "rf_model.joblib")

# --- Helpers ---
def load_audit_csv(path=CSV_IN):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df

def load_str_jsonl(path=STR_JSONL):
    if not os.path.exists(path):
        # fallback: if not present, generate labels from audit_summary flag column
        return None
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def is_round_number(amount, multiple=1000):
    try:
        return (abs(amount - round(amount)) < 1e-6) and (amount % multiple == 0)
    except Exception:
        return False

def engineer_features(df):
    # df must contain: tx_id, timestamp, amount, payment_system, kyc_verified, last_active, status, payer_id
    X = pd.DataFrame()
    # numeric amount, and log amount (stabilize)
    X['amount'] = df['amount'].astype(float)
    X['log_amount'] = np.log1p(X['amount'])
    # time features
    X['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    # kyc
    X['is_kyc'] = df.get('kyc_verified', pd.Series([False]*len(df))).fillna(False).astype(bool).astype(int)
    # payment system one-hot (be robust if column missing)
    ps_series = df.get('payment_system', pd.Series(['UPI']*len(df))).fillna('UPI').astype(str)
    ps = pd.Categorical(ps_series)
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # OneHotEncoder expects 2D input; feed codes reshaped
    codes = ps.codes.reshape(-1,1)
    try:
        ps_enc = ohe.fit_transform(codes)
    except Exception:
        # fallback: simple manual one-hot via pandas
        ps_enc = pd.get_dummies(ps_series, prefix='ps').values
        ps_names = [f"ps_{c}" for c in pd.Categorical(ps_series).categories]
        ps_df = pd.DataFrame(ps_enc, columns=ps_names, index=df.index)
        X = pd.concat([X, ps_df], axis=1)
    else:
        ps_names = [f"ps_{c}" for c in ps.categories]
        ps_df = pd.DataFrame(ps_enc, columns=ps_names, index=df.index)
        X = pd.concat([X, ps_df], axis=1)

    # round-number
    X['is_round_1000'] = df['amount'].apply(lambda a: is_round_number(a, 1000)).astype(int)
    # days dormant -- handle missing last_active gracefully
    def days_dormant(last_active):
        try:
            if pd.isna(last_active) or last_active == "" or last_active is None:
                return 999
            la = pd.to_datetime(last_active)
            return (pd.Timestamp.utcnow() - la).days
        except Exception:
            return 999
    # use df.get to avoid KeyError if column absent
    last_active_series = df.get('last_active', pd.Series([None]*len(df)))
    X['days_dormant'] = last_active_series.apply(days_dormant).astype(int)

    # placeholder velocity & reversals - compute from chronological grouping
    # Build simple in-memory rolling counters per payer
    df_sorted = df.sort_values('timestamp')
    window_minutes = 10
    velocity_map = defaultdict(deque)
    reversal_map = defaultdict(lambda: 0)
    velocity_counts = []
    reversal_counts = []
    for _, row in df_sorted.iterrows():
        payer = row.get('payer_id', None)
        ts = pd.to_datetime(row['timestamp'])
        dq = velocity_map[payer]
        # pop old
        while dq and (ts - dq[0]).total_seconds() > window_minutes * 60:
            dq.popleft()
        dq.append(ts)
        velocity_counts.append(len(dq))
        reversal_map[payer] += 1 if str(row.get('status','')).lower() == 'reversed' else 0
        reversal_counts.append(reversal_map[payer])
    # Because we built these in df_sorted order, we need to align back to original df index
    v_series = pd.Series(velocity_counts, index=df_sorted.index).reindex(df.index).fillna(0).astype(int)
    r_series = pd.Series(reversal_counts, index=df_sorted.index).reindex(df.index).fillna(0).astype(int)
    X['velocity_10m'] = v_series.values
    X['reversal_count'] = r_series.values
    return X

def build_labels(df, df_str=None):
    # If df_str present, map tx_id -> flag and create labels:
    # STR -> 'STR_review'; High-Risk/Medium -> 'Suspicious'; Low -> 'Normal'
    if df_str is not None:
        flag_map = {r['tx_id']: r['flag'] for _, r in df_str.iterrows()}
        def map_flag(txid, default_flag):
            f = flag_map.get(txid, default_flag)
            if f == 'STR':
                return 'STR_review'
            if f in ('High-Risk','Medium'):
                return 'Suspicious'
            return 'Normal'
        y = df.apply(lambda row: map_flag(row['tx_id'], row.get('flag','Low')), axis=1)
    else:
        # fallback to using 'flag' column from audit_summary.csv (if present)
        if 'flag' in df.columns:
            def fallback_map(f):
                if f == 'STR': return 'STR_review'
                if f in ('High-Risk','Medium'): return 'Suspicious'
                return 'Normal'
            y = df['flag'].fillna('Low').apply(fallback_map)
        else:
            # if nothing present, label everything 'Normal' (degenerate)
            y = pd.Series(['Normal'] * len(df))
    return y

def train_and_save(X, y):
    # Encode y to numbers
    labels = sorted(y.unique())
    label2idx = {lab:i for i,lab in enumerate(labels)}
    y_idx = y.map(label2idx)
    X_train, X_test, y_train, y_test = train_test_split(X, y_idx, test_size=0.2, random_state=42, stratify=y_idx)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    # Evaluate
    y_pred = clf.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    # Save model + label map
    joblib.dump({'model': clf, 'label_map': label2idx}, MODEL_PKL)
    print(f"Saved model to {MODEL_PKL}")
    return clf, labels

def predict_and_save(clf, labels, X, df):
    # Get probability for STR_review class (if exists), else max prob of suspicious classes
    probs = clf.predict_proba(X)
    # Map label indexes back to label names
    # If 'STR_review' in labels, use that column; else combine 'Suspicious' probs
    label_idx = {lab:i for i,lab in enumerate(labels)}
    prob_str = None
    if 'STR_review' in label_idx:
        prob_str = probs[:, label_idx['STR_review']]
    elif 'Suspicious' in label_idx:
        prob_str = probs[:, label_idx['Suspicious']]
    else:
        prob_str = probs.max(axis=1)
    df_out = df.copy()
    df_out['ml_prob_STR'] = prob_str
    # derive ml_label
    ml_labels = []
    for p in prob_str:
        if p >= 0.65:
            ml_labels.append('STR_review')
        elif p >= 0.4:
            ml_labels.append('Suspicious')
        else:
            ml_labels.append('Normal')
    df_out['ml_label'] = ml_labels
    # Save predictions
    df_out.to_csv(CSV_ML_OUT, index=False)
    print(f"Wrote ML predictions to {CSV_ML_OUT}")
    return df_out

def combine_scores_and_save(df_audit, ml_df):
    # df_audit must contain deterministic 'score' (int); ml_df contains 'ml_prob_STR'
    df_c = df_audit.copy()
    if 'score' not in df_c.columns:
        df_c['score'] = 0
    df_c = df_c.merge(ml_df[['tx_id','ml_prob_STR','ml_label']], on='tx_id', how='left')
    df_c['ml_prob_STR'] = df_c['ml_prob_STR'].fillna(0.0)
    # Combine: simple weighted sum (deterministic normalized + ml probability)
    # normalize deterministic score by (max possible approx) 200 as an example
    df_c['norm_det'] = df_c['score'] / 200.0
    df_c['combined_score'] = df_c['norm_det'] * 0.6 + df_c['ml_prob_STR'] * 0.4
    # final flag mapping thresholds (tunable)
    def final_flag(cs, ml_label):
        if ml_label == 'STR_review' or cs >= 0.75:
            return 'STR'
        if cs >= 0.5 or ml_label == 'Suspicious':
            return 'High-Risk'
        if cs >= 0.3:
            return 'Medium'
        return 'Low'
    df_c['final_flag'] = df_c.apply(lambda r: final_flag(r['combined_score'], r.get('ml_label','Normal')), axis=1)
    df_c.to_csv(CSV_COMBINED, index=False)
    print(f"Wrote combined risk file to {CSV_COMBINED}")
    return df_c

# --- Main pipeline ---
def main():
    print("Loading audit CSV...")
    df = load_audit_csv()
    print(f"Rows in audit summary: {len(df)}")
    print("Loading STR jsonl (if present)...")
    df_str = load_str_jsonl()
    if df_str is None:
        print("No str_audit.jsonl found; using audit_summary flag column as fallback.")
    else:
        print(f"STR records loaded: {len(df_str)}")

    print("Engineering features...")
    X = engineer_features(df)

    print("Building labels...")
    y = build_labels(df, df_str)

    print("Training model...")
    clf, labels = train_and_save(X, y)

    print("Predicting...")
    ml_df = predict_and_save(clf, labels, X, df[['tx_id']])

    print("Combining deterministic + ML scores...")
    combined = combine_scores_and_save(df, ml_df)

    # feature importances
    feat_names = X.columns.tolist()
    importances = clf.feature_importances_
    fi = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:20]
    print("Top feature importances:")
    for f,v in fi:
        print(f"{f}: {v:.4f}")

if __name__ == "__main__":
    main()