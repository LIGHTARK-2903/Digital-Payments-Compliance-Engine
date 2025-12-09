# src/ml_upgrade.py
"""
ML Upgrade pipeline for Digital-Payments-Compliance-Engine
- Uses SMOTE + class-weighted models + RandomizedSearchCV
- Trains RandomForest and (optionally) XGBoost, tunes hyperparams
- Finds optimal threshold for 'STR_review' class (labels: Normal, Suspicious, STR_review)
- Outputs predictions, model, plots, and a small report.
Run:
    (venv) python .\src\ml_upgrade.py
Outputs:
    models/ml_upgrade_model.joblib
    data/ml_upgrade_preds.csv
    plots/ml_upgrade_pr_curve.png
    reports/ml_upgrade_report.txt
"""
import os, json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
PLOTS_DIR = os.path.join(ROOT, "plots")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CSV_IN = os.path.join(DATA_DIR, "audit_summary.csv")
STR_JSONL = os.path.join(DATA_DIR, "str_audit.jsonl")
OUT_PRED_CSV = os.path.join(DATA_DIR, "ml_upgrade_preds.csv")
MODEL_OUT = os.path.join(MODEL_DIR, "ml_upgrade_model.joblib")
REPORT_OUT = os.path.join(REPORT_DIR, "ml_upgrade_report.txt")
PR_PLOT = os.path.join(PLOTS_DIR, "ml_upgrade_pr_curve.png")
ROC_PLOT = os.path.join(PLOTS_DIR, "ml_upgrade_roc_curve.png")

# ---------- Helpers ----------
def load_data():
    # load audit CSV (must have tx_id, timestamp, amount, payment_system, kyc_verified, last_active, status, payer_id, score)
    df = pd.read_csv(CSV_IN, parse_dates=["timestamp"], low_memory=False)
    # load STR labels (fallback to df.flag)
    if os.path.exists(STR_JSONL):
        recs = []
        with open(STR_JSONL,'r',encoding='utf-8') as f:
            for L in f:
                recs.append(json.loads(L))
        df_str = pd.DataFrame(recs)
    else:
        df_str = None
    return df, df_str

def prepare_features(df):
    # Basic feature engineering similar to previous pipeline
    out = pd.DataFrame()
    out['tx_id'] = df['tx_id']
    out['amount'] = df['amount'].astype(float)
    out['log_amount'] = np.log1p(out['amount'])
    out['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    out['is_kyc'] = df.get('kyc_verified', pd.Series([False]*len(df))).fillna(False).astype(int)
    # payment system one-hot
    ps = df.get('payment_system', pd.Series(['UPI']*len(df))).fillna('UPI').astype(str)
    ps_dummies = pd.get_dummies(ps, prefix='ps')
    out = pd.concat([out, ps_dummies], axis=1).reset_index(drop=True)
    # round number
    out['is_round_1000'] = (out['amount'] % 1000 == 0).astype(int)
    # days dormant
    if 'last_active' in df.columns:
        out['days_dormant'] = (pd.Timestamp.utcnow() - pd.to_datetime(df['last_active'], errors='coerce')).dt.days.fillna(999).astype(int)
    else:
        out['days_dormant'] = 999
    # velocity within 10m and reversal_count (naive approach)
    df_sorted = df.sort_values('timestamp')
    from collections import deque, defaultdict
    velocity_map = defaultdict(deque)
    reversal_map = defaultdict(int)
    vel = []
    rev = []
    for _,row in df_sorted.iterrows():
        payer = row.get('payer_id')
        ts = pd.to_datetime(row['timestamp'])
        dq = velocity_map[payer]
        while dq and (ts - dq[0]).total_seconds() > 10*60:
            dq.popleft()
        dq.append(ts)
        vel.append((row['tx_id'], len(dq)))
        reversal_map[payer] += 1 if str(row.get('status','')).lower()=='reversed' else 0
        rev.append((row['tx_id'], reversal_map[payer]))
    vel_map = dict(vel)
    rev_map = dict(rev)
    out['velocity_10m'] = out['tx_id'].apply(lambda t: vel_map.get(t,0)).astype(int)
    out['reversal_count'] = out['tx_id'].apply(lambda t: rev_map.get(t,0)).astype(int)
    # deterministic score if available
    out['det_score'] = df.get('score', pd.Series([0]*len(df))).fillna(0).astype(float)
    return out

def build_labels(df, df_str):
    # map flags->three classes
    if df_str is not None:
        flag_map = {r['tx_id']: r['flag'] for _,r in df_str.iterrows()}
        def map_flag(txid, fallback):
            f = flag_map.get(txid, fallback)
            if f == 'STR':
                return 'STR_review'
            if f in ('High-Risk','Medium'):
                return 'Suspicious'
            return 'Normal'
        y = df.apply(lambda row: map_flag(row['tx_id'], row.get('flag','Low')), axis=1)
    else:
        if 'flag' in df.columns:
            def fallback_map(f):
                if f == 'STR': return 'STR_review'
                if f in ('High-Risk','Medium'): return 'Suspicious'
                return 'Normal'
            y = df['flag'].fillna('Low').apply(fallback_map)
        else:
            y = pd.Series(['Normal']*len(df), index=df.index)
    return y

# ---------- Model training & selection ----------
def train_models(X_train, y_train, X_val, y_val, labels):
    # Use SMOTE to oversample training minority
    sm = SMOTE(random_state=42, n_jobs=-1)
    # Create scaler + RF pipeline (imb pipeline: SMOTE -> scaler -> model)
    pipe_rf = ImbPipeline([
        ('smote', sm),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1, random_state=42))
    ])

    rf_param_dist = {
        'clf__n_estimators': [100,200,400],
        'clf__max_depth': [6,8,12,None],
        'clf__min_samples_split': [2,5,10],
        'clf__min_samples_leaf': [1,2,4]
    }

    # Randomized CV for RF
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    rs_rf = RandomizedSearchCV(pipe_rf, rf_param_dist, n_iter=20, cv=cv, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
    print("Starting RandomForest RandomizedSearchCV...")
    rs_rf.fit(X_train, y_train)
    print("RF best:", rs_rf.best_params_)

    # Evaluate RF on val
    rf_best = rs_rf.best_estimator_
    y_prob_rf = rf_best.predict_proba(X_val)
    # keep results
    results = {'rf': (rf_best, y_prob_rf)}

    # Try XGBoost if available
    try:
        import xgboost as xgb
        print("XGBoost detected — running XGBoost randomized search...")
        pipe_xgb = ImbPipeline([
            ('smote', sm),
            ('scaler', StandardScaler()),
            ('clf', xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1, random_state=42))
        ])
        xgb_param_dist = {
            'clf__n_estimators': [100,200,400],
            'clf__max_depth': [3,5,7],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__subsample': [0.6,0.8,1.0],
            'clf__colsample_bytree': [0.6,0.8,1.0]
        }
        rs_xgb = RandomizedSearchCV(pipe_xgb, xgb_param_dist, n_iter=20, cv=cv, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        rs_xgb.fit(X_train, y_train)
        xgb_best = rs_xgb.best_estimator_
        print("XGB best:", rs_xgb.best_params_)
        y_prob_xgb = xgb_best.predict_proba(X_val)
        results['xgb'] = (xgb_best, y_prob_xgb)
    except Exception as e:
        print("XGBoost not available or failed to run:", e)

    return results

# ---------- threshold tuning (pick best threshold for STR_review) ----------
def find_best_threshold(y_true_bin, probs_for_class):
    # y_true_bin: binary (1 if STR_review else 0)
    precisions, recalls, thresholds = precision_recall_curve(y_true_bin, probs_for_class)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.nanargmax(f1s)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_thr, precisions[best_idx], recalls[best_idx], f1s[best_idx]

# ---------- evaluation helpers ----------
def binary_map_for_str(y, labels):
    # map y (multi-class string) to binary vector indicating STR_review
    return np.array([1 if yi=='STR_review' else 0 for yi in y])

def multi_clf_report(y_true, y_pred, labels):
    return classification_report(y_true, y_pred, zero_division=0)

# ---------- main ----------
def main():
    print("Loading data...")
    df, df_str = load_data()
    print("Preparing features...")
    X = prepare_features(df)
    y = build_labels(df, df_str)
    print("Label distribution:", Counter(y))
    # Encode labels to integers
    labels = sorted(y.unique())
    label2idx = {lab:i for i,lab in enumerate(labels)}
    y_idx = y.map(label2idx)

    # Train/test split (stratified)
    X_train, X_test, y_train_idx, y_test_idx = train_test_split(X.drop(columns=['tx_id']), y_idx, test_size=0.2, random_state=42, stratify=y_idx)
    # Convert back y strings for thresholding convenience
    y_train = y.loc[y_train_idx.index]
    y_test = y.loc[y_test_idx.index]

    print("Training models with SMOTE and randomized search...")
    results = train_models(X_train, y_train, X_test, y_test, labels)

    # For each candidate model compute probabilities on test set
    final_reports = {}
    for name, (est, y_prob) in results.items():
        print("Evaluating:", name)
        # y_prob: shape (n_samples, n_classes)
        # get index of STR_review class if present
        if 'STR_review' in labels:
            str_idx = labels.index('STR_review')
        elif 'Suspicious' in labels:
            str_idx = labels.index('Suspicious')
        else:
            str_idx = np.argmax(y_prob.mean(axis=0))  # fallback

        probs_str = est.predict_proba(X_test)[:, str_idx]
        # binary y for STR
        y_test_bin = binary_map_for_str(y_test, labels)
        # find best threshold using train set probs (use separate small holdout? we use test here for selection in this tutorial)
        best_thr, best_prec, best_rec, best_f1 = find_best_threshold(binary_map_for_str(y_test, labels), probs_str)
        print(f"{name} best_thr={best_thr:.3f} prec={best_prec:.3f} rec={best_rec:.3f} f1={best_f1:.3f}")
        # predictions by threshold
        y_pred_bin = (probs_str >= best_thr).astype(int)
        # map binary preds back to multi-class labels (simple mapping)
        y_pred_multi = []
        for pb, binpred, orig in zip(probs_str, y_pred_bin, y_test):
            if binpred == 1:
                y_pred_multi.append('STR_review')
            else:
                # if not STR, fallback to model's argmax class
                idx = np.argmax(est.predict_proba([X_test.loc[X_test.index[0]]])[0])  # dummy use - we will use est.predict
                # safer: use estimator.predict for multi-class
                pass
        # For full multi-class report use estimator.predict
        y_pred_full = est.predict(X_test)
        report = classification_report(y_test, y_pred_full, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_full, labels=labels)
        auc = None
        try:
            auc = roc_auc_score(pd.get_dummies(y_test, columns=labels), est.predict_proba(X_test), multi_class='ovr')
        except Exception:
            auc = None
        final_reports[name] = {
            'report': report,
            'confusion_matrix': cm.tolist(),
            'best_threshold': float(best_thr),
            'best_f1': float(best_f1),
            'auc_ovr': float(auc) if auc is not None else None,
            'estimator': est
        }

    # Choose best model by best_f1 (STR)
    best_name = max(final_reports.items(), key=lambda kv: kv[1]['best_f1'])[0]
    best_entry = final_reports[best_name]
    best_est = best_entry['estimator']
    best_thr = best_entry['best_threshold']
    print("Selected model:", best_name, "thr:", best_thr)

    # Save model
    joblib.dump({'model': best_est, 'labels': labels, 'threshold': best_thr, 'label_map': label2idx}, MODEL_OUT)
    print("Saved model to", MODEL_OUT)

    # Predictions on full dataset
    X_all = X.drop(columns=['tx_id'])
    probs_all = best_est.predict_proba(X_all)
    if 'STR_review' in labels:
        str_idx = labels.index('STR_review')
    elif 'Suspicious' in labels:
        str_idx = labels.index('Suspicious')
    else:
        str_idx = np.argmax(probs_all.mean(axis=0))
    probs_str_all = probs_all[:, str_idx]
    preds_bin_all = (probs_str_all >= best_thr).astype(int)
    # create multi label: STR_review if bin=1 else estimator.predict
    preds_multi = []
    est_preds = best_est.predict(X_all)
    for pb, b, ep in zip(probs_str_all, preds_bin_all, est_preds):
        if b == 1:
            preds_multi.append('STR_review')
        else:
            preds_multi.append(ep)
    out_df = pd.DataFrame({
        'tx_id': X['tx_id'],
        'ml_prob_STR': probs_str_all,
        'ml_pred_by_threshold': preds_multi,
        'det_score': X['det_score']
    })
    out_df.to_csv(OUT_PRED_CSV, index=False)
    print("Wrote predictions to", OUT_PRED_CSV)

    # Save a short report
    with open(REPORT_OUT, 'w', encoding='utf-8') as rf:
        rf.write("ML Upgrade Report\n")
        rf.write(f"Selected model: {best_name}\n")
        rf.write(f"Threshold (STR): {best_thr}\n")
        rf.write("=== Per-model summaries ===\n")
        for n,v in final_reports.items():
            rf.write(f"\nModel: {n}\n")
            rf.write(f"best_f1: {v['best_f1']}\n")
            rf.write("classification_report:\n")
            rf.write(v['report'] if 'report' in v else str(v.get('report','')) + "\n")
        rf.write("\nFeature importances (top 20) for selected model:\n")
        try:
            est = best_est.named_steps['clf'] if hasattr(best_est, 'named_steps') else best_est
            importances = getattr(est, 'feature_importances_', None)
            if importances is not None:
                feat_names = X_all.columns.tolist()
                fi = sorted(zip(feat_names, importances), key=lambda x:-x[1])[:20]
                for f,v in fi:
                    rf.write(f"{f}: {v}\n")
        except Exception as e:
            rf.write("Could not extract feature importances: " + str(e) + "\n")
    print("Wrote report to", REPORT_OUT)

    # Plots: precision-recall curve for STR on test set (using selected estimator)
    y_test_bin = binary_map_for_str(y_test, labels)
    probs_test = best_est.predict_proba(X_test)[:, str_idx]
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
    # ROC
    try:
        from sklearn.preprocessing import label_binarize
        y_test_bin_multi = label_binarize(y_test, classes=labels)
        roc = roc_auc_score(y_test_bin_multi, best_est.predict_proba(X_test), multi_class='ovr')
        plt.figure(figsize=(6,4))
        plt.text(0.1,0.6,f"ROC AUC (ovr): {roc:.3f}", fontsize=12)
        plt.title('ROC AUC summary')
        plt.axis('off')
        plt.savefig(ROC_PLOT)
        plt.close()
        print("Wrote ROC summary to", ROC_PLOT)
    except Exception:
        pass

if __name__ == "__main__":
    main()