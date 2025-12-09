# src/app.py
import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px

DATA = "data"
PLOTS = "plots"

st.set_page_config(
    page_title="Digital Payments Compliance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# Load Data
# ------------------------------------------
@st.cache_data
def load_predictions():
    paths = [
        os.path.join(DATA, "combined_risk.csv"),
        os.path.join(DATA, "ensemble_preds.csv"),
        os.path.join(DATA, "ml_upgrade_preds.csv")
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            return df, os.path.basename(p)
    return None, None

@st.cache_data
def load_str_audit():
    path = os.path.join(DATA, "str_audit.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


df, source_file = load_predictions()
df_str = load_str_audit()

if df is None:
    st.error("No prediction files found (combined_risk.csv / ensemble_preds.csv / ml_upgrade_preds.csv).")
    st.stop()

# ------------------------------------------
# Sidebar
# ------------------------------------------
st.sidebar.title("Compliance Controls")

# Choose display flag column
flag_col = None
for col in ["final_flag", "ensemble_pred", "ml_pred_by_threshold"]:
    if col in df.columns:
        flag_col = col
        break

if flag_col is None:
    st.error("No flag column found in data.")
    st.stop()

unique_flags = sorted(df[flag_col].dropna().unique().tolist())
sel_flags = st.sidebar.multiselect("Filter by Flags", unique_flags, default=unique_flags[:2])

# Score filtering
score_col = None
for c in ["combined_score", "ensemble_prob", "ml_prob_STR"]:
    if c in df.columns:
        score_col = c
        break

min_score = 0.3
if score_col:
    min_score = st.sidebar.slider("Minimum Risk Score", 0.0, 1.0, 0.3)

# Search bar
search_key = st.sidebar.text_input("Search tx_id / payer_id")

# ------------------------------------------
# Filtered data
# ------------------------------------------
filtered = df.copy()

# Apply flag filter
filtered = filtered[filtered[flag_col].isin(sel_flags)]

# Apply score filter
if score_col:
    filtered = filtered[filtered[score_col] >= min_score]

# Apply search
if search_key.strip():
    s = search_key.strip().lower()
    filtered = filtered[
        filtered.apply(
            lambda r: s in str(r.get("tx_id", "")).lower()
            or s in str(r.get("payer_id", "")).lower(),
            axis=1,
        )
    ]

# ------------------------------------------
# Dashboard Header
# ------------------------------------------
st.title("Digital Payments Risk & Compliance Engine")
st.caption("RBI-style Transaction Monitoring, STR Detection, and ML Risk Scoring")

st.info(f"Using prediction file: **{source_file}**")

st.subheader(f"Showing {len(filtered)} transactions")

# ------------------------------------------
# Data Table
# ------------------------------------------
st.dataframe(filtered.head(200), use_container_width=True)

# ------------------------------------------
# Heatmap of Risks
# ------------------------------------------
if score_col:
    st.subheader("Risk Score Distribution")

    fig = px.histogram(filtered, x=score_col, nbins=30, title="Risk Score Histogram", color=flag_col)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# Rule Hits (if present)
# ------------------------------------------
# rule columns = only binary rule hit columns (0/1)
rule_cols = []
for c in df.columns:
    if c.startswith("rule_") and c not in ["rule_config_version"]:
        # include only numerical 0/1 rule columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].max() in [0,1]:
            rule_cols.append(c)

if rule_cols:
    st.subheader("Rule Hit Frequency")

    hit_counts = df[rule_cols].sum().sort_values(ascending=False)
    fig = px.bar(
        x=hit_counts.values,
        y=hit_counts.index,
        orientation="h",
        title="Rule Hit Distribution",
        labels={"x": "Count", "y": "Rule"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# SHAP Section
# ------------------------------------------
st.subheader("Model Explainability (SHAP)")

if os.path.exists(os.path.join(PLOTS, "shap_summary.png")):
    st.image(os.path.join(PLOTS, "shap_summary.png"), caption="SHAP Summary Plot")

if os.path.exists(os.path.join(DATA, "shap_explanations.csv")):
    shap_df = pd.read_csv(os.path.join(DATA, "shap_explanations.csv"))
    st.write("Top SHAP Explanations (Head):")
    st.dataframe(shap_df.head(20))
else:
    st.warning("SHAP explanations not yet generated.")

# ------------------------------------------
# STR Audit Panel
# ------------------------------------------
if df_str is not None:
    st.subheader("STR Audit History")
    st.dataframe(df_str.head(50), use_container_width=True)
else:
    st.info("No STR audit file found.")