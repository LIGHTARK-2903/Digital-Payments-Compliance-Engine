# Digital Payments Risk & Compliance Engine

### UPI / NEFT / RTGS Flow Simulation • STR Rules Engine • ML Risk Scoring • SHAP Explainability

*A Supervisory-Grade Fintech Compliance Stack Inspired by RBI DPSS Standards*

---

## ⚡ Executive Overview

This system is a **full-stack digital payments compliance engine** designed to mirror the internal tooling used by:

* RBI – Department of Payment & Settlement Systems (DPSS)
* RBI – Financial Intelligence & AML Supervision Units
* Leading UPI / RTGS fintech risk divisions
* Bank AML Monitoring Cells

It simulates national-scale UPI/NEFT/RTGS payment flows, performs deterministic STR rule screening, applies machine-learning risk scoring using XGBoost, and generates SHAP-based explainability outputs required for **regulatory-grade auditability**.

This repository is engineered to resemble a **production-aligned compliance prototype** — not a classroom exercise.

---

## 🏗 System Architecture

```
Digital-Payments-Compliance-Engine/
│── data/
│   ├── audit_summary.csv
│   ├── str_audit.jsonl
│   ├── ml_upgrade_preds.csv
│   ├── ensemble_preds.csv
│   ├── shap_explanations.csv
│── src/
│   ├── simulation.py
│   ├── simple_transaction.py
│   ├── str_engine.py
│   ├── ml_upgrade.py               ← ML pipeline (XGBoost + SMOTEENN + temporal split)
│   ├── shap_explain.py             ← SHAP explainability engine
│   ├── app.py                      ← Streamlit Compliance Dashboard
│── reports/
│── plots/
│── models/
│── README.md
│── requirements.txt
```

---

## 🔁 End-to-End Flow Overview

### **1) Payment Simulation Layer**

Generates realistic synthetic UPI / NEFT / RTGS transactions with:

* device fingerprints and IP metadata
* KYC verification attributes
* behavioural anomalies
* dormant account patterns
* reversal loops
* high-value RTGS corridors
* time-of-day patterns

Used to create an AML-scale transaction monitoring dataset.

---

### **2) Deterministic STR Rules Engine (RBI-Inspired)**

Each transaction is evaluated through RBI-aligned AML rules:

| Rule          | Description                                      |
| ------------- | ------------------------------------------------ |
| High Value    | > ₹2,00,000 or abnormal UPI volume spikes        |
| Structuring   | Split transactions to avoid reporting thresholds |
| Velocity      | Rapid transfers in very short time windows       |
| Dormant Spike | Sudden activity after long dormancy              |
| Round Numbers | 10k / 50k / 1L pattern detection                 |
| Reversal Loop | Pay → reverse cycles                             |
| KYC Mismatch  | Risk tier mismatch vs transaction amount         |
| Time-of-Day   | Night-time / unusual behavioural patterns        |

All rules produce **risk scores**, **flags**, and are logged to **audit JSONL**.

---

### **3) Machine Learning Upgrade (XGBoost + SMOTEENN)**

A production-style ML pipeline:

* **Temporal Train/Test Split** (prevents leakage)
* **SMOTEENN** for class balancing
* **RandomizedSearchCV** hyperparameter tuning
* **XGBoost (multi:softprob)** for multi-class risk scoring
* **Threshold Optimization** to maximize STR recall
* **Ensemble Risk Score** blending deterministic + ML outputs

Produces artifacts:

* `ml_upgrade_model.joblib`
* `ml_upgrade_preds.csv`
* `ensemble_preds.csv`

---

### **4) SHAP Explainability Layer**

Regulatory-grade explainability delivering:

* `shap_explanations.csv` (per-transaction feature contributions)
* `shap_summary.png` (global importance visualization)

Supports model governance and audit-driven transparency.

---

### **5) Streamlit Compliance Dashboard**

Analyst-friendly UI for:

* filtering by flags & risk tier
* viewing STR candidates
* inspecting ML + deterministic signals
* checking SHAP explanations
* reviewing audit logs
* transaction-level drill-down

Run with:

```
streamlit run src/app.py
```

---

## 📊 Outputs Generated

* **audit_summary.csv** → All transactions + rule hits
* **str_audit.jsonl** → All STR flags emitted
* **ml_upgrade_preds.csv** → ML predictions
* **ensemble_preds.csv** → Deterministic + ML blended scores
* **shap_explanations.csv** → Per-transaction SHAP contributions
* **plots/** → SHAP summary, ROC/PR curves
* **reports/** → Compliance PDF-ready analysis

---

## 🔧 How to Run the Engine

### **1) Install Dependencies**

```
pip install -r requirements.txt
```

### **2) Run Simulation**

```
python src/simulation.py
```

### **3) Run Deterministic STR Engine**

```
python src/run_str_engine.py
```

### **4) Train the ML Model**

```
python src/ml_upgrade.py
```

### **5) Generate Explainability (SHAP)**

```
python -m src.shap_explain
```

### **6) Launch Dashboard**

```
streamlit run src/app.py
```

---

## 🎯 Policy & Regulatory Relevance

This prototype aligns with principles used by RBI:

* **AML/CFT** suspicious pattern identification
* **Supervised ML model governance expectations**
* **Payment system oversight (DPSS)**
* **Risk-based reporting thresholds**
* **Operational/behavioural anomaly detection**
* **Data lineage & auditability** (JSONL + SHAP)

It demonstrates how **transaction monitoring systems** integrate deterministic rule-based methods with machine learning to produce reliable STR escalations.

---

## 📌 Future Enhancements

* Graph-based entity linkage & network risk mapping
* LSTM sequence modelling for behavioural drift
* Real-time Kafka event ingestion simulation
* Automated SAR/STR drafting module

---

## 📣 Author

Designed and engineered by **Naman Narendra Choudhary** — fintech, ML, compliance, and engineering hybrid.

For collaborations or inquiries: connect via GitHub or LinkedIn.
