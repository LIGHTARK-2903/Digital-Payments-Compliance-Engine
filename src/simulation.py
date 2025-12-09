"""
src/simulation.py
Digital Payments Flow & Compliance Simulation
- Generates synthetic customers and transactions (UPI/RTGS/NEFT)
- Runs KYC, sanctions, device/IP checks
- Applies deterministic STR rules (velocity, structuring, spike, reversal loops)
- Produces audit logs (JSONL), summary CSV, and a transaction heatmap PNG
Run:
    # from project root
    venv\Scripts\activate     # or .\venv\Scripts\activate
    pip install -r requirements.txt   # ensure pandas, matplotlib, numpy installed
    python src\simulation.py
"""

import os
import json
import random
import string
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG (tunable like rule-config)
# ---------------------------
OUTPUT_DIR = os.path.join(os.getcwd(), "data")
PLOTS_DIR = os.path.join(os.getcwd(), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

N_CUSTOMERS = 400
N_TX = 4000
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RULE_CONFIG = {
    "high_value_threshold": 100000.0,        # absolute high-value -> strong risk
    "non_kyc_amount_threshold": 10000.0,     # non-KYC sending this triggers risk
    "velocity_count_window_minutes": 10,
    "velocity_tx_count_threshold": 5,
    "structuring_split_count": 6,            # splits into more than this many txs within window
    "dormant_spike_days": 90,                # consider account dormant if no txs in last N days
    "reversal_loop_count": 4,                # rapid repeated reversals threshold
    "rtgs_minimum_indicator": 200000.0       # RTGS generally large; RTGS with tiny amounts flagged
}

# ---------------------------
# Synthetic data generators
# ---------------------------
def make_customer(i):
    """Create a synthetic customer profile with KYC, risk tier, last_active"""
    kyc_verified = random.choices([True, False], weights=[0.88, 0.12])[0]
    risk_tier = random.choices(['low','medium','high'], weights=[0.7,0.22,0.08])[0]
    created = datetime.utcnow() - timedelta(days=random.randint(0, 2000))
    last_active = created + timedelta(days=random.randint(0, 2000))
    return {
        "customer_id": f"C{100000+i}",
        "name": f"User_{i}",
        "kyc_verified": bool(kyc_verified),
        "risk_tier": risk_tier,
        "created_at": created.isoformat(),
        "last_active": last_active.isoformat()
    }

def random_ip():
    return ".".join(str(random.randint(1,250)) for _ in range(4))

def device_id():
    return ''.join(random.choices(string.ascii_uppercase+string.digits, k=12))

# ---------------------------
# Compliance checks
# ---------------------------
def kyc_check(customer):
    return customer['kyc_verified']

def sanctions_check(entity_id):
    """Simple synthetic sanctions: a small % are 'sanctioned'"""
    if entity_id.endswith("7"):   # deterministic-ish sanction
        return True
    if random.random() < 0.001:
        return True
    return False

def device_ip_check(device, ip):
    """Simple IP reputation: block some IP ranges synthetically"""
    # rule: if last octet is in 240-250 it's suspicious in this sim
    try:
        last_octet = int(ip.split('.')[-1])
        if last_octet >= 240:
            return False, "ip_reputation"
    except Exception:
        pass
    # device reused many times? skipped here
    return True, None

# ---------------------------
# Transaction lifecycle simulation
# ---------------------------
def make_tx(txid, payer_id, ts, payment_system):
    """Constructs a transaction dict"""
    # amounts follow heavy-tailed distribution to produce some large txs
    base = np.random.exponential(2000)
    amount = round(base + random.choice([10, 50, 100, 500, 1000]), 2)
    # insert occasional large corporate values
    if random.random() < 0.01:
        amount = round(abs(np.random.normal(250000, 100000)),2)
    tx = {
        "tx_id": f"TX{txid:08d}",
        "timestamp": ts.isoformat(),
        "payer_id": payer_id,
        "beneficiary_id": f"BEN{random.randint(1000,9999)}",
        "amount": float(amount),
        "payment_system": payment_system,
        "device_id": device_id(),
        "ip": random_ip(),
        "status": "initiated",
        "meta": {}
    }
    return tx

# ---------------------------
# Rules engine (deterministic rules)
# Each rule returns (score_increment, reason)
# ---------------------------
def rule_high_value(tx, cust):
    if tx['amount'] >= RULE_CONFIG['high_value_threshold']:
        return 80, "high_absolute_value"
    return 0, None

def rule_non_kyc_large(tx, cust):
    if (not cust['kyc_verified']) and tx['amount'] >= RULE_CONFIG['non_kyc_amount_threshold']:
        return 70, "non_kyc_high_amount"
    return 0, None

def rule_rtgs_small(tx, cust):
    if tx['payment_system'] == 'RTGS' and tx['amount'] < RULE_CONFIG['rtgs_minimum_indicator']:
        return 30, "rtgs_unusual_small_amount"
    return 0, None

def rule_sanctions(tx, cust):
    if sanctions_check(tx['beneficiary_id']) or sanctions_check(tx['payer_id']):
        return 100, "sanctions_hit"
    return 0, None

# We'll implement velocity, structuring, dormant spike, and reversal loop in an aggregator that needs history.

# ---------------------------
# Flow engine: processes txs in time order, keeps small in-memory history for window checks
# ---------------------------
def process_transactions(customers, n_tx=N_TX, start_time=None):
    # create lookup
    cust_map = {c['customer_id']: c for c in customers}
    now = start_time or datetime.utcnow()
    # event stream time: spread across 3 days
    stream_start = now - timedelta(days=3)
    txs = []
    for i in range(1, n_tx+1):
        ts = stream_start + timedelta(seconds=random.randint(0, 3*24*3600))
        payer = random.choice(customers)
        ps = random.choices(['UPI','NEFT','RTGS'], weights=[0.86,0.08,0.06])[0]
        tx = make_tx(i, payer['customer_id'], ts, ps)
        txs.append(tx)

    # sort by timestamp
    txs.sort(key=lambda x: x['timestamp'])

    # in-memory sliding windows
    last_tx_time = {cid: None for cid in cust_map}
    window_deque = defaultdict(deque)  # payer_id -> deque of (timestamp, amount, txid, status)
    reversal_tracker = defaultdict(lambda: deque(maxlen=20))  # track statuses for reversal loop detection

    audit_logs = []
    for tx in txs:
        payer = cust_map.get(tx['payer_id'])
        # pre-checks
        device_ok, dev_reason = device_ip_check(tx['device_id'], tx['ip'])
        kyc_ok = kyc_check(payer)
        sanctions_score = 0
        # base deterministic rules
        score = 0
        reasons = []

        # device failure causes immediate block (for this sim)
        if not device_ok:
            tx['status'] = 'blocked'
            reasons.append(dev_reason)
            score += 90

        # apply modular deterministic rules
        r, reason = rule_high_value(tx, payer); score += r; 
        if reason: reasons.append(reason)
        r, reason = rule_non_kyc_large(tx, payer); score += r; 
        if reason: reasons.append(reason)
        r, reason = rule_rtgs_small(tx, payer); score += r;
        if reason: reasons.append(reason)
        r, reason = rule_sanctions(tx, payer); score += r;
        if reason: reasons.append(reason)

        # HISTORY-BASED RULES
        # Velocity: count txs in last N minutes
        ts_obj = datetime.fromisoformat(tx['timestamp'])
        w_minutes = RULE_CONFIG['velocity_count_window_minutes']
        dq = window_deque[tx['payer_id']]
        # remove old
        while dq and (ts_obj - dq[0][0]).total_seconds() > w_minutes*60:
            dq.popleft()
        dq.append((ts_obj, tx['amount'], tx['tx_id'], tx['status']))
        if len(dq) >= RULE_CONFIG['velocity_tx_count_threshold']:
            score += 40
            reasons.append("high_velocity")

        # Structuring: many small txs to same beneficiary in window -> split structuring
        # naive: count number of txs to same beneficiary in last 24 hours
        count_to_ben = sum(1 for t in dq if t[2].startswith("TX"))  # local proxy (could improve)
        if count_to_ben >= RULE_CONFIG['structuring_split_count'] and tx['amount'] < 20000:
            score += 50
            reasons.append("structuring_suspected")

        # Dormant spike: if last active > N days ago and now sudden tx
        last_active = datetime.fromisoformat(payer['last_active'])
        if (ts_obj - last_active).days >= RULE_CONFIG['dormant_spike_days'] and tx['amount'] > 5000:
            score += 45
            reasons.append("dormant_account_spike")

        # simulate reversals sometimes and detect reversal loops
        # random reversal: small chance; if reversal detected frequently -> add score
        reversed_flag = False
        if random.random() < 0.02:
            tx['status'] = 'reversed'
            reversed_flag = True
            reasons.append("auto_reversal_sim")
        else:
            if tx['status'] == 'initiated':
                tx['status'] = 'completed'
        reversal_tracker[tx['payer_id']].append(tx['status'])
        # detect loop: several reversals/completions alternating
        seq = list(reversal_tracker[tx['payer_id']])
        if seq.count('reversed') >= RULE_CONFIG['reversal_loop_count']:
            score += 60
            reasons.append("reversal_loop")

        # Final flagging thresholds
        flag = "Low"
        if score >= 120:
            flag = "STR"
        elif score >= 80:
            flag = "High-Risk"
        elif score >= 40:
            flag = "Medium"
        else:
            flag = "Low"

        # audit record
        audit = {
            "tx_id": tx['tx_id'],
            "timestamp": tx['timestamp'],
            "payer_id": tx['payer_id'],
            "beneficiary_id": tx['beneficiary_id'],
            "amount": tx['amount'],
            "payment_system": tx['payment_system'],
            "status": tx['status'],
            "kyc_verified": payer['kyc_verified'],
            "risk_tier": payer['risk_tier'],
            "device_ok": device_ok,
            "ip": tx['ip'],
            "score": int(score),
            "reasons": reasons,
            "flag": flag,
            "model_version": "deterministic-v1",
            "rule_config_version": f"rules-{datetime.utcnow().date().isoformat()}"
        }
        audit_logs.append(audit)

    # write audit logs to JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, "audit_logs.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in audit_logs:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote audit logs: {jsonl_path}")

    # write CSV summary
    df = pd.DataFrame(audit_logs)
    csv_path = os.path.join(OUTPUT_DIR, "audit_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV summary: {csv_path}")

    # plots: hourly heatmap of transaction counts
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp_dt'].dt.hour
    df['date'] = df['timestamp_dt'].dt.date
    heat = df.groupby(['date','hour']).size().unstack(fill_value=0)
    plt.figure(figsize=(12,6))
    # imshow like heatmap; keep default matplotlib colors (do not set colors)
    plt.imshow(heat, aspect='auto')
    plt.yticks(ticks=np.arange(len(heat.index)), labels=[str(d) for d in heat.index])
    plt.xticks(ticks=np.arange(0,24,1), labels=[str(h) for h in range(24)])
    plt.xlabel("Hour of day")
    plt.ylabel("Date")
    plt.title("Transaction volume heatmap (date vs hour)")
    heat_png = os.path.join(PLOTS_DIR, "tx_heatmap.png")
    plt.colorbar(label="tx count")
    plt.tight_layout()
    plt.savefig(heat_png)
    plt.close()
    print(f"Wrote plot: {heat_png}")

    # rule-hit frequency summary
    reasons_flat = [r for rec in audit_logs for r in rec['reasons']]
    reasons_series = pd.Series(reasons_flat).value_counts()
    rule_csv = os.path.join(OUTPUT_DIR, "rule_hits.csv")
    reasons_series.to_csv(rule_csv)
    print(f"Wrote rule hits: {rule_csv}")

    return audit_logs, df

if __name__ == "__main__":
    # Create synthetic customers
    customers = [make_customer(i) for i in range(N_CUSTOMERS)]
    # For extra realism, mark some customers as dormant (set last_active far in past)
    for i in range(0, int(0.08*N_CUSTOMERS)):
        customers[i]['last_active'] = (datetime.utcnow() - timedelta(days=random.randint(120,600))).isoformat()
    print("Starting simulation...")
    logs, df = process_transactions(customers, n_tx=N_TX)
    print("Simulation completed. Summary:")
    print(df['flag'].value_counts())