# src/run_str_engine.py
"""
Runner for STR engine.
Input: data/audit_summary.csv (from simulation) OR data/audit_logs.jsonl
Output:
 - data/str_audit.jsonl  (detailed audit per tx with rules)
 - data/str_candidates.csv (subset of flagged txs)
"""
import os
import pandas as pd
import json
from str_engine import STRRulesEngine

INPUT_CSV = os.path.join("data", "audit_summary.csv")
JSONL_OUT = os.path.join("data", "str_audit.jsonl")
CSV_OUT = os.path.join("data", "str_candidates.csv")

def load_transactions():
    if os.path.exists(INPUT_CSV):
        df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
        # convert rows to dict
        rows = df.to_dict(orient="records")
        # ensure timestamp isoformat strings
        for r in rows:
            if isinstance(r.get("timestamp"), pd.Timestamp):
                r["timestamp"] = r["timestamp"].isoformat()
        return rows
    # fallback to jsonl
    jsonl = os.path.join("data","audit_logs.jsonl")
    if os.path.exists(jsonl):
        rows = []
        with open(jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    raise FileNotFoundError("No input transaction file found. Run simulation first.")

def fake_sanctions_check(entity_id):
    # simple deterministic sanction: ids ending with 7 flagged
    try:
        return str(entity_id).endswith("7")
    except:
        return False

def main():
    print("Loading transactions...")
    txs = load_transactions()
    engine = STRRulesEngine(config_path="rules.yml", sanctions_fn=fake_sanctions_check)
    print(f"Loaded rules config {engine.rule_config_version} with {len(engine.rules)} rules.")
    audits = engine.evaluate_batch(txs)
    # write jsonl
    with open(JSONL_OUT, 'w', encoding='utf-8') as f:
        for a in audits:
            f.write(json.dumps(a) + "\n")
    # write CSV of flagged transactions (Medium+)
    flagged = [a for a in audits if a['flag'] in ("Medium","High-Risk","STR")]
    import pandas as pd
    df = pd.DataFrame(flagged)
    df_flat = df[["tx_id","timestamp","payer_id","beneficiary_id","amount","flag","score"]]
    df_flat.to_csv(CSV_OUT, index=False)
    print(f"Wrote {len(audits)} audits to {JSONL_OUT}")
    print(f"Wrote {len(flagged)} flagged records to {CSV_OUT}")

if __name__ == "__main__":
    main()