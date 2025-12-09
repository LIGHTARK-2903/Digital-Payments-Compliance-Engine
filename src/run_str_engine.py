# src/run_str_engine.py
import os
import pandas as pd
import json
from src.str_engine import STRRulesEngine

INPUT_CSV = os.path.join("data", "audit_summary.csv")
JSONL_OUT = os.path.join("data", "str_audit.jsonl")
CSV_OUT = os.path.join("data", "str_candidates.csv")

def load_transactions():
    if os.path.exists(INPUT_CSV):
        df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
        rows = df.to_dict(orient="records")
        for r in rows:
            if isinstance(r.get("timestamp"), pd.Timestamp):
                r["timestamp"] = r["timestamp"].isoformat()
        return rows
    jsonl = os.path.join("data","audit_logs.jsonl")
    if os.path.exists(jsonl):
        rows = []
        with open(jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    raise FileNotFoundError("No input transaction file found. Run simulation first.")

def fake_sanctions_check(entity_id):
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
    with open(JSONL_OUT, 'w', encoding='utf-8') as f:
        for a in audits:
            f.write(json.dumps(a) + "\n")
    flagged = [a for a in audits if a['flag'] in ("Medium","High-Risk","STR")]
    df = pd.DataFrame(flagged)
    df_flat = df[["tx_id","timestamp","payer_id","beneficiary_id","amount","flag","score"]] if not df.empty else pd.DataFrame(columns=["tx_id","timestamp","payer_id","beneficiary_id","amount","flag","score"])
    df_flat.to_csv(CSV_OUT, index=False)
    print(f"Wrote {len(audits)} audits to {JSONL_OUT}")
    print(f"Wrote {len(flagged)} flagged records to {CSV_OUT}")

if __name__ == "__main__":
    main()