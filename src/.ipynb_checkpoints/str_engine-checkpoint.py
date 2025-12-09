# src/str_engine.py
"""
STR Rules Engine
- Modular rule classes
- Config-driven thresholds
- Returns per-transaction audit records with per-rule details and final flag
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
import yaml
import json
import os
from typing import Tuple, Dict, Any, List

# --- Config loading helper ---
def load_rules_config(path="rules.yml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"rules config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

# --- Base rule ---
@dataclass
class RuleResult:
    name: str
    score: int
    reason: str
    metadata: Dict[str, Any]

class BaseRule:
    name: str = "base"
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    def evaluate(self, tx: Dict[str,Any], context: Dict[str,Any]) -> RuleResult:
        """Return RuleResult"""
        return RuleResult(self.name, 0, "", {})

# --- Rules implementations ---
class HighValueRule(BaseRule):
    name = "high_value"
    def evaluate(self, tx, context):
        thr = self.config.get("high_value_threshold", 100000.0)
        if tx['amount'] >= thr:
            return RuleResult(self.name, self.config.get("weight",80), f"amount >= {thr}", {"amount": tx['amount']})
        return RuleResult(self.name, 0, "", {})

class NonKycLargeRule(BaseRule):
    name = "non_kyc_large"
    def evaluate(self, tx, context):
        thr = self.config.get("non_kyc_amount_threshold", 10000.0)
        if (not tx.get('kyc_verified', True)) and tx['amount'] >= thr:
            return RuleResult(self.name, self.config.get("weight",70), f"non-KYC amount >= {thr}", {"amount": tx['amount']})
        return RuleResult(self.name, 0, "", {})

class RtgsSmallRule(BaseRule):
    name = "rtgs_small"
    def evaluate(self, tx, context):
        rtgs_min = self.config.get("rtgs_minimum_indicator", 200000.0)
        if tx.get('payment_system') == 'RTGS' and tx['amount'] < rtgs_min:
            return RuleResult(self.name, self.config.get("weight",30), "RTGS used for unusually small amount", {"amount": tx['amount']})
        return RuleResult(self.name, 0, "", {})

class SanctionsRule(BaseRule):
    name = "sanctions"
    def evaluate(self, tx, context):
        # context should provide a sanctions_check function if external
        sanctions_fn = context.get("sanctions_fn")
        if sanctions_fn and sanctions_fn(tx.get('payer_id')) or sanctions_fn and sanctions_fn(tx.get('beneficiary_id')):
            return RuleResult(self.name, self.config.get("weight",100), "sanctions/pep_hit", {})
        return RuleResult(self.name, 0, "", {})

class VelocityRule(BaseRule):
    name = "velocity"
    def evaluate(self, tx, context):
        # context maintains per-account deque of recent transactions under key 'velocity_deques'
        window_minutes = self.config.get("velocity_count_window_minutes", 10)
        threshold = self.config.get("velocity_tx_count_threshold", 5)
        payer = tx['payer_id']
        dq_map = context.setdefault("velocity_deques", {})
        dq = dq_map.setdefault(payer, deque())
        ts = datetime.fromisoformat(tx['timestamp'])
        # pop old
        while dq and (ts - dq[0]).total_seconds() > window_minutes * 60:
            dq.popleft()
        dq.append(ts)
        if len(dq) >= threshold:
            return RuleResult(self.name, self.config.get("weight",40), f"{len(dq)} txs in {window_minutes}m", {"count": len(dq)})
        return RuleResult(self.name, 0, "", {"count": len(dq)})

class StructuringRule(BaseRule):
    name = "structuring"
    def evaluate(self, tx, context):
        # detect many small transfers to same beneficiary within a 1-day window
        window_hours = self.config.get("structuring_window_hours", 24)
        split_count = self.config.get("structuring_split_count", 6)
        small_amount_thr = self.config.get("structuring_small_amount", 20000)
        payer = tx['payer_id']
        ben = tx['beneficiary_id']
        dq_map = context.setdefault("structuring_map", defaultdict(list))
        ts = datetime.fromisoformat(tx['timestamp'])
        # purge old entries
        recent_list = []
        for (t_ts, t_ben, t_amt) in dq_map[payer]:
            if (ts - t_ts).total_seconds() <= window_hours * 3600:
                recent_list.append((t_ts,t_ben,t_amt))
        recent_list.append((ts, ben, tx['amount']))
        dq_map[payer] = recent_list
        # count to same ben small txs
        count = sum(1 for (t,b,a) in recent_list if b==ben and a <= small_amount_thr)
        if count >= split_count:
            return RuleResult(self.name, self.config.get("weight",50), f"{count} small txs to {ben} in {window_hours}h", {"count": count})
        return RuleResult(self.name, 0, "", {"count": count})

class DormantSpikeRule(BaseRule):
    name = "dormant_spike"
    def evaluate(self, tx, context):
        dormant_days = self.config.get("dormant_spike_days", 90)
        payer_last_active_map = context.setdefault("last_active_map", {})
        # tx may carry payer_last_active in field; fallback to context map
        last_active_iso = tx.get('last_active') or payer_last_active_map.get(tx['payer_id'])
        if last_active_iso:
            last_active = datetime.fromisoformat(last_active_iso)
            ts = datetime.fromisoformat(tx['timestamp'])
            if (ts - last_active).days >= dormant_days and tx['amount'] > self.config.get("dormant_trigger_amount", 5000):
                return RuleResult(self.name, self.config.get("weight",45), f"dormant { (ts-last_active).days } days then spike", {"days_dormant": (ts-last_active).days})
        return RuleResult(self.name, 0, "", {})

class TimeOfDayRule(BaseRule):
    name = "time_of_day"
    def evaluate(self, tx, context):
        # flag unusual time-of-day activity for retail accounts
        start = self.config.get("unusual_start_hour", 0)
        end = self.config.get("unusual_end_hour", 5)
        ts = datetime.fromisoformat(tx['timestamp'])
        if start <= ts.hour <= end and tx['amount'] >= self.config.get("time_of_day_min_amount", 1000):
            return RuleResult(self.name, self.config.get("weight",20), f"tx at unusual hour {ts.hour}", {"hour": ts.hour})
        return RuleResult(self.name, 0, "", {})

class RoundNumberRule(BaseRule):
    name = "round_number"
    def evaluate(self, tx, context):
        # detect round-number pattern e.g., multiple txs ending with '.00' or multiples of 10000 patterns
        amount = tx['amount']
        if abs(amount - round(amount)) < 1e-6:  # integer
            # if integer and divisible by 1000 or 10000
            if amount % self.config.get("round_multiple", 1000) == 0:
                return RuleResult(self.name, self.config.get("weight",15), f"round-number pattern multiple of {self.config.get('round_multiple')}", {"amount": amount})
        return RuleResult(self.name, 0, "", {"amount": amount})

class ReversalLoopRule(BaseRule):
    name = "reversal_loop"
    def evaluate(self, tx, context):
        # context keeps per-payer last N statuses
        rev_map = context.setdefault("rev_map", defaultdict(lambda: deque(maxlen=20)))
        payer = tx['payer_id']
        status = tx.get('status')
        rev_map[payer].append(status)
        rev_count = list(rev_map[payer]).count('reversed')
        if rev_count >= self.config.get("reversal_loop_count", 4):
            return RuleResult(self.name, self.config.get("weight",60), f"{rev_count} reversals in recent history", {"rev_count": rev_count})
        return RuleResult(self.name, 0, "", {"rev_count": rev_count})

# Map names to classes for registry
RULE_REGISTRY = {
    "high_value": HighValueRule,
    "non_kyc_large": NonKycLargeRule,
    "rtgs_small": RtgsSmallRule,
    "sanctions": SanctionsRule,
    "velocity": VelocityRule,
    "structuring": StructuringRule,
    "dormant_spike": DormantSpikeRule,
    "time_of_day": TimeOfDayRule,
    "round_number": RoundNumberRule,
    "reversal_loop": ReversalLoopRule
}

# --- Engine orchestrator ---
class STRRulesEngine:
    def __init__(self, config_path="rules.yml", sanctions_fn=None):
        self.config = load_rules_config(config_path)
        self.rules = []
        rules_cfg = self.config.get("rules", {})
        for rname, rconf in rules_cfg.items():
            cls = RULE_REGISTRY.get(rname)
            if not cls:
                continue
            self.rules.append(cls(rconf))
        # context stores history maps across transactions
        self.context = {"sanctions_fn": sanctions_fn}
        self.rule_config_version = self.config.get("version", "rules-unknown")

    def evaluate_tx(self, tx: Dict[str,Any]) -> Dict[str,Any]:
        per_rule = []
        total_score = 0
        for r in self.rules:
            res = r.evaluate(tx, self.context)
            if res.score > 0:
                per_rule.append({"name":res.name, "score":res.score, "reason":res.reason, "metadata":res.metadata})
            total_score += res.score
        # final mapping
        flag = "Low"
        if total_score >= self.config.get("str_threshold", 120):
            flag = "STR"
        elif total_score >= self.config.get("high_risk_threshold", 80):
            flag = "High-Risk"
        elif total_score >= self.config.get("medium_threshold", 40):
            flag = "Medium"
        else:
            flag = "Low"
        audit = {
            "tx_id": tx.get("tx_id"),
            "timestamp": tx.get("timestamp"),
            "payer_id": tx.get("payer_id"),
            "beneficiary_id": tx.get("beneficiary_id"),
            "amount": tx.get("amount"),
            "payment_system": tx.get("payment_system"),
            "status": tx.get("status"),
            "kyc_verified": tx.get("kyc_verified"),
            "score": int(total_score),
            "flag": flag,
            "rules_triggered": per_rule,
            "rule_config_version": self.rule_config_version,
            "engine_version": "str_engine_v1"
        }
        return audit

    def evaluate_batch(self, transactions: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        out = []
        for tx in transactions:
            out.append(self.evaluate_tx(tx))
        return out