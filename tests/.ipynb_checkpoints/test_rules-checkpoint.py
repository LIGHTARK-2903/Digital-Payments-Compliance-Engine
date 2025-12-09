# tests/test_rules.py
import pytest
from src.str_engine import STRRulesEngine, RuleResult, HighValueRule, VelocityRule

def test_high_value_rule():
    cfg = {"high_value_threshold": 1000, "weight": 80}
    r = HighValueRule(cfg)
    tx = {"amount": 2000, "timestamp": "2025-12-10T00:00:00"}
    res = r.evaluate(tx, {})
    assert res.score == 80
    assert "amount" in res.metadata

def test_velocity_rule_simple():
    cfg = {"velocity_count_window_minutes": 10, "velocity_tx_count_threshold": 3, "weight": 40}
    r = VelocityRule(cfg)
    context = {}
    base_ts = "2025-12-10T00:00:00"
    tx1 = {"payer_id":"C1","timestamp":base_ts}
    r.evaluate(tx1, context)
    tx2 = {"payer_id":"C1","timestamp":"2025-12-10T00:02:00"}
    r.evaluate(tx2, context)
    tx3 = {"payer_id":"C1","timestamp":"2025-12-10T00:03:00"}
    res = r.evaluate(tx3, context)
    assert res.score == 40