"""
src/simple_transaction.py
Tiny starter: creates a few synthetic transactions and applies a simple rule.
Run: python .\src\simple_transaction.py
"""
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import json

@dataclass
class Customer:
    id: str
    kyc_verified: bool

@dataclass
class Transaction:
    tx_id: str
    timestamp: str
    sender: str
    receiver: str
    amount: float
    payment_system: str

def make_synthetic_customer(i):
    return Customer(id=f"C{i:04d}", kyc_verified=random.choice([True, True, False]))

def make_synthetic_tx(i, sender_id):
    amount = round(random.expovariate(1/2000) + random.choice([10,50,100]),2)
    return Transaction(
        tx_id=f"TX{i:06d}",
        timestamp=datetime.utcnow().isoformat(),
        sender=sender_id,
        receiver=f"BEN{random.randint(100,999)}",
        amount=amount,
        payment_system=random.choice(['UPI','NEFT','RTGS'])
    )

def rule_non_kyc(tx, sender):
    if not sender.kyc_verified and tx.amount > 5000:
        return True, "non-kyc large amount"
    return False, ""

if __name__ == '__main__':
    random.seed(0)
    customers = [make_synthetic_customer(i) for i in range(1,11)]
    results = []
    for i in range(1,21):
        c = random.choice(customers)
        tx = make_synthetic_tx(i, c.id)
        flag, reason = rule_non_kyc(tx, c)
        res = {"tx": asdict(tx), "sender": asdict(c), "flagged": flag, "reason": reason}
        results.append(res)
    print(json.dumps(results, indent=2))
