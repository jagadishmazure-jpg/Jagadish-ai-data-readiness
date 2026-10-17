"""Dataset model and deterministic synthetic data generators (stdlib only)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Dataset:
    """A tabular dataset plus governance metadata.

    metadata may carry: classification, lineage, residency, pii_columns,
    owner. Dimensions read these keys; absence is itself a finding.
    """

    name: str
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def columns(self) -> List[str]:
        cols: List[str] = []
        for row in self.rows:
            for key in row:
                if key not in cols:
                    cols.append(key)
        return cols

    def values(self, column: str) -> List[Any]:
        return [row.get(column) for row in self.rows]

    def non_null(self, column: str) -> List[Any]:
        return [v for v in self.values(column) if v is not None and v != ""]


def _governed_metadata(residency: str = "EU") -> Dict[str, Any]:
    return {
        "classification": "confidential",
        "lineage": "crm.daily_extract v3.2",
        "residency": residency,
        "pii_columns": ["email"],
        "owner": "data-platform",
    }


# ---------------------------------------------------------------------------
# Customer churn (tabular classification / anomaly detection)
# ---------------------------------------------------------------------------

def make_customer_churn(
    n: int = 2000,
    quality: str = "good",
    seed: int = 42,
) -> Dataset:
    """Synthetic customer-churn table.

    quality: "good" | "missing" | "stale" | "skewed" | "tiny" | "unlabeled".
    Deterministic for a given seed.
    """
    if quality == "tiny":
        n = min(n, 60)
    rng = random.Random(seed)
    contracts = ["month-to-month", "one-year", "two-year"]
    payments = ["credit-card", "bank-transfer", "e-check", "mailed-check"]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        tenure = rng.randint(1, 72)
        contract = rng.choice(contracts)
        monthly = round(rng.uniform(20, 120), 2)
        if quality == "skewed" and rng.random() < 0.10:
            monthly = round(rng.uniform(500, 2000), 2)
        total = round(monthly * tenure * rng.uniform(0.9, 1.0), 2)
        churn_prob = 0.27 + (0.15 if contract == "month-to-month" else -0.08)
        churn = 1 if rng.random() < churn_prob else 0
        days_ago = rng.randint(0, 30)
        if quality == "stale":
            days_ago = rng.randint(150, 210)
        row = {
            "customer_id": f"CUST-{100000 + i}",
            "tenure_months": tenure,
            "monthly_charge": monthly,
            "total_charge": total,
            "contract_type": contract,
            "payment_method": rng.choice(payments),
            "email": f"user{i}@example.com",
            "days_ago": days_ago,
            "churn": churn,
        }
        if quality == "missing":
            if rng.random() < 0.35:
                row["total_charge"] = None
            if rng.random() < 0.10:
                row["tenure_months"] = None
            if rng.random() < 0.08:
                row["contract_type"] = None
        rows.append(row)
    if quality == "unlabeled":
        for row in rows:
            row.pop("churn", None)
    if quality == "skewed":
        dups = rng.sample(rows, max(1, int(0.08 * len(rows))))
        rows.extend(dict(r) for r in dups)
    return Dataset(
        name=f"customer_churn_{quality}",
        rows=rows,
        metadata=_governed_metadata(),
    )


# ---------------------------------------------------------------------------
# Meter telemetry (forecasting)
# ---------------------------------------------------------------------------

def make_meter_telemetry(
    n: int = 2000,
    quality: str = "good",
    seed: int = 42,
) -> Dataset:
    """Synthetic smart-meter telemetry: meter_id, days_ago, kwh, temperature_c."""
    if quality == "tiny":
        n = min(n, 80)
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for _ in range(n):
        # "good" telemetry stays within the forecasting cadence (7 days)
        days_ago = rng.randint(0, 6)
        if quality == "stale":
            days_ago = rng.randint(100, 160)
        temp = round(rng.uniform(-5, 35), 1)
        kwh = round(max(0.0, rng.gauss(30 + temp * 0.8, 8)), 2)
        if quality == "skewed" and rng.random() < 0.10:
            kwh = round(kwh * 10, 2)
        row = {
            "meter_id": f"MTR-{rng.randint(1, 50):04d}",
            "days_ago": days_ago,
            "kwh": kwh,
            "temperature_c": temp,
        }
        if quality == "missing" and rng.random() < 0.25:
            row["kwh"] = None
        rows.append(row)
    if quality == "unlabeled":
        for row in rows:
            row.pop("kwh", None)
    return Dataset(
        name=f"meter_telemetry_{quality}",
        rows=rows,
        metadata=_governed_metadata(),
    )


# ---------------------------------------------------------------------------
# Support documents (RAG)
# ---------------------------------------------------------------------------

_DOC_TITLES = [
    "Password reset procedure",
    "Refund policy overview",
    "API rate limits",
    "Data retention schedule",
    "Onboarding checklist",
    "Incident escalation path",
    "Billing dispute process",
    "Feature request workflow",
]

_DOC_SENTENCES = [
    "Follow the steps in order and confirm each one before continuing.",
    "Contact the on-call engineer if the issue persists after thirty minutes.",
    "All changes are logged with a timestamp and the operator identity.",
    "The default retention window is ninety days unless overridden by policy.",
    "Escalation requires a severity rating and a customer impact statement.",
    "Rate limits reset at the top of each hour in the account timezone.",
]


def make_support_docs(
    n: int = 300,
    quality: str = "good",
    seed: int = 42,
) -> Dataset:
    """Synthetic support-document corpus: doc_id, title, body, source, days_ago."""
    if quality == "tiny":
        n = min(n, 20)
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        days_ago = rng.randint(0, 60)
        if quality == "stale":
            days_ago = rng.randint(200, 400)
        body = " ".join(rng.sample(_DOC_SENTENCES, 3))
        if quality == "missing" and rng.random() < 0.30:
            body = ""
        rows.append(
            {
                "doc_id": f"DOC-{1000 + i}",
                "title": rng.choice(_DOC_TITLES),
                "body": body,
                "source": "helpdesk",
                "days_ago": days_ago,
            }
        )
    if quality == "unlabeled":
        for row in rows:
            row.pop("title", None)
    if quality == "skewed":
        dups = rng.sample(rows, max(1, int(0.15 * len(rows))))
        rows.extend(dict(r) for r in dups)
    return Dataset(
        name=f"support_docs_{quality}",
        rows=rows,
        metadata=_governed_metadata(),
    )
