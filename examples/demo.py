"""Score the same synthetic datasets for different AI use cases (offline, deterministic)."""
from readiness import Dataset, assess, make_customer_churn, make_meter_telemetry, make_support_docs

cases = [
    (make_customer_churn(quality="good"), "tabular_classification"),
    (make_customer_churn(quality="missing"), "tabular_classification"),
    (make_customer_churn(quality="unlabeled"), "tabular_classification"),
    (make_meter_telemetry(quality="stale"), "forecasting"),
    (make_support_docs(quality="good"), "rag_documents"),
    (make_support_docs(quality="tiny"), "rag_documents"),
]
print(f"{'dataset':28} {'use case':24} {'score':>6} grade")
for ds, uc in cases:
    r = assess(ds, uc)
    print(f"{ds.name:28} {uc:24} {r.overall:6.1f} {r.grade}")

# Same rows, but nobody filled in the governance metadata.
missing = make_customer_churn(quality="missing")
ungoverned = Dataset(name="customer_churn_ungoverned", rows=missing.rows, metadata={})
r = assess(ungoverned, "tabular_classification")
print(f"\n{r.dataset_name}: {r.overall:.1f} {r.grade}  (same rows, governance metadata removed)")
for d in r.dimensions:
    print(f"  {d.name:14} {d.score:6.1f}  findings={len(d.findings)}")
print("Top remediation steps:")
for step in r.recommendations[:5]:
    print("  -", step)
