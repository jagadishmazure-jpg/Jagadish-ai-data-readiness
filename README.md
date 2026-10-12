# Jagadish-ai-data-readiness

**AI data-readiness scorer — assess a dataset before you build.**

The bottleneck in enterprise AI is rarely the model. It is the data: missing
values in key columns, stale extracts, tiny samples, skewed distributions,
unlabeled targets, and governance metadata nobody filled in. This project
scores a dataset's readiness for a specific AI use case across six
dimensions, explains every deduction with concrete findings, and produces a
prioritized remediation checklist. It is discovery-phase tooling: the
assessment you run *before* committing to model work.

## The six dimensions

| Dimension | What it measures |
|---|---|
| Completeness | Missing-value rate per column; key columns penalized more |
| Label quality | Label presence, null rate, class balance (supervised use cases) |
| Freshness | Max record age vs the cadence the use case requires |
| Volume | Row count vs minimum thresholds per model class |
| Distribution | Skew, constant columns, duplicate-row rate |
| Governance | Classification, lineage, residency, PII handling declarations |

Weights differ per use case — label quality carries zero weight for RAG,
governance carries extra weight for document corpora, freshness dominates
for forecasting. See `docs/methodology.md` for the full scoring spec.

## Quickstart

```python
from readiness import assess, make_customer_churn, to_markdown

dataset = make_customer_churn(quality="missing")  # or load your own Dataset
report = assess(dataset, "tabular_classification")
print(report.overall, report.grade)   # 74.4 C
print(to_markdown(report))
```

Supported use cases: `tabular_classification`, `forecasting`,
`rag_documents`, `anomaly_detection`. Bring your own data with the
`Dataset(name, rows, metadata)` dataclass — rows are plain dicts, no
pandas/numpy required (stdlib only).

## Sample report excerpt

```
# Data Readiness Report — customer_churn_missing
**Overall readiness:** 74.4/100 — **Grade C**

| Dimension | Score | Weight | Findings |
|---|---|---|---|
| completeness | 57.5 | 25% | 3 |
| label_quality | 100.0 | 25% | 0 |
| freshness | 100.0 | 10% | 0 |
| volume | 100.0 | 15% | 0 |
| distribution | 100 | 10% | 0 |
| governance | 0.0 | 15% | 4 |

## Remediation checklist (prioritized)
1. [governance] declare a classification in dataset metadata
2. [governance] record the upstream source and extract version
3. [governance] declare the residency zone (e.g. EU / US / APAC)
4. [governance] declare 'email' in pii_columns; confirm masking and consent
5. [completeness] backfill or impute 'total_charge' (36% missing)
...
```

## Project structure

```
readiness/
  dataset.py      Dataset model + synthetic generators (churn, telemetry, docs)
  dimensions.py   six deterministic scorers (stdlib only)
  scorer.py       use-case weighting, grading, remediation prioritization
  report.py       markdown rendering
tests/            37 deterministic tests, no network or API keys
docs/             scoring methodology and thresholds
```

## Design notes

- **Deterministic**: same input → same scores, always. No randomness in scoring.
- **Explainable**: every point deducted traces to a named finding.
- **Use-case aware**: a 60-day-old document corpus is fine for RAG and fatal
  for forecasting; the weights encode that.
- **Governance is a dimension, not an afterthought**: classification, lineage,
  residency, and PII declarations are scored like any data-quality signal.

## Limitations

Heuristic thresholds, not learned ones — see `docs/methodology.md`. The
synthetic generators are for demo and testing; point `Dataset` at real
extracts for production assessments.
