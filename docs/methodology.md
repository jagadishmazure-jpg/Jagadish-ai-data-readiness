# Scoring methodology

## Dimensions

### 1. Completeness (0–100)
Missing-value rate per column (`None` or `""` counts as missing). Each
column deducts up to 25 points proportional to its missing rate; key
columns (e.g. the label, the natural key) deduct up to 40. Findings are
raised for any column above 5% missing.

### 2. Label quality (0–100)
- Not applicable (`None` label column, e.g. RAG): 100, informational finding.
- Label column absent: 40 — supervised training is blocked.
- Null label rate above 5%: proportional deduction up to 30.
- Binary labels with minority class below 10%: −20.
- Constant label column: −40.

### 3. Freshness (0–100)
Compares max record age (`days_ago`) against the required cadence. At or
under cadence: 100. Degrades linearly to 0 at 5× the cadence. Missing
recency signal: 30 with a finding.

### 4. Volume (0–100)
`100 × rows / min_rows`, capped at 100. Thresholds are heuristics per
model class (see below), not learned values.

### 5. Distribution (0–100)
Starts at 100, −25 per finding:
- Heavily skewed numeric column: |mean/median| > 2 or < 0.5 (n ≥ 10).
- Constant column: no signal.
- Duplicate-row rate above 5%: train/test leakage risk.

### 6. Governance (0–100)
Starts at 100, −25 per finding: missing classification, lineage, or
residency declaration; PII-like columns (`email`, `ssn`, `phone`,
`address`, `dob`, …) without declared handling.

## Use-case configuration

| Use case | Label | Min rows | Cadence (days) |
|---|---|---|---|
| tabular_classification | churn | 1,000 | 30 |
| forecasting | kwh | 500 | 7 |
| rag_documents | — | 100 | 90 |
| anomaly_detection | churn | 2,000 | 30 |

### Weights

| Dimension | tabular_classification | forecasting | rag_documents | anomaly_detection |
|---|---|---|---|---|
| completeness | 25% | 25% | 20% | 20% |
| label_quality | 25% | 10% | 0% | 10% |
| freshness | 10% | 25% | 15% | 15% |
| volume | 15% | 15% | 20% | 25% |
| distribution | 10% | 10% | 10% | 20% |
| governance | 15% | 15% | 35% | 10% |

Rationale: forecasting lives or dies on recency; RAG needs governed,
fresh-enough documents and no labels; anomaly detection needs volume and
well-behaved distributions; governance dominates for document corpora
that may contain PII.

## Overall score and grades

`overall = Σ weight × dimension_score`, rounded to one decimal.
Grades: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F < 60.

Remediation recommendations are prioritized by expected impact:
`weight × (100 − dimension_score)`, highest first, deduplicated, capped
at 10 items.

## Limitations

- Thresholds (5% missing, 10% minority class, 2× skew ratio, row minima)
  are industry heuristics, not statistically derived. Calibrate them to
  your domain before using scores contractually.
- Text-heavy RAG corpora are assessed structurally (empty bodies,
  duplicates, freshness, governance), not semantically.
- The scorer is deterministic and offline by design; it does not profile
  data already inside a warehouse — point it at extracts.
