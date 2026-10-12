"""Six deterministic data-readiness dimension scorers (stdlib only).

Each scorer returns a DimensionScore(name, score 0-100, findings,
recommendations, details). Scores are deterministic for a given dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .dataset import Dataset


@dataclass
class DimensionScore:
    name: str
    score: float  # 0-100
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# small stats helpers (no numpy/pandas)
# ---------------------------------------------------------------------------

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# ---------------------------------------------------------------------------
# 1. completeness
# ---------------------------------------------------------------------------

def score_completeness(
    dataset: Dataset, key_columns: tuple = ()
) -> DimensionScore:
    """Missing-value rate per column. Key columns deduct more per point."""
    if not dataset.rows:
        return DimensionScore(
            "completeness", 0.0,
            ["dataset has no rows"],
            ["load data before assessing readiness"],
            {},
        )
    findings: List[str] = []
    recommendations: List[str] = []
    per_col: Dict[str, float] = {}
    deduction = 0.0
    for col in dataset.columns:
        vals = dataset.values(col)
        missing = sum(1 for v in vals if v is None or v == "")
        rate = missing / len(vals)
        per_col[col] = round(rate * 100, 1)
        cap = 40.0 if col in key_columns else 25.0
        deduction += min(cap, 100 * rate)
        if rate > 0.05:
            findings.append(f"column '{col}': {rate:.0%} missing values")
            recommendations.append(
                f"backfill or impute '{col}' ({rate:.0%} missing); "
                "drop the column if it is not predictive"
            )
    score = round(max(0.0, 100 - deduction), 1)
    return DimensionScore(
        "completeness", score, findings, recommendations,
        {"missing_pct_by_column": per_col},
    )


# ---------------------------------------------------------------------------
# 2. label quality
# ---------------------------------------------------------------------------

_BINARY_TOKENS = {0, 1, "0", "1", True, False,
                  "yes", "no", "Yes", "No", "YES", "NO"}


def score_label_quality(
    dataset: Dataset, label_column: Optional[str]
) -> DimensionScore:
    """Label presence, null rate and class balance (for supervised use cases)."""
    if label_column is None:
        return DimensionScore(
            "label_quality", 100.0,
            ["label quality is not applicable to retrieval use cases"],
            [],
            {"applicable": False},
        )
    if label_column not in dataset.columns or not dataset.rows:
        return DimensionScore(
            "label_quality", 40.0,
            [f"label column '{label_column}' is absent: supervised training "
             "and evaluation are blocked"],
            [f"add a '{label_column}' label column, or switch to an "
             "unsupervised use case"],
            {"applicable": True, "present": False},
        )
    vals = dataset.non_null(label_column)
    null_rate = 1 - len(vals) / len(dataset.rows)
    findings: List[str] = []
    recommendations: List[str] = []
    score = 100.0
    if null_rate > 0.05:
        findings.append(f"label '{label_column}': {null_rate:.0%} null labels")
        recommendations.append(
            f"resolve null labels for '{label_column}' ({null_rate:.0%}); "
            "unlabeled rows cannot be used for supervised training"
        )
        score -= min(30.0, 100 * null_rate)
    uniq = set(vals)
    details: Dict[str, Any] = {"applicable": True, "present": True,
                               "null_pct": round(null_rate * 100, 1)}
    if uniq <= _BINARY_TOKENS and vals:
        pos = sum(1 for v in vals if v in (1, True, "1", "yes", "Yes", "YES"))
        minority = min(pos, len(vals) - pos) / len(vals)
        details["minority_class_pct"] = round(minority * 100, 1)
        if minority < 0.10:
            findings.append(
                f"class imbalance: minority class is {minority:.1%} of labels")
            recommendations.append(
                "collect more minority-class examples, or plan class "
                "weighting / resampling before training"
            )
            score -= 20.0
    elif len(uniq) <= 1:
        findings.append(f"label column '{label_column}' is constant: "
                        "no signal to learn")
        recommendations.append(
            f"investigate why '{label_column}' never varies; a constant "
            "target cannot train a model")
        score -= 40.0
    return DimensionScore("label_quality", round(max(0.0, score), 1),
                          findings, recommendations, details)


# ---------------------------------------------------------------------------
# 3. freshness
# ---------------------------------------------------------------------------

def score_freshness(
    dataset: Dataset, recency_column: str, required_cadence_days: int
) -> DimensionScore:
    """Max record age vs the cadence the use case requires."""
    vals = [v for v in dataset.non_null(recency_column) if _is_number(v)] \
        if dataset.rows and recency_column in dataset.columns else []
    if not vals:
        return DimensionScore(
            "freshness", 30.0,
            [f"no usable recency signal in '{recency_column}': "
             "freshness cannot be verified"],
            [f"add a '{recency_column}' (days since record update) column, "
             "or record max record age in dataset metadata"],
            {"max_age_days": None},
        )
    max_age = int(max(vals))
    if max_age <= required_cadence_days:
        return DimensionScore(
            "freshness", 100.0, [], {},
            {"max_age_days": max_age,
             "required_cadence_days": required_cadence_days},
        )
    overshoot = (max_age - required_cadence_days) / (required_cadence_days * 4)
    score = round(max(0.0, 100 - 100 * overshoot), 1)
    return DimensionScore(
        "freshness", score,
        [f"newest record is {max_age} days old; required cadence is "
         f"at most {required_cadence_days} days"],
        ["refresh the extract on the required cadence before training; "
         "stale data trains stale models"],
        {"max_age_days": max_age,
         "required_cadence_days": required_cadence_days},
    )


# ---------------------------------------------------------------------------
# 4. volume
# ---------------------------------------------------------------------------

def score_volume(
    dataset: Dataset, min_rows: int, context: str = ""
) -> DimensionScore:
    """Row count vs the minimum for the model class."""
    n = len(dataset.rows)
    if n >= min_rows:
        return DimensionScore("volume", 100.0, [], {},
                              {"rows": n, "min_rows": min_rows})
    score = round(100 * n / min_rows, 1)
    suffix = f" for {context}" if context else ""
    return DimensionScore(
        "volume", score,
        [f"only {n} rows; at least {min_rows} recommended{suffix}"],
        ["collect more rows, or choose a simpler model class that learns "
         "from less data"],
        {"rows": n, "min_rows": min_rows},
    )


# ---------------------------------------------------------------------------
# 5. distribution
# ---------------------------------------------------------------------------

def score_distribution(dataset: Dataset) -> DimensionScore:
    """Skew, constant columns and duplicate rows."""
    findings: List[str] = []
    recommendations: List[str] = []
    details: Dict[str, Any] = {}
    for col in dataset.columns:
        vals = [float(v) for v in dataset.non_null(col) if _is_number(v)]
        if len(vals) < 10:
            continue
        uniq = set(vals)
        if len(uniq) == 1:
            findings.append(f"column '{col}' is constant: carries no signal")
            recommendations.append(
                f"drop constant column '{col}' or investigate the upstream feed")
            details[col] = {"constant": True}
            continue
        med = _median(vals)
        if med != 0:
            ratio = _mean(vals) / med
            details[col] = {"mean": round(_mean(vals), 2),
                            "median": round(med, 2),
                            "mean_median_ratio": round(ratio, 2)}
            if ratio > 2 or ratio < 0.5:
                findings.append(
                    f"column '{col}' is heavily skewed "
                    f"(mean/median = {ratio:.1f})")
                recommendations.append(
                    f"apply a log or winsorized transform to '{col}', or use "
                    "a model robust to skew")
    if dataset.rows:
        keys = [tuple(sorted((k, repr(v)) for k, v in row.items()))
                for row in dataset.rows]
        dup_rate = 1 - len(set(keys)) / len(keys)
        details["duplicate_row_pct"] = round(dup_rate * 100, 1)
        if dup_rate > 0.05:
            findings.append(f"{dup_rate:.0%} duplicate rows detected")
            recommendations.append(
                "deduplicate on the natural key before training; duplicates "
                "leak between train and test splits")
    score = round(max(0.0, 100 - 25 * len(findings)), 1)
    return DimensionScore("distribution", score, findings,
                          recommendations, details)


# ---------------------------------------------------------------------------
# 6. governance
# ---------------------------------------------------------------------------

_PII_LIKE = {"email", "ssn", "phone", "address", "dob", "date_of_birth",
             "full_name"}


def score_governance(dataset: Dataset) -> DimensionScore:
    """Classification, lineage, residency and PII handling declarations."""
    findings: List[str] = []
    recommendations: List[str] = []
    for key, msg, rec in [
        ("classification",
         "no data classification declared",
         "declare a classification (public / internal / confidential / "
         "restricted) in dataset metadata"),
        ("lineage",
         "no lineage recorded",
         "record the upstream source and extract version in dataset metadata"),
        ("residency",
         "no residency zone declared",
         "declare the residency zone (e.g. EU / US / APAC) so residency "
         "policies can be enforced"),
    ]:
        if not dataset.metadata.get(key):
            findings.append(msg)
            recommendations.append(rec)
    declared_pii = set(dataset.metadata.get("pii_columns", []))
    for col in dataset.columns:
        if col.lower() in _PII_LIKE and col not in declared_pii:
            findings.append(
                f"PII-like column '{col}' has no declared handling")
            recommendations.append(
                f"declare '{col}' in pii_columns and confirm masking, "
                "consent and access controls")
    score = round(max(0.0, 100 - 25 * len(findings)), 1)
    return DimensionScore("governance", score, findings, recommendations,
                          {"metadata_keys": sorted(dataset.metadata.keys())})
