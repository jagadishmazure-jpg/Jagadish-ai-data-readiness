"""Tests for the AI data-readiness scorer. All deterministic, no network."""
import pytest

from readiness import (
    Dataset,
    assess,
    grade_for,
    make_customer_churn,
    make_meter_telemetry,
    make_support_docs,
    to_markdown,
)
from readiness.dimensions import (
    score_completeness,
    score_distribution,
    score_freshness,
    score_governance,
    score_label_quality,
    score_volume,
)


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------

def test_completeness_good():
    ds = make_customer_churn(quality="good")
    s = score_completeness(ds, key_columns=("churn", "customer_id"))
    assert s.score >= 95
    assert s.findings == []


def test_completeness_missing_names_column():
    ds = make_customer_churn(quality="missing")
    s = score_completeness(ds, key_columns=("churn", "customer_id"))
    assert s.score < 70
    assert any("total_charge" in f for f in s.findings)
    assert any("total_charge" in r for r in s.recommendations)


# ---------------------------------------------------------------------------
# label quality
# ---------------------------------------------------------------------------

def test_label_quality_good():
    ds = make_customer_churn(quality="good")
    s = score_label_quality(ds, "churn")
    assert s.score >= 95
    assert s.details["minority_class_pct"] > 10


def test_label_quality_unlabeled():
    ds = make_customer_churn(quality="unlabeled")
    s = score_label_quality(ds, "churn")
    assert s.score < 60
    assert any("churn" in f for f in s.findings)


def test_label_quality_imbalance():
    rows = [{"churn": 1 if i < 5 else 0} for i in range(100)]
    ds = Dataset(name="imbalanced", rows=rows)
    s = score_label_quality(ds, "churn")
    assert any("imbalance" in f for f in s.findings)
    assert s.score < 90


def test_label_quality_not_applicable_for_rag():
    ds = make_support_docs(quality="good")
    s = score_label_quality(ds, None)
    assert s.score == 100.0
    assert s.details["applicable"] is False


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------

def test_freshness_good():
    ds = make_customer_churn(quality="good")
    s = score_freshness(ds, "days_ago", 30)
    assert s.score == 100.0


def test_freshness_stale():
    ds = make_customer_churn(quality="stale")
    s = score_freshness(ds, "days_ago", 30)
    assert s.score < 50
    assert any("days old" in f for f in s.findings)


def test_freshness_missing_recency_column():
    ds = Dataset(name="norec", rows=[{"a": 1}])
    s = score_freshness(ds, "days_ago", 30)
    assert s.score < 50
    assert s.findings


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------

def test_volume_good():
    ds = make_customer_churn(n=2000, quality="good")
    s = score_volume(ds, 1000, context="tabular_classification")
    assert s.score == 100.0


def test_volume_tiny():
    ds = make_customer_churn(quality="tiny")
    s = score_volume(ds, 1000, context="tabular_classification")
    assert s.score < 20
    assert any("60 rows" in f or "rows" in f for f in s.findings)


# ---------------------------------------------------------------------------
# distribution
# ---------------------------------------------------------------------------

def test_distribution_good():
    ds = make_customer_churn(quality="good")
    s = score_distribution(ds)
    assert s.score >= 75


def test_distribution_skewed_and_duplicates():
    ds = make_customer_churn(quality="skewed")
    s = score_distribution(ds)
    assert any("skewed" in f for f in s.findings)
    assert any("duplicate" in f for f in s.findings)
    assert any("monthly_charge" in f for f in s.findings)


def test_distribution_constant_column():
    ds = Dataset(name="const", rows=[{"x": 5, "y": i} for i in range(20)])
    s = score_distribution(ds)
    assert any("constant" in f for f in s.findings)


# ---------------------------------------------------------------------------
# governance
# ---------------------------------------------------------------------------

def test_governance_good():
    ds = make_customer_churn(quality="good")
    s = score_governance(ds)
    assert s.score == 100.0
    assert s.findings == []


def test_governance_missing_metadata():
    ds = make_customer_churn(quality="good")
    ds.metadata = {}
    s = score_governance(ds)
    assert len(s.findings) >= 3  # classification, lineage, residency
    assert s.score <= 25


def test_governance_undeclared_pii():
    ds = make_customer_churn(quality="good")
    ds.metadata = {k: v for k, v in ds.metadata.items()
                   if k != "pii_columns"}
    s = score_governance(ds)
    assert any("email" in f for f in s.findings)


# ---------------------------------------------------------------------------
# scorer: weighting, grades, recommendations
# ---------------------------------------------------------------------------

def test_rag_ignores_label_weight():
    cfg_weights = {"label_quality": 0.0}
    from readiness.scorer import USE_CASES
    assert USE_CASES["rag_documents"]["weights"]["label_quality"] == 0.0
    ds = make_support_docs(quality="good")
    report = assess(ds, "rag_documents")
    manual = sum(USE_CASES["rag_documents"]["weights"][d.name] * d.score
                 for d in report.dimensions)
    assert report.overall == round(manual, 1)


def test_overall_is_weighted_sum():
    from readiness.scorer import USE_CASES
    ds = make_customer_churn(quality="missing")
    report = assess(ds, "tabular_classification")
    w = USE_CASES["tabular_classification"]["weights"]
    manual = sum(w[d.name] * d.score for d in report.dimensions)
    assert report.overall == round(manual, 1)


@pytest.mark.parametrize("score,expected", [
    (95, "A"), (90, "A"), (85, "B"), (80, "B"),
    (75, "C"), (70, "C"), (65, "D"), (60, "D"), (40, "F"),
])
def test_grade_boundaries(score, expected):
    assert grade_for(score) == expected


def test_recommendations_reference_findings():
    ds = make_customer_churn(quality="missing")
    report = assess(ds, "tabular_classification")
    assert any("total_charge" in r for r in report.recommendations)


def test_recommendations_prioritized():
    ds = make_customer_churn(quality="missing")
    report = assess(ds, "tabular_classification")
    # completeness is the weakest weighted dimension here
    assert report.recommendations
    assert report.recommendations[0].startswith("[completeness]")


def test_unknown_use_case_raises():
    ds = make_customer_churn(quality="good")
    with pytest.raises(ValueError):
        assess(ds, "time_travel")


def test_determinism():
    ds = make_customer_churn(quality="missing", seed=7)
    r1 = assess(ds, "tabular_classification")
    r2 = assess(ds, "tabular_classification")
    assert r1.overall == r2.overall
    assert [d.score for d in r1.dimensions] == [d.score for d in r2.dimensions]
    assert r1.recommendations == r2.recommendations


def test_empty_dataset_no_crash():
    ds = Dataset(name="empty", rows=[])
    report = assess(ds, "tabular_classification")
    assert report.grade == "F"
    assert report.overall < 60


# ---------------------------------------------------------------------------
# end-to-end per use case
# ---------------------------------------------------------------------------

def test_forecasting_telemetry_good():
    ds = make_meter_telemetry(quality="good")
    report = assess(ds, "forecasting")
    assert report.overall >= 80
    assert report.grade in ("A", "B")


def test_rag_docs_assess_runs():
    ds = make_support_docs(quality="good")
    report = assess(ds, "rag_documents")
    label_dim = next(d for d in report.dimensions
                     if d.name == "label_quality")
    assert label_dim.details["applicable"] is False
    assert len(report.dimensions) == 6


def test_tiny_dataset_gets_failing_grade():
    ds = make_customer_churn(quality="tiny")
    ds.metadata = {}  # tiny *and* ungoverned: clearly not production-ready
    report = assess(ds, "anomaly_detection")  # min_rows=2000
    assert report.grade in ("D", "F")
    assert any("volume" in r for r in report.recommendations)


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------

def test_markdown_contains_key_sections():
    ds = make_customer_churn(quality="missing")
    report = assess(ds, "tabular_classification")
    md = to_markdown(report)
    assert "# Data Readiness Report" in md
    assert "## Dimension scores" in md
    assert "## Top findings" in md
    assert "## Remediation checklist" in md
    assert f"Grade {report.grade}" in md
    assert "total_charge" in md
