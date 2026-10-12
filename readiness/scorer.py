"""Assess a dataset for an intended AI use case.

Weights differ per use case: label quality, for example, carries no weight
for RAG, while governance carries extra weight for document corpora.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .dataset import Dataset
from .dimensions import (
    DimensionScore,
    score_completeness,
    score_distribution,
    score_freshness,
    score_governance,
    score_label_quality,
    score_volume,
)

DIMENSIONS = ("completeness", "label_quality", "freshness",
              "volume", "distribution", "governance")

USE_CASES: Dict[str, Dict] = {
    "tabular_classification": {
        "label_column": "churn",
        "recency_column": "days_ago",
        "min_rows": 1000,
        "required_cadence_days": 30,
        "key_columns": ("churn", "customer_id"),
        "weights": {"completeness": 0.25, "label_quality": 0.25,
                    "freshness": 0.10, "volume": 0.15,
                    "distribution": 0.10, "governance": 0.15},
    },
    "forecasting": {
        "label_column": "kwh",
        "recency_column": "days_ago",
        "min_rows": 500,
        "required_cadence_days": 7,
        "key_columns": ("kwh", "meter_id"),
        "weights": {"completeness": 0.25, "label_quality": 0.10,
                    "freshness": 0.25, "volume": 0.15,
                    "distribution": 0.10, "governance": 0.15},
    },
    "rag_documents": {
        "label_column": None,
        "recency_column": "days_ago",
        "min_rows": 100,
        "required_cadence_days": 90,
        "key_columns": ("body", "doc_id"),
        "weights": {"completeness": 0.20, "label_quality": 0.0,
                    "freshness": 0.15, "volume": 0.20,
                    "distribution": 0.10, "governance": 0.35},
    },
    "anomaly_detection": {
        "label_column": "churn",
        "recency_column": "days_ago",
        "min_rows": 2000,
        "required_cadence_days": 30,
        "key_columns": ("customer_id",),
        "weights": {"completeness": 0.20, "label_quality": 0.10,
                    "freshness": 0.15, "volume": 0.25,
                    "distribution": 0.20, "governance": 0.10},
    },
}


def grade_for(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


@dataclass
class ReadinessReport:
    dataset_name: str
    use_case: str
    overall: float  # 0-100
    dimensions: List[DimensionScore]
    grade: str
    recommendations: List[str] = field(default_factory=list)


def assess(dataset: Dataset, use_case: str) -> ReadinessReport:
    """Score dataset readiness for use_case; deterministic."""
    if use_case not in USE_CASES:
        raise ValueError(
            f"unknown use case '{use_case}'; "
            f"choose from {sorted(USE_CASES)}")
    cfg = USE_CASES[use_case]
    weights = cfg["weights"]

    dims = [
        score_completeness(dataset, key_columns=cfg["key_columns"]),
        score_label_quality(dataset, cfg["label_column"]),
        score_freshness(dataset, cfg["recency_column"],
                        cfg["required_cadence_days"]),
        score_volume(dataset, cfg["min_rows"], context=use_case),
        score_distribution(dataset),
        score_governance(dataset),
    ]
    overall = round(
        sum(weights[d.name] * d.score for d in dims), 1)

    # Prioritize remediation by expected impact: weight * room to improve.
    prioritized = sorted(
        ((weights[d.name] * (100 - d.score), d) for d in dims
         if d.findings and weights[d.name] > 0),
        key=lambda t: t[0],
        reverse=True,
    )
    recommendations: List[str] = []
    for _, dim in prioritized:
        for rec in dim.recommendations:
            if rec not in recommendations:
                recommendations.append(f"[{dim.name}] {rec}")
    recommendations = recommendations[:10]

    return ReadinessReport(
        dataset_name=dataset.name,
        use_case=use_case,
        overall=overall,
        dimensions=dims,
        grade=grade_for(overall),
        recommendations=recommendations,
    )
