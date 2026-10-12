"""AI data-readiness scorer: assess a dataset before you build.

Given a dataset and an intended AI use case, produces a scored readiness
report across six dimensions (completeness, label quality, freshness,
volume, distribution, governance) with findings and prioritized
remediation recommendations.
"""

from .dataset import (
    Dataset,
    make_customer_churn,
    make_meter_telemetry,
    make_support_docs,
)
from .dimensions import DimensionScore
from .report import to_markdown
from .scorer import USE_CASES, ReadinessReport, assess, grade_for

__all__ = [
    "Dataset",
    "DimensionScore",
    "ReadinessReport",
    "USE_CASES",
    "assess",
    "grade_for",
    "make_customer_churn",
    "make_meter_telemetry",
    "make_support_docs",
    "to_markdown",
]
