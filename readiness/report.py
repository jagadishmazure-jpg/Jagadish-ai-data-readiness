"""Human-readable markdown rendering of a readiness report."""
from __future__ import annotations

from .scorer import USE_CASES, ReadinessReport


def to_markdown(report: ReadinessReport) -> str:
    weights = USE_CASES[report.use_case]["weights"]
    lines = [
        f"# Data Readiness Report — {report.dataset_name}",
        "",
        f"**Use case:** `{report.use_case}`",
        f"**Overall readiness:** {report.overall}/100 — **Grade {report.grade}**",
        "",
        "## Dimension scores",
        "",
        "| Dimension | Score | Weight | Findings |",
        "|---|---|---|---|",
    ]
    for dim in report.dimensions:
        lines.append(
            f"| {dim.name} | {dim.score} | {weights[dim.name]:.0%} "
            f"| {len(dim.findings)} |"
        )
    lines += ["", "## Top findings", ""]
    any_findings = False
    for dim in report.dimensions:
        for finding in dim.findings:
            any_findings = True
            lines.append(f"- **{dim.name}**: {finding}")
    if not any_findings:
        lines.append("- None. The dataset is ready for modeling.")
    lines += ["", "## Remediation checklist (prioritized)", ""]
    if report.recommendations:
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
    else:
        lines.append("No remediation needed.")
    lines.append("")
    return "\n".join(lines)
