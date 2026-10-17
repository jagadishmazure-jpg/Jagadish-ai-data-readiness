# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| `Dataset.rows` | Microsoft Fabric Lakehouse / ADLS Gen2 (sampled read) | Read-only managed identity |
| `Dataset.metadata` | Microsoft Purview (classification, lineage, sensitivity labels) | Purview becomes the source of truth for governance |
| CLI / library | Fabric notebook or Azure Container Apps job | Run on demand per dataset |
| Markdown report | Report stored in a Lakehouse Files folder or SharePoint | One report per assessment |

## Steps

1. Add a `from_lakehouse(table, sample_rows)` reader that returns a `Dataset`, using a read-only managed identity.
2. Map Purview classifications, lineage and sensitivity labels into `Dataset.metadata`.
3. Calibrate thresholds on a handful of past datasets whose AI outcome is known.
4. Run the scorer as a Fabric notebook or Container Apps job and store each report with its inputs for audit.
5. Gate model work on a minimum grade agreed with the data owner.

## Guardrails to keep when going live

- Deterministic scoring: no randomness, no model calls, so a score can be reproduced in an audit.
- Governance is scored like any quality signal; missing classification, lineage, residency or PII declarations cost points.
- The tool reads data and metadata only; it never modifies or exports the dataset.
- Synthetic generators are clearly named and used only for demos and tests.

## What to measure before calling it production-ready

- Agreement between the grade and the eventual project outcome on past datasets.
- Time taken to assess a table of realistic size.
- How often governance findings are confirmed by the data owner.
