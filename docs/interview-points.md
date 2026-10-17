# Interview points

Short, honest talking points for Jagadish-ai-data-readiness. Every number quoted is from the
real run shown in the README.

- The bottleneck in enterprise AI is usually the data; I made the discovery step a measurable, repeatable score.
- Same rows, different metadata: 89.4 (B) with governance declared, 74.4 (C) without. Governance is a first-class dimension.
- Use-case weighting matters: a month-old document set is fine for RAG and a problem for forecasting.
- Deterministic and explainable by design, so a data owner can dispute a specific finding rather than a black-box number.

## Likely follow-up questions

- How would you calibrate the thresholds? (Past datasets with known outcomes; adjust weights to maximise agreement.)
- How does this scale to a 500-million-row table? (Stratified sampling, plus pushing completeness counts down to SQL.)
- Where does governance metadata come from in practice? (Purview or the catalogue, not hand-typed dicts.)
