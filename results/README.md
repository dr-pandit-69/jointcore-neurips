# Frozen reported aggregates

`paper_aggregates.json` contains the exact-stage, model-stage, robustness, and
paired query values displayed in the submitted manuscript. It is a compact
machine-readable snapshot, not a substitute for raw model records.

`exact_metrics.json` is regenerated directly from the released deterministic
benchmark code and includes results by interaction family and candidate count.

The Llama-3.3-70B values are exploratory. Minimum-cost and logical-sufficiency
comparisons in the model stage are descriptive. Query intervals use 2,000
task-bootstrap replicates, and the displayed adjusted p-values use the
registered twelve-test family.
