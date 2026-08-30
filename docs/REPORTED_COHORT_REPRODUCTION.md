# Reported cohort reproduction package

Status: post-study reproduction packaging for the model cohort displayed in the
submitted manuscript. This file does not alter or relabel the historical
development, confirmation, or exploratory-extension records.

The package combines the unchanged 432-task grammar and selection procedure
used by the reported Qwen2.5-14B, Gemma-4-12B-it,
Mistral-Small-3.1-24B-Instruct, and Llama-3.3-70B-Instruct evaluations. The
Llama result remains exploratory. The manuscript's 12B display cutoff remains
post hoc and does not support a model-scale claim.

Use:

- `configs/experiments/reported_cohort.json` for the shared task grid, search
  procedure, interfaces, and analysis family;
- `configs/models/reported_models.json` for exact checkpoint revisions and
  loading modes; and
- `results/` for frozen aggregate values reported in the paper.

The original compact artifact did not record a Qwen revision. This reproduction
package pins a checkpoint revision that predates the reported run so the command
resolves a fixed weight and tokenizer snapshot. The reported Qwen aggregate is
unchanged.

Raw model outputs are not republished in this compact artifact. Reproduction
generates them below `outputs/reported_cohort_reproduction_v1/`, together with
task checksums, runtime metadata, and aggregate manifests.
