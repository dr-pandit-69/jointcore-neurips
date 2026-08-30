# JointCore paper-expansion protocol

Status: development protocol, created after the negative v2--v5 study and the
post-hoc revival pilots.  Nothing in this track replaces or relabels those
earlier artifacts.

## Claim boundary

The paper-expansion study is a benchmark and limits paper.  It asks when
interaction-aware deletion recovers a behaviorally sufficient, low-cost
memory on controlled workflow tasks.  It does not estimate the prevalence of
these interactions in deployed agents and it does not claim that a selected
memory contains every fact a human would regard as important.

## Development and freeze

One development pass uses 72 tasks on development checkpoints.  The only
permitted method change is the deterministic ordering used by the cost-ordered
JointCore method.  After recording that choice, all task seeds, prompts,
methods, gates, and analysis rules are frozen before confirmation.

## Exact benchmark

The exact benchmark contains 6,912 task conditions: nine outcome families,
six workflow domains, four candidate counts (8, 16, 32, 64), two support
sizes (2, 4), and four horizon labels (1, 4, 8, 16).  The task seed is a
deterministic function of the full condition, yielding four balanced seed
blocks without multiplying the stated denominator.  Methods are full memory,
random-B, semantic top-B, independent leave-one-out, cost-aware conditional
singleton pruning, legacy JointCore, cost-ordered JointCore, and the exact
oracle.  B is four.

## Model confirmation

Confirmation contains 432 tasks per checkpoint: nine families by six domains
by two support sizes by four seed blocks.  Candidate count is 16 and the
horizon label cycles through 1, 4, 8, and 16.  Target action orientation and
record order are deterministically balanced.  The primary interface scores
the direct action labels EXECUTE and HOLD by mean conditional log probability.
Every final selected memory is also evaluated at a terminal COMMIT/HALT step.
The input bound is 768 tokens.

The camera-ready checkpoint cohort is Gemma-4-12B-it BF16 and
Mistral-Small-3.1-24B-Instruct-2503 NF4 with double quantization and BF16
compute.  The separately frozen Llama-3.3-70B extension uses runtime NF4.
Quantized evidence is reported as such and is not described as BF16 checkpoint
evidence.

## Robustness studies

A fixed 108-task subset receives typed-mask, deletion, plausible same-type,
and counterfactual interventions.  The same subset receives a direct-label
versus A/B label sensitivity check.  Cross-model transfer evaluates each of
each source-model core on each target checkpoint in the frozen cohort for the
cost-ordered JointCore and cost-aware singleton methods.  If direct and A/B
conclusions disagree or the live compute budget expires, the registered
archive-fallback result is reported instead of silently changing prompts.

## Outcomes and statistics

Co-primary outcomes are exact/trajectory task retention and intervention
reduction relative to cost-aware conditional singleton.  The task-retention
non-inferiority margin is 5 percentage points.  The practical target is at
least 15% fewer interventions at n=16, at least 50% active-token savings, and
no more than a 2-point task-success loss.  Analyses retain all registered
tasks, use 2,000 task-cluster bootstrap replicates, paired permutation tests,
McNemar tests for paired binary outcomes, and Holm correction.  Results are
reported per model and as an equally weighted model-by-family pool.

## Integrity

Raw model outputs, token counts, runtime metadata, configuration hashes,
model revision hashes, and task-level records are retained.  Errors stay in
the denominator.  The manuscript must disclose that the method and interface
were developed after observing the earlier negative and post-hoc pilot
results.  Human authors must verify every claim, citation, and number before
submission.
