# JointCore paper-expansion denominator correction and final freeze v4

Before any confirmation checkpoint was evaluated, an audit found that the
72-task development iterator took the first 72 Cartesian cells.  This covered
only the first two outcome families, although the intended development design
was balanced across all nine families.  The complete and partial v1--v3
artifacts remain preserved and are not used as balanced method-development
evidence.

Version 4 changes only the development cell sampler: it chooses eight
deterministic, evenly spaced domain/support/seed cells from each of the nine
families, for exactly 72 tasks per development checkpoint.  The 432-task
confirmation iterator already exhausts the Cartesian product and is unchanged.
The exact 6,912-condition artifact is unchanged.

The frozen primary interface remains YES/NO condition-truth scoring, mapped
deterministically to balanced EXECUTE/HOLD actions, followed by checkpoint
COMMIT/HALT selection.  The frozen method remains cost-ordered JointCore:
low-relevance records are contiguous and precede the top-four relevance block,
which is ordered by decreasing token cost.  B=4, 64-intervention cap, 768-token
input bound, methods, costs, four checkpoints, robustness subset, statistics,
and gates are unchanged.  No further performance-motivated change is allowed.

The paper must disclose the v1 interface search, the v2 cross-model label
failure, and this v3 denominator implementation error.  Confirmation is a v4
evaluation and is not used to make further choices.
