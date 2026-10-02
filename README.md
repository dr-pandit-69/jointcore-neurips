# JointCore [ Accepted @ Personalized, aligned, long-term memory for AI systems Workshop, NeurIPS 2026, Paris ]

This repository contains the code for the JointCore exact, model, robustness,
and transfer experiments. It also contains frozen manuscript aggregates under
`results/`. It does not contain model weights, paper sources, or raw model
outputs.

## Setup and verification

Python 3.12 and `uv` are required. CPU checks do not install PyTorch.

```bash
uv sync --frozen
uv run python -m unittest discover -s tests -v
uv run python scripts/run_exact_benchmark.py --output-root outputs/exact
```

The exact benchmark is CPU-only and produces 6,912 deterministic task records.
All generated files are written below `outputs/`.

## Model experiments

Install the pinned CUDA dependencies:

```bash
uv sync --frozen --extra gpu
```

Download the pinned snapshots listed in `THIRD_PARTY_MODELS.md`, then export
their absolute paths as required by `configs/models/*.json`:

```bash
export JOINTCORE_GEMMA4_12B_PATH=/absolute/path/to/gemma-4-12b
export JOINTCORE_MISTRAL31_24B_PATH=/absolute/path/to/mistral-24b-4bit
export JOINTCORE_LLAMA33_70B_PATH=/absolute/path/to/llama-3.3-70b
export JOINTCORE_QWEN25_14B_PATH=/absolute/path/to/qwen2.5-14b
```

Run one registered model on an allocation exposing exactly one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_model_worker.py \
  --model gemma4_12b_it --lane a100-2
```

The `lane` value is a logical provenance label. The frozen assignment is
recorded in `configs/experiments/confirmation.json`. Workers are resumable and
can be split deterministically with `--num-shards N --shard-index I`.

To reproduce the four-model cohort displayed in the manuscript, use the
post-study packaging config without changing the historical freeze files:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_model_worker.py \
  --config configs/experiments/reported_cohort.json \
  --models configs/models/reported_models.json \
  --model qwen25_14b --lane a100-2 \
  --output-root outputs/reported_cohort_reproduction_v1
```

Run the same command for `gemma4_12b_it` and `mistral31_24b_4bit`. For
`llama33_70b_4bit`, use two deterministic shards with lane labels `a100-1` and
`a100-2`, then aggregate all four models:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_model_worker.py \
  --config configs/experiments/reported_cohort.json \
  --models configs/models/reported_models.json \
  --model llama33_70b_4bit --lane a100-1 \
  --num-shards 2 --shard-index 0 \
  --output-root outputs/reported_cohort_reproduction_v1

CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_model_worker.py \
  --config configs/experiments/reported_cohort.json \
  --models configs/models/reported_models.json \
  --model llama33_70b_4bit --lane a100-2 \
  --num-shards 2 --shard-index 1 \
  --output-root outputs/reported_cohort_reproduction_v1

uv run python scripts/aggregate_models.py \
  --config configs/experiments/reported_cohort.json \
  --models configs/models/reported_models.json \
  --output-root outputs/reported_cohort_reproduction_v1
```

After each model completes:

```bash
uv run python scripts/aggregate_models.py
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_robustness_worker.py \
  --model gemma4_12b_it --lane a100-2
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_transfer_worker.py \
  --target-model gemma4_12b_it --lane a100-2
uv run python scripts/aggregate_robustness.py
```

For the reported Llama 70B experiments, pass
`configs/experiments/llama_extension.json` and
`configs/models/llama_extension.json` explicitly. Run shards 0 and 1 on the
two registered lanes with a shared `--output-root`, then aggregate with the
same config, inventory, and output root.

## Reproducibility contract

- The experimental grids, seeds, interfaces, and gates are frozen in
  `configs/experiments/`.
- Protocol files in `docs/` are hash-checked before execution.
- Each task and aggregate carries deterministic IDs and checksums.
- Missing, duplicate, failed, or provenance-mismatched records are rejected.
- Unreported model-screening cohorts and utilities are intentionally outside
  this release.

The source code is MIT-licensed. Model checkpoints remain under their upstream
licenses; see `THIRD_PARTY_MODELS.md`.
