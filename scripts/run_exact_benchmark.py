#!/usr/bin/env python3
"""Run and aggregate the registered 6,912-task exact paper benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IDEA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = IDEA_ROOT
sys.path.insert(0, str(IDEA_ROOT / "src"))

from jointcore.paper_expansion import METHODS, iter_exact_tasks, run_exact_task, validate_paper_config
from jointcore.utils import canonical_json, config_hash, git_metadata, load_json, portable_path, sha256_file, sha256_text


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def one(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
        values = [row["methods"][method] for row in rows]
        numeric = lambda key: [float(value[key]) for value in values if isinstance(value.get(key), (int, float))]
        return {
            "records": len(values),
            "exact_sufficiency": mean(numeric("exact_sufficiency")),
            "valid_minimum_set": mean(numeric("valid_minimum_set")),
            "critical_recall": mean(numeric("critical_recall")),
            "active_token_savings": mean(numeric("active_token_savings")),
            "cost_gap": mean(numeric("cost_gap")),
            "intervention_count": mean(numeric("intervention_count")),
            "budget_exhaustion_rate": mean(numeric("budget_exhausted")),
        }

    overall = {method: one(records, method) for method in METHODS}
    by_family: dict[str, Any] = {}
    for family in sorted({row["task"]["stateful_task"]["family"] for row in records}):
        selected = [row for row in records if row["task"]["stateful_task"]["family"] == family]
        by_family[family] = {method: one(selected, method) for method in METHODS}
    by_n: dict[str, Any] = {}
    for n in sorted({int(row["task"]["stateful_task"]["candidate_count"]) for row in records}):
        selected = [row for row in records if int(row["task"]["stateful_task"]["candidate_count"]) == n]
        by_n[str(n)] = {method: one(selected, method) for method in METHODS}
    return {"overall": overall, "by_family": by_family, "by_candidate_count": by_n}


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=IDEA_ROOT / "configs/experiments/exact_benchmark.json")
    parser.add_argument("--output-root", type=Path, default=IDEA_ROOT / "outputs/exact")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    config = load_json(config_path)
    validate_paper_config(config, exact=True)
    source = REPO_ROOT / config["source_spec"]
    if sha256_file(source) != config["source_sha256"]:
        raise RuntimeError("Paper protocol SHA-256 mismatch")
    started = utc_now()
    records = [run_exact_task(task, config) for task in iter_exact_tasks(config)]
    if len(records) != 6912 or len({row["task"]["paper_task_id"] for row in records}) != 6912:
        raise RuntimeError("Exact denominator or task identity check failed")
    raw_lines = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    raw_path = output_root / "raw/exact_records.jsonl"
    atomic_text(raw_path, raw_lines)
    metrics = {
        "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"],
        "status": "complete", "task_count": len(records), "method_records": len(records) * len(METHODS),
        "config_hash": config_hash(config), "source_sha256": config["source_sha256"],
        "metrics": summarize(records), "generated_utc": utc_now(),
    }
    metrics["record_sha256"] = sha256_text(canonical_json(metrics))
    metrics_path = output_root / "processed/metrics.json"
    atomic_text(metrics_path, json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"], "status": "complete",
        "started_utc": started, "ended_utc": utc_now(), "config_path": portable_path(config_path, REPO_ROOT),
        "config_sha256": config_hash(config), "source_sha256": config["source_sha256"],
        "raw_path": portable_path(raw_path, REPO_ROOT), "raw_sha256": sha256_file(raw_path),
        "metrics_path": portable_path(metrics_path, REPO_ROOT), "metrics_sha256": sha256_file(metrics_path),
        "git": git_metadata(REPO_ROOT),
    }
    atomic_text(output_root / "manifests/RUN.manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    jc = metrics["metrics"]["overall"]["jointcore_cost_ordered"]
    baseline = metrics["metrics"]["overall"]["conditional_singleton_cost"]
    print(json.dumps({"tasks": len(records), "jointcore": jc, "conditional_singleton": baseline, "output": str(metrics_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
