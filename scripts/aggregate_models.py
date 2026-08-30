#!/usr/bin/env python3
"""Validate, aggregate, and report JointCore paper model records."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IDEA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = IDEA_ROOT
sys.path.insert(0, str(IDEA_ROOT / "src"))

from jointcore.paper_expansion import METHODS, task_record_checksum, validate_paper_config
from jointcore.utils import atomic_json, canonical_json, config_hash, git_metadata, load_json, portable_path, sha256_file, sha256_text, stable_seed


METRICS = (
    "exact_sufficiency", "behavioral_sufficiency", "final_decision_correct", "trajectory_success",
    "valid_minimum_set", "critical_recall", "active_token_savings", "cost_gap", "intervention_count",
)
BINARY = {"exact_sufficiency", "behavioral_sufficiency", "final_decision_correct", "trajectory_success", "valid_minimum_set"}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row["result"]["methods"][method] for row in rows]
    result: dict[str, Any] = {"records": len(selected)}
    for metric in METRICS:
        values = [float(item[metric]) for item in selected if isinstance(item.get(metric), (int, float))]
        result[metric] = mean(values)
    result["decision_errors"] = sum(int(item["decision_errors"]) for item in selected)
    return result


def paired(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [
        float(row["result"]["methods"]["jointcore_cost_ordered"][metric])
        - float(row["result"]["methods"]["conditional_singleton_cost"][metric])
        for row in rows
        if isinstance(row["result"]["methods"]["jointcore_cost_ordered"].get(metric), (int, float))
        and isinstance(row["result"]["methods"]["conditional_singleton_cost"].get(metric), (int, float))
    ]


def bootstrap_ci(differences: list[float], replicates: int, seed: int) -> dict[str, Any]:
    if not differences:
        return {"estimate": None, "ci95": [None, None], "replicates": replicates}
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        estimates.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    estimates.sort()
    low = estimates[max(0, math.floor(0.025 * replicates))]
    high = estimates[min(replicates - 1, math.ceil(0.975 * replicates) - 1)]
    return {"estimate": mean(differences), "ci95": [low, high], "replicates": replicates}


def mcnemar_exact(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    b = c = 0
    for row in rows:
        jc = bool(row["result"]["methods"]["jointcore_cost_ordered"][metric])
        base = bool(row["result"]["methods"]["conditional_singleton_cost"][metric])
        b += int(jc and not base)
        c += int(base and not jc)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {"jointcore_only": b, "singleton_only": c, "p_value": p, "test": "exact_mcnemar"}


def paired_permutation(differences: list[float], seed: int, replicates: int = 20000) -> dict[str, Any]:
    if not differences:
        return {"p_value": None, "replicates": replicates, "test": "paired_sign_flip"}
    observed = abs(sum(differences) / len(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(replicates):
        estimate = abs(sum(value if rng.randrange(2) else -value for value in differences) / len(differences))
        extreme += int(estimate >= observed - 1e-15)
    return {"p_value": (extreme + 1) / (replicates + 1), "replicates": replicates, "test": "paired_sign_flip"}


def holm(entries: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(entries, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def load_actor_rows(root: Path, actor: str, config: dict[str, Any], models_digest: str) -> list[dict[str, Any]]:
    task_dir = root / "raw" / actor / "tasks"
    paths = sorted(task_dir.glob("*.json"))
    expected = int(config["task_grid"]["task_count"])
    if len(paths) != expected:
        raise ValueError(f"{actor}: expected {expected} task records, found {len(paths)}")
    rows = []
    for path in paths:
        payload = load_json(path)
        observed = dict(payload)
        checksum = observed.pop("record_sha256", None)
        if checksum != task_record_checksum(observed):
            raise ValueError(f"{actor}: checksum mismatch in {path.name}")
        if payload.get("actor") != actor or payload.get("config_hash") != config_hash(config) or payload.get("models_hash") != models_digest:
            raise ValueError(f"{actor}: provenance mismatch in {path.name}")
        rows.append(payload)
    if {int(row["task_index"]) for row in rows} != set(range(expected)):
        raise ValueError(f"{actor}: task index denominator mismatch")
    if len({row["paper_task_id"] for row in rows}) != expected:
        raise ValueError(f"{actor}: duplicate task identities")
    return rows


def family_weighted_pool(by_actor: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    actor_groups = [[actor] for actor in by_actor]
    families = sorted({row["result"]["task"]["stateful_task"]["family"] for rows in by_actor.values() for row in rows})
    output: dict[str, Any] = {}
    for method in METHODS:
        method_metrics: dict[str, Any] = {"records": sum(len(rows) for rows in by_actor.values())}
        for metric in METRICS:
            family_values = []
            for family in families:
                group_values = []
                for actors in actor_groups:
                    actor_values = []
                    for actor in actors:
                        rows = [row for row in by_actor[actor] if row["result"]["task"]["stateful_task"]["family"] == family]
                        values = [float(row["result"]["methods"][method][metric]) for row in rows if isinstance(row["result"]["methods"][method].get(metric), (int, float))]
                        if values:
                            actor_values.append(float(mean(values)))
                    if actor_values:
                        group_values.append(float(mean(actor_values)))
                if group_values:
                    family_values.append(float(mean(group_values)))
            method_metrics[metric] = mean(family_values)
        output[method] = method_metrics
    return output


def gate(values: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    jc, base = values["jointcore_cost_ordered"], values["conditional_singleton_cost"]
    task_diff = float(jc["behavioral_sufficiency"]) - float(base["behavioral_sufficiency"])
    reduction = 1.0 - float(jc["intervention_count"]) / float(base["intervention_count"])
    checks = {
        "behavioral_retention_noninferiority_5pp": task_diff >= -float(config.get("gates", {}).get("noninferiority_margin", 0.05)),
        "task_loss_at_most_2pp": task_diff >= -float(config.get("gates", {}).get("task_loss_max", 0.02)),
        "intervention_reduction_at_least_15pct": reduction >= float(config.get("gates", {}).get("intervention_reduction_target", 0.15)),
        "active_token_savings_at_least_50pct": float(jc["active_token_savings"]) >= float(config.get("gates", {}).get("active_token_savings_min", 0.50)),
    }
    return {"passed": all(checks.values()), "checks": checks, "behavioral_retention_difference": task_diff, "intervention_reduction": reduction}


def report(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    lines = [f"# JointCore {config['stage']} results", "", "## Registered outcome", ""]
    pooled_gate = metrics["gates"]["family_weighted_pool"]
    lines.append(f"**{'PASS' if pooled_gate['passed'] else 'NOT ALL GATES PASSED'}** on the family-weighted descriptive gate.")
    lines.extend(["", "Except for control, paired cells are cost-ordered JointCore / cost-aware singleton.", "", "| Actor | Full control | Behavior | Exact sufficiency | Minimum set | Savings | Interventions | Reduction | Gate |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for actor in config["actors"]:
        values = metrics["actors"][actor]
        jc, base, actor_gate = values["methods"]["jointcore_cost_ordered"], values["methods"]["conditional_singleton_cost"], metrics["gates"][actor]
        lines.append(
            f"| {actor} | {values['full_memory_control_accuracy']:.3f} | {jc['behavioral_sufficiency']:.3f}/{base['behavioral_sufficiency']:.3f} | "
            f"{jc['exact_sufficiency']:.3f}/{base['exact_sufficiency']:.3f} | {jc['valid_minimum_set']:.3f}/{base['valid_minimum_set']:.3f} | "
            f"{jc['active_token_savings']:.3f}/{base['active_token_savings']:.3f} | {jc['intervention_count']:.2f}/{base['intervention_count']:.2f} | "
            f"{actor_gate['intervention_reduction']:.3f} | {'pass' if actor_gate['passed'] else 'fail'} |"
        )
    lines.extend(["", "## Interpretation", "", "Primary behavioral retention is an internal invariant for valid deterministic conditional deletion, not independent generalization evidence. Exact logical and minimum-set outcomes remain nontrivial and are shown above. These results belong to the separately registered paper-expansion study. They do not overwrite the earlier negative v2--v5 experiment. Checkpoint-specific precision and quantization are recorded in the model inventory. All errors remain in the denominator.", ""])
    return "\n".join(lines)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=IDEA_ROOT / "configs/experiments/confirmation.json")
    parser.add_argument("--models", type=Path, default=IDEA_ROOT / "configs/models/frozen_models.json")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    models_path = args.models if args.models.is_absolute() else REPO_ROOT / args.models
    config, models = load_json(config_path), load_json(models_path)
    validate_paper_config(config)
    models_digest = config_hash(models)
    root = args.output_root or IDEA_ROOT / "outputs" / config["artifact_id"]
    if not root.is_absolute():
        root = REPO_ROOT / root
    by_actor = {actor: load_actor_rows(root, actor, config, models_digest) for actor in config["actors"]}
    actor_metrics: dict[str, Any] = {}
    statistical_tests: dict[str, Any] = {}
    p_values: list[tuple[str, float]] = []
    replicates = int(config.get("statistics", {}).get("cluster_bootstrap_replicates", 2000))
    stat_seed = int(config.get("statistics", {}).get("seed", 26082899))
    for actor, rows in by_actor.items():
        methods = {method: summary(rows, method) for method in METHODS}
        actor_metrics[actor] = {
            "tasks": len(rows),
            "full_memory_control_accuracy": mean([float(row["result"]["full_memory_control_correct"]) for row in rows]),
            "methods": methods,
            "families": {
                family: {method: summary([row for row in rows if row["result"]["task"]["stateful_task"]["family"] == family], method) for method in METHODS}
                for family in config["task_grid"]["families"]
            },
        }
        tests: dict[str, Any] = {}
        for offset, metric in enumerate(("behavioral_sufficiency", "active_token_savings", "intervention_count")):
            differences = paired(rows, metric)
            tests[metric] = {"cluster_bootstrap": bootstrap_ci(differences, replicates, stat_seed + stable_seed(actor, metric) % 100000)}
            if metric in BINARY:
                test = mcnemar_exact(rows, metric)
            else:
                test = paired_permutation(differences, stat_seed + 100000 + offset + stable_seed(actor) % 100000)
            tests[metric]["paired_test"] = test
            if test["p_value"] is not None:
                p_values.append((f"{actor}:{metric}", float(test["p_value"])))
        statistical_tests[actor] = tests
    adjusted = holm(p_values)
    for actor, tests in statistical_tests.items():
        for metric, values in tests.items():
            values["holm_adjusted_p_value"] = adjusted[f"{actor}:{metric}"]
    pooled = family_weighted_pool(by_actor)
    gates = {actor: gate(values["methods"], config) for actor, values in actor_metrics.items()}
    gates["family_weighted_pool"] = gate(pooled, config)
    metrics = {
        "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"], "status": "complete",
        "config_hash": config_hash(config), "models_hash": models_digest, "source_sha256": config["source_sha256"],
        "actors": actor_metrics, "family_weighted_pool": pooled, "statistical_tests": statistical_tests,
        "holm_family_size": len(p_values), "gates": gates, "generated_utc": utc_now(),
    }
    metrics["record_sha256"] = sha256_text(canonical_json(metrics))
    metrics_path = root / "processed/metrics.json"
    atomic_json(metrics_path, metrics)
    report_path = root / "reports/RESULTS.md"
    atomic_text(report_path, report(metrics, config))
    manifest = {
        "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"], "status": "complete",
        "config_path": portable_path(config_path, REPO_ROOT), "config_sha256": config_hash(config),
        "models_path": portable_path(models_path, REPO_ROOT), "models_sha256": models_digest,
        "metrics_path": portable_path(metrics_path, REPO_ROOT), "metrics_sha256": sha256_file(metrics_path),
        "report_path": portable_path(report_path, REPO_ROOT), "report_sha256": sha256_file(report_path),
        "aggregated_utc": utc_now(), "git": git_metadata(REPO_ROOT),
    }
    atomic_json(root / "manifests/AGGREGATE.manifest.json", manifest)
    print(json.dumps({"actors": config["actors"], "gate": gates["family_weighted_pool"], "metrics": str(metrics_path), "report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
