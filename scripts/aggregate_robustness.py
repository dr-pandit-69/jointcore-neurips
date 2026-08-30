#!/usr/bin/env python3
"""Aggregate the fixed robustness subset and four-by-four core transfer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

IDEA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = IDEA_ROOT
sys.path.insert(0, str(IDEA_ROOT / "src"))

from jointcore.paper_expansion import ROBUSTNESS_MODES, task_record_checksum, validate_paper_config
from jointcore.utils import atomic_json, canonical_json, config_hash, load_json, sha256_text


CORE_METHODS = ("conditional_singleton_cost", "jointcore_cost_ordered")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def checked(path: Path, config_digest: str, models_digest: str) -> dict[str, Any]:
    payload = load_json(path)
    observed = dict(payload)
    checksum = observed.pop("record_sha256", None)
    if checksum != task_record_checksum(observed):
        raise ValueError(f"Checksum mismatch: {path}")
    if payload.get("config_hash") != config_digest or payload.get("models_hash") != models_digest:
        raise ValueError(f"Provenance mismatch: {path}")
    return payload


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
    digest, models_digest = config_hash(config), config_hash(models)
    root = args.output_root or IDEA_ROOT / "outputs" / config["artifact_id"]
    if not root.is_absolute():
        root = REPO_ROOT / root
    subset_count = int(config["robustness"]["fixed_subset_count"])
    robustness: dict[str, Any] = {}
    for actor in config["actors"]:
        paths = sorted((root / "robustness" / actor / "tasks").glob("*.json"))
        if len(paths) != subset_count:
            raise ValueError(f"Robustness denominator mismatch for {actor}: {len(paths)}/{subset_count}")
        rows = [checked(path, digest, models_digest) for path in paths]
        methods: dict[str, Any] = {}
        for method in CORE_METHODS:
            values = [row["methods"][method] for row in rows]
            methods[method] = {
                "records": len(values),
                "primary_trajectory_success": mean([float(value["primary_trajectory_success"]) for value in values]),
                "direct_trajectory_success": mean([float(value["direct"]["trajectory_success"]) for value in values]),
                "ab_trajectory_success": mean([float(value["ab"]["trajectory_success"]) for value in values]),
                "primary_direct_action_agreement": mean([float(value["primary_direct_action_agreement"]) for value in values]),
                "primary_ab_action_agreement": mean([float(value["direct_ab_action_agreement"]) for value in values]),
                "validity_modes": {
                    mode: mean([float(value["validity_modes"][mode]["trajectory_success"]) for value in values])
                    for mode in ROBUSTNESS_MODES
                },
            }
        robustness[actor] = {"tasks": len(rows), "methods": methods}
    transfer: dict[str, Any] = {}
    if bool(config.get("robustness", {}).get("cross_model_transfer", True)):
        expected = int(config["task_grid"]["task_count"])
        for target in config["actors"]:
            transfer[target] = {}
            for source in config["actors"]:
                paths = sorted((root / "transfer" / target / source / "tasks").glob("*.json"))
                if len(paths) != expected:
                    raise ValueError(f"Transfer denominator mismatch {source}->{target}: {len(paths)}/{expected}")
                rows = [checked(path, digest, models_digest) for path in paths]
                transfer[target][source] = {
                    method: {
                        "records": len(rows),
                        "behavior_retention": mean([
                            float(row["methods"][method]["trajectory"]["decision"]["action"] ==
                                  load_json(sorted((root / "raw" / target / "tasks").glob("*.json"))[int(row["task_index"])])["result"]["full_memory_control"]["action"])
                            for row in rows
                        ]),
                        "trajectory_correctness": mean([float(row["methods"][method]["trajectory"]["trajectory_success"]) for row in rows]),
                    }
                    for method in CORE_METHODS
                }
    interface_disagreements = []
    for actor, values in robustness.items():
        methods = values["methods"]
        direct_conclusion = (
            float(methods["jointcore_cost_ordered"]["direct_trajectory_success"])
            >= float(methods["conditional_singleton_cost"]["direct_trajectory_success"])
        )
        ab_conclusion = (
            float(methods["jointcore_cost_ordered"]["ab_trajectory_success"])
            >= float(methods["conditional_singleton_cost"]["ab_trajectory_success"])
        )
        if direct_conclusion != ab_conclusion:
            interface_disagreements.append(actor)
    archive_fallback = {
        "triggered": bool(interface_disagreements),
        "trigger_actors": interface_disagreements,
        "policy": "When direct and A/B method conclusions disagree, report full-memory/archive retention instead of selecting a favorable label interface.",
        "behavior_retention": 1.0 if interface_disagreements else None,
        "active_token_savings": 0.0 if interface_disagreements else None,
    }
    metrics = {
        "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"] + "_robustness_transfer",
        "status": "complete", "config_hash": digest, "models_hash": models_digest,
        "robustness": robustness, "transfer": transfer, "archive_fallback": archive_fallback,
    }
    metrics["record_sha256"] = sha256_text(canonical_json(metrics))
    output = root / "processed/robustness_transfer_metrics.json"
    atomic_json(output, metrics)
    lines = ["# JointCore robustness and transfer", "", "## Intervention and interface robustness", "", "The raw operator named `plausible_same_type` writes the schema-compatible PENDING sentinel; it is not an in-distribution plausible replacement.", "", "| Actor | Method | Primary | Deletion | Typed mask | PENDING sentinel | Counterfactual | Direct | A/B |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for actor in config["actors"]:
        for method in CORE_METHODS:
            value = robustness[actor]["methods"][method]
            modes = value["validity_modes"]
            lines.append(f"| {actor} | {method} | {value['primary_trajectory_success']:.3f} | {modes['deletion']:.3f} | {modes['typed_mask']:.3f} | {modes['plausible_same_type']:.3f} | {modes['counterfactual']:.3f} | {value['direct_trajectory_success']:.3f} | {value['ab_trajectory_success']:.3f} |")
    lines.extend(["", "## Cross-model behavior retention", ""])
    if transfer:
        lines.extend(["Rows are target checkpoints; columns are source cores.", ""])
        for method in CORE_METHODS:
            lines.extend([f"### {method}", "", "| Target \\ Source | " + " | ".join(config["actors"]) + " |", "|---|" + "---:|" * len(config["actors"])])
            for target in config["actors"]:
                lines.append("| " + target + " | " + " | ".join(f"{transfer[target][source][method]['behavior_retention']:.3f}" for source in config["actors"]) + " |")
            lines.append("")
    else:
        lines.extend(["Not run for this one-model extension; no cross-model source/target comparison is defined.", ""])
    lines.extend([
        "## Registered archive fallback", "",
        f"Triggered: {'yes' if archive_fallback['triggered'] else 'no'}.",
        (f"Trigger actors: {', '.join(interface_disagreements)}." if interface_disagreements else "Direct and A/B method conclusions did not conflict."),
        "If triggered, the reported fallback retains full memory (behavior retention 1.0, active-payload savings 0.0).", "",
    ])
    report_path = root / "reports/ROBUSTNESS_TRANSFER.md"
    atomic_text(report_path, "\n".join(lines))
    print(json.dumps({"metrics": str(output), "report": str(report_path), "status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
