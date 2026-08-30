#!/usr/bin/env python3
"""Evaluate source-model JointCore/singleton cores on one target checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IDEA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = IDEA_ROOT
sys.path.insert(0, str(IDEA_ROOT / "src"))

from jointcore.llm_runtime import load_model, model_spec, score_choices
from jointcore.paper_expansion import evaluate_retained_trajectory, paper_task_from_dict, task_record_checksum, validate_paper_config
from jointcore.utils import atomic_json, canonical_json, config_hash, git_metadata, load_json, sha256_text


CORE_METHODS = ("conditional_singleton_cost", "jointcore_cost_ordered")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def valid_existing(path: Path, source: str, target: str, task_id: str, config_digest: str, models_digest: str) -> bool:
    try:
        payload = load_json(path)
        observed = dict(payload)
        checksum = observed.pop("record_sha256")
        return bool(
            checksum == task_record_checksum(observed) and payload.get("source_actor") == source
            and payload.get("target_actor") == target and payload.get("paper_task_id") == task_id
            and payload.get("config_hash") == config_digest and payload.get("models_hash") == models_digest
            and payload.get("status") == "complete"
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--lane", choices=("a100-1", "a100-2"), required=True)
    parser.add_argument("--config", type=Path, default=IDEA_ROOT / "configs/experiments/confirmation.json")
    parser.add_argument("--models", type=Path, default=IDEA_ROOT / "configs/models/frozen_models.json")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    models_path = args.models if args.models.is_absolute() else REPO_ROOT / args.models
    config, models = load_json(config_path), load_json(models_path)
    validate_paper_config(config)
    config_digest, models_digest = config_hash(config), config_hash(models)
    target = args.target_model
    if target not in config["actors"] or config["actor_lanes"][target] != args.lane:
        raise SystemExit("Target/lane is outside the registered configuration")
    root = args.output_root or IDEA_ROOT / "outputs" / config["artifact_id"]
    if not root.is_absolute():
        root = REPO_ROOT / root
    expected = int(config["task_grid"]["task_count"])
    jobs = []
    for source in config["actors"]:
        source_paths = sorted((root / "raw" / source / "tasks").glob("*.json"))
        if len(source_paths) != expected:
            raise RuntimeError(f"Primary source records incomplete for {source}: {len(source_paths)}/{expected}")
        output_dir = root / "transfer" / target / source / "tasks"
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(source_paths):
            source_payload = load_json(source_path)
            output = output_dir / f"{index:04d}_{source_payload['paper_task_id'][:12]}.json"
            if not output.exists() or not valid_existing(output, source, target, source_payload["paper_task_id"], config_digest, models_digest):
                jobs.append((source, index, source_payload, output))
    off_diagonal = [job for job in jobs if job[0] != target]
    if off_diagonal:
        spec = model_spec(models, target, set(config["model_roles"]))
        tokenizer, model, runtime = load_model(spec)
    else:
        tokenizer = model = None
        runtime = {"resume_without_model_load": True}
    calls = 0
    started = time.perf_counter()
    primary_labels = str(config["interface"].get("decision_label_mode", "direct"))
    trajectory_cache: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}

    def invoker(messages: list[dict[str, str]], choices: tuple[str, ...], seed: int) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result = score_choices(tokenizer, model, messages, choices, int(config["interface"]["maximum_input_tokens"]))
        result["decoding"]["seed"] = seed
        return result

    for completed, (source, index, source_payload, output) in enumerate(jobs, start=1):
        source_result = source_payload["result"]
        item = paper_task_from_dict(source_result["task"])
        methods: dict[str, Any] = {}
        for method in CORE_METHODS:
            selected = source_result["methods"][method]["selected_variables"]
            if source == target:
                original = source_result["methods"][method]
                trajectory = {
                    "retained_variables": selected, "intervention_mode": "typed_mask", "labels": primary_labels,
                    "decision": original["final_decision"], "terminal": original["terminal_decision"],
                    "trajectory_success": original["trajectory_success"], "diagonal_reused": True,
                }
            else:
                cache_key = (item.paper_task_id, tuple(sorted(selected)))
                cache_hit = cache_key in trajectory_cache
                if not cache_hit:
                    trajectory_cache[cache_key] = evaluate_retained_trajectory(
                        item, selected, invoker, item.task.seed + 40000 + 101 * index + len(source) + len(method),
                        labels=primary_labels,
                    )
                trajectory = deepcopy(trajectory_cache[cache_key])
                trajectory["diagonal_reused"] = False
                trajectory["transfer_cache_hit"] = cache_hit
            methods[method] = {"selected_variables": selected, "trajectory": trajectory}
        payload = {
            "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"] + "_transfer",
            "status": "complete", "source_actor": source, "target_actor": target, "task_index": index,
            "paper_task_id": item.paper_task_id, "config_hash": config_digest, "models_hash": models_digest,
            "methods": methods, "ended_utc": utc_now(),
        }
        payload["record_sha256"] = task_record_checksum(payload)
        atomic_json(output, payload)
        if completed == 1 or completed % 72 == 0 or completed == len(jobs):
            print(json.dumps({"target": target, "completed": completed, "remaining": len(jobs) - completed, "calls": calls}), flush=True)
    counts = {
        source: len(list((root / "transfer" / target / source / "tasks").glob("*.json")))
        for source in config["actors"]
    }
    complete = all(value == expected for value in counts.values())
    runtime.update({"calls_this_run": calls, "elapsed_seconds_this_run": time.perf_counter() - started})
    manifest = {
        "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"] + "_transfer",
        "status": "complete" if complete else "partial", "target_actor": target, "source_counts": counts,
        "expected_per_source": expected, "config_hash": config_digest, "models_hash": models_digest,
        "runtime": runtime, "ended_utc": utc_now(), "git": git_metadata(REPO_ROOT),
    }
    manifest["record_sha256"] = sha256_text(canonical_json(manifest))
    atomic_json(root / "manifests" / f"{target}.transfer.manifest.json", manifest)
    print(json.dumps({"target": target, "status": manifest["status"], "source_counts": counts, "calls_this_run": calls}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
