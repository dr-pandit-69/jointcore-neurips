#!/usr/bin/env python3
"""Run fixed-subset intervention-validity and label-sensitivity analyses."""

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
from jointcore.paper_expansion import ROBUSTNESS_MODES, evaluate_retained_trajectory, paper_task_from_dict, task_record_checksum, validate_paper_config
from jointcore.utils import atomic_json, canonical_json, config_hash, git_metadata, load_json, sha256_text


CORE_METHODS = ("conditional_singleton_cost", "jointcore_cost_ordered")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def valid_existing(path: Path, actor: str, task_id: str, config_digest: str, models_digest: str) -> bool:
    try:
        payload = load_json(path)
        observed = dict(payload)
        checksum = observed.pop("record_sha256")
        return bool(
            checksum == task_record_checksum(observed) and payload.get("actor") == actor
            and payload.get("paper_task_id") == task_id and payload.get("config_hash") == config_digest
            and payload.get("models_hash") == models_digest and payload.get("status") == "complete"
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lane", choices=("a100-1", "a100-2"), required=True)
    parser.add_argument("--config", type=Path, default=IDEA_ROOT / "configs/experiments/confirmation.json")
    parser.add_argument("--models", type=Path, default=IDEA_ROOT / "configs/models/frozen_models.json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    models_path = args.models if args.models.is_absolute() else REPO_ROOT / args.models
    config, models = load_json(config_path), load_json(models_path)
    validate_paper_config(config)
    config_digest, models_digest = config_hash(config), config_hash(models)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Invalid deterministic shard assignment")
    registered_lanes = config.get("actor_lanes", {}).get(args.model, ())
    if isinstance(registered_lanes, str):
        registered_lanes = (registered_lanes,)
    if args.model not in config["actors"] or args.lane not in registered_lanes:
        raise SystemExit("Actor/lane is outside the registered configuration")
    root = args.output_root or IDEA_ROOT / "outputs" / config["artifact_id"]
    if not root.is_absolute():
        root = REPO_ROOT / root
    primary = sorted((root / "raw" / args.model / "tasks").glob("*.json"))
    expected = int(config["task_grid"]["task_count"])
    if len(primary) != expected:
        raise RuntimeError(f"Primary confirmation is incomplete for {args.model}: {len(primary)}/{expected}")
    subset_count = int(config["robustness"]["fixed_subset_count"])
    stride = expected // subset_count
    selected_paths = primary[::stride][:subset_count]
    if len(selected_paths) != subset_count:
        raise RuntimeError("Fixed robustness subset construction failed")
    output_dir = root / "robustness" / args.model / "tasks"
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = []
    selected_shard = [
        (subset_index, path)
        for subset_index, path in enumerate(selected_paths)
        if subset_index % args.num_shards == args.shard_index
    ]
    for subset_index, path in selected_shard:
        primary_payload = load_json(path)
        output = output_dir / f"{subset_index:04d}_{primary_payload['paper_task_id'][:12]}.json"
        if not output.exists() or not valid_existing(output, args.model, primary_payload["paper_task_id"], config_digest, models_digest):
            remaining.append((subset_index, primary_payload, output))
    if remaining:
        spec = model_spec(models, args.model, set(config["model_roles"]))
        tokenizer, model, runtime = load_model(spec)
    else:
        tokenizer = model = None
        runtime = {"resume_without_model_load": True}
    calls = 0
    started = time.perf_counter()
    primary_labels = str(config["interface"].get("decision_label_mode", "direct"))
    trajectory_cache: dict[tuple[str, tuple[int, ...], str, str], dict[str, Any]] = {}

    def invoker(messages: list[dict[str, str]], choices: tuple[str, ...], seed: int) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result = score_choices(tokenizer, model, messages, choices, int(config["interface"]["maximum_input_tokens"]))
        result["decoding"]["seed"] = seed
        return result

    def evaluate(item: Any, retained: list[int], seed: int, mode: str, labels: str) -> dict[str, Any]:
        key = (item.paper_task_id, tuple(sorted(retained)), mode, labels)
        hit = key in trajectory_cache
        if not hit:
            trajectory_cache[key] = evaluate_retained_trajectory(
                item, retained, invoker, seed, intervention_mode=mode, labels=labels,
            )
        value = deepcopy(trajectory_cache[key])
        value["robustness_cache_hit"] = hit
        return value

    for completed, (subset_index, primary_payload, output) in enumerate(remaining, start=1):
        primary_result = primary_payload["result"]
        item = paper_task_from_dict(primary_result["task"])
        methods: dict[str, Any] = {}
        for method in CORE_METHODS:
            retained = primary_result["methods"][method]["selected_variables"]
            direct = primary_result["methods"][method]
            modes = {
                mode: evaluate(
                    item, retained, item.task.seed + 30000 + 100 * subset_index + offset,
                    mode, primary_labels,
                )
                for offset, mode in enumerate(ROBUSTNESS_MODES)
            }
            ab = evaluate(item, retained, item.task.seed + 31000 + subset_index, "typed_mask", "ab")
            direct_sensitivity = evaluate(item, retained, item.task.seed + 32000 + subset_index, "typed_mask", "direct")
            methods[method] = {
                "selected_variables": retained,
                "primary_labels": primary_labels,
                "primary_action": direct["final_decision"]["action"],
                "primary_trajectory_success": direct["trajectory_success"],
                "validity_modes": modes,
                "ab": ab,
                "direct": direct_sensitivity,
                "direct_ab_action_agreement": direct["final_decision"]["action"] == ab["decision"]["action"],
                "primary_direct_action_agreement": direct["final_decision"]["action"] == direct_sensitivity["decision"]["action"],
            }
        payload = {
            "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"] + "_robustness",
            "status": "complete", "actor": args.model, "subset_index": subset_index,
            "paper_task_id": item.paper_task_id, "config_hash": config_digest, "models_hash": models_digest,
            "methods": methods, "ended_utc": utc_now(),
        }
        payload["record_sha256"] = task_record_checksum(payload)
        atomic_json(output, payload)
        if completed == 1 or completed % 12 == 0 or completed == len(remaining):
            print(json.dumps({"actor": args.model, "completed": completed, "remaining": len(remaining) - completed, "calls": calls}), flush=True)
    observed = len(list(output_dir.glob("*.json")))
    validated_shard_tasks = sum(
        valid_existing(
            output_dir / f"{subset_index:04d}_{load_json(path)['paper_task_id'][:12]}.json",
            args.model,
            load_json(path)["paper_task_id"],
            config_digest,
            models_digest,
        )
        for subset_index, path in selected_shard
    )
    expected_shard_tasks = len(selected_shard)
    runtime.update({"calls_this_run": calls, "elapsed_seconds_this_run": time.perf_counter() - started})
    manifest = {
        "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"] + "_robustness",
        "status": "complete" if validated_shard_tasks == expected_shard_tasks else "partial", "actor": args.model,
        "expected_tasks": subset_count, "observed_tasks": observed, "config_hash": config_digest,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "expected_shard_tasks": expected_shard_tasks, "validated_shard_tasks": validated_shard_tasks,
        "models_hash": models_digest, "runtime": runtime, "ended_utc": utc_now(), "git": git_metadata(REPO_ROOT),
    }
    manifest["record_sha256"] = sha256_text(canonical_json(manifest))
    manifest_name = (
        f"{args.model}.robustness.manifest.json"
        if args.num_shards == 1
        else f"{args.model}.robustness.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.manifest.json"
    )
    atomic_json(root / "manifests" / manifest_name, manifest)
    print(json.dumps({"actor": args.model, "status": manifest["status"], "tasks": observed, "calls_this_run": calls}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
