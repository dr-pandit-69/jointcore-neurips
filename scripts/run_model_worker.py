#!/usr/bin/env python3
"""Resumable one-checkpoint worker for JointCore paper experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IDEA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = IDEA_ROOT
sys.path.insert(0, str(IDEA_ROOT / "src"))

from jointcore.llm_runtime import load_model, model_spec, score_choices
from jointcore.paper_expansion import iter_model_tasks, run_model_task, task_record_checksum, validate_paper_config
from jointcore.utils import atomic_json, canonical_json, config_hash, git_metadata, load_json, sha256_file, sha256_text


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def valid_existing(path: Path, actor: str, task_id: str, config_digest: str, models_digest: str) -> bool:
    try:
        payload = load_json(path)
        observed = dict(payload)
        checksum = observed.pop("record_sha256")
        return bool(
            checksum == task_record_checksum(observed)
            and payload.get("actor") == actor
            and payload.get("paper_task_id") == task_id
            and payload.get("config_hash") == config_digest
            and payload.get("models_hash") == models_digest
            and payload.get("status") == "complete"
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
    parser.add_argument("--task-limit", type=int, default=None, help="Diagnostic partial run; never marked complete.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    models_path = args.models if args.models.is_absolute() else REPO_ROOT / args.models
    config, models = load_json(config_path), load_json(models_path)
    validate_paper_config(config)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Invalid deterministic shard assignment")
    registered_lanes = config.get("actor_lanes", {}).get(args.model, ())
    if isinstance(registered_lanes, str):
        registered_lanes = (registered_lanes,)
    if args.model not in config["actors"] or args.lane not in registered_lanes:
        raise SystemExit("Actor/lane is outside the registered configuration")
    if sha256_file(REPO_ROOT / config["source_spec"]) != config["source_sha256"]:
        raise RuntimeError("Paper protocol SHA-256 mismatch")
    config_digest, models_digest = config_hash(config), config_hash(models)
    output_root = args.output_root or (IDEA_ROOT / "outputs" / config["artifact_id"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    task_dir = output_root / "raw" / args.model / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = list(enumerate(iter_model_tasks(config)))
    tasks = [(index, item) for index, item in all_tasks if index % args.num_shards == args.shard_index]
    if args.task_limit is not None:
        tasks = tasks[: max(0, int(args.task_limit))]
    remaining = []
    for index, item in tasks:
        path = task_dir / f"{index:04d}_{item.paper_task_id[:12]}.json"
        if not path.exists() or not valid_existing(path, args.model, item.paper_task_id, config_digest, models_digest):
            remaining.append((index, item, path))
    started_utc = utc_now()
    if remaining:
        spec = model_spec(models, args.model, set(config["model_roles"]))
        tokenizer, model, runtime = load_model(spec)
    else:
        runtime = {"resume_without_model_load": True}
        tokenizer = model = None
    calls = 0
    started = time.perf_counter()

    def invoker(messages: list[dict[str, str]], choices: tuple[str, ...], seed: int) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result = score_choices(tokenizer, model, messages, choices, int(config["interface"]["maximum_input_tokens"]))
        result["decoding"]["seed"] = int(seed)
        return result

    for completed, (index, item, path) in enumerate(remaining, start=1):
        result = run_model_task(item, config, invoker)
        payload = {
            "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"], "status": "complete",
            "actor": args.model, "lane": args.lane, "task_index": index, "paper_task_id": item.paper_task_id,
            "config_hash": config_digest, "models_hash": models_digest, "source_sha256": config["source_sha256"],
            "result": result, "ended_utc": utc_now(),
        }
        payload["record_sha256"] = task_record_checksum(payload)
        atomic_json(path, payload)
        if completed == 1 or completed % 6 == 0 or completed == len(remaining):
            print(json.dumps({"actor": args.model, "completed_this_run": completed, "remaining_this_run": len(remaining) - completed, "task_index": index, "calls": calls}), flush=True)
    expected_shard_tasks = sum(1 for index, _ in all_tasks if index % args.num_shards == args.shard_index)
    valid_shard_tasks = sum(
        1
        for index, item in tasks
        if valid_existing(task_dir / f"{index:04d}_{item.paper_task_id[:12]}.json", args.model, item.paper_task_id, config_digest, models_digest)
    )
    complete = args.task_limit is None and valid_shard_tasks == expected_shard_tasks
    runtime.update({
        "scored_decision_calls_this_run": calls, "worker_elapsed_seconds_this_run": time.perf_counter() - started,
        "worker_started_utc": started_utc, "worker_ended_utc": utc_now(), "resumed_valid_tasks": len(tasks) - len(remaining),
    })
    manifest = {
        "schema_version": 1, "idea_id": "jointcore", "stage": config["stage"],
        "status": "complete" if complete else "partial", "actor": args.model, "lane": args.lane,
        "expected_tasks": int(config["task_grid"]["task_count"]), "observed_task_files": len(list(task_dir.glob("*.json"))),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "expected_shard_tasks": expected_shard_tasks, "validated_shard_tasks": valid_shard_tasks,
        "config_path": str(config_path.relative_to(REPO_ROOT)), "config_hash": config_digest,
        "models_path": str(models_path.relative_to(REPO_ROOT)), "models_hash": models_digest,
        "source_sha256": config["source_sha256"], "runtime": runtime, "git": git_metadata(REPO_ROOT),
    }
    manifest["record_sha256"] = sha256_text(canonical_json(manifest))
    manifest_name = (
        f"{args.model}.manifest.json"
        if args.num_shards == 1
        else f"{args.model}.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.manifest.json"
    )
    atomic_json(output_root / "manifests" / manifest_name, manifest)
    print(json.dumps({"actor": args.model, "status": manifest["status"], "tasks": manifest["observed_task_files"], "calls_this_run": calls, "output_root": str(output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
