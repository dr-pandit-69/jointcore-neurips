from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jointcore.llm_runtime import model_spec, validate_model_config
from jointcore.paper_expansion import (
    METHODS,
    expected_decision,
    iter_exact_tasks,
    iter_model_tasks,
    render_decision_messages,
    run_exact_task,
    validate_paper_config,
)
from jointcore.utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class PaperExpansionTests(unittest.TestCase):
    def test_registered_denominators_and_protocol_hashes(self) -> None:
        exact = load("configs/experiments/exact_benchmark.json")
        confirmation = load("configs/experiments/confirmation.json")
        validate_paper_config(exact, exact=True)
        validate_paper_config(confirmation)
        exact_tasks = list(iter_exact_tasks(exact))
        model_tasks = list(iter_model_tasks(confirmation))
        self.assertEqual(len(exact_tasks), 6_912)
        self.assertEqual(len({task.paper_task_id for task in exact_tasks}), 6_912)
        self.assertEqual(len(model_tasks), 432)
        self.assertEqual(len({task.paper_task_id for task in model_tasks}), 432)
        self.assertEqual(confirmation["actors"], ["gemma4_12b_it", "mistral31_24b_4bit"])
        self.assertEqual(sha256_file(ROOT / exact["source_spec"]), exact["source_sha256"])
        self.assertEqual(sha256_file(ROOT / confirmation["source_spec"]), confirmation["source_sha256"])

    def test_confirmation_grid_is_balanced(self) -> None:
        config = load("configs/experiments/confirmation.json")
        tasks = list(iter_model_tasks(config))
        self.assertEqual({task.action_when_sufficient for task in tasks}, {"EXECUTE", "HOLD"})
        for family in config["task_grid"]["families"]:
            self.assertEqual(sum(task.task.family == family for task in tasks), 48)
        for domain in config["task_grid"]["domains"]:
            self.assertEqual(sum(task.task.domain == domain for task in tasks), 72)

    def test_llama_extension_reuses_grid_and_partitions_cleanly(self) -> None:
        config = load("configs/experiments/llama_extension.json")
        validate_paper_config(config)
        tasks = list(iter_model_tasks(config))
        shards = [set(range(index, len(tasks), 2)) for index in range(2)]
        self.assertEqual([len(shard) for shard in shards], [216, 216])
        self.assertFalse(shards[0] & shards[1])
        self.assertEqual(shards[0] | shards[1], set(range(432)))
        self.assertTrue(config["freeze"]["posthoc_model_extension"])

    def test_exact_record_has_complete_method_matrix(self) -> None:
        config = load("configs/experiments/exact_benchmark.json")
        task = next(
            item
            for item in iter_exact_tasks(config)
            if item.task.family == "synergy_and" and item.task.candidate_count == 16
        )
        record = run_exact_task(task, config)
        self.assertEqual(tuple(record["methods"]), METHODS)
        self.assertTrue(record["methods"]["jointcore_cost_ordered"]["exact_sufficiency"])
        self.assertLess(
            record["methods"]["jointcore_cost_ordered"]["intervention_count"],
            record["methods"]["conditional_singleton_cost"]["intervention_count"],
        )

    def test_boolean_renderer_preserves_action_semantics(self) -> None:
        config = load("configs/experiments/confirmation.json")
        task = next(iter_model_tasks(config))
        messages, choices, mapping = render_decision_messages(
            task, task.task.full_memory, labels="boolean"
        )
        self.assertEqual(choices, ("YES", "NO"))
        self.assertEqual(mapping["YES"], task.action_when_sufficient)
        self.assertEqual(expected_decision(task, task.task.full_memory), task.action_when_sufficient)
        self.assertIn("Memory records:", messages[1]["content"])

    def test_model_paths_are_supplied_only_through_environment(self) -> None:
        inventory = load("configs/models/frozen_models.json")
        validate_model_config(inventory, {"paper_expansion_actor"})
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "JOINTCORE_GEMMA4_12B_PATH"):
                model_spec(inventory, "gemma4_12b_it", {"paper_expansion_actor"})
        with patch.dict(os.environ, {"JOINTCORE_GEMMA4_12B_PATH": "/models/gemma12"}):
            spec = model_spec(inventory, "gemma4_12b_it", {"paper_expansion_actor"})
        self.assertEqual(spec["local_path"], "/models/gemma12")


if __name__ == "__main__":
    unittest.main()
