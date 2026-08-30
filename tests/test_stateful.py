from __future__ import annotations

import sys
import unittest
from pathlib import Path

IDEA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IDEA_ROOT / "src"))

from jointcore.stateful import build_stateful_task, oracle_minimal_sets, parse_action, render_messages, run_stateful_trajectory
from jointcore.utils import load_json


def fake_invoker(messages: list[dict[str, str]], seed: int) -> dict[str, object]:
    user = messages[-1]["content"]
    if "step=CHECK" in user:
        action = "CHECK"
    elif "step=DECIDE" in user:
        action = "EXECUTE" if "| UNKNOWN" not in user else "HOLD"
    else:
        action = "COMMIT" if "state=approved" in user else "HALT"
    return {"completion": '{"action":"' + action + '"}', "input_tokens": 12, "output_tokens": 5, "generation_seconds": 0.01}


class StatefulTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_json(IDEA_ROOT / "configs/experiments/confirmation.json")

    def test_fenced_and_bare_json_are_registered_but_extra_text_is_not(self) -> None:
        self.assertTrue(parse_action('{"action":"CHECK"}', ("CHECK",))["ok"])
        self.assertTrue(parse_action('```json\n{"action":"CHECK"}\n```', ("CHECK",))["ok"])
        self.assertFalse(parse_action('answer: {"action":"CHECK"}', ("CHECK",))["ok"])

    def test_stateful_trajectory_has_fixed_snapshot_and_exact_steps(self) -> None:
        task = build_stateful_task("orders", "synergy_and", 8, 2, 2, 7201, 3, self.config["costs"])
        trajectory = run_stateful_trajectory(task, task.full_memory, fake_invoker, 1)
        self.assertEqual(trajectory["snapshot_id"], task.snapshot_id)
        self.assertEqual(trajectory["task_success"], 1)
        self.assertEqual(trajectory["parser_failures"], 0)
        self.assertEqual(len(trajectory["steps"]), 3)

    def test_multiple_minimal_family_is_not_collapsed(self) -> None:
        task = build_stateful_task("access_control", "multiple_minimal_sets", 8, 4, 2, 7202, 3, self.config["costs"])
        self.assertEqual(len(oracle_minimal_sets(task)), 2)

    def test_final_prompt_is_state_specific_and_excludes_decide_actions(self) -> None:
        task = build_stateful_task("orders", "single_variable", 8, 2, 1, 7201, 3, self.config["costs"], prompt_protocol_version=2)
        approved = render_messages(task, task.full_memory, 2, "approved")[-1]["content"]
        blocked = render_messages(task, task.full_memory, 2, "blocked")[-1]["content"]
        self.assertIn("Allowed actions now: COMMIT.", approved)
        self.assertIn('Return exactly {"action":"COMMIT"}', approved)
        self.assertIn("Allowed actions now: HALT.", blocked)
        self.assertIn('Return exactly {"action":"HALT"}', blocked)
        self.assertIn("Do not repeat CHECK, EXECUTE, or HOLD", approved)

    def test_compact_final_prompt_keeps_terminal_action_explicit(self) -> None:
        task = build_stateful_task("orders", "single_variable", 8, 2, 1, 7201, 3, self.config["costs"], prompt_protocol_version=3)
        blocked = render_messages(task, task.full_memory, 2, "blocked")
        self.assertEqual(blocked[0]["content"], "JSON only.")
        self.assertIn('NOW FINAL state=blocked: {"action":"HALT"} only', blocked[-1]["content"])
        self.assertIn("never repeat a DECIDE action", blocked[-1]["content"])

    def test_terminal_suffix_is_last_for_protocol_v4(self) -> None:
        task = build_stateful_task("orders", "single_variable", 8, 2, 1, 7201, 3, self.config["costs"], prompt_protocol_version=4)
        blocked = render_messages(task, task.full_memory, 2, "blocked")
        self.assertEqual(blocked[0]["content"], "JSON only.")
        self.assertTrue(blocked[-1]["content"].endswith('Final answer only: {"action":"HALT"}.'))
        self.assertNotIn("NOW DECIDE", blocked[-1]["content"])


if __name__ == "__main__":
    unittest.main()
