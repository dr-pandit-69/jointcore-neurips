"""Small exact stateful workflows with natural-language memory renderings.

The actor is an LLM only at invocation time.  Task dynamics, snapshot resets,
valid sufficient-set families, state transitions, and task success are exact.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Iterable

from .utils import canonical_json, sha256_text, stable_rng, unit_interval


@dataclass(frozen=True)
class StatefulVariable:
    identifier: int
    name: str
    type_name: str
    value: str
    token_cost: int
    semantic_score: float


@dataclass(frozen=True)
class StatefulTask:
    task_id: str
    domain: str
    family: str
    candidate_count: int
    essential_set_size: int
    interaction_order: int
    seed: int
    horizon: int
    support: tuple[int, ...]
    proxy_variables: tuple[int, ...]
    harmful_variables: tuple[int, ...]
    variables: tuple[StatefulVariable, ...]
    snapshot_id: str
    prompt_protocol_version: int = 1

    @property
    def full_memory(self) -> frozenset[int]:
        return frozenset(range(self.candidate_count))

    @property
    def token_costs(self) -> tuple[int, ...]:
        return tuple(variable.token_cost for variable in self.variables)


def _support_size(family: str, requested: int) -> int:
    return 1 if family == "single_variable" else requested


def _domain_prefix(domain: str) -> str:
    return {"orders": "ORD", "access_control": "ACL", "deployment": "DEP", "calendar": "CAL", "database": "DB", "file_debug": "FIX"}[domain]


def _support_names(family: str, count: int) -> list[tuple[str, str, str]]:
    if family == "single_variable":
        return [("authorization", "authorization", "ALLOW")]
    if family == "redundant_or":
        return [(f"approval_{index + 1}", "approval", "ALLOW") for index in range(count)]
    if family in {"synergy_and", "proxy_confounded", "delayed_fsm"}:
        base = [
            ("case_id", "case_id", "MATCH"),
            ("current_status", "current_status", "CONFIRMED"),
            ("policy_version", "policy_version", "CURRENT"),
            ("approval", "approval", "ALLOW"),
            ("completion", "completion", "VERIFIED"),
            ("constraint", "constraint", "SATISFIED"),
            ("authority", "authority", "VALID"),
            ("receipt", "receipt", "PRESENT"),
        ]
        return base[:count]
    if family == "synergy_xor":
        return [(f"key_flag_{index + 1}", "key_flag", "ON" if index == 0 else "OFF") for index in range(count)]
    if family == "threshold":
        return [(f"approval_{index + 1}", "approval", "ALLOW") for index in range(count)]
    if family == "multiple_minimal_sets":
        if count <= 1:
            return [("fallback_authorization", "authorization", "ALLOW")]
        half = count // 2
        values: list[tuple[str, str, str]] = []
        for index in range(half):
            values.append((f"channel_a_{index + 1}", "channel_a", "VALID"))
        for index in range(count - half):
            values.append((f"channel_b_{index + 1}", "channel_b", "VALID"))
        return values
    if family == "harmful_stale":
        base = [("current_authorization", "authorization", "ALLOW")]
        base.extend((f"support_{index + 1}", "support", "VALID") for index in range(max(0, count - 2)))
        base.append(("stale_policy", "stale_policy", "BLOCK"))
        return base[:count]
    raise ValueError(family)


def _rule_text(task: StatefulTask) -> str:
    family = task.family
    if family == "single_variable":
        condition = "the authorization record is ALLOW"
    elif family == "redundant_or":
        condition = "at least one approval record is ALLOW"
    elif family in {"synergy_and", "proxy_confounded"}:
        condition = "every required case/status/policy record is present with its shown valid value"
    elif family == "synergy_xor":
        condition = "every key_flag record is known and exactly one key_flag is ON"
    elif family == "threshold":
        threshold = max(1, (len(task.support) + 1) // 2)
        condition = f"at least {threshold} approval records are ALLOW"
    elif family == "multiple_minimal_sets":
        condition = "every record in channel_a is valid OR every record in channel_b is valid"
    elif family == "delayed_fsm":
        condition = "every required case/status/policy record is present with its shown valid value"
    elif family == "harmful_stale":
        condition = "current authorization is ALLOW; however a visible stale_policy BLOCK directs HOLD"
    else:
        raise ValueError(family)
    return (
        "At CHECK, return CHECK. At DECIDE, return EXECUTE exactly when " + condition
        + "; otherwise return HOLD. At FINAL, return COMMIT after an approved decision and HALT after a blocked decision."
    )


def _task_title(domain: str) -> str:
    return {
        "orders": "order workflow",
        "access_control": "access-control workflow",
        "deployment": "deployment workflow",
        "calendar": "calendar workflow",
        "database": "database workflow",
        "file_debug": "file-debug workflow",
    }[domain]


def build_stateful_task(
    domain: str,
    family: str,
    candidate_count: int,
    essential_set_size: int,
    interaction_order: int,
    seed: int,
    horizon: int,
    costs: dict[str, Any],
    prompt_protocol_version: int = 1,
) -> StatefulTask:
    if candidate_count < 2 or horizon != 3:
        raise ValueError("Stateful pilot uses at least two variables and exactly three steps")
    support_count = min(candidate_count, _support_size(family, essential_set_size))
    key = f"{domain}|{family}|n={candidate_count}|s={essential_set_size}|d={interaction_order}|seed={seed}"
    order = list(range(candidate_count))
    stable_rng(key, "support-order").shuffle(order)
    support = tuple(sorted(order[:support_count]))
    support_names = _support_names(family, support_count)
    support_by_identifier = {identifier: support_names[index] for index, identifier in enumerate(support)}
    remaining = [identifier for identifier in range(candidate_count) if identifier not in support]
    proxy = (remaining[0],) if family == "proxy_confounded" and remaining else ()
    harmful = (support[-1],) if family == "harmful_stale" else ()
    variables: list[StatefulVariable] = []
    equal_support = family in {"redundant_or", "multiple_minimal_sets"}
    support_cost = int(costs["support_token_cost_min"]) + seed % (int(costs["support_token_cost_max"]) - int(costs["support_token_cost_min"]) + 1)
    for identifier in range(candidate_count):
        if identifier in support_by_identifier:
            name, type_name, value = support_by_identifier[identifier]
            cost = support_cost if equal_support else int(costs["support_token_cost_min"]) + int(unit_interval(key, "support-cost", identifier) * (int(costs["support_token_cost_max"]) - int(costs["support_token_cost_min"]) + 1))
            score = 0.76 + 0.18 * unit_interval(key, "support-score", identifier)
        elif identifier in proxy:
            name, type_name, value = "historical_priority_proxy", "proxy", "URGENT"
            cost, score = 5, 0.995
        else:
            name, type_name, value = f"context_note_{identifier + 1}", "context_note", f"NOTE_{identifier + 1}"
            cost = int(costs["distractor_token_cost_min"]) + int(unit_interval(key, "distractor-cost", identifier) * (int(costs["distractor_token_cost_max"]) - int(costs["distractor_token_cost_min"]) + 1))
            score = 0.02 + 0.70 * unit_interval(key, "distractor-score", identifier)
        variables.append(StatefulVariable(identifier, name, type_name, value, cost, round(score, 8)))
    task_id = sha256_text(canonical_json({"task": key, "version": 1}))
    return StatefulTask(
        task_id, domain, family, candidate_count, essential_set_size, interaction_order,
        seed, horizon, support, proxy, harmful, tuple(variables),
        sha256_text(f"snapshot|{key}"), int(prompt_protocol_version),
    )


def iter_stateful_tasks(config: dict[str, Any]) -> Iterable[StatefulTask]:
    spec = config["stateful_llm"]
    families = list(spec["outcome_families"])
    domains = list(spec["domains"])
    sizes = list(spec["essential_set_sizes"])
    orders = list(spec["interaction_orders"])
    seeds = list(spec["seeds"])
    for index in range(int(spec["task_count"])):
        yield build_stateful_task(
            domains[index % len(domains)],
            families[index % len(families)],
            int(spec["candidate_count"]),
            int(sizes[index % len(sizes)]),
            int(orders[index % len(orders)]),
            int(seeds[index % len(seeds)]) + 100 * (index // len(seeds)),
            int(spec["horizon"]),
            config["costs"],
            int(spec.get("prompt_protocol_version", 1)),
        )


def task_to_dict(task: StatefulTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "family": task.family,
        "candidate_count": task.candidate_count,
        "essential_set_size": task.essential_set_size,
        "interaction_order": task.interaction_order,
        "seed": task.seed,
        "horizon": task.horizon,
        "support": list(task.support),
        "proxy_variables": list(task.proxy_variables),
        "harmful_variables": list(task.harmful_variables),
        "snapshot_id": task.snapshot_id,
        "prompt_protocol_version": task.prompt_protocol_version,
        "variables": [variable.__dict__ for variable in task.variables],
    }


def task_from_dict(payload: dict[str, Any]) -> StatefulTask:
    variables = tuple(StatefulVariable(**dict(item)) for item in payload["variables"])
    return StatefulTask(
        str(payload["task_id"]), str(payload["domain"]), str(payload["family"]), int(payload["candidate_count"]),
        int(payload["essential_set_size"]), int(payload["interaction_order"]), int(payload["seed"]), int(payload["horizon"]),
        tuple(int(item) for item in payload["support"]), tuple(int(item) for item in payload["proxy_variables"]),
        tuple(int(item) for item in payload["harmful_variables"]), variables, str(payload["snapshot_id"]),
        int(payload.get("prompt_protocol_version", 1)),
    )


def is_task_sufficient(task: StatefulTask, retained: Iterable[int]) -> bool:
    available = frozenset(retained)
    support = frozenset(task.support)
    if task.family == "single_variable":
        return task.support[0] in available
    if task.family == "redundant_or":
        return bool(support & available)
    if task.family in {"synergy_and", "synergy_xor", "proxy_confounded", "delayed_fsm", "harmful_stale"}:
        return support <= available
    if task.family == "threshold":
        return len(support & available) >= max(1, (len(support) + 1) // 2)
    if task.family == "multiple_minimal_sets":
        if len(task.support) <= 1:
            return bool(support & available)
        midpoint = len(task.support) // 2
        return frozenset(task.support[:midpoint]) <= available or frozenset(task.support[midpoint:]) <= available
    raise ValueError(task.family)


def oracle_minimal_sets(task: StatefulTask) -> tuple[tuple[int, ...], ...]:
    if task.family == "single_variable":
        candidates = [(task.support[0],)]
    elif task.family == "redundant_or":
        candidates = [(item,) for item in task.support]
    elif task.family in {"synergy_and", "synergy_xor", "proxy_confounded", "delayed_fsm", "harmful_stale"}:
        candidates = [task.support]
    elif task.family == "threshold":
        candidates = list(combinations(task.support, max(1, (len(task.support) + 1) // 2)))
    elif task.family == "multiple_minimal_sets":
        midpoint = len(task.support) // 2
        candidates = [task.support[:midpoint], task.support[midpoint:]] if midpoint else [task.support]
    else:
        raise ValueError(task.family)
    costs = [sum(task.variables[item].token_cost for item in candidate) for candidate in candidates]
    minimum = min(costs)
    return tuple(sorted(tuple(sorted(candidate)) for candidate, cost in zip(candidates, costs, strict=True) if cost == minimum))


def oracle_participants(task: StatefulTask) -> frozenset[int]:
    return frozenset(item for candidate in oracle_minimal_sets(task) for item in candidate)


def oracle_cost(task: StatefulTask) -> int:
    return sum(task.variables[item].token_cost for item in oracle_minimal_sets(task)[0])


def screened_candidates(task: StatefulTask) -> tuple[int, ...]:
    return tuple(sorted(range(task.candidate_count), key=lambda item: (task.variables[item].semantic_score, item)))


def _stage_name(stage: int) -> str:
    return ("CHECK", "DECIDE", "FINAL")[stage]


def _allowed_actions(stage: int, state: str | None = None) -> tuple[str, ...]:
    if stage == 0:
        return ("CHECK",)
    if stage == 1:
        return ("EXECUTE", "HOLD")
    if state == "approved":
        return ("COMMIT",)
    if state == "blocked":
        return ("HALT",)
    return ("COMMIT", "HALT")


def expected_action(task: StatefulTask, stage: int, state: str) -> str:
    if stage == 0:
        return "CHECK"
    if stage == 1:
        if task.family == "delayed_fsm":
            return "EXECUTE"
        return "EXECUTE"
    return "COMMIT" if state == "approved" else "HALT"


def transition(task: StatefulTask, stage: int, state: str, action: str) -> tuple[str, bool]:
    expected = expected_action(task, stage, state)
    if stage == 0:
        return ("checked", action == expected)
    if stage == 1:
        return ("approved" if action == expected else "blocked", action == expected)
    return ("success" if action == expected and state == "approved" else "failed", action == expected and state == "approved")


def render_messages(task: StatefulTask, retained: Iterable[int], stage: int, state: str) -> list[dict[str, str]]:
    retained_set = frozenset(retained)
    records = []
    for variable in task.variables:
        value = variable.value if variable.identifier in retained_set else "UNKNOWN"
        records.append(f"- m{variable.identifier} | {variable.type_name} | {value}")
    if task.prompt_protocol_version == 2:
        allowed = _allowed_actions(stage, state)
        if stage == 0:
            step_contract = 'CURRENT STEP ONLY (CHECK): return exactly {"action":"CHECK"}. Do not choose a DECIDE or FINAL action.'
        elif stage == 1:
            step_contract = 'CURRENT STEP ONLY (DECIDE): use the rule and return exactly one of {"action":"EXECUTE"} or {"action":"HOLD"}. Do not repeat CHECK or choose a FINAL action.'
        else:
            terminal = allowed[0]
            step_contract = (
                f'CURRENT STEP ONLY (FINAL): the previous decision state is {state}. '
                f'Return exactly {{"action":"{terminal}"}}. Do not repeat CHECK, EXECUTE, or HOLD, and do not re-evaluate the DECIDE rule.'
            )
        lines = [
            f'You control a {_task_title(task.domain)}. Reply with exactly one JSON object and no other text: {{"action":"ACTION"}}.',
            f"Allowed actions now: {', '.join(allowed)}.",
            f"External state snapshot: {task.snapshot_id}; state={state}; step={_stage_name(stage)}.",
            step_contract,
            f"Rule: {_rule_text(task)}",
            "Memory records (UNKNOWN is a schema-valid missing value):",
            *records,
        ]
        system = "Return only the current-stage JSON action. Do not explain or repeat an earlier stage."
    elif task.prompt_protocol_version >= 3:
        allowed = _allowed_actions(stage, state)
        if task.prompt_protocol_version == 3:
            if stage == 0:
                current = 'NOW CHECK: {"action":"CHECK"} only.'
            elif stage == 1:
                current = 'NOW DECIDE: {"action":"EXECUTE"} or {"action":"HOLD"} only.'
            else:
                terminal = allowed[0]
                current = f'NOW FINAL state={state}: {{"action":"{terminal}"}} only; never repeat a DECIDE action.'
            lines = [
                f'Control {_task_title(task.domain)}. JSON only: {{"action":"ACTION"}}.',
                current,
                f"Snapshot={task.snapshot_id}; state={state}; step={_stage_name(stage)}.",
                f"Rule: {_rule_text(task)}",
                "Memory (UNKNOWN is missing):",
                *records,
            ]
        else:
            lines = [
                f'Reply JSON only: {{"action":"ACTION"}}.',
                f"Allowed: {', '.join(allowed)}.",
                f"Snapshot={task.snapshot_id}; state={state}; step={_stage_name(stage)}.",
                f"Rule: {_rule_text(task)}",
                "Memory (UNKNOWN is missing):",
                *records,
            ]
            if stage == 2:
                lines.append(f'Final answer only: {{"action":"{allowed[0]}"}}.')
        system = "JSON only."
    else:
        allowed = _allowed_actions(stage)
        lines = [
            f'You control a {_task_title(task.domain)}. Reply with exactly one JSON object and no other text: {{"action":"ACTION"}}.',
            f"Allowed actions now: {', '.join(allowed)}.",
            f"External state snapshot: {task.snapshot_id}; state={state}; step={_stage_name(stage)}.",
            f"Rule: {_rule_text(task)}",
            "Memory records (UNKNOWN is a schema-valid missing value):",
            *records,
        ]
        system = "Return only the requested JSON action. Do not explain."
    user = "\n".join(lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_action(completion: str, allowed_actions: Iterable[str]) -> dict[str, Any]:
    """Strict structural parser with a registered fenced-JSON compatibility path."""
    text = completion.strip()
    candidates: list[tuple[str, str]] = []
    if text.startswith("{") and text.endswith("}"):
        candidates.append(("bare_json_object", text))
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(("markdown_fenced_json_object", fenced.group(1)))
    if len(candidates) != 1:
        return {"ok": False, "parser_status": "no_registered_json_structure", "action": "INVALID"}
    structure, candidate = candidates[0]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        return {"ok": False, "parser_status": "json_decode_error", "action": "INVALID", "parser_error": str(error)}
    if not isinstance(payload, dict) or set(payload) != {"action"} or not isinstance(payload.get("action"), str):
        return {"ok": False, "parser_status": "schema_error", "action": "INVALID"}
    action = payload["action"].strip().upper()
    if action not in set(allowed_actions):
        return {"ok": False, "parser_status": "invalid_action", "action": "INVALID", "parsed_action": action}
    return {"ok": True, "parser_status": structure, "action": action}


Invoker = Callable[[list[dict[str, str]], int], dict[str, Any]]


def run_stateful_trajectory(task: StatefulTask, retained: Iterable[int], invoker: Invoker, seed: int) -> dict[str, Any]:
    """Run a fully reset, finite-horizon actor/environment trajectory."""
    retained_set = frozenset(retained)
    state = "start"
    steps: list[dict[str, Any]] = []
    parser_failures = 0
    total_input_tokens = total_output_tokens = 0
    total_generation_seconds = 0.0
    for stage in range(task.horizon):
        messages = render_messages(task, retained_set, stage, state)
        try:
            generation = invoker(messages, seed + stage)
        except Exception as error:  # Preserve an infrastructure/parser path in raw output rather than killing a task.
            generation = {"error": f"{type(error).__name__}: {error}", "completion": "", "input_tokens": 0, "output_tokens": 0, "generation_seconds": 0.0}
        completion = str(generation.get("completion", ""))
        allowed = _allowed_actions(stage, state) if task.prompt_protocol_version >= 2 else _allowed_actions(stage)
        parsed = parse_action(completion, allowed) if "error" not in generation else {"ok": False, "parser_status": "runtime_error", "action": "INVALID", "runtime_error": generation["error"]}
        action = str(parsed["action"])
        previous_state = state
        state, success = transition(task, stage, state, action)
        parser_failures += int(not bool(parsed["ok"]))
        total_input_tokens += int(generation.get("input_tokens", 0) or 0)
        total_output_tokens += int(generation.get("output_tokens", 0) or 0)
        total_generation_seconds += float(generation.get("generation_seconds", 0.0) or 0.0)
        steps.append({
            "stage": stage,
            "state_before": previous_state,
            "expected_action": expected_action(task, stage, previous_state),
            "action": action,
            "state_after": state,
            "step_success": success,
            "parser": parsed,
            "generation": generation,
        })
    signature = tuple(f"{item['stage']}:{item['action']}:{item['state_after']}" for item in steps)
    return {
        "retained_variables": sorted(retained_set),
        "snapshot_id": task.snapshot_id,
        "steps": steps,
        "signature": list(signature),
        "parser_failures": parser_failures,
        "task_success": int(state == "success"),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "generation_seconds": total_generation_seconds,
    }
