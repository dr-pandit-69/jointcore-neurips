"""Isolated JointCore paper-expansion benchmark and model evaluator.

The earlier v2--v5 study and post-hoc pilot remain immutable.  This module
implements the separately registered benchmark in PAPER_EXPANSION_PROTOCOL.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any

from .stateful import (
    StatefulTask,
    build_stateful_task,
    is_task_sufficient,
    oracle_cost,
    oracle_minimal_sets,
    oracle_participants,
    task_from_dict,
    task_to_dict,
)
from .utils import canonical_json, sha256_text, stable_rng, stable_seed


METHODS = (
    "full_memory",
    "random_b",
    "semantic_top_b",
    "independent_leave_one_out",
    "conditional_singleton_cost",
    "jointcore_legacy",
    "jointcore_cost_ordered",
    "exact_oracle",
)
FAMILIES = (
    "single_variable",
    "redundant_or",
    "synergy_and",
    "synergy_xor",
    "threshold",
    "multiple_minimal_sets",
    "proxy_confounded",
    "delayed_fsm",
    "harmful_stale",
)
DOMAINS = ("orders", "access_control", "deployment", "calendar", "database", "file_debug")
ROBUSTNESS_MODES = ("deletion", "typed_mask", "plausible_same_type", "counterfactual")


@dataclass(frozen=True)
class PaperTask:
    """A stateful logical task plus paper-only counterbalancing metadata."""

    task: StatefulTask
    paper_task_id: str
    horizon_label: int
    seed_block: int
    action_when_sufficient: str
    record_order: tuple[int, ...]

    @property
    def action_when_insufficient(self) -> str:
        return "HOLD" if self.action_when_sufficient == "EXECUTE" else "EXECUTE"


ChoiceInvoker = Callable[[list[dict[str, str]], tuple[str, ...], int], dict[str, Any]]


def validate_paper_config(config: dict[str, Any], *, exact: bool = False) -> None:
    if config.get("schema_version") != 1 or config.get("idea_id") != "jointcore":
        raise ValueError("Wrong JointCore paper-expansion configuration")
    expected_stage = "paper_expansion_exact_v1" if exact else {
        "paper_expansion_development_v1", "paper_expansion_development_v2",
        "paper_expansion_development_v3", "paper_expansion_confirmation_v1",
        "paper_expansion_development_v4", "paper_expansion_confirmation_v2",
        "paper_expansion_confirmation_v3", "paper_expansion_confirmation_v4",
        "paper_expansion_model_extension_v1",
        "paper_expansion_reported_cohort_reproduction_v1",
    }
    if exact:
        if config.get("stage") != expected_stage:
            raise ValueError("Wrong exact paper-expansion stage")
        if tuple(config.get("candidate_counts", ())) != (8, 16, 32, 64):
            raise ValueError("Exact candidate-count grid changed")
        if exact_task_count(config) != 6912:
            raise ValueError("Exact task denominator is not 6,912")
    elif config.get("stage") not in expected_stage:
        raise ValueError("Wrong model paper-expansion stage")
    if tuple(config.get("methods", ())) != METHODS:
        raise ValueError("Paper method matrix changed")
    grid = config if exact else config.get("task_grid", {})
    if tuple(grid.get("families", ())) != FAMILIES or tuple(grid.get("domains", ())) != DOMAINS:
        raise ValueError("Paper task family/domain grid changed")
    if tuple(grid.get("support_sizes", ())) != (2, 4) or tuple(grid.get("horizons", ())) != (1, 4, 8, 16):
        raise ValueError("Paper support/horizon grid changed")
    if not exact:
        interface = config.get("interface", {})
        allowed_interfaces = {
            "direct_constrained_choice_mean_log_probability",
            "condition_truth_mean_log_probability",
            "boolean_condition_mean_log_probability",
        }
        if interface.get("name") not in allowed_interfaces:
            raise ValueError("Primary model interface changed")
        if int(interface.get("maximum_input_tokens", 0)) != 768:
            raise ValueError("Frozen input bound changed")
        expected = 72 if "development" in config["stage"] else 432
        if int(grid.get("task_count", 0)) != expected:
            raise ValueError("Model task denominator changed")


def exact_task_count(config: dict[str, Any]) -> int:
    return (
        len(config.get("candidate_counts", ()))
        * len(config.get("support_sizes", ()))
        * len(config.get("horizons", ()))
        * len(config.get("families", ()))
        * len(config.get("domains", ()))
        * len(config.get("seed_blocks", ()))
    )


def _paper_task(
    *, domain: str, family: str, candidate_count: int, support_size: int,
    horizon: int, seed_block: int, seed_base: int, costs: dict[str, Any],
) -> PaperTask:
    condition = f"{domain}|{family}|n={candidate_count}|s={support_size}|h={horizon}|b={seed_block}|{seed_base}"
    seed = int(seed_base) + 1009 * int(seed_block) + stable_seed(condition, "task") % 997
    task = build_stateful_task(
        domain, family, candidate_count, support_size, min(support_size, 4), seed, 3, costs,
        prompt_protocol_version=6,
    )
    order = list(range(candidate_count))
    stable_rng(condition, "record-order").shuffle(order)
    action = "EXECUTE" if stable_seed(condition, "orientation") % 2 == 0 else "HOLD"
    paper_id = sha256_text(canonical_json({"condition": condition, "stateful_task_id": task.task_id, "version": 1}))
    return PaperTask(task, paper_id, int(horizon), int(seed_block), action, tuple(order))


def iter_exact_tasks(config: dict[str, Any]) -> Iterable[PaperTask]:
    validate_paper_config(config, exact=True)
    for domain, family, n, support_size, horizon, block in product(
        config["domains"], config["families"], config["candidate_counts"],
        config["support_sizes"], config["horizons"], config["seed_blocks"],
    ):
        yield _paper_task(
            domain=str(domain), family=str(family), candidate_count=int(n),
            support_size=int(support_size), horizon=int(horizon), seed_block=int(block),
            seed_base=int(config["seed_base"]), costs=config["costs"],
        )


def iter_model_tasks(config: dict[str, Any]) -> Iterable[PaperTask]:
    validate_paper_config(config)
    grid = config["task_grid"]
    cells = list(product(grid["families"], grid["domains"], grid["support_sizes"], grid["seed_blocks"]))
    count = int(grid["task_count"])
    if count > len(cells):
        raise ValueError("Requested task count exceeds the balanced grid")
    if count < len(cells) and grid.get("sampling") == "balanced_by_family":
        per_family = count // len(grid["families"])
        if per_family * len(grid["families"]) != count:
            raise ValueError("Balanced development count must divide evenly across families")
        chosen = []
        for family in grid["families"]:
            family_cells = [cell for cell in cells if cell[0] == family]
            indices = [round(index * (len(family_cells) - 1) / (per_family - 1)) for index in range(per_family)]
            chosen.extend(family_cells[index] for index in indices)
        cells = chosen
    else:
        cells = cells[:count]
    for family, domain, support_size, block in cells:
        yield _paper_task(
            domain=str(domain), family=str(family), candidate_count=int(grid["candidate_count"]),
            support_size=int(support_size), horizon=int(grid["horizons"][int(block) % 4]),
            seed_block=int(block), seed_base=int(grid["seed_base"]), costs=config["costs"],
        )


def paper_task_to_dict(item: PaperTask) -> dict[str, Any]:
    return {
        "paper_task_id": item.paper_task_id,
        "horizon_label": item.horizon_label,
        "seed_block": item.seed_block,
        "action_when_sufficient": item.action_when_sufficient,
        "record_order": list(item.record_order),
        "stateful_task": task_to_dict(item.task),
    }


def paper_task_from_dict(payload: dict[str, Any]) -> PaperTask:
    return PaperTask(
        task_from_dict(dict(payload["stateful_task"])), str(payload["paper_task_id"]),
        int(payload["horizon_label"]), int(payload["seed_block"]),
        str(payload["action_when_sufficient"]), tuple(int(item) for item in payload["record_order"]),
    )


def expected_decision(item: PaperTask, retained: Iterable[int]) -> str:
    return item.action_when_sufficient if is_task_sufficient(item.task, retained) else item.action_when_insufficient


def expected_terminal(decision: str) -> str:
    return "COMMIT" if decision == "EXECUTE" else "HALT"


def _condition_text(item: PaperTask) -> str:
    task, support = item.task, [item.task.variables[index] for index in item.task.support]
    if task.family == "single_variable":
        condition = f"{support[0].type_name}={support[0].value}"
    elif task.family == "redundant_or":
        condition = "at least one approval record is ALLOW"
    elif task.family in {"synergy_and", "proxy_confounded", "delayed_fsm"}:
        condition = " AND ".join(f"{record.type_name}={record.value}" for record in support)
    elif task.family == "synergy_xor":
        names = ", ".join(record.name for record in support)
        condition = f"all flags ({names}) are known and exactly one is ON"
    elif task.family == "threshold":
        condition = f"at least {max(1, (len(support) + 1) // 2)} approval records are ALLOW"
    elif task.family == "multiple_minimal_sets":
        condition = "all channel_a records are VALID OR all channel_b records are VALID"
    elif task.family == "harmful_stale":
        condition = "current authorization is ALLOW AND every support record is VALID AND stale_policy is BLOCK"
    else:
        raise ValueError(task.family)
    return condition


def _logical_rule(item: PaperTask) -> str:
    return (
        f"Choose {item.action_when_sufficient} iff {_condition_text(item)}; otherwise choose "
        f"{item.action_when_insufficient}."
    )


def _altered_value(value: str, mode: str) -> str:
    if mode == "typed_mask":
        return "UNKNOWN"
    if mode == "plausible_same_type":
        return "PENDING"
    if mode == "counterfactual":
        opposites = {"ALLOW": "BLOCK", "BLOCK": "ALLOW", "ON": "OFF", "OFF": "ON"}
        return opposites.get(value, "INVALID")
    raise ValueError(mode)


def render_decision_messages(
    item: PaperTask, retained: Iterable[int], *, intervention_mode: str = "typed_mask", labels: str = "direct",
) -> tuple[list[dict[str, str]], tuple[str, ...], dict[str, str]]:
    if intervention_mode not in ROBUSTNESS_MODES:
        raise ValueError(intervention_mode)
    retained_set = frozenset(retained)
    records: list[str] = []
    for identifier in item.record_order:
        variable = item.task.variables[identifier]
        if identifier not in retained_set and intervention_mode == "deletion":
            continue
        value = variable.value if identifier in retained_set else _altered_value(variable.value, intervention_mode)
        records.append(f"m{identifier}: name={variable.name}; type={variable.type_name}; value={value}")
    if labels == "direct":
        choices, mapping = ("EXECUTE", "HOLD"), {"EXECUTE": "EXECUTE", "HOLD": "HOLD"}
        answer = "Choose EXECUTE or HOLD."
    elif labels == "ab":
        # Mapping is counterbalanced independently of the sufficient-action orientation.
        if stable_seed(item.paper_task_id, "ab-map") % 2 == 0:
            mapping = {"A": "EXECUTE", "B": "HOLD"}
        else:
            mapping = {"A": "HOLD", "B": "EXECUTE"}
        choices = ("A", "B")
        answer = f"Choose A or B. A means {mapping['A']}; B means {mapping['B']}."
    elif labels == "condition":
        choices = ("SATISFIED", "NOT_SATISFIED")
        mapping = {"SATISFIED": item.action_when_sufficient, "NOT_SATISFIED": item.action_when_insufficient}
        answer = (
            "Judge the condition itself: choose SATISFIED if it is true, or NOT_SATISFIED if it is false. "
            f"SATISFIED maps to workflow action {item.action_when_sufficient}; NOT_SATISFIED maps to {item.action_when_insufficient}."
        )
    elif labels == "boolean":
        choices = ("YES", "NO")
        mapping = {"YES": item.action_when_sufficient, "NO": item.action_when_insufficient}
        answer = (
            "Is the stated condition true? Choose YES if it is true, or NO if it is false. "
            f"YES maps to workflow action {item.action_when_sufficient}; NO maps to {item.action_when_insufficient}."
        )
    else:
        raise ValueError(labels)
    user = "\n".join([
        f"Evaluate a {item.task.domain.replace('_', '-')} workflow at horizon {item.horizon_label}.",
        (f"Condition: {_condition_text(item)}." if labels in {"condition", "boolean"} else _logical_rule(item)),
        "UNKNOWN, PENDING, INVALID, and absent records never satisfy a stated requirement.",
        "Records with types not named in the rule are distractors.",
        "Memory records:", *records, answer,
    ])
    return ([
        {"role": "system", "content": "Apply the stated workflow rule exactly and select one allowed label without explanation."},
        {"role": "user", "content": user},
    ], choices, mapping)


def render_terminal_messages(item: PaperTask, decision: str, *, labels: str = "direct") -> tuple[list[dict[str, str]], tuple[str, ...], dict[str, str]]:
    if labels in {"direct", "condition", "boolean"}:
        choices, mapping = ("COMMIT", "HALT"), {"COMMIT": "COMMIT", "HALT": "HALT"}
        answer = "Choose COMMIT or HALT."
    elif labels == "ab":
        if stable_seed(item.paper_task_id, "terminal-ab-map") % 2 == 0:
            mapping = {"A": "COMMIT", "B": "HALT"}
        else:
            mapping = {"A": "HALT", "B": "COMMIT"}
        choices, answer = ("A", "B"), f"Choose A or B. A means {mapping['A']}; B means {mapping['B']}."
    else:
        raise ValueError(labels)
    user = "\n".join([
        f"The workflow decision was {decision}.",
        "COMMIT exactly after EXECUTE; HALT exactly after HOLD.", answer,
    ])
    return ([
        {"role": "system", "content": "Complete the workflow by selecting one allowed terminal label without explanation."},
        {"role": "user", "content": user},
    ], choices, mapping)


def legacy_order(item: PaperTask) -> tuple[int, ...]:
    return tuple(sorted(range(item.task.candidate_count), key=lambda i: (item.task.variables[i].semantic_score, i)))


def cost_ordered_jointcore_order(item: PaperTask, budget_b: int = 4) -> tuple[int, ...]:
    """Pack low-relevance records, then test the likely core expensive-first."""
    ranked = sorted(
        range(item.task.candidate_count),
        key=lambda i: (-item.task.variables[i].semantic_score, i),
    )
    likely = set(ranked[: min(int(budget_b), len(ranked))])
    low = sorted(
        (i for i in range(item.task.candidate_count) if i not in likely),
        key=lambda i: (item.task.variables[i].semantic_score, i),
    )
    high = sorted(likely, key=lambda i: (-item.task.variables[i].token_cost, i))
    return tuple([*low, *high])


def _static_selection(item: PaperTask, method: str, budget_b: int) -> set[int]:
    task = item.task
    if method == "full_memory":
        return set(range(task.candidate_count))
    if method == "random_b":
        order = list(range(task.candidate_count))
        stable_rng(item.paper_task_id, method).shuffle(order)
        return set(order[: min(budget_b, len(order))])
    if method == "semantic_top_b":
        order = sorted(range(task.candidate_count), key=lambda i: (-task.variables[i].semantic_score, i))
        return set(order[: min(budget_b, len(order))])
    if method == "exact_oracle":
        return set(oracle_minimal_sets(task)[0])
    raise ValueError(method)


def _adaptive_group_select(
    initial: set[int], ordered: tuple[int, ...], equivalent_after_deletion: Callable[[set[int]], bool], maximum: int,
) -> tuple[set[int], list[dict[str, Any]], bool]:
    selected = set(initial)
    queue: list[tuple[int, ...]] = [tuple(i for i in ordered if i in selected)]
    traces: list[dict[str, Any]] = []
    while queue and len(traces) < maximum:
        group = tuple(i for i in queue.pop(0) if i in selected)
        if not group:
            continue
        retained = selected - set(group)
        equivalent = bool(equivalent_after_deletion(retained))
        traces.append({"tested_group": list(group), "retained_variables": sorted(retained), "equivalent": equivalent})
        if equivalent:
            selected = retained
        elif len(group) > 1:
            midpoint = len(group) // 2
            queue.extend((group[:midpoint], group[midpoint:]))
    return selected, traces, bool(queue)


def _selection_metrics(item: PaperTask, method: str, selected: set[int], interventions: int, budget_exhausted: bool = False) -> dict[str, Any]:
    task = item.task
    selected_set = frozenset(selected)
    minimum_sets = tuple(frozenset(core) for core in oracle_minimal_sets(task))
    participants = oracle_participants(task)
    selected_cost = sum(task.variables[index].token_cost for index in selected_set)
    full_cost = sum(variable.token_cost for variable in task.variables)
    sufficient = bool(is_task_sufficient(task, selected_set))
    return {
        "method": method,
        "selected_variables": sorted(selected_set),
        "selected_cost_tokens": selected_cost,
        "full_memory_cost_tokens": full_cost,
        "oracle_cost_tokens": oracle_cost(task),
        "active_token_savings": (full_cost - selected_cost) / full_cost,
        "exact_sufficiency": sufficient,
        "valid_minimum_set": selected_set in minimum_sets,
        "critical_recall": len(selected_set & participants) / len(participants) if participants else 1.0,
        "cost_gap": (selected_cost - oracle_cost(task)) / oracle_cost(task) if sufficient else None,
        "intervention_count": int(interventions),
        "budget_exhausted": bool(budget_exhausted),
    }


def run_exact_task(item: PaperTask, config: dict[str, Any]) -> dict[str, Any]:
    """Run all methods against the exact logical evaluator."""
    budget_b = int(config["selection"]["budget_b"])
    maximum = int(config["selection"]["maximum_interventions"])
    task, reference = item.task, expected_decision(item, item.task.full_memory)
    methods: dict[str, dict[str, Any]] = {}
    for method in ("full_memory", "random_b", "semantic_top_b", "exact_oracle"):
        selected = _static_selection(item, method, budget_b)
        methods[method] = _selection_metrics(item, method, selected, 0)

    selected: set[int] = set()
    for variable in range(task.candidate_count):
        retained = set(range(task.candidate_count)) - {variable}
        if expected_decision(item, retained) != reference:
            selected.add(variable)
    methods["independent_leave_one_out"] = _selection_metrics(
        item, "independent_leave_one_out", selected, task.candidate_count,
    )

    selected = set(range(task.candidate_count))
    interventions = 0
    for variable in sorted(selected, key=lambda i: (-task.variables[i].token_cost, i)):
        if interventions >= maximum:
            break
        retained = selected - {variable}
        interventions += 1
        if expected_decision(item, retained) == reference:
            selected = retained
    methods["conditional_singleton_cost"] = _selection_metrics(
        item, "conditional_singleton_cost", selected, interventions, interventions < task.candidate_count,
    )

    for method, order in (
        ("jointcore_legacy", legacy_order(item)),
        ("jointcore_cost_ordered", cost_ordered_jointcore_order(item, budget_b)),
    ):
        selected, traces, exhausted = _adaptive_group_select(
            set(range(task.candidate_count)), order,
            lambda retained: expected_decision(item, retained) == reference,
            maximum,
        )
        methods[method] = _selection_metrics(item, method, selected, len(traces), exhausted)
    return {"task": paper_task_to_dict(item), "methods": {method: methods[method] for method in METHODS}}


def _invoke_decision(
    item: PaperTask, retained: Iterable[int], invoker: ChoiceInvoker, seed: int,
    cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]],
    *, intervention_mode: str = "typed_mask", labels: str = "direct",
) -> dict[str, Any]:
    key = (intervention_mode, labels, tuple(sorted(set(retained))))
    if key in cache:
        result = deepcopy(cache[key])
        result["cache_hit"] = True
        return result
    messages, choices, mapping = render_decision_messages(
        item, key[2], intervention_mode=intervention_mode, labels=labels,
    )
    try:
        generation = invoker(messages, choices, seed)
        label = str(generation.get("completion", "")).strip().upper()
        action = mapping.get(label, "INVALID")
        error = None
    except Exception as exc:
        generation = {"completion": "", "error": f"{type(exc).__name__}: {exc}", "input_tokens": 0, "output_tokens": 0, "generation_seconds": 0.0}
        label, action, error = "INVALID", "INVALID", generation["error"]
    expected = expected_decision(item, key[2])
    result = {
        "retained_variables": list(key[2]), "intervention_mode": intervention_mode, "labels": labels,
        "label": label, "action": action, "expected_action": expected,
        "valid_choice": action in {"EXECUTE", "HOLD"}, "decision_correct": action == expected,
        "generation": generation, "error": error, "cache_hit": False,
    }
    cache[key] = deepcopy(result)
    return result


def _invoke_terminal(item: PaperTask, decision: str, invoker: ChoiceInvoker, seed: int, *, labels: str = "direct") -> dict[str, Any]:
    messages, choices, mapping = render_terminal_messages(item, decision, labels=labels)
    try:
        generation = invoker(messages, choices, seed)
        label = str(generation.get("completion", "")).strip().upper()
        terminal, error = mapping.get(label, "INVALID"), None
    except Exception as exc:
        generation = {"completion": "", "error": f"{type(exc).__name__}: {exc}", "input_tokens": 0, "output_tokens": 0, "generation_seconds": 0.0}
        label, terminal, error = "INVALID", "INVALID", generation["error"]
    expected = expected_terminal(decision)
    return {
        "labels": labels, "label": label, "terminal": terminal, "expected_terminal": expected,
        "valid_choice": terminal in {"COMMIT", "HALT"}, "terminal_correct": terminal == expected,
        "generation": generation, "error": error,
    }


def _equivalent(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return bool(reference["valid_choice"] and candidate["valid_choice"] and reference["action"] == candidate["action"])


def evaluate_retained_trajectory(
    item: PaperTask,
    retained: Iterable[int],
    invoker: ChoiceInvoker,
    seed: int,
    *,
    intervention_mode: str = "typed_mask",
    labels: str = "direct",
) -> dict[str, Any]:
    """Evaluate a fixed core without performing any selection interventions."""
    cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    decision = _invoke_decision(
        item, retained, invoker, seed, cache,
        intervention_mode=intervention_mode, labels=labels,
    )
    terminal = _invoke_terminal(item, decision["action"], invoker, seed + 1, labels=labels)
    return {
        "retained_variables": sorted(set(retained)),
        "intervention_mode": intervention_mode,
        "labels": labels,
        "decision": decision,
        "terminal": terminal,
        "trajectory_success": bool(decision["decision_correct"] and terminal["terminal_correct"]),
    }


def run_model_task(item: PaperTask, config: dict[str, Any], invoker: ChoiceInvoker) -> dict[str, Any]:
    """Run the complete registered method matrix for one model task."""
    budget_b = int(config["selection"]["budget_b"])
    maximum = int(config["selection"]["maximum_interventions"])
    cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    terminal_cache: dict[str, dict[str, Any]] = {}
    labels = str(config.get("interface", {}).get("decision_label_mode", "direct"))
    reference = _invoke_decision(item, item.task.full_memory, invoker, item.task.seed, cache, labels=labels)
    methods: dict[str, dict[str, Any]] = {}

    def finish(method: str, selected: set[int], traces: list[dict[str, Any]], exhausted: bool = False) -> None:
        final = _invoke_decision(item, selected, invoker, item.task.seed + 8000 + len(method), cache, labels=labels)
        terminal_hit = final["action"] in terminal_cache
        if not terminal_hit:
            terminal_cache[final["action"]] = _invoke_terminal(
                item, final["action"], invoker, item.task.seed + 9000 + len(method), labels=labels,
            )
        terminal = deepcopy(terminal_cache[final["action"]])
        terminal["cache_hit"] = terminal_hit
        metrics = _selection_metrics(item, method, selected, len(traces), exhausted)
        metrics.update({
            "behavioral_sufficiency": _equivalent(reference, final),
            "final_decision_correct": bool(final["decision_correct"]),
            "trajectory_success": bool(final["decision_correct"] and terminal["terminal_correct"]),
            "selection_decision_accuracy": (
                sum(int(trace["decision"]["decision_correct"]) for trace in traces) / len(traces) if traces else 1.0
            ),
            "decision_errors": sum(int(not trace["decision"]["valid_choice"]) for trace in traces) + int(not final["valid_choice"]) + int(not terminal["valid_choice"]),
            "final_decision": final, "terminal_decision": terminal, "interventions": traces,
        })
        methods[method] = metrics

    for method in ("full_memory", "random_b", "semantic_top_b", "exact_oracle"):
        finish(method, _static_selection(item, method, budget_b), [])

    selected = set()
    traces: list[dict[str, Any]] = []
    for variable in range(item.task.candidate_count):
        retained = set(range(item.task.candidate_count)) - {variable}
        decision = _invoke_decision(item, retained, invoker, item.task.seed + 101 * (variable + 1), cache, labels=labels)
        traces.append({"tested_variable": variable, "retained_variables": sorted(retained), "decision": decision, "equivalent": _equivalent(reference, decision)})
        if not _equivalent(reference, decision):
            selected.add(variable)
    finish("independent_leave_one_out", selected, traces)

    selected = set(range(item.task.candidate_count))
    traces = []
    cost_order = sorted(selected, key=lambda i: (-item.task.variables[i].token_cost, i))
    for variable in cost_order:
        if len(traces) >= maximum:
            break
        retained = selected - {variable}
        decision = _invoke_decision(item, retained, invoker, item.task.seed + 211 * (len(traces) + 1), cache, labels=labels)
        equivalent = _equivalent(reference, decision)
        traces.append({"tested_variable": variable, "retained_variables": sorted(retained), "decision": decision, "equivalent": equivalent})
        if equivalent:
            selected = retained
    finish("conditional_singleton_cost", selected, traces, len(traces) < len(cost_order))

    for method, order, salt in (
        ("jointcore_legacy", legacy_order(item), 307),
        ("jointcore_cost_ordered", cost_ordered_jointcore_order(item, budget_b), 401),
    ):
        trace_decisions: list[dict[str, Any]] = []

        def test(retained: set[int]) -> bool:
            decision = _invoke_decision(item, retained, invoker, item.task.seed + salt * (len(trace_decisions) + 1), cache, labels=labels)
            equivalent = _equivalent(reference, decision)
            trace_decisions.append({"retained_variables": sorted(retained), "decision": decision, "equivalent": equivalent})
            return equivalent

        selected, structural, exhausted = _adaptive_group_select(
            set(range(item.task.candidate_count)), order, test, maximum,
        )
        for trace, structure in zip(trace_decisions, structural, strict=True):
            trace["tested_group"] = structure["tested_group"]
        finish(method, selected, trace_decisions, exhausted)

    return {
        "task": paper_task_to_dict(item),
        "full_memory_control": reference,
        "full_memory_control_correct": bool(reference["decision_correct"]),
        "methods": {method: methods[method] for method in METHODS},
        "cache_entries": len(cache),
    }


def task_record_checksum(payload: dict[str, Any]) -> str:
    return sha256_text(canonical_json(payload))
