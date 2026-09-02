"""Explicit dependency DAG for the Part B extracted representation."""

from dataclasses import asdict, dataclass
from typing import Any

from part_a_imports import PART_A_DIR  # noqa: F401
from translator_solver import ExtractedParams, extracted_params_to_scenario
from solver import solve


SOURCE_FIELDS = (
    "unilateral_defector_payoff",
    "mutual_cooperation_payoff",
    "mutual_defection_payoff",
    "unilateral_cooperator_payoff",
    "horizon_is_fixed",
    "horizon_rounds",
    "continuation_probability_percent",
)
DERIVED_FIELDS = ("critical_discount_factor", "solved_outcome")

# An edge A -> B means B must be recomputed if A changes.  The graph is
# intentionally small but explicit: payoffs feed delta*, and all decision
# inputs feed the final solved outcome.
EDGES = (
    ("unilateral_defector_payoff", "critical_discount_factor"),
    ("mutual_cooperation_payoff", "critical_discount_factor"),
    ("mutual_defection_payoff", "critical_discount_factor"),
    ("critical_discount_factor", "solved_outcome"),
    ("unilateral_cooperator_payoff", "solved_outcome"),
    ("horizon_is_fixed", "solved_outcome"),
    ("horizon_rounds", "solved_outcome"),
    ("continuation_probability_percent", "solved_outcome"),
)


def downstream_fields(changed_field: str) -> set[str]:
    """Return the transitive downstream closure, excluding the changed input."""
    if changed_field not in SOURCE_FIELDS:
        raise ValueError(f"Unsupported revision field: {changed_field}")
    adjacency: dict[str, set[str]] = {}
    for parent, child in EDGES:
        adjacency.setdefault(parent, set()).add(child)
    reached, pending = set(), list(adjacency.get(changed_field, set()))
    while pending:
        node = pending.pop()
        if node in reached:
            continue
        reached.add(node)
        pending.extend(adjacency.get(node, set()) - reached)
    return reached


def graph_record(extracted: ExtractedParams) -> dict[str, Any]:
    """Store field values and explicit edges in a JSON-serialisable DAG record."""
    values = extracted.model_dump()
    nodes = [{"id": name, "kind": "source", "value": values[name]} for name in SOURCE_FIELDS]
    nodes.extend({"id": name, "kind": "derived", "value": None} for name in DERIVED_FIELDS)
    return {"nodes": nodes, "edges": [{"from": a, "to": b} for a, b in EDGES]}


def params_from_scenario(scenario) -> ExtractedParams:
    """Canonical Part A extraction corresponding to a generated scenario."""
    if scenario.horizon == "unknown":
        return ExtractedParams(
            unilateral_defector_payoff=scenario.T,
            mutual_cooperation_payoff=scenario.R,
            mutual_defection_payoff=scenario.P,
            unilateral_cooperator_payoff=scenario.S,
            horizon_is_fixed=False,
            horizon_rounds=None,
            continuation_probability_percent=round(scenario.discount_factor * 100),
        )
    return ExtractedParams(
        unilateral_defector_payoff=scenario.T,
        mutual_cooperation_payoff=scenario.R,
        mutual_defection_payoff=scenario.P,
        unilateral_cooperator_payoff=scenario.S,
        horizon_is_fixed=True,
        horizon_rounds=scenario.horizon,
        continuation_probability_percent=None,
    )


def structure_record(extracted: ExtractedParams) -> dict[str, Any]:
    """The value object compared for B5 correctness scoring."""
    result = solve(extracted_params_to_scenario(extracted))
    return {
        "extracted_params": extracted.model_dump(),
        "derived": {
            "critical_discount_factor": result["delta_star"],
            "solved_outcome": result["action"],
        },
    }
