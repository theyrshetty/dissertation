"""Generate one-field, valid-PD revisions and render them as follow-ups."""

from dataclasses import replace
import random

from part_a_imports import PART_A_DIR  # noqa: F401
from generator import Scenario


FIELD_TO_SCENARIO_ATTR = {
    "unilateral_defector_payoff": "T",
    "mutual_cooperation_payoff": "R",
    "mutual_defection_payoff": "P",
    "unilateral_cooperator_payoff": "S",
}
FOLLOW_UP_TEMPLATES = {
    "unilateral_defector_payoff": "Correction: if one party defects while the other cooperates, the defector now earns {value} points.",
    "mutual_cooperation_payoff": "Correction: when both parties cooperate, each now earns {value} points.",
    "mutual_defection_payoff": "Correction: when both parties defect, each now earns {value} points.",
    "unilateral_cooperator_payoff": "Correction: if one party cooperates while the other defects, the cooperator now earns {value} points.",
}


def _valid(s: Scenario) -> bool:
    return s.T > s.R > s.P > s.S and 2 * s.R > s.T + s.S


def generate_revision(scenario: Scenario, rng: random.Random | None = None) -> dict:
    """Change exactly one payoff, retaining the original valid-PD constraints."""
    rng = rng or random.Random()
    candidates = []
    for field, attribute in FIELD_TO_SCENARIO_ATTR.items():
        old_value = getattr(scenario, attribute)
        for value in range(1, 21):
            after = replace(scenario, **{attribute: value})
            if value != old_value and _valid(after):
                candidates.append((field, attribute, value, after))
    if not candidates:
        raise RuntimeError("No valid one-field revision is available for this scenario.")
    field, attribute, value, after = rng.choice(candidates)
    return {
        "changed_field": field,
        "old_value": getattr(scenario, attribute),
        "new_value": value,
        "after_scenario": after,
        "follow_up": FOLLOW_UP_TEMPLATES[field].format(value=value),
    }
