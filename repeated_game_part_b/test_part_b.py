"""Offline tests for B1-B6; no Ollama server is required."""

import random

from part_a_imports import PART_A_DIR  # noqa: F401
from generator import generate_scenario
from dependency_graph import downstream_fields, params_from_scenario, structure_record
from revision_generator import generate_revision
from scoring import aggregate_scores, score_revision


def test_dag_closure_is_explicit_and_transitive():
    assert downstream_fields("unilateral_defector_payoff") == {"critical_discount_factor", "solved_outcome"}
    assert downstream_fields("mutual_cooperation_payoff") == {"critical_discount_factor", "solved_outcome"}
    assert downstream_fields("unilateral_cooperator_payoff") == {"solved_outcome"}


def test_revision_changes_one_field_and_preserves_valid_pd():
    scenario = generate_scenario(rng=random.Random(7))
    revision = generate_revision(scenario, rng=random.Random(9))
    after = revision["after_scenario"]
    changed = sum(getattr(scenario, key) != getattr(after, key) for key in ("T", "R", "P", "S", "horizon", "discount_factor"))
    assert changed == 1
    assert after.T > after.R > after.P > after.S
    assert 2 * after.R > after.T + after.S
    assert str(revision["new_value"]) in revision["follow_up"]


def test_scoring_detects_perfect_and_nonminimal_responses():
    scenario = generate_scenario(rng=random.Random(17))
    revision = generate_revision(scenario, rng=random.Random(18))
    after_params = params_from_scenario(revision["after_scenario"])
    expected = downstream_fields(revision["changed_field"])
    perfect = score_revision(after_params, expected, {"updated_structure": structure_record(after_params),
                                                       "recomputed_fields": list(expected)})
    assert perfect["structure_correct"] and perfect["minimality_correct"]
    excessive = score_revision(after_params, expected, {"updated_structure": structure_record(after_params),
                                                         "recomputed_fields": list(expected | {"solved_outcome", "extra"})})
    assert excessive["structure_correct"] and not excessive["minimality_correct"]
    report = aggregate_scores([perfect, excessive])
    assert report["updated_structure_correctness"] == 1.0
    assert report["exact_minimal_recomputation"] == 0.5


if __name__ == "__main__":
    test_dag_closure_is_explicit_and_transitive()
    test_revision_changes_one_field_and_preserves_valid_pd()
    test_scoring_detects_perfect_and_nonminimal_responses()
    print("All Part B offline tests passed.")
