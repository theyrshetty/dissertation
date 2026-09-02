"""
Unit tests for the deterministic parts of translator_solver.py: converting
extracted params into a Scenario, internal-consistency validation, and
end-to-end solving. No network calls — ExtractedParams objects are
constructed directly, standing in for what the LLM would have returned.
"""

from translator_solver import (
    ExtractedParams,
    extracted_params_to_scenario,
    validate_extraction,
)
from solver import solve


def test_fixed_horizon_conversion_and_solve():
    extracted = ExtractedParams(
        unilateral_defector_payoff=15, mutual_cooperation_payoff=11, mutual_defection_payoff=6, unilateral_cooperator_payoff=5,
        horizon_is_fixed=True, horizon_rounds=3,
        continuation_probability_percent=None,
    )
    assert validate_extraction(extracted) == []
    scenario = extracted_params_to_scenario(extracted)
    assert scenario.horizon == 3
    result = solve(scenario)
    assert result["action"] == "defect"  # finite horizon -> always defect
    print("PASS: fixed-horizon extraction converts and solves correctly")


def test_unknown_horizon_conversion_and_solve_cooperate():
    # T=9,R=8,P=4 -> delta* = 1/5 = 0.20; 71% continuation -> delta=0.71 >= 0.20 -> cooperate
    extracted = ExtractedParams(
        unilateral_defector_payoff=9, mutual_cooperation_payoff=8, mutual_defection_payoff=4, unilateral_cooperator_payoff=1,
        horizon_is_fixed=False, horizon_rounds=None,
        continuation_probability_percent=71,
    )
    assert validate_extraction(extracted) == []
    scenario = extracted_params_to_scenario(extracted)
    assert scenario.horizon == "unknown"
    assert scenario.discount_factor == 0.71
    result = solve(scenario)
    assert result["action"] == "cooperate"
    print("PASS: unknown-horizon extraction (above delta*) converts and solves to cooperate")


def test_unknown_horizon_conversion_and_solve_defect():
    # T=18,R=14,P=3 -> delta* = 4/15 = 0.2667; 24% continuation -> defect
    extracted = ExtractedParams(
        unilateral_defector_payoff=18, mutual_cooperation_payoff=14, mutual_defection_payoff=3, unilateral_cooperator_payoff=2,
        horizon_is_fixed=False, horizon_rounds=None,
        continuation_probability_percent=24,
    )
    assert validate_extraction(extracted) == []
    scenario = extracted_params_to_scenario(extracted)
    result = solve(scenario)
    assert result["action"] == "defect"
    print("PASS: unknown-horizon extraction (below delta*) converts and solves to defect")


def test_validation_catches_bad_payoff_ordering():
    extracted = ExtractedParams(
        # unilateral_defector < mutual_cooperation, violates required ordering
        unilateral_defector_payoff=5, mutual_cooperation_payoff=8, mutual_defection_payoff=6, unilateral_cooperator_payoff=1,
        horizon_is_fixed=True, horizon_rounds=4,
        continuation_probability_percent=None,
    )
    warnings = validate_extraction(extracted)
    assert len(warnings) == 1
    assert "Payoff ordering violated" in warnings[0]
    print("PASS: validation flags bad payoff ordering")


def test_validation_flags_contradictory_horizon_fixed_but_probability_set():
    # Reproduces the real scenario_id=12 case from a live run: model said
    # horizon_is_fixed=True but also filled in continuation_probability_percent.
    extracted = ExtractedParams(
        unilateral_defector_payoff=20, mutual_cooperation_payoff=19, mutual_defection_payoff=13, unilateral_cooperator_payoff=6,
        horizon_is_fixed=True, horizon_rounds=2,
        continuation_probability_percent=19,  # should be null when horizon_is_fixed=True
    )
    warnings = validate_extraction(extracted)
    assert any("internally inconsistent" in w for w in warnings)
    print("PASS: validation flags horizon_is_fixed=True with continuation_probability_percent also set")


def test_validation_flags_contradictory_horizon_not_fixed_but_rounds_set():
    extracted = ExtractedParams(
        unilateral_defector_payoff=15, mutual_cooperation_payoff=11, mutual_defection_payoff=6, unilateral_cooperator_payoff=5,
        horizon_is_fixed=False, horizon_rounds=4,  # should be null when horizon_is_fixed=False
        continuation_probability_percent=30,
    )
    warnings = validate_extraction(extracted)
    assert any("internally inconsistent" in w for w in warnings)
    print("PASS: validation flags horizon_is_fixed=False with horizon_rounds also set")



def test_validation_catches_missing_horizon_rounds():
    extracted = ExtractedParams(
        unilateral_defector_payoff=10, mutual_cooperation_payoff=8, mutual_defection_payoff=4, unilateral_cooperator_payoff=2,
        horizon_is_fixed=True, horizon_rounds=None,  # missing despite fixed=True
        continuation_probability_percent=None,
    )
    warnings = validate_extraction(extracted)
    assert any("horizon_rounds is missing" in w for w in warnings)
    print("PASS: validation flags missing horizon_rounds when horizon_is_fixed=True")


def test_validation_catches_missing_continuation_percent():
    extracted = ExtractedParams(
        unilateral_defector_payoff=10, mutual_cooperation_payoff=8, mutual_defection_payoff=4, unilateral_cooperator_payoff=2,
        horizon_is_fixed=False, horizon_rounds=None,
        continuation_probability_percent=None,  # missing despite fixed=False
    )
    warnings = validate_extraction(extracted)
    assert any("continuation_probability_percent is missing" in w for w in warnings)
    print("PASS: validation flags missing continuation_probability_percent when horizon_is_fixed=False")


def test_validation_catches_out_of_range_percent():
    extracted = ExtractedParams(
        unilateral_defector_payoff=10, mutual_cooperation_payoff=8, mutual_defection_payoff=4, unilateral_cooperator_payoff=2,
        horizon_is_fixed=False, horizon_rounds=None,
        continuation_probability_percent=150,  # out of [0,100]
    )
    warnings = validate_extraction(extracted)
    assert any("must be in [0,100]" in w for w in warnings)
    print("PASS: validation flags out-of-range continuation_probability_percent")


def test_extraction_matching_true_scenario_reproduces_ground_truth():
    # End-to-end: simulate a "perfect" extraction (LLM got every number right)
    # and confirm it reproduces the exact same ground truth the solver
    # would have produced directly from the true Scenario.
    import random
    from generator import generate_scenario

    rng = random.Random(55)
    for _ in range(50):
        true_scenario = generate_scenario(rng=rng)
        true_result = solve(true_scenario)

        if true_scenario.horizon == "unknown":
            pct = round(true_scenario.discount_factor * 100)
            extracted = ExtractedParams(
                unilateral_defector_payoff=true_scenario.T, mutual_cooperation_payoff=true_scenario.R, mutual_defection_payoff=true_scenario.P, unilateral_cooperator_payoff=true_scenario.S,
                horizon_is_fixed=False, horizon_rounds=None,
                continuation_probability_percent=pct,
            )
        else:
            extracted = ExtractedParams(
                unilateral_defector_payoff=true_scenario.T, mutual_cooperation_payoff=true_scenario.R, mutual_defection_payoff=true_scenario.P, unilateral_cooperator_payoff=true_scenario.S,
                horizon_is_fixed=True, horizon_rounds=true_scenario.horizon,
                continuation_probability_percent=None,
            )

        assert validate_extraction(extracted) == []
        scenario = extracted_params_to_scenario(extracted)
        result = solve(scenario)
        assert result["action"] == true_result["action"], (true_scenario, extracted, result, true_result)

    print("PASS: a perfect extraction always reproduces the true ground truth (50 random scenarios)")


if __name__ == "__main__":
    test_fixed_horizon_conversion_and_solve()
    test_unknown_horizon_conversion_and_solve_cooperate()
    test_unknown_horizon_conversion_and_solve_defect()
    test_validation_catches_bad_payoff_ordering()
    test_validation_catches_missing_horizon_rounds()
    test_validation_catches_missing_continuation_percent()
    test_validation_catches_out_of_range_percent()
    test_validation_flags_contradictory_horizon_fixed_but_probability_set()
    test_validation_flags_contradictory_horizon_not_fixed_but_rounds_set()
    test_extraction_matching_true_scenario_reproduces_ground_truth()
    print("\nAll translator_solver unit tests passed.")