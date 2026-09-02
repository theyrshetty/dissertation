"""
Unit tests for the deterministic parts of translator_solver.py: converting
extracted params into a Scenario, internal-consistency validation, and
end-to-end solving. No network calls -- ExtractedParams objects are
constructed directly, standing in for what the LLM would have returned.
"""

from translator_solver import (
    ExtractedParams,
    extracted_params_to_scenario,
    validate_extraction,
)
from solver import solve


def test_fixed_horizon_conversion_and_solve():
    # Same numbers verified directly against solver.py: finite T=3 -> aggressive.
    extracted = ExtractedParams(
        low_state_cautious_reward=5, low_state_aggressive_reward=9,
        high_state_cautious_reward=11, high_state_aggressive_reward=15,
        cautious_advance_probability_percent=60, aggressive_advance_probability_percent=20,
        cautious_stay_probability_percent=80, aggressive_stay_probability_percent=30,
        horizon_is_fixed=True, horizon_periods=3,
        continuation_probability_percent=0,
    )
    assert validate_extraction(extracted) == []
    scenario = extracted_params_to_scenario(extracted)
    assert scenario.horizon == 3
    result = solve(scenario)
    assert result["action"] == "aggressive"
    print("PASS: fixed-horizon extraction converts and solves correctly")


def test_unknown_horizon_conversion_and_solve_cautious():
    # Same numbers verified directly: unknown horizon, gamma=0.9 -> cautious.
    extracted = ExtractedParams(
        low_state_cautious_reward=3, low_state_aggressive_reward=6,
        high_state_cautious_reward=15, high_state_aggressive_reward=18,
        cautious_advance_probability_percent=85, aggressive_advance_probability_percent=10,
        cautious_stay_probability_percent=90, aggressive_stay_probability_percent=20,
        horizon_is_fixed=False, horizon_periods=0,
        continuation_probability_percent=90,
    )
    assert validate_extraction(extracted) == []
    scenario = extracted_params_to_scenario(extracted)
    assert scenario.horizon == "unknown"
    assert scenario.discount_factor == 0.9
    result = solve(scenario)
    assert result["action"] == "cautious"
    print("PASS: unknown-horizon extraction (high gamma) converts and solves to cautious")


def test_unknown_horizon_conversion_and_solve_aggressive():
    # Same numbers verified directly: unknown horizon, gamma=0.05 -> aggressive (myopic).
    extracted = ExtractedParams(
        low_state_cautious_reward=6, low_state_aggressive_reward=20,
        high_state_cautious_reward=10, high_state_aggressive_reward=21,
        cautious_advance_probability_percent=90, aggressive_advance_probability_percent=5,
        cautious_stay_probability_percent=90, aggressive_stay_probability_percent=5,
        horizon_is_fixed=False, horizon_periods=0,
        continuation_probability_percent=5,
    )
    assert validate_extraction(extracted) == []
    scenario = extracted_params_to_scenario(extracted)
    result = solve(scenario)
    assert result["action"] == "aggressive"
    print("PASS: unknown-horizon extraction (low gamma, myopic) converts and solves to aggressive")


def test_validation_catches_bad_reward_ordering():
    extracted = ExtractedParams(
        # low_state_aggressive_reward < low_state_cautious_reward, violates required ordering
        low_state_cautious_reward=8, low_state_aggressive_reward=5,
        high_state_cautious_reward=11, high_state_aggressive_reward=15,
        cautious_advance_probability_percent=60, aggressive_advance_probability_percent=20,
        cautious_stay_probability_percent=80, aggressive_stay_probability_percent=30,
        horizon_is_fixed=True, horizon_periods=4,
        continuation_probability_percent=0,
    )
    warnings = validate_extraction(extracted)
    assert any("low_state_aggressive_reward > low_state_cautious_reward" in w for w in warnings)
    print("PASS: validation flags bad reward ordering")


def test_validation_catches_bad_probability_ordering():
    extracted = ExtractedParams(
        low_state_cautious_reward=5, low_state_aggressive_reward=9,
        high_state_cautious_reward=11, high_state_aggressive_reward=15,
        # cautious_advance should exceed aggressive_advance; here it's reversed
        cautious_advance_probability_percent=15, aggressive_advance_probability_percent=60,
        cautious_stay_probability_percent=80, aggressive_stay_probability_percent=30,
        horizon_is_fixed=True, horizon_periods=4,
        continuation_probability_percent=0,
    )
    warnings = validate_extraction(extracted)
    assert any("cautious_advance_probability_percent > aggressive_advance_probability_percent" in w for w in warnings)
    print("PASS: validation flags bad probability ordering")


def test_validation_flags_contradictory_horizon_fixed_but_probability_set():
    extracted = ExtractedParams(
        low_state_cautious_reward=5, low_state_aggressive_reward=9,
        high_state_cautious_reward=11, high_state_aggressive_reward=15,
        cautious_advance_probability_percent=60, aggressive_advance_probability_percent=20,
        cautious_stay_probability_percent=80, aggressive_stay_probability_percent=30,
        horizon_is_fixed=True, horizon_periods=2,
        continuation_probability_percent=19,  # should be the 0 sentinel when horizon_is_fixed=True
    )
    warnings = validate_extraction(extracted)
    assert any("continuation_probability_percent was also" in w for w in warnings)
    print("PASS: validation flags contradictory horizon_is_fixed + continuation_probability_percent")


def test_validation_flags_out_of_range_percent():
    extracted = ExtractedParams(
        low_state_cautious_reward=5, low_state_aggressive_reward=9,
        high_state_cautious_reward=11, high_state_aggressive_reward=15,
        cautious_advance_probability_percent=60, aggressive_advance_probability_percent=20,
        cautious_stay_probability_percent=130, aggressive_stay_probability_percent=30,  # out of range
        horizon_is_fixed=True, horizon_periods=4,
        continuation_probability_percent=0,
    )
    warnings = validate_extraction(extracted)
    assert any("cautious_stay_probability_percent must be in [0,100]" in w for w in warnings)
    print("PASS: validation flags an out-of-[0,100]-range percent field")


if __name__ == "__main__":
    test_fixed_horizon_conversion_and_solve()
    test_unknown_horizon_conversion_and_solve_cautious()
    test_unknown_horizon_conversion_and_solve_aggressive()
    test_validation_catches_bad_reward_ordering()
    test_validation_catches_bad_probability_ordering()
    test_validation_flags_contradictory_horizon_fixed_but_probability_set()
    test_validation_flags_out_of_range_percent()
    print("\nAll translator_solver_test checks passed.")
