"""
Step 4: Combine generator + solver + renderer into one function that
produces (paragraph, ground_truth_answer, scenario, solver_detail) for a
single test case. No manual authorship anywhere in this chain.
"""

import random
from typing import Optional

from generator import generate_scenario, Scenario
from solver import solve
from renderer import render_paragraph


def generate_test_case(rng: Optional[random.Random] = None, **generator_kwargs) -> dict:
    """
    Produce one complete test case.

    Returns a dict:
      {
        "paragraph": str,               # natural-language scenario text
        "ground_truth": "cautious"|"aggressive",
        "scenario": Scenario,           # full sampled parameters (for logging/debug)
        "solver_detail": dict,          # solver's action + reason + margin + q_values
      }
    """
    rng = rng or random
    scenario = generate_scenario(rng=rng, **generator_kwargs)
    solver_result = solve(scenario)
    paragraph = render_paragraph(scenario, rng=rng)

    return {
        "paragraph": paragraph,
        "ground_truth": solver_result["action"],
        "scenario": scenario,
        "solver_detail": solver_result,
    }


if __name__ == "__main__":
    rng = random.Random(2024)
    n = 8
    for i in range(n):
        case = generate_test_case(rng=rng)
        print(f"=== Test case {i} ===")
        print(case["paragraph"])
        print()
        print(f"Ground truth: {case['ground_truth']}")
        s = case["scenario"]
        print(f"Raw params:   r_low=({s.r_low_cautious},{s.r_low_aggressive}) "
              f"r_high=({s.r_high_cautious},{s.r_high_aggressive}) "
              f"p_high_low=({s.p_high_low_cautious},{s.p_high_low_aggressive}) "
              f"p_high_high=({s.p_high_high_cautious},{s.p_high_high_aggressive}) "
              f"horizon={s.horizon} discount={s.discount_factor}")
        print(f"margin: {case['solver_detail']['margin']:.4f}")
        print("-" * 80)
        print()
