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
        "ground_truth": "cooperate"|"defect",
        "scenario": Scenario,           # full sampled parameters (for logging/debug)
        "solver_detail": dict,          # solver's action + reason + delta_star
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
        print(f"Raw params:   T={case['scenario'].T} R={case['scenario'].R} "
              f"P={case['scenario'].P} S={case['scenario'].S} "
              f"horizon={case['scenario'].horizon} "
              f"delta={case['scenario'].discount_factor}")
        if case["solver_detail"]["delta_star"] is not None:
            print(f"delta*: {case['solver_detail']['delta_star']:.4f}")
        print("-" * 80)
        print()