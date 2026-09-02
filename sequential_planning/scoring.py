"""
Step 7: Scoring.

Two layers:
  - score_case(): scores ONE test case against both arms' outputs, and
    tags it with metadata useful for breakdowns (horizon type, how close
    the two actions' Q-values were at the optimum, parse failures,
    extraction validity).
  - aggregate_scores(): rolls up a list of per-case score dicts into
    overall + breakdown accuracy stats for both arms.

Design choices, stated explicitly so they're auditable:
  - A parse failure (action is None) counts as INCORRECT for accuracy
    purposes, but is also tracked separately so it's never silently
    conflated with a confident-but-wrong answer.
  - "Near-threshold" cases (where |Q(Cautious) - Q(Aggressive)| at the
    optimum is small) are flagged as a separate breakdown bucket -- this
    is the sequential-planning analogue of the repeated-game module's
    delta-proximity-to-delta* bucket: the hardest cases for an arm to get
    right even with perfect extraction, since small numeric errors in the
    dynamic-programming solve can flip the answer.
  - Parameter-extraction accuracy is tracked SEPARATELY from action
    accuracy, for the same reason as the repeated-game module: a
    translator arm can get the action right on a garbled extraction by
    luck (especially in finite-horizon cases where the correct action can
    be robust to some parameter noise), so action accuracy alone can be a
    misleading proxy for whether the model actually read the paragraph
    correctly.
  - All comparisons are direct string/value equality against the ground
    truth computed once, upstream, at generation time. No judgment calls.
"""

from typing import Optional

from generator import Scenario

NEAR_THRESHOLD_EPSILON = 0.75  # |Q(Cautious) - Q(Aggressive)| below this counts as "near-threshold"


def _horizon_type(scenario: Scenario) -> str:
    return "unknown" if scenario.horizon == "unknown" else "finite"


def _near_threshold_info(margin: Optional[float]) -> tuple:
    """Returns (near_threshold: bool, margin: float or None)."""
    if margin is None:
        return False, None
    return margin < NEAR_THRESHOLD_EPSILON, margin


def _params_extraction_correct(scenario: Scenario, extracted_params) -> Optional[bool]:
    """
    Returns:
      True  -- all eight extracted numbers (four rewards, four transition
               percentages) exactly match the true scenario
      False -- extraction succeeded structurally but at least one number is wrong
      None  -- extraction failed entirely (no extracted_params to compare),
               kept distinct from False so a hard parse failure isn't
               silently counted as an "attempted but wrong" extraction

    Uses the self-describing field names, matching
    translator_solver.ExtractedParams. Reward fields are compared against
    `scenario_value * 1000`: the paragraph states rewards as "N thousand
    dollars", the Scenario stores the raw N, but ExtractedParams now asks
    for (and a live run confirmed the model reliably returns) the full
    dollar amount -- see ExtractedParams' docstring for why that
    convention was chosen over trying to instruct the model to divide by
    1000 itself.
    """
    if extracted_params is None:
        return None
    try:
        return (
            extracted_params.low_state_cautious_reward == scenario.r_low_cautious * 1000 and
            extracted_params.low_state_aggressive_reward == scenario.r_low_aggressive * 1000 and
            extracted_params.high_state_cautious_reward == scenario.r_high_cautious * 1000 and
            extracted_params.high_state_aggressive_reward == scenario.r_high_aggressive * 1000 and
            extracted_params.cautious_advance_probability_percent == round(scenario.p_high_low_cautious * 100) and
            extracted_params.aggressive_advance_probability_percent == round(scenario.p_high_low_aggressive * 100) and
            extracted_params.cautious_stay_probability_percent == round(scenario.p_high_high_cautious * 100) and
            extracted_params.aggressive_stay_probability_percent == round(scenario.p_high_high_aggressive * 100)
        )
    except AttributeError:
        return None


def score_case(case: dict, direct_result: dict, translator_result: dict) -> dict:
    """
    case: output of generate_test_case() -- has 'ground_truth', 'scenario', 'solver_detail'.
    direct_result: output of run_direct_ai() -- has 'action'.
    translator_result: output of run_translator_plus_solver() -- has 'action'
        and 'extraction_warnings'.
    """
    ground_truth = case["ground_truth"]
    scenario = case["scenario"]
    margin = case.get("solver_detail", {}).get("margin")

    direct_action = direct_result.get("action")
    translator_action = translator_result.get("action")

    near_threshold, margin_value = _near_threshold_info(margin)

    return {
        "ground_truth": ground_truth,
        "horizon_type": _horizon_type(scenario),
        "near_threshold": near_threshold,
        "margin": margin_value,

        "direct_action": direct_action,
        "direct_parse_failure": direct_action is None,
        "direct_correct": (direct_action == ground_truth) if direct_action is not None else False,

        "translator_action": translator_action,
        "translator_parse_failure": translator_action is None,
        "translator_correct": (translator_action == ground_truth) if translator_action is not None else False,
        "translator_extraction_warnings": translator_result.get("extraction_warnings", []) or [],
        "translator_had_extraction_warnings": bool(translator_result.get("extraction_warnings")),
        "translator_params_extraction_correct": _params_extraction_correct(
            scenario, translator_result.get("extracted_params")
        ),
    }


def _accuracy(scored: list, correct_key: str) -> Optional[float]:
    if not scored:
        return None
    return sum(1 for s in scored if s[correct_key]) / len(scored)


def _parse_failure_rate(scored: list, failure_key: str) -> Optional[float]:
    if not scored:
        return None
    return sum(1 for s in scored if s[failure_key]) / len(scored)


def _rate_among_non_none(scored: list, key: str) -> Optional[float]:
    """
    Fraction of True among entries where scored[key] is not None. Used for
    param-extraction accuracy, where None means "extraction failed
    entirely" and shouldn't be averaged in as if it were a wrong-but-
    attempted extraction.
    """
    values = [s[key] for s in scored if s[key] is not None]
    if not values:
        return None
    return sum(1 for v in values if v) / len(values)


def _breakdown(scored: list, filter_fn) -> dict:
    subset = [s for s in scored if filter_fn(s)]
    return {
        "n": len(subset),
        "direct_accuracy": _accuracy(subset, "direct_correct"),
        "direct_parse_failure_rate": _parse_failure_rate(subset, "direct_parse_failure"),
        "translator_accuracy": _accuracy(subset, "translator_correct"),
        "translator_parse_failure_rate": _parse_failure_rate(subset, "translator_parse_failure"),
        "translator_params_extraction_accuracy": _rate_among_non_none(
            subset, "translator_params_extraction_correct"
        ),
    }


def aggregate_scores(scored_cases: list) -> dict:
    """
    scored_cases: list of dicts from score_case().
    Returns a report dict with overall stats plus breakdowns by horizon
    type, near-threshold status, and extraction-warning status.
    """
    overall = _breakdown(scored_cases, lambda s: True)

    finite = _breakdown(scored_cases, lambda s: s["horizon_type"] == "finite")
    unknown = _breakdown(scored_cases, lambda s: s["horizon_type"] == "unknown")

    near = _breakdown(scored_cases, lambda s: s["near_threshold"])
    far = _breakdown(scored_cases, lambda s: not s["near_threshold"])

    clean_extraction = _breakdown(scored_cases, lambda s: not s["translator_had_extraction_warnings"])
    warned_extraction = _breakdown(scored_cases, lambda s: s["translator_had_extraction_warnings"])

    return {
        "overall": overall,
        "by_horizon_type": {"finite": finite, "unknown": unknown},
        "by_qvalue_margin": {
            "near_threshold (margin < {:.2f})".format(NEAR_THRESHOLD_EPSILON): near,
            "far_from_threshold": far,
        },
        "by_translator_extraction_validity": {
            "no_warnings": clean_extraction,
            "had_warnings": warned_extraction,
        },
    }


def format_report(report: dict) -> str:
    """Pretty-print an aggregate_scores() report for console/log output."""
    lines = []

    def fmt_pct(x):
        return f"{x*100:.1f}%" if x is not None else "n/a"

    def fmt_breakdown(name, b):
        lines.append(f"  {name} (n={b['n']}):")
        lines.append(f"      Direct-AI accuracy:              {fmt_pct(b['direct_accuracy'])}"
                      f"   (parse failures: {fmt_pct(b['direct_parse_failure_rate'])})")
        lines.append(f"      Translator-plus-Solver accuracy: {fmt_pct(b['translator_accuracy'])}"
                      f"   (parse failures: {fmt_pct(b['translator_parse_failure_rate'])})")
        lines.append(f"      Translator PARAM extraction accuracy: "
                      f"{fmt_pct(b['translator_params_extraction_accuracy'])}"
                      f"   <- honest signal, independent of action correctness")

    lines.append("=== OVERALL ===")
    fmt_breakdown("All cases", report["overall"])

    lines.append("")
    lines.append("=== BY HORIZON TYPE ===")
    fmt_breakdown("Finite horizon", report["by_horizon_type"]["finite"])
    fmt_breakdown("Unknown horizon", report["by_horizon_type"]["unknown"])

    lines.append("")
    lines.append("=== BY Q-VALUE MARGIN AT THE OPTIMUM ===")
    for name, b in report["by_qvalue_margin"].items():
        fmt_breakdown(name, b)

    lines.append("")
    lines.append("=== BY TRANSLATOR EXTRACTION VALIDITY ===")
    for name, b in report["by_translator_extraction_validity"].items():
        fmt_breakdown(name, b)

    return "\n".join(lines)


if __name__ == "__main__":
    # Sanity check with hand-constructed mock cases (no network / no LLM
    # calls) so the scoring math itself can be verified in isolation.
    from generator import Scenario
    from translator_solver import ExtractedParams

    mock_cases = []

    # Case 1: finite horizon, both arms correct, translator's extraction is
    # also genuinely correct (not just lucky).
    s1 = Scenario(
        r_low_cautious=5, r_low_aggressive=9, r_high_cautious=11, r_high_aggressive=15,
        p_high_low_cautious=0.6, p_high_low_aggressive=0.2,
        p_high_high_cautious=0.8, p_high_high_aggressive=0.3,
        num_states=2, num_actions=2, start_state="Low", horizon=3, discount_factor=0.5,
    )
    mock_cases.append((
        {"ground_truth": "aggressive", "scenario": s1, "solver_detail": {"margin": 2.1}},
        {"action": "aggressive"},
        {
            "action": "aggressive",
            "extraction_warnings": [],
            "extracted_params": ExtractedParams(
                low_state_cautious_reward=5000, low_state_aggressive_reward=9000,
                high_state_cautious_reward=11000, high_state_aggressive_reward=15000,
                cautious_advance_probability_percent=60, aggressive_advance_probability_percent=20,
                cautious_stay_probability_percent=80, aggressive_stay_probability_percent=30,
                horizon_is_fixed=True, horizon_periods=3, continuation_probability_percent=0,
            ),
        },
    ))

    # Case 2: finite horizon -- action comes out "correct" by luck even
    # though the extraction itself has swapped/garbled numbers, exposing
    # the same blind spot the repeated-game module's scoring.py documents.
    s2 = Scenario(
        r_low_cautious=4, r_low_aggressive=8, r_high_cautious=9, r_high_aggressive=14,
        p_high_low_cautious=0.5, p_high_low_aggressive=0.15,
        p_high_high_cautious=0.75, p_high_high_aggressive=0.25,
        num_states=2, num_actions=2, start_state="Low", horizon=6, discount_factor=0.3,
    )
    mock_cases.append((
        {"ground_truth": "aggressive", "scenario": s2, "solver_detail": {"margin": 1.4}},
        {"action": "aggressive"},
        {
            "action": "aggressive",  # "correct" action, but for the wrong reason
            "extraction_warnings": ["Reward ordering violated: ..."],
            "extracted_params": ExtractedParams(
                low_state_cautious_reward=8000, low_state_aggressive_reward=4000,  # swapped
                high_state_cautious_reward=9000, high_state_aggressive_reward=14000,
                cautious_advance_probability_percent=50, aggressive_advance_probability_percent=15,
                cautious_stay_probability_percent=75, aggressive_stay_probability_percent=25,
                horizon_is_fixed=True, horizon_periods=6, continuation_probability_percent=0,
            ),
        },
    ))

    # Case 3: unknown horizon, near threshold, translator has a genuine
    # parse failure (no extracted_params at all).
    s3 = Scenario(
        r_low_cautious=6, r_low_aggressive=8, r_high_cautious=10, r_high_aggressive=13,
        p_high_low_cautious=0.4, p_high_low_aggressive=0.25,
        p_high_high_cautious=0.6, p_high_high_aggressive=0.35,
        num_states=2, num_actions=2, start_state="Low", horizon="unknown", discount_factor=0.4,
    )
    mock_cases.append((
        {"ground_truth": "cautious", "scenario": s3, "solver_detail": {"margin": 0.2}},
        {"action": "aggressive"},  # wrong -- near-threshold cases are hard
        {"action": None, "extraction_warnings": ["Response could not be parsed."], "extracted_params": None},
    ))

    # Case 4: unknown horizon, far from threshold, both correct, translator
    # extraction also genuinely correct.
    s4 = Scenario(
        r_low_cautious=3, r_low_aggressive=6, r_high_cautious=15, r_high_aggressive=18,
        p_high_low_cautious=0.85, p_high_low_aggressive=0.1,
        p_high_high_cautious=0.9, p_high_high_aggressive=0.2,
        num_states=2, num_actions=2, start_state="Low", horizon="unknown", discount_factor=0.9,
    )
    mock_cases.append((
        {"ground_truth": "cautious", "scenario": s4, "solver_detail": {"margin": 5.0}},
        {"action": "cautious"},
        {
            "action": "cautious",
            "extraction_warnings": [],
            "extracted_params": ExtractedParams(
                low_state_cautious_reward=3000, low_state_aggressive_reward=6000,
                high_state_cautious_reward=15000, high_state_aggressive_reward=18000,
                cautious_advance_probability_percent=85, aggressive_advance_probability_percent=10,
                cautious_stay_probability_percent=90, aggressive_stay_probability_percent=20,
                horizon_is_fixed=False, horizon_periods=0, continuation_probability_percent=90,
            ),
        },
    ))

    scored = [score_case(case, direct, translator) for case, direct, translator in mock_cases]

    print("--- Per-case scores ---")
    for i, s in enumerate(scored):
        print(i, s)
    print()

    report = aggregate_scores(scored)
    print(format_report(report))

    # The point of this demo: case 2 shows translator_correct=True (action
    # matched by luck) alongside translator_params_extraction_correct=False
    # (the extraction itself was wrong). Confirm both.
    assert scored[1]["translator_correct"] is True
    assert scored[1]["translator_params_extraction_correct"] is False
    print("\nConfirmed: case 2 demonstrates action-correct-but-extraction-wrong "
          "is captured correctly by the two separate metrics.")
