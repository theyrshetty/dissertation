"""
Step 7: Scoring.

Two layers:
  - score_case(): scores ONE test case against both arms' outputs, and
    tags it with metadata useful for breakdowns (horizon type, how close
    delta was to the grim-trigger threshold, parse failures, extraction
    validity).
  - aggregate_scores(): rolls up a list of per-case score dicts into
    overall + breakdown accuracy stats for both arms.

Design choices, stated explicitly so they're auditable:
  - A parse failure (action is None, i.e. we couldn't extract a clean
    cooperate/defect from the model) counts as INCORRECT for accuracy
    purposes, but is also tracked separately so it's never silently
    conflated with a confident-but-wrong answer. If parse failures are
    common for one arm, that's itself a finding worth reporting, not
    noise to average away.
  - "Near-threshold" unknown-horizon cases (where the sampled discount
    factor is close to the grim-trigger critical value delta*) are flagged
    as a separate breakdown bucket, since these are the hardest cases for
    an arm to get right even with perfect extraction (small numeric errors
    in delta can flip the answer) — useful to see if errors concentrate here.
  - Payoff-extraction accuracy is tracked SEPARATELY from action accuracy.
    This matters because action accuracy alone is misleading for
    finite-horizon scenarios: the solver returns "defect" unconditionally
    for any finite horizon regardless of the extracted payoffs, so a
    translator arm can get the action right by luck even with completely
    garbled payoffs. Payoff-extraction accuracy asks a narrower, honest
    question: did the four extracted numbers actually match the four true
    numbers, independent of what the solver did with them afterward.
  - All comparisons are direct string/value equality against the ground
    truth computed once, upstream, at generation time. No judgment calls.
"""

from typing import Optional

from generator import Scenario
from solver import critical_discount_factor

NEAR_THRESHOLD_EPSILON = 0.05  # |delta - delta*| below this counts as "near-threshold"


def _horizon_type(scenario: Scenario) -> str:
    return "unknown" if scenario.horizon == "unknown" else "finite"


def _near_threshold_info(scenario: Scenario) -> tuple:
    """Returns (near_threshold: bool, delta_gap: float or None)."""
    if scenario.horizon != "unknown":
        return False, None
    delta_star = critical_discount_factor(scenario.T, scenario.R, scenario.P)
    gap = abs(scenario.discount_factor - delta_star)
    return gap < NEAR_THRESHOLD_EPSILON, gap


def _payoff_extraction_correct(scenario: Scenario, extracted_params) -> Optional[bool]:
    """
    Returns:
      True  -- all four extracted payoffs exactly match the true scenario
      False -- extraction succeeded structurally but at least one payoff is wrong
      None  -- extraction failed entirely (no extracted_params to compare),
               kept distinct from False so a hard parse failure isn't
               silently counted as an "attempted but wrong" extraction

    Uses the self-describing field names (mutual_cooperation_payoff, etc.)
    rather than T/R/P/S, matching translator_solver.ExtractedParams.
    """
    if extracted_params is None:
        return None
    try:
        return (
            extracted_params.unilateral_defector_payoff == scenario.T and
            extracted_params.mutual_cooperation_payoff == scenario.R and
            extracted_params.mutual_defection_payoff == scenario.P and
            extracted_params.unilateral_cooperator_payoff == scenario.S
        )
    except AttributeError:
        return None


def score_case(case: dict, direct_result: dict, translator_result: dict) -> dict:
    """
    case: output of generate_test_case() — has 'ground_truth' and 'scenario'.
    direct_result: output of run_direct_ai() — has 'action'.
    translator_result: output of run_translator_plus_solver() — has 'action'
        and 'extraction_warnings'.
    """
    ground_truth = case["ground_truth"]
    scenario = case["scenario"]

    direct_action = direct_result.get("action")
    translator_action = translator_result.get("action")

    near_threshold, delta_gap = _near_threshold_info(scenario)

    return {
        "ground_truth": ground_truth,
        "horizon_type": _horizon_type(scenario),
        "near_threshold": near_threshold,
        "delta_gap": delta_gap,

        "direct_action": direct_action,
        "direct_parse_failure": direct_action is None,
        "direct_correct": (direct_action == ground_truth) if direct_action is not None else False,

        "translator_action": translator_action,
        "translator_parse_failure": translator_action is None,
        "translator_correct": (translator_action == ground_truth) if translator_action is not None else False,
        "translator_extraction_warnings": translator_result.get("extraction_warnings", []) or [],
        "translator_had_extraction_warnings": bool(translator_result.get("extraction_warnings")),
        "translator_payoff_extraction_correct": _payoff_extraction_correct(
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
    payoff-extraction accuracy, where None means "extraction failed
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
        "translator_payoff_extraction_accuracy": _rate_among_non_none(
            subset, "translator_payoff_extraction_correct"
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

    unknown_near = _breakdown(
        scored_cases, lambda s: s["horizon_type"] == "unknown" and s["near_threshold"]
    )
    unknown_far = _breakdown(
        scored_cases, lambda s: s["horizon_type"] == "unknown" and not s["near_threshold"]
    )

    clean_extraction = _breakdown(scored_cases, lambda s: not s["translator_had_extraction_warnings"])
    warned_extraction = _breakdown(scored_cases, lambda s: s["translator_had_extraction_warnings"])

    return {
        "overall": overall,
        "by_horizon_type": {"finite": finite, "unknown": unknown},
        "by_delta_proximity_unknown_horizon_only": {
            "near_threshold (gap < {:.2f})".format(NEAR_THRESHOLD_EPSILON): unknown_near,
            "far_from_threshold": unknown_far,
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
        lines.append(f"      Translator PAYOFF extraction accuracy: "
                      f"{fmt_pct(b['translator_payoff_extraction_accuracy'])}"
                      f"   <- honest signal, independent of action correctness")

    lines.append("=== OVERALL ===")
    fmt_breakdown("All cases", report["overall"])

    lines.append("")
    lines.append("=== BY HORIZON TYPE ===")
    fmt_breakdown("Finite horizon", report["by_horizon_type"]["finite"])
    fmt_breakdown("Unknown horizon", report["by_horizon_type"]["unknown"])

    lines.append("")
    lines.append("=== UNKNOWN-HORIZON CASES, BY DELTA PROXIMITY TO THRESHOLD ===")
    for name, b in report["by_delta_proximity_unknown_horizon_only"].items():
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
    s1 = Scenario(T=15, R=11, P=6, S=5, num_players=2, horizon=3, discount_factor=0.5)
    mock_cases.append((
        {"ground_truth": "defect", "scenario": s1},
        {"action": "defect"},
        {
            "action": "defect",
            "extraction_warnings": [],
            "extracted_params": ExtractedParams(
                unilateral_defector_payoff=15, mutual_cooperation_payoff=11,
                mutual_defection_payoff=6, unilateral_cooperator_payoff=5,
                horizon_is_fixed=True, horizon_rounds=3,
            ),
        },
    ))

    # Case 2: finite horizon -- this is the case that exposed the blind
    # spot. Action comes out "correct" (finite horizon always solves to
    # defect regardless of payoffs), but the extraction itself is the
    # exact real swap pattern observed with llama3.2:3b: true T=9,R=8,P=4,S=1
    # extracted as unilateral_defector=8, mutual_cooperation=4,
    # mutual_defection=9 (T/R/P rotated). Payoff-extraction accuracy
    # correctly flags this as wrong even though action accuracy doesn't.
    s2 = Scenario(T=9, R=8, P=4, S=1, num_players=2, horizon=6, discount_factor=0.3)
    mock_cases.append((
        {"ground_truth": "defect", "scenario": s2},
        {"action": "defect"},
        {
            "action": "defect",  # "correct" action, but for the wrong reason
            "extraction_warnings": ["Payoff ordering violated: ..."],
            "extracted_params": ExtractedParams(
                unilateral_defector_payoff=8, mutual_cooperation_payoff=4,
                mutual_defection_payoff=9, unilateral_cooperator_payoff=1,
                horizon_is_fixed=True, horizon_rounds=6,
            ),
        },
    ))

    # Case 3: unknown horizon, near threshold, translator has a genuine
    # parse failure (no extracted_params at all).
    s3 = Scenario(T=9, R=8, P=4, S=1, num_players=2, horizon="unknown", discount_factor=0.22)
    mock_cases.append((
        {"ground_truth": "cooperate", "scenario": s3},
        {"action": "defect"},  # wrong -- near-threshold cases are hard
        {"action": None, "extraction_warnings": ["Response could not be parsed."], "extracted_params": None},
    ))

    # Case 4: unknown horizon, far from threshold, both correct, translator
    # extraction also genuinely correct.
    s4 = Scenario(T=20, R=15, P=5, S=1, num_players=2, horizon="unknown", discount_factor=0.9)
    mock_cases.append((
        {"ground_truth": "cooperate", "scenario": s4},
        {"action": "cooperate"},
        {
            "action": "cooperate",
            "extraction_warnings": [],
            "extracted_params": ExtractedParams(
                unilateral_defector_payoff=20, mutual_cooperation_payoff=15,
                mutual_defection_payoff=5, unilateral_cooperator_payoff=1,
                horizon_is_fixed=False, continuation_probability_percent=90,
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
    # matched by luck) alongside translator_payoff_extraction_correct=False
    # (the extraction itself was wrong). Confirm both.
    assert scored[1]["translator_correct"] is True
    assert scored[1]["translator_payoff_extraction_correct"] is False
    print("\nConfirmed: case 2 demonstrates action-correct-but-extraction-wrong "
          "is captured correctly by the two separate metrics.")