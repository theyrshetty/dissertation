"""
Step 2: Deterministic solver — computes the ground-truth optimal first-round
action for a repeated classic-PD scenario, given the sampled parameters.

Logic (as agreed):

Finite / known horizon (an integer):
    Backward induction. In a single-shot PD, "defect" strictly dominates
    "cooperate" no matter what the opponent does (T > R and P > S).
    A finitely repeated game with a stage game that has a strictly dominant
    action has a unique subgame-perfect equilibrium: defect in every round,
    including round 1. Horizon length therefore does not change the answer,
    only whether it is finite vs. unknown.

Unknown / infinite horizon:
    Cooperation can be sustained via grim-trigger if the discount factor
    delta is high enough. The critical discount factor is:

        delta_star = (T - R) / (T - P)

    derived from the standard grim-trigger sustainability condition:

        R / (1 - delta) >= T + delta * P / (1 - delta)

    If the sampled delta >= delta_star, grim-trigger cooperation is a
    subgame-perfect equilibrium, so ground truth = "cooperate".
    Otherwise, the only equilibrium is always-defect, so ground truth
    = "defect".
"""

from generator import Scenario


def critical_discount_factor(T: int, R: int, P: int) -> float:
    """delta* = (T - R) / (T - P). Always in (0, 1) given T > R > P."""
    return (T - R) / (T - P)


def solve(scenario: Scenario) -> dict:
    """
    Returns a dict:
      {
        "action": "cooperate" | "defect",
        "reason": short human-readable justification,
        "delta_star": float or None (only computed for unknown horizon),
      }
    """
    T, R, P, S = scenario.T, scenario.R, scenario.P, scenario.S

    if scenario.horizon != "unknown":
        # Finite known horizon -> backward induction -> always defect.
        return {
            "action": "defect",
            "reason": (
                f"Finite horizon ({scenario.horizon} rounds): stage game has "
                f"a strictly dominant action (defect, since T={T}>R={R} and "
                f"P={P}>S={S}), so backward induction yields defect in every "
                f"round including round 1."
            ),
            "delta_star": None,
        }

    delta = scenario.discount_factor

    # T <= P means the "punishment gap" the grim-trigger threshold formula
    # depends on (T - P) is zero or negative. This never happens for a
    # generator-produced Scenario (always T > R > P > S by construction),
    # but the Translator arm feeds this same function malformed Scenarios
    # built from LLM-extracted payoffs, which can absolutely violate that
    # ordering. Rather than crash the whole evaluation on one bad
    # extraction, treat this as: no positive punishment gap means
    # grim-trigger cooperation cannot be analyzed via this formula at all,
    # so default to defect and say why, exactly like the internal-
    # consistency warnings validate_extraction already raises for this
    # same condition.
    if T <= P:
        return {
            "action": "defect",
            "reason": (
                f"Unknown horizon, but input payoffs are degenerate for the "
                f"grim-trigger formula (unilateral_defector_payoff={T} <= "
                f"mutual_defection_payoff={P}, so T-P is zero or negative). "
                f"This is not possible for a well-formed PD scenario and "
                f"indicates a bad extraction upstream; defaulting to defect "
                f"since a positive punishment gap is required to sustain "
                f"cooperation."
            ),
            "delta_star": None,
        }

    # Unknown / infinite horizon -> grim-trigger sustainability check.
    delta_star = critical_discount_factor(T, R, P)

    if delta >= delta_star:
        action = "cooperate"
        reason = (
            f"Unknown horizon, delta={delta} >= delta*={delta_star:.4f}: "
            f"grim-trigger cooperation is sustainable as a subgame-perfect "
            f"equilibrium, so the optimal round-1 action is cooperate."
        )
    else:
        action = "defect"
        reason = (
            f"Unknown horizon, delta={delta} < delta*={delta_star:.4f}: "
            f"players do not value the future enough to sustain grim-trigger "
            f"cooperation, so the unique equilibrium is always-defect."
        )

    return {"action": action, "reason": reason, "delta_star": delta_star}


if __name__ == "__main__":
    import random
    from generator import generate_scenario

    rng = random.Random(42)
    for i in range(8):
        s = generate_scenario(rng=rng)
        result = solve(s)
        print(i, s)
        print("   ->", result["action"], "|", result["reason"])
        print()