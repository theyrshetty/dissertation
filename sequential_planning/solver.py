"""
Step 2: Deterministic solver -- computes the ground-truth optimal current-
period action for a sequential-planning (2-state/2-action MDP) scenario,
given the sampled parameters. Real dynamic programming, not an AI guess.

Finite / known horizon (an integer number of remaining periods):
    Exact backward induction. Terminal value V_T(s) = 0 for both states.
    For t = T-1 down to 0:
        Q_t(s, a) = R(s, a) + P(High | s, a) * V_{t+1}(High)
                             + (1 - P(High | s, a)) * V_{t+1}(Low)
        V_t(s) = max_a Q_t(s, a)
        pi_t(s) = argmax_a Q_t(s, a)
    Ground truth = pi_0(start_state), i.e. the optimal action to take
    right now, T periods from the end.

Unknown / indefinitely-continuing horizon:
    Discounted infinite-horizon MDP. Value iteration on the Bellman
    optimality equation:
        V(s) = max_a [ R(s, a) + gamma * ( P(High|s,a) V(High)
                                           + (1-P(High|s,a)) V(Low) ) ]
    is a gamma-contraction in sup-norm (gamma < 1), so iterating from
    V=0 converges to the unique fixed point V* geometrically fast. We
    iterate until the sup-norm change drops below a tight tolerance (or a
    large iteration cap is hit), then read off the greedy action at the
    start state as the ground truth.
"""

from typing import Dict, Tuple

from generator import Scenario, STATES, ACTIONS, START_STATE

_MAX_ITERATIONS = 100_000
_CONVERGENCE_TOL = 1e-12


def _reward(scenario: Scenario, state: str, action: str) -> float:
    return {
        ("Low", "Cautious"): scenario.r_low_cautious,
        ("Low", "Aggressive"): scenario.r_low_aggressive,
        ("High", "Cautious"): scenario.r_high_cautious,
        ("High", "Aggressive"): scenario.r_high_aggressive,
    }[(state, action)]


def _prob_high(scenario: Scenario, state: str, action: str) -> float:
    return {
        ("Low", "Cautious"): scenario.p_high_low_cautious,
        ("Low", "Aggressive"): scenario.p_high_low_aggressive,
        ("High", "Cautious"): scenario.p_high_high_cautious,
        ("High", "Aggressive"): scenario.p_high_high_aggressive,
    }[(state, action)]


def _q_value(scenario: Scenario, state: str, action: str, v_next: Dict[str, float],
             gamma: float) -> float:
    p_high = _prob_high(scenario, state, action)
    expected_future = p_high * v_next["High"] + (1 - p_high) * v_next["Low"]
    return _reward(scenario, state, action) + gamma * expected_future


def _greedy_action(scenario: Scenario, state: str, v_next: Dict[str, float],
                    gamma: float) -> Tuple[str, Dict[str, float]]:
    """Returns (best_action, {action: q_value}) at `state`, breaking ties
    in favor of "Cautious" (an arbitrary but fixed, documented tie-break;
    exact ties are numerically rare given the generator's min_reward_gap /
    min_prob_gap constraints, but must still resolve deterministically)."""
    qs = {a: _q_value(scenario, state, a, v_next, gamma) for a in ACTIONS}
    best = "Cautious" if qs["Cautious"] >= qs["Aggressive"] else "Aggressive"
    return best, qs


def solve_finite_horizon(scenario: Scenario) -> dict:
    """Exact backward induction over `scenario.horizon` remaining periods."""
    T = scenario.horizon
    assert isinstance(T, int) and T >= 1

    v_next = {"Low": 0.0, "High": 0.0}  # V_T(s) = 0
    policy_t0 = None
    qs_t0 = None

    # gamma = 1.0 in the finite-horizon branch: a known, fixed endpoint
    # means there's no reason to discount periods within the horizon --
    # the standard finite-horizon convention (the repeated-game module's
    # finite-horizon branch makes the same simplifying choice implicitly,
    # since backward induction there uses undiscounted per-round payoffs).
    for t in range(T - 1, -1, -1):
        v_t = {}
        for s in STATES:
            best_action, qs = _greedy_action(scenario, s, v_next, gamma=1.0)
            v_t[s] = qs[best_action]
            if t == 0 and s == scenario.start_state:
                policy_t0 = best_action
                qs_t0 = qs
        v_next = v_t

    margin = abs(qs_t0["Cautious"] - qs_t0["Aggressive"])

    return {
        "action": policy_t0.lower(),
        "reason": (
            f"Finite horizon ({T} periods remaining): exact backward "
            f"induction from V_T=0 gives Q_0({scenario.start_state}, Cautious)="
            f"{qs_t0['Cautious']:.4f} and Q_0({scenario.start_state}, Aggressive)="
            f"{qs_t0['Aggressive']:.4f}; the optimal action right now is "
            f"{policy_t0.lower()}."
        ),
        "margin": margin,
        "q_values": {k.lower(): v for k, v in qs_t0.items()},
    }


def solve_infinite_horizon(scenario: Scenario) -> dict:
    """Value iteration to convergence on the discounted Bellman optimality
    equation; gamma = scenario.discount_factor."""
    gamma = scenario.discount_factor
    assert 0 < gamma < 1

    v = {"Low": 0.0, "High": 0.0}
    for iteration in range(_MAX_ITERATIONS):
        v_new = {}
        for s in STATES:
            best_action, qs = _greedy_action(scenario, s, v, gamma=gamma)
            v_new[s] = qs[best_action]
        delta = max(abs(v_new[s] - v[s]) for s in STATES)
        v = v_new
        if delta < _CONVERGENCE_TOL:
            break

    best_action, qs = _greedy_action(scenario, scenario.start_state, v, gamma=gamma)
    margin = abs(qs["Cautious"] - qs["Aggressive"])

    return {
        "action": best_action.lower(),
        "reason": (
            f"Unknown/indefinite horizon, discount factor gamma={gamma}: "
            f"value iteration converged (after {iteration + 1} iterations, "
            f"sup-norm change < {_CONVERGENCE_TOL}) to V*(Low)={v['Low']:.4f}, "
            f"V*(High)={v['High']:.4f}. At the start state "
            f"({scenario.start_state}), Q*(Cautious)={qs['Cautious']:.4f} and "
            f"Q*(Aggressive)={qs['Aggressive']:.4f}; the optimal action right "
            f"now is {best_action.lower()}."
        ),
        "margin": margin,
        "q_values": {k.lower(): v_ for k, v_ in qs.items()},
    }


def solve(scenario: Scenario) -> dict:
    """
    Returns a dict:
      {
        "action": "cautious" | "aggressive",
        "reason": short human-readable justification,
        "margin": float, |Q(Cautious) - Q(Aggressive)| at the start state
                  under the optimal value function -- the sequential-
                  planning analogue of the repeated-game module's
                  |delta - delta*| "how close was this call" signal.
        "q_values": {"cautious": float, "aggressive": float},
      }
    """
    if scenario.horizon != "unknown":
        return solve_finite_horizon(scenario)
    return solve_infinite_horizon(scenario)


if __name__ == "__main__":
    import random
    from generator import generate_scenario

    rng = random.Random(42)
    for i in range(8):
        s = generate_scenario(rng=rng)
        result = solve(s)
        print(i, s)
        print("   ->", result["action"], "| margin=%.4f" % result["margin"])
        print("     ", result["reason"])
        print()
