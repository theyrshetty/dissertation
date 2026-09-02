"""
Step 1: Scenario generator for repeated Prisoner's-Dilemma-style games.

Produces a random, well-formed classic PD payoff matrix plus a horizon
specification (fixed integer or "unknown"), and a discount factor (only
meaningful when horizon is "unknown", but always sampled for consistency
and reproducibility).
"""

import random
from dataclasses import dataclass, asdict
from typing import Optional, Union


@dataclass
class Scenario:
    # Payoffs, from the perspective of a single (symmetric) player:
    # T = Temptation (I defect, other cooperates)
    # R = Reward      (both cooperate)
    # P = Punishment  (both defect)
    # S = Sucker      (I cooperate, other defects)
    T: int
    R: int
    P: int
    S: int
    num_players: int          # fixed at 2 for this module type
    horizon: Union[int, str]  # int, or the literal string "unknown"
    discount_factor: float    # delta in (0,1); only decisive when horizon == "unknown"

    def as_dict(self):
        return asdict(self)


def generate_scenario(
    payoff_min: int = 1,
    payoff_max: int = 20,
    min_gap: int = 1,
    horizon_choices: Optional[list] = None,
    unknown_horizon_prob: float = 0.5,
    discount_range: tuple = (0.05, 0.95),
    rng: Optional[random.Random] = None,
) -> Scenario:
    """
    Randomly sample a single classic-PD scenario.

    Constraints enforced:
      - T > R > P > S  (strictly, with at least `min_gap` between each,
        so the ordering is unambiguous and never borderline/tied)
      - 2R > T + S      (mutual cooperation beats alternating C/D on average;
        standard convention so the game is a "clean" PD)
      - all payoffs within [payoff_min, payoff_max]

    horizon:
      - with probability `unknown_horizon_prob`, horizon = "unknown"
      - otherwise, horizon is a fixed integer drawn from `horizon_choices`
        (default range 2..10)

    discount_factor is always sampled from `discount_range`, uniformly.
    It is only used by the solver when horizon == "unknown".
    """
    rng = rng or random
    if horizon_choices is None:
        horizon_choices = list(range(2, 11))

    # Rejection sampling: draw 4 sorted-distinct-enough values for S < P < R < T,
    # then check the 2R > T + S convention. This keeps the code simple and is
    # cheap enough at these small integer ranges.
    max_attempts = 2000
    for _ in range(max_attempts):
        vals = rng.sample(range(payoff_min, payoff_max + 1), 4)
        vals.sort()
        S, P, R, T = vals  # ascending

        # Enforce minimum gaps so no two payoffs are close enough to be
        # borderline/ambiguous.
        if (P - S) < min_gap or (R - P) < min_gap or (T - R) < min_gap:
            continue

        if not (2 * R > T + S):
            continue

        horizon: Union[int, str]
        if rng.random() < unknown_horizon_prob:
            horizon = "unknown"
        else:
            horizon = rng.choice(horizon_choices)

        # Round to the nearest whole percent *before* it's used anywhere.
        # This must match exactly what the renderer will display (e.g. "71%"),
        # so that a Translator-plus-Solver arm which faithfully extracts the
        # displayed percentage and recomputes always agrees with the ground
        # truth. If we instead kept a high-precision float for ground truth
        # but only displayed a rounded percentage in the paragraph, a
        # borderline scenario could round across the delta* threshold and
        # unfairly penalize a translator that read the text correctly.
        discount_factor = round(rng.uniform(*discount_range), 2)

        return Scenario(
            T=T, R=R, P=P, S=S,
            num_players=2,
            horizon=horizon,
            discount_factor=discount_factor,
        )

    raise RuntimeError(
        "Could not sample a valid scenario satisfying constraints; "
        "widen payoff_min/payoff_max or reduce min_gap."
    )


if __name__ == "__main__":
    # Quick manual sanity check: generate a handful and print them.
    rng = random.Random(42)
    for i in range(8):
        s = generate_scenario(rng=rng)
        print(i, s)