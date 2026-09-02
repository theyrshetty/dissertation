"""
Step 1: Scenario generator for sequential-planning (finite MDP) problems.

Mirrors the repeated-game module's generator.py in structure and intent,
adapted from a one-shot cooperate/defect matrix to a small Markov Decision
Process that genuinely requires looking ahead, not just reading off a
dominant action.

Structure, fixed across every sampled scenario:
  - 2 states: "Low" and "High" (an operating condition -- e.g. low vs
    high demand / reserves / standing -- with High objectively the more
    profitable state to occupy).
  - 2 actions: "Cautious" and "Aggressive".
  - A decision-maker starts in the "Low" state and must choose an action
    for the *current* period -- this is the sequential-planning analogue
    of the repeated-game module's "what should party A do in round 1".
  - horizon: a known fixed integer number of remaining periods (solved by
    exact backward induction), or "unknown" (an indefinitely continuing
    process with a discount factor, solved by value iteration to a fixed
    point).

Payoff/transition structure enforced by generation (deliberately parallel
to the repeated-game module's T > R > P > S, 2R > T+S convention -- a
structural constraint that creates genuine tension rather than a scenario
with an obviously dominant action):

  1. High is objectively the better state to be in: for BOTH actions,
     the immediate reward earned in "High" exceeds the reward earned in
     "Low".
  2. "Aggressive" pays a higher IMMEDIATE reward than "Cautious", in both
     states (the temptation to be greedy).
  3. "Cautious" gives a higher PROBABILITY of reaching/staying in the
     better "High" state than "Aggressive" does, from both states (the
     safer choice better protects the future).

This produces a genuine short-term-gain-vs-long-term-position tradeoff:
whether "Cautious" or "Aggressive" is optimal in the current "Low" state
depends on the horizon length / discount factor, and must be computed by
real dynamic programming -- it is never a foregone conclusion from the
reward numbers alone.
"""

import random
from dataclasses import dataclass, asdict
from typing import Optional, Union

STATES = ("Low", "High")
ACTIONS = ("Cautious", "Aggressive")
START_STATE = "Low"


@dataclass
class Scenario:
    # Immediate reward earned in a (state, action) pair, this period.
    r_low_cautious: int
    r_low_aggressive: int
    r_high_cautious: int
    r_high_aggressive: int

    # Probability of transitioning to "High" next period, given the
    # current state and chosen action. (Probability of ending up in "Low"
    # next period is always 1 minus this.) Rounded to the nearest whole
    # percent at sampling time -- see the discount_factor comment below
    # for why exact-percent rounding at generation time matters.
    p_high_low_cautious: float
    p_high_low_aggressive: float
    p_high_high_cautious: float
    p_high_high_aggressive: float

    num_states: int          # fixed at 2 for this module type
    num_actions: int         # fixed at 2 for this module type
    start_state: str         # fixed at "Low"
    horizon: Union[int, str]  # int, or the literal string "unknown"
    discount_factor: float    # gamma in (0,1); only decisive when horizon == "unknown"

    def as_dict(self):
        return asdict(self)


def generate_scenario(
    reward_min: int = 1,
    reward_max: int = 30,
    min_reward_gap: int = 1,
    max_reward_gap: int = 5,
    min_state_gap: int = 4,
    max_state_gap: int = 12,
    prob_min: float = 0.05,
    prob_max: float = 0.95,
    min_prob_gap: float = 0.20,
    max_prob_gap: float = 0.60,
    horizon_choices: Optional[list] = None,
    unknown_horizon_prob: float = 0.5,
    discount_range: tuple = (0.05, 0.95),
    rng: Optional[random.Random] = None,
) -> Scenario:
    """
    Randomly sample a single 2-state/2-action MDP scenario.

    Rewards and transition probabilities are built additively (rather than
    drawn independently and rejection-sampled against inequalities), so
    every one of the four structural constraints in the module docstring
    holds by construction, with a tunable, roughly-balanced gap between the
    two actions in both directions:

      r_high_cautious   = r_low_cautious  + (a "state gap": min_state_gap..max_state_gap)
      r_low_aggressive   = r_low_cautious  + (a "temptation gap": min_reward_gap..max_reward_gap)
      r_high_aggressive  = r_high_cautious + (a "temptation gap": min_reward_gap..max_reward_gap)

      p_high_low_cautious  = p_high_low_aggressive  + (min_prob_gap..max_prob_gap)
      p_high_high_cautious = p_high_high_aggressive + (min_prob_gap..max_prob_gap)

    max_state_gap is kept comfortably larger than max_reward_gap so that
    r_high_aggressive > r_low_aggressive holds in the large majority of
    draws (checked explicitly below, with a cheap retry on the rare
    violation) without needing combinatorial rejection sampling.

    The (min_reward_gap, max_reward_gap) vs. (min_prob_gap, max_prob_gap)
    ranges were tuned empirically (see stress_test.py) so that neither
    "Cautious" nor "Aggressive" is optimal in a lopsided majority of
    generated scenarios -- i.e. the tradeoff the structural constraints
    are meant to create actually bites, rather than one action being a
    disguised dominant strategy.

    horizon:
      - with probability `unknown_horizon_prob`, horizon = "unknown"
      - otherwise, horizon is a fixed integer drawn from `horizon_choices`
        (default range 2..10)

    discount_factor is always sampled from `discount_range`, uniformly,
    rounded to 2 decimal places. It is only used by the solver when
    horizon == "unknown".

    Probabilities are sampled as whole percents directly, then converted,
    so the ground truth is always computed from exactly the number that
    will later be displayed in the paragraph (e.g. "62%") -- a
    Translator-plus-Solver arm that faithfully reads the displayed
    percentage can never be unfairly penalized by a rounding mismatch
    against a higher-precision internal value.
    """
    rng = rng or random
    if horizon_choices is None:
        horizon_choices = list(range(2, 11))

    pct_min, pct_max = round(prob_min * 100), round(prob_max * 100)
    pct_min_gap, pct_max_gap = round(min_prob_gap * 100), round(max_prob_gap * 100)

    headroom = reward_max - reward_min - max_state_gap - max_reward_gap
    if headroom < 0:
        raise RuntimeError(
            "reward_max is too small for the given max_state_gap/max_reward_gap."
        )

    max_attempts = 2000
    for _ in range(max_attempts):
        r_low_cautious = rng.randint(reward_min, reward_min + headroom)
        state_gap = rng.randint(min_state_gap, max_state_gap)
        r_high_cautious = r_low_cautious + state_gap

        r_low_aggressive = r_low_cautious + rng.randint(min_reward_gap, max_reward_gap)
        r_high_aggressive = r_high_cautious + rng.randint(min_reward_gap, max_reward_gap)

        if r_high_aggressive - r_low_aggressive < min_reward_gap:
            continue  # rare: temptation-gap noise ate into the state gap

        p_high_low_aggressive_pct = rng.randint(pct_min, pct_max - pct_min_gap)
        p_high_low_cautious_pct = min(
            pct_max, p_high_low_aggressive_pct + rng.randint(pct_min_gap, pct_max_gap)
        )
        p_high_high_aggressive_pct = rng.randint(pct_min, pct_max - pct_min_gap)
        p_high_high_cautious_pct = min(
            pct_max, p_high_high_aggressive_pct + rng.randint(pct_min_gap, pct_max_gap)
        )

        if p_high_low_cautious_pct - p_high_low_aggressive_pct < pct_min_gap:
            continue  # rare: only happens if the min() clamp above ate the gap
        if p_high_high_cautious_pct - p_high_high_aggressive_pct < pct_min_gap:
            continue

        horizon: Union[int, str]
        if rng.random() < unknown_horizon_prob:
            horizon = "unknown"
        else:
            horizon = rng.choice(horizon_choices)

        discount_factor = round(rng.uniform(*discount_range), 2)

        return Scenario(
            r_low_cautious=r_low_cautious,
            r_low_aggressive=r_low_aggressive,
            r_high_cautious=r_high_cautious,
            r_high_aggressive=r_high_aggressive,
            p_high_low_cautious=p_high_low_cautious_pct / 100,
            p_high_low_aggressive=p_high_low_aggressive_pct / 100,
            p_high_high_cautious=p_high_high_cautious_pct / 100,
            p_high_high_aggressive=p_high_high_aggressive_pct / 100,
            num_states=2,
            num_actions=2,
            start_state=START_STATE,
            horizon=horizon,
            discount_factor=discount_factor,
        )

    raise RuntimeError(
        "Could not sample a valid scenario satisfying constraints; "
        "widen reward_max or adjust the gap parameters."
    )


if __name__ == "__main__":
    rng = random.Random(42)
    for i in range(8):
        s = generate_scenario(rng=rng)
        print(i, s)
