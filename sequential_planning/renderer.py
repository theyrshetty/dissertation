"""
Step 3: Paragraph renderer.

Turns a Scenario's numeric parameters into a natural-language paragraph
describing a small sequential-planning problem: a decision-maker currently
in the "Low" state choosing between a cautious and an aggressive action
this period, under either a known fixed number of remaining periods or an
indefinitely continuing process. Template-based only, no LLM involved. All
numbers that matter for solving the scenario are stated explicitly (the
"clean" rendering mode, matching the repeated-game module's Part A scope;
a "noisy" mode is future work, per Part D of the study design).

Framing choices:
  - horizon == int      -> stated as an exact number of remaining periods.
  - horizon == "unknown" -> the discount factor is rendered as a
    continuation probability ("an X% chance operations continue into
    another period"), exactly mirroring the repeated-game module's
    treatment of discount factors as honest, standard natural language.
"""

import random
from typing import Optional

from generator import Scenario

# A small pool of cover stories. Each story needs one decision-making
# entity, two named operating states ("Low"/"High" underneath), and two
# named actions ("Cautious"/"Aggressive" underneath). Numbers are injected
# afterward; the story only supplies flavor and verbs.
_COVER_STORIES = [
    {
        "actor": "Marlow Outfitters",
        "context": "a seasonal outdoor-gear retailer deciding, period by period, "
                   "how hard to push sales given its current standing with suppliers",
        "low_state": "weak supplier standing",
        "high_state": "strong supplier standing",
        "verb_cautious": "restock conservatively and keep relationships steady",
        "verb_aggressive": "over-order aggressively to chase short-term sales",
    },
    {
        "actor": "Dr. Ilsa Renn",
        "context": "a clinic director deciding, period by period, how hard to run "
                   "her short-staffed clinic given its current staff morale",
        "low_state": "low staff morale",
        "high_state": "high staff morale",
        "verb_cautious": "keep a sustainable pace for the team",
        "verb_aggressive": "push the team hard for extra throughput",
    },
    {
        "actor": "Thornebridge Farms",
        "context": "a farm cooperative deciding, period by period, how intensively "
                   "to work its land given its current soil condition",
        "low_state": "depleted soil condition",
        "high_state": "healthy soil condition",
        "verb_cautious": "farm conservatively and let the land recover",
        "verb_aggressive": "farm intensively for a bigger immediate yield",
    },
    {
        "actor": "Voss Analytics",
        "context": "a small consultancy deciding, period by period, how hard to "
                   "chase new billable work given its current team capacity",
        "low_state": "strained team capacity",
        "high_state": "healthy team capacity",
        "verb_cautious": "take on a measured amount of new work",
        "verb_aggressive": "take on as much new work as possible",
    },
]


def _horizon_sentence(scenario: Scenario, story: dict) -> str:
    if scenario.horizon == "unknown":
        pct = round(scenario.discount_factor * 100)
        return (
            f"After this period, there's roughly a {pct}% chance "
            f"{story['actor']} continues operating for another period under "
            f"the same choice each time; otherwise operations wind down for good."
        )
    else:
        return (
            f"{story['actor']} knows in advance that there are exactly "
            f"{scenario.horizon} periods left, including this one, and this "
            f"number is fixed and known."
        )


def _payoff_sentence(scenario: Scenario, story: dict) -> str:
    cau, agg = story["verb_cautious"], story["verb_aggressive"]
    low, high = story["low_state"], story["high_state"]

    return (
        f"Right now, {story['actor']} is in a state of {low}. "
        f"In any period spent in {low}: choosing to {cau} earns "
        f"{scenario.r_low_cautious} thousand dollars that period, and leaves a "
        f"{round(scenario.p_high_low_cautious * 100)}% chance of moving into a "
        f"state of {high} next period (otherwise it remains in {low}). Choosing "
        f"to {agg} earns {scenario.r_low_aggressive} thousand dollars that "
        f"period instead, but leaves only a "
        f"{round(scenario.p_high_low_aggressive * 100)}% chance of moving into "
        f"{high} next period. "
        f"In any period spent in a state of {high} instead: choosing to {cau} "
        f"earns {scenario.r_high_cautious} thousand dollars that period, and "
        f"leaves a {round(scenario.p_high_high_cautious * 100)}% chance of "
        f"remaining in {high} next period (otherwise it falls back to {low}). "
        f"Choosing to {agg} earns {scenario.r_high_aggressive} thousand dollars "
        f"that period instead, but leaves only a "
        f"{round(scenario.p_high_high_aggressive * 100)}% chance of remaining "
        f"in {high} next period."
    )


def render_paragraph(scenario: Scenario, rng: Optional[random.Random] = None) -> str:
    """
    Render a Scenario into a natural-language paragraph. Clean mode: every
    decision-relevant number (rewards, transition percentages, horizon
    info) is stated explicitly, wrapped in a short cover story.
    """
    rng = rng or random
    story = rng.choice(_COVER_STORIES)
    cau, agg = story["verb_cautious"], story["verb_aggressive"]

    payoff_sentence = _payoff_sentence(scenario, story)
    horizon_sentence = _horizon_sentence(scenario, story)

    paragraph = (
        f"{story['actor']} is {story['context']}. {payoff_sentence} "
        f"{horizon_sentence} "
        f"If you were advising {story['actor']} on the decision for this "
        f"very period, should it choose to {cau}, or to {agg}?"
    )
    return paragraph


def render_paraphrases(
    scenario: Scenario,
    num_variants: int = 3,
    rng: Optional[random.Random] = None,
) -> list:
    """Render differently worded descriptions of one already-sampled scenario.

    This function never samples a scenario or changes any of its fields.
    Each returned paragraph states the same eight numbers (four rewards,
    four transition percentages) and the same horizon information; only
    syntax and ordering of the prose differ. Intended for the
    consistency-across-paraphrases evaluation, where every wording must
    describe identical underlying numbers.

    ``num_variants`` may be 2 or 3, matching the planned evaluation design.
    """
    if num_variants not in (2, 3):
        raise ValueError("num_variants must be 2 or 3.")

    rng = rng or random
    story = rng.choice(_COVER_STORIES)
    actor = story["actor"]
    cau, agg = story["verb_cautious"], story["verb_aggressive"]
    low, high = story["low_state"], story["high_state"]

    r_lc, r_la = scenario.r_low_cautious, scenario.r_low_aggressive
    r_hc, r_ha = scenario.r_high_cautious, scenario.r_high_aggressive
    p_lc = round(scenario.p_high_low_cautious * 100)
    p_la = round(scenario.p_high_low_aggressive * 100)
    p_hc = round(scenario.p_high_high_cautious * 100)
    p_ha = round(scenario.p_high_high_aggressive * 100)

    if scenario.horizon == "unknown":
        pct = round(scenario.discount_factor * 100)
        horizon_variants = [
            f"After this period, there is a {pct}% chance operations continue into another period; otherwise they end for good.",
            f"There is no announced final period: after this one, a new period follows with probability {pct}%, and otherwise things stop.",
            f"No end date is fixed. Following this period, the probability of another period is {pct}%.",
        ]
    else:
        horizon_variants = [
            f"{actor} knows in advance that exactly {scenario.horizon} periods remain, including this one.",
            f"It is known ahead of time that there will be a fixed total of {scenario.horizon} periods left, starting with this one.",
            f"The remaining timeline is common knowledge: precisely {scenario.horizon} periods remain, this one included.",
        ]

    variants = [
        (
            f"{actor} is {story['context']}. Right now it is in a state of {low}. "
            f"From {low}, choosing to {cau} earns {r_lc} thousand dollars this period with a {p_lc}% chance of reaching {high} next period; "
            f"choosing to {agg} earns {r_la} thousand dollars instead with only a {p_la}% chance of reaching {high}. "
            f"From {high}, choosing to {cau} earns {r_hc} thousand dollars with a {p_hc}% chance of staying in {high}; "
            f"choosing to {agg} earns {r_ha} thousand dollars with only a {p_ha}% chance of staying in {high}. "
            f"{horizon_variants[0]} For this period, should {actor} {cau}, or {agg}?"
        ),
        (
            f"Consider {story['context']}: {actor} must choose an action every period, currently starting from {low}. "
            f"The payoffs work as follows. In {low}: {cau} yields {r_lc} (and a {p_lc}% shot at {high} next period), while "
            f"{agg} yields {r_la} (and only a {p_la}% shot at {high}). In {high}: {cau} yields {r_hc} (and a {p_hc}% chance of remaining there), "
            f"while {agg} yields {r_ha} (and only a {p_ha}% chance of remaining there). "
            f"{horizon_variants[1]} What should {actor} do this period: {cau} or {agg}?"
        ),
        (
            f"{actor} must decide what to do this period, currently in a state of {low}. It is {story['context']}. "
            f"Being in {low} and choosing {cau} pays {r_lc} this period and gives a {p_lc}% chance of moving to {high}; choosing {agg} instead "
            f"pays {r_la} this period but only gives a {p_la}% chance of moving to {high}. Being in {high} and choosing {cau} pays {r_hc} with "
            f"a {p_hc}% chance of staying, while choosing {agg} pays {r_ha} with only a {p_ha}% chance of staying. "
            f"{horizon_variants[2]} Should {actor} choose {cau} or {agg} this period?"
        ),
    ]
    return variants[:num_variants]


if __name__ == "__main__":
    import random as _random
    from generator import generate_scenario
    from solver import solve

    rng = _random.Random(7)
    for i in range(4):
        s = generate_scenario(rng=rng)
        para = render_paragraph(s, rng=rng)
        result = solve(s)
        print(f"--- Example {i} ---")
        print(para)
        print(f"[ground truth: {result['action']}]")
        print()
