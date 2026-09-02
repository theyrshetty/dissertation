"""
Step 3: Paragraph renderer.

Turns a Scenario's numeric parameters into a natural-language paragraph
describing a repeated cooperate/defect negotiation between two parties.
Template-based only, no LLM involved. All numbers that matter for solving
the scenario are stated explicitly in the text (this is the "clean"
rendering mode; a "noisy" mode that obscures/buries these numbers is future
work, per Part D of the study design).

Framing choices:
  - The 2x2 payoff matrix is described as a plain payout table.
  - horizon == int   -> stated as an exact number of rounds.
  - horizon == "unknown" -> the discount factor is rendered as a
    continuation probability ("an X% chance they'll deal with each other
    again next round"), which is the standard game-theoretic reading of a
    discount factor and is honest, natural language.
"""

import random
from typing import Optional

from generator import Scenario

# A small pool of cover stories so paragraphs aren't all identical in
# surface form. Each story just needs two interchangeable parties and a
# repeated-interaction framing; the numbers are injected afterward.
_COVER_STORIES = [
    {
        "party_a": "Alto Freight",
        "party_b": "Bexley Logistics",
        "context": "two shipping companies that repeatedly decide whether to "
                   "honor an informal capacity-sharing agreement or quietly "
                   "take extra cargo for themselves",
        "verb_cooperate": "honor the agreement",
        "verb_defect": "take the extra cargo for themselves",
    },
    {
        "party_a": "Nairi",
        "party_b": "Voss",
        "context": "two market vendors at a shared stall who repeatedly decide "
                   "whether to split the best selling spot fairly or push "
                   "each other out of it",
        "verb_cooperate": "split the spot fairly",
        "verb_defect": "push the other out of the spot",
    },
    {
        "party_a": "Kessler Labs",
        "party_b": "Orinth Biotech",
        "context": "two research labs that repeatedly decide whether to share "
                   "preliminary data with each other or withhold it to gain "
                   "an edge",
        "verb_cooperate": "share their preliminary data",
        "verb_defect": "withhold their data",
    },
    {
        "party_a": "Priya",
        "party_b": "Dumont",
        "context": "two co-founders of a small studio who repeatedly decide "
                   "whether to split a shared client fairly or quietly "
                   "over-bill the client themselves",
        "verb_cooperate": "split the fee fairly",
        "verb_defect": "over-bill the client",
    },
]


def _horizon_sentence(scenario: Scenario, story: dict) -> str:
    if scenario.horizon == "unknown":
        pct = round(scenario.discount_factor * 100)
        return (
            f"After each round, there's roughly a {pct}% chance "
            f"{story['party_a']} and {story['party_b']} will end up dealing "
            f"with each other again; otherwise, this is their last interaction."
        )
    else:
        return (
            f"{story['party_a']} and {story['party_b']} know in advance that "
            f"they will interact for exactly {scenario.horizon} rounds, and "
            f"both are aware of this fixed number."
        )


def render_paragraph(scenario: Scenario, rng: Optional[random.Random] = None) -> str:
    """
    Render a Scenario into a natural-language paragraph. Clean mode: all
    decision-relevant numbers are stated explicitly (payoffs and horizon
    info), just wrapped in a short cover story rather than presented as a
    bare spec sheet.
    """
    rng = rng or random
    story = rng.choice(_COVER_STORIES)
    a, b = story["party_a"], story["party_b"]
    coop, defect = story["verb_cooperate"], story["verb_defect"]

    T, R, P, S = scenario.T, scenario.R, scenario.P, scenario.S

    payoff_sentence = (
        f"In any single round: if both choose to {coop}, each earns {R} points. "
        f"If both choose to {defect}, each earns {P} points. "
        f"If one chooses to {defect} while the other chooses to {coop}, "
        f"the one who defects earns {T} points and the one who cooperates "
        f"earns only {S} points."
    )

    horizon_sentence = _horizon_sentence(scenario, story)

    paragraph = (
        f"{a} and {b} are {story['context']}. {payoff_sentence} "
        f"{horizon_sentence} "
        f"If you were advising {a} on the very first round, "
        f"should {a} {coop} or {defect}?"
    )
    return paragraph


def render_paraphrases(
    scenario: Scenario,
    num_variants: int = 3,
    rng: Optional[random.Random] = None,
) -> list[str]:
    """Render differently worded descriptions of one already-sampled scenario.

    This function never samples a scenario or changes any of its fields.  Each
    returned paragraph states the same four payoffs and the same horizon
    information; only syntax and ordering of the prose differ.  It is intended
    for the consistency-across-paraphrases evaluation, where every wording must
    describe identical underlying numbers.

    ``num_variants`` may be 2 or 3, matching the planned evaluation design.
    A supplied ``rng`` is used only to select a cover story, so reproducibility
    can be controlled without changing the scenario itself.
    """
    if num_variants not in (2, 3):
        raise ValueError("num_variants must be 2 or 3.")

    rng = rng or random
    story = rng.choice(_COVER_STORIES)
    a, b = story["party_a"], story["party_b"]
    coop, defect = story["verb_cooperate"], story["verb_defect"]
    T, R, P, S = scenario.T, scenario.R, scenario.P, scenario.S

    if scenario.horizon == "unknown":
        pct = round(scenario.discount_factor * 100)
        horizon_variants = [
            f"After each round, there is a {pct}% chance that {a} and {b} will meet again; otherwise the interaction ends.",
            f"The relationship has no announced endpoint: after any round it continues with probability {pct}% and ends otherwise.",
            f"Neither party knows a final round. Following each round, the probability of another interaction is {pct}%.",
        ]
    else:
        horizon_variants = [
            f"They know beforehand that the interaction lasts exactly {scenario.horizon} rounds.",
            f"Both parties are told that there will be a fixed total of {scenario.horizon} rounds.",
            f"The endpoint is common knowledge: precisely {scenario.horizon} rounds will be played.",
        ]

    variants = [
        (
            f"{a} and {b} are {story['context']}. In a round where both {coop}, each receives {R} points. "
            f"When both choose to {defect}, each receives {P} points. If one chooses to {defect} while the other chooses to {coop}, "
            f"the defector gets {T} points and the cooperator gets {S} points. {horizon_variants[0]} "
            f"On the first round, should {a} {coop} or {defect}?"
        ),
        (
            f"Consider {story['context']}: {a} and {b} repeatedly choose whether to {coop} or {defect}. "
            f"Their payoffs are as follows. Mutual {coop} gives {R} points to each, whereas mutual {defect} gives {P} points to each. "
            f"With different choices, the party choosing to {defect} earns {T} points and the party choosing to {coop} earns {S} points. "
            f"{horizon_variants[1]} What should {a} do first: {coop} or {defect}?"
        ),
        (
            f"{a} must make an initial recommendation in a repeated interaction with {b}. They are {story['context']}. "
            f"The one-round outcome is {R} points apiece if both {coop}, and {P} points apiece if both {defect}. "
            f"If only one party chooses to {defect}, that party obtains {T} points while the other obtains {S} points. "
            f"{horizon_variants[2]} Should {a} begin by choosing to {coop} or to {defect}?"
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
