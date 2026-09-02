"""
Step 6: Translator-plus-Solver arm.

Sends the same paragraph to the LLM as the Direct-AI arm, but the LLM's
ONLY job is to extract structured MDP parameters -- it never computes a
decision itself. That structured output is validated, converted into a
Scenario, and handed to the same deterministic solver used to build the
ground-truth answer key. The LLM never performs the dynamic programming.

Uses Ollama's structured-output mode: passing a JSON schema (generated
from our Pydantic model via .model_json_schema()) as the `format`
parameter. Ollama enforces this via constrained decoding, so the output is
guaranteed syntactically valid JSON matching our schema shape -- though the
*values* inside still depend on the model actually reading the paragraph
correctly, which is exactly what we're measuring.

NOTE ON TESTING: same caveat as direct_ai.py -- I cannot reach an Ollama
server from this sandbox. The schema/parsing/conversion logic is unit-
tested against mocked structured objects in translator_solver_test.py. Run
`python3 translator_solver.py` yourself with Ollama running to confirm the
live call path.
"""

import os
import json
from typing import Optional

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import ollama

from generator import Scenario
from solver import solve

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


class ExtractedParams(BaseModel):
    """
    Structured extraction target.

    Field names are deliberately self-describing (not abstract shorthand)
    for the same reason as the repeated-game module's ExtractedParams: a
    small model can correctly read the right numbers off the page but bind
    them to the wrong slot if the slot names require holding an arbitrary
    letter-to-meaning mapping in mind. Naming each field after the concrete
    (state, action) situation it describes removes that failure mode.

    Two deliberate design choices, both added after a live run against
    Ollama exposed real problems with the naive version:

    1. UNITS. The paragraph states rewards as "11 thousand dollars"; the
       first version of this schema just said "the reward earned", and the
       model frequently (reasonably) converted that to a raw dollar amount
       (11000) instead of echoing the number as stated (11). That silently
       broke every extraction-accuracy comparison even when the model had
       read the paragraph correctly.

       A second version tried to instruct the model NOT to do that
       conversion ("respond with 11, not 11000"). A live run against
       llama3.2:3b showed this instruction is simply not reliably followed
       by a model this size -- it kept returning 11000 regardless. Rather
       than keep fighting a small model's natural behavior, this version
       goes with it: the field descriptions now explicitly ask for the
       full dollar amount (matching what the model does unprompted), and
       scoring.py compares against `scenario_value * 1000` accordingly.
       Lesson: when a small model consistently does A instead of the
       instructed B, it's usually more robust to redefine the target as A
       than to keep re-wording the instruction for B.

    2. NO NULLABLE FIELDS. An earlier version used
       `Optional[int] = None` for horizon_periods /
       continuation_probability_percent (only one is meaningful, depending
       on horizon_is_fixed). In practice, Ollama's JSON-schema-constrained
       decoding handled the resulting `anyOf: [integer, null]` schema
       unreliably: some calls failed outright (empty-message exceptions),
       others silently forced a spurious integer into the field that
       should have been null. Both fields are now required ints with a
       documented sentinel (0) for "not applicable", which every JSON
       schema backend can represent as a plain `integer` type. NOTE: a
       live run also showed the model doesn't reliably zero out the
       inapplicable field either (e.g. still reporting a nonzero
       continuation_probability_percent when horizon_is_fixed=True) --
       this is now understood to be a genuine small-model instruction-
       following limitation, not a pipeline bug, and it's already handled
       safely: extracted_params_to_scenario() below only ever reads
       whichever field horizon_is_fixed says is the relevant one, so a
       stray value in the other field cannot corrupt the solve -- it only
       shows up as a validate_extraction() warning, which is exactly the
       point of tracking that warning separately from action correctness.
    """
    low_state_cautious_reward: int = Field(
        description="The reward earned THIS period if currently in the Low state and the Cautious action is "
                    "chosen, as a full dollar amount. Example: if the text says 'earns 11 thousand dollars', "
                    "respond with 11000."
    )
    low_state_aggressive_reward: int = Field(
        description="The reward earned THIS period if currently in the Low state and the Aggressive action is "
                    "chosen, as a full dollar amount (e.g. 11000 for 'earns 11 thousand dollars')."
    )
    high_state_cautious_reward: int = Field(
        description="The reward earned THIS period if currently in the High state and the Cautious action is "
                    "chosen, as a full dollar amount (e.g. 11000 for 'earns 11 thousand dollars')."
    )
    high_state_aggressive_reward: int = Field(
        description="The reward earned THIS period if currently in the High state and the Aggressive action is "
                    "chosen, as a full dollar amount (e.g. 11000 for 'earns 11 thousand dollars')."
    )
    cautious_advance_probability_percent: int = Field(
        description="The percent chance (0-100) of moving from the Low state into the High state next period, "
                    "given the Cautious action is chosen while in the Low state."
    )
    aggressive_advance_probability_percent: int = Field(
        description="The percent chance (0-100) of moving from the Low state into the High state next period, "
                    "given the Aggressive action is chosen while in the Low state."
    )
    cautious_stay_probability_percent: int = Field(
        description="The percent chance (0-100) of remaining in the High state next period, "
                    "given the Cautious action is chosen while already in the High state."
    )
    aggressive_stay_probability_percent: int = Field(
        description="The percent chance (0-100) of remaining in the High state next period, "
                    "given the Aggressive action is chosen while already in the High state."
    )
    horizon_is_fixed: bool = Field(
        description="True if the text states an exact, known number of remaining periods. "
                    "False if the text instead describes an indefinitely continuing process "
                    "(e.g. a percent chance of continuing each period)."
    )
    horizon_periods: int = Field(
        description="If horizon_is_fixed is true: the exact number of periods remaining, as stated in the text. "
                    "If horizon_is_fixed is false: this field is not applicable -- respond with exactly 0."
    )
    continuation_probability_percent: int = Field(
        description="If horizon_is_fixed is false: the percent chance (0-100) that operations continue into "
                    "another period, as stated in the text. If horizon_is_fixed is true: this field is not "
                    "applicable -- respond with exactly 0."
    )


_EXTRACTION_PROMPT_TEMPLATE = """You are a precise information-extraction system.

Read the scenario below and extract ONLY the structured numeric parameters
of the sequential-planning problem it describes. Do NOT decide what action
to take, do NOT reason about strategy, and do NOT compute anything. Just
extract the numbers exactly as stated in the text. Return as JSON.

Scenario:
{paragraph}
"""


def _get_client(host: Optional[str] = None) -> ollama.Client:
    if host is None:
        load_dotenv()
        host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
    return ollama.Client(host=host)


def _get_model_name() -> str:
    load_dotenv()
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def build_extraction_prompt(paragraph: str) -> str:
    return _EXTRACTION_PROMPT_TEMPLATE.format(paragraph=paragraph)


def extracted_params_to_scenario(extracted: ExtractedParams) -> Scenario:
    """
    Convert the LLM's extracted parameters into a Scenario the solver can
    consume. horizon_periods / continuation_probability_percent use the
    documented 0-sentinel for "not applicable" (see ExtractedParams
    docstring); discount_factor is a placeholder 0.0 when the horizon is
    fixed, since the solver ignores discount_factor entirely in that
    branch (per solver.py's logic), so the placeholder is never actually
    used in the ground-truth-comparable computation.
    """
    if extracted.horizon_is_fixed:
        horizon = extracted.horizon_periods
        discount_factor = 0.0  # unused by solver when horizon is fixed
    else:
        horizon = "unknown"
        discount_factor = extracted.continuation_probability_percent / 100

    return Scenario(
        r_low_cautious=extracted.low_state_cautious_reward,
        r_low_aggressive=extracted.low_state_aggressive_reward,
        r_high_cautious=extracted.high_state_cautious_reward,
        r_high_aggressive=extracted.high_state_aggressive_reward,
        p_high_low_cautious=extracted.cautious_advance_probability_percent / 100,
        p_high_low_aggressive=extracted.aggressive_advance_probability_percent / 100,
        p_high_high_cautious=extracted.cautious_stay_probability_percent / 100,
        p_high_high_aggressive=extracted.aggressive_stay_probability_percent / 100,
        num_states=2,
        num_actions=2,
        start_state="Low",
        horizon=horizon,
        discount_factor=discount_factor,
    )


def validate_extraction(extracted: ExtractedParams) -> list:
    """
    Returns a list of human-readable warning strings for structurally
    invalid extractions (e.g. reward/probability ordering violated, horizon
    fields inconsistent with the horizon_is_fixed flag, or a percent field
    out of [0,100]). An empty list means the extraction is internally
    well-formed enough to solve meaningfully. This does NOT check the
    extraction against the true scenario (that's the scoring function's
    job) -- only internal consistency.
    """
    warnings = []

    r_lc = extracted.low_state_cautious_reward
    r_la = extracted.low_state_aggressive_reward
    r_hc = extracted.high_state_cautious_reward
    r_ha = extracted.high_state_aggressive_reward

    if not (r_ha > r_hc):
        warnings.append(
            f"Reward ordering violated: expected high_state_aggressive_reward > "
            f"high_state_cautious_reward, got {r_ha} vs {r_hc}."
        )
    if not (r_la > r_lc):
        warnings.append(
            f"Reward ordering violated: expected low_state_aggressive_reward > "
            f"low_state_cautious_reward, got {r_la} vs {r_lc}."
        )
    if not (r_hc > r_lc):
        warnings.append(
            f"Reward ordering violated: expected high_state_cautious_reward > "
            f"low_state_cautious_reward, got {r_hc} vs {r_lc}."
        )
    if not (r_ha > r_la):
        warnings.append(
            f"Reward ordering violated: expected high_state_aggressive_reward > "
            f"low_state_aggressive_reward, got {r_ha} vs {r_la}."
        )

    for name in ("cautious_advance_probability_percent", "aggressive_advance_probability_percent",
                 "cautious_stay_probability_percent", "aggressive_stay_probability_percent"):
        val = getattr(extracted, name)
        if not (0 <= val <= 100):
            warnings.append(f"{name} must be in [0,100], got {val}.")

    if extracted.cautious_advance_probability_percent <= extracted.aggressive_advance_probability_percent:
        warnings.append(
            "Probability ordering violated: expected cautious_advance_probability_percent > "
            f"aggressive_advance_probability_percent, got "
            f"{extracted.cautious_advance_probability_percent} vs "
            f"{extracted.aggressive_advance_probability_percent}."
        )
    if extracted.cautious_stay_probability_percent <= extracted.aggressive_stay_probability_percent:
        warnings.append(
            "Probability ordering violated: expected cautious_stay_probability_percent > "
            f"aggressive_stay_probability_percent, got "
            f"{extracted.cautious_stay_probability_percent} vs "
            f"{extracted.aggressive_stay_probability_percent}."
        )

    if extracted.horizon_is_fixed:
        if extracted.horizon_periods < 1:
            warnings.append(
                f"horizon_is_fixed=True but horizon_periods is {extracted.horizon_periods} "
                "(expected >= 1)."
            )
        if extracted.continuation_probability_percent != 0:
            warnings.append(
                "horizon_is_fixed=True but continuation_probability_percent was also "
                f"set (got {extracted.continuation_probability_percent}, should be the 0 sentinel) -- "
                "the model may be internally inconsistent about horizon type."
            )
    else:
        if not (0 <= extracted.continuation_probability_percent <= 100):
            warnings.append(
                f"continuation_probability_percent must be in [0,100], "
                f"got {extracted.continuation_probability_percent}."
            )
        if extracted.horizon_periods != 0:
            warnings.append(
                "horizon_is_fixed=False but horizon_periods was also set "
                f"(got {extracted.horizon_periods}, should be the 0 sentinel) -- the model may be "
                "internally inconsistent about horizon type."
            )

    return warnings


def run_translator_plus_solver(paragraph: str, client: Optional[ollama.Client] = None,
                                model: Optional[str] = None) -> dict:
    """
    Returns:
      {
        "action": "cautious" | "aggressive" | None,   # None if extraction failed entirely
        "extracted_params": ExtractedParams | None,
        "extraction_warnings": list[str],
        "raw_response_text": str,
        "solver_detail": dict | None,
        "model": str,
      }
    """
    client = client or _get_client()
    model = model or _get_model_name()
    prompt = build_extraction_prompt(paragraph)

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=ExtractedParams.model_json_schema(),
            options={"temperature": 0.0},
        )
    except Exception as e:
        # Surface type(e).__name__ explicitly -- some Ollama/httpx errors
        # (e.g. a rejected `format` schema, a dropped connection) have an
        # empty str(e), which otherwise silently turns into an
        # undiagnosable "ERROR calling Translator-plus-Solver: " with no
        # information in it. Also try client.chat() without a `format`
        # first if you see this recur -- it isolates whether the schema
        # itself is the problem.
        return {
            "action": None,
            "extracted_params": None,
            "extraction_warnings": [
                f"Ollama call failed: {type(e).__name__}: {e!r}"
            ],
            "raw_response_text": "",
            "solver_detail": None,
            "model": model,
        }

    raw_text = response["message"]["content"] or ""

    try:
        extracted = ExtractedParams.model_validate_json(raw_text)
    except (ValidationError, json.JSONDecodeError) as e:
        return {
            "action": None,
            "extracted_params": None,
            "extraction_warnings": [f"Response could not be validated against ExtractedParams schema: {e}"],
            "raw_response_text": raw_text,
            "solver_detail": None,
            "model": model,
        }

    warnings = validate_extraction(extracted)
    scenario = extracted_params_to_scenario(extracted)
    solver_result = solve(scenario)

    return {
        "action": solver_result["action"],
        "extracted_params": extracted,
        "extraction_warnings": warnings,
        "raw_response_text": raw_text,
        "solver_detail": solver_result,
        "model": model,
    }


if __name__ == "__main__":
    # Live smoke test. Requires Ollama running with the model already pulled.
    from test_case import generate_test_case
    import random

    rng = random.Random(1)
    case = generate_test_case(rng=rng)
    print("Paragraph:")
    print(case["paragraph"])
    print()
    print(f"Ground truth: {case['ground_truth']}")
    s = case["scenario"]
    print(f"True params: low=({s.r_low_cautious},{s.r_low_aggressive}) "
          f"high=({s.r_high_cautious},{s.r_high_aggressive}) "
          f"p_advance=({s.p_high_low_cautious},{s.p_high_low_aggressive}) "
          f"p_stay=({s.p_high_high_cautious},{s.p_high_high_aggressive}) "
          f"horizon={s.horizon} discount={s.discount_factor}")
    print()

    result = run_translator_plus_solver(case["paragraph"])
    print(f"Model used: {result['model']}")
    print("Extracted params:", result["extracted_params"])
    print("Extraction warnings:", result["extraction_warnings"])
    print(f"Translator-plus-Solver action: {result['action']}")
