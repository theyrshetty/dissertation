"""
Step 6: Translator-plus-Solver arm.

Sends the same paragraph to the LLM as the Direct-AI arm, but the LLM's
ONLY job is to extract structured game parameters -- it never computes a
decision itself. That structured output is validated, converted into a
Scenario, and handed to the same deterministic solver used to build the
ground-truth answer key. The LLM never performs the actual calculation.

Uses Ollama's structured-output mode: passing a JSON schema (generated
from our Pydantic model via .model_json_schema()) as the `format`
parameter. Ollama enforces this via constrained decoding (it zeroes out
the probability of any token that would violate the schema), so the
output is guaranteed syntactically valid JSON matching our schema shape --
though the *values* inside still depend on the model actually reading the
paragraph correctly, which is exactly what we're measuring.

NOTE ON TESTING: same caveat as direct_ai.py -- I cannot reach an Ollama
server from this sandbox. The schema/parsing/conversion logic is unit-
tested against mocked structured objects in translator_solver_test.py.
Run `python3 translator_solver.py` yourself with Ollama running to
confirm the live call path.
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

    Field names are deliberately self-describing (not the game-theory
    shorthand T/R/P/S) after real runs showed a small model extracting the
    right four numbers but binding them to the wrong abstract letter --
    e.g. correctly reading "8" as the mutual-cooperation payoff and "4" as
    the mutual-defection payoff, but writing them into the schema as R=4,
    P=8 (swapped). Naming each field after the concrete situation it
    describes removes the need for the model to hold an arbitrary
    letter-to-meaning mapping in mind while extracting.

    We avoid a union type for horizon (int | "unknown") because that's
    awkward to express in a JSON schema reliably; instead we split it into
    an explicit boolean flag plus two optional fields, only one of which
    is meaningful depending on the flag.
    """
    mutual_cooperation_payoff: int = Field(
        description="The payoff each party earns in the specific case where BOTH parties choose "
                    "to cooperate (e.g. both honor the agreement / both share / both split fairly)."
    )
    mutual_defection_payoff: int = Field(
        description="The payoff each party earns in the specific case where BOTH parties choose "
                    "to defect (e.g. both take extra cargo / both withhold / both over-bill)."
    )
    unilateral_defector_payoff: int = Field(
        description="The payoff earned by whichever party defects, in the specific case where "
                    "ONE party defects while the OTHER cooperates. This is normally the single "
                    "highest number in the scenario."
    )
    unilateral_cooperator_payoff: int = Field(
        description="The payoff earned by whichever party cooperates, in the specific case where "
                    "ONE party defects while the OTHER cooperates. This is normally the single "
                    "lowest number in the scenario."
    )
    horizon_is_fixed: bool = Field(
        description="True if the text states an exact, known number of rounds. "
                    "False if the text instead describes an ongoing/uncertain "
                    "relationship (e.g. a percent chance of continuing)."
    )
    horizon_rounds: Optional[int] = Field(
        default=None,
        description="The exact number of rounds, ONLY if horizon_is_fixed is true. Otherwise null.",
    )
    continuation_probability_percent: Optional[int] = Field(
        default=None,
        description="The percent chance (0-100) the relationship continues after each round, "
                    "ONLY if horizon_is_fixed is false. Otherwise null.",
    )


_EXTRACTION_PROMPT_TEMPLATE = """You are a precise information-extraction system.

Read the scenario below and extract ONLY the structured numeric parameters
of the repeated cooperate/defect game it describes. Do NOT decide what
action to take, do NOT reason about strategy, and do NOT compute anything.
Just extract the numbers exactly as stated in the text. Return as JSON.

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
    consume. Fills in a placeholder discount_factor of 0.0 when the horizon
    is fixed (the solver ignores discount_factor entirely in that branch,
    per solver.py's logic, so the placeholder value is never actually used
    in the ground-truth-comparable computation).
    """
    if extracted.horizon_is_fixed:
        horizon = extracted.horizon_rounds if extracted.horizon_rounds is not None else 0
        discount_factor = 0.0  # unused by solver when horizon is fixed
    else:
        horizon = "unknown"
        pct = extracted.continuation_probability_percent
        discount_factor = (pct / 100) if pct is not None else 0.0

    return Scenario(
        T=extracted.unilateral_defector_payoff,
        R=extracted.mutual_cooperation_payoff,
        P=extracted.mutual_defection_payoff,
        S=extracted.unilateral_cooperator_payoff,
        num_players=2,
        horizon=horizon,
        discount_factor=discount_factor,
    )


def validate_extraction(extracted: ExtractedParams) -> list:
    """
    Returns a list of human-readable warning strings for structurally
    invalid extractions (e.g. payoff ordering violated, or horizon fields
    inconsistent with the horizon_is_fixed flag). An empty list means the
    extraction is internally well-formed enough to solve meaningfully.
    This does NOT check the extraction against the true scenario (that's
    the scoring function's job) -- only internal consistency.
    """
    warnings = []
    T = extracted.unilateral_defector_payoff
    R = extracted.mutual_cooperation_payoff
    P = extracted.mutual_defection_payoff
    S = extracted.unilateral_cooperator_payoff

    if not (T > R > P > S):
        warnings.append(
            f"Payoff ordering violated: expected unilateral_defector_payoff > "
            f"mutual_cooperation_payoff > mutual_defection_payoff > "
            f"unilateral_cooperator_payoff, got {T} > {R} > {P} > {S}."
        )

    if extracted.horizon_is_fixed:
        if extracted.horizon_rounds is None:
            warnings.append("horizon_is_fixed=True but horizon_rounds is missing.")
        elif extracted.horizon_rounds < 1:
            warnings.append(f"horizon_rounds must be >= 1, got {extracted.horizon_rounds}.")
        if extracted.continuation_probability_percent is not None:
            warnings.append(
                "horizon_is_fixed=True but continuation_probability_percent was also "
                f"set (got {extracted.continuation_probability_percent}, should be null) -- "
                "the model may be internally inconsistent about horizon type."
            )
    else:
        if extracted.continuation_probability_percent is None:
            warnings.append("horizon_is_fixed=False but continuation_probability_percent is missing.")
        elif not (0 <= extracted.continuation_probability_percent <= 100):
            warnings.append(
                f"continuation_probability_percent must be in [0,100], "
                f"got {extracted.continuation_probability_percent}."
            )
        if extracted.horizon_rounds is not None:
            warnings.append(
                "horizon_is_fixed=False but horizon_rounds was also set "
                f"(got {extracted.horizon_rounds}, should be null) -- the model may be "
                "internally inconsistent about horizon type."
            )

    return warnings


def run_translator_plus_solver(paragraph: str, client: Optional[ollama.Client] = None,
                                model: Optional[str] = None) -> dict:
    """
    Returns:
      {
        "action": "cooperate" | "defect" | None,   # None if extraction failed entirely
        "extracted_params": ExtractedParams | None,
        "extraction_warnings": list[str],           # internal-consistency issues, if any
        "raw_response_text": str,
        "solver_detail": dict | None,
        "model": str,
      }
    """
    client = client or _get_client()
    model = model or _get_model_name()
    prompt = build_extraction_prompt(paragraph)

    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=ExtractedParams.model_json_schema(),
        options={"temperature": 0.0},
    )
    raw_text = response["message"]["content"] or ""

    try:
        extracted = ExtractedParams.model_validate_json(raw_text)
    except (ValidationError, json.JSONDecodeError) as e:
        # The schema constrains token-level structure, but a small/weak
        # model can still produce syntactically-valid JSON that fails
        # Pydantic's stricter validation (e.g. a field of the wrong type
        # slipping through). Treat this the same as a parse failure.
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
    print(f"True params: mutual_cooperation={case['scenario'].R} mutual_defection={case['scenario'].P} "
          f"unilateral_defector={case['scenario'].T} unilateral_cooperator={case['scenario'].S} "
          f"horizon={case['scenario'].horizon} delta={case['scenario'].discount_factor}")
    print()

    result = run_translator_plus_solver(case["paragraph"])
    print(f"Model used: {result['model']}")
    print("Extracted params:", result["extracted_params"])
    print("Extraction warnings:", result["extraction_warnings"])
    print(f"Translator-plus-Solver action: {result['action']}")