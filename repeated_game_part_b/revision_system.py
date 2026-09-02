"""The system under test: structured model revision plus deterministic solver."""

import json
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from part_a_imports import PART_A_DIR  # noqa: F401
from translator_solver import ExtractedParams, _get_client, _get_model_name
from dependency_graph import DERIVED_FIELDS, structure_record


class RevisionResponse(BaseModel):
    updated_params: ExtractedParams
    recomputed_fields: list[str] = Field(
        description="Names of DERIVED fields recomputed after the correction. Valid names: " + ", ".join(DERIVED_FIELDS)
    )


PROMPT = """You revise a structured repeated-game extraction after one correction.

Original scenario paragraph:
{paragraph}

Original extracted structure (authoritative starting state):
{original}

Follow-up correction:
{follow_up}

Return the complete updated extraction. Also report ONLY the derived fields
that must be recomputed because of this correction. The valid derived field
names are critical_discount_factor and solved_outcome. Do not include input
fields merely because their values were updated.
"""


def run_revision_system(paragraph: str, original_params, follow_up: str,
                        client=None, model: Optional[str] = None) -> dict:
    client = client or _get_client()
    model = model or _get_model_name()
    prompt = PROMPT.format(
        paragraph=paragraph,
        original=json.dumps(original_params.model_dump()),
        follow_up=follow_up,
    )
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=RevisionResponse.model_json_schema(),
        options={"temperature": 0.0},
    )
    raw_text = response["message"]["content"] or ""
    try:
        parsed = RevisionResponse.model_validate_json(raw_text)
    except (ValidationError, json.JSONDecodeError) as exc:
        return {"updated_structure": None, "recomputed_fields": [], "raw_response_text": raw_text,
                "error": f"Response could not be validated: {exc}", "model": model}
    return {"updated_structure": structure_record(parsed.updated_params),
            "recomputed_fields": parsed.recomputed_fields, "raw_response_text": raw_text,
            "error": None, "model": model}
