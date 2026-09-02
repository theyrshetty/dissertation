"""
Step 5: Direct-AI arm.

Sends the generated paragraph straight to the LLM and asks it to recommend
the current-period action, with NO structure extraction and NO solver
involved. The model reasons directly from natural language, exactly as an
unaided user query would.

Runs against a local or remote Ollama server -- no API key, no rate
limits, no daily quota. Reads two settings from .env (both optional, with
sensible defaults):

  OLLAMA_HOST   e.g. http://localhost:11434 (default), or
                http://<hpc-compute-node-hostname>:11434 for a cluster run
  OLLAMA_MODEL  e.g. llama3.2:3b (default), or a larger model for a real
                run on HPC GPU, e.g. qwen2.5:14b-instruct or llama3.1:8b

NOTE ON TESTING: this file makes live calls to an Ollama server, which
isn't reachable from my sandbox (no such server running here, and my
network egress is allowlisted to a fixed set of domains). I've unit-tested
the deterministic parts (prompt construction, answer extraction) with a
mocked response in direct_ai_test.py. Run `python3 direct_ai.py` yourself
with Ollama running to confirm the live call path.
"""

import os
import re
from typing import Optional

from dotenv import load_dotenv
import ollama

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

_PROMPT_TEMPLATE = """You are an expert in sequential decision-making and dynamic optimization.

Read the following scenario carefully and decide what action should be taken THIS period.

Scenario:
{paragraph}

Think through the tradeoffs if it helps you, but you MUST end your response
with a single final line in exactly this format (nothing after it):

FINAL ANSWER: <CAUTIOUS or AGGRESSIVE>
"""


def _get_client(host: Optional[str] = None) -> ollama.Client:
    """
    Build an Ollama client. Loads OLLAMA_HOST from .env if not passed
    explicitly, defaulting to a local server.
    """
    if host is None:
        load_dotenv()
        host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
    return ollama.Client(host=host)


def _get_model_name() -> str:
    load_dotenv()
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def build_prompt(paragraph: str) -> str:
    return _PROMPT_TEMPLATE.format(paragraph=paragraph)


def extract_action(raw_text: str) -> Optional[str]:
    """
    Pull a clean 'cautious' or 'aggressive' out of the model's raw response.

    Strategy:
      1. Look for the required "FINAL ANSWER: <...>" line first (most
         reliable, since we explicitly instruct the model to produce it).
      2. If that's missing/malformed, fall back to scanning the whole
         response for the last standalone mention of either word.
      3. Return None if neither approach finds a clear answer (this should
         be logged/counted as a parse failure by the runner, NOT silently
         coerced into a guess).
    """
    match = re.search(r"FINAL ANSWER:\s*(CAUTIOUS|AGGRESSIVE)", raw_text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    occurrences = list(re.finditer(r"\b(cautious|aggressive)\b", raw_text, re.IGNORECASE))
    if occurrences:
        return occurrences[-1].group(1).lower()

    return None


def run_direct_ai(paragraph: str, client: Optional[ollama.Client] = None,
                   model: Optional[str] = None) -> dict:
    """
    Send `paragraph` to the local/remote Ollama model and return:
      {
        "action": "cautious" | "aggressive" | None,   # None = parse failure
        "raw_response": str,
        "model": str,                                 # exact model name used
      }
    """
    client = client or _get_client()
    model = model or _get_model_name()
    prompt = build_prompt(paragraph)

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},  # deterministic-as-possible for reproducible scoring
        )
    except Exception as e:
        return {"action": None, "raw_response": f"[ERROR: {type(e).__name__}: {e!r}]", "model": model}

    raw_text = response["message"]["content"] or ""
    action = extract_action(raw_text)

    return {"action": action, "raw_response": raw_text, "model": model}


if __name__ == "__main__":
    # Live smoke test. Requires Ollama running (locally or via OLLAMA_HOST)
    # with the model in OLLAMA_MODEL already pulled (e.g. `ollama pull llama3.2:3b`).
    from test_case import generate_test_case
    import random

    rng = random.Random(1)
    case = generate_test_case(rng=rng)
    print("Paragraph:")
    print(case["paragraph"])
    print()
    print(f"Ground truth: {case['ground_truth']}")
    print()

    result = run_direct_ai(case["paragraph"])
    print(f"Model used: {result['model']}")
    print("Direct-AI raw response:")
    print(result["raw_response"])
    print()
    print(f"Extracted action: {result['action']}")
