"""
Step 8: Runner.

Generates N test cases, runs the Direct-AI arm and the Translator-plus-
Solver arm on each, scores both against the ground truth, and prints an
aggregate report. Also writes every per-case result to a JSONL log file so
individual cases can be inspected later (e.g. to read exactly which
paragraphs tripped up which arm).

Requires Ollama running (locally, or on an HPC node with OLLAMA_HOST set
in .env) with the model in OLLAMA_MODEL already pulled. Cannot be executed
from my sandbox (no Ollama server reachable here). Run this yourself:

    python3 runner.py --n 20

Start small (n=20) as a sanity check before scaling to several hundred.
Since local inference has no rate limit or daily quota, --sleep defaults
to 0 -- only set it if you're running against a shared/remote Ollama
instance and want to space out requests.
"""

import argparse
import json
import time
from dataclasses import asdict

from test_case import generate_test_case
from direct_ai import run_direct_ai, _get_client as get_direct_client
from translator_solver import run_translator_plus_solver
from scoring import score_case, aggregate_scores, format_report


def _scenario_to_jsonable(scenario) -> dict:
    d = asdict(scenario)
    return d


def _extracted_params_to_jsonable(extracted) -> dict:
    if extracted is None:
        return None
    return extracted.model_dump()


def run_evaluation(n: int, log_path: str = "run_log.jsonl", seed: int = None, sleep_between_calls: float = 0.0):
    import random
    rng = random.Random(seed) if seed is not None else random.Random()

    # Both arms use the same underlying Ollama client (one Client instance,
    # reused across calls, rather than reconnecting every time).
    client = get_direct_client()

    scored_cases = []
    log_records = []

    for i in range(n):
        case = generate_test_case(rng=rng)

        try:
            direct_result = run_direct_ai(case["paragraph"], client=client)
        except Exception as e:
            direct_result = {"action": None, "raw_response": f"[ERROR calling Direct-AI: {e}]"}

        if sleep_between_calls:
            time.sleep(sleep_between_calls)

        try:
            translator_result = run_translator_plus_solver(case["paragraph"], client=client)
        except Exception as e:
            translator_result = {
                "action": None,
                "extracted_params": None,
                "extraction_warnings": [f"ERROR calling Translator-plus-Solver: {e}"],
                "raw_response_text": "",
                "solver_detail": None,
            }

        if sleep_between_calls:
            time.sleep(sleep_between_calls)

        scored = score_case(case, direct_result, translator_result)
        scored_cases.append(scored)

        log_records.append({
            "index": i,
            "paragraph": case["paragraph"],
            "scenario": _scenario_to_jsonable(case["scenario"]),
            "ground_truth": case["ground_truth"],
            "direct_ai": {
                "action": direct_result.get("action"),
                "raw_response": direct_result.get("raw_response"),
                "model": direct_result.get("model"),
            },
            "translator_plus_solver": {
                "action": translator_result.get("action"),
                "extracted_params": _extracted_params_to_jsonable(translator_result.get("extracted_params")),
                "extraction_warnings": translator_result.get("extraction_warnings"),
                "model": translator_result.get("model"),
            },
            "score": scored,
        })

        status_direct = "OK" if scored["direct_correct"] else "WRONG"
        status_translator = "OK" if scored["translator_correct"] else "WRONG"
        print(f"[{i+1}/{n}] truth={scored['ground_truth']:<9} "
              f"direct={str(scored['direct_action']):<10}({status_direct:<5}) "
              f"translator={str(scored['translator_action']):<10}({status_translator})")

    with open(log_path, "w") as f:
        for rec in log_records:
            f.write(json.dumps(rec) + "\n")

    report = aggregate_scores(scored_cases)
    print()
    print(format_report(report))
    print()
    print(f"Full per-case log written to: {log_path}")

    return report, log_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Direct-AI vs Translator-plus-Solver evaluation.")
    parser.add_argument("--n", type=int, default=20, help="Number of test cases to generate and evaluate.")
    parser.add_argument("--log", type=str, default="run_log.jsonl", help="Path to write the per-case JSONL log.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible scenario generation.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls (rate-limit safety).")
    args = parser.parse_args()

    run_evaluation(n=args.n, log_path=args.log, seed=args.seed, sleep_between_calls=args.sleep)