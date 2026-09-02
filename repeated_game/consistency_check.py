"""Consistency-across-paraphrases evaluation.

For each generated scenario, this module renders 2--3 differently worded
paragraphs that retain exactly the same sampled parameters.  It then runs both
evaluation arms on every wording and scores consistency by exact equality of
the arms' extracted actions.  This is deliberately separate from accuracy:
the answer key is not used when computing consistency.
"""

import argparse
import json
import random
import time
from dataclasses import asdict
from typing import Callable, Optional

from direct_ai import _get_client as get_direct_client
from direct_ai import run_direct_ai
from generator import generate_scenario
from renderer import render_paraphrases
from translator_solver import run_translator_plus_solver


ARM_DIRECT = "direct_ai"
ARM_TRANSLATOR = "translator_plus_solver"


def _action_is_consistent(actions: list[Optional[str]]) -> bool:
    """Return true only when every wording yielded the exact same action.

    ``None`` is retained as an extracted outcome rather than discarded. Thus a
    run is consistent if all variants parse-failed identically, and inconsistent
    if some variants parse while others fail. This is the literal, automated
    interpretation of "same answer across all wordings."

    NOTE: this only sees the extracted action, not *why* it was None. A model
    that returned unparseable text and a model that was never reachable both
    show up as None here. Callers that care about the difference (see
    ``_is_connection_error``) must filter before calling this.
    """
    return len(actions) > 0 and len(set(actions)) == 1


def _is_connection_error(raw_response: Optional[str]) -> bool:
    """Detect calls that never reached the model at all (vs. a real reply the
    parser simply couldn't extract an action from).

    This matters because ``_action_is_consistent`` treats repeated ``None``s
    as "consistent" -- correct when every wording produced a real-but-
    unparseable response, but actively misleading when every wording just
    failed to connect to Ollama (e.g. it wasn't running yet). A scenario built
    entirely out of connection errors says nothing about the model's actual
    consistency and must not be counted as a data point either way.
    """
    return bool(raw_response) and raw_response.startswith("[ERROR calling")


def aggregate_consistency(scenario_results: list[dict]) -> dict:
    """Compute arm-level rates from per-scenario consistency flags only.

    Rows with ``contaminated=True`` (every wording for that arm/scenario
    failed to connect to the model, rather than the model actually
    answering) are excluded from both the numerator and denominator -- they
    are not evidence of consistency OR inconsistency, so counting them either
    way would bias the rate.
    """
    def rate(arm: str) -> Optional[float]:
        flags = [row["consistent"] for row in scenario_results
                  if row["arm"] == arm and not row.get("contaminated", False)]
        return (sum(flags) / len(flags)) if flags else None

    n_contaminated = len({row["scenario_id"] for row in scenario_results
                           if row.get("contaminated", False)})

    return {
        "overall": {
            "n": len({row["scenario_id"] for row in scenario_results}),
            "n_excluded_contaminated": n_contaminated,
            "direct_consistency_rate": rate(ARM_DIRECT),
            "translator_consistency_rate": rate(ARM_TRANSLATOR),
        }
    }


def format_consistency_report(report: dict) -> str:
    """Format the two rates in the style of ``scoring.format_report``."""
    overall = report["overall"]

    def fmt_pct(value: Optional[float]) -> str:
        return f"{value * 100:.1f}%" if value is not None else "n/a"

    lines = [
        "=== CONSISTENCY ACROSS PARAPHRASES ===",
        f"  All scenarios (n={overall['n']}):",
        f"      Direct-AI consistency:              {fmt_pct(overall['direct_consistency_rate'])}",
        f"      Translator-plus-Solver consistency: {fmt_pct(overall['translator_consistency_rate'])}",
    ]
    if overall.get("n_excluded_contaminated"):
        lines.append(
            f"  WARNING: {overall['n_excluded_contaminated']} scenario(s) excluded from "
            "the rates above because every wording failed to connect to the "
            "model (not a real answer either way). See raw_records / the log "
            "for scenario_ids where raw_response starts with '[ERROR calling'."
        )
    return "\n".join(lines)


def print_consistency_report(report: dict) -> None:
    """Print a table-ready, side-by-side consistency summary."""
    print(format_consistency_report(report))


def run_consistency_check(
    sample_size: int = 50,
    num_variants: int = 3,
    log_path: str = "consistency_log.jsonl",
    seed: Optional[int] = None,
    sleep_between_calls: float = 0.0,
    *,
    direct_runner: Callable = run_direct_ai,
    translator_runner: Callable = run_translator_plus_solver,
    client=None,
) -> tuple[dict, list[dict], list[dict]]:
    """Evaluate both arms on paraphrases of ``sample_size`` sampled scenarios.

    The returned tuple is ``(report, per_scenario_results, raw_records)``.
    ``raw_records`` is also written as JSONL, one row per scenario/arm/wording,
    with the paragraph and normalized extracted answer retained for manual
    disagreement spot checks.  ``per_scenario_results`` contains the required
    Boolean ``consistent`` flag for each scenario and arm.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")
    if num_variants not in (2, 3):
        raise ValueError("num_variants must be 2 or 3.")

    rng = random.Random(seed) if seed is not None else random.Random()
    client = client or get_direct_client()

    # Preflight: fail fast and loudly if Ollama isn't actually reachable,
    # rather than burning through `sample_size` scenarios that all silently
    # come back as connection-error contamination (see run from 2026-07-21,
    # scenarios 0-4, where this happened and inflated the reported rates).
    # Only applies when using the real Ollama-backed runners -- tests that
    # inject mocked direct_runner/translator_runner (and a dummy client)
    # intentionally skip this, since there's nothing real to reach.
    using_real_runners = direct_runner is run_direct_ai and translator_runner is run_translator_plus_solver
    if using_real_runners:
        try:
            client.list()
        except Exception as exc:
            raise RuntimeError(
                "Could not reach Ollama before starting the consistency check "
                f"(error: {exc}). Start it with `ollama serve` (and confirm "
                "`ollama pull llama3.2:3b` has been run) before re-running this "
                "script -- otherwise every scenario will silently fail and get "
                "excluded from the report."
            ) from exc

    raw_records = []
    per_scenario_results = []

    for scenario_id in range(sample_size):
        # Sample exactly once. All wording variants below refer to this object.
        scenario = generate_scenario(rng=rng)
        paragraphs = render_paraphrases(scenario, num_variants=num_variants, rng=rng)
        actions_by_arm = {ARM_DIRECT: [], ARM_TRANSLATOR: []}
        scenario_records = {ARM_DIRECT: [], ARM_TRANSLATOR: []}

        for wording_id, paragraph in enumerate(paragraphs, start=1):
            try:
                direct = direct_runner(paragraph, client=client)
            except Exception as exc:
                direct = {"action": None, "raw_response": f"[ERROR calling Direct-AI: {exc}]"}
            actions_by_arm[ARM_DIRECT].append(direct.get("action"))
            scenario_records[ARM_DIRECT].append({
                "scenario_id": scenario_id,
                "arm": ARM_DIRECT,
                "wording_id": wording_id,
                "paragraph": paragraph,
                "extracted_answer": direct.get("action"),
                "raw_response": direct.get("raw_response"),
                "model": direct.get("model"),
                "scenario": asdict(scenario),
            })
            if sleep_between_calls:
                time.sleep(sleep_between_calls)

            try:
                translator = translator_runner(paragraph, client=client)
            except Exception as exc:
                translator = {"action": None, "raw_response_text": f"[ERROR calling Translator-plus-Solver: {exc}]"}
            actions_by_arm[ARM_TRANSLATOR].append(translator.get("action"))
            scenario_records[ARM_TRANSLATOR].append({
                "scenario_id": scenario_id,
                "arm": ARM_TRANSLATOR,
                "wording_id": wording_id,
                "paragraph": paragraph,
                "extracted_answer": translator.get("action"),
                "raw_response": translator.get("raw_response_text"),
                "model": translator.get("model"),
                "scenario": asdict(scenario),
            })
            if sleep_between_calls:
                time.sleep(sleep_between_calls)

        for arm in (ARM_DIRECT, ARM_TRANSLATOR):
            consistent = _action_is_consistent(actions_by_arm[arm])
            # Contaminated = every wording for this arm/scenario failed to
            # reach the model at all. That's not evidence of consistency
            # (nor inconsistency) and must be excluded from the rate, not
            # silently counted as "consistent" just because None == None.
            contaminated = all(
                _is_connection_error(rec["raw_response"])
                for rec in scenario_records[arm]
            )
            per_scenario_results.append({
                "scenario_id": scenario_id,
                "arm": arm,
                "answers": actions_by_arm[arm],
                "consistent": consistent,
                "contaminated": contaminated,
            })
            for record in scenario_records[arm]:
                record["consistent"] = consistent
                record["contaminated"] = contaminated
                raw_records.append(record)

        direct_flag = per_scenario_results[-2]["consistent"]
        translator_flag = per_scenario_results[-1]["consistent"]
        print(f"[{scenario_id + 1}/{sample_size}] "
              f"direct_consistent={direct_flag} "
              f"translator_consistent={translator_flag}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        for record in raw_records:
            log_file.write(json.dumps(record) + "\n")

    report = aggregate_consistency(per_scenario_results)
    print()
    print_consistency_report(report)
    print()
    print(f"Full per-wording audit log written to: {log_path}")
    return report, per_scenario_results, raw_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model-answer consistency across scenario paraphrases.")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of sampled scenarios to evaluate.")
    parser.add_argument("--variants", type=int, choices=(2, 3), default=3, help="Number of wordings per scenario.")
    parser.add_argument("--log", default="consistency_log.jsonl", help="JSONL path for per-wording audit records.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible scenario sampling.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between model calls.")
    args = parser.parse_args()

    run_consistency_check(
        sample_size=args.sample_size,
        num_variants=args.variants,
        log_path=args.log,
        seed=args.seed,
        sleep_between_calls=args.sleep,
    )