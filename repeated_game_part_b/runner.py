"""Batch runner for Part B's DAG-based revision evaluation."""

import argparse
from dataclasses import asdict
import json
import random
import time

from part_a_imports import PART_A_DIR  # noqa: F401
from test_case import generate_test_case
from dependency_graph import downstream_fields, graph_record, params_from_scenario, structure_record
from revision_generator import generate_revision
from revision_system import run_revision_system
from scoring import aggregate_scores, format_report, score_revision


def _oracle_result(after_params, expected_fields):
    """Offline end-to-end plumbing check; not a model evaluation."""
    return {"updated_structure": structure_record(after_params), "recomputed_fields": sorted(expected_fields),
            "raw_response_text": "[dry-run oracle]", "error": None, "model": "oracle"}


def run_evaluation(n=20, seed=None, log_path="part_b_run_log.jsonl", dry_run=False, sleep_between_calls=0.0):
    rng = random.Random(seed)
    scores, records = [], []
    for index in range(n):
        original_case = generate_test_case(rng=rng)
        before_params = params_from_scenario(original_case["scenario"])
        revision = generate_revision(original_case["scenario"], rng=rng)
        after_params = params_from_scenario(revision["after_scenario"])
        expected_fields = downstream_fields(revision["changed_field"])
        if dry_run:
            result = _oracle_result(after_params, expected_fields)
        else:
            try:
                result = run_revision_system(original_case["paragraph"], before_params, revision["follow_up"])
            except Exception as exc:
                result = {"updated_structure": None, "recomputed_fields": [], "raw_response_text": "",
                          "error": f"System call failed: {exc}", "model": None}
        score = score_revision(after_params, expected_fields, result)
        scores.append(score)
        records.append({
            "index": index, "original_paragraph": original_case["paragraph"],
            "before_scenario": asdict(original_case["scenario"]), "before_structure": structure_record(before_params),
            "dependency_graph": graph_record(before_params), "revision": {**{k: v for k, v in revision.items() if k != "after_scenario"},
                                                                     "after_scenario": asdict(revision["after_scenario"])},
            "expected_after_structure": structure_record(after_params), "system_result": result, "score": score,
        })
        print(f"[{index + 1}/{n}] field={revision['changed_field']:<34} "
              f"structure={'OK' if score['structure_correct'] else 'WRONG':<5} "
              f"minimality={'OK' if score['minimality_correct'] else 'WRONG'}")
        if sleep_between_calls and not dry_run:
            time.sleep(sleep_between_calls)
    with open(log_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    report = aggregate_scores(scores)
    print("\n" + format_report(report))
    print(f"\nFull per-revision log written to: {log_path}")
    return report, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Part B DAG-based revision evaluation.")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log", default="part_b_run_log.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Use an oracle response; validates only evaluation plumbing.")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    run_evaluation(args.n, args.seed, args.log, args.dry_run, args.sleep)
