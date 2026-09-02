"""Direct value and set comparisons for the two Part B measures."""

from dependency_graph import structure_record


def score_revision(after_params, expected_recomputed_fields: set[str], system_result: dict) -> dict:
    expected_structure = structure_record(after_params)
    actual_structure = system_result.get("updated_structure")
    actual_fields = set(system_result.get("recomputed_fields", []))
    return {
        "structure_correct": actual_structure == expected_structure,
        "minimality_correct": actual_fields == expected_recomputed_fields,
        "expected_recomputed_fields": sorted(expected_recomputed_fields),
        "reported_recomputed_fields": sorted(actual_fields),
    }


def aggregate_scores(scores: list[dict]) -> dict:
    n = len(scores)
    return {
        "n": n,
        "updated_structure_correctness": sum(x["structure_correct"] for x in scores) / n if n else None,
        "exact_minimal_recomputation": sum(x["minimality_correct"] for x in scores) / n if n else None,
    }


def format_report(report: dict) -> str:
    def pct(value):
        return "n/a" if value is None else f"{value * 100:.1f}%"
    return ("=== PART B REVISION RESULTS ===\n"
            f"Revisions evaluated: {report['n']}\n"
            f"Updated-structure correctness: {pct(report['updated_structure_correctness'])}\n"
            f"Exact-minimal recomputation:  {pct(report['exact_minimal_recomputation'])}")
