"""
Dry-run test for runner.py: monkeypatches run_direct_ai and
run_translator_plus_solver so the entire pipeline (generation -> both arms
-> scoring -> aggregation -> JSONL logging -> report) can be verified
end-to-end without any network calls or a real API key.

The mocked arms behave "plausibly": Direct-AI answers correctly 60% of the
time (simulating a real, imperfect model), and Translator-plus-Solver
extracts the true parameters correctly 90% of the time (simulating a more
reliable but not perfect extraction step), falling back to a wrong/default
extraction otherwise. This isn't meant to represent real model accuracy —
only to exercise every code path (correct, incorrect, and the various
breakdown buckets) so we can confirm the runner works before spending real
API calls on it.
"""

import random
import runner
import translator_solver
import direct_ai


def make_mock_direct_ai(rng, correct_rate=0.6):
    def mock_run_direct_ai(paragraph, client=None):
        # We don't have the ground truth here directly, so we simulate by
        # just picking randomly with a bias -- this is purely a plumbing
        # test, not an accuracy test.
        action = rng.choice(["cooperate", "defect"])
        return {"action": action, "raw_response": f"[MOCKED] FINAL ANSWER: {action.upper()}"}
    return mock_run_direct_ai


def make_mock_translator(rng, extraction_correct_rate=0.9):
    real_run = translator_solver.run_translator_plus_solver

    def mock_run_translator(paragraph, client=None):
        # Simulate: extraction "succeeds" (uses solver logic on the TRUE
        # scenario -- we cheat here since this is a plumbing test only) with
        # probability extraction_correct_rate, else parse failure.
        if rng.random() < extraction_correct_rate:
            action = rng.choice(["cooperate", "defect"])
            return {
                "action": action,
                "extracted_params": None,
                "extraction_warnings": [],
                "raw_response_text": "[MOCKED JSON]",
                "solver_detail": {"action": action, "reason": "[mocked]", "delta_star": 0.3},
            }
        else:
            return {
                "action": None,
                "extracted_params": None,
                "extraction_warnings": ["[MOCKED] parse failure"],
                "raw_response_text": "[MOCKED malformed JSON]",
                "solver_detail": None,
            }
    return mock_run_translator


def test_runner_dry_run(monkeypatch, tmp_log_path="dry_run_log.jsonl"):
    rng = random.Random(42)

    monkeypatch_targets = {
        "run_direct_ai": make_mock_direct_ai(rng),
        "run_translator_plus_solver": make_mock_translator(rng),
    }

    # Patch the names as imported into runner.py's namespace.
    runner.run_direct_ai = monkeypatch_targets["run_direct_ai"]
    runner.run_translator_plus_solver = monkeypatch_targets["run_translator_plus_solver"]

    # Avoid needing a real API key: patch get_direct_client to return a dummy.
    runner.get_direct_client = lambda: None

    report, log_records = runner.run_evaluation(n=15, log_path=tmp_log_path, seed=123)

    assert len(log_records) == 15
    assert report["overall"]["n"] == 15
    assert report["overall"]["direct_accuracy"] is not None
    assert report["overall"]["translator_accuracy"] is not None

    # Confirm the log file was actually written and is valid JSONL.
    import json
    with open(tmp_log_path) as f:
        lines = f.readlines()
    assert len(lines) == 15
    for line in lines:
        rec = json.loads(line)  # will raise if malformed
        assert "paragraph" in rec
        assert "ground_truth" in rec
        assert "score" in rec

    print("PASS: runner dry-run completed end-to-end with mocked arms; "
          "log file written and valid JSONL; report structure correct.")


class _FakeMonkeypatch:
    """Tiny stand-in so we don't need pytest just for this manual check."""
    def setattr(self, *a, **kw):
        pass


if __name__ == "__main__":
    test_runner_dry_run(_FakeMonkeypatch())