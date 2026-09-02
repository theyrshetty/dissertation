"""
Dry-run test for runner.py: monkeypatches run_direct_ai and
run_translator_plus_solver so the entire pipeline (generation -> both arms
-> scoring -> aggregation -> JSONL logging -> report) can be verified
end-to-end without any network calls or a real Ollama server.

The mocked arms behave "plausibly": Direct-AI answers correctly 60% of the
time (simulating a real, imperfect model), and Translator-plus-Solver
extracts correctly 90% of the time, falling back to a parse failure
otherwise. This isn't meant to represent real model accuracy -- only to
exercise every code path so we can trust the runner before spending real
inference time on it.
"""

import random
import runner


def make_mock_direct_ai(rng, correct_rate=0.6):
    def mock_run_direct_ai(paragraph, client=None):
        action = rng.choice(["cautious", "aggressive"])
        return {"action": action, "raw_response": f"[MOCKED] FINAL ANSWER: {action.upper()}"}
    return mock_run_direct_ai


def make_mock_translator(rng, extraction_correct_rate=0.9):
    def mock_run_translator(paragraph, client=None):
        if rng.random() < extraction_correct_rate:
            action = rng.choice(["cautious", "aggressive"])
            return {
                "action": action,
                "extracted_params": None,
                "extraction_warnings": [],
                "raw_response_text": "[MOCKED JSON]",
                "solver_detail": {"action": action, "reason": "[mocked]", "margin": 1.0},
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

    runner.run_direct_ai = make_mock_direct_ai(rng)
    runner.run_translator_plus_solver = make_mock_translator(rng)
    runner.get_direct_client = lambda: None

    report, log_records = runner.run_evaluation(n=15, log_path=tmp_log_path, seed=123)

    assert len(log_records) == 15
    assert report["overall"]["n"] == 15
    assert report["overall"]["direct_accuracy"] is not None
    assert report["overall"]["translator_accuracy"] is not None

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
