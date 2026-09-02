"""
Dry-run test for consistency_check.py: monkeypatches run_direct_ai and
run_translator_plus_solver so the entire consistency pipeline (scenario
sampling -> paraphrase rendering -> both arms on every wording -> per-scenario
consistency scoring -> aggregation -> JSONL logging -> report) can be verified
end-to-end without any network calls or a real API key.

Two mock behaviors are exercised deliberately:
  - an arm that is perfectly consistent (always returns the same action for
    a given scenario_id, regardless of wording) -> should score 100%.
  - an arm that answers randomly, independent of scenario/wording -> should
    score well below 100% (but not exactly 0%, since 2-3 random draws can
    coincidentally match).

This isn't meant to represent real model behavior -- only to exercise every
code path (matching, mismatching, and the None/parse-failure branch) so the
consistency pipeline can be trusted before spending real API calls on it.
"""

import json
import random

import consistency_check as cc


def make_consistent_direct(rng):
    """Always returns the same action for a given scenario, any wording."""
    memo = {}

    def mock_run_direct_ai(paragraph, client=None):
        # We don't have scenario_id here, so key off the paragraph's
        # invariant substring set is impractical; instead cheat via a
        # closure-scoped counter reset per scenario by the caller pattern:
        # consistency_check calls this once per wording within a scenario
        # loop, so we key off nothing and instead rely on a fixed action
        # chosen the FIRST time and reused via `memo` keyed by paragraph
        # length bucket is unreliable. Simplest reliable approach: key off
        # a rolling scenario counter incremented every N calls is fragile.
        # Instead, just always return the same fixed action -- still a
        # valid "perfectly consistent arm" for plumbing purposes.
        return {"action": "cooperate", "raw_response": "[MOCKED] FINAL ANSWER: COOPERATE"}

    return mock_run_direct_ai


def make_random_translator(rng):
    def mock_run_translator(paragraph, client=None):
        action = rng.choice(["cooperate", "defect", None])
        return {
            "action": action,
            "extracted_params": None,
            "extraction_warnings": [] if action is not None else ["[MOCKED] parse failure"],
            "raw_response_text": "[MOCKED JSON]",
            "solver_detail": None,
        }
    return mock_run_translator


def test_consistency_check_dry_run(tmp_log_path="dry_run_consistency_log.jsonl"):
    rng = random.Random(7)

    report, per_scenario_results, raw_records = cc.run_consistency_check(
        sample_size=10,
        num_variants=3,
        log_path=tmp_log_path,
        seed=99,
        direct_runner=make_consistent_direct(rng),
        translator_runner=make_random_translator(rng),
        client=object(),  # dummy, never touched since both runners are mocked
    )

    # --- Structural checks ---
    assert report["overall"]["n"] == 10
    assert report["overall"]["direct_consistency_rate"] is not None
    assert report["overall"]["translator_consistency_rate"] is not None

    # The always-"cooperate" arm must be perfectly consistent by construction.
    assert report["overall"]["direct_consistency_rate"] == 1.0, (
        "An arm returning the identical action on every wording of every "
        "scenario must score 100% consistency; got "
        f"{report['overall']['direct_consistency_rate']}"
    )

    # The random arm should NOT be perfectly consistent across 10 scenarios
    # x 3 wordings each with a 3-way random choice (would require every
    # triple to coincidentally match by chance in every one of 10 scenarios).
    assert report["overall"]["translator_consistency_rate"] < 1.0, (
        "A random-per-call arm coincidentally scoring 100% consistency "
        "across 10 independent scenarios is vanishingly unlikely; check "
        "that wording_id / actions_by_arm bookkeeping isn't accidentally "
        "collapsing distinct calls together."
    )

    # per_scenario_results must contain exactly 2 rows per scenario (one per arm).
    assert len(per_scenario_results) == 20
    for row in per_scenario_results:
        assert row["arm"] in (cc.ARM_DIRECT, cc.ARM_TRANSLATOR)
        assert isinstance(row["consistent"], bool)
        assert len(row["answers"]) == 3  # num_variants=3

    # raw_records must have 2 arms x 3 wordings x 10 scenarios = 60 rows,
    # and every row must be valid, loggable JSON with the fields consistency
    # spot-checking depends on.
    assert len(raw_records) == 60
    for rec in raw_records:
        assert "paragraph" in rec
        assert "extracted_answer" in rec
        assert "scenario" in rec
        assert "consistent" in rec
        assert rec["wording_id"] in (1, 2, 3)

    # Confirm the log file was actually written and is valid JSONL, matching
    # raw_records exactly.
    with open(tmp_log_path) as f:
        lines = f.readlines()
    assert len(lines) == 60
    for line in lines:
        json.loads(line)  # will raise if malformed

    # Sanity-check the direct arm's per-scenario consistency flag directly
    # against its raw answers list, independent of the aggregate rate above.
    for row in per_scenario_results:
        if row["arm"] == cc.ARM_DIRECT:
            assert row["answers"] == ["cooperate", "cooperate", "cooperate"]
            assert row["consistent"] is True

    print("PASS: consistency_check dry-run completed end-to-end with mocked "
          "arms; log file written and valid JSONL; report structure correct; "
          "a perfectly-consistent arm scores 100% and a random arm scores "
          "below 100%.")


def test_action_is_consistent_helper():
    assert cc._action_is_consistent(["cooperate", "cooperate", "cooperate"]) is True
    assert cc._action_is_consistent(["cooperate", "defect", "cooperate"]) is False
    assert cc._action_is_consistent([None, None, None]) is True  # identical parse failures
    assert cc._action_is_consistent([None, "cooperate", None]) is False
    assert cc._action_is_consistent([]) is False
    print("PASS: _action_is_consistent handles matches, mismatches, "
          "identical-None, and the empty-list edge case correctly.")


if __name__ == "__main__":
    test_action_is_consistent_helper()
    test_consistency_check_dry_run()