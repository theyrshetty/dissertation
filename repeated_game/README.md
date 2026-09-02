# Direct-AI vs Translator-plus-Solver Eval Pipeline

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this directory:

```
GEMINI_API_KEY=your_key_here
```

## Files, in build order

1. `generator.py` — samples random valid classic-PD scenarios (T>R>P>S, 2R>T+S),
   with either a fixed integer horizon or "unknown" horizon + a discount factor.
2. `solver.py` — computes ground truth: backward induction (finite horizon
   -> always defect) or grim-trigger delta* threshold (unknown horizon).
3. `renderer.py` — turns a scenario into a natural-language paragraph.
   The discount factor is rendered as a rounded whole-percent "chance of
   continuing" so the number displayed always matches the number driving
   the ground truth (no rounding-induced mismatch).
4. `test_case.py` — combines 1-3 into `generate_test_case()`.
5. `direct_ai.py` — Direct-AI arm: paragraph -> Gemini -> extracted
   cooperate/defect answer (no structure, no solver).
6. `translator_solver.py` — Translator-plus-Solver arm: paragraph ->
   Gemini structured JSON extraction (Pydantic schema + Gemini's
   `response_schema`) -> fed into the *same* solver from step 2.
7. `scoring.py` — scores both arms against ground truth, with breakdowns
   by horizon type, delta-proximity-to-threshold, and extraction validity.
8. `runner.py` — runs N test cases end to end, prints a report, writes a
   full JSONL log.

Test files (`*_test.py`) and `stress_test.py` use mocked/synthetic inputs
and require NO network access — safe to run anytime to sanity-check logic.

## Running the real evaluation

```bash
python3 runner.py --n 20 --seed 42
```

Start at n=20 as a sanity check (per the study design's own guidance)
before scaling to several hundred. Use `--sleep 1.0` if you hit rate limits.

Output:
- Console: per-case pass/fail as it runs, then the aggregate report.
- `run_log.jsonl`: one JSON record per test case (paragraph, true scenario,
  ground truth, both arms' raw + extracted outputs, and the full score
  breakdown) — use this to read exactly which paragraphs tripped up which arm.

## Known limitation carried over from the original design doc

Every scenario originates from a known generative structure. This measures
recoverability and robustness to controlled synthetic messiness, not the
full unbounded variety of real human phrasing (Part D of the original
design partially closes this gap with noisy renderings; not yet built here).