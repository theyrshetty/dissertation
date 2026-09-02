# Direct-AI vs Translator-plus-Solver Eval Pipeline

## Setup

```bash
pip install -r requirements.txt
```

This pipeline runs against a local or remote **Ollama** server -- no API key,
no rate limits, no daily quota (it previously used the Gemini API; see git
history / conversation log if you need that version back).

1. Install Ollama: https://ollama.com/download (or on an HPC cluster, follow
   your institution's instructions for installing/running it in your own
   environment -- you likely won't have root, so a container approach like
   Singularity/Apptainer, or a user-space install, may be needed. Check your
   HPC's documentation or support desk for the specifics of your cluster.)
2. Pull a model:
   ```bash
   ollama pull llama3.2:3b        # small, fast, good for local CPU sanity checks
   # or, for a real full-scale run on an HPC GPU node:
   ollama pull qwen2.5:14b-instruct
   ```
3. Start the server (if not already running as a service):
   ```bash
   ollama serve
   ```
4. (Optional) Create a `.env` file in this directory if you want to override
   the defaults:
   ```
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=llama3.2:3b
   ```
   For an HPC run, `OLLAMA_HOST` should point at the compute node running
   `ollama serve` (e.g. `http://<node-hostname>:11434`) -- this typically
   means requesting a GPU node via your scheduler (SLURM/PBS), starting
   `ollama serve` there, and pointing OLLAMA_HOST at that node's hostname
   from wherever you run `runner.py`. Whether that's reachable directly or
   needs an SSH tunnel/port-forward depends on your cluster's network setup.

Both defaults (`http://localhost:11434`, `llama3.2:3b`) work out of the box
for local CPU testing with no `.env` file at all.

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
5. `direct_ai.py` — Direct-AI arm: paragraph -> Ollama -> extracted
   cooperate/defect answer (no structure, no solver).
6. `translator_solver.py` — Translator-plus-Solver arm: paragraph ->
   Ollama structured JSON extraction (Pydantic schema + Ollama's
   constrained-decoding `format` parameter) -> fed into the *same* solver
   from step 2.
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
before scaling to several hundred. Local inference has no rate limit or
daily quota, so `--sleep` defaults to 0 -- only set it (e.g. `--sleep 1.0`)
if you're pointing at a shared/remote Ollama instance and want to be
polite to other users of that machine.

Output:
- Console: per-case pass/fail as it runs, then the aggregate report.
- `run_log.jsonl`: one JSON record per test case (paragraph, true scenario,
  ground truth, both arms' raw + extracted outputs including the exact
  model name used, and the full score breakdown) — use this to read
  exactly which paragraphs tripped up which arm, and to confirm which
  Ollama model tag produced a given run (important for reproducibility,
  since you may switch model sizes between a local sanity check and a
  full HPC run).

## Known limitation carried over from the original design doc

Every scenario originates from a known generative structure. This measures
recoverability and robustness to controlled synthetic messiness, not the
full unbounded variety of real human phrasing (Part D of the original
design partially closes this gap with noisy renderings; not yet built here).

<!-- FOR HPC run

ollama pull qwen2.5:14b-instruct
ollama serve

then in .env
OLLAMA_HOST=http://<compute-node-hostname>:11434
OLLAMA_MODEL=qwen2.5:14b-instruct -->