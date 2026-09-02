# Part B — DAG-Based Revision Evaluation

This is a self-contained Part B implementation.  It imports (but does not
modify) the Part A generator, renderer, solver, and extraction schema in
`../repeated_game`.

For each Part A scenario it:

1. records the extracted fields in an explicit dependency DAG;
2. changes exactly one payoff field while preserving a valid PD;
3. renders that edit as a natural-language follow-up;
4. asks the revision system for the updated extraction and the fields it
   recomputed; and
5. scores exact structure correctness and exact downstream-set minimality.

The original extraction supplied to the system is the gold Part A structure.
That deliberately isolates the secondary RQ (revision) from first-pass
extraction quality.  The original paragraph is still provided as context.

## Run

From this folder (or with its path from the repository root):

```powershell
cd E:\dissertation\repeated_game_part_b
python test_part_b.py
python runner.py --dry-run --n 100 --seed 42
```

`--dry-run` uses an oracle revision response.  It verifies the generator,
DAG reachability, scoring, reporting, and JSONL logging without Ollama and
should report 100.0% for both measures.

For the actual model evaluation, first make sure the same Ollama setup used
by Part A is available, then run:

```powershell
python runner.py --n 100 --seed 42 --log part_b_run_log.jsonl
```

Use a small `--n 20` smoke test first.  A run is perfect only when the final
report says both **updated-structure correctness** and **exact-minimal
recomputation** are 100.0%, and the JSONL log contains no records where
`score.structure_correct` or `score.minimality_correct` is false.

The log records the before/after scenario, graph, changed field, expected
downstream set, model response, and both boolean scores for every revision.
