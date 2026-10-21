# Reversed Reversi AI

Single-file submission for the Fall 2026 Project 1. Fewer own discs wins.
Only `ai.py` is uploaded. Course PDFs are excluded from Git.

Platform baseline v05 passed **10/10 usability cases** and supplied 20 verified
match traces. User confirmed that Round 2 v12 also passed all 10 usability cases,
earned the 10 basic points, and remained fixed throughout its new 20-game batch.
The accepted baseline remains in `versions/v05_platform_passed_ai.py`.
The working `ai.py` is Round 4 v14: Round 3 v13 (PVS at all levels, inline leaf
evaluation) plus end-phase weights fitted on exact labels. v13 and then v14 were
uploaded on 2026-10-21 and each passed all 10 usability cases (user report); v14 is
the platform submission.

## Environment

Tested with Python 3.10.22, NumPy 1.26.4 and Numba 0.59.1 on macOS arm64.

```sh
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

VS Code uses `.venv/bin/python`. A project-local `.python/` interpreter was installed
for this checkout; both it and `.venv/` are ignored by Git.

## Validate

```sh
.venv/bin/python tools/lint_submission.py ai.py
.venv/bin/python tools/check_engine.py --games 40
.venv/bin/python tests/test_search.py
.venv/bin/python tests/test_resumption.py
.venv/bin/python tests/test_interface.py
.venv/bin/python tests/test_fallback.py
.venv/bin/python tests/test_terminal.py
.venv/bin/python tests/test_arena_limits.py
.venv/bin/python tests/test_variants.py ai.py
.venv/bin/python tests/test_advanced_search.py ai.py
.venv/bin/python tests/test_pvs_inline.py
.venv/bin/python tests/test_eval_table.py
.venv/bin/python tools/bench_first_move.py ai.py
.venv/bin/python tools/bench_timing.py
.venv/bin/python tools/bench_memory.py ai.py --moves 60 --time-out 0.1
.venv/bin/python tools/check_engine.py --games 200 --self-play --time-out 0.005
```

The RSS benchmark supports Linux `/proc` and macOS `ps`; a restricted sandbox may
require permission to run `ps`. Tiny timeouts above test correctness and recovery,
not competition strength. All development tools enable `DEBUG_RAISE`.

```sh
.venv/bin/python tools/arena.py ai.py tools/baseline_random.py --games 6 --time-out 0.5
.venv/bin/python tools/arena.py ai.py tools/baseline_greedy.py --games 10 --time-out 0.5
.venv/bin/python tools/arena.py ai.py versions/v00_ai.py --games 100 --time-out 0.1 --opening-plies 6 --json artifacts/comparison.json
```

Opening pairs share a seed and exchange colors. The arena checks the full move set,
input immutability, per-move limits, separate 5-second import/initialization allowances,
and independent 180-second limits for each player.
Its paired-bootstrap confidence intervals resample opening pairs together.
The bounded experiment scripts explicitly treat their millisecond search settings
as soft compute budgets, while retaining the official 5-second hard limit and
recording each overrun. The arena CLI honors `time_out` as a hard limit by default.

## Search and experiments

The stable search uses uint64 bitboards and an explicit-stack negamax alpha-beta
implementation. Iterative deepening keeps the last completed result. Search yields
about every 2048 nodes (an inline depth-1 node may add its leaves first) so Python
can check elapsed time. Passes do not spend depth;
both players unable to move is terminal, even with empty squares remaining.
The candidate uses a fixed 6 MiB transposition table with full-key checks, retained
across turns and invalidated when weights change. Ordering favors cached moves and
uses opponent mobility. The opening and middle-game weights (seven active controls)
were tuned offline with CSO; the optimizer is not part of the submission. The earlier
SPSA pilot was rejected.
Round 3 enables PVS inside the tree and at the root (0.56x the nodes at depth 11 with
identical values), and depth-1 nodes evaluate their leaves inline (same tree, about
1.6x faster). Against v12 over 100 paired games each, v13 scored 0.640 at 0.4 s and
0.630 at 0.8 s of search per move (both paired CIs above 0.5) and 0.500 at 0.2 s.
Round 4 replaces the end-phase row with one row per empties count 1-16, fitted by
ridge regression on exact final margins from 1,200 self-play games. Against v13 over
100 paired games each, v14 scored 0.635 at 0.4 s and at 0.8 s (CIs above 0.5) and
0.565 at 2 s (CI [0.50, 0.63], inconclusive); the real 4 s budget is unmeasured.
Non-8x8 boards use an independent generic legal-move engine and a greedy choice.

Instructor-confirmed timing rules: startup has its own 5 seconds, every move has
5 seconds, and each player has 180 seconds per game. Conservative defaults use
160 seconds for our move clock and at most 4.0 seconds per search.

`versions/` contains reproducible snapshots. `tools/run_experiments.py` runs bounded
feature sign sweeps and a small SPSA weight-tuning pilot. It records results and
promotion evidence without changing `ai.py`. Results are budget-specific; training
scores alone do not demonstrate improvement. Round 2 tools are `cso_tune.py`,
`benchmark_search.py`, `round2_experiments.py`, and `round2_final_trials.py`.
Round 2's final candidate scored 68/5/27 against the baseline on 100 fresh matched-opening
games at equal 15 ms search budgets (score 70.5%, paired 95% CI 62.5%–78.5%). A
two-game 1-second pilot scored 2/0/0; this larger-budget sample is small.
See `NOTES.md` and `versions/round2_experiment_report.json` for all settings/results.

## Platform handoff

1. Upload `ai.py` and record the usability-test result (Phase 1).
2. After the upload passes, use `playto` against several opponents and inspect replays.
3. Return timeout/error messages and downloaded replays for local reproduction.
   The platform does not expose match timing telemetry to the user.
4. Keep a stable passing upload before October 23, 2026 at 23:59.

The platform must first accept an upload before practice play is available.
Local checks do not establish platform acceptance or a round-robin rank.

## Round 5 handoff

`ai.py` remains the platform-accepted **v14**. Middle-game fitting did not pass
independent confirmation: 48 wins / 10 draws / 42 losses against v14 at equal
0.2-second search budgets, score 53%, paired 95% CI [47%, 59.5%]. No new strategy
was promoted. `versions/v15_mid19_experimental_ai.py` preserves that rejected-for-
promotion experiment; this result is inconclusive rather than proof of inferiority.

`versions/v15_fast_features_candidate_ai.py` preserves a separate, exact-equivalent
feature-kernel prototype: edge-chain lookups and fused adjacency computation.
It passes value/node/cache equality checks. Small fixed-depth samples show about
5% less time when fitted end-phase features are active, and about 1% overall.
This is a performance prototype, not a demonstrated playing-strength improvement.

```sh
.venv/bin/python tests/test_stable_edge_lookup.py
.venv/bin/python tools/check_common_probes.py --candidate versions/v15_mid19_experimental_ai.py
```

`tools/fit_midgame.py` reuses the 1,200-game dataset recovered into ignored
`artifacts/codex_round5/claude_round5_inputs/`; the raw dataset is not shipped in Git.
Rows 17-19 use exact labels; later rows use bootstrapped teacher labels. Pass an
explicit `--data` directory when restoring the dataset elsewhere. The fitter and
`tools/run_paired_trial.py` use one process and save source hashes. The match runner
checks its time budget before each new opening pair; an active pair may finish
after that deadline. Full records and the next-step rationale are in `NOTES.md` and
`versions/round5_validation.json`.
