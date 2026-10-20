# Reversed Reversi AI

Single-file submission for the Fall 2026 Project 1. Fewer own discs wins.
Only `ai.py` is uploaded. Course PDFs are excluded from Git.

Platform baseline v05 passed **10/10 usability cases** and supplied 20 verified
match traces. The current Round 2 candidate needs a new upload and usability test.
The accepted baseline remains in `versions/v05_platform_passed_ai.py`.

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
at most every 2048 nodes so Python can check elapsed time. Passes do not spend depth;
both players unable to move is terminal, even with empty squares remaining.
The candidate uses a fixed 6 MiB transposition table with full-key checks, retained
across turns and invalidated when weights change. Ordering favors cached moves and
uses opponent mobility. Seven active evaluation controls were tuned offline with
CSO; the optimizer is not part of the submission. PVS remains disabled because its
isolated playing-strength gain was not established. The earlier SPSA pilot was rejected.
Non-8x8 boards use an independent generic legal-move engine and a greedy choice.

Instructor-confirmed timing rules: startup has its own 5 seconds, every move has
5 seconds, and each player has 180 seconds per game. Conservative defaults use
160 seconds for our move clock and at most 4.0 seconds per search.

`versions/` contains reproducible snapshots. `tools/run_experiments.py` runs bounded
feature sign sweeps and a small SPSA weight-tuning pilot. It records results and
promotion evidence without changing `ai.py`. Results are budget-specific; training
scores alone do not demonstrate improvement. Round 2 tools are `cso_tune.py`,
`benchmark_search.py`, `round2_experiments.py`, and `round2_final_trials.py`.
The final candidate scored 68/5/27 against the baseline on 100 fresh matched-opening
games at equal 15 ms search budgets (score 70.5%, paired 95% CI 62.5%–78.5%). A
two-game 1-second pilot scored 2/0/0; this larger-budget sample is small.
See `NOTES.md` and `versions/round2_experiment_report.json` for all settings/results.

## Platform handoff

1. Upload `ai.py` and record the usability-test result (Phase 1).
2. After the upload passes, use `playto` against several opponents and inspect replays.
3. Return timeout/error messages, actual `time_out`, and a downloaded replay for local reproduction.
4. Keep a stable passing upload before October 23, 2026 at 23:59.

The platform must first accept an upload before practice play is available.
Local checks do not establish platform acceptance or a round-robin rank.
