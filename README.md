# Reversed Reversi AI

Single-file submission for the Fall 2026 Project 1. Fewer own discs wins.
Only `ai.py` is uploaded. Course PDFs are excluded from Git.

Platform status: successful upload and **10/10 usability cases passed**, reported by
the user on 2026-10-20. Live matches and final round-robin results are pending.

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
input immutability, per-move limits, and the conservative shared 180-second game limit.
Its paired-bootstrap confidence intervals resample opening pairs together.
The bounded experiment scripts explicitly treat their millisecond search settings
as soft compute budgets, while retaining the official 5-second hard limit and
recording each overrun. The arena CLI honors `time_out` as a hard limit by default.

## Search and experiments

The stable search uses uint64 bitboards and an explicit-stack negamax alpha-beta
implementation. Iterative deepening keeps the last completed result. Search yields
at most every 512 nodes so Python can check elapsed time. Passes do not spend depth;
both players unable to move is terminal, even with empty squares remaining.
The final evaluator adds a validated -4 own-frontier term to the supplied six-feature
baseline. Other experimental features are disabled; TT and SPSA weights were rejected.
Non-8x8 boards use an independent generic legal-move engine and a greedy choice.

`versions/` contains reproducible snapshots. `tools/run_experiments.py` runs bounded
feature sign sweeps and a small SPSA weight-tuning pilot. It records results and
promotion evidence without changing `ai.py`. Results are budget-specific; training
scores alone do not demonstrate improvement. See `NOTES.md` for measured results.

## Platform handoff

1. Upload `ai.py` and record the usability-test result (Phase 1).
2. After the upload passes, use `playto` against several opponents and inspect replays.
3. Return timeout/error messages, actual `time_out`, and a downloaded replay for local reproduction.
4. Keep a stable passing upload before October 23, 2026 at 23:59.

The platform must first accept an upload before practice play is available.
Local checks do not establish platform acceptance or a round-robin rank.
