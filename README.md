# Reversed Reversi AI

A single-file game AI for **Reversed Reversi** (anti-Othello: the player with *fewer*
discs at the end wins). It was written for the Fall 2026 Artificial Intelligence course
project at SUSTech and played on the course's online platform. There it faced a 5 s
limit per move, 180 s per game and 100 MB of memory, with NumPy and Numba as the only
libraries.

The final engine, **v20** ([`ai.py`](ai.py)), is the result of 20 versions and 10
research rounds. Every promotion had to pass a gate fixed in advance and win paired
matches against the previous champion. The whole story is in
**[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**.

## Results

| Step | Change | Score vs previous version (paired 95% CI) |
|---|---|---|
| v05 | Robust search, frontier feature | 0.660 vs v01 [0.565, 0.750] |
| v12 | Transposition table, move ordering, CSO-tuned weights | 0.705 vs v05 [0.625, 0.785] |
| v13 | PVS at every level, inline leaves | 0.630 vs v12 at 0.8 s [0.555, 0.705] |
| v14 | Endgame rows fitted on exact margins | 0.635 vs v13 at 0.8 s [0.565, 0.705] |
| v18 | Shape-pattern tables, rows 10–19 | 0.55 vs v14, 200 games [0.515, 0.588] |
| v20 | Shape-pattern tables, rows 20–22 | 0.530 vs v18, 1,000 games [0.511, 0.549]; 0.55 at 2 s |

Each uploaded version (v05, v12, v13, v14, v18, v20) passed all 10 platform usability
tests. In 120 platform games, all ended normally (no timeouts or illegal moves), and an
exact solver confirmed every one of our 841 moves at 14 or fewer empties.

## How the engine works

**Board and moves.** Each side is a 64-bit bitboard. Legal moves and flips come from
shifted masks in Numba-compiled functions. Input boards of any type (lists, floats,
strided arrays) are converted without being modified. Non-8×8 boards use a simple
generic engine.

**Search.** Negamax alpha-beta with an explicit stack instead of recursion, which
Numba handles poorly:

- iterative deepening that keeps the last completed iteration;
- principal variation search (PVS) inside the tree and at the root;
- a 6 MiB transposition table with full-key checks, kept across moves;
- ordering by the cached best move, then by fewest opponent replies;
- depth-1 nodes evaluate their leaves inline;
- passes cost no depth, and a position where neither side can move is terminal;
- the search reaches exact solves near the end of the game.

**Time safety.** The search runs in 2,048-node slices and checks the clock between them.
A deadline can stop it anywhere, and it returns the best completed result. The budget is
160 s per game, at most 80% of each move's allowance, capped at 4 s. If anything goes
wrong, the AI falls back to a legal move.

**Evaluation.** A weight table with one row per number of empty squares (0–64). Each row
weights 11 features: corners, X and C squares, edges, discs, mobility, stable edge discs,
frontier, potential mobility, parity and a positional table. Rows 1–16 are fitted on
exactly solved endgames. Rows 10–22 add two **shape patterns** read from the side to move:

- edge + both X-squares (3^10 states);
- the 3×3 corner block (3^9 states).

Each pattern is shared by its symmetric instances. The tables were fitted by ridge
regression on 31,200 exactly solved positions and stored as int8 to stay within memory.
Rows with different scales are never compared, because every leaf of one search
iteration has the same number of empties.

## Repository layout

```text
ai.py            the submission (v20); the only file uploaded
docs/            DEVELOPMENT.md (v00 → v20 story), NOTES.md (full lab notebook),
                 RESEARCH_STATUS.md (final status), SUBMISSION_CHECKLIST.md
versions/        every engine snapshot v00–v20, round protocols (roundN_protocol.md),
                 validation records (roundN_validation.json), sha256.json
tools/           arena and paired-match runner, data generation, exact labelling,
                 fitting, builds (build_v18/v20.py), benchmarks, replay analysis
tests/           correctness suite (minimax equality, resumable search, interface, …)
```

`tools/` and `tests/` are flat: each script finds the repository root from its own path.
Raw data and platform replays are not in Git. The replays contain other students' IDs;
opponents appear only as anonymous labels (`opp01` …) in the committed records.

## Quick start

Requires Python 3.10, NumPy 1.26.4 and Numba 0.59.1 (the platform versions).

```sh
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Play v20 against a greedy baseline, then against the original v00
.venv/bin/python tools/arena.py ai.py tools/baseline_greedy.py --games 10 --time-out 0.5
.venv/bin/python tools/arena.py ai.py versions/v00_ai.py --games 100 --time-out 0.1 --opening-plies 6
```

The arena plays opening pairs with colours exchanged and enforces the platform rules:
the full legal-move list, input immutability, 5 s per move, a separate 5 s for import and
initialisation, and 180 s per player. It reports a paired-bootstrap confidence interval.

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
.venv/bin/python tests/test_patterns.py ai.py      # Round 8 pattern engines only
.venv/bin/python tests/test_patterns_hi.py ai.py   # Round 10 pattern engines only
.venv/bin/python tests/test_outcome_root.py versions/v17_outcome_first_candidate_ai.py
.venv/bin/python tools/bench_first_move.py ai.py
.venv/bin/python tools/bench_timing.py
.venv/bin/python tools/bench_memory.py ai.py --moves 60 --time-out 0.1
.venv/bin/python tools/check_engine.py --games 200 --self-play --time-out 0.005
```

The tests check the search against unpruned minimax, check that interrupted and resumed
searches give identical results, and compare the pattern engines entry by entry with
reference implementations. Tiny timeouts test correctness and recovery, not strength.
The memory benchmark uses Linux `/proc` or macOS `ps`.

## Acknowledgements

Course project for Artificial Intelligence (Fall 2026, SUSTech). The engine started
from a framework written with an AI assistant and was developed with Anthropic's Claude
Code and OpenAI's Codex; [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) records who did
what in each round. The CSO tuner follows Cheng & Jin, *A Competitive Swarm Optimizer
for Large Scale Optimization*, IEEE Trans. Cybernetics, 2015.
