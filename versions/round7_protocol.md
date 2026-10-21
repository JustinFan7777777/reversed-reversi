# Reversed Reversi research handoff

Updated: 2026-10-21 (Asia/Shanghai)

## Active ownership

- Owner: Claude, Round 7 outcome-first endgame, branch `claude/v17-outcome-first`
  (from `codex/v16-move-ranking` 8d94d8d). Codex: read only until this says complete.
- Active upload: v14. `ai.py` stays byte-identical to
  `versions/v14_round4_fitted_end_ai.py`; the candidate lives in
  `versions/v17_outcome_first_candidate_ai.py` until every gate below passes.
- Status: running. No new strategy has been accepted.

## Why this round (evidence gathered before the protocol, scratchpad probes)

- Codex's next hypothesis (prioritize WDL in the weight fit) was checked first on
  existing exact data: a win/loss objective ranked win-versus-loss pairs no better than
  the margin objective (0.945 vs 0.948 at 17-19 empties). The principle is applied to
  the search instead: v14's last iteration proves exact disc margins with a full
  window, which Phase 2 does not score.
- Platform audits (outcome-only solves, v14 batch 04 and v13 batch 03 non-wins): every
  loss was already lost back to 21-27 empties; one draw (38_opp27, white) was a
  proven win at 21 empties. v14 at 4 s repeats the losing (2,1) (depth 17, unsolved);
  an outcome-only root solve proves (7,5) wins in 2.9 s (44M nodes).
- v14 discards its unfinished last iteration (about 40% of a 4 s move). On 40 held-out
  positions, keeping its proven-better leader changes 4 moves; a 20 s reference agrees
  with 3 (all fail-high cases).

## Round 7 hypothesis and fixed protocol (written before any labels)

At the real 4 s budget, an outcome-only root solve attempted when it is predicted to
fit, plus keeping a move already proven better in the unfinished iteration, reduces
outcome-losing choices at 20-23 empties without harming play elsewhere.

- Candidate v17 changes only the Python root driver `AI._search`; `alphabeta`, the
  evaluation, weights and table layout are v14's.
  - Outcome-first: if empties <= 24 and the median node cost C[e] of a complete
    outcome-only root solve at this empties count is at most `nps * remaining time`
    once 15% of the move budget has elapsed, the heuristic iterations stop there and
    the root is solved with window (-1, 1) to the end of the game, moves in the last
    completed iteration's order, stopping at the first proven win. Choice: proven win,
    else proven draw, else the first unproven move in that order, else (every move a
    proven loss) v14's iterations continue for the least loss. Nothing proven: the
    heuristic choice.
  - Partial iteration: when the deadline cuts an iteration after its first move
    finished, play that iteration's leader (a later move that finished higher or
    failed high on its null window); otherwise the last completed iteration's move.
  - C[e] is measured once on training positions (cold table, depth-8 ordering,
    median per empties 15-24) and written into the candidate before any holdout run.
  - No parameter search. One documented revision is allowed only if the training
    sanity check shows v17 making more outcome errors than v14.
- Positions: v14 self-play at 10 ms per move, eight random opening plies, then random
  moves with probability 0.1. One position per game at a seeded target of 20-23 empties
  (nearest qualifying position, side to move with at least two legal moves). Training:
  200 games, seeds 13000000+; fresh holdout: 400 games, seeds 14000000+.
- Labels: the exact outcome of every legal move (outcome-only solve of each child, one
  table per position, cap 300M nodes per child). Positions with any unfinished child
  are excluded and counted. Critical positions (moves with different outcomes) are the
  only ones where an outcome error is possible; the critical share is reported.
- Benchmark: v14 and v17 each choose on every critical position at 4.0 s from a cold
  table, in the same process, alternating which engine goes first. At most three
  timing processes, nothing else heavy running. Outcome error = chosen move's outcome
  below the best available outcome.
- Primary gate (holdout, opened once after v17 is frozen): fewer outcome errors than
  v14 with exact two-sided McNemar p < 0.05 on discordant positions; a paired
  bootstrap interval is reported alongside.
- Non-regression: v17 versus v14, 100 paired games at 0.4 s and 100 at 0.8 s (fresh
  opening seeds 15000000+): pooled 200-game paired 95% score interval lower bound at
  least 0.45 and each budget's lower bound at least 0.40. Full test suite, new tests for
  the outcome solve and partial-iteration rules, timing stress, startup and memory.
- Compute: labels at most 30 minutes wall with up to 6 workers; benchmarks and matches
  at most 3 timing processes, about 1-1.5 hours in total. No commit, push or platform
  upload without the user's approval.

## Previous round (Round 6, Codex) — reproduction

All four static gates failed; v14 retained. Details: `NOTES.md` Round 6,
`versions/round6_validation.json`, `versions/round6_protocol.md`.

```sh
.venv/bin/python tools/rank_model.py --data versions/round6_rank_dataset.json --output artifacts/round6_reproduced.json
.venv/bin/python tests/test_move_rank.py
.venv/bin/python tests/test_rank_data.py
```

Re-run by Claude on 2026-10-21: output identical to `artifacts/round6_reproduced.json`
(ignoring the timestamp); 12 + 6 tests pass.

## Current outputs

- Protocol: this file (archived unchanged as `versions/round7_protocol.md`).
- Raw labels and logs: `artifacts/claude_round7/` (ignored).
- Final report: pending.
