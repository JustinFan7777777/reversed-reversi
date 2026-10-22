# Reversed Reversi research handoff

Updated: 2026-10-21 (Asia/Shanghai)

## Active ownership

- Owner: Claude, Round 8 shape patterns in the exact rows, branch `claude/v18-patterns`
  (from `claude/v17-outcome-first` 7fb3c1e). Codex: read only until this says complete.
- Working `ai.py` = v18 (byte-identical to `versions/v18_patterns_candidate_ai.py`) since
  2026-10-22, user-approved; platform upload and usability check pending (v14 is the
  last confirmed upload).
- Status: **all gates passed** (2026-10-22). v18 = `versions/v18_patterns_candidate_ai.py`
  (sha 7bb4ed79). Stage A +3.2 ranking points [+2.7, +3.7]; Stage B exact, 0.857x node
  rate, memory/startup/timing as v14; Stage C vs v14: 0.55 pooled 200 games [0.515, 0.588],
  0.61 at 2 s [0.52, 0.70]. ai.py replaced with v18 on 2026-10-22 (user-approved); upload pending.
  Details: `NOTES.md` Round 8, `versions/round8_validation.json`.

## Round 8 hypothesis and fixed protocol (written before any data)

Losses are decided in the middlegame (every platform loss already lost back to 21-27
empties). At 4 s the leaves of the 25-35-empties decisions fall at 10-20 empties, where
exact labels are cheap. Adding shape patterns (edge + both X squares, 3^10; 3x3 corner,
3^9; each shared by its four symmetric instances) to the 11 evaluation features of rows
10-19 ranks those leaves better, and the gain shows in games.

- Data: v14 self-play at 10 ms per move, eight random opening plies, then random moves
  with probability 0.1; for every empties count 10-19, the first position whose side to
  move has a legal move; exact final margin (full-window solve). 3,000 games, seeds
  18000000+; split by game: game id % 5 == 0 is the untouched test set.
- Fit (train games only): per row 10-19 its own 11 feature weights, plus pattern tables
  shared within the buckets 10-13, 14-16, 17-19; ridge on 8 x exact margin (v14's unit);
  one strength per bucket chosen on a game-split validation fifth of train; then refit
  on all train. Integer rounding: tables in the same units, scale chosen on train only.
- Stage A gate (test games, opened once after the fit is frozen): mean within-row
  pairwise ranking accuracy over rows 10-19 at least 1.5 points above v14's rows, with a
  game-clustered bootstrap 95% interval excluding 0, using the rounded integer model.
- Stage B (only if A passes): engine implementation in a separate candidate file; exact
  equality of engine and offline evaluation on 10,000 positions; full test suite; node
  rate loss at most 20% on a fixed position set; startup <= 1.5 s; memory growth small.
- Stage C promotion: 100 paired games versus v14 at 0.4 s and 100 at 0.8 s (fresh
  seeds 19000000+): pooled 200-game paired 95% interval entirely above 0.5; then 50
  paired games at 2 s must not fall below 0.5 in point estimate. Timing stress before
  any upload recommendation.
- Compute: Stage A about 50 minutes on up to 6 workers (data), fitting in seconds.
  Stages B-C need a new go-ahead with their estimate. No commit, push or upload without
  the user's approval.

## Previous rounds

- Round 7 (Claude, outcome-first endgame): gate failed (12 vs 10, p = 0.83); v14 kept.
  `NOTES.md` Round 7, `versions/round7_validation.json`.
- Round 6 (Codex, sibling ranking): all four static gates failed. Reproduction:

```sh
.venv/bin/python tools/rank_model.py --data versions/round6_rank_dataset.json --output artifacts/round6_reproduced.json
.venv/bin/python tests/test_move_rank.py
.venv/bin/python tests/test_rank_data.py
```

## Current outputs

- Protocol: this file (archived unchanged as `versions/round8_protocol.md`).
- Raw data and logs: `artifacts/claude_round8/` (ignored).
- Final report: `NOTES.md` Round 8 and `versions/round8_validation.json`.
