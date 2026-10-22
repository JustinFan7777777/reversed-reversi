# Reversed Reversi research handoff

Updated: 2026-10-22 (Asia/Shanghai)

## Active ownership

- Owner: Claude, branch `claude/v18-patterns`. Codex: read only until this says complete.
- Active upload: **v18** (`ai.py` == `versions/v18_patterns_candidate_ai.py`, sha 7bb4ed79),
  uploaded and passed 10/10 on 2026-10-22. It stays until a successor passes every gate.
- Status: Round 9 complete, gate failed (+0.81 vs +1.0 required), v18 retained.
  **Round 10 approved and running** (the user's final round): frozen protocol with revisions
  1-3 in `versions/round10_protocol.md` (P0 patterns, linear-only fallback, Stage C
  500+500 pooled then 300 at 2 s); the section below is the original proposal.
  Labelling started 2026-10-22 13:22 (8 workers; expected to finish 2026-10-23 afternoon;
  `artifacts/claude_round10/`). Rounds 0-7 artifacts are packed in
  `artifacts/archive_history.tar.xz` (restore with `tar -xJkf` from the repo root).
- Round 10 execution notes (from the pipeline reviews): fit only after every
  `old_wJ.log` ends with "worker J: 1125 records"; run Stage B's suite in a fresh repo
  copy without `Project Requirements/`, `.venv/`, `.python/` or `artifacts/`, pass
  `--frozen` an absolute path to `artifacts/claude_round10/fit_frozen*.json`, run
  `tools/bench_node_rate.py` from the main repo, and give `tools/bench_timing.py` a
  `--json` under `artifacts/claude_round10/` so runs do not overwrite each other.
  Run `tools/run_paired_trial.py` without `-I` (it imports `arena` from `tools/`).
- Stage C mechanics (fixed before any Stage C data): `tools/run_stage_c10.py` runs the
  plan and `tools/pool_paired.py` pools and gates it (dry run: all three stages, FAILED
  when the short gate fails). `tools/run_paired_trial.py` now saves a game lost at
  initialization (it has no move times) before stopping on it; a dry run under the
  labelling load had crashed there instead (import+init 2.3-2.5 s at load 10, over 5 s
  in a load spike). If a Stage C process stops on a non-normal game, that Stage C is
  void. An initialization timeout with the baseline's import+init also inflated
  (machine load) is mechanical: the whole Stage C is rerun with the same seeds into a
  new folder on an idle machine, and the void folder is kept and reported, as is a
  Stage C stopped by the unattended-run thermal watchdog (it pauses only the clock-free
  labelling and fit; it terminates timed runs only at emergency temperatures). A move
  timeout, illegal move or mutated input by the candidate fails Stage C.
- Platform evidence (NOTES.md "Platform batch 05"): the platform's 4 s move equals about
  2 s on the development Mac (97.6% move agreement at 2 s, 86% at 4 s).

## Round 10 proposal: shape patterns at rows 20-22 (approved; see revisions in versions/round10_protocol.md)

At the platform's effective budget (about 2 s here, depth about 13 at 30-40 empties) the
leaves of the decisive middlegame moves fall at about 20-27 empties, where v18 still uses
v14's 11 linear features. Rows 20-22 can still be labelled exactly (median solve cost
3.7M / 13.8M / 22.4M nodes), so the Round 8 recipe applies without bootstrapped labels
(Codex's bootstrapped rows 17-24 failed in Round 5).

- Data: the 9,000 unlabelled Round 9 positions at 20-22 empties (from 3,000 games, split
  already fixed: id % 5 == 0 test) get exact margins, plus 3,000 new games (seeds
  22000000+) labelled at 20-22 only, split the same way. The Round 9 test games were
  never used for rows 20-22, so their 20-22 positions stay an untouched test set.
- Fit: Round 9 P2 shapes and recipe, one table bucket 20-22, per-row 11 weights; strength
  and integer scale chosen without test data. A smaller shape set (P0) is fitted too and
  chosen on validation if it is within 0.2 points (fewer tables, less node-rate cost).
- Stage A gate: within-row ranking on test at least +1.5 points over v18's rows 20-22,
  game-bootstrap 95% interval excluding 0.
- Stage B: engine/reference equality on 10,000 positions, full suite, node rate at least
  0.85x v18, startup <= 1.5 s, memory as v18, timing stress.
- Stage C vs v18 (fresh seeds 23000000+): 100 paired games at 0.5 s and 100 at 1.0 s,
  pooled 95% interval entirely above 0.5; then 100 paired games at 2.0 s (the platform
  equivalent) with point estimate at least 0.5.
- Compute: labelling about 1.5-2 h on 6 background workers; Stage C about 1.5 h.

## Round 9 hypothesis and fixed protocol (written before any Round 9 fit)

v18's pattern rows 10-19 were fitted on 24,000 positions and validation kept favouring
weaker regularization, which points to more data and richer shapes helping.

- Data: Round 8's 3,000 games (all of them now training) plus 3,000 new games, seeds
  20000000+, same self-play policy and exact labels at 10-19 empties (positions at 20-23
  are kept unlabelled for a later step). The new games with id % 5 == 0 are the untouched
  test set; validation = new train games with id // 5 % 5 == 1.
- Candidate shape sets, each fitted with the Round 8 recipe (per-row 11 weights, tables
  shared within 10-13, 14-16, 17-19, strength and integer scale chosen without test data):
  P0 = v18's shapes (edge+2X, 3x3 corner); P1 = P0 + 2x5 corner (8 instances);
  P2 = P1 + the three inner lines per side and the diagonals of length 4-8. The set is
  chosen on validation ranking, then scored once on test.
- Stage A gate: the chosen model's within-row ranking on the new test games at least
  1.0 point above v18's frozen model, game-bootstrap 95% interval excluding 0.
- Stages B-C as in Round 8 but against v18: exact engine/reference equality, full suite,
  node rate, startup, memory (int8 tables), timing stress; 200 paired games versus v18
  at 0.4/0.8 s (fresh seeds 21000000+) with the pooled interval entirely above 0.5, then
  50 games at 2 s with point estimate at least 0.5.
- Rows 20-23 (bootstrapped labels from v18 searches whose leaves are rows <= 19) are a
  separate later step with its own protocol entry.

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
