# Reversed Reversi research handoff

Updated: 2026-10-21 (Asia/Shanghai)

## Active ownership

- Owner: Codex, Round 6 move-ranking research, branch `codex/v16-move-ranking`.
- Active upload: v14. `ai.py` must remain byte-identical to
  `versions/v14_round4_fitted_end_ai.py` during this experiment.
- Claude: read this file, `NOTES.md` Round 5, and `versions/round5_validation.json`
  before continuing. Avoid concurrent source edits or heavy Python runs in this
  shared checkout while the status below is running. Source snapshots and reports
  will be committed on this branch; raw labels stay in ignored `artifacts/`.
- Status: running. No new strategy has been accepted.

## Round 6 hypothesis and fixed protocol

Within a single parent position, comparing alternative moves may be easier than
predicting an absolute board value. Fit exact sibling margin differences rather
than the value of one historical continuation.

- One parent per source game, balanced over 18, 19, and 20 empty squares. Training:
  at most 120 roots from original training games only. Fresh holdout: 60 new games,
  seeds starting at 10010000, eight random opening moves, then fixed-depth-5 v14
  with 15% random moves. The fresh games are independent of previously inspected
  Round 5 test games. Remove symmetry-equivalent parent/child overlap across splits.
- Solve every legal child exactly. Only complete sibling sets enter fitting or
  regret comparisons. Record censored roots, time spent and completion by empties.
  At most 240 seconds for training labels and 150 seconds for holdout generation
  plus labels, one worker. Each child gets at most one second. These are diagnostic
  data; timeout censoring can favor easier positions.
- One pooled integer row for child empties 17-19; all other rows stay v14's.
  Center features and exact margins within each root; give each root total weight
  one. Use ridge `1e-4 * trace(X.T @ X) / p` around the v14 row, with no grid.
  Freeze parity coefficient 9, which ordinary sibling comparisons cannot identify.
  Passes follow depth-zero search semantics. Terminal children use proven values.
- Fit and freeze the model before opening fresh holdout results. Report exact
  outcome errors, mean margin regret, optimal-choice rate, pairwise accuracy and
  paired bootstrap uncertainty. Absolute-value ridge is an offline comparator,
  not another candidate selected for matches.
- Diagnostic gate: at least 72 complete training roots and 30 complete fresh
  holdout roots, no more outcome mistakes than v14, and at least 0.5-disc reduction
  in mean exact regret. This is a screening rule, not a strength claim.
- If the gate passes, freeze one candidate before 100 new paired games versus
  v14, 0.2-second search budgets, seeds 10020000-10020049, at most 800 seconds.
  Promote only after correctness/resource checks and paired 95% score interval
  entirely above 50%; longer-budget validation remains required before upload.
- Use at most roughly 24 minutes remaining from the previously approved
  45-minute compute allowance (Round 5 used approximately 21 minutes). No parallel
  heavy jobs, no additional parameter search, and no platform upload.

## Completed evidence

- Round 5: v14 retained; independent candidate confirmation 48/10/42, score 53%,
  paired interval [47%, 59.5%]. Exact-preserving kernel remains experimental.
- v14 platform batch: 18 games, 9/1/8. All 144 own decisions at <=16 empties had
  zero exact outcome or terminal-margin errors. See committed batch04 reports.

## Current outputs

- Protocol: this file (written before labels/fitting).
- Raw dataset and bounded-run logs: `artifacts/codex_round6/`.
- Final report and concrete next steps: pending.
