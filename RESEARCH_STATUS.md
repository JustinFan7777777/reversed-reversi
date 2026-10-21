# Reversed Reversi research handoff

Updated: 2026-10-21 (Asia/Shanghai)

## Current state

- Round 6 is complete. No active experiment process remains.
- Branch: `codex/v16-move-ranking`; owner: Codex, ready for Claude handoff.
- Active platform/source version: **v14**. `ai.py` is byte-identical to
  `versions/v14_round4_fitted_end_ai.py`. No new strategy was accepted or uploaded.
- Read this file first before switching branches or starting another heavy job.
  Keep one heavy local process and avoid concurrent edits in this shared checkout.

## What was tested

- A single fixed exact-sibling ranking fit: compare all legal moves from a parent
  instead of predicting absolute board value. Protocol was written before labels;
  its unchanged content is archived as `versions/round6_protocol.md`.
- Labels: 52 complete old-training roots / 427 alternatives and 23 complete fresh
  holdout roots / 184 alternatives. Another 105 attempted roots remain incomplete.
  Fit uses 51 roots; one complete root has only one move. No complete parent-20
  root was obtained. Do not claim child-19 coverage from the pooled fitted row.
- Fresh static choices: mean margin regret 3.174 -> 2.783 discs, but WDL mistakes
  1 -> 4. The confidence interval includes no improvement. All four screening
  gates failed; no matches followed. These are static evaluations, not timed play.
- Integer rounding of the positional coefficient amplifies three bad switches.
  Post hoc frozen-float arithmetic still loses against the baseline diagnostics;
  it is explanatory evidence, not a new candidate or independent validation.
- Twelve fitter tests, six independent generator tests, 192 minimax comparisons,
  112 table checks and lint all pass. Current-round measured compute about 5.8 min;
  cumulative with Round 5 approximately 27 min, below the approved 45 min.

## Durable files and reproduction

- `NOTES.md`, Round 6: rationale, result interpretation and next hypothesis.
- `versions/round6_validation.json`: all gates, metrics, hashes, tests and failures.
- `versions/round6_rank_dataset.json`: all 180 attempted families, exact labels and
  explicit unknowns. Reusable without the original private scratch inputs.
- `versions/v16_rank_only_experimental_ai.py`: research snapshot, not an upload.
- `artifacts/codex_round6/`: ignored raw logs, frozen model and phase compute ledger.

```sh
.venv/bin/python tools/rank_model.py --data versions/round6_rank_dataset.json --output artifacts/round6_reproduced.json
.venv/bin/python tests/test_move_rank.py
.venv/bin/python tests/test_rank_data.py
```

## Next hypothesis, not implemented

Prioritize win/draw/loss mistakes in sibling training and use final disc margin
secondarily. Choose coefficient precision using training only. Use an untouched
new holdout, and collect complete parent-20 alternatives before changing child-19
rows. Do not retune on this round's holdout or promote the absolute-fit comparator.
