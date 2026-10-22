# Round 10 protocol: shape patterns at rows 20-22 (approved by the user 2026-10-22, frozen before data)

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

Details fixed now:
- Validation = train games (both sources) with id // 5 % 5 == 1; Round 9 and new games are
  kept apart by source, so equal ids do not collide.
- Shape set: P2 is the default; P0 is chosen instead if its validation ranking is within
  0.2 points of P2. Strength grid as Round 9 (0.001-10), int8-feasible integer scale.
- The comparison is v18's own evaluation (v14's linear rows 20-22).
- Data tools: `tools/label_rows.py` (exact margins for the unlabelled Round 9 records) and
  `tools/build_exact_rows.py --min-empties 20 --max-empties 22` (new games).

Revision 1 (2026-10-22, before any label was used; user-approved): a 30-position timing
probe showed full-window solves at 20 empties average about 6 s (95M nodes), roughly x3 per
extra empty, so labelling costs about 60 CPU-hours. Data is therefore only the 9,000
Round 9 positions at 20-22 (no new games), labelled on 8 overnight workers; the split
(test id % 5 == 0, validation id // 5 % 5 == 1) is unchanged. The probe output
(`artifacts/claude_round10/probe.json`) is a timing check only and is not used.

Revision 2 (2026-10-22, before any label was used; labelling in progress, no fit run).
Measured on smoke builds fitted to synthetic labels (cost depends on the shape set, not on
the fitted values; `tools/bench_memory.py` 3 runs each, `tools/bench_node_rate.py`
leaf-heavy probe with the labelling workers paused):

| Set | import + init | vs v18 | nodes/s, all leaves at rows 20-22 |
|---|---:|---:|---:|
| v18 | 85-86 MB | - | 1.00 |
| P0 | 86-89 MB | +1 MB | 0.80 |
| P1 | 87-91 MB | +3 MB | 0.68 |
| P2 | 105-107 MB | +20 MB | 0.45 |

P2 exceeds the 100 MB limit (Round 8 rejected a 101 MB build) and P1's extra 2x5 corner
cost 15% more node rate for +0.1 validation points in Round 9. Changes:
- Shape set: **P0 only** (edge+2X, 3x3 corner); no set selection. Strength and integer
  scale as before.
- Stage A unchanged (+1.5 points over v18's rows 20-22, game-bootstrap 95% interval
  excluding 0), scored once on the frozen integer model:
  `tools/fit_patterns10.py ... --open-test artifacts/claude_round10/fit_frozen.json`.
- Stage B definitions, idle CPU (labelling finished):
  node rate = `tools/bench_node_rate.py` standard set (Round 8's yardstick: 30 random
  positions at 22-34 empties, seed 8_300_001, 2 s cold searches, alternating) at least
  0.85x v18; the leaf-heavy probe is reported, not gated.
  Memory: import + init within +5 MB of v18 (3 runs each, same session).
  Startup (import + init) <= 1.5 s. Timing stress `tools/bench_timing.py`: maximum move
  time no worse than v18's + 0.05 s and below 4.5 s. Equality `tests/test_patterns_hi.py`
  and the full README suite on the candidate as `ai.py` in a repo copy.
- Stage C unchanged in size and thresholds: 100 paired games at 0.5 s (seed 23000000) and
  100 at 1.0 s (seed 23100000), pooled paired 95% interval entirely above 0.5; then 100
  paired games at 2.0 s (seed 23200000), point estimate at least 0.5. At most 4 match
  processes at once (the M4 has 4 performance cores), nothing else heavy running.

Revision 3 (2026-10-22, before any fit; each point approved by the user). Prompted by a
four-agent review of the pipeline (no defect in scoring, engine or labels; guards added).
- Revision 2 is approved as written (P0 only; memory within +5 MB of v18).
- Speed: the gated `tools/bench_node_rate.py` standard set (Round 8's yardstick, >= 0.85x
  v18) is a general sanity check only. It cannot detect this change's cost, since only
  about 1 in 6 of its searches end at rows 20-22. The leaf-heavy ratio (P0 smoke: 0.80x)
  is the cost on the affected decisions and is always reported next to it; the fixed-time
  Stage C matches decide the speed/accuracy trade-off.
- Data guard: the fit runs only after all 8 workers have finished (every log ends with
  "worker J: 1125 records"). `tools/fit_patterns10.py` asserts 9,000 unique positions, 3
  per game, records a data hash, and `--open-test` refuses a frozen file fitted on other
  data. Neither mode overwrites an existing output.
- Linear-only fallback L: the same recipe with the 11 linear features and no tables
  (`--set L`), fitted and frozen at the same time as P0 (`fit_frozen_linear.json`),
  before the test is opened. Order: P0 through Stage A, B, C. If P0 fails Stage A or C,
  L goes through the same Stage A gate (+1.5 points over v18, interval excluding 0),
  Stage B and Stage C (same sizes, fresh seeds). If P0 fails Stage B, the user is asked
  before the fallback runs. Two sequential candidates put the chance of promoting a
  candidate that is not truly better at about 5% instead of 2.5%.
- Stage C sizes: 500 paired games at 0.5 s plus 500 at 1.0 s, pooled paired 95% interval
  entirely above 0.5; then 300 paired games at 2.0 s with point estimate at least 0.5.
  Budgets run in sequence, 3 match processes each (`tools/run_paired_trial.py`), seed
  blocks S, S+1000, S+2000 with 168/166/166 games (2.0 s: 100/100/100) and
  `--seconds 100000`. S = 23000000 / 23100000 / 23200000 for 0.5 / 1.0 / 2.0 s; the
  fallback uses 23300000 / 23400000 / 23500000. After each stage: opening seeds disjoint
  across processes, every game complete with reason "normal".
- Engine identity: Stage B runs `tests/test_patterns_hi.py ai.py --frozen FIT.json` in
  the candidate repo copy (engine rows, every table entry and 900 values equal the scored
  model). The validation record stores the sha256 of this protocol, the frozen files, the
  test outputs and the candidate.
- Compute: Stage C about 4.6 h per candidate on an otherwise idle machine.
