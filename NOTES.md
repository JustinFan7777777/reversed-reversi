# Reversed Reversi development notes

## Status — 2026-10-20

- Source: user-supplied Claude handoff; Appendix A extracted verbatim.
- `versions/v00_ai.py` preserves the original submission.
- M0/M1: locally validated. M2: measured stress suite passed; platform verification pending.
- Platform submission has not been performed.
- M3–M5: at most 10 minutes of single-process experiment compute authorized.
- Phase 1 deadline: 2026-10-23 23:59; Phase 2: 2026-11-06 23:59.

## Platform facts and open questions

- User's handoff reports the instructor confirmed incremental AI memory accounting; not independently verified here.
- [ ] Whether allocations during initialization count toward memory.
- [ ] Whether import/initialization time counts toward the 5-second limit.
- [ ] Whether 180 seconds is shared or per player.
- [ ] Actual `time_out` units and values.
- [ ] Starting position/colors, checked against a downloaded platform replay.
- [ ] Whether Phase 1 checks move quality as well as the legal-move set.
- [ ] Phase 1 raw points versus course points, and which submission determines the grade.
- [ ] Whether each game receives a new AI object/process.
- [ ] Module restriction checker and platform CPU characteristics.

## Experiments

Results will identify environment, seeds, budgets, W/D/L, and timing. Local checks are not platform acceptance.

### M0 — Original framework

- Environment: macOS arm64, CPython 3.10.22, NumPy 1.26.4, Numba 0.59.1, llvmlite 0.42.0.
- Engine: 40 games / 2,497 positions; alpha-beta equals minimax on 192 position/depth pairs.
- Standard-opening sanity games, timeout 0.5 s: random 6/0/0, greedy 10/0/0 (W/D/L).
- Repeated deterministic openings make these sanity checks, not independent strength evidence.
- Fresh process: import 0.18 s, initialization 0.75 s, first go 1.56 s; total 2.50 s.
- Memory: post-init delta -4.0 MiB, conservative high-water delta +0.0 MiB.
- Original Linux-only RSS tool required macOS support; sandbox blocks `ps`, so RSS measurement ran with explicit escalation.
- Log: `artifacts/m0-validation.log` (memory initially failed in the sandbox, then passed separately).

### M1/M2 — v01 robustness

- Complete legal moves are published before search; errors use independent generic choice.
- Added fallback/warm-up failure counters, including intentional error-injection tests.
- Node-estimate-only timing replaced by resumable 512-node slices and monotonic clock checks.
- Removed minimum budgets that could exceed a small timeout or an exhausted game budget.
- Endgames retain completed shallow iterations before attempting the exact remaining depth.
- Move buffer expanded to 64 slots, sufficient even for arbitrary supplied 8x8 positions.
- Engine: 40 games / 2,495 positions. Minimax: 192 pairs. Resume equivalence: 144 comparisons.
- Interface: 492 cases including forced pass, one move, eight directions, row wrapping,
  full/early terminal boards, int32/int64/float/list/noncontiguous/negative-stride inputs,
  6x6 and 10x10 boards, native-int tuples and input immutability.
- Seeded self-play: 200 games / 12,447 positions, after randomized opening plies;
  zero fallback and warm-up failures. Timeout 0.005 s is a correctness/stability budget.
- A/B against v00: 2/3/5 over 10 games, six random opening plies, timeout 0.1 s,
  seed 2026; score 0.350, paired-bootstrap 95% CI [0.150, 0.500]. No strength claim.
- Fresh process: import 0.08 s, initialization 0.71 s, first go 2.00 s; total 2.79 s.
- Stress: 14 sampled middle/endgame cases, including synthetic 3x slice slowdown
  and nps deliberately set to 1e12. Maximum observed go: 2.000225 s; not a universal guarantee.
- Full-game RSS sampling: 30 own moves; post-init delta -43.7 MiB, high-water delta +0.5 MiB.
  Negative RSS growth reflects memory reclamation, not negative allocation.
- Recommended conservative controls: `GAME_BUDGET_SEC=80`, `PER_MOVE_FRACTION=0.40`,
  per-search cap 2.0 s. The opponent's time and platform initialization accounting remain external facts.
- Logs: `artifacts/m2-validation.log`, `artifacts/m2-stress.log`, `artifacts/timing.json`.

### Experimental candidates

- v02: 1.50 MiB fixed-size transposition table with full position-key verification,
  exact-depth/bound flags, cached move ordering, and resumable stack support.
- v03: optional stable edge chains (a conservative stability subset), frontier,
  potential mobility, side-to-move parity, and grouped positional masks.
- Each passed 72 TT/feature comparisons against unpruned minimax, including active
  extra-feature weights and unregistered importlib loading.
- Exploratory feature sign sweeps and SPSA use training seeds; promotion requires
  at least 100 games and a paired-bootstrap interval entirely above 0.5 on held-out seeds.
- `tools/run_experiments.py` runs sequentially with a 540-second stop budget; the
  reserve covers setup/correctness checks under the user's 10-minute experiment allowance.
- PVS/aspiration windows, a specialized 14–16-empty solver, CSO, and opening books
  remain future experiments; none is claimed complete by these candidates.

### M3–M5 bounded pilot results

- Complete run: 309.6 s, 1,380 games, single process; a first attempt (~24 s)
  stopped on an overly strict 10 ms experiment timeout. No course 5 s timeout occurred.
- The first TT run scored 56/8/36; a full repeat scored 53/6/41, interval [0.465, 0.655].
  The repeat does not establish improvement, so TT is not merged on its own.
- Feature ablations: 100 games per sign, six seeded opening plies, paired colors,
  0.01 s soft timeout setting (nominal 4 ms search); 5 s hard limit.

| Feature | Positive weight W/D/L | Negative weight W/D/L |
|---|---:|---:|
| Stable edge chains (magnitude 10) | 47/7/46 | 2/0/98 |
| Own frontier advantage (4) | 14/1/85 | 59/9/32 |
| Potential mobility (4) | 53/5/42 | 4/2/94 |
| Side-to-move parity (4) | 45/6/49 | 46/5/49 |
| Positional penalty (1) | 22/6/72 | 5/5/90 |

- Only negative frontier weight qualified for separate confirmation; these are
  exploratory comparisons, not ten independent confirmatory claims.
- SPSA: four iterations, plus/minus perturbations, eight paired-opening games each,
  historical opponent pool. Proposed weights failed held-out v01: 38/10/52,
  score 0.430, interval [0.340, 0.515]. Rejected; tuning is a pilot, not convergence.
- The pilot against v00 scored 21/1/78, but its 50 ms floor gives it more time than
  the 4 ms candidate budget. It is not a fair strength comparison. The separate
  confirmation uses timeout 0.125 s so both sides receive a 50 ms intended budget.
- Sanity checks for the rejected tuned candidate: greedy 9/1/0, random 6/0/0.
- Maximum observed move in the pilot: 0.0724 s; combined go time per game: 0.7770 s.
- Raw records: `artifacts/experiments.json`; committed summary: `versions/experiment_results.json`.
- Extra features, TT, and the rejected weights are preserved as experimental versions.

### Final candidate — v05 Phase 1 local-ready

- Promoted `v04` evaluator without TT, with only the new frontier column enabled
  at weight -4 in all three phases. SPSA weights were rejected.
- Held-out seed 60000, 100 games vs v01, timeout 0.01 s: **64/4/32**, score 0.660,
  paired-bootstrap 95% interval **[0.565, 0.750]**.
- Held-out seed 70000, 100 games vs v00, timeout **0.125 s** (both receive the same
  nominal 50 ms search budget): **64/3/33**, score 0.655, interval **[0.555, 0.745]**.
- Confirmation took 185.5 s. Successful experiment batches totaled 495.1 s;
  including the aborted first pilot (~24 s) and candidate checks, M3–M5 experiments
  stayed below the authorized 10-minute compute allowance.
- Snapshot: `versions/v05_phase1_ai.py`; summary: `versions/confirmation_results.json`;
  complete records: `artifacts/no-tt-confirmation.json`.
- Final regression: lint passed; engine 40 games / 2,494 positions; minimax 192 pairs;
  resumable search 144 comparisons; interface 492 cases; explicit terminal/pass 9 cases;
  injected exception recovery passed.
- Final self-play: **200 games / 12,452 positions**, zero fallback/warm-up failures.
- Final baselines (six seeded random opening plies, timeout 0.1 s): random **6/0/0**,
  greedy **9/0/1**. These are sanity checks, not broad competition ranking estimates.
- Fresh process: import **0.08 s**, initialization **0.88 s**, first go **2.00 s**;
  combined import/init/first go **2.96 s**.
- Final timing: **14 stress cases**, including synthetic 3x slowdown and broken nps;
  maximum observed go **2.000193 s**. Recommended controls remain 80 s / 0.40 / 2 s.
- The complete final regression log is `artifacts/final-validation.log`.
- Platform upload, usability acceptance, practice replays, and actual server resource
  accounting remain pending. No local result establishes course points or final rank.

- Final complete-game RSS measurement: **31 own moves**, post-init reference **123.2 MiB**,
  end RSS **113.2 MiB**, delta **-10.0 MiB**, conservative peak delta **+0.0 MiB**
  (rounded to 0.1 MiB). Import plus initialization growth **53.1 MiB**, also under 100 MiB.
  Log: `artifacts/final-memory.log`. RSS reclamation explains the negative delta.

## Remaining work

- M0/M1/M2: completed locally; platform acceptance still requires an upload.
- M3: fixed-size TT implemented, tested, and rejected for lack of significant improvement;
  PVS/aspiration windows, richer ordering, and a specialized 14–16-empty solver remain pending.
- M4: five added feature directions tested both ways; frontier-only improvement confirmed
  on independent opening seeds and merged. Stability feature is a conservative subset.
- M5: reproducible SPSA pilot and historical snapshots exist; pilot weights rejected.
  Larger tuning runs, scaled perturbations, and CSO remain pending a separate compute allowance.
- M6 optional opening books/MCTS comparison: not undertaken.
- M7 platform freeze and round-robin verification: pending; final course outcomes unverified.
