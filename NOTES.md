# Reversed Reversi development notes

## Status — 2026-10-20

- Source: user-supplied Claude handoff; Appendix A extracted verbatim.
- `versions/v00_ai.py` preserves the original submission.
- Baseline v05 passed platform usability and supplied 20 validated match traces. User confirmed Round 2 v12 also passed all 10 usability cases, earned 10 basic points, and stayed fixed for a second 20-game batch.
- Baseline v05 platform usability passed all 10 cases; user reported successful upload on 2026-10-20. Current ai.py is the new Round 2 candidate.
- Initial M3–M5 pilot: 10-minute single-process allowance. User subsequently authorized Round 2 implementation, experiments, and compute; main Round 2 training/matches took about 9 min 39 s, with correctness/resource checks separately measured.
- Phase 1 deadline: 2026-10-23 23:59; Phase 2: 2026-11-06 23:59.

## Platform facts and open questions

- User's handoff reports the instructor confirmed incremental AI memory accounting; not independently verified here.
- [ ] Whether allocations during initialization count toward memory.
- [x] User clarification (2026-10-20): import + initialization have their own combined 5-second allowance, separate from each move.
- [x] User clarification (2026-10-20): 180 seconds is PER PLAYER, not shared.
- [ ] Actual `time_out` units and values.
- [x] Initial board, black first, move legality, and terminal handling verified by reconstructing 20 platform traces (1,199 placements).
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
- User confirmed successful upload and all 10 usability cases passing on 2026-10-20.
  Practice replays and actual match resource behavior remain pending; final course points and rank are unverified.

- Final complete-game RSS measurement: **31 own moves**, post-init reference **123.2 MiB**,
  end RSS **113.2 MiB**, delta **-10.0 MiB**, conservative peak delta **+0.0 MiB**
  (rounded to 0.1 MiB). Import plus initialization growth **53.1 MiB**, also under 100 MiB.
  Log: `artifacts/final-memory.log`. RSS reclamation explains the negative delta.

## Round 1 remaining work (superseded where Round 2 resolves it)

- M0/M1/M2: completed locally; platform upload accepted all 10 usability cases (user report, 2026-10-20). Live-match verification remains pending.
- M3: fixed-size TT implemented, tested, and rejected for lack of significant improvement;
  PVS/aspiration windows, richer ordering, and a specialized 14–16-empty solver remain pending.
- M4: five added feature directions tested both ways; frontier-only improvement confirmed
  on independent opening seeds and merged. Stability feature is a conservative subset.
- M5: reproducible SPSA pilot and historical snapshots exist; pilot weights rejected.
  Larger tuning runs, scaled perturbations, and CSO remain pending a separate compute allowance.
- M6 optional opening books/MCTS comparison: not undertaken.
- M7 platform freeze and round-robin verification: pending; final course outcomes unverified.

## Platform acceptance — 2026-10-20

- User reported: "Upload success, usability test pass, you have pass all 10 test cases".
- This records acceptance of the submitted version; platform matches and final course outcomes are separate.
- GitHub backup: private repository requested by the user; course PDFs, environments, and caches excluded.

## Round 2 — Instructor timing clarification and candidate experiments

- User authorized implementation, experiments, and compute on 2026-10-20.
- Correct limits: 180 s per player; import + initialization separately <=5 s; each move <=5 s.
- Arena now checks each side independently and times startup separately; simulated-clock regression tests pass.
- Stable platform-accepted v05 retained in `versions/v05_platform_passed_ai.py`; main submission remains unchanged during experiments.
- New resource policy under test: 160 s own-game budget, 80% of the move allowance, capped at 4.0 s; startup stays outside the game clock.
- Search candidates: fixed 6 MiB full-key TT retained across turns with weight-change invalidation; cached move and opponent-mobility ordering; resumable explicit-stack PVS.
- CSO is an independent NumPy implementation of loser-learning pairwise competition (Cheng/Jin, DOI 10.1109/TCYB.2014.2322602). Seven normalized controls adjust the original active evaluator across phases. No optimizer enters the submission.
- CSO training: 8 particles, 6 generations, 32 evaluations x 16 deterministic depth-5 matches = 512 games; about 13.2 s. Training openings use seeds 110000–110007 and do not include the 20 platform replays.
- Training best: 15/1/0. This is exploratory training fitness, not a held-out strength claim.
- Fixed-depth benchmark: baseline 3,020,115 nodes / 0.2358 s; TT+ordering 1,049,515 / 0.0938 s; PVS version 866,243 / 0.0839 s. All 48 engine/position/depth values agree.
- Equal 0.25 s searches on eight fixtures: baseline depths [9,10,9,9,11,10,12,11]; PVS candidate [10,11,10,10,12,12,16,15].
- Fresh PVS startup: import 0.23 s + init 0.88 s; first and second go about 4.00 s each. Startup and go are separate limits.
- Matched-opening timed trials, independent confirmation, and transfer checks are in progress; results are budget-specific.

### Round 2 completed results — 2026-10-20

All timed trials use matched fresh random openings with colors exchanged. Confidence
intervals resample opening pairs, and all hard 5 s / 180 s checks passed. The training
openings, screening openings, and final confirmation openings are disjoint.

| Trial | Games | Search budget | W/D/L | Score | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| TT+ordering+PVS vs baseline | 100 | 20 ms | 65/6/29 | 0.680 | [0.605, 0.755] |
| CSO weights alone vs baseline | 100 | 20 ms | 63/5/32 | 0.655 | [0.565, 0.740] |
| PVS+CSO vs PVS+original weights | 100 | 20 ms | 64/8/28 | 0.680 | [0.590, 0.770] |
| PVS vs TT+ordering, original weights | 100 | 5 ms | 49/3/48 | 0.505 | [0.450, 0.560] |
| PVS+CSO vs baseline, independent set | 100 | 25 ms | 82/5/13 | 0.845 | [0.780, 0.910] |
| PVS+CSO transfer sanity pair | 2 | 250 ms | 2/0/0 | 1.000 | Insufficient sample |
| PVS vs TT+ordering, learned weights | 100 | 5 ms | 51/5/44 | 0.535 | [0.480, 0.590] |
| **Final TT+ordering+CSO vs baseline** | **100** | **15 ms** | **68/5/27** | **0.705** | **[0.625, 0.785]** |
| Final candidate transfer sanity pair | 2 | 1 s | 2/0/0 | 1.000 | Insufficient sample |

- PVS remains an experimental option. Its extra strength over TT+ordering was not
  established in either ablation, so **PVS_ENABLED=False** in the submitted candidate.
- Final selected snapshot: `versions/v12_round2_selected_ai.py` (same bytes as ai.py).
- Baseline rollback: `versions/v05_platform_passed_ai.py`, already platform-accepted.
- Seven-control CSO produced the following embedded weights; columns are corner,
  X, C, edge, discs, mobility, stable, frontier, potential mobility, parity, position:
  opening [70,-2,2,2,2,5,0,-2,0,0,0]; middle [70,-2,2,2,2,7,0,-2,0,0,0];
  ending [70,-2,2,0,10,2,0,-2,0,0,0].
- Main training/match time: 13.2 + 400.4 + 165.6 = 579.3 seconds (~9 min 39 s),
  covering 512 training games and 704 timed validation/ablation/pilot games.
- Regression: engine 40 games/2,491 positions; minimax 192 pairs; resumable search
  144 comparisons; interface 492 cases; terminal/pass 9; TT/minimax 72;
  bound/window/resume 48 plus poisoned-cache invalidation; 139 existing exact
  platform-position oracle values matched. No illegal moves or hidden fallbacks.
- Self-play: 200 games/12,429 positions at a small correctness budget; zero fallback
  and warm-up failures. These games are stability checks, not a strength estimate.
- Fresh startup: import 0.09 s + init 0.84 s = 0.93 s, within its separate 5 s limit.
- Timing: 14 sampled middle/endgame stress cases, including synthetic slice delays
  and corrupted nps telemetry. Maximum observed go 4.003145 s (<4.5 s target, <5 s rule).
- Memory: TT array exactly 6 MiB. Import/init growth 61.6 MiB in total; post-init
  complete-game RSS delta -6.7 MiB and rounded peak delta +0.0 MiB, with 30 own moves.
  The interpreter/NumPy/Numba baseline is excluded per instructor guidance reported by user.
- Raw logs/results: `artifacts/round2_*`; committed summary:
  `versions/round2_experiment_report.json`, `versions/final_validation.json`.

### Remaining validation (updated after the second platform batch)

- User confirmed that current Round 2 v12 was uploaded, passed all 10 usability cases, and earned the 10 basic points.
- The 100-game evidence above is at short budgets; the 1 s two-game check is only
  a transfer sanity test. The second platform batch supplies 20 formal-setting games;
  opponents differ from the first batch, and their AI versions are unknown.
- The platform does not expose move/game timing to the user. Complete terminal traces
  establish normal game outcomes, not precise measured resource headroom.
- Specialized endgame code, aspiration windows, and stronger pattern evaluators remain
  optional future work; no untested search pruning was enabled in this release.


## Platform batch 02 — v12 locked, 2026-10-20

- User confirmed v12 passed all 10 usability cases and earned the 10 basic points.
  The user kept the version fixed throughout this batch; ai.py remains byte-identical
  to `versions/v12_round2_selected_ai.py`. No strategy or weight changes were made.
- Input: Desktop/new_chess_log, 20 traces (groups 11–20), 10 opponent IDs, 1,199
  independently verified legal placements, all terminal, all result labels correct.
- Results: **11 wins / 3 draws / 6 losses**; black 7/2/1, white 4/1/5 (W/D/L).
  Score is 62.5% counting draws as half, equal to batch 01. Opponent IDs have zero
  overlap, and opponent AI versions are unknown, so this does not estimate the
  causal effect of our upgrade. The black/white difference is exploratory (10 each).
- Exact-to-terminal audit completed **all 160 own decisions with <=16 empties**:
  zero W/D/L errors and zero terminal-margin regret. One initially interrupted
  16-empty position was retried and completed; it was a draw before the opponent
  subsequently allowed the recorded win. Incomplete earlier searches remain unknown.
- Friend opp19: two recorded losses, both 34–30, own/opponent corners 3/1.
  User reports six consecutive losses overall; unrecorded games are contextual
  evidence, not additional rows in the 20-game statistics.
- Exact narrow-window terminal proofs establish already-lost positions before
  black game ply 41 (20 empties) and white game ply 38 (23 empties). These are
  latest proven bounds on when a loss had happened, not the first losing move.
  Narrow-window bound magnitudes are not exact final margins.
- White game: after ply 44 mobility is 4 vs 9; after ply 46, 3 vs 8. All three
  legal placements at ply 48 are corners. Black game: corner at ply 45 was a
  choice among seven moves, including five noncorner moves. The mechanisms differ.
  Both games were already lost before those corners, so corner ownership alone
  does not identify the causal mistake.
- Separate NumPy-reference recursive minimax checked six final decisions across
  303 cached states (~0.041 s), confirming optimal margin -4 in both recorded
  endings. Some noncorner alternatives lose by much more; avoiding corners
  unconditionally would be incorrect.
- Next experimental priorities: quality of middle-game mobility, smoother phase
  weights (17 -> 16 empties changes discs 2 -> 10 and mobility 7 -> 2), and bounded
  stable-edge/potential-mobility ablations. These remain hypotheses; train/test
  splits must avoid overfitting to the friend's two traces.
- Platform timing is unavailable to the user. No abnormal termination appears in
  these 20 traces; precise resource usage cannot be recovered from board states.
- Complete machine-readable findings: `versions/platform_batch02_analysis.json`.
  Raw searches: `artifacts/platform_batch02_*` (ignored by Git).
