# Reversed Reversi development notes

## Status — 2026-10-21

- Source: user-supplied Claude handoff; Appendix A extracted verbatim.
- `versions/v00_ai.py` preserves the original submission.
- Baseline v05 passed platform usability and supplied 20 validated match traces. User confirmed Round 2 v12 also passed all 10 usability cases, earned 10 basic points, and stayed fixed for a second 20-game batch.
- Baseline v05 platform usability passed all 10 cases; user reported successful upload on 2026-10-20. Round 2 then selected v12 (superseded by Round 3 v13 and Round 4 v14, below).
- Initial M3–M5 pilot: 10-minute single-process allowance. User subsequently authorized Round 2 implementation, experiments, and compute; main Round 2 training/matches took about 9 min 39 s, with correctness/resource checks separately measured.
- Phase 1 deadline: 2026-10-23 23:59; Phase 2: 2026-11-06 23:59.
- Round 3: v13 (PVS at all levels, inline leaves), validated locally against v12 and
  committed as 1261282. The user uploaded it on 2026-10-21, before Phase 1 closed; it
  passed all 10 usability cases (user report).
- Round 4: working ai.py is v14 (end-phase weights fitted on exact labels), validated
  against v13. Uploaded on 2026-10-21; it passed all 10 usability cases (user report)
  and is the platform submission.

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
  (Round 2 decision; Round 3 showed the 5 ms ablations could not detect the depth-
  dependent saving and enabled PVS in v13.)
- Final selected snapshot: `versions/v12_round2_selected_ai.py` (same bytes as ai.py
  until Round 3).
- Baseline rollback: `versions/v05_platform_passed_ai.py`, already platform-accepted.
- Seven-control CSO produced the following embedded weights; columns are corner,
  X, C, edge, discs, mobility, stable, frontier, potential mobility, parity, position:
  opening [70,-2,2,2,2,5,0,-2,0,0,0]; middle [70,-2,2,2,2,7,0,-2,0,0,0];
  ending [70,-2,2,0,10,2,0,-2,0,0,0] (replaced in Round 4 by fitted rows per empties).
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
  optional future work; no untested search pruning was enabled in this release (v12).


## Platform batch 02 — v12 locked, 2026-10-20

- User confirmed v12 passed all 10 usability cases and earned the 10 basic points.
  The user kept the version fixed throughout this batch; ai.py remained byte-identical
  to `versions/v12_round2_selected_ai.py` until Round 3. No strategy or weight changes
  were made during the batch.
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
  splits must avoid overfitting to the friend's two traces. (Round 4 replaced the
  end-phase row with fitted rows that use stable edges and potential mobility.)
- Platform timing is unavailable to the user. No abnormal termination appears in
  these 20 traces; precise resource usage cannot be recovered from board states.
- Complete machine-readable findings: `versions/platform_batch02_analysis.json`.
  Raw searches: `artifacts/platform_batch02_*` (ignored by Git).

## Round 3 — PVS at all levels and inline leaves (v13), 2026-10-20

- User approved an algorithm review and then v13 implementation with validation
  compute. v12 stayed on the platform at the time; v13 and then v14 were uploaded on
  2026-10-21 (see Status).
- Review: three independent reviewers (endgame, midgame search, evaluation) measured
  on a shared seeded set of 2,064 self-play positions; an adversarial verifier re-ran
  the key claims on held-out positions. Only re-verified numbers are recorded here.

### Changes (ai.py)

- `PVS_ENABLED = True`: the existing interior PVS (null windows for later siblings
  at remaining depth >= 2, fail-soft re-search on improvement).
- `ROOT_PVS`: root moves after the first are searched with a null window
  `(alpha, alpha+1)` and re-searched with `(alpha, INF)` on fail-high, through
  `AI._child_value`, which starts a new resumable search per call (the engine ignores
  windows while a suspended frame exists, so stats are reset every call).
- `LEAF_INLINE`: depth-1 nodes score their depth-0 children inline instead of pushing
  a frame per child. Same LSB order, node counts, values and TT contents.
- `SLICE_NODES` names the 2048-node slice; `AI.last_value` records the root value of
  the last completed iteration (telemetry and tests only).

### Measurements

- Why the 5 ms ablation missed PVS: savings grow with depth. Fixed-depth searches
  with a TT, 24 held-out positions (empties 29-47): interior PVS 0.88x / 0.75x /
  0.70x nodes at depths 9 / 10 / 11; with root PVS 0.82x / 0.64x / 0.56x (depth 11
  CI [0.50, 0.63]); identical values and moves; time 0.52x. Without a TT PVS can
  cost more (601 vs 538 nodes, depth 5 from the opening).
- Inline leaves: identical tree on 8 positions at depth 10; 1.58x faster.
- Real 4.0 s budget, 16 positions (games 40-41 of the shared set): mean completed
  depth 12.50 -> 13.25 at 32-50 empties and 15.75 -> 16.88 at 20-28 empties;
  same move in 15/16 positions.

| Trial (v13 vs v12, 6 random opening plies) | Games | Search budget | W/D/L | Score | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Seeds 720000-724009 | 100 | 0.2 s | 44/12/44 | 0.500 | [0.43, 0.57] |
| Seeds 710000-714009 | 100 | 0.4 s | 57/14/29 | 0.640 | [0.57, 0.71] |
| Seeds 730000-734009 | 100 | 0.8 s | 59/8/33 | 0.630 | [0.555, 0.705] |

- The gain appears at the larger budgets, as expected from depth-dependent savings.
  The 0.2 s run includes one v12 timeout caused by machine load (5 parallel arena
  processes); all other games ended normally.
- Colour split (v13 as black / as white): 0.4 s 35/6/9 and 22/8/20; 0.8 s 36/2/12 and
  23/6/21. White remains the weaker colour, matching the platform record.
- Friend replay (opp19, both recorded losses): of 17 own decisions at 24-40
  empties searched for 4 s each, v13 changes one move (27 empties, too early to prove);
  mean depth 13.47 -> 14.47. Those losses are not explained by search depth.

### Validation

- Lint; minimax 192 pairs; resumption 144; PVS/TT window+resume 48 and cache
  invalidation; TT/feature/minimax 72; interface 492; fallback; terminal 9; arena
  limits; engine vs reference 40 games / 2,494 positions; 139 exact platform-position
  oracles (batch 01).
- New `tests/test_pvs_inline.py`: 1,024 same-tree checks (values, nodes and TT
  arrays), 2,048 PVS value/bound checks, 32 root-PVS/slice checks with exact root
  values, and a fake-clock sweep that expires the deadline at every clock read
  (200 runs, cut inside null windows and re-searches): legal moves, exact values.
- Self-play: 200 games / 12,433 positions at 0.005 s; zero fallback and warm-up
  failures.
- Startup: import 0.08-0.09 s + init 0.93-0.97 s (v12: 0.85 s). Timing stress:
  maximum 4.0033 s over 14 cases including 3x slowdown and corrupted nps.
- Memory (same machine, back to back): import + init growth 77.0 MiB (v12 70.6 MiB);
  play delta and peak delta +0.0 MiB. If initialization counts toward the 100 MB
  limit (unconfirmed), the margin is smaller than in Round 2.

### Review findings kept for later (not in v13)

- End-phase weight row (empties <= 16) ranks sibling moves worse than the middle row:
  +3.5 pp pooled [1.1, 6.0] at 12-17 empties (static ranking against exact WLD).
  Next: a per-empties or tapered table fitted on exact labels, accepted only on
  match evidence. Done in Round 4 (v14).
- Eval-based move ordering on top of PVS: plausible further node savings, but chosen
  on the same positions it was measured on; needs held-out confirmation.
- Direct WLD proofs (window [-1, 1], first-win stop) prove 21/24 positions at 20
  empties and 11/24 at 22 within ~4 s, but the current search made 0/15
  outcome-changing errors at 20 empties: low priority, nps-gated if added.
- Rejected: TT cutoffs from deeper entries (cannot fire within one iterative
  deepening run), slice tuning (0.35% overhead), a 1M-entry TT (48 MiB), parity
  endgame ordering (+5% to +50% nodes), Multi-ProbCut before the evaluation is fixed.
- Even-empties positions in the shared set are almost all black to move, so
  colour-specific conclusions need odd-level or colour-split tests.

## Round 4 — Fitted end-phase evaluation (v14), 2026-10-20/21

- User approved item 5 of the Round 3 review (end-phase weights fitted on exact labels),
  to be adopted only on match evidence; v13 was committed first (1261282). The user also
  approved a 2 s confirmation match after a mixed depth signal at the real 4 s budget.
- Platform: the user uploaded v13 on 2026-10-21, before Phase 1 closed, to play
  `playto` games, and then v14 the same day; both passed all 10 usability cases (user
  report). A paired platform comparison (v14 against the opponents of the v13 batch,
  same colours) is in progress.
- Records: snapshot `versions/v14_round4_fitted_end_ai.py`; machine-readable results
  `versions/round4_validation.json`; new test `tests/test_eval_table.py`.

### Changes (ai.py)

- `evaluate()` indexes the weight table by empties when it has more than three rows;
  three-row phase tables (offline tools, older versions) keep the phase lookup.
- `DEFAULT_W` has 65 rows (empties 0-64): rows 1-16 are the fitted `END_W` (units of
  1/8 disc), rows 17-44 the unchanged CSO middle row, rows 45-64 the unchanged opening
  row; row 0 copies row 1 (terminal positions are never evaluated). Rows 61-64 only
  cover direct calls on near-empty boards: `go()` searches only with two or more legal
  moves, so evaluated positions have at most 60 empties.
- Fitted and CSO rows use different scales. This is safe because every leaf of one
  iteration has the same number of empties (passes are free) and TT cutoffs require
  equal depth, so values from different rows are never compared. The correctness
  review confirmed this by randomising every other row (identical values and nodes).
- `AI._search` now copies the weights for its TT invalidation check, so offline tools
  can still assign a 3-row table (review finding; play is unaffected).

### Fit

- Data: 1,200 self-play games (v13 search at 10 ms per move; 8 random opening plies,
  then a random move with probability 0.1), 28,762 positions with 1-24 empties; exact
  final margin for every position with at most 16 empties. Split by game: 960 train,
  240 test.
- Features: the engine's own 11 columns, extracted by calling `evaluate()` with unit
  weights. Within one row the empties-parity column is constant, so it acts as the
  row's intercept.
- Ridge regression per empties 1-16 (penalty 1e-4 x trace/11): in-sample R^2 0.76-0.79
  (0.90 at 1 empty). Rounded to 1/8 disc, the shipped rows have a held-out MAE of
  4.1-5.7 discs (3.5 at 1 empty), within 0.13 disc of the unrounded fit. The
  positional-table column rounds to 0 in every row.
- Learned weights: stable edge discs (every corner is one) are the largest
  non-intercept feature at 5-16 empties, about 0.9-1.25 discs each (0 in v13). Corner, disc and stable columns together make
  a corner cost 11-16 eighths of a disc: 1.8x an ordinary disc at 1 empty, 7.5x at 16
  (v13's end row: 8x). At odd empties the intercept is -3 to -6 discs for the side to
  move (it makes the last move and gains discs).

### Offline comparison (240 held-out games)

| End rows (<= 16 empties) | Sibling pairs ranked correctly (38,191) | W/D/L pairs (10,356) | Per root vs v13 [95% CI] |
|---|---:|---:|---:|
| v13 end row | 0.714 | 0.773 | — |
| Middle row reused | 0.749 | 0.814 | +3.6 pp [2.8, 4.3] |
| **Fitted, 1/8 disc (selected)** | **0.760** | **0.825** | **+5.0 pp [4.2, 5.7]** |
| Fitted, 1/16 disc | 0.759 | 0.824 | +4.8 pp [4.1, 5.5] |
| Fitted, ridge 1e-2, 1/8 disc | 0.750 | 0.815 | +4.1 pp [3.4, 4.8] |

- Sibling ranking: children of test positions at 11-17 empties, static evaluation
  against exact margins (ties count half).
- Rows 1-9 (children of 1,806 held-out roots at 2-10 empties, solved exactly; added
  after review): v14 0.686 vs v13's end row 0.650, +5.5 pp per root [4.1, 6.8]. The
  middle row is no better there (0.660, -0.8 pp [-1.9, 0.4]); v14 is at least as good
  as v13's end row in every row.
- Fixed-depth decisions at 20-21 empties (201 roots, every child W/D/L-proven): 0-6
  errors per table at depths 4/6/8; no difference resolvable.
- Cost, fixed depth from 20-24 empties: 0.95x / 0.85x / 0.82x nodes at depths 6 / 8 /
  10 and 0.90x time at depth 10 (about 9% fewer nodes per second). Full iterative
  deepening to 4-5 empties, 12 held-out positions at 23 empties (12 of 80 planned
  roots; the run was stopped early): 0.76-0.77x nodes (geometric means; 0.73x in
  total), but worse on 2-3 of 12 and up to 1.5x on one. Removing the parity intercept
  changed little (0.97x / 1.00x nodes; 11 and 12 of 12 same moves).

### Matches (v14 vs v13, 6 random opening plies)

| Trial | Games | Search budget | W/D/L | Score | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Seeds 810000-814009 | 100 | 0.4 s | 59/9/32 | 0.635 | [0.565, 0.705] |
| Seeds 830000-834009 | 100 | 0.8 s | 61/5/34 | 0.635 | [0.565, 0.705] |
| Seeds 850000-854009 | 100 | 2.0 s | 50/13/37 | 0.565 | [0.50, 0.63] |

- The 0.4 s and 0.8 s runs pass the promotion rule (paired CI entirely above 0.5). The
  2 s run does not (lower bound 0.500): it is inconclusive. Its score is lower (0.565
  vs 0.635), but the drop is not statistically resolved (z = 1.43). At the real budget
  of about 4 s per move the effect is unmeasured and may be near zero.
- Pooled over the three budgets (300 games; mixed budgets, not an estimate at 4 s):
  0.612 [0.572, 0.650].
- Colour split (v14 as black / as white): 0.4 s 34/5/11 and 25/4/21; 0.8 s 36/0/14 and
  25/5/20; 2.0 s 32/3/15 and 18/10/22. All games ended normally (no timeouts or illegal
  moves); the largest per-game side total at 2 s was 40.6 s.
- The 0.4 s and 0.8 s runs used the 61-row candidate (rows 0-60 identical to the final
  table); the 2.0 s run used the final table and search (the later one-line TT-weights
  fix is not exercised in play).

### Diagnostics at the real 4.0 s budget

- Depth (16 positions, games 40-41 of the shared set): 13.25 -> 13.25 at 32-50 empties;
  16.88 -> 16.38 at 20-28 empties (v14 solved one position, was 1-2 plies shallower at
  four); same move in 13/16. This is repeatable per position, not timing noise: v13 gave
  the same depth and move at 16/16 positions in Rounds 3 and 4. In the same 4 s v14
  searches 0.89-0.96x as many nodes (slower evaluation). The fixed-depth and deep
  samples above are cheaper on average but costlier on some positions; eight
  positions cannot fix the average effect.
- Friend replay (opp19, 17 decisions at 24-40 empties): v14 changes 2 moves; at 24
  empties both choices are proven losses, the other (27 empties) is unjudged. Mean depth
  14.47 -> 14.71. v13 reproduces its Round 3 run exactly (same move and completed depth
  at 17/17 decisions); as in Round 3 it differs from the platform's v12 moves at plies
  32 and 34.

### Validation (final bytes)

- All README checks pass: lint; minimax 192 pairs; resumption 144; interface 492;
  fallback; terminal 9; arena limits; TT/feature/minimax 72; PVS/TT window+resume 48
  and cache invalidation; PVS/inline same-tree 1,024, value/bound 2,048, root/slice 32,
  deadline sweep 196; engine vs reference 40 games / 2,492 positions; self-play 200
  games / 12,450 positions at 0.005 s; zero fallbacks and warm-up failures.
- New `tests/test_eval_table.py`: table layout; a 65-row expansion of v13's phase rows
  matches the 3-row lookup in evaluation and search; row isolation (randomising every
  row except empties - depth leaves values and node counts unchanged); 112 checks. A
  planted off-by-one row index fails it (checked once; not part of the suite).
- Startup: import 0.09 s + init 0.94 s. Timing stress: maximum 4.0005 s over 14 cases.
- Memory (same machine, two back-to-back runs each): import + init growth 63.5 / 77.1
  MB (v13 74.0 / 76.9 MB). RSS readings vary by over 10 MB between runs, so the two
  versions do not differ measurably. Play peak delta at most +0.1 MB.

### Limitations

- All match evidence is head-to-head against v13; the rows were fitted on v13 self-play
  at 10 ms per move, not on platform opponents.
- Rows 17-64 are the unchanged CSO rows; only the end phase was refitted.
- No measurement at the real 4 s budget beyond the 16-position depth table and the
  17-decision replay.


## Round 5 — Codex continuation, 2026-10-21

### Decision and resource scope

- User authorized at most 45 minutes of local computation, with one worker. Reused
  Claude's completed data rather than generating more games. Work is isolated on
  `codex/v15-midgame`; active `ai.py` stays byte-identical to accepted v14.
- **No new strategy promoted.** The independently confirmed middle-game candidate
  scored 48/10/42 against v14 at equal 0.2 s search budgets: score 0.530, paired 95%
  interval [0.470, 0.595]. All 100 games completed normally.
- Candidate/teacher source hashes, actual move/startup measurements, opening seeds,
  full confirmation records, and all rejected screening summaries are committed in
  `versions/round5_validation.json`. Planned secondary transfer matches were skipped
  after the primary acceptance rule failed, rather than used to rescue a claim.

### Recovered data and fitting

- Preserved 31 Claude scratch input/script files in ignored
  `artifacts/codex_round5/claude_round5_inputs/`, with checksums in `input_manifest.json`.
- Data: 1,200 games / 52,800 positions at 17-60 empties. Exact margins are available
  for all 3,600 states at 17-19 empties. Split by game: 960 train / 240 test; no
  shared board between splits for the fitted 17-32 range.
- `tools/fit_midgame.py` implements the existing progressive fit in one process,
  loading a frozen v14 source. Ridge coefficients are rounded before reporting
  errors. Exact rows target eight times the terminal margin; later rows imitate
  fixed-depth search (depth capped at six), with terminal scores converted back to
  the same units. TT is cleared per label. Entire pipeline: 9.4 s.
- Held-out exact MAE at 17-19 empties: approximately 4.54 / 4.32 / 4.10 discs. Lower
  bootstrap MAE at later rows is teacher-imitation error, not new exact evidence.

### Exploratory matched-opening screens

| Candidate versus unchanged-weight kernel control | Games | Search budget | W/D/L | Score | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Fitted rows 17-24 | 40 | 0.04 s | 15/3/22 | 0.4125 | [0.2875, 0.5375] |
| Fitted rows 17-32 | 40 | 0.04 s | 12/11/17 | 0.4375 | [0.3125, 0.5625] |
| Child-evaluation ordering at depth >=5 | 40 | 0.08 s | 14/8/18 | 0.4500 | [0.3500, 0.5500] |
| Exact-fitted rows 17-19 only | 40 | 0.04 s | 18/10/12 | 0.5750 | [0.4875, 0.6750] |

- Only the last candidate cleared the exploratory 55% score gate. Its weights were
  frozen before the independent 100-game confirmation (fresh seeds 940000-940049).
  The final candidate additionally used only exact-equivalent feature calculations.
- No further ridge/parameter search followed the inconclusive confirmation.
- Ordering probe: identical depth-10 scores on 12 new fixtures, 0.87x nodes / 0.94x
  total time, but individual time regressions reached 1.7x. Match screening did not
  justify adopting it.

### Exact-equivalent evaluation prototype

- `versions/v15_fast_features_candidate_ai.py`: two 256-entry uint64 tables (4 KiB)
  replace repeated edge-chain propagation. Horizontal-band expansion computes the
  same eight-neighbor adjacency mask. Weights/search policy remain v14's.
- New `tests/test_stable_edge_lookup.py`: 21,024 oracle/mask checks, 6,156 evaluations
  with production and random three-/65-row weights, 1,152 score/node/cache comparisons,
  including passes, early terminals and 496 resumed searches. Independent adjacency
  oracle: 1,000 further checks.
- Twelve fresh depth-10 fixtures, two timed repetitions per engine: identical scores
  and node counts; time ratio 0.988 overall, 0.951 when fitted-phase leaves are used.
  These small timing samples do not establish stronger timed play. Prototype kept
  separate; not uploaded and not substituted for the active v14.
- Standard regression passed, including 192 minimax pairs, 144 resumptions, 492
  interface cases, 72 TT/minimax checks, 48 bound/resume checks, 1,024 PVS same-tree
  checks, 2,048 bound checks, 32 root/slice checks, 196 injected deadline runs, and
  112 row-isolation checks. The fitted candidate also passed minimax and table checks.
- Final resource checks on the feature prototype: import plus initialization about
  1.01 s; maximum move 4.0016 s across 14 timing-stress cases, including simulated
  slowdown and an inaccurate initial node-rate estimate. Self-play: 200 games /
  12,451 positions, zero fallbacks and zero warm-up failures (43.6 s).
- Memory in one local run: Python/NumPy/Numba baseline 69.9 MiB, import growth
  3.0 MiB, initialization growth 71.1 MiB, then 144.1 MiB after initialization and
  after 30 own moves. Counted post-initialization growth and peak growth both round
  to 0.0 MiB. RSS varies across runs; these figures do not establish platform usage.
- Recorded fitting and matches took 1,105.4 s (18.4 minutes). Including the short
  correctness, resource and replay diagnostics, local computation was approximately
  21 minutes, below the 45-minute allowance. This excludes chat wall-clock time.

### Platform v14 batch (Desktop/v4_chess_log)

- Complete supplied inventory is 18 files: groups 31-39, nine opponents, both colors.
  Independently verified 1,079 legal placements / six legal passes; all terminal,
  all result labels correct, no duplicate complete trace.
- Results: **9 wins / 1 draw / 8 losses**, score 52.8%; black 5/0/4, white 4/1/4.
- Exact audit completed all 144 own decisions with <=16 empties: **zero outcome
  errors and zero terminal-margin regret**. Search took approximately 3.56 s.
- Only opp14 overlaps v3 by opponent ID, both colors. The two batches' aggregate
  scores are not a controlled estimate of upgrade strength; opponent AI versions
  are unknown. Friend opp19 is absent from this batch.
- Black versus opp14 shares 30 placements: at ply31 (30 empties), v13 chooses
  (5,7), v14 chooses (7,4). Equal 4 s cold-table replay reproduces both choices; the
  new middle-game candidate also chooses (7,4). This root remains unproven.
- White shares 41 placements. The ply42 root (19 empties) has exact best margin -8.
  Original v13 move (3,0) achieves -8, v14's recorded (6,0) achieves -9; both lose.
  Cold-table 4 s searches now solve the position and choose (3,0) in all three
  engines. This is a one-disc inaccuracy in an already-lost game, not a missed draw.
- Records: `versions/platform_batch04_analysis.json`,
  `versions/platform_batch04_common_states.json`, `versions/platform_batch04_probe_results.json`.

### Next research step

- Keep v14 on the platform. Avoid treating better regression MAE, fewer nodes, or a
  higher aggregate platform score against different opponents as proven improvement.
- Prefer a decision-ranking objective using counterfactual sibling moves and exact
  labels, plus more diverse opponent positions, over another arbitrary ridge sweep.
  This is a hypothesis for the next round, not an implemented improvement.
- Retain the two shared platform boards as diagnostics; changing a move requires
  fresh continuation play, not replaying the remaining historical moves verbatim.
- Validate promising future candidates at longer budgets on fresh openings before
  replacing the accepted upload.

## Round 6 — exact sibling ranking, 2026-10-21

### Decision and coordination

- User authorized starting the next research step and requested durable records
  for Claude/Codex coordination. Branch: `codex/v16-move-ranking`; handoff entry:
  `RESEARCH_STATUS.md`. Frozen preregistered protocol: `versions/round6_protocol.md`.
- **No new strategy promoted.** All four diagnostic gates failed, so no match
  screen or upload was run. Active `ai.py` remains byte-identical to v14.
- Current-round measured computation: about 347.4 s (5.8 minutes), one heavy worker.
  Together with Round 5's approximately 21 minutes, this remains below the original
  45-minute allowance. Small synthetic tests/analysis add less than one minute.

### Hypothesis and exact labels

- Hypothesis: comparing sibling moves removes unpredictable between-position value
  offsets. This might improve decisions more than absolute-margin regression.
- Selected 120 roots from old training games, one parent per game, balanced at
  18/19/20 empties. Exact full-depth search attempted every legal child with a
  one-second child limit and 240-second total limit. Only fully solved families
  entered fitting. Train labels: 232.6 s, 52 complete roots, 427 exact siblings.
- Fitting used 51 roots / 426 nonterminal children; one complete root had only one
  move. Center features and eight-times exact margins inside each root, normalize
  each root to total weight one, and ridge around the original v14 middle row.
  Lambda was fixed at `1e-4 * trace(X.T @ X) / 10`; parity column 9 stayed fixed.
  There was no hyperparameter grid. A separate absolute ridge is an offline control.
- Pooled integer row for child empties 17-19:
  `[59, -14, -15, 5, 2, 5, 6, -2, 1, 0, -1]`.
- Models were frozen before generating 60 new games (seeds 10010000-10010059).
  Fresh policy: eight random plies, then fixed-depth-5 v14 with 15% random moves.
  Fresh generation plus labels: 104.0 s, 23 complete roots, 184 exact siblings.
- Independence audit verified one root per game and no within-/cross-split family
  overlap under D4 board symmetries, preserving turn/player roles. Pass features
  follow depth-zero search; terminals retain exact encoded values.
- Timeout censoring is severe: train complete roots at 18/19/20 empties are
  33/19/0; holdout 17/6/0. The shared row's child-19 behavior is extrapolation,
  without complete parent-20 supervision. Do not generalize results to that stratum.

### Fresh static-choice results

These are one-step evaluation choices among exact-labeled alternatives, not the
behavior of the normal timed search and not full-game wins or losses.

| Model | Complete holdout roots | WDL mistakes | Mean exact margin regret | Optimal choices |
|---|---:|---:|---:|---:|
| v14 middle row | 23 | 1 | 3.174 discs | 11 |
| Sibling-centered integer row | 23 | 4 | 2.783 discs | 14 |
| Absolute ridge control | 23 | 2 | 2.435 discs | 10 |

- Centered-minus-v14 mean regret: -0.391 discs, game-clustered paired bootstrap
  95% interval [-2.174, 1.435]. This does not establish an improvement.
- Gates: fitting roots below 72; holdout roots below 30; WDL errors increased;
  regret reduction below 0.5 discs. All four failed. Do not relax the gates or
  select the absolute-fit control for matches after viewing this holdout.

### Concrete failure mechanisms

- Three new WDL mistakes occur in fresh roots 18, 33, and 51 (all parent-18).
  Relative to v14's safe choices, the changed positional feature contributes
  +32/+32/+36 toward the bad choices. Its fitted coefficient -0.5714 rounded to -1,
  increasing its magnitude by about 75%. This exposes feature-scale/quantization
  sensitivity in addition to the mismatch between disc-margin and WDL objectives.
- Post hoc scoring with the already stored unrounded row removes two new errors,
  but still has two WDL mistakes and 3.348 discs mean regret. No refitting or new
  candidate followed. Finer precision alone is not shown to solve the problem.
- One holdout sibling pair has identical 11 features but exact margins 0 and -2
  (root 51, squares 40/58). The representation cannot distinguish that pair;
  this collision does not explain all regressions. Full contributions and the
  sensitivity calculation are preserved in `versions/round6_validation.json`.

### Verification and reusable outputs

- Twelve synthetic fitter/metric tests and six generator tests pass. Generator
  tests use independent array move generation and terminal minimax, explicit
  passes/early terminals, expired deadlines, and real one-node resumption slices.
- Experimental snapshot passes lint, 192 minimax comparisons and 112 table/layout
  checks. Its only behavior change from v14 is the three fitted weight rows; it
  is not an accepted upload and was not benchmarked as a competition replacement.
- Final review found the older search/table test scripts ignored a positional
  candidate path. Both now load the requested file and print its resolved path;
  the snapshot checks were rerun successfully. Original misdirected runs remain
  recorded as active-v14 checks, explicitly excluded from candidate evidence.
- `versions/round6_rank_dataset.json` preserves all 180 attempted roots, including
  105 incomplete families with unknown labels. It can reproduce the fit and
  diagnostics without the original scratch files. Raw logs remain in ignored
  `artifacts/codex_round6/`; source/model/data hashes are in the validation report.
- Next hypothesis: prioritize WDL errors, use margin secondarily, choose integer
  precision using training data alone, and test on a new untouched holdout. Obtain
  complete root-20 supervision before modifying child-19 evaluation. No such
  successor is implemented or accepted in this round.
