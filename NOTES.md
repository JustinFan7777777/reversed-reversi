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
