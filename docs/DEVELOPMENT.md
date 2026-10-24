# Development history: v00 → v20

How the engine grew from a working framework into the final submission, round by round.
Each version is kept byte-for-byte in [`versions/`](../versions/). Each research round has
a pre-registered protocol (`versions/roundN_protocol.md`) and a machine-readable record
(`versions/roundN_validation.json`). The full lab notebook, with every number, is
[`NOTES.md`](NOTES.md).

Built from 2026-10-20 to 2026-10-24 by [JustinFan7777777](https://github.com/JustinFan7777777), working with two AI coding assistants
(Anthropic's Claude Code and OpenAI's Codex). The assistants wrote much of the code and ran
the experiments; every promotion decision followed a gate fixed before the data was seen.

## At a glance

| Version | What changed | Evidence it was better | Platform |
|---|---|---|---|
| v00 | Original framework: bitboards, alpha-beta, hand-set evaluation | Engine equals minimax on 192 checks | |
| v01 | Robustness: resumable time slices, safe fallbacks, pass/terminal handling | 492 interface cases, 200 self-play games, zero fallbacks | |
| v02–v04 | Transposition table, extra features (stable edges, frontier, potential mobility, parity) | Exploratory; most features rejected | |
| **v05** | v04 evaluator with only a negative-frontier term added | 0.660 vs v01, 95% CI [0.565, 0.750] | **10/10** |
| v06–v11 | Round 2 ablations: TT + move ordering, PVS, CSO-tuned weights | See Round 2 table | |
| **v12** | TT + ordering + CSO weights, 160 s game clock | 0.705 vs v05 at 15 ms [0.625, 0.785] | **10/10** |
| **v13** | PVS at every level and at the root; leaves evaluated inline | 0.640 / 0.630 vs v12 at 0.4 / 0.8 s | **10/10** |
| **v14** | One evaluation row per empties count 1–16, fitted on exact endgame margins | 0.635 vs v13 at 0.4 and 0.8 s | **10/10** |
| v15 | Midgame rows fitted on bootstrapped labels; fast-feature prototype | 0.530 [0.470, 0.595]: not promoted | |
| v16 | Weights fitted by ranking sibling moves | Four diagnostic gates failed | |
| v17 | Outcome-first endgame solve at the root | 12 better vs 10 worse, p = 0.83: not promoted | |
| **v18** | Shape-pattern tables (edge + 2X, 3×3 corner) on rows 10–19 | 0.55 vs v14 over 200 games [0.515, 0.588]; 0.61 at 2 s | **10/10** |
| v19 | Richer shape sets on rows 10–19 | +0.81 ranking points, gate +1.0: failed (never built as a file) | |
| **v20** | The same shapes on rows 20–22 (final) | 0.530 vs v18 over 1,000 games [0.511, 0.549]; 0.55 over 300 at 2 s | **10/10** |

Bold versions were uploaded to the course platform; every upload passed all 10
usability tests. "0.640" means the score (win = 1, draw = ½) over paired games against
the previous version; the bracket is a paired-bootstrap 95% confidence interval.

## How a change was accepted

The same discipline applied from Round 2 onward, and it is the main reason the
project ends with a version that is measurably stronger than every earlier one.

1. **Write the gate before the data.** Each round fixed its data split, metric,
   threshold and match plan in a protocol file, committed before any fit.
2. **Paired openings.** Matches start from random openings; each opening is played
   twice with colours swapped, and confidence intervals resample the pairs. This
   removes most of the luck of the opening.
3. **Staged evidence.** Stage A: offline ranking on an untouched test set. Stage B:
   correctness, memory, startup and timing on the final bytes. Stage C: paired
   matches against the current champion at several time budgets.
4. **Frozen fits.** From Round 8 on, fitted models were hashed and committed before
   the test set was opened, so the test could not influence the choice.
5. **Negative results are kept.** Five rounds failed their gates. They are recorded
   just as fully as the successes, and their versions stay in `versions/`.

## Phase 1: a correct, safe engine (v00–v05)

**v00, the framework.** A single-file Numba engine: uint64 bitboards for both sides,
legal-move generation by shifted masks, an explicit-stack negamax alpha-beta (Numba
cannot recurse efficiently), iterative deepening, and an exact solve near the end.
Evaluation is a weighted sum of 11 hand-chosen features, scored for the side to move.
The goal is reversed, so "good" features (corners, mobility) have the opposite sign
from normal Reversi.

**v01, robustness first.** Phase 1 scored only usability, so the first job was never to
crash or time out:
- The search runs in resumable 512-node slices (later 2,048) and checks a monotonic
  clock between slices. A deadline can cut a search anywhere and still return the best
  completed iteration.
- The full legal-move list is published before any search starts, and any exception
  falls back to a generic legal choice.
- Passes, early terminal positions, odd input types (lists, floats, strided arrays,
  6×6 and 10×10 boards) and input immutability are covered by 492 interface tests.

**v02–v04, first experiments.** A transposition table (TT) and five extra features were
each tested in both signs over 100 games. Only a *negative* frontier weight helped
(59/9/32). An SPSA weight-tuning pilot failed on held-out openings and was rejected.

**v05, the Phase 1 submission.** The v04 evaluator, no TT, frontier weight −4. It scored
0.660 against v01 [0.565, 0.750] and 0.655 against v00. It was uploaded on 2026-10-20
and passed all 10 usability tests.

## Round 2: search speed and tuned weights (v06–v12)

Once the instructor confirmed the timing rules (5 s per move, 180 s per player, a
separate 5 s for import and initialisation), the clock policy became: 160 s for our
own game, at most 80% of each move's allowance, capped at 4 s.

| Ablation | Games | Budget | Score | 95% CI |
|---|---:|---:|---:|---:|
| TT + ordering + PVS vs baseline | 100 | 20 ms | 0.680 | [0.605, 0.755] |
| CSO weights alone vs baseline | 100 | 20 ms | 0.655 | [0.565, 0.740] |
| PVS vs TT + ordering (same weights) | 100 | 5 ms | 0.505 | [0.450, 0.560] |
| **Final: TT + ordering + CSO vs baseline** | 100 | 15 ms | **0.705** | **[0.625, 0.785]** |

- **Transposition table:** 131,072 entries (6 MiB) with full-key verification, kept
  across turns and cleared when the weights change.
- **Move ordering:** the cached best move first, then fewest opponent replies.
- **CSO:** a competitive swarm optimiser (Cheng & Jin, 2015) tuned seven weight
  controls over 512 training games. The optimiser is not part of the submission;
  only its weights are.
- **PVS (principal variation search)** showed no gain at 5 ms, so v12 shipped
  without it. Round 3 found out why.

## Round 3: PVS done right (v13)

The 5 ms ablation was too shallow to see PVS work. PVS saves more nodes the deeper the
search goes: 0.88× the nodes at depth 9, and 0.56× at depth 11 with PVS also at the
root, with identical values. v13 enables PVS everywhere. It also evaluates the leaves
of depth-1 nodes inline instead of pushing a stack frame for each; this gives the same
tree about 1.6× faster. At a real budget the search went about one ply deeper.

| v13 vs v12 | Games | Budget | Score | 95% CI |
|---|---:|---:|---:|---:|
| | 100 | 0.2 s | 0.500 | [0.43, 0.57] |
| | 100 | 0.4 s | 0.640 | [0.57, 0.71] |
| | 100 | 0.8 s | 0.630 | [0.555, 0.705] |

Lesson: a speed-up has to be measured at the budget where it matters.

## Round 4: learn the endgame from exact answers (v14)

Near the end of the game the exact final margin can be computed. 1,200 self-play games
gave exact labels for every position with at most 16 empty squares. v14 replaces the
single end-phase weight row with **one row per empties count (1–16)**, fitted by ridge
regression on 8 × the exact margin, using the engine's own 11 features. The model was
tested on 240 held-out games:

- Sibling moves ranked correctly: 0.760 vs v13's 0.714 (+5.0 points per root [4.2, 5.7]).
- Learned fact: stable edge discs are the most important feature at 5–16 empties.

| v14 vs v13 | Games | Budget | Score | 95% CI |
|---|---:|---:|---:|---:|
| | 100 | 0.4 s | 0.635 | [0.565, 0.705] |
| | 100 | 0.8 s | 0.635 | [0.565, 0.705] |
| | 100 | 2.0 s | 0.565 | [0.50, 0.63] |

Because every leaf in one iteration has the same number of empties, rows with different
scales are never compared, so per-row fitting is safe inside alpha-beta.

## Rounds 5–7: three ideas that did not pass

**Round 5 (v15), midgame rows from bootstrapped labels.** Midgame positions cannot be
solved exactly, so rows 17–32 were fitted to imitate a deeper search. Screens were
mixed, and the one promising candidate scored 0.530 [0.470, 0.595] against v14 in
confirmation. Not promoted. A side product, an exact-equivalent fast-feature kernel,
was kept as a prototype.

**Round 6 (v16), rank sibling moves instead of predicting values.** Fitting the
*differences* between legal moves should cancel position-level offsets. The holdout had
too few complete roots, and decision errors rose (1 → 4). All four pre-registered gates
failed, so no matches were run.

**Round 7 (v17), outcome-first endgame.** At the root, try a win/draw/loss-only solve
when it is predicted to fit the time. On 600 fresh positions it changed 22 outcomes:
12 better, 10 worse (p = 0.83). When the solve finished it helped (6 vs 1); when it was
cut off, the shallower fallback hurt (6 vs 9). Not promoted.

The common finding: platform losses were already decided in the **midgame**. Audits of
platform games showed that every loss was lost by 21–27 empties, while the endgame made
almost no outcome errors. The next lever had to be midgame evaluation.

## Round 8: shape patterns (v18)

Linear features cannot see shapes. Two pattern families were added, read from the
side to move:

- **edge + 2X:** the 8 edge squares plus both adjacent X-squares, 3^10 = 59,049 states;
- **3×3 corner block:** 3^9 = 19,683 states.

Each pattern is shared by its four symmetric instances and folded by mirror image.
The tables were fitted on 24,000 exactly solved positions at 10–19 empties, the leaves
of the decisive midgame searches. Tables are shared within buckets (10–13, 14–16,
17–19), and each row keeps its own 11 linear weights.

- **Stage A:** within-row ranking 0.882 vs v14's 0.851 (+3.2 points [+2.7, +3.7]).
- **Stage B:** the engine equals an independent reference on 10,000 positions.
  Storing the tables as int8 instead of int64 cut Numba's JIT memory by about 96 MB,
  back to v14's level.
- **Stage C:** 0.55 vs v14 over 200 paired games [0.515, 0.588], and 0.61 over 50 at 2 s.

## Round 9: richer shapes saturate (v19, rejected)

Adding a 2×5 corner, inner lines and diagonals, with 3,000 more games, improved
ranking by only +0.81 points [+0.51, +1.10] against a +1.0 gate. Patterns at 10–19
empties were saturating.

## Platform calibration

Replaying v18's 20 platform games with our engine at different budgets showed which
local budget reproduces the platform's moves:

| Local budget | 0.25 s | 0.5 s | 1 s | 2 s | 4 s |
|---|---:|---:|---:|---:|---:|
| Same move as on the platform | 80.4% | 83.9% | 92.9% | **97.6%** | 86.3% |

The platform's 4 s move equals about **2 s on the development machine**. So the
leaves of real midgame searches fall at 20–27 empties, beyond v18's pattern rows.

## Round 10: patterns one bucket further out (v20, final)

The same two shapes were added to rows 20–22, with one shared table bucket. These are
the deepest rows that can still be solved exactly: a median of 3.7M, 13.8M and 22.4M
nodes per position.

- **Data:** exact margins for 9,000 positions at 20–22 empties: 18–20 hours on each of
  8 parallel workers, run overnight with a thermal watchdog.
- **Fits:** the pattern model (P0) and a linear-only fallback (L) were both frozen and
  committed before the test set was opened.
- **Stage A:** 0.885 vs v18's 0.852 (+3.4 points [+2.5, +4.3]); every row +3.1 to +3.5.
- **Stage B:** suite 19/19; engine equals the frozen model entry by entry; startup 1.20 s;
  import + init 80 MB (v18: 77 MB); maximum move 4.004 s; node rate 0.96× v18.
- **Stage C vs v18:**

| Budget | Games | Score | 95% CI |
|---|---:|---:|---:|
| 0.5 s | 500 | 0.527 | |
| 1.0 s | 500 | 0.532 | |
| pooled | 1,000 | 0.530 | [0.511, 0.549] |
| 2.0 s (platform equivalent) | 300 | 0.550 | [0.513, 0.587] |

A first Stage C run was cut off when the agent session ended. Under the pre-registered
rule it was declared void, kept, and rerun in full.

## Platform games

Twenty practice games per uploaded version (10 opening pairs, both colours):

| Batch | Version | W / D / L | Note |
|---|---|---|---|
| 01 | v05 | 12 / 1 / 7 | first opponents met |
| 02 | v12 | 11 / 3 / 6 | different opponents |
| 03 | v13 | 5 / 2 / 13 | |
| 04 | v14 | 9 / 2 / 9 | |
| 05 | v18 | 7 / 3 / 10 | +10 points vs v14 on the same opponent/colour pairs |
| 06 | v20 | 4 / 2 / 14 | deliberately chosen top-ranked opponents |

Opponents were chosen differently each time and kept updating their own programs, so
these batches are **not** a strength comparison between versions. The paired local
matches above are. What the batches do verify:
- all 120 games ended normally, with no timeouts, crashes or illegal moves;
- all 841 of our own moves at 14 or fewer empties were checked by an exact solver, and
  none of them turned a win, draw or loss into a worse result.

## Lessons

- **Correctness and time safety come first.** A resumable, interruptible search made
  every later experiment safe to ship.
- **Measure at the budget that matters.** PVS looked useless at 5 ms and was worth
  about a ply at real budgets; the platform's 4 s equals about 2 s locally.
- **Exact labels beat imitation.** Every successful evaluation change (v14, v18, v20)
  was fitted on exactly solved positions; the bootstrapped midgame fit (v15) was not.
- **Gains shrink.** +5.0 → +3.2 → +3.4 ranking points, then saturation at
  rows 10–19. The next step out (rows 23+) would need labels that cannot be computed
  exactly, which is where Round 5 failed. That is why development stopped at v20.
- **Engineering details decide feasibility:** int8 tables to stay under the memory
  limit, mirror-folded patterns, and per-row evaluation scales that are never compared.
