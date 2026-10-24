# Phase 1 submission checklist

Local validation date: 2026-10-24. **v12, v13, v14, v18 and v20 each passed all 10 usability cases (user reports); v20 is uploaded (since 2026-10-24) and is the final submission.**
Baseline `versions/v05_platform_passed_ai.py` passed all 10 cases on 2026-10-20.

- [x] Single `ai.py`; class `AI` and required signatures preserved.
- [x] Submission imports limited to NumPy, random, time, and Numba.
- [x] Forbidden module absent, including comments and strings checked by lint.
- [x] Complete legal move set; selected move last; tuples contain native Python integers.
- [x] No legal moves produces an empty list; repeated calls clear stale candidates.
- [x] Input board remains unchanged, including list, float, int and strided inputs.
- [x] Non-8x8 boards tested against an independent reference.
- [x] Search values compared with unpruned minimax; pass and terminal handling covered.
- [x] Development exceptions propagate; normal self-play produces zero fallbacks.
- [x] Tested with Python 3.10.22 / NumPy 1.26.4 / Numba 0.59.1.
- [x] Local first-call timing, sampled middle/endgame timing, and complete-game memory measured.
- [x] Baseline upload: user reported "Upload success, usability test pass, you have pass all 10 test cases" on 2026-10-20; 20 complete match traces verified.
- [x] User confirmed separate 5-second startup, 5 seconds per move, and 180 seconds per side.
- [x] Round 2: 72 TT/minimax comparisons, 48 bound/resume checks, and 139 exact platform-position oracles passed.
- [x] Upload Round 2's `ai.py` (v12) and confirm all 10 usability cases again (user report).
- [x] Use `playto` to validate the new version: 20 complete traces from 10 opponents, 11 wins / 3 draws / 6 losses.
- [x] Keep a stable passing submission through 2026-10-23 23:59 (v18 passing since 2026-10-22); nothing new was uploaded before Phase 1 closed.

Local timing and memory observations are in `NOTES.md`. Usability acceptance of all
six uploads (v05, v12, v13, v14, v18, v20) is confirmed by user reports; final course rank
remains unverified.

## Round 3 v13 (Phase 2 candidate, validated locally)

- [x] PVS in the tree and at the root, inline leaf evaluation; v12 kept in `versions/` for rollback.
- [x] Full regression plus `tests/test_pvs_inline.py`; 139 platform oracles; 200 self-play games, zero fallbacks.
- [x] Matches vs v12, 100 paired games each: 0.640 at 0.4 s and 0.630 at 0.8 s search (CIs above 0.5); 0.500 at 0.2 s.
- [x] Startup about 1.0 s; maximum stress move 4.0033 s; import+init growth 77.0 MiB (v12 70.6 MiB); play delta +0.0 MiB.
- [x] Uploaded 2026-10-21, before Phase 1 closed (user's choice); all 10 usability cases passed (user report).
- [x] `playto` with v13: 20 games against 8 opponents including opp19 (platform batch 03; analysis in progress).

## Round 4 v14 (fitted end-phase rows)

- [x] One fitted row per empties count 1-16 (ridge on exact margins); v13 kept in `versions/` for rollback (sha256 96daf02b...).
- [x] Full regression plus `tests/test_eval_table.py` (112 checks); 200 self-play games, zero fallbacks; correctness review: ship; platform-risk review: ship with record fixes (applied).
- [x] Matches vs v13, 100 paired games each: 0.635 at 0.4 s and 0.8 s (CIs above 0.5); 0.565 at 2 s (CI [0.50, 0.63], inconclusive); 4 s unmeasured.
- [x] Startup about 1.0 s; maximum stress move 4.0005 s; import+init growth 63.5-77.1 MB over two runs (v13 74.0-76.9 MB); play peak delta at most +0.1 MB.
- [x] Uploaded 2026-10-21; all 10 usability cases passed (user report).
- [x] Platform batch 04 recorded with v14 (NOTES.md); superseded by Round 8.
- [x] Rollback item superseded: v18 replaced v14, and v18's rollback target is `versions/v14_round4_fitted_end_ai.py`.

## Round 8 v18 (shape patterns on rows 10-19)

- [x] Pattern tables (edge plus both X squares, 3x3 corner) on rows 10-19, fitted on exact margins of 24,000 self-play training positions (6,000 held out); offline ranking +3.2 points [+2.7, +3.7] over v14.
- [x] Engine equals an independent reference on 10,000 pattern-row positions; other rows equal v14; full suite 15/15 on the final bytes; node rate 0.857x v14; startup 0.09 + 1.02 s; timing stress maximum 4.003 s; tables stored as int8 to keep JIT memory at v14's level.
- [x] Matches vs v14: 0.55 over 200 paired games at 0.4/0.8 s (CI [0.515, 0.588]) and 0.61 over 50 at 2 s.
- [x] Uploaded 2026-10-22; all 10 usability cases passed (user report). Platform batch 05: 7W/3D/10L over 20 games; on the opponent/colour pairings v14 also played, +10 points net versus v14's results.

## Round 10 v20 (shape patterns on rows 20-22)

- [x] Same two shapes on rows 20-22, fitted on exact margins of 7,200 training positions (fits frozen and committed before the test); offline ranking +3.4 points [+2.5, +4.3] over v18 on 1,800 test positions.
- [x] Stage B on the final bytes: suite 19/19, engine equals the frozen model entry by entry, startup 1.20 s, memory +3.0 MB over v18 (import + init 80 MB), timing stress maximum 4.004 s, node rate 0.96x (0.79x when every leaf is at rows 20-22).
- [x] Matches vs v18: 0.530 over 1,000 paired games at 0.5/1.0 s (CI [0.511, 0.549]) and 0.55 over 300 at 2 s; all games normal.

## Before uploading a successor (Round 10 v20 passed every gate)

- [x] Wait until Phase 1 has closed (after 2026-10-23 23:59).
- [x] Upload the file whose sha256 matches the promoted candidate in `versions/round10_validation.json`.
- [x] Confirm all 10 usability cases: v20 uploaded 2026-10-24, 10/10 passed (user report).
- [x] `playto` against strong opponents: platform batch 06, 20 games (4W/2D/14L), all normal, no endgame outcome errors; replays kept.
- [x] No platform problem found, so no rollback (target would be `versions/v18_patterns_candidate_ai.py`).
- [ ] Leave v20 uploaded: the upload live at 2026-11-06 23:59 plays the round-robin.
