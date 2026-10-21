# Phase 1 submission checklist

Local validation date: 2026-10-21. **v12, v13 and v14 each passed all 10 usability cases and earned the 10 basic points (user reports); v14 is uploaded.**
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
- [x] Upload the current Round 2 `ai.py` and confirm all 10 usability cases again (user report).
- [x] Use `playto` to validate the new version: 20 complete traces from 10 opponents, 11 wins / 3 draws / 6 losses.
- [ ] Keep a stable passing submission before 2026-10-23 23:59 (v14 passing since 2026-10-21).

Local timing and memory observations are in `NOTES.md`. Usability acceptance of all
four uploads (v05, v12, v13, v14) is confirmed by user reports; final course rank
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
- [ ] Paired platform comparison: v14 against the batch 03 opponents with the same colours; keep the replays.
- [ ] If v14 does worse on the platform, roll back to `versions/v13_round3_pvs_inline_ai.py`.
