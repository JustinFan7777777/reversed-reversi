# Phase 1 submission checklist

Local validation date: 2026-10-20. **Round 2 v12: user confirmed platform acceptance and 10 basic points.**
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
- [ ] Keep a stable passing submission before 2026-10-23 23:59.

Local timing and memory observations are in `NOTES.md`. Both uploads' usability
acceptance is confirmed by user reports; final course rank remains unverified.
