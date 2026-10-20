# Phase 1 submission checklist

Local validation date: 2026-10-20. **Platform acceptance is pending.**

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
- [ ] Upload the final `ai.py` on the campus platform and record all usability-test results.
- [ ] Use `playto` after a successful upload; check actual platform time/memory behavior.
- [ ] Confirm initialization accounting and time limits with the TA/platform.
- [ ] Keep a stable passing submission before 2026-10-23 23:59.

Local timing and memory observations are in `NOTES.md`. The final round-robin
ranking and Phase 1 acceptance cannot be established by local testing.
