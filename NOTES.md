# Reversed Reversi development notes

## Status — 2026-10-20

- Source: user-supplied Claude handoff; Appendix A extracted verbatim.
- `versions/v00_ai.py` preserves the original submission.
- M0/M1/M2: in progress. Platform submission has not been performed.
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
