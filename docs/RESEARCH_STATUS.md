# Research status

Updated: 2026-10-24 (Asia/Shanghai). **Project complete; no further rounds planned.**

- Final engine: **v20** (`ai.py` == `versions/v20_patterns_candidate_ai.py`, sha256 2abbedd9).
  Uploaded 2026-10-24, 10/10 usability cases; it stays live through the Phase 2 deadline
  (2026-11-06 23:59). Rollback target: `versions/v18_patterns_candidate_ai.py`.
- Last platform check: batch 06 (20 games, v20; opponents chosen among the strongest),
  all games normal, no endgame outcome errors (NOTES.md "Platform batch 06").
- Why development stopped: gains per round shrank (+5.0, +3.2, +3.4 ranking points; Round 9
  saturated at +0.8). The next rows (23+) cannot be labelled exactly, and the bootstrapped
  alternative failed in Round 5. A new round would need about two days of compute and a fresh
  upload close to the deadline, for an expected gain of a few percent.
- History: `docs/DEVELOPMENT.md` (summary, v00 → v20) and `docs/NOTES.md` (full lab
  notebook). Round protocols and records: `versions/roundN_protocol.md`,
  `versions/roundN_validation.json` (Round 8's protocol was archived from this file).
- Raw data (ignored by Git): `artifacts/archive_history.tar.xz` holds every round's data,
  including Rounds 8-10 and the batch 05/06 audits; restore with `tar -xJkf
  artifacts/archive_history.tar.xz` from the repo root. Platform replays:
  `artifacts/platform_logs/batch01_v05.zip` ... `batch06_v20.zip`.
- Ownership: Claude (rounds 3-4, 7-10) and Codex (rounds 1-2, 5-6) both finished; no handoff
  is pending.

## Reusable execution notes

- Run long jobs detached (`nohup ... & disown`) and watch them by PID: background tasks of
  an agent session die with the session.
- Exact labelling runs at lowered priority; side jobs slow it. Timed benchmarks (startup,
  Stage B, Stage C) need an otherwise idle machine.
- Run `tools/run_paired_trial.py` without `python -I` (it imports `arena` from `tools/`).
- Stage B: `tools/run_stage_b10.py CANDIDATE FIT --out-dir DIR`; Stage C:
  `tools/run_stage_c10.py` and `tools/pool_paired.py`.
- The platform's 4 s move equals about 2 s on the development Mac (MacBook Air M4).
