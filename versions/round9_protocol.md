## Active ownership

- Owner: Claude, Round 9 richer patterns, branch `claude/v18-patterns` (after a66624f).
  Codex: read only until this says complete.
- Active upload: **v18** (`ai.py` == `versions/v18_patterns_candidate_ai.py`, sha 7bb4ed79),
  uploaded and passed 10/10 on 2026-10-22. It stays until a successor passes every gate.
- Status: Round 9 running. Round 8: complete, v18 promoted (details below and NOTES.md).

## Round 9 hypothesis and fixed protocol (written before any Round 9 fit)

v18's pattern rows 10-19 were fitted on 24,000 positions and validation kept favouring
weaker regularization, which points to more data and richer shapes helping.

- Data: Round 8's 3,000 games (all of them now training) plus 3,000 new games, seeds
  20000000+, same self-play policy and exact labels at 10-19 empties (positions at 20-23
  are kept unlabelled for a later step). The new games with id % 5 == 0 are the untouched
  test set; validation = new train games with id // 5 % 5 == 1.
- Candidate shape sets, each fitted with the Round 8 recipe (per-row 11 weights, tables
  shared within 10-13, 14-16, 17-19, strength and integer scale chosen without test data):
  P0 = v18's shapes (edge+2X, 3x3 corner); P1 = P0 + 2x5 corner (8 instances);
  P2 = P1 + the three inner lines per side and the diagonals of length 4-8. The set is
  chosen on validation ranking, then scored once on test.
- Stage A gate: the chosen model's within-row ranking on the new test games at least
  1.0 point above v18's frozen model, game-bootstrap 95% interval excluding 0.
- Stages B-C as in Round 8 but against v18: exact engine/reference equality, full suite,
  node rate, startup, memory (int8 tables), timing stress; 200 paired games versus v18
  at 0.4/0.8 s (fresh seeds 21000000+) with the pooled interval entirely above 0.5, then
  50 games at 2 s with point estimate at least 0.5.
- Rows 20-23 (bootstrapped labels from v18 searches whose leaves are rows <= 19) are a
  separate later step with its own protocol entry.

