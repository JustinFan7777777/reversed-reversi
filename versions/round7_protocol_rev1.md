# Round 7 protocol, revision 1

Full labels of every move cost 30-60 s per position at 22-23 empties and were often
capped; the first run was stopped after about 13 minutes (114 positions, never used for
any engine comparison) and is discarded. Replacements for the Positions, Labels,
Benchmark and Primary gate items above; everything else stands.

- Positions only, no labels up front: new seeds, training 150 games (16000000+, with the
  15-24 empties cost probes), fresh holdout 600 games (17000000+), one position per game
  at a seeded target of 20-23 empties, same self-play policy and selection rule.
- Both engines choose on every position at 4.0 s from a cold table (same process,
  alternating order, at most three timing processes).
- Labels only where the two chosen moves differ: the exact outcome of each chosen move
  (outcome-only solve, cap 1B nodes per child; capped positions are reported, excluded).
- Primary gate (holdout, opened once after v17 is frozen): among positions where the two
  choices reach different outcomes, v17's is better more often than worse, exact
  two-sided sign test p < 0.05.
- Training sanity check: the same procedure on the training positions after C[e] is set.
- Compute for this revision: about 1 hour plus the non-regression matches (about
  40 minutes); positions, labels and cost probes may use up to 6 workers.

