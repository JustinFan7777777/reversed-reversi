"""The edge-chain lookup must preserve the frozen v14 engine exactly.

    python tests/test_stable_edge_lookup.py [candidate.py] [--reference baseline.py]

The mask oracle deliberately uses Python square coordinates rather than the
candidate's bit packing. Search comparisons require identical scores, node counts
and cache contents, including resuming after one-node slices. This is a bounded
correctness regression, not a playing-strength or timing benchmark.
"""
import argparse
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ref_engine as R  # noqa: E402

FULL = (1 << 64) - 1
EDGES = (tuple(range(8)), tuple(range(56, 64)),
         tuple(range(0, 64, 8)), tuple(range(7, 64, 8)))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEBUG_RAISE = True
    return module


def stable_oracle(discs):
    """Only uninterrupted own edge chains from an occupied corner are stable."""
    stable = 0
    for edge in EDGES:
        for squares in (edge, reversed(edge)):
            for square in squares:
                bit = 1 << square
                if not discs & bit:
                    break
                stable |= bit
    return stable


def random_disjoint(rng, occupied):
    """Include every occupancy count; unreachable boards also stress bit packing."""
    squares = rng.sample(range(64), occupied)
    me = opp = 0
    for square in squares:
        if rng.getrandbits(1):
            me |= 1 << square
        else:
            opp |= 1 << square
    return np.uint64(me), np.uint64(opp)


def reachable_position(rng, placements, engine):
    board, color = R.initial_board(), R.BLACK
    for _ in range(placements):
        moves = R.legal_moves(board, color)
        if not moves:
            color = -color
            moves = R.legal_moves(board, color)
            if not moves:
                break
        board = R.apply_move(board, color, rng.choice(moves))
        color = -color
    return engine.board_to_bb(board, color)


def search(engine, me, opp, depth, alpha, beta, weights, use_tt, chunk):
    su = np.zeros((engine.MAXPLY, engine.SU_COLS), dtype=np.uint64)
    si = np.zeros((engine.MAXPLY, engine.SI_COLS), dtype=np.int64)
    stats = np.zeros(7, dtype=np.int64)
    table = np.zeros((engine.TT_SIZE if use_tt else 0, 6), dtype=np.uint64)
    nodes = yields = 0
    while True:
        stats[0], stats[1], stats[2] = 0, chunk, 0
        value = int(engine.alphabeta(me, opp, depth, alpha, beta,
                                     stats, weights, su, si, table))
        nodes += int(stats[0])
        if stats[2] != 2:
            assert stats[2] == 0 and stats[3] == 0, stats
            return (value, nodes), table, yields
        yields += 1
        assert yields < 1_000_000, "Search failed to finish after bounded node slices"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?",
                        default="versions/v15_fast_features_candidate_ai.py")
    parser.add_argument("--reference", default="versions/v14_round4_fitted_end_ai.py")
    args = parser.parse_args()
    reference = load(ROOT / args.reference, "edge_lookup_reference")
    candidate = load(ROOT / args.candidate, "edge_lookup_candidate")
    assert np.array_equal(reference.DEFAULT_W, candidate.DEFAULT_W), (
        "This exact-equivalence test requires unchanged evaluation weights")

    mask_checks = 0
    for pattern in range(256):
        for edge_index, edge in enumerate(EDGES):
            discs = sum(1 << square for i, square in enumerate(edge)
                        if pattern & (1 << i))
            expected = stable_oracle(discs)
            for engine in (reference, candidate):
                actual = int(engine.stable_edge_mask(np.uint64(discs)))
                assert actual == expected, ("edge pattern", edge_index, pattern,
                                            hex(actual), hex(expected))
            mask_checks += 1

    rng = random.Random(514081)
    for i in range(20_000):
        discs = rng.getrandbits(64)
        # Explicit density extremes supplement unbiased full-width random masks.
        if i % 4 == 0:
            discs &= rng.getrandbits(64)
        elif i % 4 == 1:
            discs |= rng.getrandbits(64)
        expected = stable_oracle(discs)
        for engine in (reference, candidate):
            assert int(engine.stable_edge_mask(np.uint64(discs))) == expected, (
                "random mask", i, hex(discs))
        mask_checks += 1

    # Random weights exercise every feature, even when the production phase row
    # intentionally has zero coefficients. Both supported row layouts are used.
    noise = np.random.default_rng(514081)
    weights65 = noise.integers(-100, 101, size=(65, 11), dtype=np.int64)
    weights3 = noise.integers(-100, 101, size=(3, 11), dtype=np.int64)
    weights65[:, 6] = 17
    weights3[:, 6] = -11
    tables = (reference.DEFAULT_W, weights65, weights3)
    # Forced pass: me has b1, opponent owns all except a1 and b1. Only the
    # opponent can play a1. Also include early terminal and full-board states.
    special = [(np.uint64(2), np.uint64(FULL ^ 3)),
               (np.uint64(FULL ^ 3), np.uint64(2)),
               (np.uint64(0), np.uint64(FULL ^ 1)),
               (np.uint64(0x5555555555555555), np.uint64(0xAAAAAAAAAAAAAAAA))]
    assert reference.legal_moves(*special[0]) == 0
    assert reference.legal_moves(*special[1]) != 0
    eval_checks = 0
    for i in range(2_048 + len(special)):
        me, opp = (random_disjoint(rng, i % 65) if i < 2_048
                   else special[i - 2_048])
        moves = np.uint64(reference.legal_moves(me, opp))
        assert moves == candidate.legal_moves(me, opp)
        for weights in tables:
            expected = int(reference.evaluate(me, opp, moves, weights))
            actual = int(candidate.evaluate(me, opp, moves, weights))
            assert actual == expected, ("evaluation", i, weights.shape,
                                        hex(int(me)), hex(int(opp)), actual, expected)
            eval_checks += 1

    fixtures = [reachable_position(rng, plies, reference)
                for plies in (12, 28, 44, 53)] + special
    search_checks = resumed = 0
    for fixture, (me, opp) in enumerate(fixtures):
        for weights in (reference.DEFAULT_W, weights3):
            for depth in (0, 1, 3):
                exact, _, _ = search(reference, me, opp, depth,
                                     -reference.INF, reference.INF, weights, False, 1 << 60)
                for alpha, beta in ((-reference.INF, reference.INF),
                                    (exact[0] - 1, exact[0])):
                    for use_tt in (False, True):
                        whole, whole_tt, _ = search(reference, me, opp, depth,
                                                    alpha, beta, weights, use_tt, 1 << 60)
                        for chunk in (1 << 60, 7, 1):
                            for engine in (reference, candidate):
                                actual, table, yields = search(engine, me, opp, depth,
                                                               alpha, beta, weights,
                                                               use_tt, chunk)
                                assert actual == whole, ("search value/nodes", fixture,
                                                          depth, weights.shape, use_tt,
                                                          alpha, beta, chunk, actual, whole)
                                assert np.array_equal(table, whole_tt), (
                                    "TT contents", fixture, depth, use_tt, chunk)
                                resumed += yields > 0
                                search_checks += 1
    assert resumed > 0, "No test search actually exercised resumable slices"
    print("OK: %d mask/oracle, %d full evaluation, %d score/node/cache checks "
          "(%d searches resumed)" % (mask_checks, eval_checks, search_checks, resumed))


if __name__ == "__main__":
    main()
