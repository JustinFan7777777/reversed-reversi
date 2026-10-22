"""Round 8 pattern evaluation (v18 candidate) against an independent reference.

  * Rows 10-19: on 10,000 random positions with 10-19 empties, the engine's evaluate()
    equals a pure-Python reference: v14's 11 features times the candidate's row, plus the
    edge+2X and 3x3-corner entries read square by square and folded by mirror image,
    looked up in tables parsed directly from the embedded text.
  * Every other row, and the legacy 3-row tables, give exactly v14's values (2,000
    positions over all empties counts); fixed-depth search values match unpruned minimax
    with the candidate's evaluation (40 positions, depths 1-3).

    python tests/test_patterns.py [path/to/candidate.py]
"""
import importlib.util
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import test_search as T                     # noqa: E402
from fit_patterns import CORNER9, EDGE2X, TRANSPOSE9  # noqa: E402


def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DEBUG_RAISE = True
    return mod


def parse(text):
    out = {}
    vals = [int(x) for x in text.split()]
    for i in range(0, len(vals), 2):
        out[vals[i]] = vals[i + 1]
    return out


def index(me, opp, squares):
    return sum((1 if (me >> s) & 1 else 2 if (opp >> s) & 1 else 0) * 3 ** k for k, s in enumerate(squares))


def reference(X, V, me, opp, edge, corner):
    """Candidate value from first principles (Python ints)."""
    me, opp = int(me), int(opp)
    empties = 64 - bin(me | opp).count("1")
    mv = np.uint64(V.legal_moves(np.uint64(me), np.uint64(opp)))
    value = 0
    for k in range(11):
        unit = np.zeros((3, 11), np.int64)
        unit[:, k] = 1
        value += int(X.DEFAULT_W[empties, k]) * int(V.evaluate(np.uint64(me), np.uint64(opp), mv, unit))
    b = 0 if empties <= 13 else 1 if empties <= 16 else 2
    for inst in EDGE2X:
        a, m = index(me, opp, inst), index(me, opp, inst[::-1])
        value += edge[b].get(min(a, m), 0)
    for inst in CORNER9:
        a, m = index(me, opp, inst), index(me, opp, [inst[j] for j in TRANSPOSE9])
        value += corner[b].get(min(a, m), 0)
    return value


def position(rnd, empties):
    while True:
        board, color = T.random_position(rnd, 60 - empties)
        if int((board == 0).sum()) == empties:
            return board, color


def main():
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "versions/v18_patterns_candidate_ai.py")
    print("source:", path.resolve())
    X = load(path, "t8_candidate")
    V = load(ROOT / "versions/v14_round4_fitted_end_ai.py", "t8_v14")
    edge = [parse(t) for t in X._EDGE_TEXT]
    corner = [parse(t) for t in X._CORNER_TEXT]
    rnd = random.Random(8_200_001)
    checked = 0
    for n in range(10_000):
        board, color = position(rnd, 10 + n % 10)
        me, opp = X.board_to_bb(board, color)
        mv = np.uint64(X.legal_moves(me, opp))
        got = int(X.evaluate(me, opp, mv, X.DEFAULT_W))
        assert got == reference(X, V, me, opp, edge, corner), (n, got)
        checked += 1
    same = 0
    legacy = np.array([[3, -1, 2, 0, 1, 2, 1, -1, 1, 0, 1]] * 3, dtype=np.int64)
    for n in range(2_000):
        e = rnd.choice([e for e in range(1, 61) if not 10 <= e <= 19])
        board, color = position(rnd, e)
        me, opp = X.board_to_bb(board, color)
        mv = np.uint64(X.legal_moves(me, opp))
        assert int(X.evaluate(me, opp, mv, X.DEFAULT_W)) == int(V.evaluate(me, opp, mv, V.DEFAULT_W)), e
        assert int(X.evaluate(me, opp, mv, legacy)) == int(V.evaluate(me, opp, mv, legacy)), e
        same += 1
    assert np.array_equal(np.delete(X.DEFAULT_W, range(10, 20), 0), np.delete(V.DEFAULT_W, range(10, 20), 0))

    def minimax(me, opp, depth, passed=False):
        moves = X.bb_to_squares(X.legal_moves(me, opp))
        if not moves:
            if passed or not X.legal_moves(opp, me):
                return int(X.terminal_value(me, opp))
            return -minimax(opp, me, depth, True)
        if depth == 0:
            return int(X.evaluate(me, opp, np.uint64(X.legal_moves(me, opp)), X.DEFAULT_W))
        best = -X.INF
        for sq in moves:
            b = np.uint64(1 << sq)
            f = np.uint64(X.flips(me, opp, b))
            best = max(best, -minimax(np.uint64(opp & ~f), np.uint64(me | b | f), depth - 1))
        return best

    su = np.zeros((X.MAXPLY, X.SU_COLS), dtype=np.uint64)
    si = np.zeros((X.MAXPLY, X.SI_COLS), dtype=np.int64)
    searched = 0
    for n in range(40):
        board, color = position(rnd, 12 + n % 10)
        me, opp = X.board_to_bb(board, color)
        for depth in (1, 2, 3):
            stats = np.array([0, 1 << 60, 0], dtype=np.int64)
            got = int(X.alphabeta(me, opp, depth, -X.INF, X.INF, stats, X.DEFAULT_W, su, si))
            assert got == minimax(me, opp, depth), (n, depth)
            searched += 1
    print("OK: %d pattern-row evaluations equal the reference, %d other-row/legacy positions "
          "equal v14, %d search values equal minimax" % (checked, same, searched))


if __name__ == "__main__":
    main()
