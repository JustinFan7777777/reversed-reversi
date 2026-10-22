"""Round 10 pattern evaluation (v20 candidate) against an independent reference.

  * Rows 20-22: on 6,000 random positions with 20-22 empties, the engine's evaluate()
    equals a pure-Python reference: v14's 11 features times the candidate's row, plus every
    shape's entries read square by square (instances and mirror maps from
    tools/fit_patterns9.SHAPES) and looked up in tables parsed from the embedded text.
  * Rows 10-19 give exactly v18's values, every other row and the legacy 3-row tables
    exactly v14's (3,000 positions); fixed-depth search values match unpruned minimax with
    the candidate's evaluation (40 positions at 21-25 empties, depths 1-3).

  * With --frozen FIT.json (the file Stage A scored): the engine's rows equal the model's,
    every entry of every table equals the model's value at the fitter's folded index (all
    3^digits indices), and on 900 random positions at 20-22 empties evaluate() equals the
    fitter's own predictor (v14 unit features times the rows, plus tables at
    tools/fit_patterns9.shape_index). This ties the built engine to the scored model.

    python tests/test_patterns_hi.py [path/to/candidate.py] [--frozen FIT.json]
"""
import argparse
import json
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
from fit_patterns9 import SHAPES, shape_index            # noqa: E402
from test_patterns import index, load, parse, position   # noqa: E402


def reference(X, V, me, opp, tables):
    me, opp = int(me), int(opp)
    empties = 64 - bin(me | opp).count("1")
    mv = np.uint64(V.legal_moves(np.uint64(me), np.uint64(opp)))
    value = 0
    for k in range(11):
        unit = np.zeros((3, 11), np.int64)
        unit[:, k] = 1
        value += int(X.DEFAULT_W[empties, k]) * int(V.evaluate(np.uint64(me), np.uint64(opp), mv, unit))
    for name, table in tables.items():
        inst, mirror = SHAPES[name]
        for s in inst:
            a = index(me, opp, s)
            if mirror is not None:
                a = min(a, index(me, opp, [s[j] for j in mirror]))
            value += table.get(a, 0)
    return value


def check_frozen(X, V, frozen, rnd):
    """Engine == the frozen model: rows, every table entry, and values on random positions."""
    model = json.loads(pathlib.Path(frozen).read_text())["model"]
    rows = np.array([model["rows"][str(e)] for e in range(20, 23)], dtype=np.int64)
    assert np.array_equal(X.PATTERN_ROWS_HI, rows) and np.array_equal(X.DEFAULT_W[20:23], rows)
    engine_shapes = sorted(n for n in SHAPES if hasattr(X, "HI_PAT_" + n.upper()))
    assert engine_shapes == sorted(model["shapes"]), (engine_shapes, model["shapes"])
    entries = 0
    for name in model["shapes"]:
        inst, mirror = SHAPES[name]
        digits = len(inst[0])
        t = np.array(model["tables"][name], dtype=np.int64)
        i = np.arange(3 ** digits)
        if mirror is None:
            fold = i
        else:
            dig = [(i // 3 ** k) % 3 for k in range(digits)]
            fold = np.minimum(i, sum(dig[mirror[k]] * 3 ** k for k in range(digits)))
        eng = getattr(X, "HI_PAT_" + name.upper())
        assert eng.dtype == np.int8 and np.array_equal(eng.astype(np.int64), t[fold]), name
        entries += len(i)
    unit = []
    for k in range(11):
        w = np.zeros((3, 11), np.int64)
        w[:, k] = 1
        unit.append(w)
    values = 0
    for n in range(900):
        board, color = position(rnd, 20 + n % 3)
        me, opp = X.board_to_bb(board, color)
        mv = np.uint64(V.legal_moves(me, opp))
        e = 64 - bin(int(me | opp)).count("1")
        pred = sum(int(rows[e - 20, k]) * int(V.evaluate(me, opp, mv, unit[k])) for k in range(11))
        a, b = np.array([me], dtype=np.uint64), np.array([opp], dtype=np.uint64)
        for name in model["shapes"]:
            pred += int(np.array(model["tables"][name])[shape_index(a, b, name)].sum())
        assert int(X.evaluate(me, opp, np.uint64(X.legal_moves(me, opp)), X.DEFAULT_W)) == pred, n
        values += 1
    return entries, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="versions/v20_patterns_candidate_ai.py")
    ap.add_argument("--frozen")
    args = ap.parse_args()
    path = ROOT / args.path
    print("source:", path.resolve())
    X = load(path, "t10_candidate")
    V = load(ROOT / "versions/v14_round4_fitted_end_ai.py", "t10_v14")
    E = load(ROOT / "versions/v18_patterns_candidate_ai.py", "t10_v18")
    tables = {name: parse(getattr(X, "_HI_TEXT_" + name.upper())) for name in SHAPES
              if hasattr(X, "_HI_TEXT_" + name.upper())}
    assert hasattr(X, "PATTERN_ROWS_HI"), "not a Round 10 engine"   # the linear-only fallback has no tables
    rnd = random.Random(10_200_001)
    checked = 0
    for n in range(6_000):
        board, color = position(rnd, 20 + n % 3)
        me, opp = X.board_to_bb(board, color)
        mv = np.uint64(X.legal_moves(me, opp))
        got = int(X.evaluate(me, opp, mv, X.DEFAULT_W))
        assert got == reference(X, V, me, opp, tables), (n, got)
        checked += 1
    same = 0
    legacy = np.array([[3, -1, 2, 0, 1, 2, 1, -1, 1, 0, 1]] * 3, dtype=np.int64)
    for n in range(3_000):
        e = rnd.choice([e for e in range(1, 61) if not 20 <= e <= 22])
        board, color = position(rnd, e)
        me, opp = X.board_to_bb(board, color)
        mv = np.uint64(X.legal_moves(me, opp))
        assert int(X.evaluate(me, opp, mv, X.DEFAULT_W)) == int(E.evaluate(me, opp, mv, E.DEFAULT_W)), e
        assert int(X.evaluate(me, opp, mv, legacy)) == int(V.evaluate(me, opp, mv, legacy)), e
        same += 1
    assert np.array_equal(np.delete(X.DEFAULT_W, range(20, 23), 0), np.delete(E.DEFAULT_W, range(20, 23), 0))

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
        board, color = position(rnd, 21 + n % 5)
        me, opp = X.board_to_bb(board, color)
        for depth in (1, 2, 3):
            stats = np.array([0, 1 << 60, 0], dtype=np.int64)
            got = int(X.alphabeta(me, opp, depth, -X.INF, X.INF, stats, X.DEFAULT_W, su, si))
            assert got == minimax(me, opp, depth), (n, depth)
            searched += 1
    print("OK: %d rows 20-22 evaluations equal the reference, %d other-row/legacy positions "
          "equal v18/v14, %d search values equal minimax" % (checked, same, searched))
    if args.frozen:
        entries, values = check_frozen(X, V, args.frozen, rnd)
        print("OK: engine == frozen model %s (rows, %d table entries, %d values)" % (args.frozen, entries, values))


if __name__ == "__main__":
    main()
