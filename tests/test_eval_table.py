"""Per-empties weight table: layout, legacy-table equivalence and row isolation.

    python tests/test_eval_table.py [path/to/ai.py]
"""
import importlib.util
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import ai as A
from test_search import random_position

V13_W = np.array([[70, -2, 2, 2, 2, 5, 0, -2, 0, 0, 0],
                  [70, -2, 2, 2, 2, 7, 0, -2, 0, 0, 0],
                  [70, -2, 2, 0, 10, 2, 0, -2, 0, 0, 0]], dtype=np.int64)


def expand(rows3):
    """65-row table holding the legacy phase rows (end <= 16, middle 17-44, opening > 44)."""
    return np.array([rows3[2] if e <= 16 else rows3[1] if e <= 44 else rows3[0]
                     for e in range(65)], dtype=np.int64)


def search(me, opp, depth, W):
    su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
    si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
    tt = np.zeros((A.TT_SIZE, 6), dtype=np.uint64)
    stats = np.array([0, 1 << 60, 0], dtype=np.int64)
    value = A.alphabeta(me, opp, depth, -A.INF, A.INF, stats, W, su, si, tt)
    return int(value), int(stats[0])


def main(path=None):
    global A
    if path is not None:
        spec = importlib.util.spec_from_file_location("table_test_candidate", path)
        A = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(A)
        print("source:", pathlib.Path(path).resolve())
    A.DEBUG_RAISE = True
    W = A.DEFAULT_W
    # Layout: one C-contiguous int64 row per empties count 0..64.
    assert W.shape == (65, 11) and W.dtype == np.int64 and W.flags["C_CONTIGUOUS"]
    expected = np.vstack((A.END_W[:1], A.END_W, np.repeat(A.PHASE_W[1:2], 28, axis=0),
                          np.repeat(A.PHASE_W[:1], 20, axis=0)))
    if hasattr(A, "MID_W"):
        assert A.MID_W.ndim == 2 and A.MID_W.shape[1] == 11
        assert 1 <= len(A.MID_W) <= 28
        expected[17:17 + len(A.MID_W)] = A.MID_W
    if hasattr(A, "PATTERN_ROWS"):           # Round 8: rows 10-19 carry pattern-fitted weights
        assert A.PATTERN_ROWS.shape == (10, 11)
        expected[10:20] = A.PATTERN_ROWS
    if hasattr(A, "PATTERN_ROWS_HI"):        # Round 10: rows 20-22 likewise
        assert A.PATTERN_ROWS_HI.shape == (3, 11)
        expected[20:23] = A.PATTERN_ROWS_HI
    assert np.array_equal(W, expected)
    legacy = expand(V13_W)
    patterns = hasattr(A, "PAT_EDGE")
    pattern_rows = set(range(10, 20)) | (set(range(20, 23)) if hasattr(A, "PATTERN_ROWS_HI") else set())
    rng, noise, checked = random.Random(4417), np.random.default_rng(4417), 0
    for plies in (8, 24, 36, 44, 48, 52, 56):
        for _ in range(4):
            board, color = random_position(rng, plies)
            me, opp = A.board_to_bb(board, color)
            moves = np.uint64(A.legal_moves(me, opp))
            # The per-empties lookup of expanded phase rows equals the 3-row phase lookup;
            # Round 8/10 engines add their shape-pattern term (rows 10-19 and 20-22, 65-row tables only),
            # which an all-zero 65-row table isolates.
            pattern = A.evaluate(me, opp, moves, np.zeros((65, 11), np.int64)) if patterns else 0
            assert A.evaluate(me, opp, moves, legacy) == A.evaluate(me, opp, moves, V13_W) + pattern
            empties = 64 - bin(int(me | opp)).count("1")
            for depth in (1, 2, 3, 4):
                if not (patterns and empties - depth in pattern_rows):
                    assert search(me, opp, depth, legacy) == search(me, opp, depth, V13_W)
                # Row isolation: every leaf of a search has empties - depth empties (passes
                # are free), so randomising all other rows changes neither value nor tree.
                noisy = noise.integers(-50, 50, size=W.shape).astype(np.int64)
                leaf = max(empties - depth, 0)
                noisy[leaf] = W[leaf]
                assert search(me, opp, depth, noisy) == search(me, opp, depth, W), (plies, depth)
                checked += 1
    print("OK: per-empties table layout; %d legacy-equivalence and row-isolation checks" % checked)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: test_eval_table.py [path/to/ai.py]")
    main(sys.argv[1] if len(sys.argv) == 2 else None)
