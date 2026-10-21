"""Per-empties weight table: layout, legacy-table equivalence and row isolation.

    python tests/test_eval_table.py
"""
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


def main():
    A.DEBUG_RAISE = True
    W = A.DEFAULT_W
    # Layout: one C-contiguous int64 row per empties count 0..64.
    assert W.shape == (65, 11) and W.dtype == np.int64 and W.flags["C_CONTIGUOUS"]
    assert (W[0] == W[1]).all() and (W[1:17] == A.END_W).all()
    assert (W[17:45] == A.PHASE_W[1]).all() and (W[45:] == A.PHASE_W[0]).all()
    legacy = expand(V13_W)
    rng, noise, checked = random.Random(4417), np.random.default_rng(4417), 0
    for plies in (8, 24, 36, 44, 48, 52, 56):
        for _ in range(4):
            board, color = random_position(rng, plies)
            me, opp = A.board_to_bb(board, color)
            moves = np.uint64(A.legal_moves(me, opp))
            # The per-empties lookup of expanded phase rows equals the 3-row phase lookup.
            assert A.evaluate(me, opp, moves, legacy) == A.evaluate(me, opp, moves, V13_W)
            empties = 64 - bin(int(me | opp)).count("1")
            for depth in (1, 2, 3, 4):
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
    main()
