"""Explicit pass-at-horizon and early-terminal score checks."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import ai as A
from test_search import minimax


def main():
    A.DEBUG_RAISE = True
    board = np.ones((8, 8), dtype=np.int64)
    board[0, 0], board[0, 1] = 0, -1
    su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
    si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
    checked = 0
    for color in (-1, 1):
        me, opp = A.board_to_bb(board, color)
        for depth in (0, 1, 2):
            stats = np.array([0, 1 << 60, 0], dtype=np.int64)
            got = A.alphabeta(me, opp, depth, -A.INF, A.INF, stats,
                             A.DEFAULT_W, su, si)
            assert got == minimax(me, opp, depth)
            checked += 1
    for me, opp, expected in ((0, 1, A.WIN_BASE + 1),
                              (1, 0, -A.WIN_BASE - 1), (1, 2, 0)):
        stats = np.array([0, 1 << 60, 0], dtype=np.int64)
        got = A.alphabeta(np.uint64(me), np.uint64(opp), 0, -A.INF, A.INF,
                         stats, A.DEFAULT_W, su, si)
        # For (1, 2), there is a legal move, so test terminal scoring directly.
        assert int(A.terminal_value(np.uint64(me), np.uint64(opp))) == expected
        if me == 0 or opp == 0:
            assert got == expected
        checked += 1
    print("OK: %d explicit pass/horizon/terminal checks" % checked)


if __name__ == "__main__":
    main()
