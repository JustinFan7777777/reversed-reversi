"""Sliced search must match uninterrupted search across yields, passes and pruning."""
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import ai as A
from test_search import random_position


def main():
    rnd, checked = random.Random(81), 0
    A.DEBUG_RAISE = True
    for plies in (6, 24, 48, 55):
        for _ in range(4):
            board, color = random_position(rnd, plies)
            me, opp = A.board_to_bb(board, color)
            for depth in (1, 3, 5):
                su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
                si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
                whole = np.array([0, 1 << 60, 0], dtype=np.int64)
                expected = A.alphabeta(me, opp, depth, -A.INF, A.INF,
                                       whole, A.DEFAULT_W, su, si)
                for chunk in (1, 7, 512):
                    stats = np.zeros(7, dtype=np.int64)
                    while True:
                        stats[0], stats[1], stats[2] = 0, chunk, 0
                        got = A.alphabeta(me, opp, depth, -A.INF, A.INF,
                                         stats, A.DEFAULT_W, su, si)
                        if stats[2] != 2:
                            break
                    assert got == expected, (plies, depth, chunk, got, expected)
                    checked += 1
    # The endgame must run shallow iterations before attempting an exact solve.
    ai = A.AI(8, 1, 0.05)
    for _ in range(30):
        board, color = random_position(rnd, 52)
        me, opp = A.board_to_bb(board, color)
        moves = A.bb_to_squares(A.legal_moves(me, opp))
        if len(moves) > 1:
            chosen = ai._search(me, opp, moves, 0.01)
            assert chosen in moves and ai.last_depth >= 1
            break
    else:
        raise AssertionError("No playable endgame fixture")
    assert ai._budget(20) == 0 if ai.used >= A.GAME_BUDGET_SEC else True
    ai.used = A.GAME_BUDGET_SEC
    assert ai._budget(20) == 0
    print("OK: %d resumable-search comparisons; bounded endgame fallback" % checked)


if __name__ == "__main__":
    main()
