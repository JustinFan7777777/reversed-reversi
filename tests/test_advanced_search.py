"""PVS/TT bounds and resumable re-search must agree with exact depth-limited minimax."""
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import test_search as T


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "versions/v07_pvs_ai.py"
    spec = importlib.util.spec_from_file_location("unregistered_advanced", ROOT / path)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    A.DEBUG_RAISE = True
    player = A.AI(8, -1, 5)
    T.A, T.W = A, player.W
    rnd, checks = random.Random(8701), 0
    su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
    si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
    table = player.tt
    for ply in (12, 30, 48, 54):
        board, color = T.random_position(rnd, ply)
        me, opp = A.board_to_bb(board, color)
        for depth in (3, 5):
            expected = T.minimax(me, opp, depth)
            windows = [(-A.INF, A.INF), (-100, -50), (-5, 5),
                       (expected, expected+1), (expected-1, expected), (-A.INF, A.INF)]
            for alpha, beta in windows:
                stats = np.zeros(7, dtype=np.int64)
                while True:
                    stats[0], stats[1], stats[2] = 0, 3, 0
                    got = int(A.alphabeta(me, opp, depth, alpha, beta, stats,
                                         player.W, su, si, table))
                    if stats[2] != 2:
                        break
                if expected <= alpha:
                    assert got <= alpha, (ply, depth, expected, got, alpha, beta)
                elif expected >= beta:
                    assert got >= beta, (ply, depth, expected, got, alpha, beta)
                else:
                    assert got == expected, (ply, depth, expected, got, alpha, beta)
                checks += 1
    # Poison a starting-position child with a bogus proven win under old weights.
    # If weight changes fail to clear the table, depth two would accept this entry
    # and report a solved opening. A real opening cannot finish in two placements.
    board = T.R.initial_board()
    me, opp = A.board_to_bb(board, player.color)
    sq = A.bb_to_squares(A.legal_moves(me, opp))[0]
    move = np.uint64(1 << sq)
    flipped = np.uint64(A.flips(me, opp, move))
    child_me, child_opp = np.uint64(opp & ~flipped), np.uint64(me | move | flipped)
    table.fill(0)
    slot = int(A.tt_index(child_me, child_opp))
    poison = np.array([-A.WIN_BASE-64], dtype=np.int64).view(np.uint64)[0]
    table[slot] = [child_me, child_opp, 1, poison, 0, 1]
    player.W[0, 0] += 1
    player.time_out = 0.05
    player.go(board)
    assert np.array_equal(player._tt_weights, player.W)
    assert player.last_depth >= 2 and not player.last_solved
    assert player.fallback_count == player.warmup_failures == 0
    print("OK: %d PVS/TT window+resume checks and weight-cache invalidation" % checks)


if __name__ == "__main__":
    main()
