"""alpha-beta (numba, iterative) must return EXACTLY the same value as a plain minimax
without pruning.  Covers pruning, passes and terminal scoring (late-game positions).

    python tests/test_search.py [path/to/ai.py]
"""
import importlib.util
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
import ai as A                      # noqa: E402
import ref_engine as R              # noqa: E402
A.DEBUG_RAISE = True

U = np.uint64
W = A.DEFAULT_W


def minimax(me, opp, depth):
    """slow reference: plain negamax, no pruning, no ordering, python recursion."""
    moves = int(A.legal_moves(me, opp))
    if moves == 0:
        if int(A.legal_moves(opp, me)) == 0:
            return int(A.terminal_value(me, opp))
        return -minimax(opp, me, depth)
    if depth <= 0:
        return int(A.evaluate(me, opp, U(moves), W))
    best = -A.INF
    for sq in range(64):
        if (moves >> sq) & 1:
            b = U(1 << sq)
            f = U(A.flips(me, opp, b))
            best = max(best, -minimax(opp & ~f, me | b | f, depth - 1))
    return best


def random_position(rnd, plies):
    board, color = R.initial_board(), R.BLACK
    for _ in range(plies):
        mv = R.legal_moves(board, color)
        if not mv:
            color = -color
            mv = R.legal_moves(board, color)
            if not mv:
                break
        board = R.apply_move(board, color, rnd.choice(mv))
        color = -color
    return board, color


def main(path=None):
    global A, W
    if path is not None:
        spec = importlib.util.spec_from_file_location("minimax_test_candidate", path)
        A = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(A)
        A.DEBUG_RAISE = True
        W = A.DEFAULT_W
        print("source:", pathlib.Path(path).resolve())
    rnd = random.Random(7)
    su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
    si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
    checked = 0
    for plies, depths in ((8, (1, 2, 3, 4)), (20, (1, 2, 3)), (44, (1, 2, 3, 4)), (52, (2, 4, 6, 8, 12))):
        for _ in range(12):
            board, color = random_position(rnd, plies)
            me, opp = A.board_to_bb(board, color)
            for d in depths:
                stats = np.array([0, 1 << 60, 0], dtype=np.int64)
                got = A.alphabeta(me, opp, d, -A.INF, A.INF, stats, W, su, si)
                want = minimax(me, opp, d)
                assert got == want, ("alphabeta != minimax", plies, d, got, want)
                checked += 1
    print("OK: alphabeta == minimax on %d (position, depth) pairs" % checked)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: test_search.py [path/to/ai.py]")
    main(sys.argv[1] if len(sys.argv) == 2 else None)
