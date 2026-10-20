"""Platform contract checks against an independent array engine; no pytest required."""
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ai as A
import ref_engine as R


def fixtures():
    yield "opening", R.initial_board()
    yield "empty", np.zeros((8, 8), dtype=np.int64)
    yield "full", np.ones((8, 8), dtype=np.int64)
    one = np.ones((8, 8), dtype=np.int64)
    one[0, 0], one[0, 1] = 0, -1
    assert R.legal_moves(one, 1) == [(0, 0)]
    assert not R.legal_moves(one, -1)  # Forced pass, opponent can still play.
    yield "one_move_and_pass", one
    early = np.zeros((8, 8), dtype=np.int64)
    early[3, 3] = 1
    yield "early_terminal", early
    rays = np.zeros((8, 8), dtype=np.int64)
    for dr, dc in R.DIRS:
        rays[3 + dr, 3 + dc] = -1
        rays[3 + 2 * dr, 3 + 2 * dc] = 1
    assert len(R.flips_for(rays, 1, 3, 3)) == 8
    yield "eight_directions", rays
    wrap = np.zeros((8, 8), dtype=np.int64)
    wrap[0, 7], wrap[1, 0] = 1, -1
    assert not R.legal_moves(wrap, 1)
    yield "no_row_wrapping", wrap
    rnd = random.Random(2026)
    for size in (6, 8, 10):
        board, color = R.initial_board(size), -1
        yield "size_%d_opening" % size, board.copy()
        for ply in range(size * size - 4):
            moves = R.legal_moves(board, color)
            if not moves:
                color = -color
                moves = R.legal_moves(board, color)
                if not moves:
                    break
            board = R.apply_move(board, color, rnd.choice(moves))
            color = -color
            if ply % 9 == 0 or np.count_nonzero(board == 0) <= 2:
                yield "size_%d_ply_%d" % (size, ply), board.copy()


def main():
    A.DEBUG_RAISE = True
    players = {(size, color): A.AI(size, color, 0.005)
               for size in (6, 8, 10) for color in (-1, 1)}
    checked = 0
    for label, board in fixtures():
        wide = np.zeros((board.shape[0], board.shape[1] * 2), dtype=np.int64)
        wide[:, ::2] = board
        variants = (board, board.astype(np.int32), board.astype(float),
                    wide[:, ::2], board.tolist(), board[::-1, ::-1])
        for variant in variants:
            before = np.asarray(variant).copy()
            for color in (-1, 1):
                ai = players[(len(board), color)]
                ai.candidate_list[:] = [(999, 999)]  # Verify clearing, including pass.
                ai.go(variant)
                legal, got = set(R.legal_moves(before, color)), ai.candidate_list
                assert set(got) == legal, (label, color, got, legal)
                assert all(type(m) is tuple and len(m) == 2 and
                           all(type(x) is int for x in m) for m in got)
                assert (got[-1] in legal) if legal else got == []
                assert np.array_equal(np.asarray(variant), before), label
                assert getattr(ai, "fallback_count", 0) == 0
                checked += 1
    print("OK: %d interface cases, array/list/dtype/view/size/pass checks" % checked)


if __name__ == "__main__":
    main()
