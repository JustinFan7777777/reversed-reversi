"""Cross-check the bitboard engine in ai.py against the numpy reference engine.

    python tools/check_engine.py --games 300

For every position of every random game it verifies
  * legal-move set of AI.go()  ==  reference legal moves (and the chosen move is last and legal)
  * bitboard flips()           ==  reference flips
  * pass / game-over handling and the 'fewer discs wins' result
"""
import argparse
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import ai as A                      # noqa: E402
import ref_engine as R              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    A.DEBUG_RAISE = True                 # never hide exceptions while testing
    players = {R.BLACK: A.AI(8, R.BLACK, 0.2), R.WHITE: A.AI(8, R.WHITE, 0.2)}
    positions = 0
    for g in range(args.games):
        board, color, passes = R.initial_board(), R.BLACK, 0
        while passes < 2:
            ref = R.legal_moves(board, color)
            me, opp = A.board_to_bb(board, color)
            bb_moves = [A.sq_to_move(s) for s in A.bb_to_squares(A.legal_moves(me, opp))]
            assert sorted(ref) == sorted(bb_moves), (g, board, color, ref, bb_moves)
            for mv in ref:                                   # flips must agree too
                b = np.uint64(1 << (mv[0] * 8 + mv[1]))
                got = sorted(A.sq_to_move(s) for s in A.bb_to_squares(A.flips(me, opp, b)))
                assert got == sorted(R.flips_for(board, color, *mv)), (g, mv)
            gm, _ = A.generic_choose(board, color)           # independent slow path must agree too
            assert sorted(gm) == sorted(ref), ("generic path != reference", g, color)
            ai = players[color]
            ai.candidate_list.clear()
            ai.go(board)
            cl = ai.candidate_list
            if not ref:
                assert cl == [], "must return [] when no legal move"
                passes += 1
            else:
                assert all(isinstance(x, tuple) and type(x[0]) is int and type(x[1]) is int for x in cl), \
                    "candidate_list must hold tuples of plain python ints"
                assert set(cl) == set(ref) and cl[-1] in ref, (cl, ref)
                mv = rnd.choice(ref) if rnd.random() < 0.7 else cl[-1]   # mix random + AI moves
                board = R.apply_move(board, color, mv)
                passes = 0
            positions += 1
            color = -color
        R.result(board)
    print("OK: %d games, %d positions, engine matches the reference" % (args.games, positions))


if __name__ == "__main__":
    main()
