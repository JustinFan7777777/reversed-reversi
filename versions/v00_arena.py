"""Head-to-head matches between two submission files (colours swapped every game).

    python tools/arena.py ai.py other_ai.py --games 20 --time-out 1.0

Reports W/D/L from the point of view of the first file.  Fewer discs wins.
"""
import argparse
import importlib.util
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import ref_engine as R              # noqa: E402


def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(mod_black, mod_white, time_out, per_move_limit=5.0, game_limit=180.0):
    for m in (mod_black, mod_white):
        if hasattr(m, "DEBUG_RAISE"):
            m.DEBUG_RAISE = True
    ais = {R.BLACK: mod_black.AI(8, R.BLACK, time_out), R.WHITE: mod_white.AI(8, R.WHITE, time_out)}
    used = {R.BLACK: 0.0, R.WHITE: 0.0}
    board, color, passes = R.initial_board(), R.BLACK, 0
    while passes < 2:
        ai = ais[color]
        t = time.time()
        ai.go(board.copy())
        dt = time.time() - t
        used[color] += dt
        legal = R.legal_moves(board, color)
        cl = list(ai.candidate_list)
        if dt > per_move_limit or used[color] > game_limit:
            return -color, "timeout", used            # opponent wins
        if not legal:
            assert cl == [], "illegal: returned moves although none are legal"
            passes += 1
        else:
            if not cl or tuple(cl[-1]) not in legal:
                return -color, "illegal move", used
            board = R.apply_move(board, color, tuple(cl[-1]))
            passes = 0
        color = -color
    return R.result(board)[2], "normal", used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--time-out", type=float, default=1.0)
    args = ap.parse_args()
    A, B = load(args.a, "mod_a"), load(args.b, "mod_b")
    w = d = l = 0
    for g in range(args.games):
        a_is_black = (g % 2 == 0)
        winner, why, used = play(A, B, args.time_out) if a_is_black else play(B, A, args.time_out)
        a_color = R.BLACK if a_is_black else R.WHITE
        if winner == 0:
            d += 1
        elif winner == a_color:
            w += 1
        else:
            l += 1
        print("game %2d  A=%s  winner=%s (%s)  time(black/white)=%.1f/%.1f" % (
            g, "black" if a_is_black else "white", {0: "draw", -1: "black", 1: "white"}[winner],
            why, used[R.BLACK], used[R.WHITE]))
    print("A vs B:  W %d  D %d  L %d" % (w, d, l))


if __name__ == "__main__":
    main()
