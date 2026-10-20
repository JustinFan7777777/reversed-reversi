"""Head-to-head matches between two submission files (colours swapped every game).

    python tools/arena.py ai.py other_ai.py --games 20 --time-out 1.0

Reports W/D/L from the point of view of the first file.  Fewer discs wins.
"""
import argparse
import importlib.util
import pathlib
import random
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import ref_engine as R              # noqa: E402


def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(mod_black, mod_white, time_out, per_move_limit=5.0, game_limit=180.0,
         opening_plies=0, seed=2026, strict_time_out=True, metrics=None):
    for m in (mod_black, mod_white):
        if hasattr(m, "DEBUG_RAISE"):
            m.DEBUG_RAISE = True
    ais = {R.BLACK: mod_black.AI(8, R.BLACK, time_out), R.WHITE: mod_white.AI(8, R.WHITE, time_out)}
    used = {R.BLACK: 0.0, R.WHITE: 0.0}
    board, color, passes = R.initial_board(), R.BLACK, 0
    rnd = random.Random(seed)
    for _ in range(opening_plies):
        legal = R.legal_moves(board, color)
        if not legal:
            color = -color
            legal = R.legal_moves(board, color)
            if not legal:
                break
        board = R.apply_move(board, color, rnd.choice(legal))
        color = -color
    while passes < 2:
        ai = ais[color]
        t = time.time()
        given = board.copy()
        ai.go(given)
        dt = time.time() - t
        used[color] += dt
        if metrics is not None:
            metrics["max_move_seconds"] = max(metrics.get("max_move_seconds", 0.0), dt)
            metrics["soft_limit_overruns"] = metrics.get("soft_limit_overruns", 0) + int(dt > time_out)
        legal = R.legal_moves(board, color)
        cl = list(ai.candidate_list)
        hard_limit = min(per_move_limit, time_out) if strict_time_out else per_move_limit
        if dt > hard_limit or sum(used.values()) > game_limit:
            return -color, "timeout", used            # opponent wins
        if not np.array_equal(board, given):
            return -color, "mutated input", used
        if getattr(ai, "fallback_count", 0) or getattr(ai, "warmup_failures", 0):
            raise AssertionError("unexpected fallback or JIT warmup failure")
        if not legal:
            if cl:
                return -color, "illegal pass", used
            passes += 1
        else:
            if (not cl or set(cl) != set(legal) or cl[-1] not in legal or
                    not all(type(m) is tuple and len(m) == 2 and
                            all(type(x) is int for x in m) for m in cl)):
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
    ap.add_argument("--opening-plies", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()
    A, B = load(args.a, "mod_a"), load(args.b, "mod_b")
    w = d = l = 0
    scores, records = [], []
    for g in range(args.games):
        a_is_black = (g % 2 == 0)
        black, white = (A, B) if a_is_black else (B, A)
        winner, why, used = play(black, white, args.time_out,
                                 opening_plies=args.opening_plies, seed=args.seed + g // 2)
        a_color = R.BLACK if a_is_black else R.WHITE
        if winner == 0:
            d += 1
        elif winner == a_color:
            w += 1
        else:
            l += 1
        scores.append(0.5 if winner == 0 else float(winner == a_color))
        records.append(dict(game=g, a_color=a_color, winner=winner, reason=why,
                            times=used, opening_seed=args.seed + g // 2))
        print("game %2d  A=%s  winner=%s (%s)  time(black/white)=%.1f/%.1f" % (
            g, "black" if a_is_black else "white", {0: "draw", -1: "black", 1: "white"}[winner],
            why, used[R.BLACK], used[R.WHITE]))
    print("A vs B:  W %d  D %d  L %d" % (w, d, l))
    ci = paired_interval(scores)
    print("score %.3f; paired-bootstrap 95%% CI [%.3f, %.3f]" % (np.mean(scores), *ci))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(a=args.a, b=args.b, games=args.games,
            time_out=args.time_out, opening_plies=args.opening_plies, seed=args.seed,
            wins=w, draws=d, losses=l, score=float(np.mean(scores)),
            paired_ci95=ci, records=records), indent=2) + "\n")


def paired_interval(scores):
    """Resample opening pairs, keeping the two colors together to avoid fake precision."""
    if len(scores) < 4 or len(scores) % 2:
        return [0.0, 1.0]
    pairs = np.asarray(scores).reshape(-1, 2).mean(axis=1)
    rng = np.random.default_rng(517)
    samples = rng.choice(pairs, size=(5000, len(pairs)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


if __name__ == "__main__":
    main()
