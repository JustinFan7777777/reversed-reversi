"""Round 8 data: exact final margins of self-play positions at 10-19 empties.

    .venv/bin/python -I tools/build_exact_rows.py --seed-base 18000000 --games 3000 \
        --worker J --workers K --out artifacts/claude_round8/rows_wJ.json

Protocol: versions/round8_protocol.md. Self-play with the frozen v14 snapshot: v14 search at
10 ms per move, eight random opening plies, then a random move with probability 0.1
(seeded per game: seed-base + game id). For every empties count 10-19 (up to --max-empties), the first position
in the game whose side to move has a legal move is kept (from --min-empties up) with its exact final margin
(full-window solve; margin = opponent's discs - own discs at the end, > 0 wins).
With --unlabelled-max N, positions with max-empties+1..N empties are kept with margin None.
Split by game: game id % 5 == 0 is test. The output is rewritten after every game.
"""
import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("exact_rows_v14", ROOT / "versions/v14_round4_fitted_end_ai.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
A.DEBUG_RAISE = True
U = np.uint64
START = ((1 << 35) | (1 << 28), (1 << 27) | (1 << 36))     # black to move, standard setup


def bits(x):
    x = int(x)
    return [i for i in range(64) if (x >> i) & 1]


def child(me, opp, sq):
    b = U(1 << sq)
    f = U(A.flips(U(me), U(opp), b))
    return int(U(opp) & ~f), int(U(me) | b | f)


class Solver:
    def __init__(self):
        self.su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
        self.si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
        self.tt = np.zeros((A.TT_SIZE, 6), dtype=np.uint64)

    def margin(self, me, opp):
        empties = 64 - bin(me | opp).count("1")
        stats = np.array([0, 1 << 60, 0], dtype=np.int64)
        v = int(A.alphabeta(U(me), U(opp), empties, -A.INF, A.INF, stats, A.DEFAULT_W,
                            self.su, self.si, self.tt))
        assert not stats[2] and (v == 0 or abs(v) > A.WIN_BASE)
        return v - A.WIN_BASE if v > 0 else v + A.WIN_BASE if v < 0 else 0, int(stats[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-empties", type=int, default=10)
    ap.add_argument("--max-empties", type=int, default=19)
    ap.add_argument("--unlabelled-max", type=int, default=0)    # also keep 20..N without labels
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    players = {c: A.AI(8, c, 5) for c in (-1, 1)}
    solver, out, t0 = Solver(), [], time.perf_counter()
    for gid in range(args.worker, args.games, args.workers):
        rnd = random.Random(args.seed_base + gid)
        split = "test" if gid % 5 == 0 else "train"
        for p in players.values():
            p.tt.fill(0)
        (me, opp), colour, passes, ply, seen = START, -1, 0, 0, set()
        while passes < 2:
            moves = bits(A.legal_moves(U(me), U(opp)))
            if not moves:
                passes += 1
                me, opp, colour = opp, me, -colour
                continue
            passes = 0
            empties = 64 - bin(me | opp).count("1")
            if empties < args.min_empties:
                break
            if args.max_empties < empties <= args.unlabelled_max and empties not in seen:
                seen.add(empties)
                out.append(dict(game=gid, split=split, empties=empties, colour=colour,
                                me=me, opp=opp, margin=None, nodes=0))
            if empties <= args.max_empties and empties not in seen:
                seen.add(empties)
                m, n = solver.margin(me, opp)
                out.append(dict(game=gid, split=split, empties=empties, colour=colour,
                                me=me, opp=opp, margin=m, nodes=n))
            if ply < 8 or rnd.random() < 0.1:
                sq = rnd.choice(moves)
            else:
                sq = players[colour]._search(U(me), U(opp), moves, 0.01)
            me, opp = child(me, opp, sq)
            colour = -colour
            ply += 1
        args.out.write_text(json.dumps(out))
    args.out.write_text(json.dumps(out))
    print(f"worker {args.worker}: {len(out)} positions, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
