"""Round 7 positions with the exact outcome of every legal move (versions/round7_protocol.md).

    .venv/bin/python -I tools/build_outcome_set.py --split train --seed-base 13000000 \
        --games 200 --worker 0 --workers 6 --out artifacts/claude_round7/train_w0.json [--probes]

Self-play with the frozen v14 snapshot: v14 search at 10 ms per move, eight random opening
plies, then a random move with probability 0.1 (seeded per game: seed-base + game id). One
position per game: a seeded target of 20-23 empties, then the qualifying position (side to
move has at least two legal moves) nearest to it, earlier on ties. Labels: an outcome-only
solve of every legal child (window -1..1 to the end of the game, one table per position,
at most --cap nodes per child). A position with any unfinished child keeps `outcomes: null`.
With --probes (training only), one extra unlabelled position per empties count 15-24 is kept
for measuring solve costs. With --no-labels (protocol revision 1) positions are kept
without solving any move. Outcomes are from the side to move: 1 win, 0 draw, -1 loss.
The output is rewritten after every game, so an interrupted run keeps finished games.
"""
import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("outcome_set_v14", ROOT / "versions/v14_round4_fitted_end_ai.py")
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
    def __init__(self, cap):
        self.cap = cap
        self.su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
        self.si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
        self.tt = np.zeros((A.TT_SIZE, 6), dtype=np.uint64)

    def outcome(self, me, opp):
        """(outcome for the side to move, nodes) or (None, nodes) past the node cap."""
        empties = 64 - bin(me | opp).count("1")
        stats = np.zeros(7, dtype=np.int64)
        used = 0
        while True:
            stats[0], stats[1], stats[2] = 0, 1 << 20, 0
            v = int(A.alphabeta(U(me), U(opp), empties, -1, 1, stats, A.DEFAULT_W,
                                self.su, self.si, self.tt))
            used += int(stats[0])
            if stats[2] != 2:
                return (1 if v > 0 else -1 if v < 0 else 0), used
            if used >= self.cap:
                return None, used


def play(gid, seed, probes, players):
    """Self-play one game; returns (target, candidates, probe positions)."""
    rnd = random.Random(seed)
    target = rnd.choice([20, 21, 22, 23])
    for p in players.values():
        p.tt.fill(0)
    (me, opp), colour, passes, ply = START, -1, 0, 0
    cands, probe = [], {}
    floor = 15 if probes else 20
    while passes < 2:
        moves = bits(A.legal_moves(U(me), U(opp)))
        if not moves:
            passes += 1
            me, opp, colour = opp, me, -colour
            continue
        passes = 0
        empties = 64 - bin(me | opp).count("1")
        if empties < floor:
            break
        if len(moves) >= 2:
            if 20 <= empties <= 23:
                cands.append((empties, me, opp, colour, moves))
            if probes and 15 <= empties <= 24 and empties not in probe:
                probe[empties] = dict(empties=empties, me=me, opp=opp, colour=colour)
        if ply < 8 or rnd.random() < 0.1:
            sq = rnd.choice(moves)
        else:
            sq = players[colour]._search(U(me), U(opp), moves, 0.01)
        me, opp = child(me, opp, sq)
        colour = -colour
        ply += 1
    return target, cands, [probe[e] for e in sorted(probe)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cap", type=int, default=300_000_000)
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    players = {c: A.AI(8, c, 5) for c in (-1, 1)}
    solver = Solver(args.cap)
    out, t0 = [], time.perf_counter()
    for gid in range(args.worker, args.games, args.workers):
        seed = args.seed_base + gid
        target, cands, probes = play(gid, seed, args.probes, players)
        rec = dict(split=args.split, game=gid, seed=seed, target=target, probes=probes)
        if cands:
            empties, me, opp, colour, moves = min(cands, key=lambda c: abs(c[0] - target))
            rec.update(empties=empties, colour=colour, me=me, opp=opp, moves=moves)
        if cands and not args.no_labels:
            solver.tt.fill(0)
            t, nodes, outcomes = time.perf_counter(), 0, {}
            for sq in moves:
                o, n = solver.outcome(*child(me, opp, sq))
                nodes += n
                if o is None:
                    outcomes = None
                    break
                outcomes[str(sq)] = -o                   # child's side to move is the opponent
            rec.update(outcomes=outcomes, nodes=nodes, seconds=round(time.perf_counter() - t, 3))
        out.append(rec)
        args.out.write_text(json.dumps(out))
        print(json.dumps({k: rec.get(k) for k in ("game", "empties", "outcomes", "nodes", "seconds")}), flush=True)
    args.out.write_text(json.dumps(out))
    print(f"{args.split} worker {args.worker}: {len(out)} games, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
