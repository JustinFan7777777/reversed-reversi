"""Round 7: median node cost of a complete outcome-only root solve per empties count.

    .venv/bin/python -I tools/outcome_cost.py ENGINE --data artifacts/claude_round7/train_w*.json \
        --per-empties 40 --cap 200000000 --worker J --workers K --out PART.json
    .venv/bin/python -I tools/outcome_cost.py --merge PART.json ... --out TABLE.json

Training probe positions only (one per game and empties count 15-24, written by
build_outcome_set.py --probes); per empties count, the first --per-empties games by id.
Each position: cold table, the driver's own heuristic iterations to depth
min(8, empties - 1) for the move order, then the engine's own _outcome_root with no time
limit but a node cap (a clock shim reports the deadline as passed once the cap is
reached). Only the outcome solve's nodes count. A capped solve is censored; the median is
reported as the cap when at least half of an empties count is censored.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


class NodeClock:
    """Stands in for the time module inside the engine: the deadline passes at the cap."""

    def __init__(self, ai, cap):
        self.ai, self.cap = ai, cap

    def perf_counter(self):
        return float("inf") if self.ai.last_nodes >= self.cap else 0.0


def measure(args):
    spec = importlib.util.spec_from_file_location("outcome_cost_engine", args.engine)
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)
    X.DEBUG_RAISE = True
    ai = X.AI(8, -1, 5)
    games = []
    for f in args.data:
        games += json.loads(Path(f).read_text())
    games.sort(key=lambda g: g["game"])
    jobs = []
    for e in range(15, 25):
        with_probe = [(g["game"], p) for g in games for p in g["probes"] if p["empties"] == e]
        jobs += [(e, gid, p) for gid, p in with_probe[:args.per_empties]]
    su = np.zeros((X.MAXPLY, X.SU_COLS), dtype=np.uint64)
    si = np.zeros((X.MAXPLY, X.SI_COLS), dtype=np.int64)
    stats = np.zeros(7, dtype=np.int64)
    real_time, rows = X.time, []
    for k, (e, gid, p) in enumerate(jobs):
        if k % args.workers != args.worker:
            continue
        me, opp = np.uint64(p["me"]), np.uint64(p["opp"])
        moves = X.bb_to_squares(X.legal_moves(me, opp))
        ai.tt.fill(0)
        order = sorted(moves, key=lambda sq: (bool((1 << sq) & int(X.CORNERS)),
                                              int(X.popcnt(np.uint64(X.flips(me, opp, np.uint64(1 << sq)))))))
        best = order[0]
        for depth in range(1, min(8, e - 1) + 1):
            results, best, _, complete = ai._iteration(me, opp, order, depth, best, stats, su, si,
                                                        float("inf"))
            assert complete
            results.sort(key=lambda item: -item[0])
            order = [sq for _, sq in results]
        order = [best] + [sq for sq in order if sq != best]
        ai.last_nodes = 0
        X.time = NodeClock(ai, args.cap)
        try:
            ai._outcome_root(me, opp, order, stats, su, si, 1.0)
        finally:
            X.time = real_time
        rows.append(dict(empties=e, game=gid, nodes=int(ai.last_nodes),
                         censored=bool(ai.last_nodes >= args.cap)))
        print(json.dumps(rows[-1]), flush=True)
    return dict(engine=str(args.engine), cap=args.cap, per_empties=args.per_empties, rows=rows)


def merge(parts, cap):
    rows = [r for p in parts for r in p["rows"]]
    table = {}
    for e in range(15, 25):
        # a censored solve counts as the cap: the median equals the cap once half are censored
        sub = sorted(min(r["nodes"], cap) if not r["censored"] else cap for r in rows if r["empties"] == e)
        if not sub:
            continue
        med = float(np.median(sub))
        table[e] = dict(n=len(sub), censored=sum(1 for r in rows if r["empties"] == e and r["censored"]),
                        median=int(med),
                        q25=int(min(np.percentile(sub, 25), cap)), q75=int(min(np.percentile(sub, 75), cap)))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", nargs="?", type=Path)
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--per-empties", type=int, default=40)
    ap.add_argument("--cap", type=int, default=200_000_000)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--merge", nargs="*")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.merge:
        parts = [json.loads(Path(f).read_text()) for f in args.merge]
        cap = parts[0]["cap"]
        out = dict(engine=parts[0]["engine"], cap=cap, per_empties=parts[0]["per_empties"],
                   table=merge(parts, cap), rows=[r for p in parts for r in p["rows"]])
        for e, t in out["table"].items():
            print(e, t)
    else:
        out = measure(args)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
