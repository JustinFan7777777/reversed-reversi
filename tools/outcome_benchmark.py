"""Round 7 paired benchmark: two engines choose on the same fresh positions.

    .venv/bin/python -I tools/outcome_benchmark.py --data artifacts/claude_round7/pos_holdout_w*.json \
        --a versions/v14_round4_fitted_end_ai.py --b versions/v17_outcome_first_candidate_ai.py \
        --seconds 4.0 --worker J --workers K --out PART.json

Every recorded position (protocol revision 1: no labels needed). Each engine searches from
a cold table with the given budget through its own AI._search, in one process; which
engine goes first alternates by position. Before the first position each engine
calibrates its node-rate estimate with a 1-second search on a fixed position with 30
empties, as its earlier moves would in a game. Where the two moves differ,
tools/outcome_disagree.py later solves both chosen moves exactly.
(Round 7 data is archived: run `tar -xJkf artifacts/archive_history.tar.xz` from the repo root first.)
"""
import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np

U = np.uint64


def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DEBUG_RAISE = True
    return mod


def calibration_position(X):
    """Fixed position with 30 empties: seeded random moves from the start."""
    rnd = random.Random(7_200_001)
    me, opp = U((1 << 35) | (1 << 28)), U((1 << 27) | (1 << 36))
    while 64 - int(X.popcnt(me | opp)) > 30:
        moves = X.bb_to_squares(X.legal_moves(me, opp))
        if not moves:
            me, opp = opp, me
            continue
        b = U(1 << rnd.choice(moves))
        f = U(X.flips(me, opp, b))
        me, opp = U(opp & ~f), U(me | b | f)
    return me, opp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    paths = {"a": args.a.resolve(), "b": args.b.resolve()}
    sha = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    mods = {k: load(p, "outcome_bench_" + k) for k, p in paths.items()}
    ais = {k: m.AI(8, -1, 5) for k, m in mods.items()}
    for k, ai in ais.items():
        me, opp = calibration_position(mods[k])
        ai._search(me, opp, mods[k].bb_to_squares(mods[k].legal_moves(me, opp)), 1.0)
        ai.tt.fill(0)
    recs = []
    for f in args.data:
        recs += json.loads(Path(f).read_text())
    recs = sorted((r for r in recs if r.get("moves")), key=lambda r: r["game"])
    rows, started = [], time.perf_counter()
    for idx, r in enumerate(recs):
        if idx % args.workers != args.worker:
            continue
        row = dict(game=r["game"], empties=r["empties"], legal=len(r["moves"]), me=r["me"],
                   opp=r["opp"], first="a" if idx % 2 == 0 else "b")
        for k in (("a", "b") if idx % 2 == 0 else ("b", "a")):
            X, ai = mods[k], ais[k]
            me, opp = U(r["me"]), U(r["opp"])
            moves = X.bb_to_squares(X.legal_moves(me, opp))
            assert sorted(moves) == sorted(r["moves"]), (r["game"], moves, r["moves"])
            ai.tt.fill(0)
            t = time.perf_counter()
            sq = ai._search(me, opp, moves, args.seconds)
            row[k] = dict(move=int(sq), depth=ai.last_depth, last_outcome=getattr(ai, "last_outcome", None),
                          nodes=int(ai.last_nodes), seconds=round(time.perf_counter() - t, 3))
        rows.append(row)
        print(json.dumps(row), flush=True)
    for k, p in paths.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha[k], "source changed during the run"
    args.out.write_text(json.dumps(dict(paths={k: str(p) for k, p in paths.items()}, sha256=sha,
                                        seconds=args.seconds, data=args.data, worker=args.worker,
                                        workers=args.workers, wall=time.perf_counter() - started,
                                        rows=rows), indent=1))


if __name__ == "__main__":
    main()
