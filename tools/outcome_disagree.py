"""Round 7 (protocol revision 1): exact outcomes of both chosen moves where two engines differ.

    .venv/bin/python -I tools/outcome_disagree.py BENCH_PART.json ... --worker J --workers K \
        --cap 1000000000 --out LABELS_PART.json

For every benchmark row whose two chosen moves differ, each chosen move is solved for its
outcome only (window -1..1 to the end of the game, one table per position, at most --cap
nodes per move) with the frozen v14 snapshot's search. Outcomes are from the mover's side:
1 win, 0 draw, -1 loss; None when the cap was reached.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script dir out
import build_outcome_set as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cap", type=int, default=1_000_000_000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = sorted((r for f in args.parts for r in json.loads(Path(f).read_text())["rows"]
                   if r["a"]["move"] != r["b"]["move"]), key=lambda r: r["game"])
    solver, out = B.Solver(args.cap), []
    for k, r in enumerate(rows):
        if k % args.workers != args.worker:
            continue
        solver.tt.fill(0)
        t, rec = time.perf_counter(), dict(game=r["game"], empties=r["empties"], nodes=0)
        for side in ("a", "b"):
            o, n = solver.outcome(*B.child(r["me"], r["opp"], r[side]["move"]))
            rec[side] = None if o is None else -o           # the child's mover is the opponent
            rec["nodes"] += n
        rec["seconds"] = round(time.perf_counter() - t, 3)
        out.append(rec)
        args.out.write_text(json.dumps(dict(cap=args.cap, rows=out)))
        print(json.dumps(rec), flush=True)
    args.out.write_text(json.dumps(dict(cap=args.cap, rows=out)))


if __name__ == "__main__":
    main()
