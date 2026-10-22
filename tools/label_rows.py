"""Round 10 data: exact final margins for unlabelled records of tools/build_exact_rows.py.

    .venv/bin/python -I tools/label_rows.py artifacts/claude_round9/rows_w*.json \
        --min-empties 20 --max-empties 22 --worker J --workers K \
        --out artifacts/claude_round10/old_wJ.json

Protocol: versions/round10_protocol.md. Records with margin None and empties in range are
solved with the same full-window solver as build_exact_rows (exact margin, > 0 wins for
the side to move). Records keep game, split and colour. Output rewritten every 50 records
through a temporary file (atomic replace). --resume continues a dead worker from its own
output after checking that it is an exact prefix of the worker's deterministic shard.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_exact_rows import Solver  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--min-empties", type=int, default=20)
    ap.add_argument("--max-empties", type=int, default=22)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)       # timing probe: stop after N records
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    recs = [r for f in args.inputs for r in json.loads(f.read_text())
            if r["margin"] is None and args.min_empties <= r["empties"] <= args.max_empties]
    recs.sort(key=lambda r: (r["game"], r["empties"]))
    mine = recs[args.worker::args.workers]
    if args.limit:
        mine = mine[:args.limit]
    out = []
    if args.resume and args.out.exists():
        out = json.loads(args.out.read_text())
        done = [(r["game"], r["empties"]) for r in out]
        assert done == [(r["game"], r["empties"]) for r in mine[:len(done)]], "not a prefix of this shard"
        print(f"resuming after {len(out)} records", flush=True)
    elif args.out.exists():
        raise SystemExit(f"{args.out} exists; pass --resume to continue it")

    def save():
        tmp = args.out.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            fh.write(json.dumps(out))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, args.out)

    solver, t0 = Solver(), time.perf_counter()
    for k in range(len(out), len(mine)):
        r = mine[k]
        t = time.perf_counter()
        m, n = solver.margin(r["me"], r["opp"])
        out.append(dict(r, margin=m, nodes=n, seconds=round(time.perf_counter() - t, 3)))
        if k % 50 == 49:
            save()
            print(f"{k + 1}/{len(mine)} {time.perf_counter() - t0:.0f} s", flush=True)
    save()
    print(f"worker {args.worker}: {len(out)} records, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
