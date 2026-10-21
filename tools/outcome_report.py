"""Round 7 (protocol revision 1): paired outcome comparison of two engines' choices.

    .venv/bin/python -I tools/outcome_report.py --bench PART.json ... --labels LAB.json ... \
        --out REPORT.json

Per position, delta = outcome of b's choice - outcome of a's choice (0 when both chose the
same move; otherwise from the exact labels of both chosen moves; capped labels excluded
and counted). The gate statistic: positions with delta > 0 (b better) and delta < 0 (a
better), exact two-sided sign test on the nonzero ones. Also by empties count, the share
of differing choices, b's reported outcomes and the slowest move per engine.
"""
import argparse
import json
import math
from pathlib import Path


def sign_test(plus, minus):
    n = plus + minus
    if n == 0:
        return 1.0
    k = min(plus, minus)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(rows):
    plus = sum(r["delta"] > 0 for r in rows if r["delta"] is not None)
    minus = sum(r["delta"] < 0 for r in rows if r["delta"] is not None)
    return dict(positions=len(rows), differing_moves=sum(r["differ"] for r in rows),
                capped=sum(r["delta"] is None for r in rows), b_better=plus, a_better=minus,
                sign_test_p=round(sign_test(plus, minus), 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    parts = [json.loads(Path(f).read_text()) for f in args.bench]
    assert len({json.dumps(p["sha256"], sort_keys=True) for p in parts}) == 1, "mixed engine sources"
    labels = {r["game"]: r for f in args.labels for r in json.loads(Path(f).read_text())["rows"]}
    rows = []
    for r in sorted((r for p in parts for r in p["rows"]), key=lambda r: r["game"]):
        differ = r["a"]["move"] != r["b"]["move"]
        if not differ:
            delta = 0
        else:
            lab = labels[r["game"]]                       # every differing row must be labelled
            delta = None if lab["a"] is None or lab["b"] is None else lab["b"] - lab["a"]
        rows.append(dict(game=r["game"], empties=r["empties"], differ=differ, delta=delta,
                         a_move=r["a"]["move"], b_move=r["b"]["move"],
                         b_last_outcome=r["b"]["last_outcome"],
                         a_seconds=r["a"]["seconds"], b_seconds=r["b"]["seconds"]))
    assert len({r["game"] for r in rows}) == len(rows), "duplicate positions"
    report = dict(paths=parts[0]["paths"], sha256=parts[0]["sha256"], seconds=parts[0]["seconds"],
                  overall=summarize(rows),
                  by_empties={e: summarize([r for r in rows if r["empties"] == e])
                              for e in sorted({r["empties"] for r in rows})},
                  b_reported_outcomes=sum(r["b_last_outcome"] is not None for r in rows),
                  max_move_seconds=dict(a=max(r["a_seconds"] for r in rows),
                                        b=max(r["b_seconds"] for r in rows)))
    print(json.dumps({k: v for k, v in report.items() if k != "paths"}, indent=1))
    args.out.write_text(json.dumps(dict(report, rows=rows), indent=1))


if __name__ == "__main__":
    main()
