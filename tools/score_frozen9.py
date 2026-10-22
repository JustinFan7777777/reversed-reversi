"""Round 9 correction: score the frozen Round 9 model on the complete test set.

    .venv/bin/python -I tools/score_frozen9.py artifacts/claude_round9/fit_frozen.json \
        artifacts/claude_round9/rows_w*.json --out artifacts/claude_round9/test_complete.json \
        [--engine versions/v19_check_ai.py]

The Round 9 gate (versions/round9_protocol.md) was first scored while 5 of the 6 data
workers were still writing, on 5,710 of the 6,000 test positions (NOTES.md Round 9,
correction). This scores the same frozen integer model, without refitting, on every
labelled new-game record with game id % 5 == 0: within-row pairwise ranking at rows
10-19 against v18's own evaluation, game-bootstrap (1,000 resamples, seed 9_100_001, as
tools/fit_patterns9.py). With --engine (a tools/build_v19.py build of the same frozen
file) the engine's evaluate() is checked to equal the scored prediction exactly.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_patterns9 import load, pair_terms, shape_index  # noqa: E402

BUCKETS = ((10, 13), (14, 16), (17, 19))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frozen", type=Path)
    ap.add_argument("rows", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine", type=Path)
    args = ap.parse_args()
    model = json.loads(args.frozen.read_text())["model"]
    recs = [r for f in args.rows for r in json.loads(f.read_text())
            if r["margin"] is not None and r["game"] % 5 == 0 and 10 <= r["empties"] <= 19]
    recs.sort(key=lambda r: (r["game"], r["empties"]))
    keys = [(r["game"], r["empties"]) for r in recs]
    assert len(keys) == len(set(keys)), "duplicate records"
    V14 = load("versions/v14_round4_fitted_end_ai.py", "sf9_v14")
    V18 = load("versions/v18_patterns_candidate_ai.py", "sf9_v18")
    unit = []
    for k in range(11):
        w = np.zeros((3, 11), np.int64)
        w[:, k] = 1
        unit.append(w)
    F = np.zeros((len(recs), 11), np.int64)
    v18 = np.zeros(len(recs))
    for i, r in enumerate(recs):
        me, opp = np.uint64(r["me"]), np.uint64(r["opp"])
        mv = np.uint64(V14.legal_moves(me, opp))
        for k in range(11):
            F[i, k] = V14.evaluate(me, opp, mv, unit[k])
        v18[i] = V18.evaluate(me, opp, mv, V18.DEFAULT_W)
    me = np.array([r["me"] for r in recs], dtype=np.uint64)
    opp = np.array([r["opp"] for r in recs], dtype=np.uint64)
    empties = np.array([r["empties"] for r in recs])
    y = 8.0 * np.array([r["margin"] for r in recs])
    game = np.array([r["game"] for r in recs])
    I = {n: shape_index(me, opp, n) for n in model["shapes"]}
    rows = np.array([model["rows"][str(e)] for e in range(10, 20)], dtype=np.int64)
    pred = (F * rows[empties - 10]).sum(1).astype(np.int64)
    for lo, hi in BUCKETS:
        m = (empties >= lo) & (empties <= hi)
        for n in model["shapes"]:
            t = np.array(model["tables"][f"{lo}-{hi}"][n], dtype=np.int64)
            pred[m] += t[I[n][m]].sum(1)
    checked = 0
    if args.engine:
        X = load(str(args.engine), "sf9_engine")
        for i in range(0, len(recs), 7):
            a, b = np.uint64(recs[i]["me"]), np.uint64(recs[i]["opp"])
            got = int(X.evaluate(a, b, np.uint64(X.legal_moves(a, b)), X.DEFAULT_W))
            assert got == int(pred[i]), (i, got, int(pred[i]))
            checked += 1
    tgames = np.unique(game)
    pos = {g: k for k, g in enumerate(tgames)}
    mats = []
    for e in range(10, 20):
        m = empties == e
        gi = np.array([pos[g] for g in game[m]])
        mats.append((pair_terms(pred[m].astype(float), y[m]), pair_terms(v18[m], y[m]), gi))

    def acc(which, wts, rows=range(10)):
        out = []
        for k in rows:
            pm, pv, gi = mats[k]
            M, K = pm if which == "model" else pv
            ww = wts[gi]
            out.append(ww @ M @ ww / (ww @ K @ ww))
        return float(np.mean(out))

    ones = np.ones(len(tgames))
    rng = np.random.default_rng(9_100_001)
    diffs = []
    for _ in range(1000):
        wts = np.bincount(rng.integers(0, len(tgames), len(tgames)), minlength=len(tgames)).astype(float)
        diffs.append(acc("model", wts) - acc("v18", wts))
    t = dict(positions=len(recs), games=len(tgames), model=acc("model", ones), v18=acc("v18", ones),
             diff_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
             by_row={10 + k: dict(model=acc("model", ones, [k]), v18=acc("v18", ones, [k])) for k in range(10)},
             frozen=str(args.frozen), frozen_sha256=hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
             engine_checked=checked)
    t["diff"] = t["model"] - t["v18"]
    t["gate_pass"] = bool(t["diff"] >= 0.010 and t["diff_ci95"][0] > 0)
    args.out.write_text(json.dumps(t, indent=1))
    print(json.dumps({k: t[k] for k in ("positions", "games", "model", "v18", "diff", "diff_ci95", "gate_pass",
                                        "engine_checked")}))


if __name__ == "__main__":
    main()
