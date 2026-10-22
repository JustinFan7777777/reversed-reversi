"""Round 10: shape patterns for rows 20-22 (versions/round10_protocol.md).

    .venv/bin/python -I tools/fit_patterns10.py artifacts/claude_round10/old_w*.json \
        --out artifacts/claude_round10/fit_frozen.json
    .venv/bin/python -I tools/fit_patterns10.py artifacts/claude_round10/old_w*.json --set L \
        --out artifacts/claude_round10/fit_frozen_linear.json
    .venv/bin/python -I tools/fit_patterns10.py artifacts/claude_round10/old_w*.json \
        --out artifacts/claude_round10/test.json --open-test artifacts/claude_round10/fit_frozen.json

Round 9's recipe (tools/fit_patterns9.py: per-row 11 weights centred within each row,
ridge on 8 x exact margin by conjugate gradient, strength from validation, int8-feasible
integer scale from train) with one table bucket 20-22 and shape set P0 (edge+2X, 3x3
corner; protocol revision 2: P1/P2 excluded on measured memory and node-rate cost), or with
--set L the linear-only fallback (the same rows, no tables; revision 3). Split by Round 9
game id: id % 5 == 0 test, validation = train games with id // 5 % 5 == 1. The test games
are scored only with --open-test FROZEN, which scores that file's integer model exactly
(no refit) against v18's own evaluation, after checking that the frozen fit saw the same
records (counts and a data hash). Inputs must be the complete protocol data (9,000 unique
positions, 3 per game, 3,000 games) unless --smoke (synthetic checks); --out is never
overwritten.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_patterns9 import GRID, SETS, SHAPES, Design, load, pair_terms, ridge_cg, shape_index  # noqa: E402

LO, HI = 20, 22
SETS10 = {"P0": SETS["P0"], "L": []}          # L: linear-only fallback (revision 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--open-test", type=Path)
    ap.add_argument("--smoke", action="store_true")      # synthetic data: skip the completeness check
    ap.add_argument("--set", choices=sorted(SETS10), default="P0")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    recs = [r for f in args.inputs for r in json.loads(f.read_text())
            if r["margin"] is not None and LO <= r["empties"] <= HI]
    recs.sort(key=lambda r: (r["game"], r["empties"]))
    keys = [(r["game"], r["empties"]) for r in recs]
    assert len(keys) == len(set(keys)), "duplicate (game, empties) records"
    per_game = Counter(r["game"] for r in recs)
    if not args.smoke:
        # the protocol data: the 9,000 Round 9 positions at 20-22 empties, 3 per game
        assert len(recs) == 9000 and len(per_game) == 3000 and set(per_game.values()) == {3}, \
            ("incomplete or malformed data", len(recs), len(per_game))
    data_sha = hashlib.sha256(json.dumps(
        [(r["game"], r["empties"], r["me"], r["opp"], r["margin"]) for r in recs]).encode()).hexdigest()
    V14 = load("versions/v14_round4_fitted_end_ai.py", "fp10_v14")
    V18 = load("versions/v18_patterns_candidate_ai.py", "fp10_v18")
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
    I = {name: shape_index(me, opp, name) for name in SHAPES}
    empties = np.array([r["empties"] for r in recs])
    y = 8.0 * np.array([r["margin"] for r in recs])
    game = np.array([r["game"] for r in recs])
    test = game % 5 == 0
    val = ~test & ((game // 5) % 5 == 1)
    train = ~test

    def design(mask, names):
        idx = np.flatnonzero(mask)
        rws = empties[idx] - LO
        Fc, yc = F[idx].astype(float), y[idx].copy()
        for k in range(HI - LO + 1):
            m = rws == k
            Fc[m] -= Fc[m].mean(0)
            yc[m] -= yc[m].mean()
        Fc[:, 9] = 0.0
        return idx, Design(Fc, rws, [(n, I[n][idx]) for n in names], HI - LO + 1), yc

    def predict(X, w, idx, S=None):
        d, tabs = X.split(w)
        if S is not None:
            d, tabs = np.round(S * d), [np.round(S * t) for t in tabs]
        out = np.einsum("ij,ij->i", F[idx].astype(float), d[empties[idx] - LO])
        for (n, _), t in zip(X.pats, tabs):
            out += t[I[n][idx]].sum(1)
        return out

    def row_acc(pred, idx):
        accs = []
        for e in np.unique(empties[idx]):
            m = empties[idx] == e
            M, K = pair_terms(pred[m], y[idx][m])
            accs.append(M.sum() / K.sum())
        return float(np.mean(accs))

    if args.open_test:
        frozen = json.loads(args.open_test.read_text())
        model = frozen["model"]
        now = dict(positions=len(recs), train=int(train.sum()), validation=int(val.sum()),
                   test=int(test.sum()), data_sha256=data_sha)
        seen = {k: frozen["report"].get(k) for k in now}
        assert seen == now, ("the frozen fit saw different data", seen, now)
        report = dict(frozen["report"], frozen=str(args.open_test),
                      frozen_sha256=hashlib.sha256(args.open_test.read_bytes()).hexdigest())
        ti = np.flatnonzero(test)
        rows = np.array([model["rows"][str(e)] for e in range(LO, HI + 1)])
        pred = np.einsum("ij,ij->i", F[ti].astype(float), rows[empties[ti] - LO])
        for n in model["shapes"]:
            pred += np.array(model["tables"][n])[I[n][ti]].sum(1)
        tgames = np.unique(game[ti])
        pos = {g: k for k, g in enumerate(tgames)}
        mats = []
        for e in range(LO, HI + 1):
            m = empties[ti] == e
            gi = np.array([pos[g] for g in game[ti][m]])
            mats.append((pair_terms(pred[m], y[ti][m]), pair_terms(v18[ti][m], y[ti][m]), gi))

        def acc(which, wts, rows=range(HI - LO + 1)):
            out = []
            for k in rows:
                pm, pv, gi = mats[k]
                M, K = pm if which == "model" else pv
                ww = wts[gi]
                out.append(ww @ M @ ww / (ww @ K @ ww))
            return float(np.mean(out))

        ones = np.ones(len(tgames))
        rng = np.random.default_rng(10_100_001)
        diffs = []
        for _ in range(1000):
            wts = np.bincount(rng.integers(0, len(tgames), len(tgames)), minlength=len(tgames)).astype(float)
            diffs.append(acc("model", wts) - acc("v18", wts))
        t = dict(positions=int(test.sum()), games=len(tgames), model=acc("model", ones), v18=acc("v18", ones),
                 diff_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                 by_row={LO + k: dict(model=acc("model", ones, [k]), v18=acc("v18", ones, [k]))
                         for k in range(HI - LO + 1)})
        t["diff"] = t["model"] - t["v18"]
        t["gate_pass"] = bool(t["diff"] >= 0.015 and t["diff_ci95"][0] > 0)
        report["test_result"] = t
        print("TEST", json.dumps(t), flush=True)
    else:
        vi = np.flatnonzero(val)
        report = dict(positions=len(recs), train=int(train.sum()), validation=int(val.sum()),
                      test=int(test.sum()), data_sha256=data_sha, smoke=args.smoke,
                      v18_validation=row_acc(v18[vi], vi), sets={})
        for name in (args.set,):
            _, Xf, yf = design(train & ~val, SETS10[name])
            scale = float(np.mean(Xf.Fc ** 2)) * len(yf) / 11
            acc, g = max((row_acc(predict(Xf, ridge_cg(Xf, yf, g * scale / 100), vi), vi), g) for g in GRID)
            report["sets"][name] = dict(validation=acc, strength=g)
            print(name, json.dumps(report["sets"][name]), flush=True)
        sets = report["sets"]
        chosen = max(sets, key=lambda k: sets[k]["validation"])
        report["chosen"] = chosen
        idx, X, yt = design(train, SETS10[chosen])
        scale = float(np.mean(X.Fc ** 2)) * len(yt) / 11
        w = ridge_cg(X, yt, sets[chosen]["strength"] * scale / 100)
        d, tabs = X.split(w)
        # integer scale: only scales whose rounded tables fit int8 (engine storage)
        feasible = [S for S in (1, 2, 4) if max((np.abs(np.round(S * t)).max() for t in tabs), default=0) <= 127]
        clipped = 0
        if feasible:
            S = -max((row_acc(predict(X, w, idx, S), idx), -S) for S in feasible)[1]
        else:
            S = 1
            clipped = int(sum((np.abs(np.round(t)) > 127).sum() for t in tabs))
            tabs = [np.clip(t, -127, 127) for t in tabs]
            w = np.concatenate([w[:X.nd]] + tabs)
        d = np.round(S * w[:X.nd].reshape(X.n_rows, 11)).astype(int)
        model = dict(shapes=SETS10[chosen], rows={}, tables={})
        for k, e in enumerate(range(LO, HI + 1)):
            row = d[k].tolist()
            row[9] = int(S * V14.DEFAULT_W[e, 9])
            model["rows"][e] = row
        model["tables"] = {n: np.round(S * t).astype(int).tolist() for (n, _), t in zip(X.pats, tabs)}
        model["scale"], model["clipped"] = S, clipped
        print("chosen", chosen, "scale", S, "clipped", clipped, flush=True)
    args.out.write_text(json.dumps(dict(report=report, model=model)))


if __name__ == "__main__":
    main()
