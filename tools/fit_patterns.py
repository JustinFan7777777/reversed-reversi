"""Round 8: fit shape-pattern evaluation rows 10-19 on exact margins (versions/round8_protocol.md).

    .venv/bin/python -I tools/fit_patterns.py --data artifacts/claude_round8/rows_w*.json \
        --out artifacts/claude_round8/fit.json [--open-test]

Features per position: the 11 evaluate() columns of the frozen v14 snapshot (unit weights)
and two shape patterns read from the side to move (0 empty, 1 own, 2 opponent):
  EDGE2X  four instances: X square, the eight edge squares corner to corner, X square (3^10)
  CORNER9 four instances: the 3x3 block from each corner, row by row from the corner (3^9)
Each index is folded with its mirror image (reversed edge / transposed block), so one
table entry serves every symmetric instance.

Model: row e in 10-19 has its own 11 weights; pattern tables are shared within the buckets
10-13, 14-16 and 17-19. Ridge on 8 x margin (v14's units) after centring features and
target within each row (a per-row constant never changes a move; parity keeps v14's
weight). Solved by conjugate gradient. The strength per bucket is chosen on a validation
fifth of the train games (game id // 5 % 5 == 1), then the bucket is refitted on all of
train. Integers: weights are scaled by S in {1, 2, 4} (chosen on train pairwise accuracy)
and rounded. The test games are scored only with --open-test, once, after this is frozen.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = ((10, 13), (14, 16), (17, 19))
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def sq(r, c):
    return 8 * r + c


EDGE8 = [[sq(0, c) for c in range(8)], [sq(7, c) for c in range(8)],
         [sq(r, 0) for r in range(8)], [sq(r, 7) for r in range(8)]]
EDGE2X = [[sq(1, 1)] + EDGE8[0] + [sq(1, 6)], [sq(6, 1)] + EDGE8[1] + [sq(6, 6)],
          [sq(1, 1)] + EDGE8[2] + [sq(6, 1)], [sq(1, 6)] + EDGE8[3] + [sq(6, 6)]]
CORNER9 = [[sq(rf(r), cf(c)) for r in range(3) for c in range(3)]
           for rf, cf in ((lambda r: r, lambda c: c), (lambda r: r, lambda c: 7 - c),
                          (lambda r: 7 - r, lambda c: c), (lambda r: 7 - r, lambda c: 7 - c))]
TRANSPOSE9 = [3 * c + r for r in range(3) for c in range(3)]


def ternary(me, opp, squares):
    idx = np.zeros(me.shape, np.int64)
    for k, s in enumerate(squares):
        st = ((me >> np.uint64(s)) & np.uint64(1)).astype(np.int64) + \
            2 * ((opp >> np.uint64(s)) & np.uint64(1)).astype(np.int64)
        idx += st * 3 ** k
    return idx


def folded(me, opp, instances, mirror):
    return np.stack([np.minimum(ternary(me, opp, s), ternary(me, opp, [s[j] for j in mirror(len(s))]))
                     for s in instances], axis=1)


def features(recs):
    spec = importlib.util.spec_from_file_location("fit_patterns_v14", ROOT / "versions/v14_round4_fitted_end_ai.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    unit = []
    for k in range(11):
        w = np.zeros((3, 11), np.int64)
        w[:, k] = 1
        unit.append(w)
    F = np.zeros((len(recs), 11), np.int64)
    for i, r in enumerate(recs):
        me, opp = np.uint64(r["me"]), np.uint64(r["opp"])
        mv = np.uint64(A.legal_moves(me, opp))
        for k in range(11):
            F[i, k] = A.evaluate(me, opp, mv, unit[k])
    me = np.array([r["me"] for r in recs], dtype=np.uint64)
    opp = np.array([r["opp"] for r in recs], dtype=np.uint64)
    return (A, F, folded(me, opp, EDGE2X, lambda m: range(m - 1, -1, -1)),
            folded(me, opp, CORNER9, lambda m: TRANSPOSE9))


class Design:
    """Implicit sparse design for one bucket: per-row dense blocks + two shared tables."""

    def __init__(self, Fc, rows, E, C, n_rows):
        self.Fc, self.rows, self.E, self.C, self.n_rows = Fc, rows, E, C, n_rows
        self.nd = 11 * n_rows
        self.n = self.nd + 3 ** 10 + 3 ** 9

    def mul(self, w):
        d = w[:self.nd].reshape(self.n_rows, 11)
        out = np.einsum("ij,ij->i", self.Fc, d[self.rows])
        out += w[self.nd:self.nd + 3 ** 10][self.E].sum(1)
        out += w[self.nd + 3 ** 10:][self.C].sum(1)
        return out

    def rmul(self, v):
        g = np.zeros(self.n)
        np.add.at(g[:self.nd].reshape(self.n_rows, 11), self.rows, self.Fc * v[:, None])
        ge = np.zeros(3 ** 10)
        np.add.at(ge, self.E.ravel(), np.repeat(v, self.E.shape[1]))
        gc = np.zeros(3 ** 9)
        np.add.at(gc, self.C.ravel(), np.repeat(v, self.C.shape[1]))
        g[self.nd:self.nd + 3 ** 10], g[self.nd + 3 ** 10:] = ge, gc
        return g


def ridge_cg(X, y, lam, iters=400):
    """Minimise |Xw - y|^2 + lam |w|^2 by conjugate gradient on the normal equations."""
    w = np.zeros(X.n)
    r = X.rmul(y)
    p, rs = r.copy(), r @ r
    for _ in range(iters):
        Ap = X.rmul(X.mul(p)) + lam * p
        a = rs / (p @ Ap)
        w += a * p
        r -= a * Ap
        new = r @ r
        if new < 1e-12 * max(rs, 1e-30):
            break
        p, rs = r + (new / rs) * p, new
    return w


def pair_matrix(pred, y):
    """Per-row pairwise ranking terms: (agree - 0.5*tie) and count for label-ordered pairs."""
    dy = np.sign(y[:, None] - y[None, :])
    dp = np.sign(pred[:, None] - pred[None, :])
    keep = dy > 0
    return np.where(keep, (dp > 0) + 0.5 * (dp == 0), 0.0), keep.astype(float)


def mean_row_accuracy(rows_data, weights=None):
    accs = []
    for (M, K, games) in rows_data:
        w = np.ones(len(games)) if weights is None else weights[games]
        accs.append(float(w @ M @ w / (w @ K @ w)))
    return float(np.mean(accs)), accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--open-test", action="store_true")
    args = ap.parse_args()
    recs = [r for f in args.data for r in json.loads(Path(f).read_text())]
    recs.sort(key=lambda r: (r["game"], r["empties"]))
    A, F, E, C = features(recs)
    empties = np.array([r["empties"] for r in recs])
    game = np.array([r["game"] for r in recs])
    test = np.array([r["split"] == "test" for r in recs])
    y = 8.0 * np.array([r["margin"] for r in recs])
    v14 = np.einsum("ij,ij->i", A.DEFAULT_W[empties].astype(float), F)
    val = ~test & ((game // 5) % 5 == 1)
    fitp = ~test & ~val
    rows_out = {}
    tables = {}
    report = dict(positions=len(recs), train=int((~test).sum()), test=int(test.sum()), buckets={})
    for lo, hi in BUCKETS:
        rlist = list(range(lo, hi + 1))
        inb = (empties >= lo) & (empties <= hi)

        def build(mask):
            idx = np.flatnonzero(inb & mask)
            rws = empties[idx] - lo
            Fc = F[idx].astype(float)
            yc = y[idx].copy()
            for k in range(len(rlist)):            # centre within each row
                m = rws == k
                Fc[m] -= Fc[m].mean(0)
                yc[m] -= yc[m].mean()
            Fc[:, 9] = 0.0                          # parity is a per-row constant
            return idx, Design(Fc, rws, E[idx], C[idx], len(rlist)), yc

        def predict(w, idx, S=None):
            d = w[:11 * len(rlist)].reshape(len(rlist), 11)
            te, tc = w[11 * len(rlist):11 * len(rlist) + 3 ** 10], w[11 * len(rlist) + 3 ** 10:]
            if S is not None:
                d, te, tc = np.round(S * d), np.round(S * te), np.round(S * tc)
            return (np.einsum("ij,ij->i", F[idx].astype(float), d[empties[idx] - lo])
                    + te[E[idx]].sum(1) + tc[C[idx]].sum(1))

        def row_acc(pred_all, mask):
            out = []
            for e in rlist:
                m = np.flatnonzero(inb & mask & (empties == e))
                M, K = pair_matrix(pred_all[m], y[m])
                out.append(float(M.sum() / K.sum()))
            return float(np.mean(out))

        _, Xf, yf = build(fitp)
        scale = float(np.mean(Xf.Fc ** 2)) * len(yf) / 11
        choice = []
        for g in GRID:
            w = ridge_cg(Xf, yf, g * scale / 100)
            pred = np.zeros(len(recs))
            vi = np.flatnonzero(inb & val)
            pred[vi] = predict(w, vi)
            choice.append((row_acc(pred, val), g))
        best_g = max(choice)[1]
        _, Xt, yt = build(~test)
        w = ridge_cg(Xt, yt, best_g * scale / 100)
        tri = np.flatnonzero(inb & ~test)
        s_choice = []
        for S in (1, 2, 4):
            pred = np.zeros(len(recs))
            pred[tri] = predict(w, tri, S)
            s_choice.append((row_acc(pred, ~test), -S))
        S = -max(s_choice)[1]
        d = np.round(S * w[:11 * len(rlist)].reshape(len(rlist), 11)).astype(int)
        for k, e in enumerate(rlist):
            row = d[k].tolist()
            row[9] = int(S * A.DEFAULT_W[e, 9])     # parity: v14's value in the new units
            rows_out[e] = row
        tables[f"{lo}-{hi}"] = dict(
            edge2x=np.round(S * w[11 * len(rlist):11 * len(rlist) + 3 ** 10]).astype(int).tolist(),
            corner9=np.round(S * w[11 * len(rlist) + 3 ** 10:]).astype(int).tolist(), scale=S)
        pv = np.zeros(len(recs))
        pv[tri] = v14[tri]
        report["buckets"][f"{lo}-{hi}"] = dict(validation=choice, strength=best_g, integer_scale=S,
                                               train_acc_v14=row_acc(pv, ~test),
                                               train_acc_model=max(s_choice)[0])
        print(f"{lo}-{hi}", report["buckets"][f"{lo}-{hi}"], flush=True)
    model = dict(rows=rows_out, tables=tables)
    if args.open_test:
        pm = np.zeros(len(recs))
        for lo, hi in BUCKETS:
            t = tables[f"{lo}-{hi}"]
            for i in np.flatnonzero((empties >= lo) & (empties <= hi)):
                pm[i] = (np.array(rows_out[empties[i]]) @ F[i] + sum(t["edge2x"][j] for j in E[i])
                         + sum(t["corner9"][j] for j in C[i]))
        data_m, data_v = [], []
        test_games = np.unique(game[test])
        gpos = {g: k for k, g in enumerate(test_games)}
        for e in range(10, 20):
            m = np.flatnonzero(test & (empties == e))
            gi = np.array([gpos[g] for g in game[m]])
            data_m.append(pair_matrix(pm[m], y[m]) + (gi,))
            data_v.append(pair_matrix(v14[m], y[m]) + (gi,))
        acc_m, rows_m = mean_row_accuracy(data_m)
        acc_v, rows_v = mean_row_accuracy(data_v)
        rng = np.random.default_rng(8_100_001)
        diffs = []
        for _ in range(1000):
            wts = np.bincount(rng.integers(0, len(test_games), len(test_games)),
                              minlength=len(test_games)).astype(float)
            diffs.append(mean_row_accuracy(data_m, wts)[0] - mean_row_accuracy(data_v, wts)[0])
        report["test"] = dict(model=acc_m, v14=acc_v, diff=acc_m - acc_v,
                              diff_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                              by_row={e: dict(model=a, v14=b) for e, a, b in zip(range(10, 20), rows_m, rows_v)},
                              gate_pass=bool(acc_m - acc_v >= 0.015 and np.percentile(diffs, 2.5) > 0))
        print("TEST", json.dumps({k: v for k, v in report["test"].items() if k != "by_row"}), flush=True)
    args.out.write_text(json.dumps(dict(report=report, model=model)))


if __name__ == "__main__":
    main()
