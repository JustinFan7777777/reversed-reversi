"""Round 9: richer shape patterns for rows 10-19 (versions/round9_protocol.md).

    .venv/bin/python -I tools/fit_patterns9.py --old artifacts/claude_round8/rows_w*.json \
        --new artifacts/claude_round9/rows_w*.json --out artifacts/claude_round9/fit.json [--open-test]

Same recipe as tools/fit_patterns.py (per-row 11 weights centred within each row, pattern
tables shared within the buckets 10-13, 14-16, 17-19, ridge on 8 x exact margin by
conjugate gradient, strength per bucket from validation, integer scale from train), for
three candidate shape sets:
  P0  edge+2X (3^10), 3x3 corner (3^9)                                  -- v18's shapes
  P1  P0 + 2x5 corner block, both orientations at each corner (3^10, 8 instances)
  P2  P1 + inner lines 2-4 (3^8 each, 4 instances) + diagonals of length 8..4
Folding: a configuration and its mirror image share an entry (reversed line or edge,
transposed corner block); the 2x5 block has no internal symmetry. Data: Round 8 games are
all training; new games with id % 5 == 0 are test; validation = new train games with
id // 5 % 5 == 1. The set with the best mean validation ranking is refitted on all
training data; the test games are scored only with --open-test, against v18's own values.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = ((10, 13), (14, 16), (17, 19))
GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def sq(r, c):
    return 8 * r + c


def rev(n):
    return list(range(n - 1, -1, -1))


def dihedral_lines(row):
    """A horizontal line at `row` and its 3 images (opposite row, both columns)."""
    return [[sq(row, c) for c in range(8)], [sq(7 - row, c) for c in range(8)],
            [sq(r, row) for r in range(8)], [sq(r, 7 - row) for r in range(8)]]


def diagonals(length):
    if length == 8:
        return [[sq(i, i) for i in range(8)], [sq(i, 7 - i) for i in range(8)]]
    k = 8 - length
    return [[sq(i, i + k) for i in range(length)], [sq(i + k, i) for i in range(length)],
            [sq(i, 7 - k - i) for i in range(length)], [sq(i + k, 7 - i) for i in range(length)]]


EDGE8 = [[sq(0, c) for c in range(8)], [sq(7, c) for c in range(8)],
         [sq(r, 0) for r in range(8)], [sq(r, 7) for r in range(8)]]
EDGE2X = [[sq(1, 1)] + EDGE8[0] + [sq(1, 6)], [sq(6, 1)] + EDGE8[1] + [sq(6, 6)],
          [sq(1, 1)] + EDGE8[2] + [sq(6, 1)], [sq(1, 6)] + EDGE8[3] + [sq(6, 6)]]
FLIPS = ((lambda r: r, lambda c: c), (lambda r: r, lambda c: 7 - c),
         (lambda r: 7 - r, lambda c: c), (lambda r: 7 - r, lambda c: 7 - c))
CORNER9 = [[sq(rf(r), cf(c)) for r in range(3) for c in range(3)] for rf, cf in FLIPS]
TRANSPOSE9 = [3 * c + r for r in range(3) for c in range(3)]
CORNER25 = ([[sq(rf(r), cf(c)) for r in range(2) for c in range(5)] for rf, cf in FLIPS] +
            [[sq(rf(c), cf(r)) for r in range(2) for c in range(5)] for rf, cf in FLIPS])

SHAPES = {
    "edge2x": (EDGE2X, rev(10)), "corner9": (CORNER9, TRANSPOSE9), "corner25": (CORNER25, None),
    "line2": (dihedral_lines(1), rev(8)), "line3": (dihedral_lines(2), rev(8)),
    "line4": (dihedral_lines(3), rev(8)),
    "diag8": (diagonals(8), rev(8)), "diag7": (diagonals(7), rev(7)), "diag6": (diagonals(6), rev(6)),
    "diag5": (diagonals(5), rev(5)), "diag4": (diagonals(4), rev(4)),
}
SETS = {"P0": ["edge2x", "corner9"], "P1": ["edge2x", "corner9", "corner25"],
        "P2": ["edge2x", "corner9", "corner25", "line2", "line3", "line4",
               "diag8", "diag7", "diag6", "diag5", "diag4"]}


def ternary(me, opp, squares):
    idx = np.zeros(me.shape, np.int64)
    for k, s in enumerate(squares):
        st = ((me >> np.uint64(s)) & np.uint64(1)).astype(np.int64) + \
            2 * ((opp >> np.uint64(s)) & np.uint64(1)).astype(np.int64)
        idx += st * 3 ** k
    return idx


def shape_index(me, opp, name):
    inst, mirror = SHAPES[name]
    cols = []
    for s in inst:
        a = ternary(me, opp, s)
        cols.append(a if mirror is None else np.minimum(a, ternary(me, opp, [s[j] for j in mirror])))
    return np.stack(cols, axis=1)


def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Design:
    """Per-row dense 11-column blocks + one shared table per shape (implicit sparse)."""

    def __init__(self, Fc, rows, pats, n_rows):
        self.Fc, self.rows, self.pats, self.n_rows = Fc, rows, pats, n_rows
        self.nd = 11 * n_rows
        self.sizes = [3 ** len(SHAPES[name][0][0]) for name, _ in pats]
        self.n = self.nd + sum(self.sizes)

    def split(self, w):
        out, at = [], self.nd
        for size in self.sizes:
            out.append(w[at:at + size])
            at += size
        return w[:self.nd].reshape(self.n_rows, 11), out

    def mul(self, w):
        d, tabs = self.split(w)
        out = np.einsum("ij,ij->i", self.Fc, d[self.rows])
        for (_, I), t in zip(self.pats, tabs):
            out += t[I].sum(1)
        return out

    def rmul(self, v):
        g = np.zeros(self.n)
        d, tabs = self.split(g)
        np.add.at(d, self.rows, self.Fc * v[:, None])
        for (_, I), t in zip(self.pats, tabs):
            np.add.at(t, I.ravel(), np.repeat(v, I.shape[1]))
        return g


def ridge_cg(X, y, lam, iters=500):
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


def pair_terms(pred, y):
    dy = np.sign(y[:, None] - y[None, :])
    dp = np.sign(pred[:, None] - pred[None, :])
    keep = dy > 0
    return np.where(keep, (dp > 0) + 0.5 * (dp == 0), 0.0), keep.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", nargs="+", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--open-test", action="store_true")
    args = ap.parse_args()
    old = [dict(r, split="train", src="r8") for f in args.old for r in json.loads(Path(f).read_text())]
    new = [dict(r, src="r9") for f in args.new for r in json.loads(Path(f).read_text()) if r["margin"] is not None]
    recs = sorted(old + new, key=lambda r: (r["src"], r["game"], r["empties"]))
    V14 = load("versions/v14_round4_fitted_end_ai.py", "fp9_v14")
    V18 = load("versions/v18_patterns_candidate_ai.py", "fp9_v18")
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
    is_new = np.array([r["src"] == "r9" for r in recs])
    game = np.array([r["game"] for r in recs])
    test = is_new & (game % 5 == 0)
    val = is_new & ~test & ((game // 5) % 5 == 1)
    train = ~test

    def bucket_design(lo, hi, mask, names):
        idx = np.flatnonzero((empties >= lo) & (empties <= hi) & mask)
        rws = empties[idx] - lo
        Fc, yc = F[idx].astype(float), y[idx].copy()
        for k in range(hi - lo + 1):
            m = rws == k
            Fc[m] -= Fc[m].mean(0)
            yc[m] -= yc[m].mean()
        Fc[:, 9] = 0.0
        return idx, Design(Fc, rws, [(n, I[n][idx]) for n in names], hi - lo + 1), yc

    def predict(lo, X, w, idx, S=None):
        d, tabs = X.split(w)
        if S is not None:
            d, tabs = np.round(S * d), [np.round(S * t) for t in tabs]
        out = np.einsum("ij,ij->i", F[idx].astype(float), d[empties[idx] - lo])
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

    report = dict(positions=len(recs), train=int(train.sum()), validation=int(val.sum()),
                  test=int(test.sum()), sets={})
    fitmask = train & ~val
    for name, shapes in SETS.items():
        per_bucket, accs = {}, []
        for lo, hi in BUCKETS:
            _, Xf, yf = bucket_design(lo, hi, fitmask, shapes)
            scale = float(np.mean(Xf.Fc ** 2)) * len(yf) / 11
            vi = np.flatnonzero((empties >= lo) & (empties <= hi) & val)
            best = max((row_acc(predict(lo, Xf, ridge_cg(Xf, yf, g * scale / 100), vi), vi), g) for g in GRID)
            per_bucket[f"{lo}-{hi}"] = dict(validation=best[0], strength=best[1])
            accs.append(best[0])
        v18_val = float(np.mean([row_acc(v18[np.flatnonzero((empties >= lo) & (empties <= hi) & val)],
                                         np.flatnonzero((empties >= lo) & (empties <= hi) & val))
                                 for lo, hi in BUCKETS]))
        report["sets"][name] = dict(buckets=per_bucket, validation=float(np.mean(accs)), v18_validation=v18_val)
        print(name, json.dumps(report["sets"][name]), flush=True)
    chosen = max(report["sets"], key=lambda k: report["sets"][k]["validation"])
    report["chosen"] = chosen
    model = dict(shapes=SETS[chosen], rows={}, tables={})
    pred_test = np.zeros(len(recs))
    for lo, hi in BUCKETS:
        g = report["sets"][chosen]["buckets"][f"{lo}-{hi}"]["strength"]
        idx, X, yt = bucket_design(lo, hi, train, SETS[chosen])
        scale = float(np.mean(X.Fc ** 2)) * len(yt) / 11
        w = ridge_cg(X, yt, g * scale / 100)
        d, tabs = X.split(w)
        # integer scale: only scales whose rounded tables fit int8 (engine storage)
        feasible = [S for S in (1, 2, 4) if max(np.abs(np.round(S * t)).max() for t in tabs) <= 127]
        if feasible:
            S = -max((row_acc(predict(lo, X, w, idx, S), idx), -S) for S in feasible)[1]
            clipped = 0
        else:
            S = 1
            clipped = int(sum((np.abs(np.round(t)) > 127).sum() for t in tabs))
            tabs = [np.clip(t, -127, 127) for t in tabs]
            w = np.concatenate([w[:X.nd]] + tabs)
        d = np.round(S * w[:X.nd].reshape(X.n_rows, 11)).astype(int)
        for k, e in enumerate(range(lo, hi + 1)):
            row = d[k].tolist()
            row[9] = int(S * V14.DEFAULT_W[e, 9])
            model["rows"][e] = row
        model["tables"][f"{lo}-{hi}"] = {n: np.round(S * t).astype(int).tolist() for (n, _), t in zip(X.pats, tabs)}
        model["tables"][f"{lo}-{hi}"]["scale"] = S
        model["tables"][f"{lo}-{hi}"]["clipped"] = clipped
        ti = np.flatnonzero((empties >= lo) & (empties <= hi) & test)
        pred_test[ti] = predict(lo, X, w, ti, S)
    if args.open_test:
        tgames = np.unique(game[test])
        pos = {g: k for k, g in enumerate(tgames)}
        mats = []
        for e in range(10, 20):
            m = np.flatnonzero(test & (empties == e))
            gi = np.array([pos[g] for g in game[m]])
            mats.append((pair_terms(pred_test[m], y[m]), pair_terms(v18[m], y[m]), gi))

        def acc(which, wts):
            out = []
            for (pm, pv, gi) in mats:
                M, K = pm if which == "model" else pv
                w = wts[gi]
                out.append(w @ M @ w / (w @ K @ w))
            return float(np.mean(out))

        ones = np.ones(len(tgames))
        rng = np.random.default_rng(9_100_001)
        diffs = []
        for _ in range(1000):
            wts = np.bincount(rng.integers(0, len(tgames), len(tgames)), minlength=len(tgames)).astype(float)
            diffs.append(acc("model", wts) - acc("v18", wts))
        report["test"] = dict(model=acc("model", ones), v18=acc("v18", ones),
                              diff=acc("model", ones) - acc("v18", ones),
                              diff_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
        report["test"]["gate_pass"] = bool(report["test"]["diff"] >= 0.010 and report["test"]["diff_ci95"][0] > 0)
        print("TEST", json.dumps(report["test"]), flush=True)
    args.out.write_text(json.dumps(dict(report=report, model=model)))


if __name__ == "__main__":
    main()
