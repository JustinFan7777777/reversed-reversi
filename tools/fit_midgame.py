"""Fit bounded, reproducible middle-game tables from previously generated games.

Rows 17--19 use exact margins. Higher rows imitate shallow searches whose leaves
use already fitted rows. Held-out bootstrap MAE measures teacher imitation, not
independent playing strength. All fitting and labeling run in one process.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    spec = importlib.util.spec_from_file_location("midgame_frozen_teacher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEBUG_RAISE = True
    module.AI(8, -1, 5)
    return module


def fit(X, y):
    penalty = 1e-4 * float(np.trace(X.T @ X)) / X.shape[1]
    return np.rint(np.linalg.solve(X.T @ X + penalty * np.eye(X.shape[1]), X.T @ y)).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, default=ROOT / "versions/v14_round4_fitted_end_ai.py")
    ap.add_argument("--data", type=Path, default=ROOT / "artifacts/codex_round5/claude_round5_inputs")
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/codex_round5/fitted")
    ap.add_argument("--max-empties", type=int, default=32)
    ap.add_argument("--depth-cap", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=300)
    args = ap.parse_args()
    assert 19 <= args.max_empties <= 44 and 1 <= args.depth_cap <= 12
    started = time.perf_counter()
    deadline = started + args.seconds
    args.output.mkdir(parents=True, exist_ok=True)
    A = load(args.engine)
    files = sorted(args.data.glob("mid_w*.json"))
    rows = [r for f in files for r in json.loads(f.read_text())
            if 17 <= r["empties"] <= args.max_empties]
    assert files and rows
    games = {split: {r["game"] for r in rows if r["split"] == split}
             for split in ("train", "test")}
    assert not games["train"] & games["test"]
    positions = {split: {(r["me"], r["opp"]) for r in rows if r["split"] == split}
                 for split in ("train", "test")}
    assert not positions["train"] & positions["test"], "Training/test board overlap"
    units = [np.tile(np.eye(11, dtype=np.int64)[i], (3, 1)) for i in range(11)]
    for r in rows:
        me, opp = np.uint64(r["me"]), np.uint64(r["opp"])
        moves = np.uint64(A.legal_moves(me, opp))
        assert moves != 0
        r["features"] = [int(A.evaluate(me, opp, moves, w)) for w in units]
    table = A.DEFAULT_W.copy()
    su = np.zeros((A.MAXPLY, A.SU_COLS), np.uint64)
    si = np.zeros((A.MAXPLY, A.SI_COLS), np.int64)
    tt = np.zeros((A.TT_SIZE, 6), np.uint64)
    report = dict(engine=str(args.engine), engine_sha256=hashlib.sha256(args.engine.read_bytes()).hexdigest(),
                  data_files={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  train_games=len(games["train"]), test_games=len(games["test"]),
                  board_overlap=0, rows={}, completed_through=16,
                  bootstrap_mae_caveat="Rows20+ measure imitation of a fitted search teacher; independent matches decide adoption.")
    for e in range(17, args.max_empties + 1):
        if time.perf_counter() >= deadline:
            break
        subset = [r for r in rows if r["empties"] == e]
        labels, nodes = [], 0
        complete = True
        for r in subset:
            if e <= 19:
                labels.append(8 * r["margin"])
                continue
            if time.perf_counter() >= deadline:
                complete = False
                break
            tt.fill(0)
            stats = np.zeros(7, np.int64)
            depth = min(e - 16, args.depth_cap)
            while True:
                if time.perf_counter() >= deadline:
                    complete = False
                    break
                stats[0], stats[1], stats[2] = 0, 4096, 0
                value = int(A.alphabeta(np.uint64(r["me"]), np.uint64(r["opp"]), depth,
                                       -A.INF, A.INF, stats, table, su, si, tt))
                nodes += int(stats[0])
                if stats[2] != 2:
                    if abs(value) >= A.WIN_BASE:
                        margin = value - A.WIN_BASE if value > 0 else value + A.WIN_BASE
                        value = 8 * margin
                    labels.append(value)
                    break
            if not complete:
                break
        if not complete:
            break  # Never fit a partially selected row after the budget expires.
        X = np.asarray([r["features"] for r in subset], dtype=float)
        y = np.asarray(labels, dtype=float)
        train = np.asarray([r["split"] == "train" for r in subset])
        fitted = fit(X[train], y[train])
        metrics = dict(label="exact" if e <= 19 else "bootstrap_depth_%d" % min(e - 16, args.depth_cap),
                       n_train=int(train.sum()), n_test=int((~train).sum()), nodes=nodes,
                       row=fitted.tolist(),
                       train_mae_discs=float(np.abs(X[train] @ fitted - y[train]).mean() / 8),
                       test_mae_discs=float(np.abs(X[~train] @ fitted - y[~train]).mean() / 8))
        table[e] = fitted
        report["rows"][str(e)] = metrics
        report["completed_through"] = e
        report["seconds"] = time.perf_counter() - started
        np.save(args.output / ("through_%02d.npy" % e), table)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("row", e, metrics["label"], "heldout MAE %.3f discs" % metrics["test_mae_discs"],
              "nodes", nodes, "elapsed %.1fs" % report["seconds"], flush=True)
    report["seconds"] = time.perf_counter() - started
    report["complete"] = report["completed_through"] == args.max_empties
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Completed through", report["completed_through"], "in %.1fs" % report["seconds"], flush=True)


if __name__ == "__main__":
    main()
