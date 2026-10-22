"""Node rate of a candidate against a base engine (Stage B, versions/round10_protocol.md).

    .venv/bin/python -I tools/bench_node_rate.py versions/v18_patterns_candidate_ai.py \
        versions/v20_patterns_candidate_ai.py --out artifacts/claude_round10/node_rate.json

Run on an otherwise idle CPU. Two position sets, engines alternated per position:
  standard    Round 8's yardstick: 30 random positions at 22-34 empties (seed 8_300_001,
              tests/test_search.random_position), each searched cold for 2 s through
              AI._search; reports nodes/s and mean depth. This is the gated ratio.
  leaf_heavy  every position with 29-31 empties and >= 2 legal moves in v18's platform
              batch 05, searched cold to fixed depth 9 (all leaves at rows 20-22): the
              worst case for evaluation cost at those rows. Reported, not gated.
"""
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import analyze_replays as P  # noqa: E402
import test_search as T      # noqa: E402

ARCHIVE = ROOT / "artifacts/platform_batch05/platform_v18_input.zip"


def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DEBUG_RAISE = True
    return mod


def standard_positions(X):
    rnd, pos = random.Random(8_300_001), []
    while len(pos) < 30:
        e = 22 + 2 * (len(pos) % 7)
        b, c = T.random_position(rnd, 60 - e)
        if int((b == 0).sum()) == e:
            me, opp = X.board_to_bb(b, c)
            if len(X.bb_to_squares(X.legal_moves(me, opp))) >= 2:
                pos.append((me, opp))
    return pos


def leaf_heavy_positions(X):
    pos = []
    for g in P.load_games(ARCHIVE):
        for i, m in enumerate(g["moves"]):
            if 29 <= m["empties_before"] <= 31 and m["legal_count"] >= 2:
                pos.append(X.board_to_bb(g["_before"][i], m["actor"]))
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    E = {"base": load(args.base, "nr_base"), "candidate": load(args.candidate, "nr_candidate")}
    ais = {k: m.AI(8, -1, 5) for k, m in E.items()}       # also the JIT warm-up
    out = {"base": str(args.base), "candidate": str(args.candidate)}

    tot = {k: [0, 0.0, 0] for k in E}
    pos = standard_positions(E["base"])
    for i, (me, opp) in enumerate(pos):
        for k in (("base", "candidate") if i % 2 == 0 else ("candidate", "base")):
            X, ai = E[k], ais[k]
            ai.tt.fill(0)
            t = time.perf_counter()
            ai._search(me, opp, X.bb_to_squares(X.legal_moves(me, opp)), 2.0)
            tot[k][0] += ai.last_nodes
            tot[k][1] += time.perf_counter() - t
            tot[k][2] += ai.last_depth
    out["standard"] = {k: dict(nodes=n, seconds=s, mnps=n / s / 1e6, mean_depth=d / len(pos))
                       for k, (n, s, d) in tot.items()}
    out["standard"]["nps_ratio"] = out["standard"]["candidate"]["mnps"] / out["standard"]["base"]["mnps"]

    tot = {k: [0, 0.0] for k in E}
    pos = leaf_heavy_positions(E["base"])
    for i, (me, opp) in enumerate(pos):
        for k in (("base", "candidate") if i % 2 == 0 else ("candidate", "base")):
            M = E[k]
            su = np.zeros((M.MAXPLY, M.SU_COLS), dtype=np.uint64)
            si = np.zeros((M.MAXPLY, M.SI_COLS), dtype=np.int64)
            tt = np.zeros((M.TT_SIZE, 6), dtype=np.uint64)
            stats = np.array([0, 1 << 60, 0], dtype=np.int64)
            t = time.perf_counter()
            M.alphabeta(np.uint64(me), np.uint64(opp), 9, -M.INF, M.INF, stats, M.DEFAULT_W, su, si, tt)
            tot[k][1] += time.perf_counter() - t
            tot[k][0] += int(stats[0])
    out["leaf_heavy"] = {k: dict(nodes=n, seconds=s, mnps=n / s / 1e6) for k, (n, s) in tot.items()}
    out["leaf_heavy"]["positions"] = len(pos)
    out["leaf_heavy"]["nps_ratio"] = out["leaf_heavy"]["candidate"]["mnps"] / out["leaf_heavy"]["base"]["mnps"]
    out["leaf_heavy"]["time_ratio"] = out["leaf_heavy"]["candidate"]["seconds"] / out["leaf_heavy"]["base"]["seconds"]
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k]["nps_ratio"] for k in ("standard", "leaf_heavy")}))
    for k in ("base", "candidate"):
        print(k, "standard %.2f Mnps depth %.2f; leaf-heavy %.2f Mnps" % (
            out["standard"][k]["mnps"], out["standard"][k]["mean_depth"], out["leaf_heavy"][k]["mnps"]))


if __name__ == "__main__":
    main()
