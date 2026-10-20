"""Equal-depth correctness and equal-time search benchmarks for candidate engines."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import arena
import test_search as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/round2_search_benchmark.json")
    args = ap.parse_args()
    rng = random.Random(98172)
    fixtures = [(ply, *T.random_position(rng, ply)) for ply in (12, 24, 36, 44)
                for _ in range(2)]
    paths = ["versions/v05_phase1_ai.py", "versions/v06_tt_order_ai.py", "versions/v07_pvs_ai.py"]
    records, expected, startups = [], {}, {}
    for index, path in enumerate(paths):
        mod = arena.load(str(ROOT / path), "search_benchmark_%d" % index)
        mod.DEBUG_RAISE = True
        start = time.perf_counter()
        ai = mod.AI(8, -1, 5)
        startups[path] = time.perf_counter() - start + mod._import_seconds
        assert startups[path] < 5, (path, "startup exceeds five seconds")
        for f, (ply, board, color) in enumerate(fixtures):
            me, opp = mod.board_to_bb(board, color)
            for depth in (6, 8):
                su = np.zeros((mod.MAXPLY, mod.SU_COLS), dtype=np.uint64)
                si = np.zeros((mod.MAXPLY, mod.SI_COLS), dtype=np.int64)
                stats = np.array([0, 1 << 60, 0], dtype=np.int64)
                if hasattr(ai, "tt"):
                    ai.tt.fill(0)
                start = time.perf_counter()
                values = (me, opp, depth, -mod.INF, mod.INF, stats, ai.W, su, si)
                value = int(mod.alphabeta(*values, ai.tt) if hasattr(ai, "tt")
                            else mod.alphabeta(*values))
                seconds = time.perf_counter() - start
                if index == 0:
                    expected[(f, depth)] = value
                assert value == expected[(f, depth)], (path, f, depth, value, expected[(f, depth)])
                records.append(dict(path=path, fixture=f, ply=ply, depth=depth,
                                    value=value, nodes=int(stats[0]), seconds=seconds))
            if hasattr(ai, "tt"):
                ai.tt.fill(0)
            moves = mod.bb_to_squares(mod.legal_moves(me, opp))
            start = time.perf_counter()
            move = ai._search(me, opp, moves, 0.25)
            elapsed = time.perf_counter() - start
            assert move in moves and elapsed < 0.35
            records.append(dict(path=path, fixture=f, ply=ply, budget_seconds=0.25,
                                seconds=elapsed, completed_depth=ai.last_depth,
                                nodes=ai.last_nodes, move=move))
        fixed = [r for r in records if r["path"] == path and "depth" in r]
        timed = [r for r in records if r["path"] == path and "completed_depth" in r]
        print(path, "startup %.3fs" % startups[path], "fixed nodes", sum(r["nodes"] for r in fixed),
              "fixed seconds %.4f" % sum(r["seconds"] for r in fixed),
              "timed depths", [r["completed_depth"] for r in timed], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(startups=startups, records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
