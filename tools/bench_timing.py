"""Bounded timing stress: broad middlegames, exact endgames and distorted nps.

The optional slowdown delays each search slice to model a slower platform. It is
not a claim about platform hardware. Run sequentially, on an otherwise idle CPU.
"""
import argparse
import importlib.util
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ai as A
import ref_engine as R


def positions():
    rnd, found = random.Random(761), {}
    for _ in range(12):
        board, color = R.initial_board(), -1
        for ply in range(60):
            legal = R.legal_moves(board, color)
            if not legal:
                color = -color
                legal = R.legal_moves(board, color)
                if not legal:
                    break
            if ply in (12, 24, 36, 48, 50, 52, 54):
                if ply not in found or len(legal) > found[ply][0]:
                    found[ply] = (len(legal), board.copy(), color)
            board = R.apply_move(board, color, rnd.choice(legal))
            color = -color
    return found


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=pathlib.Path, default=ROOT / "ai.py")
    ap.add_argument("--time-out", type=float, default=5.0)
    ap.add_argument("--max-move-seconds", type=float, default=4.5)
    ap.add_argument("--slow-factor", type=float, default=3.0)
    ap.add_argument("--json", type=pathlib.Path, default=ROOT / "artifacts/timing.json")
    args = ap.parse_args()
    if args.path.resolve() != (ROOT / "ai.py").resolve():
        spec = importlib.util.spec_from_file_location("timing_candidate", args.path)
        A = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(A)
    A.DEBUG_RAISE = True
    # Instantiate both colors before installing the synthetic slowdown.
    players = {c: A.AI(8, c, args.time_out) for c in (-1, 1)}
    actual = A.alphabeta

    def delayed(*values):
        start = time.perf_counter()
        result = actual(*values)
        time.sleep(max(0.0, (time.perf_counter() - start) * (args.slow_factor - 1)))
        return result

    rows = []
    fixtures = positions()
    for slow in (False, True):
        A.alphabeta = delayed if slow else actual
        for ply, (branching, board, color) in fixtures.items():
            ai = players[color]
            ai.used, ai.nps = 0.0, 1e12  # Deliberately useless speed estimate.
            start = time.perf_counter()
            ai.go(board)
            elapsed = time.perf_counter() - start
            assert set(ai.candidate_list) == set(R.legal_moves(board, color))
            assert ai.fallback_count == 0
            assert elapsed < min(args.max_move_seconds, args.time_out), (ply, slow, elapsed)
            row = dict(ply=ply, branching=branching, slow=slow, seconds=elapsed,
                       depth=ai.last_depth, nodes=ai.last_nodes, solved=ai.last_solved)
            rows.append(row)
            print(row, flush=True)
    A.alphabeta = actual
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rows, indent=2) + "\n")
    print("OK: maximum %.4fs across %d timing stress cases" %
          (max(r["seconds"] for r in rows), len(rows)))


if __name__ == "__main__":
    main()
