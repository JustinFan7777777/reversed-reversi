"""Confirm the tuned evaluator without an unproven TT before final promotion.

This is a separate validation set; weights come from the earlier bounded pilot.
Output records evidence only. It never changes the submission automatically.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import arena


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--time-out", type=float, default=0.01)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--weights", type=pathlib.Path, default=ROOT / "artifacts/proposed_weights.npy")
    ap.add_argument("--original-time-out", type=float, default=0.125)
    args = ap.parse_args()
    start = time.perf_counter()
    deadline = start + args.seconds
    candidate = arena.load(str(ROOT / "versions/v04_features_no_tt_ai.py"), "no_tt_confirmation")
    weights = np.load(args.weights)
    candidate.DEFAULT_W = weights.copy()
    results = []
    for path, seed in (("versions/v01_ai.py", 60000), ("versions/v00_ai.py", 70000)):
        # v00 imposes a 50 ms search-budget floor. A >=125 ms timeout makes
        # timeout*0.40 >=50 ms, so both algorithms receive equal intended budgets.
        time_out = args.original_time_out if "v00" in path else args.time_out
        opponent = arena.load(str(ROOT / path), "confirmation_opponent_" + str(seed))
        scores, records = [], []
        for game in range(args.games):
            if game % 2 == 0 and time.perf_counter() > deadline - 3:
                break
            color = -1 if game % 2 == 0 else 1
            black, white = (candidate, opponent) if color == -1 else (opponent, candidate)
            metrics = {}
            winner, reason, used = arena.play(black, white, time_out, opening_plies=6,
                seed=seed + game // 2, strict_time_out=False, metrics=metrics)
            if reason != "normal":
                raise AssertionError((game, reason, used))
            scores.append(0.5 if winner == 0 else float(winner == color))
            records.append(dict(game=game, a_color=color, winner=winner, seconds=used, **metrics))
            if (game + 1) % 20 == 0:
                print(path, game + 1, float(np.mean(scores)), flush=True)
        row = dict(opponent=path, seed=seed, time_out=time_out, games=len(scores), wins=scores.count(1.0),
                   draws=scores.count(0.5), losses=scores.count(0.0),
                   score=float(np.mean(scores)) if scores else None,
                   ci95=arena.paired_interval(scores), records=records)
        results.append(row)
        print({k: v for k, v in row.items() if k != "records"}, flush=True)
    approved = all(r["games"] >= 100 and r["ci95"][0] > 0.5 for r in results)
    result = dict(seconds=time.perf_counter() - start, time_out=args.time_out,
                  weights=weights.tolist(), results=results, promotion=approved)
    (ROOT / "artifacts/no-tt-confirmation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("No-TT candidate promotion evidence:", approved, flush=True)


if __name__ == "__main__":
    main()
