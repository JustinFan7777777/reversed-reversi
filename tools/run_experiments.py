"""Bounded single-process M3/M4/M5 experiments; no automatic promotion.

All comparisons use matched random openings with colors swapped. Feature sweeps
and SPSA are exploratory. Independent confirmation is required before promotion.
The timer stops between games; reserve a margin below the authorized compute cap.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=540)
    ap.add_argument("--time-out", type=float, default=0.01)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--output", type=pathlib.Path, default=ROOT / "artifacts/experiments.json")
    args = ap.parse_args()
    start, results = time.perf_counter(), []
    deadline = start + args.seconds
    base = arena.load(str(ROOT / "versions/v01_ai.py"), "baseline_m2")
    tt = arena.load(str(ROOT / "versions/v02_tt_ai.py"), "candidate_tt")
    features = arena.load(str(ROOT / "versions/v03_features_ai.py"), "candidate_features")
    old = arena.load(str(ROOT / "versions/v00_ai.py"), "baseline_original")
    random_opponent = arena.load(str(ROOT / "tools/baseline_random.py"), "random_opponent")
    greedy = arena.load(str(ROOT / "tools/baseline_greedy.py"), "greedy_opponent")

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(seconds=time.perf_counter() - start,
            budget_seconds=args.seconds, time_out=args.time_out, results=results), indent=2) + "\n")

    def match(label, a, b, games, seed, weights=None):
        if weights is not None:
            a.DEFAULT_W = weights.copy()
        scores, records = [], []
        began = time.perf_counter()
        for game in range(games):
            # Finish whole opening pairs. Maximum overshoot is a single bounded game.
            if game % 2 == 0 and time.perf_counter() >= deadline - 3:
                break
            a_color = -1 if game % 2 == 0 else 1
            black, white = (a, b) if a_color == -1 else (b, a)
            metrics = {}
            # Millisecond search budgets limit local compute; the official hard
            # limit stays 5 s. Record scheduling overruns rather than mistaking a
            # 10 ms experimental allowance for the course's actual timeout.
            winner, reason, used = arena.play(black, white, args.time_out,
                opening_plies=6, seed=seed + game // 2, strict_time_out=False, metrics=metrics)
            if reason != "normal":
                raise AssertionError((label, game, reason, used))
            scores.append(0.5 if winner == 0 else float(winner == a_color))
            records.append(dict(game=game, a_color=a_color, winner=winner,
                                opening_seed=seed + game // 2, seconds=used, **metrics))
            if (game + 1) % 20 == 0:
                print("%s: %d/%d games, score %.3f" %
                      (label, game + 1, games, np.mean(scores)), flush=True)
        result = dict(label=label, games=len(scores), seed=seed,
                      wins=scores.count(1.0), draws=scores.count(0.5), losses=scores.count(0.0),
                      score=float(np.mean(scores)) if scores else None,
                      ci95=arena.paired_interval(scores), records=records,
                      seconds=time.perf_counter() - began,
                      weights=weights.tolist() if weights is not None else None)
        results.append(result)
        print({k: v for k, v in result.items() if k not in ("records", "weights")}, flush=True)
        save()
        return result

    def significant(result):
        return result["games"] >= 100 and result["ci95"][0] > 0.5

    # M3: Search enhancement is tested on independent matched openings.
    table_result = match("M3_tt_vs_m2", tt, base, args.games, 4100)
    champion = tt if significant(table_result) else base

    # M4: One feature at a time, testing BOTH signs to check intuitions empirically.
    initial = features.DEFAULT_W.copy()
    selected = initial.copy()
    for column, name, magnitude in ((6, "stable_edges", 10), (7, "frontier", 4),
                                    (8, "potential_mobility", 4), (9, "parity", 4),
                                    (10, "position", 1)):
        if time.perf_counter() >= min(start + args.seconds * 0.60, deadline - 120):
            break
        signs = []
        for sign in (1, -1):
            weights = initial.copy()
            weights[:, column] = magnitude * sign
            result = match("M4_%s_%+d" % (name, sign), features, tt,
                           args.games, 5200 + column * 100, weights)
            signs.append((result, weights))
        good = [(r, w) for r, w in signs if significant(r)]
        if good:
            _, winner_weights = max(good, key=lambda item: item[0]["score"])
            selected[:, column] = winner_weights[:, column]

    # M5: Small reproducible SPSA pilot against historical versions, never the
    # held-out validation openings. Optimizing a few noisy games proves nothing.
    best_weights = selected.astype(float)
    rnd = np.random.default_rng(827)
    for iteration in range(4):
        if time.perf_counter() >= deadline - 35:
            break
        direction = rnd.choice([-1, 1], size=best_weights.shape)
        c = 2.0 / ((iteration + 1) ** 0.101)
        opponents = (base, old, tt, base)
        plus = np.rint(best_weights + c * direction).astype(np.int64)
        minus = np.rint(best_weights - c * direction).astype(np.int64)
        positive = match("M5_spsa_%d_plus" % iteration, features,
                         opponents[iteration], 8, 10000 + iteration * 20, plus)
        negative = match("M5_spsa_%d_minus" % iteration, features,
                         opponents[iteration], 8, 10000 + iteration * 20, minus)
        if positive["games"] < 8 or negative["games"] < 8:
            break
        gradient = (positive["score"] - negative["score"]) / (2 * c) * direction
        best_weights = np.clip(best_weights + 8.0 / ((iteration + 1) ** 0.602) * gradient,
                               -100, 100)
    proposed = np.rint(best_weights).astype(np.int64)
    np.save(ROOT / "artifacts/proposed_weights.npy", proposed)

    # Confirm the combined/tuned candidate on unseen opening seeds and two
    # historical opponents. Both must pass; baseline matches do not select weights.
    first = match("heldout_tuned_vs_champion", features, champion, args.games,
                  30000, proposed)
    second = match("heldout_tuned_vs_original", features, old, args.games,
                   40000, proposed)
    match("sanity_tuned_vs_greedy", features, greedy, 10, 50000, proposed)
    match("sanity_tuned_vs_random", features, random_opponent, 6, 51000, proposed)
    promotion = dict(tt=significant(table_result),
                     tuned=significant(first) and significant(second),
                     note="Independent confirmation required; no file is changed automatically.")
    (ROOT / "artifacts/promotion.json").write_text(json.dumps(promotion, indent=2) + "\n")
    save()
    print("Promotion evidence:", promotion, flush=True)


if __name__ == "__main__":
    main()
