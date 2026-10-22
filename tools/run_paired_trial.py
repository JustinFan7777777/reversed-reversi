"""Single-process matched-opening trial with incremental records and source hashes.

The search budget is equal for both players. Official 5-second startup and move
limits and independent 180-second player banks remain enforced. Save after every
game. The run's --seconds budget is checked before starting each opening pair;
an active pair may finish after it. Re-running overwrites the output, without resume.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

import arena


def budgeted(module, seconds):
    class BudgetAI(module.AI):
        def _budget(self, empties):
            return min(seconds, max(0.0, 160 - self.used))
    return SimpleNamespace(AI=BudgetAI, DEBUG_RAISE=True,
                           _import_seconds=module._import_seconds)


def summarize(records):
    scores = [0.5 if r["winner"] == 0 else float(r["winner"] == r["a_color"]) for r in records]
    colors = {}
    for color in (-1, 1):
        subset = [r for r in records if r["a_color"] == color]
        colors["black" if color == -1 else "white"] = dict(Counter(
            "draw" if r["winner"] == 0 else "win" if r["winner"] == color else "loss"
            for r in subset))
    return dict(games=len(records), wins=scores.count(1), draws=scores.count(0.5),
                losses=scores.count(0), score=float(np.mean(scores)) if scores else None,
                paired_ci95=arena.paired_interval(scores), by_color=colors,
                reasons=dict(Counter(r["reason"] for r in records)),
                # a game lost at initialization never moved, so it has no move times
                max_move_seconds=max((r.get("max_move_seconds", 0.0) for r in records), default=0),
                max_side_seconds=max((max(r["used"].values()) for r in records), default=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--search-seconds", type=float, default=0.2)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=1800)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert args.games > 0 and args.games % 2 == 0 and 0 < args.search_seconds <= 4
    started = time.perf_counter()
    deadline = started + args.seconds
    paths = {"a": args.a.resolve(), "b": args.b.resolve()}
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    modules = {k: arena.load(str(p), "paired_trial_" + k) for k, p in paths.items()}
    players = {k: budgeted(m, args.search_seconds) for k, m in modules.items()}
    records = []
    report = dict(paths={k: str(p) for k, p in paths.items()}, sha256=hashes,
                  target_games=args.games, search_seconds=args.search_seconds, seed=args.seed,
                  budget_seconds=args.seconds, startup_limit=5, move_limit=5, per_side_limit=180,
                  opening_plies="6 and 8 alternating by pair", records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report.update(summarize(records))
        report["seconds"] = time.perf_counter() - started
        report["complete"] = len(records) == args.games
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for game in range(args.games):
        if game % 2 == 0:
            # Reserve enough time to finish a pair at the observed recent cost.
            pair_estimate = max(15.0, 2.4 * np.mean([r["wall_seconds"] for r in records[-10:]])) if records else 15.0
            if time.perf_counter() + pair_estimate >= deadline:
                break
        a_color = -1 if game % 2 == 0 else 1
        black, white = (players["a"], players["b"]) if a_color == -1 else (players["b"], players["a"])
        metrics = {}
        game_started = time.perf_counter()
        winner, reason, used = arena.play(black, white, 5, opening_plies=6 + 2 * ((game // 2) % 2),
                                          seed=args.seed + game // 2, metrics=metrics)
        records.append(dict(game=game, opening_seed=args.seed + game // 2, a_color=a_color,
                            winner=winner, reason=reason, used=used,
                            wall_seconds=time.perf_counter() - game_started, **metrics))
        save()
        if (game + 1) % 10 == 0 or reason != "normal":
            print("games", len(records), "W/D/L", report["wins"], report["draws"], report["losses"],
                  "score %.3f" % report["score"], "elapsed %.1fs" % report["seconds"], flush=True)
        if reason != "normal":
            raise AssertionError("Non-normal trial game: " + reason)
    save()
    for k, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[k], "Source changed during trial"
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
