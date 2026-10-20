"""Held-out matched-opening trials for search, CSO evaluation, and their combination.

Every comparison has explicit per-search seconds, 5-second hard move limits,
separate 5-second startup limits, and 180 seconds PER PLAYER. Results are saved
after each batch. Short-budget screening and a two-game longer-budget pilot do
not establish final competition rank. No automatic edit of the main submission.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import arena


def budgeted(module, seconds):
    class BudgetAI(module.AI):
        def _budget(self, empties):
            return min(seconds, max(0.0, 160 - self.used))
    return SimpleNamespace(AI=BudgetAI, DEBUG_RAISE=True,
                           _import_seconds=getattr(module, "_import_seconds", 0))


def write_weights(template, output, weights):
    text = (ROOT / template).read_text()
    a = text.index("DEFAULT_W = np.array(["); b = text.index("], dtype=np.int64)", a) + len("], dtype=np.int64)")
    body = "\n".join("    " + repr(row) + "," for row in weights.tolist())
    text = text[:a] + "DEFAULT_W = np.array([\n" + body + "\n], dtype=np.int64)" + text[b:]
    (ROOT / output).write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=500)
    ap.add_argument("--games", type=int, default=100)
    args = ap.parse_args()
    started = time.perf_counter(); deadline = started + args.seconds
    weights = np.load(ROOT / "artifacts/round2_cso_weights.npy")
    write_weights("versions/v05_time_budget_ai.py", "versions/v08_cso_ai.py", weights)
    write_weights("versions/v07_pvs_ai.py", "versions/v09_pvs_cso_ai.py", weights)
    modules = {}
    for name, path in (("baseline", "versions/v05_time_budget_ai.py"),
                       ("tt", "versions/v06_tt_order_ai.py"),
                       ("search", "versions/v07_pvs_ai.py"),
                       ("cso", "versions/v08_cso_ai.py"),
                       ("combined", "versions/v09_pvs_cso_ai.py")):
        modules[name] = arena.load(str(ROOT / path), "round2_" + name)
    paths = {"baseline":"versions/v05_time_budget_ai.py", "search":"versions/v07_pvs_ai.py",
             "cso":"versions/v08_cso_ai.py", "combined":"versions/v09_pvs_cso_ai.py"}
    results = []
    decision = {}

    def save():
        (ROOT / "artifacts/round2_results.json").write_text(json.dumps(dict(
            seconds=time.perf_counter()-started, budget_seconds=args.seconds,
            results=results, decision=decision), indent=2) + "\n")

    def match(label, a, b, seconds, count, seed):
        scores, records = [], []
        began = time.perf_counter()
        left, right = budgeted(modules[a], seconds), budgeted(modules[b], seconds)
        for game in range(count):
            if game % 2 == 0 and time.perf_counter() > deadline - 4:
                break
            color = -1 if game % 2 == 0 else 1
            black, white = (left, right) if color == -1 else (right, left)
            metrics = {}
            winner, reason, used = arena.play(black, white, 5,
                opening_plies=6 + 2*((game//2) % 2), seed=seed + game//2, metrics=metrics)
            assert reason == "normal", (label, game, reason, used)
            scores.append(0.5 if winner == 0 else float(winner == color))
            records.append(dict(game=game, opening_seed=seed+game//2, a_color=color,
                                winner=winner, used=used, **metrics))
            if (game+1) % 20 == 0:
                print(label, game+1, "/", count, "score", float(np.mean(scores)), flush=True)
        row = dict(label=label, a=a, b=b, search_seconds=seconds, seed=seed,
                   games=len(scores), wins=scores.count(1), draws=scores.count(0.5), losses=scores.count(0),
                   score=float(np.mean(scores)) if scores else None,
                   ci95=arena.paired_interval(scores), records=records,
                   seconds=time.perf_counter()-began)
        results.append(row); save()
        print({k:v for k,v in row.items() if k!="records"}, flush=True)
        return row

    def accepted(result):
        return result["games"] >= 100 and result["ci95"][0] > 0.5

    search = match("search_vs_baseline", "search", "baseline", 0.02, args.games, 210000)
    cso = match("cso_vs_baseline", "cso", "baseline", 0.02, args.games, 220000)
    combined = match("combined_vs_search", "combined", "search", 0.02, args.games, 230000)
    # An ablation isolates PVS from TT+ordering, using another opening set.
    if time.perf_counter() < deadline - 180:
        match("pvs_ablation_vs_tt", "search", "tt", 0.005, args.games, 240000)
    if accepted(search) and accepted(combined):
        selected = "combined"
    elif accepted(cso):
        selected = "cso"
    elif accepted(search):
        selected = "search"
    else:
        selected = "baseline"
    decision.update(screen_selected=selected, screening_only=True)
    save()
    if selected != "baseline":
        holdout = match("independent_confirmation", selected, "baseline", 0.025,
                        args.games, 310000)
        decision.update(confirm_accepted=accepted(holdout), selected_path=paths[selected])
        # This pair is a transfer sanity check, not a statistically powered claim.
        remaining = deadline - time.perf_counter()
        if remaining >= 125:
            pilot = match("longer_budget_pilot", selected, "baseline", 1.0, 2, 410000)
            decision["longer_budget_pilot_completed"] = pilot["games"] == 2
        elif remaining >= 40:
            pilot = match("medium_budget_pilot", selected, "baseline", 0.25, 2, 410000)
            decision["medium_budget_pilot_completed"] = pilot["games"] == 2
        # Keep a concrete copy of the candidate under validation.
        if accepted(holdout):
            (ROOT / "versions/v10_round2_candidate_ai.py").write_text((ROOT / paths[selected]).read_text())
    save()
    print("Final experiment decision:", decision, flush=True)


if __name__ == "__main__":
    main()
