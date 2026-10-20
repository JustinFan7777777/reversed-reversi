"""Predeclared PVS ablation, fresh final confirmation, and longer-budget transfer pair."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import arena
from round2_experiments import budgeted, write_weights


def main():
    began = time.perf_counter()
    weights = np.load(ROOT / "artifacts/round2_cso_weights.npy")
    write_weights("versions/v06_tt_order_ai.py", "versions/v11_tt_cso_ai.py", weights)
    paths = dict(pvs="versions/v09_pvs_cso_ai.py", tt="versions/v11_tt_cso_ai.py",
                 baseline="versions/v05_time_budget_ai.py")
    modules = {k:arena.load(str(ROOT / p), "final_trial_"+k) for k,p in paths.items()}
    results = []

    def match(label, a, b, count, seconds, seed):
        scores, records = [], []
        left, right = budgeted(modules[a], seconds), budgeted(modules[b], seconds)
        for game in range(count):
            color = -1 if game % 2 == 0 else 1
            black, white = (left,right) if color == -1 else (right,left)
            metrics = {}
            winner, reason, used = arena.play(black, white, 5, opening_plies=6+2*((game//2)%2),
                                               seed=seed+game//2, metrics=metrics)
            assert reason == "normal", (label, game, reason, used)
            scores.append(0.5 if winner == 0 else float(winner == color))
            records.append(dict(game=game, a_color=color, winner=winner,
                                opening_seed=seed+game//2, used=used, **metrics))
            if (game+1)%20 == 0:
                print(label, game+1, float(np.mean(scores)), flush=True)
        row = dict(label=label,a=a,b=b,games=count,search_seconds=seconds,seed=seed,
                   wins=scores.count(1),draws=scores.count(0.5),losses=scores.count(0),
                   score=float(np.mean(scores)),ci95=arena.paired_interval(scores),records=records)
        results.append(row)
        (ROOT / "artifacts/round2_final_trials.json").write_text(json.dumps(dict(
            seconds=time.perf_counter()-began,results=results),indent=2)+"\n")
        print({k:v for k,v in row.items() if k!="records"},flush=True)
        return row

    # Retain PVS only if it demonstrates an additional strength gain. Otherwise
    # the simpler TT+ordering engine is the default for this release candidate.
    ablation = match("pvs_with_final_weights", "pvs", "tt", 100, 0.005, 500000)
    selected = "pvs" if ablation["ci95"][0] > 0.5 else "tt"
    confirm = match("final_fresh_confirmation", selected,"baseline",100,0.015,510000)
    transfer = match("one_second_transfer_pair", selected,"baseline",2,1.0,610000)
    accept = confirm["ci95"][0] > 0.5 and transfer["losses"] < 2
    decision = dict(selected=selected,selected_path=paths[selected],
                    accepted=accept,seconds=time.perf_counter()-began,
                    reason="Fresh 100-game confirmation plus non-catastrophic transfer pair; larger-budget sample remains small.")
    (ROOT / "artifacts/round2_final_decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    if accept:
        (ROOT / "versions/v12_round2_selected_ai.py").write_text((ROOT / paths[selected]).read_text())
    print("Release-candidate decision:",decision,flush=True)


if __name__ == "__main__":
    main()
