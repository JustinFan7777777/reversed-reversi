"""Pool several tools/run_paired_trial.py outputs into one paired result (Round 10 Stage C).

    .venv/bin/python tools/pool_paired.py OUT1.json OUT2.json ... --out POOLED.json \
        [--gate interval|point]

Checks before pooling: every input is complete, every game ended normally, all inputs ran
the same two engines (sha256 of a and b), and opening seeds are disjoint across inputs
with each seed played exactly twice (once per colour) inside its input. The per-game
scores of candidate `a` are then concatenated input by input (whole opening pairs, in
game order) and summarized with arena.paired_interval, which resamples pairs together,
as each single run does. --gate interval passes when the pooled 95% interval lies
entirely above 0.5; --gate point when the pooled score is at least 0.5.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arena  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gate", choices=("interval", "point"))
    args = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in args.inputs]
    for p, r in zip(args.inputs, runs):
        assert r["complete"] and len(r["records"]) == r["target_games"], ("incomplete run", str(p))
        assert all(g["reason"] == "normal" for g in r["records"]), ("non-normal game", str(p))
        seeds = Counter(g["opening_seed"] for g in r["records"])
        assert set(seeds.values()) == {2}, ("opening seed not played exactly twice", str(p))
        for a, b in zip(r["records"][::2], r["records"][1::2]):
            assert a["opening_seed"] == b["opening_seed"] and a["a_color"] == -b["a_color"], str(p)
    assert len({(r["sha256"]["a"], r["sha256"]["b"]) for r in runs}) == 1, "inputs ran different engines"
    all_seeds = [s for r in runs for s in {g["opening_seed"] for g in r["records"]}]
    assert len(all_seeds) == len(set(all_seeds)), "opening seeds overlap across inputs"
    scores = [0.5 if g["winner"] == 0 else float(g["winner"] == g["a_color"])
              for r in runs for g in r["records"]]
    out = dict(inputs=[str(p) for p in args.inputs], sha256=runs[0]["sha256"], paths=runs[0]["paths"],
               search_seconds=sorted({r["search_seconds"] for r in runs}),
               games=len(scores), wins=scores.count(1.0), draws=scores.count(0.5), losses=scores.count(0.0),
               score=float(np.mean(scores)), paired_ci95=arena.paired_interval(scores),
               max_move_seconds=max(r["max_move_seconds"] for r in runs),
               max_side_seconds=max(r["max_side_seconds"] for r in runs))
    if args.gate == "interval":
        out["gate"] = dict(rule="pooled paired 95% interval entirely above 0.5",
                           passed=bool(out["paired_ci95"][0] > 0.5))
    elif args.gate == "point":
        out["gate"] = dict(rule="pooled score at least 0.5", passed=bool(out["score"] >= 0.5))
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("games", "wins", "draws", "losses", "score", "paired_ci95")}
                     | ({"gate": out["gate"]} if "gate" in out else {})))


if __name__ == "__main__":
    main()
