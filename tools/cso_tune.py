"""Offline Competitive Swarm Optimizer for seven normalized evaluation controls.

Independent NumPy implementation of pairwise loser-learning CSO (Cheng/Jin,
doi:10.1109/TCYB.2014.2322602). No optimizer or extra dependency enters ai.py.
Training uses deterministic equal-depth play. Fresh time-limited validation is
performed separately; a training winner is not automatically promoted.
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

COLUMNS = np.array([0, 1, 2, 3, 4, 5, 7])
SCALES = np.array([[30, 20, 12, 5, 3, 8, 6],
                   [30, 20, 12, 5, 3, 8, 6],
                   [30, 20, 12, 3, 8, 4, 6]], dtype=float)


def decode(anchor, coordinates):
    weights = anchor.copy()
    weights[:, COLUMNS] = np.rint(anchor[:, COLUMNS] + SCALES * coordinates).astype(np.int64)
    return weights


def fixed_depth(module, depth, weights=None):
    class FixedDepthAI(module.AI):
        def __init__(self, *args):
            super().__init__(*args)
            if weights is not None:
                self.W = weights.copy()

        def _search(self, me, opp, root_moves, budget):
            order = np.array(sorted(root_moves), dtype=np.int64)
            values = np.full(len(order), -module.INF, dtype=np.int64)
            stats = np.array([0, 1 << 60, 0], dtype=np.int64)
            chosen = int(module.search_root(me, opp, order, len(order), depth,
                                            stats, self.W, values))
            assert chosen >= 0 and stats[2] == 0
            return int(order[chosen])

    return SimpleNamespace(AI=FixedDepthAI, DEBUG_RAISE=True,
                           _import_seconds=getattr(module, "_import_seconds", 0.0))


def train(population=8, generations=6, games=16, depth=5, seconds=120, seed=20261020):
    assert population >= 2 and population % 2 == 0 and games % 2 == 0
    started = time.perf_counter()
    deadline = started + seconds
    mod = arena.load(str(ROOT / "versions/v05_phase1_ai.py"), "cso_candidate")
    old = arena.load(str(ROOT / "versions/v00_ai.py"), "cso_historical")
    anchor = mod.DEFAULT_W.copy()
    opponents = [fixed_depth(mod, depth), fixed_depth(old, depth)]
    rnd = np.random.default_rng(seed)
    positions = np.clip(rnd.normal(0, 0.35, (population, len(COLUMNS))), -1, 1)
    positions[0] = 0
    velocities = np.zeros_like(positions)
    fitness = np.full(population, -np.inf)
    evaluations = []

    def evaluate(coordinates, generation, particle):
        candidate = fixed_depth(mod, depth, decode(anchor, coordinates))
        scores = []
        for game in range(games):
            if time.perf_counter() > deadline:
                return None
            color = -1 if game % 2 == 0 else 1
            opponent = opponents[(game // 2) % len(opponents)]
            black, white = (candidate, opponent) if color == -1 else (opponent, candidate)
            winner, reason, _ = arena.play(black, white, 5, opening_plies=6,
                                           seed=110000 + game // 2)
            assert reason == "normal", reason
            scores.append(0.5 if winner == 0 else float(winner == color))
        row = dict(generation=generation, particle=particle, coordinates=coordinates.tolist(),
                   weights=decode(anchor, coordinates).tolist(), score=float(np.mean(scores)),
                   wins=scores.count(1), draws=scores.count(0.5), losses=scores.count(0))
        evaluations.append(row)
        print("CSO evaluation", len(evaluations), "generation", generation,
              "particle", particle, "W/D/L", row["wins"], row["draws"], row["losses"], flush=True)
        return row["score"]

    for i in range(population):
        value = evaluate(positions[i], 0, i)
        if value is None:
            break
        fitness[i] = value
    for generation in range(1, generations + 1):
        if not np.isfinite(fitness).all() or time.perf_counter() > deadline:
            break
        mean = positions.mean(axis=0)
        for a, b in rnd.permutation(population).reshape(-1, 2):
            winner, loser = (a, b) if fitness[a] >= fitness[b] else (b, a)
            r1, r2, r3 = rnd.random((3, len(COLUMNS)))
            velocities[loser] = (r1 * velocities[loser] +
                r2 * (positions[winner] - positions[loser]) + 0.1 * r3 * (mean - positions[loser]))
            proposed = positions[loser] + velocities[loser]
            positions[loser] = np.clip(proposed, -1, 1)
            velocities[loser][proposed != positions[loser]] = 0
            value = evaluate(positions[loser], generation, int(loser))
            if value is None:
                break
            fitness[loser] = value
    best = max(evaluations, key=lambda r: (r["score"], -sum(x*x for x in r["coordinates"])))
    result = dict(seed=seed, depth=depth, training_games_per_evaluation=games,
                  population=population, requested_generations=generations,
                  seconds=time.perf_counter()-started, evaluations=evaluations,
                  best=best, anchor=anchor.tolist(), optimized_columns=COLUMNS.tolist(),
                  training_opening_seeds=list(range(110000, 110000 + games // 2)),
                  note="Exploratory fixed-depth training; held-out timed matches required.")
    return np.array(best["weights"], dtype=np.int64), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--generations", type=int, default=6)
    args = ap.parse_args()
    weights, result = train(seconds=args.seconds, generations=args.generations)
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    np.save(folder / "round2_cso_weights.npy", weights)
    (folder / "round2_cso_training.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Training best:", result["best"], "seconds", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
