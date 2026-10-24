"""Create bounded, all-sibling exact labels for move-ranking research.

One root per game keeps the statistical unit explicit. Training roots reuse only
old training games; holdout games use new fixed-depth policies and seeds. A parent
is usable only when every legal child is solved. Timed-out parents remain unknown.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location("ranking_frozen_engine", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.DEBUG_RAISE = True
    engine.AI(8, -1, 5)
    return engine


def squares(mask):
    mask = int(mask)
    return [i for i in range(64) if (mask >> i) & 1]


def child(A, me, opp, square):
    move = np.uint64(1 << square)
    flipped = np.uint64(A.flips(np.uint64(me), np.uint64(opp), move))
    return int(np.uint64(opp) & ~flipped), int(np.uint64(me) | move | flipped)


def canonical(me, opp):
    """D4 symmetry key, preserving player roles and turn orientation."""
    variants = []
    for reflected in (False, True):
        for rotations in range(4):
            masks = []
            for mask in (me, opp):
                out = 0
                for sq in squares(mask):
                    row, col = divmod(sq, 8)
                    if reflected:
                        col = 7 - col
                    for _ in range(rotations):
                        row, col = col, 7 - row
                    out |= 1 << (8 * row + col)
                masks.append(out)
            variants.append(tuple(masks))
    return min(variants)


def family_keys(A, me, opp):
    return {canonical(me, opp)} | {
        canonical(*child(A, me, opp, sq))
        for sq in squares(A.legal_moves(np.uint64(me), np.uint64(opp)))
    }


class Labeler:
    def __init__(self, engine):
        self.A = engine
        self.su = np.zeros((engine.MAXPLY, engine.SU_COLS), np.uint64)
        self.si = np.zeros((engine.MAXPLY, engine.SI_COLS), np.int64)
        self.tt = np.zeros((engine.TT_SIZE, 6), np.uint64)
        self.units = [np.tile(np.eye(11, dtype=np.int64)[i], (3, 1)) for i in range(11)]
        self.nodes = 0

    def exact(self, me, opp, deadline):
        A = self.A
        stats = np.zeros(7, np.int64)
        empties = 64 - (me | opp).bit_count()
        while time.perf_counter() < deadline:
            stats[0:3] = (0, 4096, 0)
            value = int(A.alphabeta(np.uint64(me), np.uint64(opp), empties,
                                   -A.INF, A.INF, stats, A.DEFAULT_W,
                                   self.su, self.si, self.tt))
            self.nodes += int(stats[0])
            if stats[2] == 0:
                assert value == 0 or A.WIN_BASE < abs(value) <= A.WIN_BASE + 64
                return value - A.WIN_BASE if value > 0 else value + A.WIN_BASE if value < 0 else 0
            assert stats[2] == 2, "Unexpected exact-search abort"
        return None

    def features(self, me, opp):
        """Root-perspective features for a child, matching depth-zero pass rules."""
        A = self.A
        moves = np.uint64(A.legal_moves(np.uint64(me), np.uint64(opp)))
        passed = not bool(moves)
        if passed:
            other_moves = np.uint64(A.legal_moves(np.uint64(opp), np.uint64(me)))
            if not other_moves:
                return [0] * 11, True, False
            me, opp, moves, sign = opp, me, other_moves, 1
        else:
            sign = -1
        values = [sign * int(A.evaluate(np.uint64(me), np.uint64(opp), moves, W))
                  for W in self.units]
        return values, False, passed

    def label(self, source, deadline, child_seconds):
        A = self.A
        me, opp = source["me"], source["opp"]
        options = squares(A.legal_moves(np.uint64(me), np.uint64(opp)))
        assert options
        result = dict(source, legal_count=len(options), complete=False, children=[])
        started, initial_nodes = time.perf_counter(), self.nodes
        self.tt.fill(0)
        for sq in options:
            cm, co = child(A, me, opp, sq)
            began, nodes_before = time.perf_counter(), self.nodes
            features, terminal, passed = self.features(cm, co)
            value = self.exact(cm, co, min(deadline, began + child_seconds))
            result["children"].append(dict(
                square=sq, child_me=cm, child_opp=co, features=features,
                terminal=terminal, **{"pass": passed}, exact_complete=value is not None,
                margin=-value if value is not None else None,
                nodes=self.nodes - nodes_before, seconds=time.perf_counter() - began))
            if value is None:
                break
        result["complete"] = (len(result["children"]) == len(options)
                              and all(c["exact_complete"] for c in result["children"]))
        if result["complete"]:
            result["best_margin"] = max(c["margin"] for c in result["children"])
            if "source_exact_margin" in result:
                assert result["best_margin"] == result["source_exact_margin"], result["id"]
        result.update(seconds=time.perf_counter() - started, nodes=self.nodes - initial_nodes)
        return result


def training_sources(data, count, seed):
    files = sorted(data.glob("mid_w*.json"))
    assert files
    rows = [r for f in files for r in json.loads(f.read_text()) if r["split"] == "train"]
    lookup = {(r["game"], r["empties"]): r for r in rows if 18 <= r["empties"] <= 20}
    games = sorted({r["game"] for r in rows})
    random.Random(seed).shuffle(games)
    out = []
    for game in games:
        empties = 18 + len(out) % 3
        r = lookup.get((game, empties))
        if r is None:
            continue
        source = dict(id="old_%d_e%d" % (game, empties), game=game,
                      split="train", empties=empties, me=r["me"], opp=r["opp"])
        if "margin" in r:
            source["source_exact_margin"] = r["margin"]
        out.append(source)
        if len(out) == count:
            break
    assert len(out) == count
    return out, {f.name: sha(f) for f in files}


def fresh_source(A, index, seed, deadline):
    """Generate a single deterministic-policy game up to the selected parent."""
    rng = random.Random(seed)
    me, opp = (1 << 35) | (1 << 28), (1 << 27) | (1 << 36)
    target, ply, passes = 18 + index % 3, 0, 0
    su = np.zeros((A.MAXPLY, A.SU_COLS), np.uint64)
    si = np.zeros((A.MAXPLY, A.SI_COLS), np.int64)
    tt = np.zeros((A.TT_SIZE, 6), np.uint64)
    while time.perf_counter() < deadline and passes < 2:
        options = squares(A.legal_moves(np.uint64(me), np.uint64(opp)))
        if not options:
            passes += 1
            me, opp = opp, me
            continue
        passes = 0
        empties = 64 - (me | opp).bit_count()
        if empties == target:
            return dict(id="fresh_%d_e%d" % (seed, target), game="fresh_%d" % seed,
                        seed=seed, split="holdout", empties=empties, me=me, opp=opp)
        if ply < 8 or rng.random() < 0.15:
            square = rng.choice(options)
        else:
            best, square = -A.INF, options[0]
            for sq in options:
                cm, co = child(A, me, opp, sq)
                stats = np.zeros(7, np.int64)
                while time.perf_counter() < deadline:
                    stats[0:3] = (0, 4096, 0)
                    value = -int(A.alphabeta(np.uint64(cm), np.uint64(co), 4,
                                            -A.INF, -best, stats, A.DEFAULT_W, su, si, tt))
                    if stats[2] == 0:
                        break
                    assert stats[2] == 2
                else:
                    return None
                if value > best:
                    best, square = value, sq
        me, opp = child(A, me, opp, square)
        ply += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "holdout"))
    ap.add_argument("--engine", type=Path, default=ROOT / "versions/v14_round4_fitted_end_ai.py")
    ap.add_argument("--data", type=Path, default=ROOT / "artifacts/codex_round5/claude_round5_inputs")
    ap.add_argument("--count", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--child-seconds", type=float, default=1.0)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--frozen-model", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    count = args.count or (120 if args.mode == "train" else 60)
    seed = args.seed or (10000001 if args.mode == "train" else 10010000)
    budget = args.seconds if args.seconds is not None else (240 if args.mode == "train" else 150)
    assert count > 0 and budget > 0 and args.child_seconds > 0
    if args.mode == "holdout":
        assert args.frozen_model and args.frozen_model.is_file() and args.exclude
    started = time.perf_counter()
    deadline = started + budget
    A = load(args.engine)
    labeler = Labeler(A)
    exclusions = set()
    if args.exclude:
        previous = json.loads(args.exclude.read_text())
        for r in previous["roots"]:
            exclusions.update(family_keys(A, r["me"], r["opp"]))
    sources, data_hashes = training_sources(args.data, count, seed) if args.mode == "train" else ([], {})
    report = dict(manifest=dict(
        schema_version=1, mode=args.mode, engine=str(args.engine), engine_sha256=sha(args.engine),
        generator_sha256=sha(__file__), protocol_sha256=sha(ROOT / "docs/RESEARCH_STATUS.md"),
        source_data_sha256=data_hashes, requested_roots=count, seed=seed,
        budget_seconds=budget, child_seconds=args.child_seconds,
        frozen_model_sha256=sha(args.frozen_model) if args.frozen_model else None,
        exclusion_dataset_sha256=sha(args.exclude) if args.exclude else None,
        fresh_policy="eight random plies; then depth5 v14 with 15% random moves"),
        roots=[], skipped=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["summary"] = dict(
            attempted=len(report["roots"]), complete=sum(r["complete"] for r in report["roots"]),
            completion_by_empties={str(e): dict(Counter(
                "complete" if r["complete"] else "incomplete" for r in report["roots"] if r["empties"] == e))
                for e in (18, 19, 20)}, seconds=time.perf_counter() - started, nodes=labeler.nodes,
            complete_children=sum(len(r["children"]) for r in report["roots"] if r["complete"]))
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for index in range(count):
        if time.perf_counter() >= deadline:
            break
        source = sources[index] if sources else fresh_source(A, index, seed + index, deadline)
        if source is None:
            report["skipped"].append(dict(index=index, reason="generation_ended_or_timed_out"))
            continue
        keys = family_keys(A, source["me"], source["opp"])
        if exclusions & keys:
            report["skipped"].append(dict(id=source["id"], reason="symmetry_family_overlap"))
            continue
        # Holdout roots also need to be independent of earlier holdout families.
        exclusions.update(keys)
        result = labeler.label(source, deadline, args.child_seconds)
        report["roots"].append(result)
        save()
        if (index + 1) % 10 == 0:
            print(args.mode, index + 1, "roots", report["summary"], flush=True)
    save()
    assert sha(args.engine) == report["manifest"]["engine_sha256"], "Engine changed during labeling"
    if args.frozen_model:
        assert sha(args.frozen_model) == report["manifest"]["frozen_model_sha256"], "Model changed during holdout"
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
