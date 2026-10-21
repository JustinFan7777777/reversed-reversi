"""Fit and audit an exact-label sibling-ranking hypothesis without AI searches.

Input features and margins must already use the root player's perspective.  A
completed root is the labeling tool's assertion that *all* legal children were
solved.  Optional ``legal_count`` makes accidental omissions detectable here too.
Run ``python tools/rank_model.py --data roots.json --output report.json``.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ANCHOR = np.array([70, -2, 2, 2, 2, 7, 0, -2, 0, 0, 0], dtype=np.int64)
FREE_COLUMNS = np.array([i for i in range(11) if i != 9], dtype=np.int64)
WIN_BASE = 100_000
CHILD_EMPTIES = (17, 18, 19)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % name)
    if not np.iinfo(np.int64).min <= int(value) <= np.iinfo(np.int64).max:
        raise ValueError("%s exceeds int64" % name)
    return int(value)


def validate_roots(roots):
    """Validate and copy JSON-shaped data; never modify the caller's records."""
    if not isinstance(roots, list):
        raise ValueError("Input must be a list of roots")
    result, seen_ids, game_splits = [], set(), {}
    for root in roots:
        if not isinstance(root, dict):
            raise ValueError("Each root must be an object")
        ident = root.get("id")
        if not isinstance(ident, str) or not ident or ident in seen_ids:
            raise ValueError("Root ids must be unique nonempty strings")
        seen_ids.add(ident)
        split = root.get("split")
        if split not in ("train", "holdout"):
            raise ValueError("split must be train or holdout")
        game = root.get("game")
        if isinstance(game, bool) or not isinstance(game, (str, int)):
            raise ValueError("game must be a string or integer")
        # JSON integer and string spellings of the same game share a split.
        game_key = str(game)
        if game_key in game_splits and game_splits[game_key] != split:
            raise ValueError("A game cannot cross training and holdout splits")
        game_splits[game_key] = split
        empties = _integer(root.get("empties"), "empties")
        if not 1 <= empties <= 60:
            raise ValueError("Root empties must be in 1..60")
        if not isinstance(root.get("complete"), bool):
            raise ValueError("complete must be a boolean")
        children = root.get("children")
        if not isinstance(children, list) or (root["complete"] and not children):
            raise ValueError("Completed roots need a nonempty children list")
        out_children, squares = [], set()
        for child in children:
            if not isinstance(child, dict):
                raise ValueError("Each child must be an object")
            square = _integer(child.get("square"), "square")
            if not 0 <= square < 64 or square in squares:
                raise ValueError("Child squares must be unique and in 0..63")
            squares.add(square)
            features = child.get("features")
            if not isinstance(features, (list, tuple)) or len(features) != 11:
                raise ValueError("Each child needs exactly 11 integer features")
            features = [_integer(v, "feature") for v in features]
            completion_flags = []
            for key in ("complete", "exact_complete"):
                if key in child:
                    if not isinstance(child[key], bool):
                        raise ValueError("Child completion flags must be booleans")
                    completion_flags.append(child[key])
            child_complete = all(completion_flags)
            if root["complete"] and not child_complete:
                raise ValueError("A completed root cannot contain an unresolved child")
            if child.get("margin") is None:
                if root["complete"] or child.get("exact_complete") is not False:
                    raise ValueError("Unknown margins need an incomplete root and exact_complete=false")
                margin = None
            else:
                margin = _integer(child.get("margin"), "margin")
                if not -64 <= margin <= 64:
                    raise ValueError("Exact terminal margin must be in -64..64")
            if not isinstance(child.get("terminal"), bool) or not isinstance(child.get("pass"), bool):
                raise ValueError("terminal and pass must be booleans")
            if child["terminal"] and child["pass"]:
                raise ValueError("A terminal child cannot also be a single-pass child")
            out_children.append(dict(square=square, features=features, margin=margin,
                                     terminal=child["terminal"], exact_complete=child_complete,
                                     **{"pass": child["pass"]}))
        for count_key in ("legal_count", "n_legal"):
            if count_key in root:
                expected = _integer(root[count_key], count_key)
                if expected < len(children) or (root["complete"] and expected != len(children)):
                    raise ValueError("Completed root must contain every legal child")
        result.append(dict(id=ident, split=split, game=game, empties=empties,
                           complete=root["complete"], children=sorted(out_children, key=lambda c: c["square"])))
    return result


def _eligible(root):
    return root["complete"] and root["empties"] - 1 in CHILD_EMPTIES


def _row(value):
    array = np.asarray(value)
    if array.shape != (11,) or not np.isfinite(array.astype(float)).all():
        raise ValueError("A model row needs 11 finite numbers")
    # Scoring and reported metrics always use the deployed integer coefficients.
    rounded = np.rint(array.astype(float))
    if (rounded < -(2.0 ** 63)).any() or (rounded >= 2.0 ** 63).any():
        raise ValueError("Model coefficients exceed int64")
    return rounded.astype(np.int64)


def fit_model(roots, objective="centered", anchor=ANCHOR):
    """Fit one row, training on complete roots only, with root total weight one.

    Centered least squares is equivalent to within-root pairwise squared error.
    The parity coefficient is fixed, since ordinary sibling differences cancel it.
    Rank-deficient and zero-information features preserve the ridge anchor.
    """
    roots = validate_roots(roots)
    if objective not in ("centered", "absolute"):
        raise ValueError("objective must be centered or absolute")
    anchor = _row(anchor)
    x_blocks, y_blocks, used = [], [], []
    for root in roots:
        if root["split"] != "train" or not _eligible(root):
            continue
        children = [c for c in root["children"] if not c["terminal"]]
        if len(children) < 2:
            continue
        x = np.asarray([c["features"] for c in children], dtype=float)
        y = 8.0 * np.asarray([c["margin"] for c in children], dtype=float)
        # Remove the fixed parity term before fitting the other ten columns.
        y -= x[:, 9] * anchor[9]
        x = x[:, FREE_COLUMNS]
        if objective == "centered":
            x, y = x - x.mean(axis=0), y - y.mean()
        scale = np.sqrt(len(children))
        x_blocks.append(x / scale)
        y_blocks.append(y / scale)
        used.append(root["id"])
    if not used:
        raise ValueError("No eligible training roots with two nonterminal children")
    x, y = np.vstack(x_blocks), np.concatenate(y_blocks)
    gram = x.T @ x
    penalty = 1e-4 * float(np.trace(gram)) / len(FREE_COLUMNS)
    if not np.isfinite(gram).all() or not np.isfinite(y).all() or not np.isfinite(penalty):
        raise ValueError("Nonfinite fitting matrix")
    fitted = anchor.astype(float)
    if penalty > 0:
        base = anchor[FREE_COLUMNS].astype(float)
        # Solve a correction around the anchor, retaining unsupported directions.
        correction = np.linalg.solve(gram + penalty * np.eye(len(FREE_COLUMNS)),
                                     x.T @ (y - x @ base))
        fitted[FREE_COLUMNS] = base + correction
    if not np.isfinite(fitted).all():
        raise ValueError("Fit produced nonfinite coefficients")
    integer = _row(fitted)
    return dict(objective=objective, row=integer.tolist(), unrounded_row=fitted.tolist(),
                anchor=anchor.tolist(), fixed_columns=[9], ridge_lambda=penalty,
                free_feature_rank=int(np.linalg.matrix_rank(x)),
                train_roots=len(used), train_nonterminal_children=len(y),
                training_ids=used, child_empties=list(CHILD_EMPTIES))


def terminal_score(margin):
    """Encode an exact root-perspective margin as the search's terminal score."""
    return WIN_BASE + margin if margin > 0 else -WIN_BASE + margin if margin < 0 else 0


def _summarize(records):
    n = len(records)
    pairs = sum(r["pairwise_count"] for r in records)
    correct = sum(r["pairwise_correct"] for r in records)
    ties = sum(r["pairwise_predicted_ties"] for r in records)
    return dict(roots=n, wdl_errors=sum(r["wdl_error"] for r in records),
                wdl_error_rate=sum(r["wdl_error"] for r in records) / n if n else None,
                mean_margin_regret=float(np.mean([r["regret"] for r in records])) if n else None,
                optimum_hits=sum(r["regret"] == 0 for r in records),
                optimum_hit_rate=sum(r["regret"] == 0 for r in records) / n if n else None,
                pairwise_count=pairs, pairwise_correct=correct,
                pairwise_strict_mistakes=sum(r["pairwise_strict_mistakes"] for r in records),
                pairwise_predicted_ties=ties,
                pairwise_accuracy=(correct + .5 * ties) / pairs if pairs else None)


def evaluate(roots, row):
    """Measure integer static move choices against all exact child outcomes."""
    roots, row = validate_roots(roots), _row(row)
    records = []
    for root in roots:
        if not _eligible(root):
            continue
        children = root["children"]
        predictions = [terminal_score(c["margin"]) if c["terminal"] else
                       sum(int(w) * f for w, f in zip(row, c["features"])) for c in children]
        # Children were square-sorted by validation, so argmax breaks ties by square.
        chosen_index = max(range(len(children)), key=lambda i: predictions[i])
        chosen, best = children[chosen_index], max(c["margin"] for c in children)
        counts = dict(pairwise_count=0, pairwise_correct=0,
                      pairwise_strict_mistakes=0, pairwise_predicted_ties=0)
        for i, left in enumerate(children):
            for j in range(i + 1, len(children)):
                delta = left["margin"] - children[j]["margin"]
                if delta == 0:
                    continue
                counts["pairwise_count"] += 1
                predicted_delta = predictions[i] - predictions[j]
                key = ("pairwise_predicted_ties" if predicted_delta == 0 else
                       "pairwise_correct" if predicted_delta * delta > 0 else "pairwise_strict_mistakes")
                counts[key] += 1
        records.append(dict(id=root["id"], game=root["game"], split=root["split"],
                            parent_empties=root["empties"], chosen_square=chosen["square"],
                            chosen_margin=chosen["margin"], best_margin=best,
                            regret=best - chosen["margin"],
                            wdl_error=bool(np.sign(chosen["margin"]) < np.sign(best)),
                            optimal_squares=[c["square"] for c in children if c["margin"] == best],
                            **counts))
    summary = _summarize(records)
    summary["per_parent_empties"] = {
        str(e): _summarize([r for r in records if r["parent_empties"] == e])
        for e in sorted({r["parent_empties"] for r in records})}
    summary["records"] = records
    return summary


def paired_regret_interval(candidate_records, baseline_records, samples=2000, seed=20261021):
    """Paired game-cluster bootstrap of candidate minus baseline margin regret.

    Pairing remains per root, while all roots from a sampled game travel together.
    Standalone metric records without ``game`` use their root id as the cluster.
    """
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    baseline = {r["id"]: r for r in baseline_records}
    candidate_ids = {r["id"] for r in candidate_records}
    if (len(baseline) != len(baseline_records) or len(candidate_ids) != len(candidate_records)
            or set(baseline) != candidate_ids):
        raise ValueError("Paired bootstrap needs exactly matching unique roots")
    diffs = np.asarray([r["regret"] - baseline[r["id"]]["regret"] for r in candidate_records], dtype=float)
    clusters = {}
    for index, record in enumerate(candidate_records):
        other = baseline[record["id"]]
        if "game" in record and "game" in other and str(record["game"]) != str(other["game"]):
            raise ValueError("Paired roots must belong to the same game")
        key = str(record.get("game", other.get("game", record["id"])))
        clusters.setdefault(key, []).append(index)
    if not len(diffs):
        return dict(mean_difference=None, ci95=None, roots=0, samples=samples, seed=seed,
                    games=0, resampling_unit="game", negative_favors_candidate=True)
    groups = [np.asarray(indices, dtype=np.int64) for indices in clusters.values()]
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for i in range(samples):
        selected = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[j] for j in selected])
        boot[i] = diffs[indices].mean()
    return dict(mean_difference=float(diffs.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                roots=len(diffs), games=len(groups), samples=samples, seed=seed,
                resampling_unit="game", negative_favors_candidate=True)


def compare_models(roots, rows, bootstrap_samples=2000, seed=20261021):
    roots = validate_roots(roots)
    if "v14" not in rows:
        raise ValueError("rows must include the v14 baseline")
    result = {}
    for split in ("train", "holdout"):
        subset = [r for r in roots if r["split"] == split]
        engines = {name: evaluate(subset, row) for name, row in rows.items()}
        intervals = {name: paired_regret_interval(metric["records"], engines["v14"]["records"],
                                                  samples=bootstrap_samples, seed=seed)
                     for name, metric in engines.items() if name != "v14"}
        result[split] = dict(engines=engines, paired_margin_regret_vs_v14=intervals)
    return result


def run_experiment(roots, bootstrap_samples=2000, seed=20261021):
    roots = validate_roots(roots)
    centered = fit_model(roots, "centered")
    absolute = fit_model(roots, "absolute")
    rows = dict(v14=ANCHOR.tolist(), sibling_centered=centered["row"], absolute_ridge=absolute["row"])
    return dict(schema_version=1, target="Eight times exact final opp-minus-own margin, root perspective",
                feature_convention="Eleven root-perspective features supplied by label generator; passes already normalized",
                completeness="Generator asserts all legal children solved; incomplete roots are excluded",
                promotion="Offline diagnostic only; independent fresh-opening matches decide adoption",
                input_roots=len(roots), complete_roots=sum(r["complete"] for r in roots),
                eligible_roots=sum(_eligible(r) for r in roots),
                models=dict(sibling_centered=centered, absolute_ridge=absolute),
                metrics=compare_models(roots, rows, bootstrap_samples=bootstrap_samples, seed=seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    raw = args.data.read_bytes()
    data = json.loads(raw)
    roots = data["roots"] if isinstance(data, dict) else data
    report = run_experiment(roots, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    report["input_path"], report["input_sha256"] = str(args.data), hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": {name: model["row"] for name, model in report["models"].items()},
                      "holdout": {name: {k: v for k, v in metric.items() if k not in ("records", "per_parent_empties")}
                                  for name, metric in report["metrics"]["holdout"]["engines"].items()}}, indent=2))


if __name__ == "__main__":
    main()
