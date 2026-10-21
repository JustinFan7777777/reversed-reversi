"""Synthetic model checks; no game-engine imports, JIT or search required."""
import copy
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import rank_model as R


def child(square, x=0, margin=0, terminal=False, passed=False):
    features = [0] * 11
    features[0] = x
    return dict(square=square, features=features, margin=margin, terminal=terminal, **{"pass": passed})


def root(ident, children, split="train", game=None, complete=True, empties=18):
    return dict(id=ident, game=ident if game is None else game, split=split,
                empties=empties, complete=complete, children=children)


class MoveRankTests(unittest.TestCase):
    def test_centering_removes_per_root_offsets_and_keeps_unsupported_anchor(self):
        roots = [root("a", [child(0, -2, -2), child(1, 0, 0), child(2, 2, 2)]),
                 root("b", [child(3, -3, -3), child(4, 1, 1)])]
        shifted = copy.deepcopy(roots)
        for i, item in enumerate(shifted):
            for c in item["children"]:
                c["margin"] += [20, -20][i]
                c["features"][1] += [11, -11][i]
        baseline, offset = R.fit_model(roots), R.fit_model(shifted)
        self.assertEqual(baseline["row"], offset["row"])
        self.assertEqual(baseline["row"][0], 8)
        self.assertEqual(baseline["row"][1:], R.ANCHOR.tolist()[1:])
        self.assertEqual(baseline["free_feature_rank"], 1)
        self.assertNotEqual(R.fit_model(shifted, "absolute")["row"], offset["row"])

    def test_root_weight_is_independent_of_sibling_duplication(self):
        roots = [root("a", [child(0, -1, -1), child(1, 1, 1)]),
                 root("b", [child(2, -1, -2), child(3, 1, 2)])]
        duplicated = copy.deepcopy(roots)
        duplicated[0]["children"] += [child(4, -1, -1), child(5, 1, 1)]
        self.assertEqual(R.fit_model(roots)["row"], R.fit_model(duplicated)["row"])

    def test_zero_information_and_nonfinite_schema(self):
        roots = [root("same", [child(0, 1, 2), child(1, 1, -2)])]
        model = R.fit_model(roots)
        self.assertEqual(model["row"], R.ANCHOR.tolist())
        self.assertEqual(model["ridge_lambda"], 0)
        self.assertEqual(model["free_feature_rank"], 0)
        broken = copy.deepcopy(roots)
        broken[0]["children"][0]["features"][2] = float("nan")
        with self.assertRaises(ValueError):
            R.fit_model(broken)

    def test_complete_children_and_split_validation(self):
        good = root("a", [child(0), child(1)])
        bad = copy.deepcopy(good)
        bad["legal_count"] = 3
        with self.assertRaises(ValueError):
            R.validate_roots([bad])
        duplicate = copy.deepcopy(good)
        duplicate["children"][1]["square"] = 0
        with self.assertRaises(ValueError):
            R.validate_roots([duplicate])
        other = root("b", [child(2)], "holdout", game="a")
        with self.assertRaises(ValueError):
            R.validate_roots([good, other])
        bad = copy.deepcopy(good)
        bad["children"][0]["complete"] = False
        with self.assertRaises(ValueError):
            R.validate_roots([bad])

    def test_incomplete_terminal_and_holdout_do_not_train(self):
        train = root("train", [child(0, -1, -1), child(1, 1, 1)])
        excluded = [root("incomplete", [child(2, -1, 20), child(3, 1, -20)], complete=False),
                    root("terminal", [child(4, -1, 20, terminal=True), child(5, 1, -20)]),
                    root("holdout", [child(6, -1, 20), child(7, 1, -20)], split="holdout"),
                    root("wrong-row", [child(8, -1, 20), child(9, 1, -20)], empties=21)]
        self.assertEqual(R.fit_model([train])["row"], R.fit_model([train] + excluded)["row"])
        self.assertEqual(R.fit_model([train] + excluded)["training_ids"], ["train"])

    def test_censored_unknown_child_is_preserved_and_never_used(self):
        train = root("train", [child(0, -1, -1), child(1, 1, 1)])
        censored = root("censored", [child(2, -1, -1), child(3, 1, 1)], complete=False)
        censored["children"][1]["margin"] = None
        censored["children"][1]["exact_complete"] = False
        normalized = R.validate_roots([censored])[0]
        self.assertIsNone(normalized["children"][1]["margin"])
        self.assertFalse(normalized["children"][1]["exact_complete"])
        self.assertEqual(R.fit_model([train, censored])["row"], R.fit_model([train])["row"])
        self.assertEqual(R.evaluate([train, censored], R.ANCHOR)["roots"], 1)
        self.assertEqual(R.run_experiment([train, censored], bootstrap_samples=20)["eligible_roots"], 1)
        broken = copy.deepcopy(censored)
        broken["complete"] = True
        with self.assertRaises(ValueError):
            R.validate_roots([broken])
        broken = copy.deepcopy(censored)
        del broken["children"][1]["exact_complete"]
        with self.assertRaises(ValueError):
            R.validate_roots([broken])
        broken = copy.deepcopy(censored)
        broken["complete"] = True
        broken["children"][1]["margin"] = 1
        with self.assertRaises(ValueError):
            R.validate_roots([broken])

    def test_pass_is_already_root_perspective_and_input_is_unchanged(self):
        roots = [root("pass", [child(0, -1, -1), child(1, 1, 1, passed=True)])]
        before = copy.deepcopy(roots)
        model = R.fit_model(roots)
        metrics = R.evaluate(roots, model["row"])
        self.assertEqual(metrics["records"][0]["chosen_square"], 1)
        self.assertEqual(roots, before)

    def test_ties_pairwise_denominators_and_wdl_regret(self):
        roots = [root("tie", [child(2, 0, 1), child(1, 0, -1), child(3, 0, 1)])]
        metrics = R.evaluate(roots, np.zeros(11, dtype=np.int64))
        self.assertEqual(metrics["records"][0]["chosen_square"], 1)
        self.assertEqual(metrics["mean_margin_regret"], 2)
        self.assertEqual(metrics["wdl_errors"], 1)
        self.assertEqual(metrics["pairwise_count"], 2)
        self.assertEqual(metrics["pairwise_predicted_ties"], 2)
        self.assertEqual(metrics["pairwise_accuracy"], .5)
        self.assertEqual(metrics["pairwise_strict_mistakes"], 0)

    def test_terminal_priority_preserves_search_encoding(self):
        roots = [root("terminal-win", [child(0, 100, 20), child(1, 0, 1, terminal=True)]),
                 root("terminal-loss", [child(2, -100, -20), child(3, 0, -1, terminal=True)])]
        row = np.zeros(11, dtype=np.int64)
        row[0] = 8
        metrics = R.evaluate(roots, row)
        self.assertEqual([r["chosen_square"] for r in metrics["records"]], [1, 2])
        self.assertEqual(R.terminal_score(1), 100001)
        self.assertEqual(R.terminal_score(-1), -100001)
        self.assertEqual(R.terminal_score(0), 0)
        self.assertEqual(metrics["wdl_errors"], 0)

    def test_bootstrap_is_paired_and_deterministic(self):
        baseline = [dict(id="a", regret=4), dict(id="b", regret=2)]
        candidate = [dict(id="b", regret=0), dict(id="a", regret=2)]
        one = R.paired_regret_interval(candidate, baseline, samples=100, seed=42)
        self.assertEqual(one, R.paired_regret_interval(candidate, baseline, samples=100, seed=42))
        self.assertEqual(one["mean_difference"], -2)
        self.assertEqual(one["ci95"], [-2, -2])
        with self.assertRaises(ValueError):
            R.paired_regret_interval(candidate[:1], baseline)

    def test_bootstrap_keeps_all_roots_from_a_game_together(self):
        baseline = [dict(id="a", game="one", regret=4), dict(id="b", game="one", regret=4)]
        candidate = [dict(id="a", game="one", regret=0), dict(id="b", game="one", regret=4)]
        interval = R.paired_regret_interval(candidate, baseline, samples=100)
        self.assertEqual(interval["games"], 1)
        self.assertEqual(interval["resampling_unit"], "game")
        self.assertEqual(interval["ci95"], [-2, -2])

    def test_full_report_integerizes_and_groups_empties(self):
        roots = [root("train", [child(0, -1, -1), child(1, 1, 1)]),
                 root("holdout", [child(2, -1, -1), child(3, 1, 1)], split="holdout", empties=20)]
        report = R.run_experiment(roots, bootstrap_samples=30)
        self.assertIsInstance(report["models"]["sibling_centered"]["row"][0], int)
        for engine in report["metrics"]["holdout"]["engines"].values():
            self.assertEqual(engine["roots"], 1)
            self.assertEqual(set(engine["per_parent_empties"]), {"20"})


if __name__ == "__main__":
    unittest.main()
