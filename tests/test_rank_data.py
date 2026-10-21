"""Small independent checks for exact sibling labels and pass-aware features.

The reference minimax uses array move generation, not the research engine's
bitboard routines. No position with more than six empty squares is solved.
"""
from functools import lru_cache
from pathlib import Path
import random
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import build_rank_data as D
import ref_engine as R


def board_from_bits(me, opp):
    board = np.zeros((8, 8), dtype=np.int64)
    for square in D.squares(me):
        board.flat[square] = 1
    for square in D.squares(opp):
        board.flat[square] = -1
    return board


@lru_cache(maxsize=None)
def reference_margin(raw, color):
    board = np.frombuffer(raw, dtype=np.int8).reshape(8, 8)
    moves = R.legal_moves(board, color)
    if not moves:
        if not R.legal_moves(board, -color):
            return int((board == -color).sum() - (board == color).sum())
        return -reference_margin(raw, -color)
    return max(-reference_margin(R.apply_move(board, color, move).astype(np.int8).tobytes(), -color)
               for move in moves)


def sample_states():
    """Get varied boards and a verified forced pass using only the array engine."""
    selected, forced_pass = [], None
    rng = random.Random(671203)
    for game in range(24):
        board, color, placed, passes = R.initial_board(), -1, 0, 0
        while passes < 2:
            moves = R.legal_moves(board, color)
            if not moves:
                if R.legal_moves(board, -color) and forced_pass is None:
                    forced_pass = (board.copy(), color)
                passes += 1
                color = -color
                continue
            passes = 0
            if game < 2 and placed in (12, 29, 54, 56, 58):
                selected.append((board.copy(), color))
            board = R.apply_move(board, color, rng.choice(moves))
            placed += 1
            color = -color
        if game >= 1 and forced_pass is not None:
            break
    assert forced_pass is not None, "Deterministic fixture did not find a forced pass"
    return selected, forced_pass


class RankDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = D.load(ROOT / "versions/v14_round4_fitted_end_ai.py")
        cls.labeler = D.Labeler(cls.A)
        cls.states, cls.forced_pass = sample_states()

    def test_canonical_preserves_player_roles_and_all_symmetries(self):
        # Unequal populations make a role swap observably different under D4.
        me, opp = (1 << 0) | (1 << 11), (1 << 19) | (1 << 38) | (1 << 62)
        board = board_from_bits(me, opp)
        key = D.canonical(me, opp)
        for reflected in (False, True):
            image = np.fliplr(board) if reflected else board
            for turns in range(4):
                transformed = np.rot90(image, turns)
                tm, to = self.A.board_to_bb(transformed, 1)
                self.assertEqual(D.canonical(int(tm), int(to)), key)
        self.assertNotEqual(D.canonical(opp, me), key)

    def test_children_match_independent_move_generation_and_flips(self):
        for board, color in self.states:
            me, opp = self.A.board_to_bb(board, color)
            expected = R.legal_moves(board, color)
            actual = D.squares(self.A.legal_moves(me, opp))
            self.assertEqual(set(actual), {8 * r + c for r, c in expected})
            for move in expected:
                after = R.apply_move(board, color, move)
                cm, co = D.child(self.A, int(me), int(opp), 8 * move[0] + move[1])
                rm, ro = self.A.board_to_bb(after, -color)
                self.assertEqual((cm, co), (int(rm), int(ro)))
                self.assertEqual((cm | co).bit_count(), (int(me) | int(opp)).bit_count() + 1)

    def test_features_reproduce_depth_zero_including_forced_pass_and_terminal(self):
        A = self.A
        tested_pass = False
        fixtures = self.states + [self.forced_pass]
        # Nonfull early terminal: neither side can bracket a line.
        terminal = np.ones((8, 8), dtype=np.int64)
        terminal[0, 0] = 0
        fixtures.append((terminal, -1))
        for board, color in fixtures:
            me, opp = A.board_to_bb(board, color)
            features, terminal, passed = self.labeler.features(int(me), int(opp))
            stats = np.array([0, 1 << 40, 0], dtype=np.int64)
            value = -int(A.alphabeta(me, opp, 0, -A.INF, A.INF, stats,
                                     A.DEFAULT_W, self.labeler.su, self.labeler.si))
            self.assertEqual(stats[2], 0)
            if terminal:
                self.assertFalse(R.legal_moves(board, color))
                self.assertFalse(R.legal_moves(board, -color))
                self.assertEqual(features, [0] * 11)
                self.assertEqual(value, -int(A.terminal_value(me, opp)))
            else:
                empties = int((board == 0).sum())
                self.assertEqual(int(np.dot(features, A.DEFAULT_W[empties])), value)
                self.assertEqual(passed, not bool(R.legal_moves(board, color)))
                tested_pass |= passed
        self.assertTrue(tested_pass)

    def test_complete_small_roots_agree_with_independent_terminal_minimax(self):
        small = [(b, c) for b, c in self.states if int((b == 0).sum()) <= 6][:4]
        self.assertEqual(len(small), 4)
        for index, (board, color) in enumerate(small):
            me, opp = self.A.board_to_bb(board, color)
            margin = reference_margin(board.astype(np.int8).tobytes(), color)
            source = dict(id="tiny_%d" % index, me=int(me), opp=int(opp),
                          empties=int((board == 0).sum()), source_exact_margin=margin)
            record = self.labeler.label(source, time.perf_counter() + 10, 5)
            self.assertTrue(record["complete"])
            self.assertEqual(record["best_margin"], margin)
            for item in record["children"]:
                child_board = R.apply_move(board, color, divmod(item["square"], 8))
                want = -reference_margin(child_board.astype(np.int8).tobytes(), -color)
                self.assertEqual(item["margin"], want)

    def test_expired_deadline_is_unknown_and_cannot_produce_best_margin(self):
        board, color = self.states[-1]
        me, opp = self.A.board_to_bb(board, color)
        source = dict(id="expired", me=int(me), opp=int(opp), empties=int((board == 0).sum()))
        record = self.labeler.label(source, time.perf_counter() - 1, 1)
        self.assertFalse(record["complete"])
        self.assertNotIn("best_margin", record)
        self.assertTrue(all(not item["exact_complete"] and item["margin"] is None
                            for item in record["children"]))

    def test_exact_search_resumes_only_the_same_root(self):
        base = self.A

        class TinySliceEngine:
            """Force actual engine continuation after one visited frame."""
            def __init__(self):
                self.continuations = []

            def __getattr__(self, name):
                return getattr(base, name)

            def alphabeta(self, *args):
                stats = args[5]
                self.continuations.append(int(stats[3]))
                stats[1] = 1
                return base.alphabeta(*args)

        proxy = TinySliceEngine()
        labeler = D.Labeler(proxy)
        board, color = next((b, c) for b, c in self.states if int((b == 0).sum()) == 6)
        me, opp = base.board_to_bb(board, color)
        margin = labeler.exact(int(me), int(opp), time.perf_counter() + 10)
        self.assertEqual(margin, reference_margin(board.astype(np.int8).tobytes(), color))
        self.assertEqual(proxy.continuations[0], 0)
        self.assertIn(1, proxy.continuations)
        before = len(proxy.continuations)
        # The opposite turn is another root, not an allowed continuation of the
        # prior finished search. Shared TT storage does not change that rule.
        other = labeler.exact(int(opp), int(me), time.perf_counter() + 10)
        self.assertEqual(other, reference_margin(board.astype(np.int8).tobytes(), -color))
        self.assertEqual(proxy.continuations[before], 0)


if __name__ == "__main__":
    unittest.main()
