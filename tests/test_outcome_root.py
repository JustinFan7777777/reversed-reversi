"""Round 7 root driver (v17 candidate): outcome-first solve and partial iterations.

  * Outcome solve: on random positions with 8-14 empties, _outcome_root (no deadline)
    chooses a move with the best exact outcome (oracle: a full-window exact solve of
    every child on a fresh table), reports every_move_lost exactly when every move
    loses, and last_outcome is the chosen move's exact outcome.
  * _search end to end (generous budget): the chosen move has the best outcome; when
    every move loses it also has the exact least loss.
  * Deadlines: a fake clock expires the search at every clock read in turn (8-12
    empties, 4-node slices); every returned move is legal and every reported
    outcome is the chosen move's exact outcome.
  * Scripted cut points: child values come from a script, so every fallback rule is
    hit exactly -- a move that failed high or finished higher in a cut iteration is
    kept (and never when PARTIAL_ITERATION is off); in the outcome solve a proven win
    stops it, a proven draw is kept when the deadline cuts a later test, the first
    move not proven to lose replaces proven losses, and when every move loses the
    exact iteration chooses the least loss.

    python tests/test_outcome_root.py [path/to/ai.py]
"""
import importlib.util
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import test_search as T                 # noqa: E402
from test_pvs_inline import FakeClock   # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location("t7_outcome_root", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DEBUG_RAISE = True
    return mod


def children(X, me, opp, moves):
    out = {}
    for sq in moves:
        bit = np.uint64(1 << sq)
        f = np.uint64(X.flips(me, opp, bit))
        out[sq] = (np.uint64(opp & ~f), np.uint64(me | bit | f))
    return out


def exact_values(X, me, opp, moves):
    """Exact value of every move from the mover's side (full window, fresh table)."""
    su = np.zeros((X.MAXPLY, X.SU_COLS), dtype=np.uint64)
    si = np.zeros((X.MAXPLY, X.SI_COLS), dtype=np.int64)
    table = np.zeros((X.TT_SIZE, 6), dtype=np.uint64)
    empties = 64 - int(X.popcnt(me | opp))
    vals = {}
    for sq, (cme, cop) in children(X, me, opp, moves).items():
        stats = np.zeros(7, dtype=np.int64)
        stats[1] = 1 << 60
        vals[sq] = -int(X.alphabeta(cme, cop, empties - 1, -X.INF, X.INF, stats,
                                    X.DEFAULT_W, su, si, table))
    return vals


def outcome(v):
    return (v > 0) - (v < 0)


def endgame(X, rnd, empties_choices, min_moves=2):
    while True:
        board, color = T.random_position(rnd, 60 - rnd.choice(empties_choices))
        me, opp = X.board_to_bb(board, color)
        moves = X.bb_to_squares(X.legal_moves(me, opp))
        if len(moves) >= min_moves:
            return me, opp, moves


def check_exact(X, rnd):
    ai = X.AI(8, -1, 5)
    su = np.zeros((X.MAXPLY, X.SU_COLS), dtype=np.uint64)
    si = np.zeros((X.MAXPLY, X.SI_COLS), dtype=np.int64)
    stats = np.zeros(7, dtype=np.int64)
    root = search = lost_all = 0
    for _ in range(120):
        me, opp, moves = endgame(X, rnd, (8, 10, 12, 14))
        vals = exact_values(X, me, opp, moves)
        best_out = max(outcome(v) for v in vals.values())
        order = list(moves)
        rnd.shuffle(order)
        ai.tt.fill(0)
        ai.last_outcome = None
        move, every_lost = ai._outcome_root(me, opp, order, stats, su, si, float("inf"))
        if best_out < 0:
            assert move is None and every_lost and ai.last_outcome == -1, (move, every_lost)
            lost_all += 1
        else:
            assert move in moves and not every_lost, (move, every_lost)
            assert outcome(vals[move]) == best_out == ai.last_outcome, (vals, move, ai.last_outcome)
        root += 1
        ai.tt.fill(0)
        ai.nps = 1e12
        chosen = ai._search(me, opp, moves, 1e9)
        assert outcome(vals[chosen]) == best_out, ("search missed the best outcome", vals, chosen)
        if best_out < 0 and ai.last_depth == 64 - int(X.popcnt(me | opp)):
            # the exact iteration finished (not a forced result within the horizon)
            assert vals[chosen] == max(vals.values()), ("not the least loss", vals, chosen)
        search += 1
    return root, search, lost_all


def check_deadlines(X, rnd):
    ai = X.AI(8, -1, 5)
    real_time, slices = X.time, X.SLICE_NODES
    runs = reported = positions = 0
    try:
        while positions < 8:
            me, opp, moves = endgame(X, rnd, (8, 10, 12), min_moves=3)
            vals = exact_values(X, me, opp, moves)
            positions += 1
            X.SLICE_NODES = 4
            X.time = FakeClock()
            ai.tt.fill(0)                       # never cleared again for this position
            for budget in range(1, 100000):
                ai.nps = 1e12                   # keep the outcome-first path active
                move = ai._search(me, opp, moves, float(budget))
                runs += 1
                assert move in moves, (budget, move)
                if ai.last_outcome is not None:
                    assert outcome(vals[move]) == ai.last_outcome, (budget, vals, move)
                    reported += 1
                    if ai.last_outcome == max(outcome(v) for v in vals.values()):
                        break
            else:
                raise AssertionError("deadline sweep did not finish")
    finally:
        X.time, X.SLICE_NODES = real_time, slices
    assert runs >= 200, ("too few cut points exercised", runs)
    return runs, reported


class Script:
    """Stands in for AI._child_value: values by (iteration depth, move, kind) from
    tables; kind is 'outcome' for the outcome solve's (-1, 1) and (0, 1) windows at
    full depth, else 'null' or 'full'. A cut point returns None (deadline)."""

    def __init__(self, X, me, opp, moves):
        self.X = X
        self.empties = 64 - int(X.popcnt(me | opp))
        self.sq_of = {(int(a), int(b)): sq for sq, (a, b) in children(X, me, opp, moves).items()}
        self.reset({}, {}, set())

    def reset(self, value, final, cuts):
        self.value, self.final, self.cuts, self.calls = value, final, cuts, []

    def __call__(self, ai, child_me, child_opp, depth, lo, hi, stats, su, si, deadline):
        sq = self.sq_of[(int(child_me), int(child_opp))]
        if depth == self.empties - 1 and (lo, hi) in ((-1, 1), (0, 1)):
            kind, v = "outcome", self.final[sq]
        else:
            kind = "null" if hi == lo + 1 else "full"
            v = self.value.get((depth + 1, sq), self.value.get(sq, -5))
        key = (depth + 1, sq, kind)
        self.calls.append(key)
        return None if key in self.cuts else v


def check_scripted(X, rnd):
    original = X.AI._child_value
    flags = (X.OUTCOME_FIRST, X.PARTIAL_ITERATION)
    checks = 0
    try:
        ai = X.AI(8, -1, 5)
        # ---- partial iterations (outcome-first off, a midgame position) -----------
        me, opp, moves = endgame(X, rnd, (40,), min_moves=4)
        script = Script(X, me, opp, moves)
        X.AI._child_value = lambda self, *a: script(self, *a)
        a, b, c = moves[:3]
        base = {a: 30, b: 20, c: 10}            # depth-1 values decide the depth-2 order
        cases = [  # depth-2 values, cut point, expected move with / without the rule
            ({a: 5, b: 6, c: 1}, (2, b, "full"), b),     # b failed high, re-search cut
            ({a: 5, b: 6, c: 1}, (2, c, "null"), b),     # b finished higher, then cut
            ({a: 5, b: 4, c: 1}, (2, c, "null"), a),     # nothing better
            ({a: 5, b: 6, c: 1}, (2, a, "full"), a),     # the first move itself was cut
        ]
        X.OUTCOME_FIRST = False
        for d2, cut, expected in cases:
            for partial in (True, False):
                X.PARTIAL_ITERATION = partial
                value = {sq: base.get(sq, -5) for sq in moves}
                value.update({(2, sq): v for sq, v in d2.items()})
                script.reset(value, {}, {cut})
                move = ai._search(me, opp, moves, 1e9)
                assert move == (expected if partial else a), (d2, cut, partial, move)
                assert script.calls[-1] == cut, ("search continued after the cut", script.calls)
                checks += 1
        # ---- outcome solve (outcome-first on, an endgame position) -----------------
        X.OUTCOME_FIRST, X.PARTIAL_ITERATION = True, True
        me, opp, moves = endgame(X, rnd, (16, 18, 20), min_moves=4)
        script = Script(X, me, opp, moves)
        X.AI._child_value = lambda self, *a: script(self, *a)
        e = script.empties
        a, b, c = moves[:3]
        base = {a: 30, b: 20, c: 10}            # every heuristic depth orders a, b, c
        W, L, D = X.WIN_BASE + 2, -X.WIN_BASE - 2, 0
        cases = [  # outcomes by move, cut points, expected move, expected last_outcome
            ({a: L, b: W, c: D}, set(), b, 1),
            ({a: D, b: L, c: W}, set(), c, 1),
            ({a: L, b: D, c: L}, set(), b, 0),
            ({a: L, b: D, c: W}, {(e, c, "outcome")}, b, 0),      # draw kept at the cut
            ({a: L, b: L, c: W}, {(e, b, "outcome")}, b, None),   # first move not proven lost
            ({a: W, b: L, c: L}, {(e, a, "outcome")}, a, None),   # nothing proven
        ]
        for final, cuts, expected, out in cases:
            final = {sq: final.get(sq, L) for sq in moves}
            script.reset({sq: base.get(sq, -5) for sq in moves}, final, cuts)
            ai.nps = 1e12
            move = ai._search(me, opp, moves, 1e9)
            assert move == expected and ai.last_outcome == out, (final, cuts, move, ai.last_outcome)
            checks += 1
        # every move loses: the exact iteration (scripted margins) picks the least loss
        final = {sq: L for sq in moves}
        exact = {(e, sq): -X.WIN_BASE - 20 for sq in moves}
        exact.update({(e, a): -X.WIN_BASE - 10, (e, b): -X.WIN_BASE - 2, (e, c): -X.WIN_BASE - 6})
        for cut, expected in ((set(), b), ({(e, b, "null")}, a)):
            value = {sq: base.get(sq, -5) for sq in moves}
            value.update(exact)
            script.reset(value, final, cut)
            ai.nps = 1e12
            move = ai._search(me, opp, moves, 1e9)
            assert move == expected and ai.last_outcome == -1, (cut, move)
            checks += 1
        # the outcome-first path is skipped when its cost does not fit the budget
        script.reset({sq: base.get(sq, -5) for sq in moves}, {sq: W for sq in moves}, set())
        ai.nps = 1.0
        ai._search(me, opp, moves, 1e-9)                  # e >= 16: C[e] > nps * budget
        assert all(k[2] != "outcome" for k in script.calls), script.calls
        checks += 1
    finally:
        X.AI._child_value = original
        X.OUTCOME_FIRST, X.PARTIAL_ITERATION = flags
    return checks


def main():
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "ai.py")
    print("source:", path.resolve())
    X = load(path)
    for name in ("OUTCOME_FIRST", "PARTIAL_ITERATION", "OUTCOME_COST"):
        assert hasattr(X, name), f"{name} missing: not a Round 7 engine"
    rnd = random.Random(7_100_001)
    root, search, lost_all = check_exact(X, rnd)
    runs, reported = check_deadlines(X, rnd)
    scripted = check_scripted(X, rnd)
    # the platform entry point still answers legally with the Round 7 driver
    ai = X.AI(8, -1, 0.05)
    for empties in (10, 18, 22, 40):
        board, color = T.random_position(rnd, 60 - empties)
        ai.color = color
        ai.go(board)
        legal = set(T.R.legal_moves(board, color))
        assert set(ai.candidate_list) == legal
        assert not legal or ai.candidate_list[-1] in legal
    assert ai.fallback_count == ai.warmup_failures == 0
    print("OK: %d outcome solves (%d with every move lost), %d end-to-end searches, "
          "%d deadline runs (%d reported outcomes), %d scripted cut-point checks"
          % (root, lost_all, search, runs, reported, scripted))


if __name__ == "__main__":
    main()
