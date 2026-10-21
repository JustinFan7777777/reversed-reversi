"""Round 3 search changes must keep the searched tree and its values.

  * LEAF_INLINE: depth-1 nodes that score their leaves inline visit exactly the same
    tree as the frame path: identical node counts, values and transposition tables,
    PVS on and off, with and without the TT, full and narrow windows.
  * PVS_ENABLED: interior null windows return the same fixed-depth values as plain
    alpha-beta, and narrow windows keep fail-soft bound semantics.
  * ROOT_PVS in AI._search: the exact root value at the completed depth with and
    without root null windows (which must change the node count), and node slices
    of 1, 7 and 2048 nodes reproduce an uninterrupted search exactly.
  * Deadlines: a fake clock expires the search at every clock read in turn (inside
    null windows, between a fail-high null window and its re-search, inside the
    re-search) on a table that is never cleared; every returned move is legal and
    every reported value is exact for its completed depth.
  * Round 7 driver switches (OUTCOME_FIRST, PARTIAL_ITERATION), where present, are off
    in the root and deadline parts; tests/test_outcome_root.py covers them.

    python tests/test_pvs_inline.py [path/to/ai.py]
"""
import importlib.util
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import test_search as T                 # noqa: E402


def load(path, tag, **flags):
    """Fresh, unregistered engine copy. Compile-time flags are set before any njit
    function runs, because Numba freezes module globals at first compilation."""
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DEBUG_RAISE = True
    for name, value in flags.items():
        assert hasattr(mod, name), name
        setattr(mod, name, value)
    return mod


def search(mod, me, opp, depth, alpha, beta, W, table, chunk=1 << 60):
    """(value, nodes) of one resumable search, sliced every `chunk` nodes."""
    su = np.zeros((mod.MAXPLY, mod.SU_COLS), dtype=np.uint64)
    si = np.zeros((mod.MAXPLY, mod.SI_COLS), dtype=np.int64)
    stats = np.zeros(7, dtype=np.int64)
    nodes = 0
    while True:
        stats[0], stats[1], stats[2] = 0, chunk, 0
        value = int(mod.alphabeta(np.uint64(me), np.uint64(opp), depth, alpha, beta,
                                  stats, W, su, si, table))
        nodes += int(stats[0])
        if stats[2] != 2:
            return value, nodes


def fresh_table(mod):
    return np.zeros((mod.TT_SIZE, 6), dtype=np.uint64)


def child(mod, me, opp, sq):
    bit = np.uint64(1 << sq)
    f = np.uint64(mod.flips(me, opp, bit))
    return np.uint64(opp & ~f), np.uint64(me | bit | f)


class FakeClock:
    """Stands in for the time module: every perf_counter() read advances one second."""

    def __init__(self):
        self.t = 0.0

    def perf_counter(self):
        self.t += 1.0
        return self.t


def main():
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "ai.py")
    engines = {
        "inline_pvs": load(path, "t3_inline_pvs", PVS_ENABLED=True, LEAF_INLINE=True),
        "frames_pvs": load(path, "t3_frames_pvs", PVS_ENABLED=True, LEAF_INLINE=False),
        "inline_ab": load(path, "t3_inline_ab", PVS_ENABLED=False, LEAF_INLINE=True),
        "frames_ab": load(path, "t3_frames_ab", PVS_ENABLED=False, LEAF_INLINE=False),
    }
    X, REF = engines["inline_pvs"], engines["frames_ab"]
    # Parts 3-4 check v14's root semantics (the last completed depth decides). Round 7
    # driver switches, where present, are off there; tests/test_outcome_root.py covers them.
    round7 = {k: getattr(X, k) for k in ("OUTCOME_FIRST", "PARTIAL_ITERATION") if hasattr(X, k)}
    for k in round7:
        setattr(X, k, False)
    defaults = (X.ROOT_PVS, X.MAX_DEPTH, X.SLICE_NODES)    # restored after each part
    W = X.DEFAULT_W.copy()
    INF = X.INF
    rnd = random.Random(30303)
    same_tree = pvs_values = 0

    # ---- 1 + 2: same tree under LEAF_INLINE; same values under PVS -------------------
    for plies, depths in ((8, (1, 2, 3, 5)), (24, (1, 2, 4, 6)), (40, (1, 3, 5, 7)),
                          (52, (2, 4, 6, 8))):
        for _ in range(4):
            board, color = T.random_position(rnd, plies)
            me, opp = X.board_to_bb(board, color)
            for depth in depths:
                exact, _ = search(REF, me, opp, depth, -INF, INF, W, None)
                windows = [(-INF, INF), (-30, 30), (exact - 1, exact), (exact, exact + 1)]
                for alpha, beta in windows:
                    for use_tt in (False, True):
                        got, tables = {}, {}
                        for name, mod in engines.items():
                            tables[name] = fresh_table(mod) if use_tt else None
                            got[name] = search(mod, me, opp, depth, alpha, beta, W,
                                               tables[name])
                        for pvs in ("pvs", "ab"):
                            a, b = got["inline_" + pvs], got["frames_" + pvs]
                            assert a == b, ("inline tree differs", plies, depth, pvs,
                                            alpha, beta, use_tt, a, b)
                            if use_tt:
                                assert np.array_equal(tables["inline_" + pvs],
                                                      tables["frames_" + pvs]), (
                                    "inline TT differs", plies, depth, pvs, alpha, beta)
                            same_tree += 1
                        for name, (value, _) in got.items():
                            if exact <= alpha:
                                assert value <= alpha, (name, plies, depth, value, alpha)
                            elif exact >= beta:
                                assert value >= beta, (name, plies, depth, value, beta)
                            else:
                                assert value == exact, (name, plies, depth, value, exact)
                            pvs_values += 1

    # ---- 3: root PVS and slicing in AI._search at a fixed depth ------------------------
    ai = X.AI(8, -1, 5)
    root_checks = positions = pvs_changed = 0
    try:
        while positions < 8:
            board, color = T.random_position(rnd, rnd.choice((10, 20, 30, 40)))
            me, opp = X.board_to_bb(board, color)
            moves = X.bb_to_squares(X.legal_moves(me, opp))
            if len(moves) < 2:
                continue
            positions += 1
            runs = {}
            for root_pvs in (False, True):
                X.ROOT_PVS, X.MAX_DEPTH, X.SLICE_NODES = root_pvs, 6, 1 << 60
                ai.tt.fill(0)
                move = ai._search(me, opp, moves, 1e9)
                depth = ai.last_depth
                runs[root_pvs] = (move, ai.last_value, depth, ai.last_nodes)
                assert depth == 6 or ai.last_solved, (root_pvs, depth)
                exact, _ = search(REF, me, opp, depth, -INF, INF, W, None)
                assert ai.last_value == exact, ("root value is not exact", root_pvs,
                                                depth, ai.last_value, exact)
                value, _ = search(REF, *child(X, me, opp, move), depth - 1,
                                  -INF, INF, W, None)
                assert -value == exact, ("chosen move does not achieve the root value",
                                         root_pvs, move)
            assert runs[False][1:3] == runs[True][1:3], runs
            pvs_changed += runs[False][3] != runs[True][3]
            root_checks += 1
            X.ROOT_PVS, X.MAX_DEPTH = True, 5
            X.SLICE_NODES = 1 << 60
            ai.tt.fill(0)
            whole = (ai._search(me, opp, moves, 1e9), ai.last_value, ai.last_depth,
                     ai.last_nodes)
            for chunk in (1, 7, 2048):
                X.SLICE_NODES = chunk
                ai.tt.fill(0)
                sliced = (ai._search(me, opp, moves, 1e9), ai.last_value, ai.last_depth,
                          ai.last_nodes)
                assert sliced == whole, ("slicing changed the root search", chunk,
                                         sliced, whole)
                root_checks += 1
        assert pvs_changed > 0, "ROOT_PVS never changed the node count"
    finally:
        X.ROOT_PVS, X.MAX_DEPTH, X.SLICE_NODES = defaults

    # ---- 4: the deadline expires at every clock read in turn ---------------------------
    calls = []
    original = X.AI._child_value

    def logged(self, child_me, child_opp, depth, lo, hi, stats, su, si, deadline):
        result = original(self, child_me, child_opp, depth, lo, hi, stats, su, si, deadline)
        calls.append((hi == lo + 1, result, lo))
        return result

    real_time = X.time
    deadline_runs = in_null = in_research = positions = 0
    try:
        X.AI._child_value = logged
        while positions < 4:
            board, color = T.random_position(rnd, rnd.choice((14, 22, 30, 40, 50)))
            me, opp = X.board_to_bb(board, color)
            moves = X.bb_to_squares(X.legal_moves(me, opp))
            if len(moves) < 3:
                continue
            positions += 1
            empties = 64 - int(X.popcnt(me | opp))
            X.MAX_DEPTH, X.SLICE_NODES = 4, 16
            top = empties if empties <= X.ENDGAME_EMPTIES else min(4, empties)
            ref = {d: search(REF, me, opp, d, -INF, INF, W, None)[0]
                   for d in range(1, top + 1)}
            static_first = sorted(moves, key=lambda sq: (
                bool((1 << sq) & int(X.CORNERS)),
                int(X.popcnt(np.uint64(X.flips(me, opp, np.uint64(1 << sq)))))))[0]
            X.time = FakeClock()
            ai.tt.fill(0)                          # never cleared again for this position
            for budget in range(1, 100000):
                calls.clear()
                move = ai._search(me, opp, moves, float(budget))
                deadline_runs += 1
                assert move in moves, (budget, move)
                depth, value = ai.last_depth, ai.last_value
                if depth == 0:
                    assert move == static_first and value == 0, (budget, move, value)
                else:
                    assert value == ref[depth], ("inexact value", budget, depth, value)
                    reply, _ = search(REF, *child(X, me, opp, move), depth - 1,
                                      -INF, INF, W, None)
                    assert -reply == value, ("move does not achieve value", budget, depth)
                if calls and calls[-1][1] is None:          # this run was cut off
                    if calls[-1][0]:
                        in_null += 1
                    elif (len(calls) >= 2 and calls[-2][0] and calls[-2][1] is not None
                          and calls[-2][1] > calls[-2][2]):
                        in_research += 1
                if depth == top or ai.last_solved:
                    break
            else:
                raise AssertionError("deadline sweep did not finish")
            X.time = real_time                         # dirty table, unbounded search
            ai._search(me, opp, moves, 1e9)
            assert ai.last_depth == top or ai.last_solved, (ai.last_depth, top)
            assert ai.last_value == ref[ai.last_depth], (
                "dirty table changed the value", ai.last_value, ref[ai.last_depth])
        assert in_null > 0 and in_research > 0, (in_null, in_research)
    finally:
        X.time = real_time
        X.AI._child_value = original
        X.ROOT_PVS, X.MAX_DEPTH, X.SLICE_NODES = defaults

    # ---- the platform entry point still answers legally with these switches --------
    for k, v in round7.items():
        setattr(X, k, v)
    board, color = T.random_position(rnd, 20)
    ai.color, ai.time_out = color, 0.05
    ai.go(board)
    legal = set(T.R.legal_moves(board, color))
    assert set(ai.candidate_list) == legal
    assert not legal or ai.candidate_list[-1] in legal
    assert ai.fallback_count == ai.warmup_failures == 0
    print("OK: %d same-tree, %d PVS value/bound, %d root-PVS/slice checks; "
          "%d deadline runs (%d cut inside null windows, %d inside re-searches)" %
          (same_tree, pvs_values, root_checks, deadline_runs, in_null, in_research))


if __name__ == "__main__":
    main()
