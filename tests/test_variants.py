"""Exact reference checks for experimental TT and feature versions."""
import importlib.util
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import test_search as T


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "versions/v02_tt_ai.py"
    spec = importlib.util.spec_from_file_location("unregistered_variant", path)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)  # Intentionally absent from sys.modules.
    A.DEBUG_RAISE = True
    T.A = A
    ai = A.AI(8, -1, 0.01)
    rng, checked = random.Random(719), 0
    weights = A.DEFAULT_W.copy()
    if weights.shape[1] > 6:
        weights[:, 6:] = np.array([3, -2, 1, 2, -1])
        for square in (0, 7, 56, 63):
            bit = np.uint64(1 << square)
            assert int(A.stable_edge_mask(bit)) == int(bit)
        assert int(A.stable_edge_mask(np.uint64(1 << 27))) == 0
    for plies in (10, 30, 50, 55):
        for _ in range(3):
            board, color = T.random_position(rng, plies)
            me, opp = A.board_to_bb(board, color)
            for depth in (1, 3, 5):
                T.W = weights
                # Reference minimax accepts W through its module-level global.
                expected = T.minimax(me, opp, depth)
                for chunk in (7, 512):
                    tt = (np.zeros((A.TT_SIZE, 6), dtype=np.uint64)
                          if hasattr(A, "TT_SIZE") else None)
                    su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
                    si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
                    stats = np.zeros(7, dtype=np.int64)
                    while True:
                        stats[0], stats[1], stats[2] = 0, chunk, 0
                        values = (me, opp, depth, -A.INF, A.INF,
                                  stats, weights, su, si)
                        got = A.alphabeta(*values, tt) if tt is not None else A.alphabeta(*values)
                        if stats[2] != 2:
                            break
                    assert got == expected, (plies, depth, chunk, got, expected)
                    checked += 1
    assert ai.fallback_count == ai.warmup_failures == 0
    print("OK: %s; %d TT/feature/minimax comparisons; TT %.2f MiB" %
          (path, checked, ai.tt.nbytes / 1024 ** 2 if hasattr(ai, "tt") else 0))


if __name__ == "__main__":
    main()
