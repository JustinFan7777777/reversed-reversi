"""Memory the way the platform counts it: INCREMENT of the process while the AI runs,
baseline (python + numpy + numba, loaded before AI exists) excluded.

    python tools/bench_memory.py ai.py --moves 30

  mem_after_init   : RSS right after AI.__init__           (the reference point)
  mem_after_moves  : RSS after `--moves` of our own moves  (self-play vs random)
  delta            : mem_after_moves - mem_after_init      <- this is what must stay < 100 MB
  peak_delta       : (peak RSS during the moves) - mem_after_init   (safer, worst case)

Also prints the 'import + __init__' growth separately, so you can see how much memory the
JIT compile needs and whether it happens BEFORE the reference point (it should: warm-up in
__init__) or leaks into the counted delta (if compile were lazy, in the first go()).
"""
import argparse
import importlib.util
import pathlib
import random
import sys
import os
import resource
import subprocess

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np                  # noqa: E402,F401  (baseline: loaded before measuring)
import numba                        # noqa: E402,F401
import ref_engine as R              # noqa: E402


def status(key):
    if sys.platform == "darwin":
        if key == "VmRSS":
            return int(subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) / 1024.0
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)
    with open("/proc/self/status") as f:          # Linux only (the platform container is Linux)
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024.0   # kB -> MB
    raise RuntimeError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="ai.py")
    ap.add_argument("--moves", type=int, default=30)
    ap.add_argument("--time-out", type=float, default=0.1)
    args = ap.parse_args()

    base = status("VmRSS")
    spec = importlib.util.spec_from_file_location("sub", args.path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DEBUG_RAISE = True
    after_import = status("VmRSS")
    me_color = R.BLACK
    ai = mod.AI(8, me_color, args.time_out)
    after_init = status("VmRSS")
    hwm_ref = status("VmHWM")

    rnd = random.Random(3)
    board, color, passes, mine, peak = R.initial_board(), R.BLACK, 0, 0, after_init
    while passes < 2 and mine < args.moves:
        legal = R.legal_moves(board, color)
        if not legal:
            passes += 1
        elif color == me_color:
            ai.go(board.copy())
            assert set(ai.candidate_list) == set(legal)
            assert ai.fallback_count == 0 if hasattr(ai, "fallback_count") else True
            mv = ai.candidate_list[-1]
            mine += 1
            peak = max(peak, status("VmHWM"))
            board = R.apply_move(board, color, tuple(mv))
            passes = 0
        else:
            board = R.apply_move(board, color, rnd.choice(legal))
            passes = 0
        color = -color
    after_moves = status("VmRSS")
    print("baseline (python+numpy+numba)   : %7.1f MB" % base)
    print("import ai.py growth             : %+7.1f MB" % (after_import - base))
    print("__init__ growth (JIT warm-up)   : %+7.1f MB   (before the reference point)" % (after_init - after_import))
    print("mem_after_init (reference)      : %7.1f MB" % after_init)
    print("mem_after_%d_moves              : %7.1f MB" % (mine, after_moves))
    print("COUNTED delta                   : %+7.1f MB   (limit 100)" % (after_moves - after_init))
    print("COUNTED peak delta              : %+7.1f MB   (limit 100)" % (max(peak, hwm_ref) - after_init))


if __name__ == "__main__":
    main()
