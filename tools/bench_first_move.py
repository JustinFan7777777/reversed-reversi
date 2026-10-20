"""Fresh-process timing of import / __init__ / first go() / later go() -- run on every version.

    python tools/bench_first_move.py ai.py

What matters: is the JIT compile time (import + __init__ + first go) comfortably below
the 5 s limit?  (Confirm with the TA/platform whether import and __init__ are timed.)
"""
import importlib.util
import sys
import time

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "ai.py"
t0 = time.time()
spec = importlib.util.spec_from_file_location("sub", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.DEBUG_RAISE = True
t1 = time.time()
ai = mod.AI(8, -1, 5)
t2 = time.time()
b = np.zeros((8, 8), dtype=np.int64)
b[3][3] = b[4][4] = 1
b[3][4] = b[4][3] = -1
ai.go(b)
t3 = time.time()
ai.go(b)
t4 = time.time()
print("import %.2fs | __init__ %.2fs | 1st go %.2fs | 2nd go %.2fs | total before 2nd go %.2fs"
      % (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t3 - t0))
