"""
Reversed Reversi (Anti-Reversi) AI -- SINGLE-FILE submission.

Objective of the game: the side with FEWER discs of its own colour at the end wins.
Everything the platform needs lives in this one file (class `AI` at the bottom).

Sections
  0. imports / constants
  1. bitboard engine (Numba)        <- Layer 0  (correctness, Phase 1)
  2. evaluation function (Numba)    <- Layer 2  (strength,    Phase 2)
  3. alpha-beta search (Numba)      <- Layer 1 + Layer 3 (endgame = deep search when few empties)
  4. python-side helpers
  5. class AI (platform interface)
"""
import numpy as np
import random
import time
from numba import njit

# ----------------------------------------------------------------------------
# 0. constants
# ----------------------------------------------------------------------------
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

# --- knobs that depend on facts we still have to confirm with the TA ---------
GAME_BUDGET_SEC = 80.0      # how much of the 180 s whole-game limit WE allow ourselves
                            # (conservative: 180 s may be shared by both sides)
PER_MOVE_FRACTION = 0.40    # use at most this fraction of `time_out` on one move
ENDGAME_EMPTIES = 10        # solve exactly (search to the end) when <= this many empties
MAX_DEPTH = 40
DEBUG_RAISE = False         # dev tools set this True so that errors inside go() are NOT swallowed
# -----------------------------------------------------------------------------

# bit index = row * 8 + col   (row 0 = chessboard[0], col 0 = chessboard[r][0])
U64 = np.uint64
S1, S7, S8, S9 = U64(1), U64(7), U64(8), U64(9)
S2, S4, S56 = U64(2), U64(4), U64(56)
NOT_COL0 = U64(0xFEFEFEFEFEFEFEFE)   # clears column 0 (wrap-around after an east-ward shift)
NOT_COL7 = U64(0x7F7F7F7F7F7F7F7F)   # clears column 7 (wrap-around after a west-ward shift)
M1 = U64(0x5555555555555555)
M2 = U64(0x3333333333333333)
M4 = U64(0x0F0F0F0F0F0F0F0F)
H01 = U64(0x0101010101010101)
ONE = U64(1)


def _bb(squares):
    v = 0
    for s in squares:
        v |= 1 << int(s)
    return U64(v)


def _sq(r, c):
    return r * 8 + c


CORNERS = _bb([_sq(0, 0), _sq(0, 7), _sq(7, 0), _sq(7, 7)])
X_SQUARES = _bb([_sq(1, 1), _sq(1, 6), _sq(6, 1), _sq(6, 6)])
C_SQUARES = _bb([_sq(0, 1), _sq(1, 0), _sq(0, 6), _sq(1, 7),
                 _sq(6, 0), _sq(7, 1), _sq(7, 6), _sq(6, 7)])
_edge_all = [_sq(r, c) for r in range(8) for c in range(8) if r in (0, 7) or c in (0, 7)]
_edge_rest = [s for s in _edge_all
              if not ((1 << s) & int(CORNERS)) and not ((1 << s) & int(C_SQUARES))]
EDGES = _bb(_edge_rest)          # edge squares that are neither corner nor C-square

INF = 1_000_000_000
WIN_BASE = 100_000

# ----------------------------------------------------------------------------
# 1. bitboard engine
# ----------------------------------------------------------------------------


@njit(cache=False)
def popcnt(x):
    x = x - ((x >> S1) & M1)
    x = (x & M2) + ((x >> S2) & M2)
    x = (x + (x >> S4)) & M4
    return np.int64((x * H01) >> S56)


@njit(cache=False)
def shift(x, d):
    """Move every bit one step in direction d (0..7); bits leaving the board are dropped."""
    if d == 0:      # E   (col+1)
        return (x << S1) & NOT_COL0
    elif d == 1:    # W   (col-1)
        return (x >> S1) & NOT_COL7
    elif d == 2:    # S   (row+1)
        return x << S8
    elif d == 3:    # N   (row-1)
        return x >> S8
    elif d == 4:    # SE
        return (x << S9) & NOT_COL0
    elif d == 5:    # NW
        return (x >> S9) & NOT_COL7
    elif d == 6:    # SW
        return (x << S7) & NOT_COL7
    else:           # NE
        return (x >> S7) & NOT_COL0


@njit(cache=False)
def legal_moves(me, opp):
    """Bitboard of all squares where `me` may play."""
    empty = ~(me | opp)
    moves = U64(0)
    for d in range(8):
        t = shift(me, d) & opp
        for _ in range(5):
            t |= shift(t, d) & opp
        moves |= shift(t, d) & empty
    return moves


@njit(cache=False)
def flips(me, opp, move):
    """Bitboard of the discs flipped if `me` plays on the single-bit board `move`."""
    f = U64(0)
    for d in range(8):
        line = U64(0)
        x = shift(move, d)
        while (x & opp) != U64(0):
            line |= x
            x = shift(x, d)
        if (x & me) != U64(0):
            f |= line
    return f


# ----------------------------------------------------------------------------
# 2. evaluation   (value is from the point of view of the side to move `me`;
#                  HIGHER = BETTER FOR `me` in the REVERSED game)
# ----------------------------------------------------------------------------
# feature columns:   0 corner   1 X-square   2 C-square   3 edge   4 discs   5 mobility
# For columns 0-4 the feature is (opp_count - me_count), so a POSITIVE weight means
# "owning this is BAD for me" (reversed game).  Column 5 is (my_mobility - opp_mobility).
# PLACEHOLDER VALUES -- to be tuned by self-play / evolution (see tools/).
# rows = game phase:  0 opening (empties > 44), 1 middle (17..44), 2 end (<= 16)
DEFAULT_W = np.array([
    [40, -10, 5, 2, 1, 4, 0, 0, 0, 0, 0],
    [40, -10, 5, 2, 1, 6, 0, 0, 0, 0, 0],
    [40, -10, 5, 0, 8, 2, 0, 0, 0, 0, 0],
], dtype=np.int64)


@njit(cache=False)
def terminal_value(me, opp):
    diff = popcnt(opp) - popcnt(me)          # reversed: fewer own discs is better
    if diff > 0:
        return WIN_BASE + diff
    elif diff < 0:
        return -WIN_BASE + diff
    return 0


# Extra feature columns: stable-edge penalty, frontier benefit, potential
# mobility benefit, side-to-move parity, positional penalty. Initially disabled;
# signs and magnitudes require paired self-play evidence before adoption.
POSITION_WEIGHTS = np.array([
    [80, -20, 8, 4, 4, 8, -20, 80],
    [-20, -30, 0, 0, 0, 0, -30, -20],
    [8, 0, 2, 1, 1, 2, 0, 8],
    [4, 0, 1, 0, 0, 1, 0, 4],
    [4, 0, 1, 0, 0, 1, 0, 4],
    [8, 0, 2, 1, 1, 2, 0, 8],
    [-20, -30, 0, 0, 0, 0, -30, -20],
    [80, -20, 8, 4, 4, 8, -20, 80],
], dtype=np.int64).reshape(-1)
# Group equal weights so evaluation uses population counts, not a square loop.
POSITION_LEVELS = np.unique(POSITION_WEIGHTS)
POSITION_MASKS = np.array([_bb(np.flatnonzero(POSITION_WEIGHTS == v))
                           for v in POSITION_LEVELS], dtype=np.uint64)
HORIZONTAL_EDGES = U64(0xFF000000000000FF)
VERTICAL_EDGES = U64(0x8181818181818181)


@njit(cache=False)
def adjacent(bits):
    out = U64(0)
    for direction in range(8):
        out |= shift(bits, direction)
    return out


@njit(cache=False)
def stable_edge_mask(discs):
    # Conservative subset: same-color uninterrupted edge chains from corners.
    stable = discs & CORNERS
    for direction in range(4):
        chain = discs & CORNERS
        edge = HORIZONTAL_EDGES if direction < 2 else VERTICAL_EDGES
        for _ in range(6):
            chain |= shift(chain, direction) & discs & edge
        stable |= chain
    return stable


@njit(cache=False)
def evaluate(me, opp, my_moves, W):
    cm = popcnt(me)
    co = popcnt(opp)
    empties = 64 - cm - co
    if empties > 44:
        ph = 0
    elif empties > 16:
        ph = 1
    else:
        ph = 2
    f0 = popcnt(opp & CORNERS) - popcnt(me & CORNERS)
    f1 = popcnt(opp & X_SQUARES) - popcnt(me & X_SQUARES)
    f2 = popcnt(opp & C_SQUARES) - popcnt(me & C_SQUARES)
    f3 = popcnt(opp & EDGES) - popcnt(me & EDGES)
    f4 = co - cm
    f5 = popcnt(my_moves) - popcnt(legal_moves(opp, me))
    value = (W[ph, 0] * f0 + W[ph, 1] * f1 + W[ph, 2] * f2 +
             W[ph, 3] * f3 + W[ph, 4] * f4 + W[ph, 5] * f5)
    empty = ~(me | opp)
    if W[ph, 6] != 0:
        value += W[ph, 6] * (popcnt(stable_edge_mask(opp)) - popcnt(stable_edge_mask(me)))
    if W[ph, 7] != 0:
        frontier = adjacent(empty)
        value += W[ph, 7] * (popcnt(me & frontier) - popcnt(opp & frontier))
    if W[ph, 8] != 0:
        value += W[ph, 8] * (popcnt(empty & adjacent(opp)) - popcnt(empty & adjacent(me)))
    if W[ph, 9] != 0:
        value += W[ph, 9] * (1 if empties % 2 else -1)
    if W[ph, 10] != 0:
        positional = 0
        for i in range(len(POSITION_LEVELS)):
            mask = POSITION_MASKS[i]
            positional += POSITION_LEVELS[i] * (popcnt(opp & mask) - popcnt(me & mask))
        value += W[ph, 10] * positional
    return value


# ----------------------------------------------------------------------------
# 3. search  --  ITERATIVE negamax alpha-beta (explicit stack, NO recursion)
#
#    Why no recursion?  numba's self-recursion can fail at run time with
#    "numba jitted function aborted due to unresolved symbol" depending on HOW the
#    platform loads this file (importlib without sys.modules, exec, ...).  Since go()
#    swallows exceptions to stay legal, that failure would silently turn the AI into
#    "play the first legal move".  An explicit stack removes the whole risk.
#
#    stats[0] = nodes visited, stats[1] = node limit, stats[2] = 1 if aborted
#    Seven-element stats allow yielding/resuming the explicit stack so the Python
#    driver can check a monotonic wall clock between short node slices.
# ----------------------------------------------------------------------------
MAXPLY = 128
NSLOT = 64                      # >= max number of legal moves in a position (33)
# columns of the uint64 frame array `su`
C_ME, C_OPP, C_MV, C_FL = 0, 1, 2, 2 + NSLOT
SU_COLS = 2 + 2 * NSLOT
# columns of the int64 frame array `si`
I_DEPTH, I_ALPHA, I_BETA, I_BEST, I_N, I_I, I_KEY = 0, 1, 2, 3, 4, 5, 6
I_ORIG_ALPHA, I_TT_MOVE = 6 + NSLOT, 7 + NSLOT
SI_COLS = 8 + NSLOT
TT_SIZE = 32768  # 1.5 MiB: full keys, depth, score, move, bound flag


@njit(cache=False)
def tt_index(me, opp):
    key = me ^ ((opp << U64(17)) | (opp >> U64(47)))
    key ^= key >> U64(30)
    key *= U64(0xBF58476D1CE4E5B9)
    key ^= key >> U64(27)
    key *= U64(0x94D049BB133111EB)
    key ^= key >> U64(31)
    return np.int64(key & U64(TT_SIZE - 1))


@njit(cache=False)
def alphabeta(me, opp, depth, alpha, beta, stats, W, su, si, tt=None):
    """Value of the position for `me` (side to move) searched `depth` plies (passes are free).
    su / si are scratch frame arrays of shape (MAXPLY, SU_COLS) / (MAXPLY, SI_COLS)."""
    # Seven-element stats enable resumable node slices. Legacy callers may still
    # use three elements for a single bounded search (including reference tests).
    if stats.size >= 7 and stats[3] != 0:
        p, mode, ret = stats[4], stats[5], stats[6]
    else:
        p = 0
        su[0, C_ME], su[0, C_OPP] = me, opp
        si[0, I_DEPTH] = depth
        si[0, I_ALPHA], si[0, I_BETA] = alpha, beta
        mode, ret = 0, 0
    while True:
        if stats[0] >= stats[1]:
            stats[2] = 1
            if stats.size >= 7:
                stats[2] = 2  # Yield to Python to check the wall clock.
                stats[3], stats[4], stats[5], stats[6] = 1, p, mode, ret
            return 0
        if mode == 0:
            cm = su[p, C_ME]
            co = su[p, C_OPP]
            d = si[p, I_DEPTH]
            stats[0] += 1
            si[p, I_ORIG_ALPHA] = si[p, I_ALPHA]
            si[p, I_TT_MOVE] = 0
            preferred = U64(0)
            if tt is not None and d > 0:
                h = tt_index(cm, co)
                if tt[h, 5] != U64(0) and tt[h, 0] == cm and tt[h, 1] == co:
                    preferred = tt[h, 4]
                    # Exact depth preserves the defined depth-limited minimax value.
                    if np.int64(tt[h, 2]) == d:
                        flag, cached = tt[h, 5], np.int64(tt[h, 3])
                        if (flag == U64(1) or
                            (flag == U64(2) and cached >= si[p, I_BETA]) or
                            (flag == U64(3) and cached <= si[p, I_ALPHA])):
                            ret, mode = cached, 2
                            continue
            moves = legal_moves(cm, co)
            if moves == U64(0):
                if legal_moves(co, cm) == U64(0):
                    ret = terminal_value(cm, co)
                    mode = 2
                else:                                   # pass: one fake move "0"
                    su[p, C_MV] = U64(0)
                    su[p, C_FL] = U64(0)
                    si[p, I_N] = 1
                    si[p, I_I] = 0
                    si[p, I_BEST] = -INF
                    mode = 1
            elif d <= 0:
                ret = evaluate(cm, co, moves, W)
                mode = 2
            else:
                n = 0
                m = moves
                while m != U64(0):
                    lsb = m & ((~m) + ONE)
                    m ^= lsb
                    f = flips(cm, co, lsb)
                    su[p, C_MV + n] = lsb
                    su[p, C_FL + n] = f
                    si[p, I_KEY + n] = popcnt(f) - (1000 if lsb == preferred else 0)        # fewest flips first
                    n += 1
                if d >= 2:
                    for a in range(n):                  # selection sort (n <= 33)
                        b = a
                        for j in range(a + 1, n):
                            if si[p, I_KEY + j] < si[p, I_KEY + b]:
                                b = j
                        if b != a:
                            tk = si[p, I_KEY + a]
                            si[p, I_KEY + a] = si[p, I_KEY + b]
                            si[p, I_KEY + b] = tk
                            tm = su[p, C_MV + a]
                            su[p, C_MV + a] = su[p, C_MV + b]
                            su[p, C_MV + b] = tm
                            tf = su[p, C_FL + a]
                            su[p, C_FL + a] = su[p, C_FL + b]
                            su[p, C_FL + b] = tf
                si[p, I_N] = n
                si[p, I_I] = 0
                si[p, I_BEST] = -INF
                mode = 1
        elif mode == 1:
            i = si[p, I_I]
            if i >= si[p, I_N]:
                ret = si[p, I_BEST]
                mode = 2
            else:
                mvb = su[p, C_MV + i]
                f = su[p, C_FL + i]
                q = p + 1
                su[q, C_ME] = su[p, C_OPP] & ~f
                su[q, C_OPP] = su[p, C_ME] | mvb | f
                if mvb == U64(0):
                    si[q, I_DEPTH] = si[p, I_DEPTH]     # pass costs no ply
                else:
                    si[q, I_DEPTH] = si[p, I_DEPTH] - 1
                si[q, I_ALPHA] = -si[p, I_BETA]
                si[q, I_BETA] = -si[p, I_ALPHA]
                p = q
                mode = 0
        else:
            if tt is not None and si[p, I_DEPTH] > 0:
                cm, co = su[p, C_ME], su[p, C_OPP]
                h = tt_index(cm, co)
                flag = U64(1)
                if ret <= si[p, I_ORIG_ALPHA]:
                    flag = U64(3)
                elif ret >= si[p, I_BETA]:
                    flag = U64(2)
                tt[h, 0], tt[h, 1] = cm, co
                tt[h, 2], tt[h, 3] = U64(si[p, I_DEPTH]), U64(ret)
                # Retain a known move when a cached result returns immediately.
                if si[p, I_TT_MOVE] != 0:
                    tt[h, 4] = U64(si[p, I_TT_MOVE])
                elif tt[h, 5] == U64(0):
                    tt[h, 4] = U64(0)
                tt[h, 5] = flag
            if p == 0:
                if stats.size >= 7:
                    stats[3] = 0
                stats[2] = 0
                return ret
            p -= 1
            v = -ret
            if v > si[p, I_BEST]:
                si[p, I_TT_MOVE] = np.int64(su[p, C_MV + si[p, I_I]])
                si[p, I_BEST] = v
                if v > si[p, I_ALPHA]:
                    si[p, I_ALPHA] = v
            si[p, I_I] += 1
            if si[p, I_ALPHA] >= si[p, I_BETA]:
                ret = si[p, I_BEST]                     # beta cut-off: return from p as well
            else:
                mode = 1


@njit(cache=False)
def search_root(me, opp, order, n, depth, stats, W, vals):
    """Search the n root moves listed in `order` (square indices) to `depth`.
    Fills vals[i] with the score of order[i]; returns the index of the best move,
    or -1 if the node budget ran out (stats[2] == 1).
    NOTE: the whole root loop lives in Numba on purpose -- numba hands uint64 values back
    to python as plain `int`, which cannot be mixed with np.uint64 in bit operations."""
    su = np.zeros((MAXPLY, SU_COLS), dtype=np.uint64)
    si = np.zeros((MAXPLY, SI_COLS), dtype=np.int64)
    alpha = -INF
    best_i = -1
    best_v = -INF
    for i in range(n):
        b = ONE << np.uint64(order[i])
        f = flips(me, opp, b)
        v = -alphabeta(opp & ~f, me | b | f, depth - 1, -INF, -alpha, stats, W, su, si)
        if stats[2] != 0:
            return -1
        vals[i] = v
        if v > best_v:
            best_v = v
            best_i = i
        if v > alpha:
            alpha = v
    return best_i


# ----------------------------------------------------------------------------
# 4. python-side helpers
# ----------------------------------------------------------------------------
_WEIGHTS = np.uint64(1) << np.arange(64, dtype=np.uint64)


def board_to_bb(chessboard, color):
    """numpy 8x8 board -> (me, opp) bitboards (np.uint64) for the player `color`."""
    flat = np.asarray(chessboard).reshape(-1)
    me = np.bitwise_or.reduce(_WEIGHTS[flat == color]) if np.any(flat == color) else U64(0)
    opp = np.bitwise_or.reduce(_WEIGHTS[flat == -color]) if np.any(flat == -color) else U64(0)
    return U64(me), U64(opp)


def bb_to_squares(bb):
    v = int(bb)
    return [i for i in range(64) if (v >> i) & 1]


def sq_to_move(sq):
    return (sq // 8, sq % 8)          # (row, col)


# generic numpy path: ONLY used when the board is not 8x8 (the statement says the
# administrators may change the board size).  Slow but correct for any n_rows x n_cols.
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def generic_flips(board, color, r, c):
    """number of discs flipped if `color` plays (r, c) on an arbitrary-size board (0 = illegal)."""
    n_r, n_c = board.shape
    if board[r][c] != COLOR_NONE:
        return 0
    total = 0
    for dr, dc in _DIRS8:
        rr, cc, seen = r + dr, c + dc, 0
        while 0 <= rr < n_r and 0 <= cc < n_c and board[rr][cc] == -color:
            rr += dr
            cc += dc
            seen += 1
        if seen and 0 <= rr < n_r and 0 <= cc < n_c and board[rr][cc] == color:
            total += seen
    return total


def generic_choose(board, color):
    """(all legal moves, chosen move) for any board size.  Heuristic: never take a corner
    if avoidable (reversed game), then fewest flips."""
    n_r, n_c = board.shape
    moves, best, best_key = [], None, None
    for r in range(n_r):
        for c in range(n_c):
            f = generic_flips(board, color, r, c)
            if f:
                mv = (int(r), int(c))
                moves.append(mv)
                is_corner = (r in (0, n_r - 1)) and (c in (0, n_c - 1))
                key = (is_corner, f)
                if best_key is None or key < best_key:
                    best, best_key = mv, key
    return moves, best


# ----------------------------------------------------------------------------
# 5. platform interface
# ----------------------------------------------------------------------------
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.W = DEFAULT_W.copy()
        self.tt = np.zeros((TT_SIZE, 6), dtype=np.uint64)
        self.nps = 300_000.0            # nodes/second estimate, re-calibrated at run time
        self.used = 0.0                 # wall-clock seconds spent in go() so far
        self.fallback_count = 0
        self.warmup_failures = 0
        self.last_depth = 0
        self.last_nodes = 0
        self.last_solved = False
        self._warmup()

    # compile everything once and calibrate the speed (also a good place to test "does
    # JIT time count against us?" -- see tools/bench_first_move.py)
    def _warmup(self):
        try:
            board = np.zeros((8, 8), dtype=np.int64)
            board[3][3] = board[4][4] = COLOR_WHITE
            board[3][4] = board[4][3] = COLOR_BLACK
            me, opp = board_to_bb(board, self.color)
            order = np.array(bb_to_squares(legal_moves(me, opp)), dtype=np.int64)
            vals = np.zeros(len(order), dtype=np.int64)
            stats = np.zeros(7, dtype=np.int64)
            stats[1] = 1 << 60
            search_root(me, opp, order, len(order), 2, stats, self.W, vals)      # compile
            stats[:] = 0
            stats[1] = 1 << 60
            t = time.perf_counter()
            search_root(me, opp, order, len(order), 7, stats, self.W, vals)      # calibrate
            dt = max(time.perf_counter() - t, 1e-4)
            self.nps = max(50_000.0, stats[0] / dt)
            # Compile the mutable-table specialization before the first go().
            su = np.zeros((MAXPLY, SU_COLS), dtype=np.uint64)
            si = np.zeros((MAXPLY, SI_COLS), dtype=np.int64)
            stats[:] = 0
            stats[1] = 1 << 60
            alphabeta(me, opp, 2, -INF, INF, stats, self.W, su, si, self.tt)
            self.tt.fill(0)
        except Exception:
            self.warmup_failures += 1
            if DEBUG_RAISE:
                raise

    # ---- time management ----------------------------------------------------
    def _budget(self, empties):
        moves_left = max(1, (empties + 1) // 2)
        remaining = max(0.0, GAME_BUDGET_SEC - self.used)
        # Respect very small limits and exhausted game budgets; never impose a
        # minimum that could exceed the supplied timeout.
        return max(0.0, min(remaining / moves_left * 1.5,
                            float(self.time_out) * PER_MOVE_FRACTION, 2.0))

    # ---- resumable root search with wall-clock checks -----------------------
    def _search(self, me, opp, root_moves, budget):
        start = time.perf_counter()
        deadline = start + budget
        self.tt.fill(0)  # Weights can change between offline experiments.
        empties = int(64 - popcnt(me | opp))
        order = sorted(root_moves, key=lambda sq: (
            bool((1 << sq) & int(CORNERS)),
            int(popcnt(U64(flips(me, opp, U64(1 << sq)))))))
        best = order[0]
        self.last_depth = self.last_nodes = 0
        self.last_solved = False
        su = np.zeros((MAXPLY, SU_COLS), dtype=np.uint64)
        si = np.zeros((MAXPLY, SI_COLS), dtype=np.int64)
        stats = np.zeros(7, dtype=np.int64)
        max_depth = empties if empties <= ENDGAME_EMPTIES else min(MAX_DEPTH, empties)
        try:
            for depth in range(1, max_depth + 1):
                alpha, iteration_best, results = -INF, best, []
                for sq in order:
                    if time.perf_counter() >= deadline:
                        return best
                    move = U64(1 << sq)
                    flipped = U64(flips(me, opp, move))
                    child_me, child_opp = U64(opp & ~flipped), U64(me | move | flipped)
                    stats[:] = 0
                    while True:
                        if time.perf_counter() >= deadline:
                            return best  # Keep only the last COMPLETED depth.
                        # A bad nps estimate cannot create a long uninterruptible
                        # search: the explicit stack yields at most every 512 nodes.
                        stats[0], stats[1], stats[2] = 0, 512, 0
                        value = -int(alphabeta(child_me, child_opp, depth - 1,
                                             -INF, -alpha, stats, self.W, su, si, self.tt))
                        self.last_nodes += int(stats[0])
                        if stats[2] != 2:
                            break
                    results.append((value, sq))
                    if value > alpha:
                        alpha, iteration_best = value, sq
                best = iteration_best
                self.last_depth = depth
                if depth >= empties or abs(alpha) >= WIN_BASE:
                    self.last_solved = True
                    break
                results.sort(key=lambda item: -item[0])
                order = [sq for _, sq in results]
        finally:
            elapsed = time.perf_counter() - start
            if self.last_nodes and elapsed > 0:
                self.nps = self.last_nodes / elapsed
        return best

    # ---- platform entry point ---------------------------------------------------
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        t_start = time.perf_counter()
        try:
            board = np.asarray(chessboard)
            if board.shape != (8, 8):                    # non-default board size: slow generic path
                moves, best = generic_choose(board, self.color)
                if moves:
                    self.candidate_list.extend(moves)
                    self.candidate_list.append(best)
                return
            me, opp = board_to_bb(board, self.color)
            moves = bb_to_squares(legal_moves(me, opp))
            if not moves:
                return                                   # no legal move -> empty list
            all_moves = [sq_to_move(s) for s in moves]
            self.candidate_list.extend(all_moves)
            if len(moves) == 1:
                self.candidate_list.append(all_moves[0])
                return
            try:
                empties = int(64 - popcnt(me | opp))
                best_sq = self._search(me, opp, moves, self._budget(empties))
            except Exception:
                self.fallback_count += 1
                if DEBUG_RAISE:
                    raise
                _, fallback = generic_choose(board, self.color)
                best_sq = fallback[0] * 8 + fallback[1]                       # never crash, never illegal
            # ALL legal moves first, the chosen one LAST (platform takes the last element)
            self.candidate_list.append(sq_to_move(best_sq))
        except Exception:
            self.fallback_count += 1
            if DEBUG_RAISE:
                raise
            # last resort: independent slow path, so we still answer with legal moves
            self.candidate_list.clear()
            try:
                moves, best = generic_choose(np.asarray(chessboard), self.color)
                if moves:
                    self.candidate_list.extend(moves)
                    self.candidate_list.append(best)
            except Exception:
                self.candidate_list.clear()
        finally:
            self.used += time.perf_counter() - t_start
