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
        v |= 1 << s
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
    [40, -10, 5, 2, 1, 4],
    [40, -10, 5, 2, 1, 6],
    [40, -10, 5, 0, 8, 2],
], dtype=np.int64)


@njit(cache=False)
def terminal_value(me, opp):
    diff = popcnt(opp) - popcnt(me)          # reversed: fewer own discs is better
    if diff > 0:
        return WIN_BASE + diff
    elif diff < 0:
        return -WIN_BASE + diff
    return 0


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
    return (W[ph, 0] * f0 + W[ph, 1] * f1 + W[ph, 2] * f2 +
            W[ph, 3] * f3 + W[ph, 4] * f4 + W[ph, 5] * f5)


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
#    (time.time() is not usable inside nopython code, so the python driver converts
#     the time budget into a node budget.)
# ----------------------------------------------------------------------------
MAXPLY = 128
NSLOT = 40                      # >= max number of legal moves in a position (33)
# columns of the uint64 frame array `su`
C_ME, C_OPP, C_MV, C_FL = 0, 1, 2, 2 + NSLOT
SU_COLS = 2 + 2 * NSLOT
# columns of the int64 frame array `si`
I_DEPTH, I_ALPHA, I_BETA, I_BEST, I_N, I_I, I_KEY = 0, 1, 2, 3, 4, 5, 6
SI_COLS = 6 + NSLOT


@njit(cache=False)
def alphabeta(me, opp, depth, alpha, beta, stats, W, su, si):
    """Value of the position for `me` (side to move) searched `depth` plies (passes are free).
    su / si are scratch frame arrays of shape (MAXPLY, SU_COLS) / (MAXPLY, SI_COLS)."""
    p = 0
    su[0, C_ME] = me
    su[0, C_OPP] = opp
    si[0, I_DEPTH] = depth
    si[0, I_ALPHA] = alpha
    si[0, I_BETA] = beta
    mode = 0                    # 0 = enter frame p, 1 = pick next child of p, 2 = return from p
    ret = 0
    while True:
        if mode == 0:
            cm = su[p, C_ME]
            co = su[p, C_OPP]
            d = si[p, I_DEPTH]
            stats[0] += 1
            if stats[0] > stats[1]:
                stats[2] = 1
                return 0
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
                    si[p, I_KEY + n] = popcnt(f)        # fewest flips first
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
            if p == 0:
                return ret
            p -= 1
            v = -ret
            if v > si[p, I_BEST]:
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
        self.nps = 300_000.0            # nodes/second estimate, re-calibrated at run time
        self.used = 0.0                 # wall-clock seconds spent in go() so far
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
            stats = np.array([0, 1 << 60, 0], dtype=np.int64)
            search_root(me, opp, order, len(order), 2, stats, self.W, vals)      # compile
            stats[:] = (0, 1 << 60, 0)
            t = time.time()
            search_root(me, opp, order, len(order), 7, stats, self.W, vals)      # calibrate
            dt = max(time.time() - t, 1e-4)
            self.nps = max(50_000.0, stats[0] / dt)
        except Exception:
            if DEBUG_RAISE:
                raise

    # ---- time management ----------------------------------------------------
    def _budget(self, empties):
        moves_left = max(1, (empties + 1) // 2)
        soft = (GAME_BUDGET_SEC - self.used) / moves_left
        b = min(soft * 1.5, self.time_out * PER_MOVE_FRACTION)
        return max(0.05, b)

    # ---- root search with iterative deepening ---------------------------------
    def _search(self, me, opp, root_moves, budget):
        t0 = time.time()
        deadline = t0 + budget
        empties = 64 - popcnt(me | opp)
        # initial order: fewest flips first
        scored = []
        for sq in root_moves:
            b = U64(1 << sq)
            scored.append((int(popcnt(U64(flips(me, opp, b)))), sq))
        scored.sort()
        order = [sq for _, sq in scored]
        best = order[0]
        stats = np.zeros(3, dtype=np.int64)

        depths = list(range(1, MAX_DEPTH + 1))
        if empties <= ENDGAME_EMPTIES:
            depths = [int(empties)]                      # exact solve, single iteration
        last_iter = 0.0
        for depth in depths:
            remaining = deadline - time.time()
            if remaining <= 0.01:
                break
            if last_iter > 0 and last_iter * 4 > remaining:   # next iteration will not fit
                break
            limit = int(remaining * self.nps)
            stats[0], stats[1], stats[2] = 0, max(limit, 1000), 0
            it0 = time.time()
            n = len(order)
            arr = np.array(order, dtype=np.int64)
            vals = np.full(n, -INF, dtype=np.int64)
            bi = search_root(me, opp, arr, n, depth, stats, self.W, vals)
            iter_best = order[bi] if bi >= 0 else None
            results = [(int(vals[i]), order[i]) for i in range(n)]
            dt = time.time() - it0
            if stats[2] != 0:
                break                                    # aborted: keep the previous depth's answer
            if stats[0] > 20_000 and dt > 1e-3:
                self.nps = 0.5 * self.nps + 0.5 * (stats[0] / dt)
            best = iter_best
            # PV move first, then by score
            results.sort(key=lambda t: -t[0])
            order = [sq for _, sq in results]
            last_iter = dt
        return best

    # ---- platform entry point ---------------------------------------------------
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        t_start = time.time()
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
            try:
                empties = int(64 - popcnt(me | opp))
                best_sq = self._search(me, opp, moves, self._budget(empties))
            except Exception:
                if DEBUG_RAISE:
                    raise
                best_sq = moves[0]                       # never crash, never illegal
            # ALL legal moves first, the chosen one LAST (platform takes the last element)
            self.candidate_list.extend(all_moves)
            self.candidate_list.append(sq_to_move(best_sq))
        except Exception:
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
            self.used += time.time() - t_start
