"""Slow-but-obviously-correct numpy reference engine, used ONLY to cross-check ai.py."""
import numpy as np

BLACK, WHITE, NONE = -1, 1, 0
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def initial_board(size=8):
    # ASSUMPTION (verify against a downloaded replay file from the platform!):
    # white on (3,3),(4,4); black on (3,4),(4,3); black moves first.
    b = np.zeros((size, size), dtype=np.int64)
    lo, hi = size // 2 - 1, size // 2
    b[lo][lo] = b[hi][hi] = WHITE
    b[lo][hi] = b[hi][lo] = BLACK
    return b


def flips_for(board, color, r, c):
    """list of (r, c) discs flipped if `color` plays (r, c); [] if the move is illegal."""
    rows, cols = board.shape
    if board[r][c] != NONE:
        return []
    out = []
    for dr, dc in DIRS:
        line = []
        rr, cc = r + dr, c + dc
        while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == -color:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == color:
            out.extend(line)
    return out


def legal_moves(board, color):
    return [(r, c) for r in range(board.shape[0]) for c in range(board.shape[1])
            if flips_for(board, color, r, c)]


def apply_move(board, color, move):
    nb = board.copy()
    r, c = move
    fl = flips_for(board, color, r, c)
    assert fl, "illegal move %r" % (move,)
    nb[r][c] = color
    for rr, cc in fl:
        nb[rr][cc] = color
    return nb


def result(board):
    """Reversed Reversi: FEWER discs wins. Returns (black_count, white_count, winner)
    where winner is BLACK, WHITE or 0 (draw)."""
    nb = int((board == BLACK).sum())
    nw = int((board == WHITE).sum())
    if nb < nw:
        return nb, nw, BLACK
    if nw < nb:
        return nb, nw, WHITE
    return nb, nw, 0
