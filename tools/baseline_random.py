"""Baseline opponent for tools/arena.py: uniformly random legal move (same interface as ai.py)."""
import random

import numpy as np

COLOR_BLACK, COLOR_WHITE, COLOR_NONE = -1, 1, 0
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _legal(board, color):
    out = []
    for r in range(8):
        for c in range(8):
            if board[r][c] != 0:
                continue
            for dr, dc in _DIRS:
                rr, cc, seen = r + dr, c + dc, 0
                while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == -color:
                    rr, cc, seen = rr + dr, cc + dc, seen + 1
                if seen and 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == color:
                    out.append((r, c))
                    break
    return out


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.color, self.candidate_list = color, []
        self.rnd = random.Random(12345)

    def go(self, chessboard):
        self.candidate_list.clear()
        moves = _legal(chessboard, self.color)
        if moves:
            self.candidate_list.extend(moves)
            self.candidate_list.append(self.rnd.choice(moves))
