"""Second baseline for tools/arena.py: reversed-Reversi greedy = never take a corner if avoidable,
then FEWEST flips, ties broken at random.  Much stronger than random; a sanity check that the
evaluation sign conventions of ai.py are not upside down.  Same interface as ai.py."""
import random

COLOR_BLACK, COLOR_WHITE, COLOR_NONE = -1, 1, 0
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _flips(board, color, r, c):
    if board[r][c] != 0:
        return 0
    total = 0
    for dr, dc in _DIRS:
        rr, cc, seen = r + dr, c + dc, 0
        while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == -color:
            rr, cc, seen = rr + dr, cc + dc, seen + 1
        if seen and 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == color:
            total += seen
    return total


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.color, self.candidate_list = color, []
        self.rnd = random.Random(2026)

    def go(self, chessboard):
        self.candidate_list.clear()
        scored = []
        for r in range(8):
            for c in range(8):
                f = _flips(chessboard, self.color, r, c)
                if f:
                    corner = (r in (0, 7)) and (c in (0, 7))
                    scored.append(((corner, f, self.rnd.random()), (r, c)))
        if scored:
            scored.sort()
            self.candidate_list.extend(m for _, m in scored)
            self.candidate_list.append(scored[0][1])
