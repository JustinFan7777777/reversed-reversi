"""Instructor clarification: 180 seconds PER SIDE; import+init separately <=5 s."""
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
import arena
import ref_engine as R


def main():
    clock = [0.0]

    class Player:
        def __init__(self, chessboard_size, color, time_out):
            self.color, self.candidate_list = color, []

        def go(self, board):
            clock[0] += 3.5
            self.candidate_list[:] = R.legal_moves(board, self.color)

    module = SimpleNamespace(AI=Player)
    with patch.object(arena.time, "perf_counter", lambda: clock[0]):
        _, reason, used = arena.play(module, module, 5)
        assert reason == "normal" and sum(used.values()) > 180
        assert max(used.values()) < 180
        _, reason, _ = arena.play(module, module, 5, game_limit=50)
        assert reason == "timeout"

        class SlowInit(Player):
            def __init__(self, *args):
                clock[0] += 5.1
                super().__init__(*args)

        winner, reason, _ = arena.play(SimpleNamespace(AI=SlowInit), module, 5)
        assert winner == 1 and reason == "initialization timeout"
    print("OK: independent 180-second player budgets and separate 5-second startup")


if __name__ == "__main__":
    main()
