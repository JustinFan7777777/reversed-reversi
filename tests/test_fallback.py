"""Production exceptions preserve the complete legal set and increment diagnostics."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ai as A
import ref_engine as R


def fail(*args):
    raise RuntimeError("injected test failure")


def main():
    A.DEBUG_RAISE = True
    ai = A.AI(8, -1, 0.01)
    assert ai.warmup_failures == 0
    board = R.initial_board()
    A.DEBUG_RAISE = False
    ai._search = fail
    ai.go(board)
    assert ai.fallback_count == 1
    assert set(ai.candidate_list) == set(R.legal_moves(board, -1))
    A.DEBUG_RAISE = True
    try:
        ai.go(board)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Debug mode hid a search failure")
    # An engine failure must use the independent array implementation.
    A.DEBUG_RAISE = False
    actual = A.legal_moves
    A.legal_moves = fail
    try:
        ai.go(board)
        assert set(ai.candidate_list) == set(R.legal_moves(board, -1))
        assert ai.fallback_count >= 3
    finally:
        A.legal_moves = actual
        A.DEBUG_RAISE = True
    print("OK: search/engine fallback and debug exception propagation")


if __name__ == "__main__":
    main()
