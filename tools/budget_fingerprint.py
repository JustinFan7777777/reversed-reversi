"""Estimate the platform's effective search budget from our own recorded moves.

    .venv/bin/python -I tools/budget_fingerprint.py ARCHIVE.zip ENGINE.py --worker J --workers K \
        --out PART.json

For each game in the archive (our moves only, from the engine that played them), one AI
object per budget replays the game in order: at each of our decisions with >= 2 legal
moves it searches the recorded position with that budget (table kept across the game, as
in real play) and the chosen move is compared with the platform's move. Agreement by
budget estimates how much search the platform actually allowed. Archives are read as data.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import analyze_replays as P  # noqa: E402

BUDGETS = (0.25, 0.5, 1.0, 2.0, 4.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("engine", type=Path)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("fingerprint_engine", args.engine)
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)
    X.DEBUG_RAISE = True
    games = sorted(P.load_games(args.archive), key=lambda g: g["name"])
    rows = []
    for k, g in enumerate(games):
        if k % args.workers != args.worker:
            continue
        ais = {b: X.AI(8, g["me"], 5) for b in BUDGETS}
        for i, m in enumerate(g["moves"]):
            if m["actor"] != g["me"] or m["legal_count"] < 2:
                continue
            me, opp = X.board_to_bb(g["_before"][i], g["me"])
            moves = X.bb_to_squares(X.legal_moves(me, opp))
            played = m["row"] * 8 + m["col"]
            row = dict(game=g["name"], ply=m["ply"], empties=m["empties_before"], legal=len(moves))
            for b, ai in ais.items():
                sq = ai._search(me, opp, moves, b)
                row[str(b)] = dict(same=int(sq) == played, depth=ai.last_depth, solved=bool(ai.last_solved))
            rows.append(row)
            print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps(rows))
    args.out.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
