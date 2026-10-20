"""Cross-check a candidate against completed exact values from the platform batch."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import analyze_replays as P
import arena


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--oracle", type=Path, default=ROOT / "artifacts/platform_replay_audit.json")
    args = ap.parse_args()
    reference = json.loads(args.oracle.read_text())
    games = {g["name"]: g for g in P.load_games(args.archive)}
    A = arena.load(str(args.candidate), "unregistered_oracle_candidate")
    A.DEBUG_RAISE = True
    player = A.AI(8, -1, 5)
    su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
    si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
    checked = 0
    for game in reference["games"]:
        for row in game["audit"]:
            if not row["complete"]:
                continue
            board = games[game["name"]]["_before"][row["ply"]-1]
            me, opp = A.board_to_bb(board, game["me"])
            stats = np.array([0, 1 << 60, 0], dtype=np.int64)
            values = (me, opp, row["empties"], -A.INF, A.INF, stats, player.W, su, si)
            got = A.alphabeta(*values, player.tt) if hasattr(player, "tt") else A.alphabeta(*values)
            assert got == row["best_value"], (game["name"], row["ply"], got, row["best_value"])
            assert stats[2] == 0
            checked += 1
    print("OK: candidate agrees with %d completed exact platform-position oracles" % checked)


if __name__ == "__main__":
    main()
