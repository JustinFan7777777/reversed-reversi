"""Validate platform JSON board traces and audit exact late-game decisions.

Reads archive members as data; never executes attachments or extracts ZIP paths.
Each transition is independently replayed with ref_engine. Bounded offline search
only reports completed exact results, never treating an interruption as a draw.
Outputs stay in artifacts/ by default; the submitted AI is not changed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ai as A
import ref_engine as R

CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
NAME = re.compile(r"(\d+)_(\d+)_(black|white)_(win|lose|loss|draw)\.txt")


def load_games(archive):
    games = []
    with ZipFile(archive) as z:
        for member in sorted(z.infolist(), key=lambda x: x.filename):
            if member.is_dir() or member.filename.startswith("__MACOSX/"):
                continue
            name = PurePosixPath(member.filename).name
            match = NAME.fullmatch(name)
            if not match:
                continue
            assert member.file_size < 2_000_000, "Replay member unexpectedly large"
            raw = json.loads(z.read(member))
            frames = np.asarray(raw)
            assert frames.ndim == 3 and frames.shape[1:] == (8, 8), name
            assert 1 <= len(frames) <= 128 and np.isin(frames, [-1, 0, 1]).all(), name
            frames = frames.astype(np.int8)
            number, opponent, color, label = match.groups()
            me = -1 if color == "black" else 1
            board, expected = R.initial_board(), -1
            before, moves, passes = [], [], []
            for index, after in enumerate(frames):
                placed = np.argwhere((board == 0) & (after != 0))
                assert len(placed) == 1, (name, index, "expected one placement")
                row, col = map(int, placed[0])
                actor = int(after[row, col])
                if actor != expected:
                    assert not R.legal_moves(board, expected), (name, index, "illegal pass")
                    passes.append(dict(before_ply=index + 1, color=expected))
                legal = R.legal_moves(board, actor)
                assert (row, col) in legal, (name, index, "illegal placement")
                assert np.array_equal(R.apply_move(board, actor, (row, col)), after), (
                    name, index, "incorrect flips")
                before.append(board.copy())
                moves.append(dict(ply=index + 1, actor=actor, row=row, col=col,
                                  empties_before=int((board == 0).sum()),
                                  legal_count=len(legal),
                                  noncorner_choices=sum(m not in CORNERS for m in legal),
                                  corner=(row, col) in CORNERS))
                board, expected = after, -actor
            mine, theirs = int((board == me).sum()), int((board == -me).sum())
            result = "win" if mine < theirs else "lose" if mine > theirs else "draw"
            assert ("lose" if label == "loss" else label) == result, (name, "result label")
            terminal = not R.legal_moves(board, -1) and not R.legal_moves(board, 1)
            games.append(dict(name=name, number=number, opponent=opponent, color=color,
                me=me, result=result, mine=mine, theirs=theirs, terminal=terminal,
                empty_at_end=int((board == 0).sum()), placements=len(moves), passes=passes,
                corners_me=sum(board[r, c] == me for r, c in CORNERS),
                corners_opp=sum(board[r, c] == -me for r, c in CORNERS),
                my_discs_after_20=int((frames[min(19, len(frames)-1)] == me).sum()),
                my_discs_after_40=int((frames[min(39, len(frames)-1)] == me).sum()),
                corner_events=[dict(ply=m["ply"], side="me" if m["actor"] == me else "opp",
                                    row=m["row"], col=m["col"],
                                    noncorner_choices=m["noncorner_choices"])
                               for m in moves if m["corner"]],
                trace_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                final_sha256=hashlib.sha256(board.tobytes()).hexdigest(),
                moves=moves, audit=[], _frames=frames, _before=before))
    return games


def outcome(value):
    return "win" if value > 0 else "lose" if value < 0 else "draw"


def margin(value):
    return value - A.WIN_BASE if value > 0 else value + A.WIN_BASE if value < 0 else 0


def audit(games, seconds, max_empties, position_seconds, min_empties=0,
          results=("win", "lose", "draw")):
    A.DEBUG_RAISE = True
    A.AI(8, -1, 5)  # Compile before starting the bounded analysis clock.
    started = time.perf_counter()
    deadline = started + seconds
    su = np.zeros((A.MAXPLY, A.SU_COLS), dtype=np.uint64)
    si = np.zeros((A.MAXPLY, A.SI_COLS), dtype=np.int64)
    cache, nodes = {}, 0

    def solve(board, color):
        nonlocal nodes
        me, opp = A.board_to_bb(board, color)
        key = (int(me), int(opp))
        if key in cache:
            return cache[key]
        stop = min(deadline, time.perf_counter() + position_seconds)
        stats = np.zeros(7, dtype=np.int64)
        while time.perf_counter() < stop:
            stats[0], stats[1], stats[2] = 0, 4096, 0
            value = int(A.alphabeta(me, opp, int((board == 0).sum()), -A.INF, A.INF,
                                   stats, A.DEFAULT_W, su, si))
            nodes += int(stats[0])
            if stats[2] != 2:
                assert value == 0 or abs(value) > A.WIN_BASE, "Exact search returned a heuristic score"
                cache[key] = value
                return value
        return None

    candidates = [(m["empties_before"], g, i, m) for g in games
                  for i, m in enumerate(g["moves"])
                  if m["actor"] == g["me"] and g["result"] in results and
                  min_empties <= m["empties_before"] <= max_empties]
    # Complete shallow suffixes for every game before attempting earlier states.
    candidates.sort(key=lambda item: (item[0], item[1]["name"]))
    for empties, game, index, move in candidates:
        if time.perf_counter() >= deadline:
            break
        before = game["_before"][index]
        after = game["_frames"][index]
        best = solve(before, game["me"])
        played_child = solve(after, -game["me"])
        played = -played_child if played_child is not None else None
        row = dict(ply=move["ply"], empties=empties,
                   move=[move["row"], move["col"]], legal_count=move["legal_count"],
                   complete=best is not None and played is not None)
        if row["complete"]:
            row.update(best_value=best, played_value=played,
                       best_outcome=outcome(best), played_outcome=outcome(played),
                       best_margin=margin(best), played_margin=margin(played),
                       outcome_blunder=outcome(best) != outcome(played),
                       margin_regret=margin(best) - margin(played))
            assert best >= played, (game["name"], row)
            if row["outcome_blunder"]:
                options = []
                for mv in R.legal_moves(before, game["me"]):
                    child = R.apply_move(before, game["me"], mv)
                    value = solve(child, -game["me"])
                    if value is not None:
                        options.append(dict(move=list(mv), outcome=outcome(-value),
                                            margin=margin(-value)))
                row["verified_alternatives"] = options
        game["audit"].append(row)
    for game in games:
        game["audit"].sort(key=lambda r: r["ply"])
    return dict(seconds=time.perf_counter()-started, budget_seconds=seconds,
                max_empties=max_empties, per_position_seconds=position_seconds,
                nodes=nodes, exact_positions_cached=len(cache))


def build_summary(games):
    counts = Counter(g["result"] for g in games)
    opponents = defaultdict(list)
    for game in games:
        opponents[game["opponent"]].append(game)
    repeats = []
    for opponent, group in opponents.items():
        for color in ("black", "white"):
            same = [g for g in group if g["color"] == color]
            for i, a in enumerate(same):
                for b in same[i+1:]:
                    common = 0
                    for x, y in zip(a["_frames"], b["_frames"]):
                        if not np.array_equal(x, y):
                            break
                        common += 1
                    repeats.append(dict(a=a["name"], b=b["name"], opponent=opponent,
                        color=color, common_prefix_placements=common,
                        same_complete_trace=a["trace_sha256"] == b["trace_sha256"],
                        same_final_board=a["final_sha256"] == b["final_sha256"]))
    return dict(games=len(games), unique_opponents=len(opponents), outcomes=dict(counts),
                placements=sum(g["placements"] for g in games),
                all_terminal=all(g["terminal"] for g in games),
                colors={c:dict(Counter(g["result"] for g in games if g["color"] == c))
                        for c in ("black", "white")},
                opponent_results={o:dict(Counter(g["result"] for g in group))
                                  for o, group in opponents.items()}, repeats=repeats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--max-empties", type=int, default=14)
    ap.add_argument("--position-seconds", type=float, default=0.25)
    ap.add_argument("--min-empties", type=int, default=0)
    ap.add_argument("--audit-results", default="win,lose,draw")
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/platform_replay_audit.json")
    args = ap.parse_args()
    games = load_games(args.archive)
    assert games, "No replay files matched the documented filename convention"
    summary = build_summary(games)
    print(json.dumps(summary, indent=2), flush=True)
    search = audit(games, args.seconds, args.max_empties, args.position_seconds,
                   args.min_empties, tuple(args.audit_results.split(",")))
    for game in games:
        complete = [r for r in game["audit"] if r["complete"]]
        errors = [r for r in complete if r["outcome_blunder"]]
        print(game["name"], "WDL", game["mine"], game["theirs"],
              "exact decisions", len(complete), "outcome errors", errors, flush=True)
        if complete:
            print("  earliest exact:", complete[0], flush=True)
    public = [{k:v for k,v in game.items() if not k.startswith("_")} for game in games]
    # Convert numpy boolean corner counts to native integers for JSON serialization.
    for game in public:
        game["corners_me"], game["corners_opp"] = int(game["corners_me"]), int(game["corners_opp"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(source=str(args.archive), me="me",
        summary=summary, search=search, games=public), indent=2) + "\n")
    print("Search:", search)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
