"""Compare frozen engines on common platform states without changing submissions.

Run this separately from match trials: each engine gets the same timed search
budget and a cold transposition table. Exact verification has one additional,
shared deadline across roots and selected children; an unfinished proof is unknown.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import arena
import ref_engine as R


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "versions/platform_batch04_common_states.json"
DEFAULT_OUTPUT = ROOT / "artifacts/codex_round5/common_state_probe_results.json"


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_exact(module, value):
    """Decode a completed all-the-way search; reject any heuristic-looking value."""
    if value == 0:
        return {"encoded_value": 0, "margin": 0, "outcome": "draw"}
    base = int(module.WIN_BASE)
    if not base < abs(value) <= base + 64:
        raise AssertionError("Full-depth search returned a nonterminal value: %s" % value)
    margin = value - base if value > 0 else value + base
    return {"encoded_value": value, "margin": margin,
            "outcome": "win" if value > 0 else "lose"}


def exact_value(module, player, board, color, deadline):
    """Full-window negamax to zero empties, including free passes and early endings.

    Every new root/child has fresh stack and stats. Only slices of that same search
    resume the seven-element stats, so windows can never leak between searches.
    Results are for ``color``; callers negate children to recover root perspective.
    """
    started = time.perf_counter()
    empties = int(np.count_nonzero(board == 0))
    legal = R.legal_moves(board, color)
    other_legal = R.legal_moves(board, -color) if not legal else None
    info = dict(depth=empties, side_to_move=color, legal_count=len(legal),
                must_pass=not legal and bool(other_legal),
                terminal=not legal and not other_legal,
                complete=False, encoded_value=None, margin=None, outcome="unknown",
                nodes=0, seconds=0.0, reason="shared proof deadline reached")
    me, opp = module.board_to_bb(board, color)
    su = np.zeros((module.MAXPLY, module.SU_COLS), dtype=np.uint64)
    si = np.zeros((module.MAXPLY, module.SI_COLS), dtype=np.int64)
    stats = np.zeros(7, dtype=np.int64)
    while time.perf_counter() < deadline:
        # Preserve continuation fields 3:7 between slices; reset them only when
        # entering this function for another board/window.
        stats[0], stats[1], stats[2] = 0, 4096, 0
        value = int(module.alphabeta(me, opp, empties, -module.INF, module.INF,
                                     stats, player.W, su, si, player.tt))
        info["nodes"] += int(stats[0])
        if stats[2] == 0:
            info.update(decode_exact(module, value))
            info.update(complete=True, reason="full-depth full-window search completed")
            break
        if stats[2] != 2:
            info["reason"] = "search interrupted without a resumable result"
            break
    info["seconds"] = time.perf_counter() - started
    return info


def timed_search(module, board, color, legal, seconds, probe):
    """Construct one warmed player and run the internal search at the exact budget."""
    initialized = time.perf_counter()
    player = module.AI(8, color, 5)
    init_seconds = time.perf_counter() - initialized
    if getattr(player, "warmup_failures", 0):
        raise AssertionError("Engine warmup failed")
    if hasattr(player, "tt"):
        player.tt.fill(0)
    me, opp = module.board_to_bb(board, color)
    moves = module.bb_to_squares(module.legal_moves(me, opp))
    assert {tuple(module.sq_to_move(sq)) for sq in moves} == set(legal)
    started = time.perf_counter()
    # The platform states have legal moves; handle a general pass probe explicitly
    # because _search itself requires a nonempty root move list.
    square = int(player._search(me, opp, moves, seconds)) if moves else None
    elapsed = time.perf_counter() - started
    move = list(module.sq_to_move(square)) if square is not None else None
    assert move is None or tuple(move) in legal
    info = dict(move=move, square=square, completed_depth=int(player.last_depth),
                nodes=int(player.last_nodes), elapsed_seconds=elapsed,
                search_budget_seconds=seconds, init_seconds=init_seconds,
                module_import_seconds=float(module._import_seconds),
                last_value=int(player.last_value) if hasattr(player, "last_value") else None,
                last_solved=bool(player.last_solved) if hasattr(player, "last_solved") else None,
                must_pass=not bool(moves),
                matches_v13_original=move == probe["v13_original_move"],
                matches_v14_original=move == probe["v14_original_move"])
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="Equal internal-search budget for each engine and probe")
    parser.add_argument("--proof-seconds", type=float, default=10.0,
                        help="One shared exact-proof budget after timed comparisons")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0 < args.seconds <= 4 or args.proof_seconds < 0:
        parser.error("--seconds must be in (0, 4]; --proof-seconds must be nonnegative")

    paths = {"v13": ROOT / "versions/v13_round3_pvs_inline_ai.py",
             "v14": ROOT / "versions/v14_round4_fitted_end_ai.py",
             "candidate": args.candidate.resolve()}
    hashes = {name: source_hash(path) for name, path in paths.items()}
    modules = {name: arena.load(str(path), "common_probe_" + name)
               for name, path in paths.items()}
    for module in modules.values():
        module.DEBUG_RAISE = True
    raw_input = args.input.read_bytes()
    source = json.loads(raw_input)
    records = []
    boards = {}
    for probe in source["probes"]:
        board = np.asarray(probe["board_before"], dtype=np.int64)
        assert board.shape == (8, 8) and np.isin(board, [-1, 0, 1]).all()
        color = int(probe["color_to_move"])
        assert color in (-1, 1)
        empties = int(np.count_nonzero(board == 0))
        assert empties == int(probe["empties_before"])
        legal = R.legal_moves(board, color)
        assert set(legal) == {tuple(move) for move in probe["legal_moves"]}
        original = board.copy()
        row = dict(probe_id=probe["probe_id"], color_to_move=color,
                   empties_before=empties, placement_number=probe["placement_number"],
                   v13_original_move=probe["v13_original_move"],
                   v14_original_move=probe["v14_original_move"],
                   v13_filename=probe["v13_filename"], v14_filename=probe["v14_filename"],
                   searches={})
        for name, module in modules.items():
            row["searches"][name] = timed_search(module, board, color, legal, args.seconds, probe)
            print(probe["probe_id"], name, row["searches"][name], flush=True)
        assert np.array_equal(board, original)
        records.append(row)
        boards[probe["probe_id"]] = board

    candidate = modules["candidate"]
    # Warmup occurs outside the shared proof deadline, just as for timed searches.
    prover = candidate.AI(8, -1, 5)
    assert not getattr(prover, "warmup_failures", 0)
    prover.tt.fill(0)
    proof_started = time.perf_counter()
    deadline = proof_started + args.proof_seconds
    for row in records:
        if row["empties_before"] > 20:
            row["proof"] = dict(status="unknown", root=None, chosen_children=[],
                                reason="more than 20 empties; exact search deliberately skipped")
            continue
        board = boards[row["probe_id"]]
        color = row["color_to_move"]
        root = exact_value(candidate, prover, board, color, deadline)
        # Include recorded choices as well as every newly selected engine choice.
        # Duplicate moves are solved once and retain every source label.
        selected = {}
        for label, move in [("v13_original", row["v13_original_move"]),
                            ("v14_original", row["v14_original_move"])]:
            selected.setdefault(tuple(move), []).append(label)
        for name, result in row["searches"].items():
            if result["move"] is not None:
                selected.setdefault(tuple(result["move"]), []).append(name + "_timed")
        children = []
        for move, labels in selected.items():
            child = R.apply_move(board, color, move)
            child_result = exact_value(candidate, prover, child, -color, deadline)
            value = -child_result["encoded_value"] if child_result["complete"] else None
            root_result = decode_exact(candidate, value) if value is not None else dict(
                encoded_value=None, margin=None, outcome="unknown")
            entry = dict(move=list(move), sources=labels, child_side_result=child_result,
                         root_perspective=root_result)
            if root["complete"] and child_result["complete"]:
                assert root["encoded_value"] >= value, "Child exceeds proven root optimum"
                entry["optimal_margin"] = value == root["encoded_value"]
                entry["margin_regret"] = root["margin"] - root_result["margin"]
                entry["outcome_matches_optimum"] = root["outcome"] == root_result["outcome"]
            children.append(entry)
        row["proof"] = dict(status="exact" if root["complete"] else "unknown",
                            root=root, chosen_children=children,
                            reason="Completed roots/children only are reported as exact")

    for name, path in paths.items():
        assert source_hash(path) == hashes[name], "Engine source changed during the run"
    assert source_hash(args.input) == hashlib.sha256(raw_input).hexdigest(), "Probe input changed"
    report = dict(input=str(args.input.resolve()), input_sha256=hashlib.sha256(raw_input).hexdigest(),
                  engine_paths={name: str(path) for name, path in paths.items()},
                  engine_sha256=hashes, search_seconds=args.seconds,
                  proof_budget_seconds=args.proof_seconds,
                  proof_elapsed_seconds=time.perf_counter() - proof_started,
                  coordinate_system="zero-based row,column", records=records,
                  limitations=[
                      "Heuristic last_value scores at different evaluation rows, scales or depths are not directly comparable.",
                      "Cold-table internal searches do not reconstruct accumulated platform cache or timing history.",
                      "A move matching a historical winner is not proof of its optimality.",
                      "Only completed full-depth full-window searches are exact; deadline interruptions remain unknown.",
                      "The proof deadline is shared across all eligible roots and children, checked between 4096-node slices.",
                      "Child values are negated to the original root player's perspective, including forced passes and early endings.",
                  ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
