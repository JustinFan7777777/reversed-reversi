"""Round 10 Stage C (versions/round10_protocol.md, revision 3): paired matches against v18.

    .venv/bin/python tools/run_stage_c10.py versions/v20_patterns_candidate_ai.py \
        --out-dir artifacts/claude_round10/stage_c_p0 [--fallback]

Budgets run in sequence, each as 3 parallel tools/run_paired_trial.py processes with seed
blocks S, S+1000, S+2000: 0.5 s (168/166/166 games, S = 23000000) and 1.0 s (same split,
S = 23100000), pooled by tools/pool_paired.py --gate interval (95% interval entirely above
0.5). Only if that passes: 2.0 s (100/100/100 games, S = 23200000), pooled with
--gate point (score at least 0.5). --fallback (the linear-only candidate) adds 300000 to
every S. The baseline is versions/v18_patterns_candidate_ai.py. Every process writes its
own JSON and log; pooled results and the decision go to stage_c.json. Nothing is
overwritten: the output folder must not exist yet. --dry-run replaces the plan with
2-game runs at 0.05 s (mechanics check only).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
BASE = ROOT / "versions/v18_patterns_candidate_ai.py"
PLAN = ((0.5, 23000000, (168, 166, 166)), (1.0, 23100000, (168, 166, 166)), (2.0, 23200000, (100, 100, 100)))
DRY = ((0.05, 99000000, (2, 2, 2)), (0.05, 99100000, (2, 2, 2)), (0.05, 99200000, (2, 2, 2)))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_budget(cand, out_dir, stage, seconds, seed, games):
    procs, outs = [], []
    for i, g in enumerate(games):
        out = out_dir / f"stage{stage}_{seconds}s_p{i}.json"
        assert not out.exists() and not out.with_suffix(".log").exists(), f"refusing to overwrite {out}"
        fh = open(out.with_suffix(".log"), "w")
        cmd = [PY, str(ROOT / "tools/run_paired_trial.py"), str(cand), str(BASE), "--games", str(g),
               "--search-seconds", str(seconds), "--seed", str(seed + 1000 * i), "--seconds", "100000",
               "--output", str(out)]
        procs.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT), fh))
        outs.append(out)
    for p, fh in procs:                              # wait for all three before judging any
        p.wait()
        fh.close()
    failed = [fh.name for p, fh in procs if p.returncode != 0]
    assert not failed, f"match process failed, see {failed}"
    return outs


def pool(outs, gate, dest):
    subprocess.run([PY, str(ROOT / "tools/pool_paired.py"), *map(str, outs), "--out", str(dest),
                    "--gate", gate], check=True, cwd=ROOT)
    return json.loads(dest.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cand = args.candidate.resolve()
    assert cand.is_file() and cand != BASE.resolve(), cand
    args.out_dir.mkdir(parents=True, exist_ok=False)
    plan = DRY if args.dry_run else PLAN
    offset = 300000 if args.fallback else 0
    (s1, b1, g1), (s2, b2, g2), (s3, b3, g3) = plan
    record = dict(protocol="versions/round10_protocol.md revision 3", candidate=str(cand),
                  baseline=str(BASE), fallback=args.fallback, dry_run=args.dry_run,
                  plan=[dict(seconds=s, seed_blocks=[b + offset + 1000 * i for i in range(3)], games=list(g))
                        for s, b, g in plan])
    log(f"{s1} s stage")
    outs = run_budget(cand, args.out_dir, 1, s1, b1 + offset, g1)
    log(f"{s2} s stage")
    outs += run_budget(cand, args.out_dir, 2, s2, b2 + offset, g2)
    first = pool(outs, "interval", args.out_dir / "pooled_short.json")
    record["pooled_short"] = {k: first[k] for k in ("games", "wins", "draws", "losses", "score",
                                                    "paired_ci95", "max_move_seconds", "gate")}
    short_ok, long_ok = bool(first["gate"]["passed"]), False
    if short_ok or args.dry_run:                     # a dry run exercises every stage
        log(f"{s3} s stage")
        long_outs = run_budget(cand, args.out_dir, 3, s3, b3 + offset, g3)
        second = pool(long_outs, "point", args.out_dir / "pooled_long.json")
        record["pooled_long"] = {k: second[k] for k in ("games", "wins", "draws", "losses", "score",
                                                        "paired_ci95", "max_move_seconds", "gate")}
        long_ok = bool(second["gate"]["passed"])
    else:
        record["pooled_long"] = "not run: the pooled short-budget gate failed"
    passed = short_ok and long_ok                    # both gates, never just the last one run
    record["passed"] = passed
    (args.out_dir / "stage_c.json").write_text(json.dumps(record, indent=1) + "\n")
    log("Stage C " + ("PASSED" if passed else "FAILED") + f": {args.out_dir / 'stage_c.json'}")


if __name__ == "__main__":
    main()
