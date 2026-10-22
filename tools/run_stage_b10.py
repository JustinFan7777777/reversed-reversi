"""Round 10 Stage B (versions/round10_protocol.md, revisions 2-3) for one candidate.

    .venv/bin/python tools/run_stage_b10.py versions/v20_patterns_candidate_ai.py \
        artifacts/claude_round10/fit_frozen.json --out-dir artifacts/claude_round10/stage_b_p0

Run on an otherwise idle CPU (labelling finished). In order:
1. Suite: the README "Validate" commands (checked against README.md) in a temporary repo
   copy without Project Requirements/, .venv/, .python/, artifacts/ or .git/, with the
   candidate as its ai.py; tests/test_patterns_hi.py also gets --frozen (absolute path),
   which checks the engine entry by entry against the frozen model. Every command exits 0.
2. Startup: tools/bench_first_move.py 3 times in the copy; median import + __init__ at
   most 1.5 s.
3. Memory: tools/bench_memory.py (README arguments) 3 times each for v18 and the
   candidate, alternating; median import + init within +5 MB of v18's.
4. Timing stress: tools/bench_timing.py on v18, then on the candidate; the candidate's
   maximum at most v18's + 0.05 s and below 4.5 s.
5. Node rate: tools/bench_node_rate.py from this repo; standard ratio at least 0.85 (a
   sanity check), leaf-heavy ratio reported.
A failed step is recorded and the remaining steps still run. Every log and stage_b.json
(numbers, gates, passed) go to the output folder, which must not exist yet.
"""
import argparse
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
V18 = "versions/v18_patterns_candidate_ai.py"
SKIP = {"Project Requirements", ".venv", ".python", "artifacts", ".git"}
SUITE = (                                            # README "Validate", in order
    ["tools/lint_submission.py", "ai.py"],
    ["tools/check_engine.py", "--games", "40"],
    ["tests/test_search.py"],
    ["tests/test_resumption.py"],
    ["tests/test_interface.py"],
    ["tests/test_fallback.py"],
    ["tests/test_terminal.py"],
    ["tests/test_arena_limits.py"],
    ["tests/test_variants.py", "ai.py"],
    ["tests/test_advanced_search.py", "ai.py"],
    ["tests/test_pvs_inline.py"],
    ["tests/test_eval_table.py"],
    ["tests/test_patterns.py", "ai.py"],
    ["tests/test_patterns_hi.py", "ai.py"],
    ["tests/test_outcome_root.py", "versions/v17_outcome_first_candidate_ai.py"],
    ["tools/bench_first_move.py", "ai.py"],
    ["tools/bench_timing.py"],
    ["tools/bench_memory.py", "ai.py", "--moves", "60", "--time-out", "0.1"],
    ["tools/check_engine.py", "--games", "200", "--self-play", "--time-out", "0.005"],
)
MEMORY_ARGS = ["--moves", "60", "--time-out", "0.1"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readme_suite():
    text = (ROOT / "README.md").read_text()
    block = text.split("## Validate", 1)[1].split("```sh", 1)[1].split("```", 1)[0]
    return [line.split("#")[0].split()[1:] for line in block.splitlines()
            if line.startswith(".venv/bin/python ")]


def run(args, cwd, log):
    started = time.perf_counter()
    with open(log, "w") as fh:
        rc = subprocess.run([PY, *args], cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                            timeout=3600).returncode
    return rc, time.perf_counter() - started, Path(log).read_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("frozen", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    cand, frozen, out = args.candidate.resolve(), args.frozen.resolve(), args.out_dir.resolve()
    assert cand.is_file() and frozen.is_file() and cand != (ROOT / V18).resolve()
    assert readme_suite() == [list(c) for c in SUITE], "README Validate list changed"
    out.mkdir(parents=True, exist_ok=False)
    record = dict(protocol="versions/round10_protocol.md revisions 2-3",
                  protocol_sha256=sha(ROOT / "versions/round10_protocol.md"),
                  candidate=str(cand), candidate_sha256=sha(cand), frozen=str(frozen),
                  frozen_sha256=sha(frozen), baseline=V18, baseline_sha256=sha(ROOT / V18))
    tmp = Path(tempfile.mkdtemp(prefix="stage_b10_"))
    copy = tmp / "repo"
    shutil.copytree(ROOT, copy, ignore=lambda d, names: [
        n for n in names if n == "__pycache__" or (Path(d) == ROOT and n in SKIP)])
    shutil.copyfile(cand, copy / "ai.py")
    assert sha(copy / "ai.py") == sha(cand) and sha(copy / V18) == sha(ROOT / V18)

    print(time.strftime("%H:%M:%S"), "suite", flush=True)
    rows = []
    for i, cmd in enumerate(SUITE):
        cmd = list(cmd) + (["--frozen", str(frozen)] if cmd[0] == "tests/test_patterns_hi.py" else [])
        rc, secs, _ = run(cmd, copy, out / f"suite_{i:02d}_{Path(cmd[0]).stem}.log")
        rows.append(dict(command=" ".join(cmd), returncode=rc, seconds=round(secs, 1)))
        print(f"  {'ok ' if rc == 0 else 'FAIL'} {secs:7.1f} s  {' '.join(cmd)}", flush=True)
    record["suite"] = dict(commands=rows, passed=all(r["returncode"] == 0 for r in rows))

    print(time.strftime("%H:%M:%S"), "startup", flush=True)
    runs = []
    for k in range(3):
        rc, _, text = run(["tools/bench_first_move.py", "ai.py"], copy, out / f"startup_{k}.log")
        m = re.search(r"import ([\d.]+)s \| __init__ ([\d.]+)s", text)
        runs.append(float(m.group(1)) + float(m.group(2)) if rc == 0 and m else None)
    ok = None not in runs and statistics.median(runs) <= 1.5
    record["startup"] = dict(import_plus_init_seconds=runs, rule="median import + __init__ <= 1.5 s",
                             median=None if None in runs else statistics.median(runs), passed=ok)

    print(time.strftime("%H:%M:%S"), "memory", flush=True)
    mem = {"v18": [], "candidate": []}
    for k in range(3):
        for name, path in (("v18", V18), ("candidate", "ai.py")):
            rc, _, text = run(["tools/bench_memory.py", path, *MEMORY_ARGS], copy, out / f"memory_{name}_{k}.log")
            a = re.search(r"import ai\.py growth\s*:\s*([+-]?[\d.]+)", text)
            b = re.search(r"__init__ growth \(JIT warm-up\)\s*:\s*([+-]?[\d.]+)", text)
            mem[name].append(float(a.group(1)) + float(b.group(1)) if rc == 0 and a and b else None)
    valid = None not in mem["v18"] + mem["candidate"]
    diff = statistics.median(mem["candidate"]) - statistics.median(mem["v18"]) if valid else None
    record["memory"] = dict(import_plus_init_mb=mem, median_difference_mb=diff,
                            rule="median import + init within +5 MB of v18's", passed=valid and diff <= 5.0)

    print(time.strftime("%H:%M:%S"), "timing stress", flush=True)
    tmax = {}
    for name, extra in (("v18", ["--path", str(copy / V18)]), ("candidate", [])):
        js = out / f"timing_{name}.json"
        rc, _, _ = run(["tools/bench_timing.py", *extra, "--json", str(js)], copy, out / f"timing_{name}.log")
        tmax[name] = max(r["seconds"] for r in json.loads(js.read_text())) if rc == 0 else None
    ok = None not in tmax.values() and tmax["candidate"] <= tmax["v18"] + 0.05 and tmax["candidate"] < 4.5
    record["timing"] = dict(max_move_seconds=tmax, passed=ok,
                            rule="candidate maximum <= v18's + 0.05 s and < 4.5 s")

    print(time.strftime("%H:%M:%S"), "node rate", flush=True)
    js = out / "node_rate.json"
    rc, _, _ = run(["-I", "tools/bench_node_rate.py", V18, str(cand), "--out", str(js)], ROOT, out / "node_rate.log")
    if rc == 0:
        nr = json.loads(js.read_text())
        record["node_rate"] = dict(
            standard_ratio=nr["standard"]["nps_ratio"], leaf_heavy_ratio=nr["leaf_heavy"]["nps_ratio"],
            leaf_heavy_time_ratio=nr["leaf_heavy"]["time_ratio"],
            mean_depth={k: nr["standard"][k]["mean_depth"] for k in ("base", "candidate")},
            rule="standard ratio >= 0.85 (sanity check; leaf-heavy reported)",
            passed=nr["standard"]["nps_ratio"] >= 0.85)
    else:
        record["node_rate"] = dict(returncode=rc, passed=False)

    shutil.rmtree(tmp)
    record["passed"] = all(record[k]["passed"] for k in ("suite", "startup", "memory", "timing", "node_rate"))
    (out / "stage_b.json").write_text(json.dumps(record, indent=1) + "\n")
    print(time.strftime("%H:%M:%S"), "Stage B " + ("PASSED" if record["passed"] else "FAILED") + ":",
          {k: record[k]["passed"] for k in ("suite", "startup", "memory", "timing", "node_rate")}, flush=True)


if __name__ == "__main__":
    main()
