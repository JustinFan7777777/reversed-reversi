"""Static pre-submit checks.  python tools/lint_submission.py ai.py   (exit code 1 on failure)"""
import ast
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ai.py"
src = open(path, encoding="utf-8").read()
problems = []

# the platform bans the os module; check text AND syntax tree (comments/strings included)
if re.search(r"\bimport\s+os\b|\bfrom\s+os\b|__import__\(\s*['\"]os", src):
    problems.append("forbidden: the os module is referenced (even in a comment, to be safe)")
tree = ast.parse(src)
for node in ast.walk(tree):
    allowed = {"numpy", "random", "time", "numba"}
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] not in allowed:
                problems.append("unsupported submission import: %s" % alias.name)
    if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in allowed:
        problems.append("unsupported submission import: %s" % node.module)
    if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "os" for a in node.names):
        problems.append("forbidden import of os at line %d" % node.lineno)
    if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "os":
        problems.append("forbidden import from os at line %d" % node.lineno)

classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
if "AI" not in classes:
    problems.append("class AI is missing")
else:
    fns = {n.name: [a.arg for a in n.args.args] for n in classes["AI"].body if isinstance(n, ast.FunctionDef)}
    if fns.get("__init__") != ["self", "chessboard_size", "color", "time_out"]:
        problems.append("AI.__init__ signature changed: %r" % fns.get("__init__"))
    if fns.get("go") != ["self", "chessboard"]:
        problems.append("AI.go signature changed: %r" % fns.get("go"))

for pkg in ("scipy", "torch", "pandas", "sklearn", "tensorflow", "multiprocessing"):
    if re.search(r"^\s*(import|from)\s+%s\b" % pkg, src, re.M):
        problems.append("package not guaranteed on the platform: %s" % pkg)
if "cache=True" in src:
    print("WARNING: numba cache=True may fail on a read-only platform file system")

print("FAILED:\n  " + "\n  ".join(problems) if problems else "lint OK")
sys.exit(1 if problems else 0)
