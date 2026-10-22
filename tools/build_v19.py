"""Round 9: build a pattern candidate from v18's source and a tools/fit_patterns9.py model.

    .venv/bin/python -I tools/build_v19.py artifacts/claude_round9/fit_frozen.json \
        versions/v19_patterns_candidate_ai.py

Deterministic text transformation of versions/v18_patterns_candidate_ai.py: the Round 8
block (rows 10-19, two shapes) is replaced by the fitted rows and one int8 lookup table per
shape and bucket, expanded at import from "folded-index value" text (mirror entries filled
in; shapes without a mirror are stored in full); evaluate()'s pattern loop is regenerated
with one unrolled lookup per shape. Everything else is v18's code.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from fit_patterns9 import SHAPES  # noqa: E402

BUCKETS = ("10-13", "14-16", "17-19")


def replace_once(src, old, new):
    assert src.count(old) == 1, ("anchor not unique", old[:70])
    return src.replace(old, new)


def text(values):
    return " ".join(f"{i} {v}" for i, v in enumerate(values) if v)


def main():
    fit, out = json.loads(Path(sys.argv[1]).read_text())["model"], Path(sys.argv[2])
    shapes = fit["shapes"]
    src = (ROOT / "versions/v18_patterns_candidate_ai.py").read_text()
    a = src.index("\n# Round 8: rows 10-19 add shape patterns")
    b = src.index("PAT_CORNER = _expand(")
    b = src.index("\n", b) + 1
    rows = "\n".join(f"    {list(fit['rows'][str(e)])},  # {e} empties" for e in range(10, 20))
    parts = [f'''
# Rounds 8-9: rows 10-19 add shape patterns ({", ".join(shapes)}), fitted on exact margins
# of self-play positions (ridge, per-bucket integer scale; NOTES.md Round 9). Each row keeps
# its own 11 weights; pattern tables are shared within the buckets 10-13, 14-16 and 17-19.
PATTERN_ROWS = np.array([
{rows}
], dtype=np.int64)
DEFAULT_W[10:20] = PATTERN_ROWS
POW3 = 3 ** np.arange(10, dtype=np.int64)


def _expand(texts, digits, mirror):
    """Full int8 lookup tables: entry i holds the fitted value of min(i, mirror(i))
    (of i itself when the shape has no mirror). Built digit by digit in int32."""
    idx = np.arange(3 ** digits, dtype=np.int32)
    key = idx
    if mirror is not None:
        rest, mirrored = idx.copy(), np.zeros(3 ** digits, dtype=np.int32)
        for k in range(digits):
            mirrored += (rest % 3) * np.int32(3 ** mirror[k])
            rest //= 3
        key = np.minimum(idx, mirrored)
    tables = np.zeros((len(texts), 3 ** digits), dtype=np.int8)
    for b, text in enumerate(texts):
        flat = np.zeros(3 ** digits, dtype=np.int32)
        if text:
            pairs = np.array(text.split(), dtype=np.int32).reshape(-1, 2)
            assert np.abs(pairs[:, 1]).max() <= 127
            flat[pairs[:, 0]] = pairs[:, 1]
        tables[b] = flat[key]
    return tables

''']
    loops = []
    for name in shapes:
        inst, mirror = SHAPES[name]
        up = name.upper()
        texts = ",\n".join(f'    "{text(fit["tables"][k][name])}"' for k in BUCKETS)
        parts.append(f'''BITS_{up} = np.array([[1 << s for s in inst] for inst in {inst}], dtype=np.uint64)
_TEXT_{up} = (
{texts},
)
PAT_{up} = _expand(_TEXT_{up}, {len(inst[0])}, {mirror})
''')
        loops.append(f'''        for i in range({len(inst)}):
            idx = 0
            for k in range({len(inst[0])}):
                bit = BITS_{up}[i, k]
                if me & bit:
                    idx += POW3[k]
                elif opp & bit:
                    idx += 2 * POW3[k]
            value += np.int64(PAT_{up}[b, idx])''')
    src = src[:a] + "".join(parts) + src[b:]
    c = src.index("    if W.shape[0] > 3 and 10 <= empties <= 19:      # Round 8 shape patterns")
    d = src.index("    return value", c)
    src = src[:c] + ("    if W.shape[0] > 3 and 10 <= empties <= 19:      # Rounds 8-9 shape patterns\n"
                     "        b = 0 if empties <= 13 else (1 if empties <= 16 else 2)\n"
                     + "\n".join(loops) + "\n") + src[d:]
    out.write_text(src)
    print(f"wrote {out} ({len(src) / 1024:.0f} KiB), shapes {shapes}")


if __name__ == "__main__":
    main()
