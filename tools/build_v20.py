"""Round 10: build a candidate from v18's source and a tools/fit_patterns10.py model.

    .venv/bin/python -I tools/build_v20.py artifacts/claude_round10/fit_frozen.json \
        versions/v20_patterns_candidate_ai.py

Deterministic text transformation of versions/v18_patterns_candidate_ai.py: v18's rows
10-19 and tables stay as they are; rows 20-22 get the fitted 11 weights and one int8 table
per shape (one bucket), expanded at import from "folded-index value" text (mirror entries
filled in; shapes without a mirror are stored in full). evaluate() gains one branch for
20-22 empties with one unrolled lookup per shape. A model without shapes (the linear-only
fallback, --set L) only replaces the three rows. Everything else is v18's code.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from fit_patterns9 import SHAPES  # noqa: E402


def replace_once(src, old, new):
    assert src.count(old) == 1, ("anchor not unique", old[:70])
    return src.replace(old, new)


def text(values):
    return " ".join(f"{i} {v}" for i, v in enumerate(values) if v)


def main():
    data, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    fit, train = data["model"], data["report"]["train"]
    shapes = fit["shapes"]
    src = (ROOT / "versions/v18_patterns_candidate_ai.py").read_text()
    rows = "\n".join(f"    {list(fit['rows'][str(e)])},  # {e} empties" for e in range(20, 23))
    if shapes:
        what = (f"add shape patterns ({', '.join(shapes)}), fitted on exact margins\n# of {train:,} self-play "
                f"training positions (ridge, integer scale {fit['scale']}; NOTES.md Round 10). Each row\n"
                "# keeps its own 11 weights; the pattern tables are shared by the three rows.")
    else:
        what = (f"are refitted on exact margins of {train:,} self-play training\n# positions (ridge, integer "
                f"scale {fit['scale']}; linear-only fallback, no patterns; NOTES.md Round 10).")
    parts = [f'''

# Round 10: rows 20-22 {what}
PATTERN_ROWS_HI = np.array([
{rows}
], dtype=np.int64)
DEFAULT_W[20:23] = PATTERN_ROWS_HI
''']
    if shapes:
        parts.append('''

def _expand_hi(text, digits, mirror):
    """Full int8 lookup table: entry i holds the fitted value of min(i, mirror(i))
    (of i itself when the shape has no mirror). Built digit by digit in int32."""
    idx = np.arange(3 ** digits, dtype=np.int32)
    key = idx
    if mirror is not None:
        rest, mirrored = idx.copy(), np.zeros(3 ** digits, dtype=np.int32)
        for k in range(digits):
            mirrored += (rest % 3) * np.int32(3 ** mirror[k])
            rest //= 3
        key = np.minimum(idx, mirrored)
    flat = np.zeros(3 ** digits, dtype=np.int32)
    if text:
        pairs = np.array(text.split(), dtype=np.int32).reshape(-1, 2)
        assert np.abs(pairs[:, 1]).max() <= 127
        flat[pairs[:, 0]] = pairs[:, 1]
    return flat[key].astype(np.int8)

''')
    loops = []
    for name in shapes:
        inst, mirror = SHAPES[name]
        up = name.upper()
        parts.append(f'''HI_BITS_{up} = np.array([[1 << s for s in inst] for inst in {inst}], dtype=np.uint64)
_HI_TEXT_{up} = "{text(fit["tables"][name])}"
HI_PAT_{up} = _expand_hi(_HI_TEXT_{up}, {len(inst[0])}, {mirror})
''')
        loops.append(f'''        for i in range({len(inst)}):
            idx = 0
            for k in range({len(inst[0])}):
                bit = HI_BITS_{up}[i, k]
                if me & bit:
                    idx += POW3[k]
                elif opp & bit:
                    idx += 2 * POW3[k]
            value += np.int64(HI_PAT_{up}[idx])''')
    anchor = "PAT_CORNER = _expand(_CORNER_TEXT, 9, [3 * c + r for r in range(3) for c in range(3)])\n"
    src = replace_once(src, anchor, anchor + "".join(parts))
    if shapes:
        tail = "            value += np.int64(PAT_CORNER[b, idx])\n    return value\n"
        src = replace_once(src, tail, tail.replace(
            "    return value\n",
            "    if W.shape[0] > 3 and 20 <= empties <= 22:      # Round 10 shape patterns\n"
            + "\n".join(loops) + "\n    return value\n"))
    out.write_text(src)
    print(f"wrote {out} ({len(src) / 1024:.0f} KiB), shapes {shapes}")


if __name__ == "__main__":
    main()
