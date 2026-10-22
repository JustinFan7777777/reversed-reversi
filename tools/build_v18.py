"""Round 8 Stage B: build the v18 pattern candidate from the frozen v14 snapshot and fit.

    .venv/bin/python -I tools/build_v18.py artifacts/claude_round8/fit_frozen.json \
        versions/v18_patterns_candidate_ai.py

Deterministic text transformation of versions/v14_round4_fitted_end_ai.py:
  * DEFAULT_W rows 10-19 become the fitted rows (11 weights each, per-bucket integer scale);
  * the fitted pattern tables are embedded as "folded-index value" text (nonzero entries
    only) and expanded at import into full lookup tables (mirror entries filled in);
  * evaluate() adds four edge+2X and four 3x3-corner lookups when the table has one row
    per empties count (W.shape[0] > 3) and 10 <= empties <= 19.
Nothing else changes; the build asserts each anchor occurs exactly once.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from fit_patterns import CORNER9, EDGE2X  # noqa: E402

BUCKET_KEYS = ("10-13", "14-16", "17-19")


def replace_once(src, old, new):
    assert src.count(old) == 1, ("anchor not unique", old[:60])
    return src.replace(old, new)


def table_text(values):
    return " ".join(f"{i} {v}" for i, v in enumerate(values) if v)


def main():
    fit, out = json.loads(Path(sys.argv[1]).read_text())["model"], Path(sys.argv[2])
    src = (ROOT / "versions/v14_round4_fitted_end_ai.py").read_text()
    rows = "\n".join(f"    {list(fit['rows'][str(e)])},  # {e} empties" for e in range(10, 20))
    edge = ",\n".join(f'    "{table_text(fit["tables"][k]["edge2x"])}"' for k in BUCKET_KEYS)
    corner = ",\n".join(f'    "{table_text(fit["tables"][k]["corner9"])}"' for k in BUCKET_KEYS)
    block = f'''
# Round 8: rows 10-19 add shape patterns, fitted on exact margins of 24,000 self-play
# positions (ridge, per-bucket integer scale; NOTES.md Round 8). Each row keeps its own
# 11 weights; pattern tables are shared within the buckets 10-13, 14-16 and 17-19.
PATTERN_ROWS = np.array([
{rows}
], dtype=np.int64)
DEFAULT_W[10:20] = PATTERN_ROWS
# Instances read from the side to move (digit k = 3**k: 0 empty, 1 own, 2 opponent).
EDGE_BITS = np.array([[1 << s for s in inst] for inst in {EDGE2X}], dtype=np.uint64)
CORNER_BITS = np.array([[1 << s for s in inst] for inst in {CORNER9}], dtype=np.uint64)
POW3 = 3 ** np.arange(10, dtype=np.int64)
# Nonzero entries "folded-index value" per bucket; a configuration and its mirror image
# (reversed edge, transposed corner block) share one entry.
_EDGE_TEXT = (
{edge},
)
_CORNER_TEXT = (
{corner},
)


def _expand(texts, digits, mirror):
    """Full lookup tables: entry i holds the fitted value of min(i, mirror(i)).
    Built digit by digit in int32 to keep import-time memory small (mirror is an
    involution, so digit k moves to position mirror[k])."""
    idx = np.arange(3 ** digits, dtype=np.int32)
    rest, mirrored = idx.copy(), np.zeros(3 ** digits, dtype=np.int32)
    for k in range(digits):
        mirrored += (rest % 3) * np.int32(3 ** mirror[k])
        rest //= 3
    key = np.minimum(idx, mirrored)
    tables = np.zeros((len(texts), 3 ** digits), dtype=np.int8)    # values fit -127..127
    for b, text in enumerate(texts):
        flat = np.zeros(3 ** digits, dtype=np.int32)
        pairs = np.array(text.split(), dtype=np.int32).reshape(-1, 2)
        flat[pairs[:, 0]] = pairs[:, 1]
        tables[b] = flat[key]
    return tables


PAT_EDGE = _expand(_EDGE_TEXT, 10, list(range(9, -1, -1)))
PAT_CORNER = _expand(_CORNER_TEXT, 9, [3 * c + r for r in range(3) for c in range(3)])
'''
    src = replace_once(src, "\n\n@njit(cache=False)\ndef terminal_value(me, opp):",
                       block + "\n\n@njit(cache=False)\ndef terminal_value(me, opp):")
    src = replace_once(src, '''        value += W[ph, 10] * positional
    return value''', '''        value += W[ph, 10] * positional
    if W.shape[0] > 3 and 10 <= empties <= 19:      # Round 8 shape patterns
        b = 0 if empties <= 13 else (1 if empties <= 16 else 2)
        for i in range(4):
            idx = 0
            for k in range(10):
                bit = EDGE_BITS[i, k]
                if me & bit:
                    idx += POW3[k]
                elif opp & bit:
                    idx += 2 * POW3[k]
            value += np.int64(PAT_EDGE[b, idx])
            idx = 0
            for k in range(9):
                bit = CORNER_BITS[i, k]
                if me & bit:
                    idx += POW3[k]
                elif opp & bit:
                    idx += 2 * POW3[k]
            value += np.int64(PAT_CORNER[b, idx])
    return value''')
    out.write_text(src)
    print(f"wrote {out} ({len(src) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
