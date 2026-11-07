#!/usr/bin/env python3
"""Duckomp-v8 sample generator.

Writes sample.txt (one offset per line), like gen.py in v5-v7. The gaps are a
row-major W-wide field with smooth drift plus 8x8 checkerboard tiles, so the
grid rules in src/grid.rs have real spatial structure to find.

Modes
  python3 gen.py                 32x32 structured field (default)
  python3 gen.py grid W H        custom field size
  python3 gen.py big             64x64 structured field
  python3 gen.py sudoku N        Latin square order N (every row/column is a
                                 permutation of 1..N: max entropy, derivable)
  python3 gen.py volatile W H   larger sample whose rows swing sharply in
                                 entropy (default 32x192)
  python3 gen.py shuffle W H     same field, gaps shuffled (control)
  python3 gen.py random N        uniform gaps 1..899 (v7-style control)

Every mode writes ./sample.txt.  The RNG is seeded, so runs repeat exactly.
"""
import random
import sys

SEED = 7
BASE = 1
DRIFT_X = 3
DRIFT_Y = 5
TILE = 8
TILE_AMP = 200
NOISE = 3
RANDOM_MAX_GAP = 899


def grid_field(w, h):
    """Row-major 2-D gap field: smooth drift + checkerboard tiles + noise."""
    rng = random.Random(SEED)
    gaps = []
    for y in range(h):
        for x in range(w):
            tile = ((x // TILE) ^ (y // TILE)) & 1
            v = BASE + DRIFT_X * x + DRIFT_Y * y + TILE_AMP * tile + rng.randint(0, NOISE)
            gaps.append(max(1, v))
    return gaps


def latin_field(n):
    """A valid Latin square of order n: rows and columns are permutations.

    Rows are maximum entropy (every symbol appears once) yet each row is fully
    derivable from its multiset + rank, which is exactly what the de-entropy
    pass targets.
    """
    rng = random.Random(SEED)
    rows = list(range(n))
    rng.shuffle(rows)
    cols = list(range(n))
    rng.shuffle(cols)
    syms = list(range(1, n + 1))
    rng.shuffle(syms)
    return [syms[(rows[y] + cols[x]) % n] for y in range(n) for x in range(n)]


def volatile_field(w, h):
    """Rows with sharply different entropy, in blocks of four:

    0. permutation of 1..w  (maximum entropy, fully derivable)
    1. exact copy of the previous row
    2. rotated copy of the previous row
    3. constant row
    4. smooth ramp with noise (delta/Haar friendly)
    5. wide uniform noise (incompressible)
    """
    rng = random.Random(SEED)
    alphabet = list(range(1, w + 1))
    order = (0, 1, 2, 3, 4, 5)
    gaps = []
    for y in range(h):
        mode = order[(y // 4) % 6]
        if mode == 0:
            row = alphabet[:]
            rng.shuffle(row)
            gaps += row
        elif mode == 1:
            gaps += gaps[-w:] if y else [7] * w
        elif mode == 2:
            if y:
                prev = gaps[-w:]
                s = rng.randrange(1, w)
                gaps += [prev[(j + s) % w] for j in range(w)]
            else:
                gaps += [rng.randint(1, 9) for _ in range(w)]
        elif mode == 3:
            gaps += [rng.randint(1, 900)] * w
        elif mode == 4:
            start = rng.randint(1, 90)
            gaps += [start + 3 * j + rng.randint(0, 2) for j in range(w)]
        else:
            gaps += [rng.randint(1, 900) for _ in range(w)]
    return gaps


def to_offsets(gaps):
    """Prefix sums, so sample.txt stays strictly increasing like a scan."""
    out = [1]
    for g in gaps:
        out.append(out[-1] + g)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if mode == "big":
        w = h = 64
    elif mode in ("grid", "shuffle"):
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        h = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    elif mode == "sudoku":
        w = h = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    elif mode == "volatile":
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        h = int(sys.argv[3]) if len(sys.argv) > 3 else 192
    elif mode == "random":
        w = h = 0
    else:
        sys.exit(
            f"unknown mode {mode!r}: use grid W H | big | sudoku N | "
            f"volatile W H | shuffle W H | random N"
        )

    if mode == "random":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        rng = random.Random(SEED)
        gaps = [rng.randint(1, RANDOM_MAX_GAP) for _ in range(n)]
    elif mode == "sudoku":
        gaps = latin_field(w)
    elif mode == "volatile":
        gaps = volatile_field(w, h)
    else:
        gaps = grid_field(w, h)
        if mode == "shuffle":
            # Same multiset of gaps, spatial structure destroyed.
            random.Random(SEED).shuffle(gaps)

    offsets = to_offsets(gaps)
    with open("sample.txt", "w") as file:
        for offset in offsets:
            file.write(str(offset) + "\n")
    print(
        f"wrote sample.txt: mode={mode} offsets={len(offsets)} gaps={len(gaps)} "
        f"range {min(gaps)}..{max(gaps)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
