#!/usr/bin/env python3
"""Independent cross-check: decode a Duckomp-v7 solver equation file and
compare to the seed derived from sample.txt (or seed.txt if present).

Usage: python3 verify_solver.py [equation_file]
       (defaults to output/solver_equation.txt)
"""
import os
import re
import sys

sys.set_int_max_str_digits(3000000)

path = sys.argv[1] if len(sys.argv) > 1 else "output/PCS.txt"

lines = open(path).read().splitlines()
magic = 0
magic_sign = "+"
eq = None
for i, l in enumerate(lines):
    if l.startswith("Solver Equation:"):
        eq = lines[i + 1]
    elif l.startswith("Magic:"):
        v = l.split(":", 1)[1].strip()
        if v.startswith("-"):
            magic_sign = "-"
            v = v[1:]
        magic = int(v)
if eq is None:
    # Fall back to the legacy `equation  : seed = ...` single-line format.
    eq = next(l for l in lines if l.startswith("equation  :"))
    eq = eq.split(":", 1)[1].strip() if eq.startswith("equation  :") else eq
else:
    eq = eq.split("=", 1)[1].strip() if "=" in eq else eq

# The equation line ends with " ± magic"; strip that tail off. The signed
# value itself comes from the Magic: line above (or defaults to 0).
body = eq
tail = re.search(r"([+-])\s*(\d+)$", body)
body_nomagic = body[: tail.start()].rstrip() if tail else body

# Split into signed tokens: [first, op, token, op, token, ...]
parts = re.split(r"\s*([+-])\s*", body_nomagic)
# Parse each token unambiguously: c·n^e  OR  n^e  (the middle-dot is REQUIRED
# for a coefficient so multi-digit bases like 98^19 can't be mis-split).
term_re = re.compile(r"^(?:(\d+)\u00b7)?(\d+)\^(\d+)$")
seed = 0
for j in range(0, len(parts), 2):
    sign = "+" if j == 0 else parts[j - 1]
    tok = parts[j].strip()
    if not tok:
        continue
    m = term_re.match(tok)
    if not m:
        print("UNPARSED TOKEN:", repr(tok))
        sys.exit(1)
    c = int(m.group(1)) if m.group(1) else 1
    v = c * (int(m.group(2)) ** int(m.group(3)))
    seed += v if sign == "+" else -v
seed += magic if magic_sign == "+" else -magic

# Reference seed.
if os.path.exists("seed.txt"):
    seed0 = int("".join(c for c in open("seed.txt").read() if c.isdigit()))
else:
    diffs = [int(l) for l in open("sample.txt") if l.strip()]
    n = len(diffs) - 1
    low = min(abs(diffs[i + 1] - diffs[i]) for i in range(n))
    high = max(abs(diffs[i + 1] - diffs[i]) for i in range(n))
    base = high - low + 1
    seed0 = 0
    for i in range(n):
        seed0 = seed0 * base + (abs(diffs[i + 1] - diffs[i]) - low)

print("reconstructed bits:", seed.bit_length())
print("reference bits    :", seed0.bit_length())
print("EXACT MATCH       :", seed == seed0)
sys.exit(0 if seed == seed0 else 1)
