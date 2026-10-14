#!/usr/bin/env python3
"""Independent cross-check for the Duckomp-v7 PoC report (output/PCS.txt).

Rebuilds the seed from sample.txt exactly like src/main.rs, then checks it
against the plain decimal and the packed hex printed in the report.

Usage: python3 verify_packed.py [report_file]
       (defaults to output/PCS.txt)
"""
import re
import sys

sys.set_int_max_str_digits(3000000)

path = sys.argv[1] if len(sys.argv) > 1 else "output/PCS.txt"
lines = open(path).read().splitlines()
text = "\n".join(lines)

# Plain decimal seed: the "Seed" label line, then its indented digit line(s).
dec = None
for i, l in enumerate(lines):
    if l.strip() == "Seed:":
        parts = []
        for nxt in lines[i + 1:]:
            s = nxt.strip()
            if not s or nxt.startswith("  "):
                break
            parts.append(s)
        dec = "".join(parts)
        break
if not dec or not dec.isdigit():
    print("FAIL: could not read the plain seed from", path)
    sys.exit(1)
plain = int(dec)

# Packed hex (optional) and byte count.
mh = re.search(r"^Seed as hex\s*:\s*([0-9a-f]+)\s*$", text, re.M)
packed = int(mh.group(1), 16) if mh else None
mb = re.search(r"^Seed as bytes\s*:\s*(\d+) bytes", text, re.M)
nbytes = int(mb.group(1)) if mb else None

# Reference: rebuild the seed from the offsets exactly like main.rs.
off = [int(l) for l in open("sample.txt") if l.strip()]
diffs = [abs(off[i + 1] - off[i]) for i in range(len(off) - 1)]
low, high = min(diffs), max(diffs)
base = high - low + 1
seed = 0
for d in diffs:
    seed = seed * base + (d - low)

want = (seed.bit_length() + 7) // 8
checks = {"plain == re-derived": plain == seed}
if packed is not None:
    checks["packed == re-derived"] = packed == seed
if nbytes is not None:
    checks["byte count correct"] = nbytes == want
for name, ok in checks.items():
    print(f"  {'OK ' if ok else 'BAD'}  {name}")
print(f"  re-derived bits/bytes: {seed.bit_length()} / {want}")
print("EXACT MATCH :", all(checks.values()))
sys.exit(0 if all(checks.values()) else 1)
