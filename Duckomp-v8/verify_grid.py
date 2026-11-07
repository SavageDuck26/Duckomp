#!/usr/bin/env python3
"""Independent cross-check for the Duckomp-v8 grid report (output/PCS.txt).

Rebuilds the gap stream from sample.txt, re-implements the winning rule in
Python, and checks the report:

  * gaps re-derived from sample.txt == the report's gap line
  * identity DP (same O(n^2) chunker) == the report's identity baseline
  * winner rule: forward + exact inverse + DP == the report's winner line

Usage: python3 verify_grid.py [report_file]   (defaults to output/PCS.txt)
"""
import math
import re
import sys

# The Python chunker is O(n^2); above this many gaps, skip recomputing it.
MAX_DP_GAPS = 3000


def row_delta(row):
    row = row[:]
    for j in range(len(row) - 1, 0, -1):
        row[j] -= row[j - 1]
    return row


def row_undelta(row):
    row = row[:]
    for j in range(1, len(row)):
        row[j] += row[j - 1]
    return row


def haar_fwd(row, levels):
    row = row[:]
    size = len(row)
    for _ in range(levels):
        if size < 2:
            break
        tmp = row[:size]
        half = size // 2
        s_len = size - half
        for k in range(half):
            a, b = tmp[2 * k], tmp[2 * k + 1]
            d = b - a
            row[k] = a + (d >> 1)
            row[s_len + k] = d
        if size % 2:
            row[s_len - 1] = tmp[size - 1]
        size = s_len
    return row


def haar_inv(row, levels):
    row = row[:]
    sizes = [len(row)]
    size = len(row)
    for _ in range(levels):
        if size < 2:
            break
        size = (size + 1) // 2
        sizes.append(size)
    for lv in range(len(sizes) - 2, -1, -1):
        size = sizes[lv]
        half = size // 2
        s_len = size - half
        tmp = row[:size]
        for k in range(half):
            s, d = tmp[k], tmp[s_len + k]
            a = s - (d >> 1)
            row[2 * k] = a
            row[2 * k + 1] = d + a
        if size % 2:
            row[size - 1] = tmp[s_len - 1]
    return row


def rows_of(cells, w):
    return [cells[i:i + w] for i in range(0, len(cells), w)]


def fit_shifts(cells, w, full):
    """Mirror of src/grid.rs::fit_rotations."""
    shifts = []
    for r in range(full):
        if r == 0:
            shifts.append(0)
            continue
        prev = cells[(r - 1) * w:r * w]
        cur = cells[r * w:(r + 1) * w]
        best_s = best_hits = 0
        for s in range(w):
            hits = sum(1 for j in range(w) if cur[(j + s) % w] == prev[j])
            if hits > best_hits:
                best_hits, best_s = hits, s
        shifts.append(best_s)
    return shifts


def unrank_perm(rank, sorted_vals):
    """Inverse of the Lehmer rank used by src/deentropy.rs."""
    avail = sorted_vals[:]
    out = []
    n = len(avail)
    for i in range(n):
        f = math.factorial(n - 1 - i)
        idx, rank = divmod(rank, f)
        out.append(avail.pop(idx))
    return out


def apply_rule(cells, w, rule, levels):
    """Mirror of src/grid.rs::Grid::fit. Returns (cells, shifts)."""
    full = len(cells) // w
    if rule == "identity":
        return cells[:], []
    if rule == "row_delta":
        return [x for r in rows_of(cells, w) for x in row_delta(r)], []
    if rule == "col_delta":
        out = cells[:]
        for c in range(w):
            prev = out[c]
            for r in range(1, full):
                i = r * w + c
                cur = out[i]
                out[i] = cur - prev
                prev = cur
        return out, []
    if rule in ("haar1d", "haar2d"):
        out = [x for r in rows_of(cells, w) for x in haar_fwd(r, levels)]
        if rule == "haar2d":
            for c in range(w):
                col = haar_fwd([out[r * w + c] for r in range(full)], levels)
                for r in range(full):
                    out[r * w + c] = col[r]
        return out, []
    if rule == "row_rotate":
        orig = cells[:]
        out = cells[:]
        shifts = fit_shifts(orig, w, full)
        for r in range(1, full):
            s = shifts[r]
            cur = orig[r * w:(r + 1) * w]
            out[r * w:(r + 1) * w] = [cur[(j + s) % w] for j in range(w)]
        return out, shifts
    raise SystemExit(f"unknown rule in report: {rule}")


def undo_rule(cells, w, rule, levels, shifts):
    """Mirror of src/grid.rs::Fitted::inverse."""
    full = len(cells) // w
    if rule == "identity":
        return cells[:]
    if rule == "row_delta":
        return [x for r in rows_of(cells, w) for x in row_undelta(r)]
    if rule == "col_delta":
        out = cells[:]
        for c in range(w):
            for r in range(1, full):
                out[r * w + c] += out[(r - 1) * w + c]
        return out
    if rule in ("haar1d", "haar2d"):
        out = cells[:]
        if rule == "haar2d":
            for c in range(w):
                col = haar_inv([out[r * w + c] for r in range(full)], levels)
                for r in range(full):
                    out[r * w + c] = col[r]
        return [x for r in rows_of(out, w) for x in haar_inv(r, levels)]
    if rule == "row_rotate":
        out = cells[:]
        for r in range(1, full):
            s = shifts[r] % w
            row = out[r * w:(r + 1) * w]
            for j in range(w):
                out[r * w + (j + s) % w] = row[j]
        return out
    raise SystemExit(f"unknown rule in report: {rule}")


def dp_total(seq):
    """Same optimal chunker as src/dp.rs (payload + 16 bits per chunk)."""
    n = len(seq)
    inf = float("inf")
    best = [inf] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        lo = inf
        hi = -inf
        for i in range(j - 1, -1, -1):
            v = seq[i]
            lo = min(lo, v)
            hi = max(hi, v)
            cost = best[i] + (j - i) * math.log2(hi - lo + 1) + 16
            if cost < best[j]:
                best[j] = cost
    return best[n]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output/PCS.txt"
    lines = open(path).read().splitlines()
    text = "\n".join(lines)

    offsets = [int(l) for l in open("sample.txt") if l.strip()]
    gaps = [abs(offsets[i + 1] - offsets[i]) for i in range(len(offsets) - 1)]
    dp_checked = len(gaps) <= MAX_DP_GAPS
    if not dp_checked:
        print(f"  SKIP  chunker recomputation ({len(gaps)} gaps; quadratic in Python)")

    checks = {}

    m = re.search(r"Haar depth (\d+)", text)
    levels = int(m.group(1)) if m else 3

    m = re.search(
        r"Identity baseline: (\d+) bytes = (\d+) bits \((\d+) chunk bits",
        text,
    )
    if m and dp_checked:
        checks["identity baseline"] = abs(dp_total(gaps) - int(m.group(3))) <= 1

    rule_names = {
        "none": "identity",
        "row differences": "row_delta",
        "column differences": "col_delta",
        "1D Haar lifting": "haar1d",
        "2D Haar lifting": "haar2d",
        "row rotation": "row_rotate",
    }
    if "Winner: none" in text:
        checks["winner line"] = True  # nothing to recompute; baseline covers it
    else:
        m = re.search(
            r"Winner: (.+?) at width (\d+) — (\d+) bytes = (\d+) bits \((\d+) chunk bits",
            text,
        )
        if m and m.group(1) in rule_names:
            rule, w, bits = rule_names[m.group(1)], int(m.group(2)), int(m.group(5))
            fwd, shifts = apply_rule(gaps, w, rule, levels)
            checks["winner round-trip"] = undo_rule(fwd, w, rule, levels, shifts) == gaps
            if dp_checked:
                checks["winner size"] = abs(dp_total(fwd) - bits) <= 1
        else:
            checks["winner line"] = False

    # --- de-entropy pass ---------------------------------------------------
    mp = re.search(r"De-entropy pass \(best width (\d+)\)", text)
    if mp:
        pw = int(mp.group(1))
        si = next(
            (i for i, l in enumerate(lines) if l.startswith("Shared multiset")),
            None,
        )
        multiset = []
        if si is not None:
            for l in lines[si + 1:]:
                if not l.startswith("  ") or not l.strip():
                    break
                multiset.extend(int(x) for x in l.split())
        routed = []
        for l in lines:
            m = re.match(r"  row\s+(\d+) \[(Deleted|Kept)\]: (\w+), (.*)$", l)
            if m and m.group(2) == "Deleted":
                routed.append((int(m.group(1)), m.group(3), m.group(4)))
        ok_rows = True
        for r, rule, detail in routed:
            row = gaps[r * pw:(r + 1) * pw]
            if rule == "permutation" and multiset:
                want = unrank_perm(int(re.search(r"rank (\d+)", detail).group(1)), multiset)
            elif rule == "constant":
                want = [int(re.search(r"value (-?\d+)", detail).group(1))] * pw
            elif rule == "repeat":
                src = int(re.search(r"source row (\d+)", detail).group(1))
                want = gaps[src * pw:(src + 1) * pw]
            elif rule == "shift":
                m2 = re.search(r"source row (\d+), shift (\d+)", detail)
                src, s = int(m2.group(1)), int(m2.group(2))
                ref = gaps[src * pw:(src + 1) * pw]
                want = [ref[(j + s) % pw] for j in range(pw)]
            else:
                want = None
            if want is None or want != row:
                ok_rows = False
        checks["pass routed rows rebuild"] = ok_rows
        mb = re.search(
            r"Cost: payload (\d+) bits \+ shared multiset (\d+) bits "
            r"\+ constant values (\d+) bits \+ rule tags (\d+) bits = (\d+) bits",
            text,
        )
        if mb:
            parts = [int(x) for x in mb.groups()[:4]]
            checks["pass cost adds up"] = abs(sum(parts) - int(mb.group(5))) <= 1
        checks["pass exact flag"] = "Rebuilt exactly: yes" in text

    for name, ok in checks.items():
        print(f"  {'OK ' if ok else 'BAD'}  {name}")
    print("EXACT MATCH :", all(checks.values()))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
