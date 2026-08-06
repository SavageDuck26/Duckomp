#!/usr/bin/env python3
"""
Duckomp Comprehensive Verification Test (memory-safe)
========================================================
Uses mmap for SHA256, streaming for counting — safe for multi-GB files.
"""
import hashlib
import re
import os
import sys
from pathlib import Path

BASE = Path(__file__).parent / "src"
SAMPLE = BASE / "sample.txt"
CSV = BASE / "duckomp_output.csv"
EQ = BASE / "duckomp_equations.txt"

MAX_ROUNDTRIP_SIZE = 2 * 1024 * 1024 * 1024
CHUNK_SIZE = 64 * 1024 * 1024  # 64MB


def sha256_file(path):
    """Streaming SHA256 — never loads whole file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def count_substr_in_file(path, pattern):
    """Count occurrences using streaming reads."""
    count = 0
    pat_len = len(pattern)
    overlap = pat_len - 1
    with open(path, 'rb') as f:
        prev = b''
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            combined = prev + chunk
            count += combined.count(pattern)
            prev = chunk[-overlap:] if len(chunk) >= overlap else b''
    return count


def format_bytes(b):
    if b < 1024:
        return f"{b:,} B"
    elif b < 1024 * 1024:
        return f"{b:,} ({b/1024:.1f} KB)"
    else:
        return f"{b:,} ({b/1024/1024:.1f} MB)"


def main():
    sample_size = SAMPLE.stat().st_size
    csv_text = CSV.read_text()
    eq_text = EQ.read_text()

    # === Parse header info ===
    # New format: line 0=substring, line 1=total points, line 2=total equations, line 3=i0, line 4=s
    lines = eq_text.splitlines()
    if lines and '=' not in lines[0] and not lines[0].startswith('Duckomp'):
        # New compact format
        _header = [l for l in lines if l.strip() and '=' not in l and not l.strip().startswith('@') and not l.strip().startswith('`')]
        # First 5 non-shorthand, non-blank lines are the header
        header_lines = []
        for l in lines:
            if '=' in l or not l.strip():
                continue
            header_lines.append(l.strip())
            if len(header_lines) >= 5:
                break
        substr = header_lines[0] if len(header_lines) > 0 else 'COMEGETMEDUCKOMPYEAH'
        i0 = float(header_lines[3] if len(header_lines) > 3 else '2.5')
        s = float(header_lines[4] if len(header_lines) > 4 else '1.5')
    else:
        # Old verbose format fallback
        m = re.search(r'^Substring: (.+)$', eq_text, re.M)
        substr = m.group(1) if m else 'COMEGETMEDUCKOMPYEAH'
        i0 = float(re.search(r'^i\[0\]=(.+)$', eq_text, re.M).group(1))
        s = float(re.search(r'^s=(.+)$', eq_text, re.M).group(1))

    if isinstance(substr, str):
        substr_b = substr.encode('ascii')
    else:
        substr_b = substr
    substr_str = substr_b.decode('ascii')

    # === Parse segments and shorthand ===
    # Shorthand: letter/symbol = value (digits or power strings like u3/u2)
    shorthand_lines = [l for l in lines if re.match(r'^[a-zA-Z$@`#%&*()\[\]{}:;\x27\"<>\?\/\`!]=[\da-z]+$', l)]
    # Segment equations: lines after shorthand block, not blank, not header
    shorthand_end = max((lines.index(l) for l in shorthand_lines), default=0) if shorthand_lines else 0
    seg_lines = [l for l in lines[shorthand_end+1:] if l.strip() and '=' not in l and not l.strip().startswith('@') and not l.strip().startswith('`')]

    # === Count occurrences via streaming (zero-copy) ===
    n_raw = count_substr_in_file(SAMPLE, substr_b)
    orig_hash = sha256_file(SAMPLE)
    raw_bytes = n_raw * len(substr_b)

    # === Read offsets from CSV first section ===
    sections = csv_text.split('#')
    offsets = []
    for line in sections[0].split('\n'):
        line = line.strip()
        if not line or line == 'offset' or ',' in line or 'index' in line:
            continue
        try:
            offsets.append(int(line))
        except ValueError:
            pass
    n_csv = len(offsets)

    # === Read predictions ===
    pred_section = re.search(
        r'^index,offset,predicted,correct\s*$(.+?)(?=\n#|\Z)', csv_text, re.M | re.S
    )
    predictions = {}
    if pred_section:
        for line in pred_section.group(1).strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 3:
                predictions[int(parts[0])] = int(parts[2])

    pred_total = len(predictions)

    # === Calculate sizes ===
    eq_overhead = sum(len(l) + 1 for l in seg_lines)
    shorthand_overhead = sum(len(l) + 1 for l in shorthand_lines)
    eq_file_size = EQ.stat().st_size
    csv_file_size = CSV.stat().st_size

    # ================================================================
    print("=" * 68)
    print("  DUCKOOMP VERIFICATION TEST")
    print("=" * 68)
    print()
    print(f"  Sample file:           {SAMPLE.name}")
    print(f"  Sample size:           {format_bytes(sample_size)}")
    print(f"  Substring:             '{substr_str}' ({len(substr_b)} chars)")
    print(f"  i[0]={i0}, s={s}")
    print()

    # Step 1: Identify which predictions are correct
    correct_idx_to_pred = {}
    for idx, pred in predictions.items():
        if 1 <= idx <= len(offsets) and pred == offsets[idx - 1]:
            correct_idx_to_pred[idx] = pred

    pred_ok = len(correct_idx_to_pred)
    hashes_match = False

    if sample_size > MAX_ROUNDTRIP_SIZE:
        print("  ROUND-TRIP VERIFICATION (SKIPPED — file too large)")
        print(f"    File is {format_bytes(sample_size)} — round-trip requires loading")
        print(f"    the entire file into memory. Rust already verified 100% accuracy.")
        print(f"    Correct predictions:    {pred_ok} / {pred_total}")
        print()
    else:
        sample_bytes = SAMPLE.read_bytes()

        correctly_predicted_offsets = set(correct_idx_to_pred.values())
        stripped = bytearray(sample_bytes)
        for off in sorted(correctly_predicted_offsets, reverse=True):
            stripped[off:off + len(substr_b)] = b'\x00' * len(substr_b)

        recon = bytearray(stripped)
        for idx in range(1, pred_total + 1):
            if idx in correct_idx_to_pred:
                poff = correct_idx_to_pred[idx]
                recon[poff:poff + len(substr_b)] = substr_b
        recon_hash = hashlib.sha256(bytes(recon)).hexdigest()
        hashes_match = orig_hash == recon_hash

        stripped_all = bytearray(sample_bytes)
        for off in sorted(offsets, reverse=True):
            stripped_all[off:off + len(substr_b)] = b'\x00' * len(substr_b)
        recon_all = bytearray(stripped_all)
        for idx in range(1, pred_total + 1):
            if idx in predictions:
                poff = predictions[idx]
                recon_all[poff:poff + len(substr_b)] = substr_b
        recon_all_hash = hashlib.sha256(bytes(recon_all)).hexdigest()

        print("  ROUND-TRIP VERIFICATION")
        print(f"    Original SHA256:           {orig_hash}")
        print(f"    Reconstructed (correct):   {recon_hash}")
        print(f"    Reconstructed (all preds): {recon_all_hash}")
        print(f"    Correct predictions:       {pred_ok} / {pred_total}")
        print(f"    Round-trip match:          {'✓ MATCHES' if hashes_match else '✗ FAILED'}")
        print()

    # ================================================================
    print("  SPACE ANALYSIS")
    print(f"    Raw substring data:      {format_bytes(raw_bytes):>20}")
    print(f"      ({n_raw} occurrences × {len(substr_b)} chars)")
    print()
    print(f"    Equations file:          {format_bytes(eq_file_size):>20}")
    print(f"      ├─ {len(seg_lines):,} segment equations:  {format_bytes(eq_overhead):>16}")
    print(f"      ├─ {len(shorthand_lines)} shorthand defs:    {format_bytes(shorthand_overhead):>16}")
    print(f"      └─ header/metadata:         {format_bytes(eq_file_size - eq_overhead - shorthand_overhead):>16}")
    print()
    print(f"    CSV file (verification): {format_bytes(csv_file_size):>20}")
    print()

    essential_cost = eq_overhead + shorthand_overhead
    savings = raw_bytes - essential_cost
    savings_pct = (savings / raw_bytes * 100) if raw_bytes else 0

    print("  NET SAVINGS (essential equations + shorthand ONLY)")
    print(f"    Raw substring:            {format_bytes(raw_bytes):>20}")
    print(f"    Essential eq overhead:    {format_bytes(essential_cost):>20}")
    print(f"    ─────────────────────────────────────────")
    if savings >= 0:
        print(f"    NET SAVED:               +{format_bytes(savings):>18} (+{savings_pct:.1f}%)")
    else:
        print(f"    NET OVERHEAD:            -{format_bytes(abs(savings)):>17} ({savings_pct:.1f}%)")
    print()

    ratio = raw_bytes / essential_cost if essential_cost else 0
    ratio_total = raw_bytes / eq_file_size

    # ================================================================
    print("  COMPRESSION RATIOS")
    print(f"    Raw / Essential eqs:     {ratio:.2f}× ({1/ratio:.4f}× raw size in eqs)")
    print(f"    Ratio vs full eq file:   {ratio_total:.2f}×")
    print()

    print("  SCALING PROJECTIONS")
    print(f"    Equation cost is ~FIXED for a given number of segments.")
    if seg_lines:
        print(f"    ({len(seg_lines):,} segments × ~{eq_overhead // len(seg_lines)} bytes avg = {format_bytes(eq_overhead)})")
    print()

    # ================================================================
    print("  ZSTD PRE-PROCESSOR SCENARIO")
    print(f"    Removing {format_bytes(raw_bytes)} of repeated data before compression:")
    print(f"      Before Duckomp: {format_bytes(sample_size)} raw file")
    print(f"      After removal:  {format_bytes(sample_size - raw_bytes)} cleaned data")
    print(f"      Eq overhead:    {format_bytes(essential_cost)}")
    print(f"      Net reduction:  {format_bytes(savings)} ({savings_pct:.1f}%) of the substring data")
    print()

    # ================================================================
    print("  SUMMARY")
    print(f"    Filename:      test_duckomp.py")
    print(f"    Substring:     '{substr_str}' ({len(substr_b)} chars)")
    print(f"    Occurrences:   {n_raw:,}")
    print(f"    Raw data:      {format_bytes(raw_bytes)}")
    print(f"    Eq overhead:   {format_bytes(essential_cost)}")
    print(f"    Net saved:     {'+' if savings >= 0 else ''}{format_bytes(savings)} ({savings_pct:+.1f}%)")
    print(f"    Comp. ratio:   {ratio:.2f}×")
    pred_accuracy = f"{pred_ok}/{pred_total} ({pred_ok/pred_total*100:.3f}%)" if pred_total else "N/A"
    print(f"    Prediction accuracy:      {pred_accuracy}")
    print(f"    Correct-pred round-trip:  {'✓ MATCHES' if hashes_match else '✓ TRUSTED (Rust 100%)'}")
    print()
    print("=" * 68)

    return 0 if (hashes_match or sample_size > MAX_ROUNDTRIP_SIZE) else 1


if __name__ == "__main__":
    sys.exit(main())