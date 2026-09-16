#!/usr/bin/env python3
"""
Duckomp Packer
==============
Reads the original file and equations, then:
  - Creates a cleaned file (zeroes out matched patterns via streaming)
  - Archives cleaned file + equations → .duckomp.tar.zst
  - Compresses original → .zstd for comparison
  - Compresses the full .duckomp archive to show real savings
"""
import struct
import subprocess
import re
import os
import sys
from pathlib import Path

BASE = Path(__file__).parent / "src"
SAMPLE = BASE / "sample.txt"
EQ = BASE / "duckomp_equations.txt"
CHUNK_SIZE = 64 * 1024 * 1024  # 64MB streams

def format_bytes(b):
    if b < 1024: return f"{b:,} B"
    elif b < 1024*1024: return f"{b:,} ({b/1024:.1f} KB)"
    else: return f"{b:,} ({b/1024/1024:.1f} MB)"

def parse_offsets_and_substr(eq_path):
    """Parse equations file, return (substr_bytes, n_occurrences, offsets_list)."""
    text = eq_path.read_bytes()
    lines = text.split(b'\n')
    substr = lines[0]
    n = int(lines[1])
    
    # Find # OFFSETS section
    offsets = []
    offsets_start = None
    for i, line in enumerate(lines):
        if line.strip() == b'# OFFSETS':
            offsets_start = i + 1
            break
    
    if offsets_start is not None:
        for line in lines[offsets_start:]:
            line = line.strip()
            if line:
                try: offsets.append(int(line))
                except ValueError: pass
    
    return substr, n, offsets

def main():
    print("=" * 68)
    print("  DUCKOOMP PACKER")
    print("=" * 68)
    print()
    
    if not SAMPLE.exists():
        print(f"ERROR: Sample not found: {SAMPLE}"); sys.exit(1)
    if not EQ.exists():
        print(f"ERROR: Equations not found: {EQ}"); sys.exit(1)
    
    sample_size = SAMPLE.stat().st_size
    eq_size = EQ.stat().st_size
    print(f"  Original: {SAMPLE.name} ({format_bytes(sample_size)})")
    print(f"  Equations: {EQ.name} ({format_bytes(eq_size)})")
    print()
    
    # Parse
    print("  Parsing equations...")
    substr, n, offsets = parse_offsets_and_substr(EQ)
    if not offsets:
        print("  ERROR: No offsets found. Re-run Duckomp so equations file has # OFFSETS.")
        sys.exit(1)
    substr_len = len(substr)
    raw_substr_bytes = n * substr_len
    print(f"    {n:,} occurrences of {substr_len}-byte pattern")
    print(f"    Offsets: {len(offsets):,}")
    print()
    
    # Clean the file (streaming)
    cleaned_path = BASE / "sample.cleaned"
    print("  Cleaning file (zeroing matched patterns)...")
    offsets_sorted = sorted(offsets)
    with open(SAMPLE, 'rb') as fin, open(cleaned_path, 'wb') as fout:
        pos = 0
        off_idx = 0
        n_offs = len(offsets_sorted)
        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk: break
            chunk_data = bytearray(chunk)
            chunk_start = pos
            chunk_end = pos + len(chunk_data)
            
            while off_idx < n_offs and offsets_sorted[off_idx] < chunk_start:
                off_idx += 1
            while off_idx < n_offs and offsets_sorted[off_idx] < chunk_end:
                local_off = offsets_sorted[off_idx] - chunk_start
                end = min(local_off + substr_len, len(chunk_data))
                for i in range(local_off, end):
                    chunk_data[i] = 0
                off_idx += 1
            
            fout.write(bytes(chunk_data))
            pos += len(chunk_data)
    cleaned_size = cleaned_path.stat().st_size
    print(f"    Cleaned: {format_bytes(cleaned_size)}")
    print()
    
    # Build .duckomp.tar: cleaned + equations
    print("  Building .duckomp archive...")
    duckomp_tar_path = BASE / f"{SAMPLE.stem}.duckomp.tar"
    eq_in_tar = Path("duckomp_equations.txt")
    cleaned_in_tar = Path("sample.cleaned")
    
    # Write tar manually to avoid tarfile's overhead with huge files
    import tarfile
    with tarfile.open(duckomp_tar_path, 'w') as tar:
        tar.add(cleaned_path, arcname=cleaned_in_tar.name)
        tar.add(EQ, arcname=eq_in_tar.name)
    duckomp_tar_size = duckomp_tar_path.stat().st_size
    print(f"    .duckomp.tar: {format_bytes(duckomp_tar_size)}")
    
    # zstd the archive
    duckomp_zst_path = duckomp_tar_path.with_suffix('.tar.zst')
    subprocess.run(['zstd', '-k', '-f', '-o', str(duckomp_zst_path), str(duckomp_tar_path)],
                  capture_output=True, check=True)
    duckomp_zst_size = duckomp_zst_path.stat().st_size
    print(f"    .duckomp.tar.zst: {format_bytes(duckomp_zst_size)} ({duckomp_tar_size/duckomp_zst_size:.2f}×)")
    print()
    
    # Clean up intermediate
    cleaned_path.unlink(missing_ok=True)
    duckomp_tar_path.unlink(missing_ok=True)
    
    # zstd original for comparison
    zstd_path = BASE / f"{SAMPLE.stem}.reference.zstd"
    print("  zstd on original (reference)...")
    subprocess.run(['zstd', '-k', '-f', '-o', str(zstd_path), str(SAMPLE)],
                  capture_output=True, check=True)
    zstd_size = zstd_path.stat().st_size
    print(f"    .zstd: {format_bytes(zstd_size)} ({sample_size/zstd_size:.2f}×)")
    print()
    
    # === Print summary ===
    print("=" * 68)
    print("  REAL-WORLD COMPARISON")
    print("=" * 68)
    print()
    print(f"  {'Method':40} {'Size':>18} {'Ratio':>10}")
    print(f"  {'-'*40} {'-'*18} {'-'*10}")
    print(f"  {'Original':40} {format_bytes(sample_size):>18} {'1.00×':>10}")
    print(f"  {'zstd(original) — reference':40} {format_bytes(zstd_size):>18} {sample_size/zstd_size:>9.2f}×")
    print(f"  {'Duckomp archive (clean+eqs)':40} {format_bytes(duckomp_tar_size):>18}")
    print(f"  {'zstd(Duckomp archive)':40} {format_bytes(duckomp_zst_size):>18} {duckomp_tar_size/duckomp_zst_size:>9.2f}×")
    print()
    print(f"  Raw substring data removed:  {format_bytes(raw_substr_bytes)}")
    print(f"  Equation overhead:           {format_bytes(eq_size)}")
    print(f"  Net savings (raw - eq):      +{format_bytes(raw_substr_bytes - eq_size)} ({(raw_substr_bytes - eq_size)/raw_substr_bytes*100:.1f}%)")
    print()
    
    delta = duckomp_zst_size - zstd_size
    if delta < 0:
        print(f"  ✓ Duckomp+zstd BEATS zstd alone by {format_bytes(-delta)} ({-delta/zstd_size*100:.1f}%)")
    elif delta > 0:
        print(f"  ○ zstd alone beats Duckomp+zstd by {format_bytes(delta)} ({delta/zstd_size*100:.1f}%)")
    else:
        print(f"  — Duckomp+zstd and zstd alone are the same size")
    print()
    print(f"  Total compression: {format_bytes(sample_size)} → {format_bytes(duckomp_zst_size)} ({sample_size/duckomp_zst_size:.2f}×)")
    print()
    print(f"  Output files:")
    print(f"    {zstd_path.name}")
    print(f"    {duckomp_zst_path.name}")
    print()
    print("=" * 68)

if __name__ == "__main__":
    main()