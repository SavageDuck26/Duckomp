use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use std::{fs, io::BufRead};
use rayon::prelude::*;
use memmap2::{Mmap, MmapMut};
use aho_corasick::{AhoCorasick, MatchKind};
use std::borrow::Cow;

const MIN_IMPROVEMENT_RATIO: f64 = 0.001; // 0.1% — commit anything that helps
const MAX_STAGES: usize = 16;
const MAX_CODES: usize = 254;
const MIN_CHUNK_SIZE: usize = 3;
const MAX_CHUNK_SIZE: usize = 128;

const SAMPLE_REGION_SIZE: usize = 256 * 1024; // 256 KB per parallel sample region
const MIN_CROSS_REF_SAMPLES: usize = 2;       // pattern must appear in >= N samples
const WRITE_BUF_SIZE: usize = 1024 * 1024;    // 1 MB write buffer
const ENCODE_CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4 MB per parallel encode chunk

// Chunked compression pipeline constants
const POST_DICT_CHUNK_SIZE: usize = 64 * 1024;         // 64 KB post-dict chunks for EGAP/LZ77
const DICT_STAGE_MIN_CHUNK: usize = 64 * 1024 * 1024;  // 64 MB minimum chunk for dict stages
const DICT_STAGE_MAX_CHUNK: usize = 512 * 1024 * 1024;  // 512 MB maximum chunk for dict stages
const RAM_BUDGET_FRACTION: f64 = 0.4;                  // use 40% of available RAM

// Adaptive stage selection constants
const ADAPTIVE_DICT_SKIP_ALL: f64 = 50.0;   // dict reduction >= 50% → skip EGAP+LZ77
const ADAPTIVE_DICT_HIGH: f64 = 20.0;       // dict reduction >= 20% → run EGAP
const ADAPTIVE_DICT_MEDIUM: f64 = 5.0;      // dict reduction >= 5% → run EGAP
const ADAPTIVE_EGAP_FLAT: f64 = 1.0;        // EGAP reduction < 1% → skip LZ77
const ADAPTIVE_EGAP_GOOD: f64 = 5.0;        // EGAP reduction >= 5% → always LZ77
const ADAPTIVE_ENTROPY_RANDOM: f64 = 7.5;   // entropy >= 7.5 → skip stage (near random)
const ADAPTIVE_ENTROPY_HIGH: f64 = 7.0;     // entropy >= 7.0 → run EGAP but skip LZ77

// Per-chunk adaptive stage flags
const CHUNK_FLAG_EGAP: u8 = 0x01;
const CHUNK_FLAG_LZ77: u8 = 0x02;
const CHUNK_FLAG_REVERSED: u8 = 0x04;
const CHUNK_FLAG_DEENTROPY: u8 = 0x08;
const CHUNK_FLAG_POST_ENTROPY: u8 = 0x10; // post-encoding entropy pass

// Format-level flags (in header format_flags byte)
const FORMAT_FLAG_FOLDER: u8 = 0x01;
const FORMAT_FLAG_LZ77: u8 = 0x02;

// ---------------------------------------------------------------------------
// Progress reporting helper for parallel phases
// ---------------------------------------------------------------------------

/// Print a carriage-return progress line. Thread-safe via atomic counter.
/// Returns the new count after incrementing.
fn report_progress(counter: &AtomicUsize, total: usize, label: &str) {
    let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
    // Only print at ~1% intervals (or every chunk if < 100 total)
    let interval = (total / 100).max(1);
    if done % interval == 0 || done == total {
        let pct = done as f64 / total as f64 * 100.0;
        eprint!("\r  [{label}] {done}/{total} ({pct:.1}%)");
        if done == total {
            eprintln!();
        }
    }
}

// ---------------------------------------------------------------------------
// De-entropy transform layer (between dict output and EGAP)
// ---------------------------------------------------------------------------

const DEENTROPY_MAX_SHIFTS: usize = 3;
const DEENTROPY_MIN_GAIN: usize = 16; // bytes — transform must save at least this much

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
enum DeentropyMode {
    None = 0,
    Shift = 1,
    Reverse = 2,
    ReverseShift = 3,
}

#[derive(Clone, Copy, Debug)]
struct ShiftEntry {
    symbol: u8,
    offset: i8, // {-1, 0, +1}
}

#[derive(Clone, Debug)]
struct DeentropyMeta {
    mode: DeentropyMode,
    shifts: Vec<ShiftEntry>,
}

impl DeentropyMeta {
    fn none() -> Self {
        DeentropyMeta { mode: DeentropyMode::None, shifts: vec![] }
    }

    /// Serialized size: 1 byte (mode) + 1 byte (count) + 2 bytes per shift entry.
    fn meta_size(&self) -> usize {
        2 + 2 * self.shifts.len()
    }

    fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.meta_size());
        out.push(self.mode as u8);
        out.push(self.shifts.len() as u8);
        for s in &self.shifts {
            out.push(s.symbol);
            out.push(s.offset as u8);
        }
        out
    }

    fn deserialize(data: &[u8]) -> io::Result<(DeentropyMeta, usize)> {
        if data.len() < 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated deentropy meta"));
        }
        let mode = match data[0] {
            0 => DeentropyMode::None,
            1 => DeentropyMode::Shift,
            2 => DeentropyMode::Reverse,
            3 => DeentropyMode::ReverseShift,
            x => return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid deentropy mode: {}", x),
            )),
        };
        let num_shifts = data[1] as usize;
        if num_shifts > DEENTROPY_MAX_SHIFTS {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Too many shift entries"));
        }
        let needed = 2 + 2 * num_shifts;
        if data.len() < needed {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated shift entries"));
        }
        let mut shifts = Vec::with_capacity(num_shifts);
        for i in 0..num_shifts {
            shifts.push(ShiftEntry {
                symbol: data[2 + 2 * i],
                offset: data[2 + 2 * i + 1] as i8,
            });
        }
        Ok((DeentropyMeta { mode, shifts }, needed))
    }
}

/// Build a conflict-free byte permutation from shift entries.
///
/// Each ShiftEntry requests `symbol → symbol + offset`. Conflicts (two symbols
/// targeting the same slot) are resolved by swapping: the displaced symbol takes
/// the vacated slot, maintaining a valid bijection at every step.
fn build_shift_permutation(shifts: &[ShiftEntry]) -> [u8; 256] {
    let mut fwd = [0u8; 256];
    let mut rev = [0u8; 256];
    for i in 0..256 {
        fwd[i] = i as u8;
        rev[i] = i as u8;
    }
    for entry in shifts {
        if entry.offset == 0 {
            continue;
        }
        let src = entry.symbol;
        let dst = (src as i16 + entry.offset as i16).rem_euclid(256) as u8;
        if fwd[src as usize] == dst {
            continue; // already in place
        }
        // Swap-based conflict resolution:
        // - `displaced_src` currently maps to `dst`; it must go somewhere else
        // - `old_dst` is where `src` currently maps; give that slot to displaced_src
        let displaced_src = rev[dst as usize];
        let old_dst = fwd[src as usize];
        fwd[src as usize] = dst;
        fwd[displaced_src as usize] = old_dst;
        rev[dst as usize] = src;
        rev[old_dst as usize] = displaced_src;
    }
    fwd
}

fn invert_permutation(perm: &[u8; 256]) -> [u8; 256] {
    let mut inv = [0u8; 256];
    for i in 0..256 {
        inv[perm[i] as usize] = i as u8;
    }
    inv
}

fn apply_byte_permutation(data: &[u8], perm: &[u8; 256]) -> Vec<u8> {
    data.iter().map(|&b| perm[b as usize]).collect()
}

/// Estimate the downstream cost (EGAP + LZ77) for a candidate buffer.
///
/// Shannon entropy alone is invariant under byte permutations (bijections preserve
/// the frequency histogram). To capture local structure changes that matter for
/// EGAP's gap phase and downstream LZ77, we blend order-0 entropy with a
/// digram-transition metric: fewer unique byte-pair transitions → more local
/// repetition → smaller LZ77 output.
fn estimate_egap_cost(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let n = data.len();
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    let nf = n as f64;

    // Order-0 Shannon entropy (bits per symbol)
    let h0: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.log2()
        })
        .sum();

    // Digram transition count — sensitive to byte permutations
    let mut transition_set = HashSet::new();
    if n > 1 {
        for w in data.windows(2) {
            transition_set.insert((w[0], w[1]));
        }
    }
    let unique_transitions = transition_set.len();

    // Blend: base cost from Shannon + penalty/bonus from transition diversity.
    // Max possible transitions = distinct^2; fewer transitions relative to that
    // means better local structure.
    let max_transitions = (distinct * distinct).max(1);
    let transition_ratio = unique_transitions as f64 / max_transitions as f64; // 0..1
    // Scale factor: high diversity → 1.0 (no help); low → down to ~0.7
    let structure_factor = 0.7 + 0.3 * transition_ratio;
    let data_bytes = ((h0 * nf * structure_factor) / 8.0).ceil() as usize;

    // EGAP header: "EGAP"(4) + original_size(4) + num_symbols(1) + padding(1) + 6*distinct
    let header = 10 + 6 * distinct;
    data_bytes + header
}

/// Run the de-entropy candidate search. Returns the best transform and transformed data.
///
/// Candidate space: 3^N offset combos × 2 byte orders (normal + reversed), where N = top-3
/// symbols by frequency. Each candidate is scored via `estimate_egap_cost`; the best is
/// accepted only if it clears the cost gate (MIN_GAIN bytes better than baseline).
fn deentropy_transform(data: &[u8]) -> (Vec<u8>, DeentropyMeta) {
    if data.len() < 64 {
        return (data.to_vec(), DeentropyMeta::none());
    }

    let baseline_cost = estimate_egap_cost(data);

    // Byte frequency analysis
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let mut freq_indices: Vec<(u64, u8)> = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (c, i as u8))
        .collect();
    freq_indices.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let top_n = freq_indices.len().min(DEENTROPY_MAX_SHIFTS);
    let top_symbols: Vec<u8> = freq_indices[..top_n].iter().map(|&(_, s)| s).collect();

    let offsets: [i8; 3] = [-1, 0, 1];
    let num_combos = 3usize.pow(top_n as u32);

    let mut best_cost = baseline_cost;
    let mut best_meta = DeentropyMeta::none();
    let mut best_data: Option<Vec<u8>> = None;

    // Try reverse-only first (no shifts)
    {
        let mut rev = data.to_vec();
        rev.reverse();
        let cost = estimate_egap_cost(&rev) + 2; // 2 bytes meta for Reverse mode
        if cost + DEENTROPY_MIN_GAIN <= best_cost {
            best_cost = cost;
            best_meta = DeentropyMeta { mode: DeentropyMode::Reverse, shifts: vec![] };
            best_data = Some(rev);
        }
    }

    // Try all 3^N offset combinations × 2 byte orders
    for combo in 0..num_combos {
        let mut shifts = Vec::with_capacity(top_n);
        let mut c = combo;
        let mut any_nonzero = false;
        for i in 0..top_n {
            let off = offsets[c % 3];
            c /= 3;
            shifts.push(ShiftEntry { symbol: top_symbols[i], offset: off });
            if off != 0 {
                any_nonzero = true;
            }
        }
        if !any_nonzero {
            continue; // skip identity
        }

        let perm = build_shift_permutation(&shifts);
        let active_shifts: Vec<ShiftEntry> = shifts.iter().copied().filter(|s| s.offset != 0).collect();
        let meta_overhead = 2 + 2 * active_shifts.len();

        for reversed in [false, true] {
            let mode = if reversed {
                DeentropyMode::ReverseShift
            } else {
                DeentropyMode::Shift
            };

            let working: Vec<u8> = if reversed {
                data.iter().copied().rev().collect()
            } else {
                data.to_vec()
            };

            let transformed = apply_byte_permutation(&working, &perm);
            let cost = estimate_egap_cost(&transformed) + meta_overhead;

            if cost + DEENTROPY_MIN_GAIN <= best_cost {
                best_cost = cost;
                best_meta = DeentropyMeta { mode, shifts: active_shifts.clone() };
                best_data = Some(transformed);
            }
        }
    }

    match best_data {
        Some(d) => (d, best_meta),
        None => (data.to_vec(), DeentropyMeta::none()),
    }
}

/// Invert the de-entropy transform. Order: undo shifts (inverse permutation), then undo reverse.
fn inverse_deentropy_transform(data: &[u8], meta: &DeentropyMeta) -> Vec<u8> {
    if meta.mode == DeentropyMode::None {
        return data.to_vec();
    }
    let mut result = data.to_vec();

    // Undo byte permutation (applied second during compression → undo first)
    if !meta.shifts.is_empty()
        && (meta.mode == DeentropyMode::Shift || meta.mode == DeentropyMode::ReverseShift)
    {
        let perm = build_shift_permutation(&meta.shifts);
        let inv = invert_permutation(&perm);
        result = apply_byte_permutation(&result, &inv);
    }

    // Undo reverse (applied first during compression → undo second)
    if meta.mode == DeentropyMode::Reverse || meta.mode == DeentropyMode::ReverseShift {
        result.reverse();
    }

    result
}

fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let len = data.len() as f64;
    counts.iter().filter(|&&c| c > 0).map(|&c| {
        let p = c as f64 / len;
        -p * p.log2()
    }).sum()
}

const REVERSE_SAMPLE_SIZE: usize = 128 * 1024; // 128 KB sample for reversal estimation

/// Estimate whether reversing a chunk's byte order would improve compression.
/// Compresses a small sample in both directions using LZ77 and compares sizes.
fn should_reverse_chunk(data: &[u8]) -> bool {
    if data.len() < 1024 {
        return false;
    }
    let sample_len = data.len().min(REVERSE_SAMPLE_SIZE);
    let sample = &data[..sample_len];
    let sample_rev: Vec<u8> = sample.iter().copied().rev().collect();
    let fwd_size = lz77_compress(sample).len();
    let rev_size = lz77_compress(&sample_rev).len();
    rev_size < fwd_size
}

/// Early-exit threshold: skip deentropy + EGAP when entropy exceeds this
const EARLY_EXIT_ENTROPY: f64 = 7.9;
/// Early-exit: skip deentropy + EGAP when all 256 byte values are present
const EARLY_EXIT_DISTINCT: usize = 256;

fn compress_adaptive_chunk_data(
    dict_data: &[u8],
    orig_chunk_size: usize,
    _chunk_idx: usize,
) -> (Vec<u8>, u8, f64, f64) {
    let dict_size = dict_data.len();
    let dict_reduction_pct = if orig_chunk_size > 0 {
        (1.0 - dict_size as f64 / orig_chunk_size as f64) * 100.0
    } else {
        0.0
    };

    let entropy = shannon_entropy(dict_data);

    // --- Early exit: near-random data ---
    let mut distinct = [false; 256];
    for &b in dict_data { distinct[b as usize] = true; }
    let distinct_count = distinct.iter().filter(|&&v| v).count();

    if entropy > EARLY_EXIT_ENTROPY || distinct_count >= EARLY_EXIT_DISTINCT {
        return (dict_data.to_vec(), 0, dict_reduction_pct, entropy);
    }

    // EGAP decides independently — early exit already filtered out
    // near-random data (entropy > 7.9 or 256 distinct symbols).
    // Everything that survived early exit is worth trying EGAP on.
    let run_egap = true;

    // --- De-entropy transform (skip for high-entropy chunks where it never helps) ---
    const DEENTROPY_SKIP_ENTROPY: f64 = 6.5;
    let (de_data, de_meta, use_deentropy) = if entropy > DEENTROPY_SKIP_ENTROPY {
        (Vec::new(), DeentropyMeta::none(), false)
    } else {
        let (d, m) = deentropy_transform(dict_data);
        let used = m.mode != DeentropyMode::None;
        (d, m, used)
    };
    let egap_input = if use_deentropy { &de_data[..] } else { dict_data };

    let mut current_data: Vec<u8>;
    let mut flags: u8 = 0;

    if use_deentropy {
        flags |= CHUNK_FLAG_DEENTROPY;
    }

    // --- EGAP ---
    if run_egap {
        let egap_data = entropy_encode(egap_input);
        if egap_data.len() < dict_size {
            flags |= CHUNK_FLAG_EGAP;
            current_data = egap_data;
        } else {
            current_data = egap_input.to_vec();
        }
    } else {
        current_data = egap_input.to_vec();
    }

    // Prepend de-entropy meta to the compressed payload
    if use_deentropy {
        let meta_bytes = de_meta.serialize();
        let mut combined = Vec::with_capacity(meta_bytes.len() + current_data.len());
        combined.extend_from_slice(&meta_bytes);
        combined.extend_from_slice(&current_data);
        current_data = combined;
    }

    (current_data, flags, dict_reduction_pct, entropy)
}

fn adaptive_stage_label(flags: u8) -> String {
    let mut parts: Vec<&str> = vec!["dict"];
    if flags & CHUNK_FLAG_DEENTROPY != 0 { parts.push("deent"); }
    if flags & CHUNK_FLAG_EGAP != 0 { parts.push("egap"); }
    if flags & CHUNK_FLAG_LZ77 != 0 { parts.push("lz77"); }
    if flags & CHUNK_FLAG_POST_ENTROPY != 0 { parts.push("post-ent"); }
    let label = parts.join("+");
    if flags & CHUNK_FLAG_REVERSED != 0 {
        format!("{} (rev)", label)
    } else if parts.len() == 1 {
        "dict only".to_string()
    } else {
        label
    }
}

fn print_adaptive_summary(chunk_flags: &[u8]) {
    let num_chunks = chunk_flags.len();
    let mut egap_count = 0usize;
    let mut lz77_count = 0usize;
    let mut post_ent_count = 0usize;
    let mut reversed_count = 0usize;
    let mut deentropy_count = 0usize;
    let mut dict_only_count = 0usize;
    for &f in chunk_flags {
        if f & CHUNK_FLAG_EGAP != 0 {
            egap_count += 1;
        }
        if f & CHUNK_FLAG_LZ77 != 0 {
            lz77_count += 1;
        }
        if f & CHUNK_FLAG_POST_ENTROPY != 0 {
            post_ent_count += 1;
        }
        if f & CHUNK_FLAG_REVERSED != 0 {
            reversed_count += 1;
        }
        if f & CHUNK_FLAG_DEENTROPY != 0 {
            deentropy_count += 1;
        }
        if f & (CHUNK_FLAG_EGAP | CHUNK_FLAG_LZ77 | CHUNK_FLAG_DEENTROPY | CHUNK_FLAG_POST_ENTROPY) == 0 {
            dict_only_count += 1;
        }
    }
    println!("[adaptive] stages used across {} chunks:", num_chunks);
    println!("    dict only:      {} chunks", dict_only_count);
    println!("    egap:           {} chunks", egap_count);
    println!("    lz77:           {} chunks", lz77_count);
    println!("    post-entropy:   {} chunks", post_ent_count);
    println!("    deentropy:      {} chunks", deentropy_count);
    println!("    reversed:       {} chunks", reversed_count);
}

// ---------------------------------------------------------------------------
// LZ77 sliding window compression
// ---------------------------------------------------------------------------

const LZ77_DEFAULT_WINDOW_EXP: u8 = 16;       // 1 << 16 = 64 KB window
const LZ77_MIN_MATCH: usize = 3;
const LZ77_MAX_MATCH: usize = 258;
const LZ77_HASH_SIZE: usize = 1 << 16;        // 65536 entries
const LZ77_MAX_DISTANCE: usize = 65535;       // 16-bit distance field
const LZ77_MAX_CHAIN: usize = 128;            // max hash chain depth
const LZ77_GLOBAL_CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4 MB global LZ77 chunks
const LZ77_EARLY_SKIP_ENTROPY: f64 = 7.95; // skip LZ77 only when chunk is near-random
const LZ77_EARLY_SKIP_DISTINCT: usize = 256;
const LZ77_REPETITION_SAMPLE: usize = 256 * 1024; // sample for repeat detection
const LZ77_REPETITION_THRESHOLD: usize = 512;    // minimum repeated 4-byte grams needed

fn lz77_write_raw_block(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(10 + data.len());
    out.extend_from_slice(b"LZ77");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.push(0); // raw block marker
    out.push(0); // no padding
    out.extend_from_slice(data);
    out
}

fn lz77_compress_maybe(data: &[u8]) -> Vec<u8> {
    let compressed = lz77_compress(data);
    if compressed.len() >= data.len() {
        return lz77_write_raw_block(data);
    }
    compressed
}

fn lz77_chunk_repetition_score(data: &[u8]) -> usize {
    let sample = &data[..data.len().min(LZ77_REPETITION_SAMPLE)];
    if sample.len() < 8 {
        return 0;
    }

    let mut counts: HashMap<u32, usize> = HashMap::with_capacity(1 << 14);
    for pos in 0..sample.len().saturating_sub(3) {
        let key = u32::from_le_bytes(sample[pos..pos + 4].try_into().unwrap());
        *counts.entry(key).or_insert(0) += 1;
    }

    counts.values().filter(|&&c| c > 1).map(|&c| c - 1).sum()
}

fn lz77_should_skip_chunk(data: &[u8]) -> bool {
    let entropy = shannon_entropy(data);

    // Most high-entropy chunks are not worth an LZ77 pass.
    if entropy > LZ77_EARLY_SKIP_ENTROPY {
        let mut distinct = [false; 256];
        for &b in data {
            distinct[b as usize] = true;
        }
        let distinct_count = distinct.iter().filter(|&&v| v).count();
        if distinct_count >= LZ77_EARLY_SKIP_DISTINCT {
            return true;
        }
    }

    // Use a cheap repeat-count heuristic to avoid skipping chunks that still
    // have enough local repetition for LZ77 to gain 10% or more.
    let repeat_score = lz77_chunk_repetition_score(data);
    repeat_score < LZ77_REPETITION_THRESHOLD
}

fn lz77_compress_chunked(data: &[u8]) -> Vec<u8> {
    if data.len() <= LZ77_GLOBAL_CHUNK_SIZE {
        return lz77_compress_maybe(data);
    }

    let num_chunks = (data.len() + LZ77_GLOBAL_CHUNK_SIZE - 1) / LZ77_GLOBAL_CHUNK_SIZE;

    // Process all chunks in parallel
    let lz77_progress = AtomicUsize::new(0);
    let chunk_results: Vec<Vec<u8>> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * LZ77_GLOBAL_CHUNK_SIZE;
            let end = ((i + 1) * LZ77_GLOBAL_CHUNK_SIZE).min(data.len());
            let chunk = &data[start..end];

            let result = if lz77_should_skip_chunk(chunk) {
                lz77_write_raw_block(chunk)
            } else {
                lz77_compress_maybe(chunk)
            };
            report_progress(&lz77_progress, num_chunks, "lz77");
            result
        })
        .collect();

    let total_size: usize = chunk_results.iter().map(|r| r.len()).sum();
    let mut out = Vec::with_capacity(total_size);
    for result in chunk_results {
        out.extend_from_slice(&result);
    }

    out
}

struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    bits_in_acc: u32,
}

impl BitWriter {
    fn with_capacity(cap: usize) -> Self {
        BitWriter { bytes: Vec::with_capacity(cap), acc: 0, bits_in_acc: 0 }
    }

    #[inline(always)]
    fn write_bits(&mut self, value: u32, num_bits: u8) {
        self.acc = (self.acc << num_bits as u32) | (value as u64);
        self.bits_in_acc += num_bits as u32;
        while self.bits_in_acc >= 8 {
            self.bits_in_acc -= 8;
            self.bytes.push((self.acc >> self.bits_in_acc) as u8);
            self.acc &= (1u64 << self.bits_in_acc) - 1;
        }
    }

    #[inline(always)]
    fn write_bit(&mut self, bit: u8) {
        self.write_bits(bit as u32 & 1, 1);
    }

    fn finish(mut self) -> (Vec<u8>, u8) {
        let padding = if self.bits_in_acc > 0 {
            let pad = 8 - self.bits_in_acc as u8;
            self.bytes.push((self.acc << pad as u32) as u8);
            pad
        } else {
            0
        };
        (self.bytes, padding)
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, byte_pos: 0, bit_pos: 0 }
    }

    #[inline(always)]
    fn read_bit(&mut self) -> io::Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "LZ77 bitstream truncated"));
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
        Ok(bit)
    }

    #[inline(always)]
    fn read_bits(&mut self, num_bits: u8) -> io::Result<u32> {
        let mut value = 0u32;
        for _ in 0..num_bits {
            value = (value << 1) | self.read_bit()? as u32;
        }
        Ok(value)
    }
}

#[inline(always)]
fn lz77_hash(a: u8, b: u8, c: u8) -> usize {
    ((a as usize) << 10 ^ (b as usize) << 5 ^ c as usize) & (LZ77_HASH_SIZE - 1)
}

/// Find the best LZ77 match at `pos` by walking the hash chain.
/// Returns (match_length, match_distance). match_length == 0 means no match found.
#[inline]
fn lz77_find_match(data: &[u8], pos: usize, head: &[u32], prev: &[u32], window_size: usize) -> (usize, usize) {
    let original_size = data.len();
    if pos + 2 >= original_size {
        return (0, 0);
    }
    let h = lz77_hash(data[pos], data[pos + 1], data[pos + 2]);
    let mut chain_pos = head[h];
    let mut chain_count = 0u32;
    let mut best_len = 0usize;
    let mut best_dist = 0usize;

    while chain_pos != u32::MAX && chain_count < LZ77_MAX_CHAIN as u32 {
        let cp = chain_pos as usize;
        if cp >= pos { break; }
        let dist = pos - cp;
        if dist > LZ77_MAX_DISTANCE { break; }

        let max_len = LZ77_MAX_MATCH.min(original_size - pos);
        let mut len = 0;
        let safe_limit = if max_len >= 8 { max_len - 7 } else { 0 };
        while len < safe_limit {
            let a = u64::from_ne_bytes(data[cp + len..cp + len + 8].try_into().unwrap());
            let b = u64::from_ne_bytes(data[pos + len..pos + len + 8].try_into().unwrap());
            let xor = a ^ b;
            if xor != 0 {
                len += (xor.to_be().leading_zeros() / 8) as usize;
                break;
            }
            len += 8;
        }
        while len < max_len && data[cp + len] == data[pos + len] {
            len += 1;
        }

        if len > best_len && len >= LZ77_MIN_MATCH {
            best_len = len;
            best_dist = dist;
            if best_len >= LZ77_MAX_MATCH { break; }
        }

        let next = prev[cp % window_size];
        if next != u32::MAX && next as usize >= cp { break; }
        chain_pos = next;
        chain_count += 1;
    }

    (best_len, best_dist)
}

fn lz77_compress(data: &[u8]) -> Vec<u8> {
    let original_size = data.len();
    if original_size == 0 {
        let mut out = Vec::with_capacity(10);
        out.extend_from_slice(b"LZ77");
        out.extend_from_slice(&0u32.to_le_bytes());
        out.push(LZ77_DEFAULT_WINDOW_EXP | 0x80);
        out.push(0);
        return out;
    }

    let window_size: usize = 1 << LZ77_DEFAULT_WINDOW_EXP;
    let mut head = vec![u32::MAX; LZ77_HASH_SIZE];
    let mut prev = vec![u32::MAX; window_size];

    // Worst case: all literals = original_size * 9/8
    let mut writer = BitWriter::with_capacity(original_size * 9 / 8 + 64);
    let mut pos = 0usize;

    while pos < original_size {
        let (best_len, best_dist) = lz77_find_match(data, pos, &head, &prev, window_size);

        if best_len >= LZ77_MIN_MATCH {
            // --- Lazy matching: check if pos+1 has a strictly better match ---
            if best_len < LZ77_MAX_MATCH {
                let (lazy_len, lazy_dist) = lz77_find_match(data, pos + 1, &head, &prev, window_size);
                if lazy_len > best_len + 1 {
                    // Emit pos as literal, then use the better match at pos+1
                    writer.write_bit(0);
                    writer.write_bits(data[pos] as u32, 8);
                    if pos + 2 < original_size {
                        let h = lz77_hash(data[pos], data[pos + 1], data[pos + 2]);
                        prev[pos % window_size] = head[h];
                        head[h] = pos as u32;
                    }
                    pos += 1;

                    // Emit the lazy match
                    writer.write_bit(1);
                    if lazy_dist <= 256 {
                        writer.write_bit(0);
                        writer.write_bits((lazy_dist - 1) as u32, 8);
                    } else {
                        writer.write_bit(1);
                        writer.write_bits((lazy_dist - 1) as u32, 16);
                    }
                    if lazy_len == LZ77_MIN_MATCH {
                        writer.write_bit(0);
                    } else {
                        writer.write_bit(1);
                        writer.write_bits((lazy_len - LZ77_MIN_MATCH - 1) as u32, 8);
                    }
                    for i in 0..lazy_len {
                        let p = pos + i;
                        if p + 2 < original_size {
                            let h = lz77_hash(data[p], data[p + 1], data[p + 2]);
                            prev[p % window_size] = head[h];
                            head[h] = p as u32;
                        }
                    }
                    pos += lazy_len;
                    continue;
                }
            }

            // Back-reference: variable-length distance + variable-length length
            writer.write_bit(1);
            if best_dist <= 256 {
                writer.write_bit(0);
                writer.write_bits((best_dist - 1) as u32, 8);
            } else {
                writer.write_bit(1);
                writer.write_bits((best_dist - 1) as u32, 16);
            }
            if best_len == LZ77_MIN_MATCH {
                writer.write_bit(0);
            } else {
                writer.write_bit(1);
                writer.write_bits((best_len - LZ77_MIN_MATCH - 1) as u32, 8);
            }

            // Update hash for all positions in match
            for i in 0..best_len {
                let p = pos + i;
                if p + 2 < original_size {
                    let h = lz77_hash(data[p], data[p + 1], data[p + 2]);
                    prev[p % window_size] = head[h];
                    head[h] = p as u32;
                }
            }
            pos += best_len;
        } else {
            // Literal: [0] [8-bit byte]
            writer.write_bit(0);
            writer.write_bits(data[pos] as u32, 8);

            if pos + 2 < original_size {
                let h = lz77_hash(data[pos], data[pos + 1], data[pos + 2]);
                prev[pos % window_size] = head[h];
                head[h] = pos as u32;
            }
            pos += 1;
        }
    }

    let (payload, padding) = writer.finish();

    // Header: "LZ77" + original_size(u32) + window_exp(u8) + padding(u8) + payload
    let mut out = Vec::with_capacity(10 + payload.len());
    out.extend_from_slice(b"LZ77");
    out.extend_from_slice(&(original_size as u32).to_le_bytes());
    out.push(LZ77_DEFAULT_WINDOW_EXP | 0x80);
    out.push(padding);
    out.extend_from_slice(&payload);
    out
}

fn lz77_decompress_block(data: &[u8]) -> io::Result<(Vec<u8>, usize)> {
    if data.len() < 10 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "LZ77 data too short"));
    }
    if &data[..4] != b"LZ77" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "LZ77 magic mismatch"));
    }
    let original_size = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let window_exp = data[8];
    let payload = &data[10..];

    if original_size == 0 {
        return Ok((Vec::new(), 10));
    }

    if window_exp == 0 {
        if payload.len() < original_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LZ77 raw block truncated",
            ));
        }
        let output = payload[..original_size].to_vec();
        return Ok((output, 10 + original_size));
    }

    let variable_len = window_exp & 0x80 != 0;
    let window_exp = window_exp & 0x7F;

    let mut reader = BitReader::new(payload);
    let mut output: Vec<u8> = Vec::with_capacity(original_size);

    while output.len() < original_size {
        let bit = reader.read_bit()?;
        if bit == 0 {
            let byte = reader.read_bits(8)? as u8;
            output.push(byte);
        } else {
            let (distance, length) = if variable_len {
                let dist_flag = reader.read_bit()?;
                let d = if dist_flag == 0 {
                    reader.read_bits(8)? as usize + 1
                } else {
                    reader.read_bits(16)? as usize + 1
                };
                let len_flag = reader.read_bit()?;
                let l = if len_flag == 0 {
                    LZ77_MIN_MATCH
                } else {
                    reader.read_bits(8)? as usize + LZ77_MIN_MATCH + 1
                };
                (d, l)
            } else {
                let d = reader.read_bits(window_exp)? as usize;
                let l = reader.read_bits(8)? as usize + LZ77_MIN_MATCH;
                (d, l)
            };

            if distance == 0 || distance > output.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("LZ77 invalid back-reference: dist={}, pos={}", distance, output.len()),
                ));
            }

            let start = output.len() - distance;
            for i in 0..length {
                let b = output[start + i];
                output.push(b);
            }
        }
    }

    if output.len() != original_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("LZ77 size mismatch: {} != {}", output.len(), original_size),
        ));
    }

    let consumed_payload = reader.byte_pos + if reader.bit_pos > 0 { 1 } else { 0 };
    Ok((output, 10 + consumed_payload))
}

fn lz77_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    let (output, consumed) = lz77_decompress_block(data)?;
    if consumed != data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "LZ77 block contains extra trailing data",
        ));
    }
    Ok(output)
}

fn lz77_decompress_stream(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut output: Vec<u8> = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        let (chunk, consumed) = lz77_decompress_block(&data[offset..])?;
        output.extend_from_slice(&chunk);
        offset += consumed;
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Threshold
// ---------------------------------------------------------------------------

fn occurrence_threshold_by_data_length(data_length: usize) -> usize {
    let t = data_length / 50000;
    if t < 2 { 2 } else { t }
}

// ---------------------------------------------------------------------------
// Escape byte selection
// ---------------------------------------------------------------------------

fn select_escape_byte(data: &[u8]) -> u8 {
    const SCAN_CHUNK: usize = 4 * 1024 * 1024;

    // Parallel frequency count across 4 MB segments
    let counts = data.par_chunks(SCAN_CHUNK)
        .fold(|| [0u64; 256], |mut acc, chunk| {
            for &b in chunk {
                acc[b as usize] += 1;
            }
            acc
        })
        .reduce(|| [0u64; 256], |mut a, b| {
            for i in 0..256 { a[i] += b[i]; }
            a
        });

    // Prefer absent byte
    for b in 0u8..=255 {
        if counts[b as usize] == 0 {
            return b;
        }
    }
    // Otherwise least frequent
    let (least, _) = counts.iter().enumerate().min_by_key(|(_, &c)| c).unwrap();
    least as u8
}

// ---------------------------------------------------------------------------
// Sampling — parallel multi-region with cross-referencing
// ---------------------------------------------------------------------------

/// Compute evenly-spaced sample region offsets across the data.
fn compute_sample_offsets(data_len: usize, num_regions: usize, region_size: usize) -> Vec<(usize, usize)> {
    if data_len <= region_size * num_regions {
        return vec![(0, data_len)];
    }
    let step = data_len / num_regions;
    (0..num_regions)
        .map(|i| {
            let start = i * step;
            let end = (start + region_size).min(data_len);
            (start, end)
        })
        .collect()
}

/// Take evenly spaced 0.5 MB chunks across the file, collected in parallel via Rayon.
fn sample_spread_parallel(data: &[u8], target: usize, num_regions: usize) -> Vec<u8> {
    if data.len() <= target {
        return data.to_vec();
    }
    let region_size = target / num_regions;
    let offsets = compute_sample_offsets(data.len(), num_regions, region_size);

    // Each region is read independently — perfect for parallel mmap access
    let regions: Vec<&[u8]> = offsets
        .par_iter()
        .map(|&(start, end)| &data[start..end])
        .collect();

    let total: usize = regions.iter().map(|r| r.len()).sum();
    let mut out = Vec::with_capacity(total);
    for r in regions {
        out.extend_from_slice(r);
    }
    out
}

/// Take samples biased toward areas with highest local repetition.
/// Scans the file in blocks, scores each by byte frequency variance,
/// and picks the most "repetitive" blocks.
fn sample_hotspots(data: &[u8], target: usize, block_size: usize) -> Vec<u8> {
    if data.len() <= target {
        return data.to_vec();
    }

    let num_blocks = data.len() / block_size;
    if num_blocks == 0 {
        return data[..target.min(data.len())].to_vec();
    }

    // Score each block by how repetitive it is:
    // high score = many repeated bytes = more compressible
    let mut block_scores: Vec<(usize, f64)> = (0..num_blocks)
        .into_par_iter()
        .map(|i| {
            let start = i * block_size;
            let end = (start + block_size).min(data.len());
            let block = &data[start..end];

            let mut counts = [0u32; 256];
            for &b in block {
                counts[b as usize] += 1;
            }
            let score: f64 = counts.iter().map(|&c| (c as f64).powi(2)).sum();
            (i, score)
        })
        .collect();

    block_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let num_to_take = (target / block_size).max(1);
    let mut out = Vec::with_capacity(target);
    for (block_idx, _) in block_scores.into_iter().take(num_to_take) {
        let start = block_idx * block_size;
        let end = (start + block_size).min(data.len());
        out.extend_from_slice(&data[start..end]);
        if out.len() >= target {
            break;
        }
    }
    out
}

/// Combine spread + hotspot samples into one analysis corpus.
fn build_analysis_corpus(data: &[u8], target_per_strategy: usize) -> Vec<u8> {
    let spread = sample_spread_parallel(data, target_per_strategy, 16);
    let hotspots = sample_hotspots(data, target_per_strategy, 65536);

    let mut corpus = Vec::with_capacity(spread.len() + hotspots.len());
    corpus.extend_from_slice(&spread);
    corpus.extend_from_slice(&hotspots);
    corpus
}

// ---------------------------------------------------------------------------
// Cross-reference sampling — parallel 256KB regions, frequency merging
// ---------------------------------------------------------------------------

/// Per-region frequency map for a single chunk size.
type RegionFreqMap = HashMap<Vec<u8>, usize>;

/// Sample multiple regions in parallel, build per-region frequency
/// dictionaries, then cross-reference: only patterns that appear above
/// threshold in >= MIN_CROSS_REF_SAMPLES distinct regions are promoted.
///
/// Uses a flat (region, size) work-item list for true 2D parallelism, and
/// applies early termination during the merge phase — once a pattern has
/// been seen in enough regions, it is promoted immediately without waiting
/// for all regions to be processed.
fn parallel_cross_reference_sample(
    data: &[u8],
    num_regions: usize,
    sizes: &[usize],
    threshold: usize,
) -> Vec<Candidate> {
    let offsets = compute_sample_offsets(data.len(), num_regions, SAMPLE_REGION_SIZE);
    let actual_regions = offsets.len();

    if actual_regions == 0 || sizes.is_empty() {
        return vec![];
    }

    // Build flat work items: (region_idx, region_start, region_end, chunk_size)
    let work_items: Vec<(usize, usize, usize, usize)> = offsets
        .iter()
        .enumerate()
        .flat_map(|(ri, &(start, end))| {
            sizes.iter().map(move |&cs| (ri, start, end, cs))
        })
        .collect();

    // Process all (region, size) pairs in parallel — flat 2D parallelism
    let region_freqs: Vec<(usize, usize, RegionFreqMap)> = work_items
        .par_iter()
        .map(|&(region_idx, start, end, chunk_size)| {
            let region_data = &data[start..end];
            let region_len = region_data.len();
            let n_positions = region_len.saturating_sub(chunk_size - 1);

            // Use a byte-hash approach: count frequencies, only allocate Vec<u8>
            // keys for patterns that exceed the threshold.
            let mut freq_raw: HashMap<Vec<u8>, usize> = HashMap::with_capacity(
                (n_positions / 8).min(16384)
            );
            for pos in 0..n_positions {
                let chunk = &region_data[pos..pos + chunk_size];
                *freq_raw.entry(chunk.to_vec()).or_insert(0) += 1;
            }

            // Pre-filter: only keep patterns above threshold in this region
            freq_raw.retain(|_, count| *count >= threshold);

            (region_idx, chunk_size, freq_raw)
        })
        .collect();

    // Group by chunk_size, then merge with early termination.
    let mut by_size: HashMap<usize, Vec<(usize, RegionFreqMap)>> = HashMap::new();
    for (region_idx, chunk_size, freq) in region_freqs {
        by_size.entry(chunk_size).or_default().push((region_idx, freq));
    }

    let min_regions = MIN_CROSS_REF_SAMPLES.min(actual_regions);
    let sample_bytes = actual_regions * SAMPLE_REGION_SIZE;
    let sample_fraction = sample_bytes as f64 / data.len().max(1) as f64;

    let mut candidates: Vec<Candidate> = Vec::new();

    for (&chunk_size, region_maps) in &by_size {
        // Merge: for each pattern, track (total_count, num_regions_seen).
        // Early termination: once a pattern hits min_regions, promote it
        // immediately and skip further counting.
        let mut merged: HashMap<Vec<u8>, (usize, usize)> = HashMap::new();
        let mut promoted: HashSet<Vec<u8>> = HashSet::new();

        for (_region_idx, freq) in region_maps {
            for (pattern, &count) in freq {
                if promoted.contains(pattern) {
                    // Already promoted — just accumulate count for scaling
                    if let Some(entry) = merged.get_mut(pattern) {
                        entry.0 += count;
                    }
                    continue;
                }
                let entry = merged.entry(pattern.clone()).or_insert((0, 0));
                entry.0 += count;
                entry.1 += 1;

                // Early promotion: pattern seen in enough regions
                if entry.1 >= min_regions {
                    promoted.insert(pattern.clone());
                }
            }
        }

        // Emit promoted patterns as candidates.
        for pattern in &promoted {
            if let Some(&(total_count, _)) = merged.get(pattern) {
                let estimated_full_count = if sample_fraction > 0.0 && sample_fraction < 1.0 {
                    (total_count as f64 / sample_fraction) as usize
                } else {
                    total_count
                };
                let size_score = chunk_size * estimated_full_count;
                candidates.push(Candidate {
                    chunk: pattern.clone(),
                    max_count: estimated_full_count,
                    chunk_size,
                    size_score,
                });
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.size_score
            .cmp(&a.size_score)
            .then(b.max_count.cmp(&a.max_count))
            .then(a.chunk.cmp(&b.chunk))
    });

    candidates
}

// ---------------------------------------------------------------------------
// Probe — score all chunk sizes cheaply on a small sample
// ---------------------------------------------------------------------------

struct ProbeResult {
    sizes: Vec<usize>,   // ordered best-first
    scores: Vec<i64>,    // parallel to sizes
}

fn probe_chunk_sizes(data: &[u8]) -> ProbeResult {
    let threshold = occurrence_threshold_by_data_length(data.len()).max(2);
    let data_length = data.len();

    let mut scores: Vec<(usize, i64)> = (MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE)
        .into_par_iter()
        .map(|chunk_size| {
            let capacity = (data_length / 4).min(65536);
            let mut freq: HashMap<&[u8], usize> = HashMap::with_capacity(capacity);
            for pos in 0..data_length.saturating_sub(chunk_size - 1) {
                let chunk = &data[pos..pos + chunk_size];
                *freq.entry(chunk).or_insert(0) += 1;
            }
            let score: i64 = freq
                .values()
                .filter(|&&c| c >= threshold)
                .map(|&c| (chunk_size as i64 - 2) * c as i64 - (1 + 2 + chunk_size) as i64)
                .filter(|&s| s > 0)
                .sum();
            (chunk_size, score)
        })
        .collect();

    scores.sort_by(|a, b| b.1.cmp(&a.1));

    let sizes = scores.iter().map(|(s, _)| *s).collect();
    let score_vals = scores.iter().map(|(_, s)| *s).collect();

    ProbeResult { sizes, scores: score_vals }
}

// ---------------------------------------------------------------------------
// Full analysis on selected sizes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Candidate {
    chunk: Vec<u8>,
    max_count: usize,
    chunk_size: usize,
    size_score: usize,
}

fn find_occurrences_for_sizes(
    data: &[u8],
    sizes: &[usize],
    threshold: Option<usize>,
) -> Vec<Candidate> {
    let threshold = threshold.unwrap_or_else(|| occurrence_threshold_by_data_length(data.len()));
    let data_length = data.len();

    if data_length == 0 || sizes.is_empty() {
        return vec![];
    }

    let freq_maps: Vec<(usize, HashMap<&[u8], usize>)> = sizes
        .into_par_iter()
        .map(|&chunk_size| {
            let capacity = (data_length / 4).min(65536);
            let mut freq: HashMap<&[u8], usize> = HashMap::with_capacity(capacity);
            for pos in 0..data_length.saturating_sub(chunk_size - 1) {
                let chunk = &data[pos..pos + chunk_size];
                *freq.entry(chunk).or_insert(0) += 1;
            }
            (chunk_size, freq)
        })
        .collect();

    let mut best_by_chunk: HashMap<Vec<u8>, (usize, usize)> = HashMap::new();

    for (chunk_size, freq) in &freq_maps {
        let mut above: Vec<(&&[u8], &usize)> = freq
            .iter()
            .filter(|(_, &count)| count >= threshold)
            .collect();
        above.sort_by(|a, b| b.1.cmp(a.1));

        for (chunk_slice, &count) in above.into_iter().take(20) {
            let chunk_vec = chunk_slice.to_vec();
            let size_score = chunk_size * count;
            let entry = best_by_chunk.entry(chunk_vec).or_insert((0, 0));
            if size_score > entry.0 * entry.1 {
                *entry = (count, *chunk_size);
            }
        }
    }

    let mut candidates: Vec<Candidate> = best_by_chunk
        .into_iter()
        .map(|(chunk, (count, chunk_size))| Candidate {
            size_score: chunk_size * count,
            chunk,
            max_count: count,
            chunk_size,
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.size_score
            .cmp(&a.size_score)
            .then(b.max_count.cmp(&a.max_count))
            .then(a.chunk.cmp(&b.chunk))
    });

    candidates
}

// ---------------------------------------------------------------------------
// Chunk selection
// ---------------------------------------------------------------------------

fn select_best_chunks(candidates: &[Candidate], max_codes: usize) -> Vec<Vec<u8>> {
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut filtered: Vec<(Vec<u8>, usize, usize, i64)> = Vec::new();

    for c in candidates {
        if seen.contains(&c.chunk) {
            continue;
        }
        seen.insert(c.chunk.clone());

        let chunk_size = c.chunk_size;
        if chunk_size <= 2 {
            continue;
        }

        let header_cost = (1 + 2 + chunk_size) as i64;
        let estimated_savings = ((chunk_size as i64 - 2) * c.max_count as i64) - header_cost;

        if estimated_savings > 0 {
            filtered.push((c.chunk.clone(), chunk_size, c.max_count, estimated_savings));
        }
    }

    filtered.sort_by(|a, b| {
        b.3.cmp(&a.3)
            .then(b.1.cmp(&a.1))
            .then(a.0.cmp(&b.0))
    });

    filtered
        .into_iter()
        .take(max_codes)
        .map(|(chunk, _, _, _)| chunk)
        .collect()
}

// ---------------------------------------------------------------------------
// Encoding — Aho-Corasick automaton, parallel chunks, memchr literal emit
// ---------------------------------------------------------------------------

/// Emit literal bytes into `out`, using memchr to bulk-copy runs of
/// non-escape bytes and only stopping to escape the escape byte.
#[inline]
fn emit_literals(out: &mut Vec<u8>, data: &[u8], escape_byte: u8) {
    let mut pos = 0;
    while pos < data.len() {
        match memchr::memchr(escape_byte, &data[pos..]) {
            Some(offset) => {
                out.extend_from_slice(&data[pos..pos + offset]);
                out.push(escape_byte);
                out.push(0);
                pos += offset + 1;
            }
            None => {
                out.extend_from_slice(&data[pos..]);
                break;
            }
        }
    }
}

/// Encode a single data slice using a pre-built Aho-Corasick automaton.
/// Single-pass O(n) scan — the automaton handles all multi-pattern matching
/// with leftmost-longest semantics (i.e. greedy longest match at each pos).
fn encode_chunk_ac(
    data: &[u8],
    escape_byte: u8,
    ac: &AhoCorasick,
    pattern_to_code: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut last_end = 0;

    for mat in ac.find_iter(data) {
        // Bulk-emit literal bytes between the previous match and this one
        emit_literals(&mut out, &data[last_end..mat.start()], escape_byte);
        // Emit the encoded token (escape + code)
        out.push(escape_byte);
        out.push(pattern_to_code[mat.pattern().as_usize()]);
        last_end = mat.end();
    }

    // Emit trailing literals after the last match
    emit_literals(&mut out, &data[last_end..], escape_byte);
    out
}

fn encode_with_map(source_data: &[u8], escape_byte: u8, code_map: &HashMap<u8, Vec<u8>>) -> Vec<u8> {
    if code_map.is_empty() {
        let mut out = Vec::with_capacity(source_data.len());
        emit_literals(&mut out, source_data, escape_byte);
        return out;
    }

    // Collect (code, pattern) pairs from the dictionary
    let entries: Vec<(u8, &[u8])> = code_map
        .iter()
        .map(|(&code, chunk)| (code, chunk.as_slice()))
        .collect();

    // Build Aho-Corasick automaton
    let ac = AhoCorasick::builder()
        .match_kind(MatchKind::LeftmostLongest)
        .build(entries.iter().map(|(_, pat)| *pat))
        .expect("AC automaton build failed");

    let pattern_to_code: Vec<u8> = entries.iter().map(|(code, _)| *code).collect();

    // For small data, encode sequentially
    if source_data.len() <= ENCODE_CHUNK_SIZE * 2 {
        return encode_chunk_ac(source_data, escape_byte, &ac, &pattern_to_code);
    }

    // Parallel encoding
    let num_chunks = (source_data.len() + ENCODE_CHUNK_SIZE - 1) / ENCODE_CHUNK_SIZE;

    let chunk_results: Vec<Vec<u8>> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * ENCODE_CHUNK_SIZE;
            let end = (start + ENCODE_CHUNK_SIZE).min(source_data.len());
            encode_chunk_ac(&source_data[start..end], escape_byte, &ac, &pattern_to_code)
        })
        .collect();

    // Parallel scatter-write: pre-allocate exact output, copy chunks in parallel
    let sizes: Vec<usize> = chunk_results.iter().map(|c| c.len()).collect();
    let total_len: usize = sizes.iter().sum();

    let mut out = vec![0u8; total_len];

    // Build non-overlapping mutable slices via split_at_mut (fully safe)
    {
        let mut remaining = out.as_mut_slice();
        let mut dest_slices: Vec<&mut [u8]> = Vec::with_capacity(chunk_results.len());
        for &sz in &sizes {
            let (left, right) = remaining.split_at_mut(sz);
            dest_slices.push(left);
            remaining = right;
        }
        dest_slices.par_iter_mut()
            .zip(chunk_results.par_iter())
            .for_each(|(dest, src)| {
                dest.copy_from_slice(src);
            });
    }

    out
}

// ---------------------------------------------------------------------------
// EGAP — Empirical Gap entropy coding (prefix-free, swap-based assignment)
// ---------------------------------------------------------------------------

/// Prefix-free code table entry: symbol → (code_length, code_bits).
/// code_bits is right-aligned: only the lowest `code_len` bits are meaningful.
struct EgapCode {
    symbol: u8,
    code_len: u8,  // 1..=32
    code_bits: u32,
}

/// Check whether code A (a_bits, a_len) is a prefix of code B (b_bits, b_len).
/// Codes are MSB-first, right-aligned in u32.
#[inline]
fn egap_is_prefix(a_bits: u32, a_len: u8, b_bits: u32, b_len: u8) -> bool {
    if a_len == 0 {
        return true;
    }
    if a_len > b_len {
        return false;
    }
    (b_bits >> (b_len - a_len)) == a_bits
}

/// Generate `n` prefix-free codes from a complete binary trie of depth
/// ceil(log2(n)), then iteratively shorten codes where a sibling is free.
fn generate_prefix_codes(n: usize) -> Vec<(u32, u8)> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![(0, 1)];
    }

    let depth = (n as f64).log2().ceil() as u8;
    // Start: first n codes of exactly `depth` bits (prefix-free by equal length)
    let mut codes: Vec<(u32, u8)> = (0..n as u32).map(|b| (b, depth)).collect();

    // Iterative shortening: try to promote each code to its parent when
    // the sibling leaf is not assigned to any other code.
    loop {
        let mut changed = false;
        for i in 0..codes.len() {
            let (bits, len) = codes[i];
            if len <= 1 {
                continue;
            }
            let parent_bits = bits >> 1;
            let parent_len = len - 1;
            // Sibling = same parent, opposite last bit
            let sibling_bits = bits ^ 1;
            let sibling_used = codes
                .iter()
                .enumerate()
                .any(|(j, &(b, l))| j != i && l == len && b == sibling_bits);
            if sibling_used {
                continue;
            }
            // Parent must be prefix-free against every other code
            let ok = codes.iter().enumerate().all(|(j, &(b, l))| {
                j == i
                    || (!egap_is_prefix(parent_bits, parent_len, b, l)
                        && !egap_is_prefix(b, l, parent_bits, parent_len))
            });
            if ok {
                codes[i] = (parent_bits, parent_len);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Sort by length ascending, then by bits
    codes.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    codes
}

/// Build EGAP prefix-free codes from byte frequency analysis.
/// Uses six phases: frequency table → trie generation → frequency-sorted
/// assignment → gap exploitation → prefix-free verification → swap pass.
fn build_egap_codes(data: &[u8]) -> Vec<EgapCode> {
    // ── Phase 1: build frequency table ────────────────────────────────────
    const FREQ_CHUNK: usize = 4 * 1024 * 1024;
    let counts = data
        .par_chunks(FREQ_CHUNK)
        .fold(
            || [0u64; 256],
            |mut acc, chunk| {
                for &b in chunk {
                    acc[b as usize] += 1;
                }
                acc
            },
        )
        .reduce(
            || [0u64; 256],
            |mut a, b| {
                for i in 0..256 {
                    a[i] += b[i];
                }
                a
            },
        );

    let mut present: Vec<(u8, u64)> = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (i as u8, c))
        .collect();
    let absent_count = 256 - present.len();

    let n = present.len();
    if n == 0 {
        return vec![];
    }

    // Sort by frequency descending (ties: lower byte value first)
    present.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    // Frequency lookup
    let mut freq_of = [0u64; 256];
    for &(sym, freq) in &present {
        freq_of[sym as usize] = freq;
    }

    // ── Phase 2: generate prefix-free code set via complete trie ──────────
    let raw_codes = generate_prefix_codes(n);

    // ── Phase 3: frequency-sorted assignment ──────────────────────────────
    // raw_codes is sorted shortest→longest; present is sorted most→least freq.
    // Most frequent symbol → shortest code.
    let mut codes: Vec<EgapCode> = present
        .iter()
        .zip(raw_codes.iter())
        .map(|(&(sym, _), &(bits, len))| EgapCode {
            symbol: sym,
            code_len: len,
            code_bits: bits,
        })
        .collect();

    // ── Phase 4: gap exploitation ─────────────────────────────────────────
    // For each code (longest first), try replacing it with any shorter
    // prefix-free code that is available thanks to absent-symbol gaps.
    let mut gap_shortenings = 0u32;
    loop {
        let mut improved = false;
        // Process longest codes first for maximum byte savings
        let mut indices: Vec<usize> = (0..codes.len()).collect();
        indices.sort_by(|&a, &b| codes[b].code_len.cmp(&codes[a].code_len));

        for &i in &indices {
            let cur_len = codes[i].code_len;
            if cur_len <= 1 {
                continue;
            }
            let mut found = false;
            // Try every shorter length (shortest first → greedy)
            for try_len in 1..cur_len {
                let num_at_len = 1u32 << try_len;
                for try_bits in 0..num_at_len {
                    let ok = codes.iter().enumerate().all(|(j, c)| {
                        j == i
                            || (!egap_is_prefix(try_bits, try_len, c.code_bits, c.code_len)
                                && !egap_is_prefix(c.code_bits, c.code_len, try_bits, try_len))
                    });
                    if ok {
                        codes[i].code_bits = try_bits;
                        codes[i].code_len = try_len;
                        gap_shortenings += 1;
                        improved = true;
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
        }
        if !improved {
            break;
        }
    }
    if gap_shortenings > 0 {
        // gap exploitations were applied
    }

    // ── Phase 5: verify prefix-free property (debug builds) ───────────────
    #[cfg(debug_assertions)]
    {
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                let a = &codes[i];
                let b = &codes[j];
                debug_assert!(
                    !egap_is_prefix(a.code_bits, a.code_len, b.code_bits, b.code_len)
                        && !egap_is_prefix(b.code_bits, b.code_len, a.code_bits, a.code_len),
                    "EGAP phase 5: prefix conflict sym {} (len {}) vs sym {} (len {})",
                    a.symbol,
                    a.code_len,
                    b.symbol,
                    b.code_len
                );
            }
        }
    }

    // ── Phase 6: swap pass — ensure frequency-optimal assignment ──────────
    // Swapping codes between two symbols preserves the code set, so the
    // prefix-free property is trivially maintained.
    let mut swap_count = 0u32;
    loop {
        let mut swapped = false;
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                let fi = freq_of[codes[i].symbol as usize];
                let fj = freq_of[codes[j].symbol as usize];
                let li = codes[i].code_len as u64;
                let lj = codes[j].code_len as u64;

                // Swap is beneficial when it reduces total bit cost
                if fi * li + fj * lj > fi * lj + fj * li {
                    let tmp_len = codes[i].code_len;
                    let tmp_bits = codes[i].code_bits;
                    codes[i].code_len = codes[j].code_len;
                    codes[i].code_bits = codes[j].code_bits;
                    codes[j].code_len = tmp_len;
                    codes[j].code_bits = tmp_bits;
                    swap_count += 1;
                    swapped = true;
                }
            }
        }
        if !swapped {
            break;
        }
    }

    // Post-swap prefix-free verification
    #[cfg(debug_assertions)]
    {
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                let a = &codes[i];
                let b = &codes[j];
                debug_assert!(
                    !egap_is_prefix(a.code_bits, a.code_len, b.code_bits, b.code_len)
                        && !egap_is_prefix(b.code_bits, b.code_len, a.code_bits, a.code_len),
                    "EGAP phase 6: prefix conflict after swap sym {} vs sym {}",
                    a.symbol,
                    b.symbol
                );
            }
        }
    }

    if swap_count > 0 {
        // swap optimizations were applied
    }

    let total_freq: u64 = present.iter().map(|(_, f)| f).sum();
    let avg_bits: f64 = codes
        .iter()
        .map(|c| c.code_len as f64 * freq_of[c.symbol as usize] as f64)
        .sum::<f64>()
        / total_freq as f64;
    codes
}

/// Serialize EGAP manifest into header bytes.
/// Format: "EGAP"(4) + original_size(4) + num_symbols(1) + padding_bits(1)
///         + [symbol(1) + code_len(1) + code_bits(4)] * n
fn serialize_egap_header(
    codes: &[EgapCode],
    original_size: u32,
    padding_bits: u8,
) -> Vec<u8> {
    let mut hdr = Vec::with_capacity(10 + codes.len() * 6);
    hdr.extend_from_slice(b"EGAP");
    hdr.extend_from_slice(&original_size.to_le_bytes());
    hdr.push(codes.len() as u8);
    hdr.push(padding_bits);
    for c in codes {
        hdr.push(c.symbol);
        hdr.push(c.code_len);
        hdr.extend_from_slice(&c.code_bits.to_le_bytes());
    }
    hdr
}

/// Deserialize EGAP header. Returns (codes, original_size, padding_bits, bytes_consumed).
fn deserialize_egap_header(data: &[u8]) -> io::Result<(Vec<EgapCode>, u32, u8, usize)> {
    if data.len() < 10 || &data[0..4] != b"EGAP" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Missing EGAP header magic",
        ));
    }
    let original_size = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let num_symbols = data[8] as usize;
    let padding_bits = data[9];
    let table_bytes = num_symbols * 6;
    let total = 10 + table_bytes;
    if data.len() < total {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated EGAP header",
        ));
    }
    let mut codes = Vec::with_capacity(num_symbols);
    let mut off = 10;
    for _ in 0..num_symbols {
        let symbol = data[off];
        let code_len = data[off + 1];
        let code_bits = u32::from_le_bytes(data[off + 2..off + 6].try_into().unwrap());
        codes.push(EgapCode {
            symbol,
            code_len,
            code_bits,
        });
        off += 6;
    }
    Ok((codes, original_size, padding_bits, total))
}

/// Entropy-encode data using EGAP prefix-free codes.
/// Returns (egap_header ++ encoded_bitstream) as a single Vec.
fn entropy_encode(data: &[u8]) -> Vec<u8> {
    let t_total = Instant::now();

    if data.is_empty() {
        return serialize_egap_header(&[], 0, 0);
    }

    let codes = build_egap_codes(data);

    // Build lookup: symbol → (code_len, code_bits)
    let mut lut = [(0u8, 0u32); 256];
    for c in &codes {
        lut[c.symbol as usize] = (c.code_len, c.code_bits);
    }

    // Calculate total bits needed
    let total_bits: u64 = data.iter().map(|&b| lut[b as usize].0 as u64).sum();
    let total_bytes = ((total_bits + 7) / 8) as usize;
    let padding_bits = ((total_bytes as u64 * 8) - total_bits) as u8;

    let hdr = serialize_egap_header(&codes, data.len() as u32, padding_bits);

    // Check if encoding actually saves space
    let overhead = hdr.len() + total_bytes;
    if overhead >= data.len() {
        let mut result = serialize_egap_header(&[], data.len() as u32, 0);
        result.extend_from_slice(data);
        return result;
    }

    // Encode: pack bits MSB-first into output bytes using a u64 bit accumulator
    let t_enc = Instant::now();
    let mut out = vec![0u8; total_bytes];
    let mut bit_acc: u64 = 0;
    let mut bits_in_acc: u32 = 0;
    let mut byte_pos: usize = 0;

    for &byte in data {
        let (len, bits) = lut[byte as usize];
        let len = len as u32;
        // Shift accumulator left and add new code bits
        bit_acc = (bit_acc << len) | (bits as u64);
        bits_in_acc += len;

        // Flush complete bytes
        while bits_in_acc >= 8 {
            bits_in_acc -= 8;
            out[byte_pos] = (bit_acc >> bits_in_acc) as u8;
            byte_pos += 1;
            // Mask off flushed bits
            bit_acc &= (1u64 << bits_in_acc) - 1;
        }
    }
    // Flush remaining bits (with zero padding)
    if bits_in_acc > 0 {
        out[byte_pos] = (bit_acc << (8 - bits_in_acc)) as u8;
    }

    let enc_elapsed = t_enc.elapsed();
    let throughput = data.len() as f64 / enc_elapsed.as_secs_f64() / 1_000_000.0;

    let mut result = Vec::with_capacity(hdr.len() + out.len());
    result.extend_from_slice(&hdr);
    result.extend_from_slice(&out);

    result
}

/// EGAP-decode: read header + bitstream, reconstruct original bytes.
fn entropy_decode(data: &[u8]) -> io::Result<Vec<u8>> {
    let t_total = Instant::now();

    let (codes, original_size, padding_bits, hdr_len) = deserialize_egap_header(data)?;

    // If no symbols in table, payload is passed through uncompressed
    if codes.is_empty() {
        return Ok(data[hdr_len..].to_vec());
    }

    let bitstream = &data[hdr_len..];
    let total_bits = (bitstream.len() as u64 * 8).saturating_sub(padding_bits as u64);
    let original_size = original_size as usize;

    // Build a decode trie. Flat array: children of node i are at 2*i+1 (bit 0)
    // and 2*i+2 (bit 1). Max code length determines tree depth.
    let max_depth = codes.iter().map(|c| c.code_len).max().unwrap_or(0) as usize;
    let tree_size = (1usize << (max_depth + 1)).saturating_sub(1).max(3);
    let mut tree_symbols: Vec<Option<u8>> = vec![None; tree_size];

    for c in &codes {
        let mut node = 0usize;
        for i in 0..c.code_len {
            let bit = ((c.code_bits >> (c.code_len - 1 - i)) & 1) as usize;
            node = 2 * node + 1 + bit;
            if node >= tree_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "EGAP code too deep for decode tree",
                ));
            }
        }
        tree_symbols[node] = Some(c.symbol);
    }

    // Decode bit-by-bit, walking the trie
    let t_dec = Instant::now();
    let mut output = Vec::with_capacity(original_size);
    let mut bit_idx: u64 = 0;
    let mut node = 0usize;

    while output.len() < original_size && bit_idx < total_bits {
        let byte_pos = (bit_idx / 8) as usize;
        let bit_off = 7 - (bit_idx % 8);
        let bit = ((bitstream[byte_pos] >> bit_off) & 1) as usize;
        bit_idx += 1;

        node = 2 * node + 1 + bit;
        if node >= tree_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "EGAP decode walked off tree",
            ));
        }

        if let Some(sym) = tree_symbols[node] {
            output.push(sym);
            node = 0;
        }
    }

    if output.len() != original_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "EGAP size mismatch: {} != {}",
                output.len(),
                original_size
            ),
        ));
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// EGAP adaptive windowing — variable-size segments based on symbol diversity
// ---------------------------------------------------------------------------

const EGAP_MIN_SEGMENT: usize = 1024; // 1 KB minimum segment to avoid manifest bloat

/// Scan forward counting unique symbols; cut a segment when 256 distinct values
/// are reached (point of maximum entropy where EGAP cannot help). Encode each
/// segment independently with entropy_encode. Returns concatenated encoded
/// segments and a manifest of per-segment compressed sizes.
fn egap_adaptive_encode(data: &[u8]) -> (Vec<u8>, Vec<u32>) {
    if data.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let t_total = Instant::now();
    let mut output: Vec<u8> = Vec::with_capacity(data.len());
    let mut segment_sizes: Vec<u32> = Vec::new();
    let mut pos = 0usize;

    while pos < data.len() {
        // Scan forward counting unique symbols
        let mut seen = [false; 256];
        let mut distinct = 0usize;
        let mut boundary = data.len(); // default: rest of data

        for i in pos..data.len() {
            let b = data[i] as usize;
            if !seen[b] {
                distinct += 1;
                seen[b] = true;
                if distinct >= 256 {
                    // Hit 256 unique symbols — cut here (before this byte)
                    // Enforce minimum segment size
                    boundary = i.max(pos + EGAP_MIN_SEGMENT);
                    break;
                }
            }
        }

        let seg_end = boundary.min(data.len());
        let segment = &data[pos..seg_end];

        // Encode this segment with EGAP
        let encoded = entropy_encode(segment);
        segment_sizes.push(encoded.len() as u32);
        output.extend_from_slice(&encoded);

        pos = seg_end;
    }

    println!(
        "[egap-adaptive] {} segments from {:.1}MB, avg {:.1}KB/seg, {:.1}MB → {:.1}MB, time={:.3}s",
        segment_sizes.len(),
        data.len() as f64 / 1_000_000.0,
        data.len() as f64 / segment_sizes.len().max(1) as f64 / 1024.0,
        data.len() as f64 / 1_000_000.0,
        output.len() as f64 / 1_000_000.0,
        t_total.elapsed().as_secs_f64()
    );

    (output, segment_sizes)
}

/// Decode EGAP adaptive segments: split concatenated data by manifest sizes,
/// decode each with entropy_decode, return concatenated original data.
fn egap_adaptive_decode(data: &[u8], segment_sizes: &[u32]) -> io::Result<Vec<u8>> {
    let mut output: Vec<u8> = Vec::new();
    let mut pos = 0usize;

    for (i, &sz) in segment_sizes.iter().enumerate() {
        let sz = sz as usize;
        if pos + sz > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "EGAP segment {} truncated: need {} bytes at offset {}, have {}",
                    i, sz, pos, data.len() - pos
                ),
            ));
        }
        let decoded = entropy_decode(&data[pos..pos + sz])?;
        output.extend_from_slice(&decoded);
        pos += sz;
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// File format writing — large buffered writes (1 MB+)
// ---------------------------------------------------------------------------

fn build_stage_header(stage_entries: &[(u8, HashMap<u8, Vec<u8>>)]) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(4096);
    let num_stages = stage_entries.len();
    if num_stages > 255 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "Too many stages"));
    }
    buf.push(num_stages as u8);

    for (escape_byte, code_map) in stage_entries {
        buf.push(*escape_byte);
        let num_codes = code_map.len();
        if num_codes > 255 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Too many codes"));
        }
        buf.push(num_codes as u8);
        for (&code, chunk_bytes) in code_map {
            buf.push(code);
            buf.extend_from_slice(&(chunk_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(chunk_bytes);
        }
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// System memory detection & chunk sizing
// ---------------------------------------------------------------------------

fn get_available_memory() -> usize {
    #[cfg(target_os = "windows")]
    {
        use std::mem;
        #[repr(C)]
        struct MemoryStatusEx {
            dw_length: u32,
            dw_memory_load: u32,
            ull_total_phys: u64,
            ull_avail_phys: u64,
            ull_total_page_file: u64,
            ull_avail_page_file: u64,
            ull_total_virtual: u64,
            ull_avail_virtual: u64,
            ull_avail_extended_virtual: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
        }
        unsafe {
            let mut status: MemoryStatusEx = mem::zeroed();
            status.dw_length = mem::size_of::<MemoryStatusEx>() as u32;
            if GlobalMemoryStatusEx(&mut status) != 0 {
                return status.ull_avail_phys as usize;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb as usize * 1024;
                        }
                    }
                }
            }
        }
    }
    8 * 1024 * 1024 * 1024 // 8 GB fallback
}

fn calculate_chunk_size(available_ram: usize) -> usize {
    let num_threads = rayon::current_num_threads();
    // Budget: 40% of available RAM, divided by (threads * 2) for input+output buffers
    let budget = (available_ram as f64 * RAM_BUDGET_FRACTION) as usize;
    let chunk = budget / (num_threads * 2).max(1);
    chunk.clamp(DICT_STAGE_MIN_CHUNK, DICT_STAGE_MAX_CHUNK)
}

// ---------------------------------------------------------------------------
// Temporary file management with cleanup-on-drop
// ---------------------------------------------------------------------------

struct TempFileManager {
    temp_dir: PathBuf,
    session_id: String,
    registered_files: Vec<PathBuf>,
}

impl TempFileManager {
    fn new() -> Self {
        let temp_dir = std::env::temp_dir().join("duckomp");
        fs::create_dir_all(&temp_dir).ok();
        let session_id = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );
        TempFileManager {
            temp_dir,
            session_id,
            registered_files: Vec::new(),
        }
    }

    fn chunk_path(&self, stage: usize, chunk: usize) -> PathBuf {
        self.temp_dir
            .join(format!("dk_{}_s{}_c{}.tmp", self.session_id, stage, chunk))
    }

    fn register(&mut self, path: PathBuf) {
        self.registered_files.push(path);
    }

    fn stage_files(&self, stage: usize) -> Vec<PathBuf> {
        let tag = format!("_s{}_c", stage);
        let mut files: Vec<PathBuf> = self
            .registered_files
            .iter()
            .filter(|p| {
                p.file_name()
                    .map(|f| f.to_string_lossy().contains(&tag))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        files.sort();
        files
    }

    fn delete_stage(&mut self, stage: usize) {
        let tag = format!("_s{}_c", stage);
        self.registered_files.retain(|p| {
            let belongs = p
                .file_name()
                .map(|f| f.to_string_lossy().contains(&tag))
                .unwrap_or(false);
            if belongs {
                fs::remove_file(p).ok();
            }
            !belongs
        });
    }

    fn cleanup_all(&mut self) {
        for path in self.registered_files.drain(..) {
            fs::remove_file(&path).ok();
        }
    }
}

impl Drop for TempFileManager {
    fn drop(&mut self) {
        self.cleanup_all();
    }
}

// ---------------------------------------------------------------------------
// Chunked compression helpers
// ---------------------------------------------------------------------------

/// Select escape byte by scanning multiple temp files (accumulate byte frequencies).
fn select_escape_byte_from_chunks(files: &[PathBuf]) -> u8 {
    let mut total_counts = [0u64; 256];
    for path in files {
        let file = fs::File::open(path).expect("open temp chunk for escape scan");
        let mmap = unsafe { Mmap::map(&file).expect("mmap temp chunk for escape scan") };
        let data: &[u8] = &mmap;
        let counts = data
            .par_chunks(4 * 1024 * 1024)
            .fold(
                || [0u64; 256],
                |mut acc, chunk| {
                    for &b in chunk {
                        acc[b as usize] += 1;
                    }
                    acc
                },
            )
            .reduce(
                || [0u64; 256],
                |mut a, b| {
                    for i in 0..256 {
                        a[i] += b[i];
                    }
                    a
                },
            );
        for i in 0..256 {
            total_counts[i] += counts[i];
        }
    }
    for b in 0u8..=255 {
        if total_counts[b as usize] == 0 {
            return b;
        }
    }
    total_counts
        .iter()
        .enumerate()
        .min_by_key(|(_, &c)| c)
        .map(|(b, _)| b as u8)
        .unwrap_or(0)
}

/// Sample spread data from a set of temp files, returning up to `target` bytes.
fn sample_from_chunks(files: &[PathBuf], target: usize) -> Vec<u8> {
    if files.is_empty() {
        return vec![];
    }
    let total_size: usize = files
        .iter()
        .map(|p| fs::metadata(p).map(|m| m.len() as usize).unwrap_or(0))
        .sum();
    if total_size <= target {
        let mut out = Vec::with_capacity(total_size);
        for path in files {
            if let Ok(data) = fs::read(path) {
                out.extend_from_slice(&data);
            }
        }
        return out;
    }
    let per_chunk = target / files.len().max(1);
    let mut out = Vec::with_capacity(target);
    for path in files {
        let file = fs::File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let data: &[u8] = &mmap;
        if data.len() <= per_chunk {
            out.extend_from_slice(data);
        } else {
            let sample = sample_spread_parallel(data, per_chunk, 4);
            out.extend_from_slice(&sample);
        }
        if out.len() >= target {
            break;
        }
    }
    out.truncate(target);
    out
}

/// Encode all chunks through one dictionary stage, writing results to temp files.
/// Returns (temp_file_paths, total_encoded_size).
fn encode_chunks_to_temp(
    input_mmap: &[u8],
    prev_stage_files: &Option<Vec<PathBuf>>,
    chunk_size: usize,
    num_chunks: usize,
    escape_byte: u8,
    code_map: &HashMap<u8, Vec<u8>>,
    stage: usize,
    temp_mgr: &mut TempFileManager,
) -> io::Result<(Vec<PathBuf>, usize)> {
    let total_size = input_mmap.len();

    // Pre-generate all temp paths (chunk_path only needs &self)
    let temp_paths: Vec<PathBuf> = (0..num_chunks)
        .map(|i| temp_mgr.chunk_path(stage, i))
        .collect();

    // Encode + write in parallel — each chunk writes its own file
    let dict_progress = AtomicUsize::new(0);
    let encoded_sizes: Vec<usize> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let encoded = if let Some(files) = prev_stage_files {
                let file = fs::File::open(&files[chunk_idx]).unwrap();
                let mmap = unsafe { Mmap::map(&file).unwrap() };
                encode_with_map(&mmap, escape_byte, code_map)
            } else {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(total_size);
                encode_with_map(&input_mmap[start..end], escape_byte, code_map)
            };

            let encoded_len = encoded.len();
            fs::write(&temp_paths[chunk_idx], &encoded).unwrap();
            report_progress(&dict_progress, num_chunks, "dict-encode");
            encoded_len
        })
        .collect();

    let encoded_total: usize = encoded_sizes.iter().sum();

    // Register paths sequentially (needs &mut self)
    for path in &temp_paths {
        temp_mgr.register(path.clone());
    }

    Ok((temp_paths, encoded_total))
}

/// Run the full dictionary stage pipeline in chunked mode.
/// All chunks share the same dictionary per stage; Rayon parallelism within-chunk only.
/// Returns (stage_entries, final_chunk_temp_files).
fn run_chunked_stages(
    input_mmap: &[u8],
    initial_files: Option<Vec<PathBuf>>,
    chunk_size: usize,
    temp_mgr: &mut TempFileManager,
) -> io::Result<(Vec<(u8, HashMap<u8, Vec<u8>>)>, Vec<PathBuf>)> {
    let t_all = Instant::now();
    let total_size = input_mmap.len();
    let num_chunks = match &initial_files {
        Some(files) => files.len(),
        None => (total_size + chunk_size - 1) / chunk_size,
    };

    println!(
        "[chunked] {} chunks of {:.0}MB from {:.1}MB input",
        num_chunks,
        chunk_size as f64 / 1_000_000.0,
        total_size as f64 / 1_000_000.0
    );

    let mut stage_entries: Vec<(u8, HashMap<u8, Vec<u8>>)> = Vec::new();
    let mut prev_stage_files: Option<Vec<PathBuf>> = initial_files;
    let mut last_good_sizes: Option<Vec<usize>> = None;

    for iteration in 1..=MAX_STAGES {
        let t_stage = Instant::now();

        let current_total_size: usize = match &prev_stage_files {
            None => total_size,
            Some(files) => files
                .iter()
                .map(|p| fs::metadata(p).map(|m| m.len() as usize).unwrap_or(0))
                .sum(),
        };

        println!(
            "\nStage {} [chunked]: (data: {:.1}MB, {} chunks)",
            iteration,
            current_total_size as f64 / 1_000_000.0,
            num_chunks
        );

        // --- Probe phase ---
        let probe_target = match current_total_size {
            0..=2_000_000 => current_total_size,
            2_000_001..=20_000_000 => 1_000_000,
            _ => 2_000_000,
        };
        print!(
            "  probing ({:.1}MB sample)...  ",
            probe_target as f64 / 1_000_000.0
        );
        io::stdout().flush().ok();

        let t_probe = Instant::now();
        let probe_sample = match &prev_stage_files {
            None => sample_spread_parallel(input_mmap, probe_target, 16),
            Some(files) => sample_from_chunks(files, probe_target),
        };
        let probe = probe_chunk_sizes(&probe_sample);
        let mut analysis_sizes = select_analysis_sizes(&probe);
        println!(
            "  [timing] probe: {:.3}s",
            t_probe.elapsed().as_secs_f64()
        );

        let top_n = probe.sizes.len().min(10);
        print!("  probe top {}: ", top_n);
        for i in 0..top_n {
            if probe.scores[i] > 0 {
                print!("{}({}k) ", probe.sizes[i], probe.scores[i] / 1000);
            }
        }
        println!();

        // Fallback: hotspot sampling
        if analysis_sizes.is_empty() {
            print!("  probe weak — trying hotspot...  ");
            io::stdout().flush().ok();
            let hotspot_data = match &prev_stage_files {
                None => sample_hotspots(input_mmap, probe_target * 2, 65536),
                Some(files) => sample_from_chunks(files, probe_target * 2),
            };
            let probe2 = probe_chunk_sizes(&hotspot_data);
            analysis_sizes = select_analysis_sizes(&probe2);
            if analysis_sizes.is_empty() {
                if let Some(ref prev) = last_good_sizes {
                    println!("reusing last good sizes {:?}", prev);
                    analysis_sizes = prev.clone();
                } else {
                    println!("no signal, stopping.");
                    break;
                }
            } else {
                println!("found sizes {:?}", analysis_sizes);
            }
        }

        // --- Analysis phase (cross-reference sampling) ---
        let use_cross_ref = current_total_size > 4_000_000;
        let max_regions = if probe_is_uniform(&probe, 0.90) {
            12
        } else {
            24
        };
        let num_xref_regions = (current_total_size / SAMPLE_REGION_SIZE).max(4).min(max_regions);

        let t_analysis = Instant::now();

        let analysis_data: Cow<'_, [u8]> = match &prev_stage_files {
            None => Cow::Borrowed(input_mmap),
            Some(files) => {
                let target = SAMPLE_REGION_SIZE * num_xref_regions * 2;
                Cow::Owned(sample_from_chunks(files, target.min(current_total_size)))
            }
        };

        let (candidates, corpus_desc) = if use_cross_ref {
            let threshold =
                occurrence_threshold_by_data_length(SAMPLE_REGION_SIZE * num_xref_regions);
            print!(
                "  cross-ref ({} x {}KB, sizes {:?})...  ",
                num_xref_regions,
                SAMPLE_REGION_SIZE / 1024,
                analysis_sizes
            );
            io::stdout().flush().ok();
            let cands = parallel_cross_reference_sample(
                &analysis_data,
                num_xref_regions,
                &analysis_sizes,
                threshold,
            );
            (
                cands,
                format!("{}x{}KB xref", num_xref_regions, SAMPLE_REGION_SIZE / 1024),
            )
        } else {
            print!(
                "  analysing (sizes {:?}, {:.1}MB)...  ",
                analysis_sizes,
                current_total_size as f64 / 1_000_000.0
            );
            io::stdout().flush().ok();
            let corpus = build_analysis_corpus(&analysis_data, current_total_size / 2);
            let cands = find_occurrences_for_sizes(&corpus, &analysis_sizes, None);
            (
                cands,
                format!("{:.1}MB full", current_total_size as f64 / 1_000_000.0),
            )
        };
        drop(analysis_data);
        println!(
            "  [timing] analysis ({}): {:.3}s, {} candidates",
            corpus_desc,
            t_analysis.elapsed().as_secs_f64(),
            candidates.len()
        );

        let best_chunks = select_best_chunks(&candidates, MAX_CODES);
        println!("  {} chunks selected", best_chunks.len());

        if best_chunks.is_empty() {
            // Try once more with a larger corpus before giving up
            print!("  retrying with larger corpus...  ");
            io::stdout().flush().ok();
            let retry_target = match current_total_size {
                0..=2_000_000 => current_total_size,
                2_000_001..=20_000_000 => current_total_size / 4,
                20_000_001..=100_000_000 => 8_000_000,
                _ => 12_000_000,
            };
            let larger_data: Cow<'_, [u8]> = match &prev_stage_files {
                None => Cow::Owned(sample_spread_parallel(input_mmap, retry_target, 16)),
                Some(files) => Cow::Owned(sample_from_chunks(files, retry_target.min(current_total_size))),
            };
            let corpus2 = build_analysis_corpus(&larger_data, larger_data.len() / 2);
            let candidates2 = find_occurrences_for_sizes(&corpus2, &analysis_sizes, None);
            let best_chunks2 = select_best_chunks(&candidates2, MAX_CODES);
            drop(larger_data);

            if best_chunks2.is_empty() {
                println!("still nothing, stopping.");
                break;
            }

            let escape_byte = match &prev_stage_files {
                None => select_escape_byte(input_mmap),
                Some(files) => select_escape_byte_from_chunks(files),
            };
            let code_map: HashMap<u8, Vec<u8>> = best_chunks2
                .into_iter()
                .enumerate()
                .map(|(i, c)| ((i + 1) as u8, c))
                .collect();

            let (new_files, encoded_total) = encode_chunks_to_temp(
                input_mmap,
                &prev_stage_files,
                chunk_size,
                num_chunks,
                escape_byte,
                &code_map,
                iteration,
                temp_mgr,
            )?;

            let improvement = (current_total_size as f64 - encoded_total as f64)
                / current_total_size as f64;
            println!(
                "  {} -> {} ({:.3}% improvement)",
                current_total_size,
                encoded_total,
                improvement * 100.0
            );

            if encoded_total < current_total_size {
                stage_entries.push((escape_byte, code_map));
                if prev_stage_files.is_some() {
                    temp_mgr.delete_stage(iteration - 1);
                }
                prev_stage_files = Some(new_files);
                last_good_sizes = Some(analysis_sizes);
            } else {
                for f in &new_files {
                    fs::remove_file(f).ok();
                }
                temp_mgr.registered_files.retain(|p| !new_files.contains(p));
            }
            println!(
                "  [timing] stage {} total: {:.3}s",
                iteration,
                t_stage.elapsed().as_secs_f64()
            );
            break;
        }

        last_good_sizes = Some(analysis_sizes.clone());

        // --- Escape byte selection ---
        let t_esc = Instant::now();
        let escape_byte = match &prev_stage_files {
            None => select_escape_byte(input_mmap),
            Some(files) => select_escape_byte_from_chunks(files),
        };
        println!(
            "  [timing] escape byte: {:.3}s (0x{:02X})",
            t_esc.elapsed().as_secs_f64(),
            escape_byte
        );

        let code_map: HashMap<u8, Vec<u8>> = best_chunks
            .into_iter()
            .enumerate()
            .map(|(i, c)| ((i + 1) as u8, c))
            .collect();

        // --- Encode chunks sequentially ---
        let (new_files, encoded_total) = encode_chunks_to_temp(
            input_mmap,
            &prev_stage_files,
            chunk_size,
            num_chunks,
            escape_byte,
            &code_map,
            iteration,
            temp_mgr,
        )?;

        let improvement =
            (current_total_size as f64 - encoded_total as f64) / current_total_size as f64;

        if improvement < MIN_IMPROVEMENT_RATIO {
            if encoded_total < current_total_size {
                println!(
                    "  {} -> {} ({:.3}% below threshold — committing and stopping.)",
                    current_total_size,
                    encoded_total,
                    improvement * 100.0
                );
                stage_entries.push((escape_byte, code_map));
                if prev_stage_files.is_some() {
                    temp_mgr.delete_stage(iteration - 1);
                }
                prev_stage_files = Some(new_files);
            } else {
                println!(
                    "  {} -> {} (no improvement, stopping.)",
                    current_total_size, encoded_total
                );
                for f in &new_files {
                    fs::remove_file(f).ok();
                }
                temp_mgr.registered_files.retain(|p| !new_files.contains(p));
            }
            println!(
                "  [timing] stage {} total: {:.3}s",
                iteration,
                t_stage.elapsed().as_secs_f64()
            );
            break;
        }

        println!(
            "  {} -> {} ({:.2}% improvement)",
            current_total_size,
            encoded_total,
            improvement * 100.0
        );
        println!(
            "  [timing] stage {} total: {:.3}s",
            iteration,
            t_stage.elapsed().as_secs_f64()
        );
        stage_entries.push((escape_byte, code_map));
        if prev_stage_files.is_some() {
            temp_mgr.delete_stage(iteration - 1);
        }
        prev_stage_files = Some(new_files);
    }

    if stage_entries.is_empty() {
        println!("No beneficial stages found in chunked mode.");
    }

    println!(
        "\n[timing] all chunked stages: {:.3}s ({} stages committed)",
        t_all.elapsed().as_secs_f64(),
        stage_entries.len()
    );

    // If no stages committed, create temp files from original raw data
    let final_files = match prev_stage_files {
        Some(files) => files,
        None => {
            let mut files = Vec::with_capacity(num_chunks);
            for chunk_idx in 0..num_chunks {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(total_size);
                let path = temp_mgr.chunk_path(0, chunk_idx);
                fs::write(&path, &input_mmap[start..end])?;
                temp_mgr.register(path.clone());
                files.push(path);
            }
            files
        }
    };

    Ok((stage_entries, final_files))
}

// ---------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------

/// Detect whether probe scores are nearly flat (uniform), meaning most
/// sizes perform similarly and there is no clear winner. Returns true if
/// the top N positive scores are within `tolerance` of each other.
fn probe_is_uniform(probe: &ProbeResult, tolerance: f64) -> bool {
    let positive: Vec<i64> = probe.scores.iter().copied().filter(|&s| s > 0).collect();
    if positive.len() < 4 {
        return false;
    }
    let top = positive[0] as f64;
    if top <= 0.0 {
        return true;
    }
    // Check if the 4th-best score is within tolerance of the best
    let fourth = positive[3.min(positive.len() - 1)] as f64;
    (fourth / top) >= tolerance
}

/// How many top sizes to use in full analysis, based on how strong the
/// probe signal is. If the best sizes score much higher than the rest,
/// focus tightly. If scores are similar, cast a wider net.
fn select_analysis_sizes(probe: &ProbeResult) -> Vec<usize> {
    let positive: Vec<(usize, i64)> = probe.sizes.iter().zip(probe.scores.iter())
        .filter(|(_, &s)| s > 0)
        .map(|(&sz, &sc)| (sz, sc))
        .collect();

    if positive.is_empty() {
        return vec![];
    }

    let top_score = positive[0].1 as f64;

    // If probe is nearly uniform, aggressively limit to top 3 sizes
    if probe_is_uniform(probe, 0.90) {
        let count = positive.len().min(3);
        println!("  [xref] uniform probe detected — limiting to top {} sizes", count);
        return positive[..count].iter().map(|(sz, _)| *sz).collect();
    }

    // Keep sizes whose score is at least 20% of the top score,
    // but always at most 8
    let good: Vec<usize> = positive.iter()
        .filter(|(_, s)| *s as f64 >= top_score * 0.20)
        .map(|(sz, _)| *sz)
        .collect();

    let count = good.len().min(8);
    good[..count].to_vec()
}

fn run_compression_stages(source_data: &[u8]) -> (Vec<(u8, HashMap<u8, Vec<u8>>)>, Vec<u8>) {
    let t_all_stages = Instant::now();
    let mut stage_entries: Vec<(u8, HashMap<u8, Vec<u8>>)> = Vec::new();
    let mut current_data: Cow<'_, [u8]> = Cow::Borrowed(source_data);
    let mut last_good_sizes: Option<Vec<usize>> = None;

    for iteration in 1..=MAX_STAGES {
        let t_stage = Instant::now();
        println!("Stage {}:  (data: {:.1}MB)", iteration, current_data.len() as f64 / 1_000_000.0);

        // --- Probe phase (parallel spread sample) ---
        let probe_target = match current_data.len() {
            0..=2_000_000 => current_data.len(),
            2_000_001..=20_000_000 => 1_000_000,
            _ => 2_000_000,
        };
        print!("  probing ({:.1}MB spread sample, sizes {}-{})...  ",
            probe_target as f64 / 1_000_000.0, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
        io::stdout().flush().ok();

        let t_probe = Instant::now();
        let probe_sample = sample_spread_parallel(&current_data, probe_target, 16);
        let t_sample_done = t_probe.elapsed();
        let probe = probe_chunk_sizes(&probe_sample);
        let t_probe_done = t_probe.elapsed();
        let mut analysis_sizes = select_analysis_sizes(&probe);
        println!("  [timing] sample: {:.3}s, probe: {:.3}s",
            t_sample_done.as_secs_f64(), t_probe_done.as_secs_f64());

        // Print probe summary
        println!();
        let top_n = probe.sizes.len().min(10);
        print!("  probe top {}: ", top_n);
        for i in 0..top_n {
            if probe.scores[i] > 0 {
                print!("{}({}k) ", probe.sizes[i], probe.scores[i] / 1000);
            }
        }
        println!();

        // If probe found nothing useful, try hotspot sampling as a second chance
        if analysis_sizes.is_empty() {
            print!("  probe weak — trying hotspot sample...  ");
            io::stdout().flush().ok();
            let hotspot_sample = sample_hotspots(&current_data, probe_target * 2, 65536);
            let probe2 = probe_chunk_sizes(&hotspot_sample);
            analysis_sizes = select_analysis_sizes(&probe2);

            if analysis_sizes.is_empty() {
                // Fall back to whatever worked last stage if we have it
                if let Some(ref prev_sizes) = last_good_sizes {
                    println!("hotspot weak — reusing last good sizes {:?}", prev_sizes);
                    analysis_sizes = prev_sizes.clone();
                } else {
                    println!("no signal found, stopping.");
                    break;
                }
            } else {
                println!("hotspot found sizes {:?}", analysis_sizes);
            }
        }

        // --- Full analysis phase ---
        // For large data, use cross-reference sampling across parallel 256KB
        // regions; for small data fall back to the original full-corpus path.
        let use_cross_ref = current_data.len() > 4_000_000;

        // Adaptive region count: if probe is uniform, fewer regions suffice
        let max_regions = if probe_is_uniform(&probe, 0.90) { 12 } else { 24 };
        let num_xref_regions = (current_data.len() / SAMPLE_REGION_SIZE).max(4).min(max_regions);

        let t_analysis = Instant::now();
        let (candidates, corpus_desc) = if use_cross_ref {
            let threshold = occurrence_threshold_by_data_length(
                SAMPLE_REGION_SIZE * num_xref_regions,
            );
            print!("  cross-ref sampling ({} x {}KB regions, sizes {:?})...  ",
                num_xref_regions, SAMPLE_REGION_SIZE / 1024, analysis_sizes);
            io::stdout().flush().ok();
            let cands = parallel_cross_reference_sample(
                &current_data,
                num_xref_regions,
                &analysis_sizes,
                threshold,
            );
            (cands, format!("{}x{}KB xref", num_xref_regions, SAMPLE_REGION_SIZE / 1024))
        } else {
            let corpus_target = current_data.len();
            print!("  analysing (sizes {:?}, {:.1}MB corpus)...  ",
                analysis_sizes, corpus_target as f64 / 1_000_000.0);
            io::stdout().flush().ok();
            let corpus = build_analysis_corpus(&current_data, corpus_target / 2);
            let cands = find_occurrences_for_sizes(&corpus, &analysis_sizes, None);
            (cands, format!("{:.1}MB full", corpus_target as f64 / 1_000_000.0))
        };
        println!("  [timing] analysis ({}): {:.3}s, {} candidates",
            corpus_desc, t_analysis.elapsed().as_secs_f64(), candidates.len());

        let t_select = Instant::now();
        let best_chunks = select_best_chunks(&candidates, MAX_CODES);
        println!("  [timing] chunk selection: {:.3}s, {} chunks selected",
            t_select.elapsed().as_secs_f64(), best_chunks.len());

        if best_chunks.is_empty() {
            println!("no beneficial chunks found ({}).", corpus_desc);

            // Try once more with a larger corpus before giving up
            print!("  retrying with larger corpus...  ");
            io::stdout().flush().ok();

            let t_retry = Instant::now();
            let corpus_target = match current_data.len() {
                0..=2_000_000 => current_data.len(),
                2_000_001..=20_000_000 => current_data.len() / 4,
                20_000_001..=100_000_000 => 8_000_000,
                _ => 12_000_000,
            };
            let larger_corpus = build_analysis_corpus(
                &current_data,
                corpus_target.min(current_data.len()),
            );
            let candidates2 = find_occurrences_for_sizes(&larger_corpus, &analysis_sizes, None);
            let best_chunks2 = select_best_chunks(&candidates2, MAX_CODES);
            println!("  [timing] retry analysis: {:.3}s, {} candidates, {} selected",
                t_retry.elapsed().as_secs_f64(), candidates2.len(), best_chunks2.len());

            if best_chunks2.is_empty() {
                println!("still nothing, stopping.");
                break;
            }

            let t_esc = Instant::now();
            let escape_byte = select_escape_byte(&current_data);
            println!("  [timing] escape byte selection: {:.3}s (byte=0x{:02X})",
                t_esc.elapsed().as_secs_f64(), escape_byte);

            let code_map: HashMap<u8, Vec<u8>> = best_chunks2
                .into_iter()
                .enumerate()
                .map(|(i, chunk)| ((i + 1) as u8, chunk))
                .collect();

            print!("encoding...  ");
            io::stdout().flush().ok();
            let compressed_stage = encode_with_map(&current_data, escape_byte, &code_map);
            let improvement = (current_data.len() as f64 - compressed_stage.len() as f64)
                / current_data.len() as f64;

            println!("{} -> {} ({:.3}% improvement)", current_data.len(), compressed_stage.len(), improvement * 100.0);
            println!("  [timing] stage {} total: {:.3}s", iteration, t_stage.elapsed().as_secs_f64());

            if compressed_stage.len() < current_data.len() {
                stage_entries.push((escape_byte, code_map));
                current_data = Cow::Owned(compressed_stage);
                last_good_sizes = Some(analysis_sizes);
            }
            break;
        }

        last_good_sizes = Some(analysis_sizes.clone());

        let t_esc = Instant::now();
        let escape_byte = select_escape_byte(&current_data);
        println!("  [timing] escape byte selection: {:.3}s (byte=0x{:02X})",
            t_esc.elapsed().as_secs_f64(), escape_byte);

        let code_map: HashMap<u8, Vec<u8>> = best_chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| ((i + 1) as u8, chunk))
            .collect();

        print!("  encoding...  ");
        io::stdout().flush().ok();

        let compressed_stage = encode_with_map(&current_data, escape_byte, &code_map);
        let improvement = (current_data.len() as f64 - compressed_stage.len() as f64)
            / current_data.len() as f64;

        if improvement < MIN_IMPROVEMENT_RATIO {
            if compressed_stage.len() < current_data.len() {
                println!(
                    "{} -> {} ({:.3}% improvement, below threshold — committing and stopping.)",
                    current_data.len(), compressed_stage.len(), improvement * 100.0
                );
                stage_entries.push((escape_byte, code_map));
                current_data = Cow::Owned(compressed_stage);
            } else {
                println!("{} -> {} (no improvement, stopping.)", current_data.len(), compressed_stage.len());
            }
            println!("  [timing] stage {} total: {:.3}s", iteration, t_stage.elapsed().as_secs_f64());
            break;
        }

        println!(
            "{} -> {} ({:.2}% improvement)",
            current_data.len(), compressed_stage.len(), improvement * 100.0
        );
        println!("  [timing] stage {} total: {:.3}s", iteration, t_stage.elapsed().as_secs_f64());
        stage_entries.push((escape_byte, code_map));
        current_data = Cow::Owned(compressed_stage);
    }

    if stage_entries.is_empty() {
        println!("No beneficial stages found; writing raw output.");
    }

    println!("\n[timing] all stages total: {:.3}s ({} stages committed)",
        t_all_stages.elapsed().as_secs_f64(), stage_entries.len());

    (stage_entries, current_data.into_owned())
}

fn compress_file(file_path: &str) -> io::Result<()> {
    let t_total = Instant::now();

    let file = fs::File::open(file_path)?;
    let metadata = file.metadata()?;
    let original_size = metadata.len() as usize;

    if original_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Cannot compress empty file",
        ));
    }

    println!(
        "File size: {} bytes ({:.1} MB)",
        original_size,
        original_size as f64 / 1_000_000.0
    );

    let mmap = unsafe { Mmap::map(&file)? };

    // Detect available memory and decide dict-stage strategy
    let available_ram = get_available_memory();
    let dict_chunk_size = calculate_chunk_size(available_ram);
    let use_chunked_dict = original_size > dict_chunk_size;

    println!(
        "[memory] available: {:.1}GB, dict chunk: {:.0}MB, dict strategy: {}",
        available_ram as f64 / 1_000_000_000.0,
        dict_chunk_size as f64 / 1_000_000.0,
        if use_chunked_dict { "chunked" } else { "in-memory" }
    );

    let output_path = format!("{}.duckomp", file_path);
    let mut temp_mgr = TempFileManager::new();

    // =================================================================
    // Step 1: Dictionary stages (chunked or in-memory based on RAM)
    // =================================================================
    let stage_entries: Vec<(u8, HashMap<u8, Vec<u8>>)>;

    // Step 2 results (populated in whichever branch applies)
    let chunk_flags: Vec<u8>;
    let chunk_sizes: Vec<u64>;
    let post_dict_output: Vec<u8>;
    let dict_output_len: usize;
    let actual_chunk_size: usize;
    let t_chunk: Instant;

    if use_chunked_dict {
        let (entries, dict_chunks) =
            run_chunked_stages(&mmap, None, dict_chunk_size, &mut temp_mgr)?;
        drop(mmap);
        drop(file);
        stage_entries = entries;

        // =============================================================
        // Step 2 (chunked): process each dict chunk as a whole piece
        //   — full-chunk EGAP + adaptive LZ77 (like old pipeline)
        // =============================================================
        t_chunk = Instant::now();
        actual_chunk_size = dict_chunk_size;
        let num_dict_chunks = dict_chunks.len();
        let mut total_dict_size: usize = 0;
        let mut all_flags: Vec<u8> = Vec::with_capacity(num_dict_chunks);
        let mut all_sizes: Vec<u64> = Vec::with_capacity(num_dict_chunks);
        let mut all_output: Vec<u8> = Vec::new();

        for (i, dict_chunk_path) in dict_chunks.iter().enumerate() {
            let t_chunk_i = Instant::now();
            let cf = fs::File::open(dict_chunk_path)?;
            let cm = unsafe { Mmap::map(&cf)? };
            let dict_data: &[u8] = &cm;
            let dict_size = dict_data.len();
            total_dict_size += dict_size;

            let orig_chunk_size = if i < num_dict_chunks - 1 {
                dict_chunk_size
            } else {
                original_size - (num_dict_chunks - 1) * dict_chunk_size
            };

            let dict_reduction_pct = if orig_chunk_size > 0 {
                (1.0 - dict_size as f64 / orig_chunk_size as f64) * 100.0
            } else {
                0.0
            };

            let entropy = shannon_entropy(dict_data);

            // Adaptive stage decision (matching old pipeline logic)
            let run_egap;
            let mut run_lz77;

            if dict_reduction_pct >= ADAPTIVE_DICT_SKIP_ALL {
                run_egap = false;
                run_lz77 = false;
            } else if entropy >= ADAPTIVE_ENTROPY_RANDOM {
                run_egap = false;
                run_lz77 = false;
            } else if dict_reduction_pct >= ADAPTIVE_DICT_HIGH {
                run_egap = true;
                run_lz77 = entropy < ADAPTIVE_ENTROPY_HIGH;
            } else if dict_reduction_pct >= ADAPTIVE_DICT_MEDIUM {
                run_egap = true;
                run_lz77 = entropy < ADAPTIVE_ENTROPY_HIGH;
            } else {
                // dict < 5%: skip EGAP, run LZ77 only
                run_egap = false;
                run_lz77 = true;
            }

            let mut current_data: Vec<u8>;
            let mut flags: u8 = 0;

            if run_egap {
                let egap_data = entropy_encode(dict_data);
                if egap_data.len() < dict_size {
                    let egap_reduction_pct =
                        (1.0 - egap_data.len() as f64 / dict_size as f64) * 100.0;
                    flags |= CHUNK_FLAG_EGAP;
                    current_data = egap_data;

                    // Post-EGAP LZ77 decision
                    if run_lz77 {
                        if egap_reduction_pct < ADAPTIVE_EGAP_FLAT {
                            run_lz77 = false;
                        } else if egap_reduction_pct >= ADAPTIVE_EGAP_GOOD {
                            run_lz77 = true;
                        }
                    }
                } else {
                    current_data = dict_data.to_vec();
                }
            } else {
                current_data = dict_data.to_vec();
            }
            drop(cm);

            // Entropy check before LZ77
            if run_lz77 {
                let pre_lz77_entropy = shannon_entropy(&current_data);
                if pre_lz77_entropy >= ADAPTIVE_ENTROPY_RANDOM {
                    run_lz77 = false;
                }
            }

            if run_lz77 {
                let lz77_data = lz77_compress(&current_data);
                if lz77_data.len() < current_data.len() {
                    flags |= CHUNK_FLAG_LZ77;
                    current_data = lz77_data;
                }
            }

            // Post-encoding entropy pass on final chunk
            {
                let egap_post = entropy_encode(&current_data);
                if egap_post.len() < current_data.len() {
                    flags |= CHUNK_FLAG_POST_ENTROPY;
                    current_data = egap_post;
                }
            }

            println!(
                "[adaptive] chunk {}/{}: dict={:.1}%, entropy={:.1} → {} ({:.1}MB in {:.3}s)",
                i + 1, num_dict_chunks, dict_reduction_pct, entropy,
                adaptive_stage_label(flags),
                current_data.len() as f64 / 1_000_000.0,
                t_chunk_i.elapsed().as_secs_f64()
            );

            all_flags.push(flags);
            all_sizes.push(current_data.len() as u64);
            all_output.extend_from_slice(&current_data);
            drop(current_data);
        }

        // Clean up dict temp files
        for path in &dict_chunks {
            fs::remove_file(path).ok();
        }
        temp_mgr.registered_files.retain(|p| !dict_chunks.contains(p));

        dict_output_len = total_dict_size;
        chunk_flags = all_flags;
        chunk_sizes = all_sizes;
        post_dict_output = all_output;
    } else {
        let (entries, data) = run_compression_stages(&mmap);
        let input_size = mmap.len();
        drop(mmap);
        drop(file);
        stage_entries = entries;

        // =============================================================
        // Step 2 (in-memory): process full dict output as one piece
        // =============================================================
        t_chunk = Instant::now();
        actual_chunk_size = data.len();
        dict_output_len = data.len();

        let dict_reduction_pct = if input_size > 0 {
            (1.0 - data.len() as f64 / input_size as f64) * 100.0
        } else {
            0.0
        };
        let entropy = shannon_entropy(&data);

        let run_egap;
        let mut run_lz77;

        if dict_reduction_pct >= ADAPTIVE_DICT_SKIP_ALL {
            run_egap = false;
            run_lz77 = false;
        } else if entropy >= ADAPTIVE_ENTROPY_RANDOM {
            run_egap = false;
            run_lz77 = false;
        } else if dict_reduction_pct >= ADAPTIVE_DICT_HIGH {
            run_egap = true;
            run_lz77 = entropy < ADAPTIVE_ENTROPY_HIGH;
        } else if dict_reduction_pct >= ADAPTIVE_DICT_MEDIUM {
            run_egap = true;
            run_lz77 = entropy < ADAPTIVE_ENTROPY_HIGH;
        } else {
            run_egap = false;
            run_lz77 = true;
        }

        let mut current_data = data;
        let mut flags: u8 = 0;

        if run_egap {
            println!(
                "\n[egap] encoding {:.1}MB dictionary output...",
                current_data.len() as f64 / 1_000_000.0
            );
            let t_entropy = Instant::now();
            let pre_size = current_data.len();
            let egap_data = entropy_encode(&current_data);
            println!(
                "[timing] egap encode: {:.3}s",
                t_entropy.elapsed().as_secs_f64()
            );
            if egap_data.len() < pre_size {
                let egap_reduction_pct =
                    (1.0 - egap_data.len() as f64 / pre_size as f64) * 100.0;
                flags |= CHUNK_FLAG_EGAP;
                current_data = egap_data;

                if run_lz77 {
                    if egap_reduction_pct < ADAPTIVE_EGAP_FLAT {
                        run_lz77 = false;
                    } else if egap_reduction_pct >= ADAPTIVE_EGAP_GOOD {
                        run_lz77 = true;
                    }
                }
            }
        }

        if run_lz77 {
            let pre_lz77_entropy = shannon_entropy(&current_data);
            if pre_lz77_entropy >= ADAPTIVE_ENTROPY_RANDOM {
                run_lz77 = false;
            }
        }

        if run_lz77 {
            println!("\n[lz77] compressing {:.1}MB...", current_data.len() as f64 / 1_000_000.0);
            let t_lz77 = Instant::now();
            let lz77_data = lz77_compress(&current_data);
            println!(
                "[timing] lz77: {:.3}s ({:.1}MB -> {:.1}MB, {:.1}% reduction)",
                t_lz77.elapsed().as_secs_f64(),
                current_data.len() as f64 / 1_000_000.0,
                lz77_data.len() as f64 / 1_000_000.0,
                (1.0 - lz77_data.len() as f64 / current_data.len() as f64) * 100.0
            );
            if lz77_data.len() < current_data.len() {
                flags |= CHUNK_FLAG_LZ77;
                current_data = lz77_data;
            }
        }

        // Post-encoding entropy pass
        {
            let egap_post = entropy_encode(&current_data);
            if egap_post.len() < current_data.len() {
                flags |= CHUNK_FLAG_POST_ENTROPY;
                current_data = egap_post;
            }
        }

        println!(
            "[adaptive] stages: {} (dict={:.1}%, entropy={:.1})",
            adaptive_stage_label(flags), dict_reduction_pct, entropy
        );

        chunk_flags = vec![flags];
        chunk_sizes = vec![current_data.len() as u64];
        post_dict_output = current_data;
    }

    let num_chunks = chunk_flags.len();
    print_adaptive_summary(&chunk_flags);
    println!(
        "[timing] adaptive stages: {:.3}s, {:.1}MB → {:.1}MB ({} chunks)",
        t_chunk.elapsed().as_secs_f64(),
        dict_output_len as f64 / 1_000_000.0,
        post_dict_output.len() as f64 / 1_000_000.0,
        num_chunks
    );

    // =================================================================
    // Step 3: Build header and write output
    // =================================================================
    // Header: "DUCKOMP1"(8) + format_flags(1) + original_size(8)
    //       + num_chunks(4) + chunk_size(4) + stage_header
    //       + per-chunk metadata [flag(1) + size(8)] + payload
    let t_write = Instant::now();
    let format_flags: u8 = 0; // bit 0 = is_folder (0 for single file)
    let payload = post_dict_output;

    let mut header: Vec<u8> = Vec::with_capacity(4096);
    header.extend_from_slice(b"DUCKOMP1");
    header.push(format_flags);
    header.extend_from_slice(&(original_size as u64).to_le_bytes());
    header.extend_from_slice(&(num_chunks as u32).to_le_bytes());
    header.extend_from_slice(&(actual_chunk_size as u32).to_le_bytes());
    header.extend_from_slice(&build_stage_header(&stage_entries)?);

    // Per-chunk metadata: flag(1) + compressed_size(8)
    for i in 0..num_chunks {
        header.push(chunk_flags[i]);
        header.extend_from_slice(&chunk_sizes[i].to_le_bytes());
    }

    let total_file_size = header.len() + payload.len();

    {
        let out_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&output_path)?;
        out_file.set_len(total_file_size as u64)?;
        let mut out_mmap = unsafe { MmapMut::map_mut(&out_file)? };

        out_mmap[..header.len()].copy_from_slice(&header);
        out_mmap[header.len()..header.len() + payload.len()].copy_from_slice(&payload);

        out_mmap.flush()?;
    }
    drop(payload);

    println!(
        "[timing] write output: {:.3}s ({:.1}MB)",
        t_write.elapsed().as_secs_f64(),
        total_file_size as f64 / 1_000_000.0
    );

    let ratio = total_file_size as f64 / original_size as f64;
    let saved = original_size as i64 - total_file_size as i64;
    println!();
    println!("Compressed:   {} -> {}", file_path, output_path);
    println!(
        "Original:     {} bytes ({:.1} MB)",
        original_size,
        original_size as f64 / 1_000_000.0
    );
    println!(
        "Compressed:   {} bytes ({:.1} MB)",
        total_file_size,
        total_file_size as f64 / 1_000_000.0
    );
    println!("Saved:        {} bytes ({:.1}%)", saved, (1.0 - ratio) * 100.0);
    println!("Ratio:        {:.3}", ratio);
    println!("Wall time:    {:.3}s", t_total.elapsed().as_secs_f64());

    Ok(())
}

// ---------------------------------------------------------------------------
// Folder compression support
// ---------------------------------------------------------------------------

struct FileEntry {
    relative_path: String,
    file_size: u64,
}

fn collect_files(base: &Path, current: &Path, files: &mut Vec<(String, PathBuf)>) -> io::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(base, &path, files)?;
        } else {
            let relative = path
                .strip_prefix(base)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            // Normalize to forward slashes for cross-platform portability
            let relative_str = relative
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            files.push((relative_str, path));
        }
    }
    Ok(())
}

fn compress_folder(folder_path: &str) -> io::Result<()> {
    let t_total = Instant::now();
    let base = Path::new(folder_path);
    let t_collect = Instant::now();
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    collect_files(base, base, &mut files)?;

    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No files found in folder",
        ));
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    println!(
        "[timing] collect {} files: {:.3}s",
        files.len(),
        t_collect.elapsed().as_secs_f64()
    );

    let mut total_size: usize = 0;
    let mut file_entries: Vec<FileEntry> = Vec::with_capacity(files.len());
    for (path, file_path) in &files {
        let size = fs::metadata(file_path)?.len();
        total_size += size as usize;
        file_entries.push(FileEntry {
            relative_path: path.clone(),
            file_size: size,
        });
    }

    println!(
        "Folder: {} file(s), {} bytes total ({:.1} MB)",
        files.len(),
        total_size,
        total_size as f64 / 1_000_000.0
    );

    let mut temp_mgr = TempFileManager::new();
    let combined_path = temp_mgr.chunk_path(0, 0);
    let t_combine = Instant::now();
    {
        let mut out_file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&combined_path)?;
        let mut writer = BufWriter::with_capacity(WRITE_BUF_SIZE, out_file);
        for (_, file_path) in &files {
            let mut in_file = fs::File::open(file_path)?;
            std::io::copy(&mut in_file, &mut writer)?;
        }
        writer.flush()?;
    }
    temp_mgr.register(combined_path.clone());
    println!(
        "[timing] combine files: {:.3}s",
        t_combine.elapsed().as_secs_f64()
    );

    let combined_file = fs::File::open(&combined_path)?;
    let combined = unsafe { Mmap::map(&combined_file)? };

    // Detect available memory and decide dict-stage strategy
    let available_ram = get_available_memory();
    let dict_chunk_size = calculate_chunk_size(available_ram);
    let use_chunked_dict = combined.len() > dict_chunk_size;

    println!(
        "[memory] available: {:.1}GB, dict chunk: {:.0}MB, dict strategy: {}",
        available_ram as f64 / 1_000_000_000.0,
        dict_chunk_size as f64 / 1_000_000.0,
        if use_chunked_dict { "chunked" } else { "in-memory" }
    );

    let folder_name = base
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "folder".to_string());
    let output_path = base
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.join(format!("{}.duckomp", folder_name)))
        .unwrap_or_else(|| PathBuf::from(format!("{}.duckomp", folder_name)));
    let output_path_str = output_path.to_string_lossy().into_owned();

    // =================================================================
    // Step 1: Dictionary stages (chunked or in-memory based on RAM)
    // =================================================================
    let stage_entries: Vec<(u8, HashMap<u8, Vec<u8>>)>;

    // Step 2 results
    let chunk_flags: Vec<u8>;
    let chunk_sizes: Vec<u64>;
    let post_dict_output: Vec<u8>;
    let dict_output_len: usize;
    let actual_chunk_size: usize;
    let t_chunk: Instant;

    if use_chunked_dict {
        let mut dict_temp_mgr = TempFileManager::new();
        let (entries, dict_chunks) =
            run_chunked_stages(&combined, None, dict_chunk_size, &mut dict_temp_mgr)?;
        drop(combined);
        stage_entries = entries;

        // =============================================================
        // Step 2 (chunked): process each dict chunk as a whole piece
        // =============================================================
        t_chunk = Instant::now();
        actual_chunk_size = dict_chunk_size;
        let num_dict_chunks = dict_chunks.len();
        let mut total_dict_size: usize = 0;
        let mut all_flags: Vec<u8> = Vec::with_capacity(num_dict_chunks);
        let mut all_sizes: Vec<u64> = Vec::with_capacity(num_dict_chunks);
        let mut all_output: Vec<u8> = Vec::new();

        for (i, dict_chunk_path) in dict_chunks.iter().enumerate() {
            let t_chunk_i = Instant::now();
            let cf = fs::File::open(dict_chunk_path)?;
            let cm = unsafe { Mmap::map(&cf)? };
            let dict_data: &[u8] = &cm;
            let dict_size = dict_data.len();
            total_dict_size += dict_size;

            let orig_chunk_size = if i < num_dict_chunks - 1 {
                dict_chunk_size
            } else {
                total_size - (num_dict_chunks - 1) * dict_chunk_size
            };

            let dict_reduction_pct = if orig_chunk_size > 0 {
                (1.0 - dict_size as f64 / orig_chunk_size as f64) * 100.0
            } else {
                0.0
            };

            let entropy = shannon_entropy(dict_data);

            let run_egap;
            let mut run_lz77;

            if dict_reduction_pct >= ADAPTIVE_DICT_SKIP_ALL {
                run_egap = false;
                run_lz77 = false;
            } else if entropy >= ADAPTIVE_ENTROPY_RANDOM {
                run_egap = false;
                run_lz77 = false;
            } else if dict_reduction_pct >= ADAPTIVE_DICT_HIGH {
                run_egap = true;
                run_lz77 = entropy < ADAPTIVE_ENTROPY_HIGH;
            } else if dict_reduction_pct >= ADAPTIVE_DICT_MEDIUM {
                run_egap = true;
                run_lz77 = entropy < ADAPTIVE_ENTROPY_HIGH;
            } else {
                run_egap = false;
                run_lz77 = true;
            }

            let mut current_data: Vec<u8>;
            let mut flags: u8 = 0;

            if run_egap {
                let egap_data = entropy_encode(dict_data);
                if egap_data.len() < dict_size {
                    let egap_reduction_pct =
                        (1.0 - egap_data.len() as f64 / dict_size as f64) * 100.0;
                    flags |= CHUNK_FLAG_EGAP;
                    current_data = egap_data;

                    if run_lz77 {
                        if egap_reduction_pct < ADAPTIVE_EGAP_FLAT {
                            run_lz77 = false;
                        } else if egap_reduction_pct >= ADAPTIVE_EGAP_GOOD {
                            run_lz77 = true;
                        }
                    }
                } else {
                    current_data = dict_data.to_vec();
                }
            } else {
                current_data = dict_data.to_vec();
            }
            drop(cm);

            if run_lz77 {
                let pre_lz77_entropy = shannon_entropy(&current_data);
                if pre_lz77_entropy >= ADAPTIVE_ENTROPY_RANDOM {
                    run_lz77 = false;
                }
            }

            if run_lz77 {
                let lz77_data = lz77_compress(&current_data);
                if lz77_data.len() < current_data.len() {
                    flags |= CHUNK_FLAG_LZ77;
                    current_data = lz77_data;
                }
            }

            // Post-encoding entropy pass on final chunk
            {
                let egap_post = entropy_encode(&current_data);
                if egap_post.len() < current_data.len() {
                    flags |= CHUNK_FLAG_POST_ENTROPY;
                    current_data = egap_post;
                }
            }

            println!(
                "[adaptive] chunk {}/{}: dict={:.1}%, entropy={:.1} → {} ({:.1}MB in {:.3}s)",
                i + 1, num_dict_chunks, dict_reduction_pct, entropy,
                adaptive_stage_label(flags),
                current_data.len() as f64 / 1_000_000.0,
                t_chunk_i.elapsed().as_secs_f64()
            );

            all_flags.push(flags);
            all_sizes.push(current_data.len() as u64);
            all_output.extend_from_slice(&current_data);
            drop(current_data);
        }

        for path in &dict_chunks {
            fs::remove_file(path).ok();
        }

        dict_output_len = total_dict_size;
        chunk_flags = all_flags;
        chunk_sizes = all_sizes;
        post_dict_output = all_output;
    } else {
        let (entries, data) = run_compression_stages(&combined);
        drop(combined);
        stage_entries = entries;

        // =============================================================
        // Step 2 (in-memory): process full dict output as one piece
        // =============================================================
        t_chunk = Instant::now();
        actual_chunk_size = data.len();
        dict_output_len = data.len();

        let dict_reduction_pct = if total_size > 0 {
            (1.0 - data.len() as f64 / total_size as f64) * 100.0
        } else {
            0.0
        };
        let entropy = shannon_entropy(&data);

        let run_egap;
        let mut run_lz77;

        if dict_reduction_pct >= ADAPTIVE_DICT_SKIP_ALL {
            run_egap = false;
            run_lz77 = false;
        } else if entropy >= ADAPTIVE_ENTROPY_RANDOM {
            run_egap = false;
            run_lz77 = false;
        } else if dict_reduction_pct >= ADAPTIVE_DICT_HIGH {
            run_egap = true;
            run_lz77 = entropy < ADAPTIVE_ENTROPY_HIGH;
        } else if dict_reduction_pct >= ADAPTIVE_DICT_MEDIUM {
            run_egap = true;
            run_lz77 = entropy < ADAPTIVE_ENTROPY_HIGH;
        } else {
            run_egap = false;
            run_lz77 = true;
        }

        let mut current_data = data;
        let mut flags: u8 = 0;

        if run_egap {
            println!(
                "\n[egap] encoding {:.1}MB dictionary output...",
                current_data.len() as f64 / 1_000_000.0
            );
            let t_entropy = Instant::now();
            let pre_size = current_data.len();
            let egap_data = entropy_encode(&current_data);
            println!(
                "[timing] egap encode: {:.3}s",
                t_entropy.elapsed().as_secs_f64()
            );
            if egap_data.len() < pre_size {
                let egap_reduction_pct =
                    (1.0 - egap_data.len() as f64 / pre_size as f64) * 100.0;
                flags |= CHUNK_FLAG_EGAP;
                current_data = egap_data;

                if run_lz77 {
                    if egap_reduction_pct < ADAPTIVE_EGAP_FLAT {
                        run_lz77 = false;
                    } else if egap_reduction_pct >= ADAPTIVE_EGAP_GOOD {
                        run_lz77 = true;
                    }
                }
            }
        }

        if run_lz77 {
            let pre_lz77_entropy = shannon_entropy(&current_data);
            if pre_lz77_entropy >= ADAPTIVE_ENTROPY_RANDOM {
                run_lz77 = false;
            }
        }

        if run_lz77 {
            println!("\n[lz77] compressing {:.1}MB...", current_data.len() as f64 / 1_000_000.0);
            let t_lz77 = Instant::now();
            let lz77_data = lz77_compress(&current_data);
            println!(
                "[timing] lz77: {:.3}s ({:.1}MB -> {:.1}MB, {:.1}% reduction)",
                t_lz77.elapsed().as_secs_f64(),
                current_data.len() as f64 / 1_000_000.0,
                lz77_data.len() as f64 / 1_000_000.0,
                (1.0 - lz77_data.len() as f64 / current_data.len() as f64) * 100.0
            );
            if lz77_data.len() < current_data.len() {
                flags |= CHUNK_FLAG_LZ77;
                current_data = lz77_data;
            }
        }

        // Post-encoding entropy pass
        {
            let egap_post = entropy_encode(&current_data);
            if egap_post.len() < current_data.len() {
                flags |= CHUNK_FLAG_POST_ENTROPY;
                current_data = egap_post;
            }
        }

        println!(
            "[adaptive] stages: {} (dict={:.1}%, entropy={:.1})",
            adaptive_stage_label(flags), dict_reduction_pct, entropy
        );

        chunk_flags = vec![flags];
        chunk_sizes = vec![current_data.len() as u64];
        post_dict_output = current_data;
    }

    // =================================================================
    // Step 2 summary
    // =================================================================
    let num_chunks = chunk_flags.len();
    print_adaptive_summary(&chunk_flags);
    println!(
        "[timing] adaptive stages: {:.3}s, {:.1}MB → {:.1}MB ({} chunks)",
        t_chunk.elapsed().as_secs_f64(),
        dict_output_len as f64 / 1_000_000.0,
        post_dict_output.len() as f64 / 1_000_000.0,
        num_chunks
    );

    // =================================================================
    // Step 3: Build header and write output
    // =================================================================
    let t_write = Instant::now();
    let format_flags: u8 = FORMAT_FLAG_FOLDER;
    let payload = post_dict_output;

    let mut header: Vec<u8> = Vec::with_capacity(4096);
    header.extend_from_slice(b"DUCKOMP1");
    header.push(format_flags);
    header.extend_from_slice(&(total_size as u64).to_le_bytes());
    header.extend_from_slice(&(num_chunks as u32).to_le_bytes());
    header.extend_from_slice(&(actual_chunk_size as u32).to_le_bytes());

    // File directory
    if file_entries.len() > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Too many files",
        ));
    }
    header.extend_from_slice(&(file_entries.len() as u32).to_le_bytes());
    for entry in &file_entries {
        let path_bytes = entry.relative_path.as_bytes();
        if path_bytes.len() > u16::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "File path too long",
            ));
        }
        header.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
        header.extend_from_slice(path_bytes);
        header.extend_from_slice(&entry.file_size.to_le_bytes());
    }

    header.extend_from_slice(&build_stage_header(&stage_entries)?);

    // Per-chunk metadata: flag(1) + compressed_size(8)
    for i in 0..num_chunks {
        header.push(chunk_flags[i]);
        header.extend_from_slice(&chunk_sizes[i].to_le_bytes());
    }

    let total_file_size = header.len() + payload.len();

    {
        let out_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&output_path_str)?;
        out_file.set_len(total_file_size as u64)?;
        let mut out_mmap = unsafe { MmapMut::map_mut(&out_file)? };
        out_mmap[..header.len()].copy_from_slice(&header);
        out_mmap[header.len()..header.len() + payload.len()].copy_from_slice(&payload);
        out_mmap.flush()?;
    }
    drop(payload);

    println!(
        "[timing] write output: {:.3}s",
        t_write.elapsed().as_secs_f64()
    );

    let ratio = total_file_size as f64 / total_size as f64;
    let saved = total_size as i64 - total_file_size as i64;
    println!();
    println!("Compressed folder: {} -> {}", folder_path, output_path_str);
    println!(
        "Original:          {} bytes ({:.1} MB)",
        total_size,
        total_size as f64 / 1_000_000.0
    );
    println!(
        "Compressed:        {} bytes ({:.1} MB)",
        total_file_size,
        total_file_size as f64 / 1_000_000.0
    );
    println!("Saved:             {} bytes ({:.1}%)", saved, (1.0 - ratio) * 100.0);
    println!("Ratio:             {:.3}", ratio);
    println!("Wall time:         {:.3}s", t_total.elapsed().as_secs_f64());

    Ok(())
}

// ---------------------------------------------------------------------------
// Unified decompression
// ---------------------------------------------------------------------------
// Header: "DUCKOMP1"(8) + format_flags(1) + original_size(8)
//       + num_chunks(4) + chunk_size(4)
//       + [if is_folder: file directory]
//       + stage_header (dict tables)
//       + per-chunk metadata [flag(1) + compressed_size(8)]
//       + payload (global LZ77 compressed if FORMAT_FLAG_LZ77 set)

fn decompress_duckomp(duckomp_path: &str) -> io::Result<()> {
    let t_total = Instant::now();

    let t_mmap = Instant::now();
    let file = fs::File::open(duckomp_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data: &[u8] = &mmap;
    println!(
        "[timing] mmap input ({:.1}MB): {:.3}s",
        data.len() as f64 / 1_000_000.0,
        t_mmap.elapsed().as_secs_f64()
    );

    // Minimum header: magic(8) + flags(1) + orig_size(8) + num_chunks(4) + chunk_size(4) = 25
    if data.len() < 25 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "File too short",
        ));
    }
    if &data[..8] != b"DUCKOMP1" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a DUCKOMP file",
        ));
    }

    let format_flags = data[8];
    let is_folder = format_flags & FORMAT_FLAG_FOLDER != 0;
    let original_size = u64::from_le_bytes(data[9..17].try_into().unwrap()) as usize;
    let num_chunks = u32::from_le_bytes(data[17..21].try_into().unwrap()) as usize;
    let _chunk_size = u32::from_le_bytes(data[21..25].try_into().unwrap()) as usize;
    let mut offset = 25usize;

    println!(
        "[decompress] format_flags=0x{:02X} ({}), original={:.1}MB, {} chunks",
        format_flags,
        if is_folder { "folder" } else { "file" },
        original_size as f64 / 1_000_000.0,
        num_chunks
    );

    // --- File directory (folder only) ---
    let mut file_entries: Vec<(String, u64)> = Vec::new();
    if is_folder {
        if offset + 4 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Truncated file count",
            ));
        }
        let num_files =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        for _ in 0..num_files {
            if offset + 2 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Truncated file directory",
                ));
            }
            let path_len =
                u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize;
            offset += 2;
            if offset + path_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Truncated file path",
                ));
            }
            let path = String::from_utf8_lossy(&data[offset..offset + path_len]).into_owned();
            offset += path_len;
            if offset + 8 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Truncated file size",
                ));
            }
            let file_size = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            file_entries.push((path, file_size));
        }
        println!("[decompress] {} files in directory", file_entries.len());
    }

    // --- Stage entries (dictionary tables) ---
    if offset + 1 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated stage count",
        ));
    }
    let num_stages = data[offset] as usize;
    offset += 1;
    let mut stage_entries: Vec<(u8, HashMap<u8, Vec<u8>>)> = Vec::new();
    for _ in 0..num_stages {
        if offset + 2 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Truncated stage header",
            ));
        }
        let escape_byte = data[offset];
        let num_codes = data[offset + 1] as usize;
        offset += 2;
        let mut code_map: HashMap<u8, Vec<u8>> = HashMap::new();
        for _ in 0..num_codes {
            if offset + 3 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Truncated code entry",
                ));
            }
            let code = data[offset];
            let chunk_len =
                u16::from_le_bytes(data[offset + 1..offset + 3].try_into().unwrap()) as usize;
            offset += 3;
            if offset + chunk_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Truncated chunk data",
                ));
            }
            code_map.insert(code, data[offset..offset + chunk_len].to_vec());
            offset += chunk_len;
        }
        stage_entries.push((escape_byte, code_map));
    }
    println!("[decompress] {} dictionary stages loaded", num_stages);

    // --- Per-chunk metadata: flag(1) + compressed_size(8) ---
    if offset + num_chunks * 9 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated chunk metadata",
        ));
    }
    let mut chunk_flags: Vec<u8> = Vec::with_capacity(num_chunks);
    let mut chunk_sizes: Vec<u64> = Vec::with_capacity(num_chunks);
    for _ in 0..num_chunks {
        let flag = data[offset];
        offset += 1;
        let sz = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;
        chunk_flags.push(flag);
        chunk_sizes.push(sz);
    }

    // =====================================================================
    // Global LZ77 decompress (if applied)
    // =====================================================================
    let payload_data = &data[offset..];

    let post_lz77 = if format_flags & FORMAT_FLAG_LZ77 != 0 {
        let t_lz77 = Instant::now();
        let decompressed_payload = lz77_decompress_stream(payload_data)?;
        println!(
            "[lz77-global] decompress: {} -> {} in {:.3}s",
            payload_data.len(),
            decompressed_payload.len(),
            t_lz77.elapsed().as_secs_f64()
        );
        decompressed_payload
    } else {
        payload_data.to_vec()
    };

    // =====================================================================
    // Decode per-chunk EGAP/deentropy → reassemble dict output
    // =====================================================================
    let t_chunks = Instant::now();
    let mut dict_output: Vec<u8> = Vec::with_capacity(original_size);
    let mut chunk_offset: usize = 0;
    for i in 0..num_chunks {
        let sz = chunk_sizes[i] as usize;
        if chunk_offset + sz > post_lz77.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Chunk {} truncated: need {} bytes at offset {}, have {}",
                    i, sz, chunk_offset, post_lz77.len() - chunk_offset),
            ));
        }
        let chunk_data = &post_lz77[chunk_offset..chunk_offset + sz];
        let decoded = decode_adaptive_chunk(chunk_data, chunk_flags[i])?;
        dict_output.extend_from_slice(&decoded);
        chunk_offset += sz;
    }
    drop(post_lz77);
    println!(
        "[timing] decode {} chunks: {:.3}s",
        num_chunks,
        t_chunks.elapsed().as_secs_f64()
    );

    // =====================================================================
    // Reverse dictionary stages
    // =====================================================================
    println!(
        "[decompress] reversing {} dict stages on {:.1}MB...",
        stage_entries.len(),
        dict_output.len() as f64 / 1_000_000.0
    );
    let mut decompressed = dict_output;
    for (stage_idx, (escape_byte, code_map)) in stage_entries.iter().rev().enumerate() {
        let t_stage = Instant::now();
        let in_len = decompressed.len();
        decompressed = decode_dict_stage(&decompressed, *escape_byte, code_map)?;
        println!(
            "  stage {}: {:.1}MB → {:.1}MB in {:.3}s",
            stage_idx + 1,
            in_len as f64 / 1_000_000.0,
            decompressed.len() as f64 / 1_000_000.0,
            t_stage.elapsed().as_secs_f64()
        );
    }

    if decompressed.len() != original_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Size mismatch: {} != {}",
                decompressed.len(),
                original_size
            ),
        ));
    }

    // =====================================================================
    // Write output
    // =====================================================================
    if is_folder {
        let duckomp_file = Path::new(duckomp_path);
        let stem = duckomp_file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "extracted".to_string());
        let out_dir = duckomp_file
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.join(&stem))
            .unwrap_or_else(|| PathBuf::from(&stem));

        let mut cursor: usize = 0;
        for (rel_path, file_size) in &file_entries {
            let size = *file_size as usize;
            if cursor + size > decompressed.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("File data truncated for '{}'", rel_path),
                ));
            }
            let file_data = &decompressed[cursor..cursor + size];
            cursor += size;

            let dest: PathBuf = rel_path
                .replace('\\', "/")
                .split('/')
                .filter(|c| !c.is_empty() && *c != "." && *c != "..")
                .fold(out_dir.clone(), |acc, c| acc.join(c));

            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&dest, file_data)?;
            println!("  extracted: {}", rel_path);
        }

        println!(
            "\nExtracted {} file(s) to {}",
            file_entries.len(),
            out_dir.display()
        );
        println!("Wall time: {:.3}s", t_total.elapsed().as_secs_f64());
    } else {
        let output_path = if duckomp_path.to_lowercase().ends_with(".duckomp") {
            duckomp_path[..duckomp_path.len() - ".duckomp".len()].to_string()
        } else {
            format!("{}.decompressed", duckomp_path)
        };

        let t_write = Instant::now();
        let out_file = fs::File::create(&output_path)?;
        let mut writer = BufWriter::with_capacity(WRITE_BUF_SIZE, out_file);
        writer.write_all(&decompressed)?;
        writer.flush()?;
        println!(
            "[timing] write output ({:.1}MB): {:.3}s",
            decompressed.len() as f64 / 1_000_000.0,
            t_write.elapsed().as_secs_f64()
        );
        println!("Decompressed: {} -> {}", duckomp_path, output_path);
        println!("Wall time:    {:.3}s", t_total.elapsed().as_secs_f64());
    }

    Ok(())
}

/// Decode a single adaptive chunk: reverse LZ77, EGAP, and deentropy as indicated by flags
fn decode_adaptive_chunk(data: &[u8], flags: u8) -> io::Result<Vec<u8>> {
    // Undo in reverse order: post-entropy → LZ77 → deentropy-meta → EGAP → inv-deentropy

    let post_ent: Vec<u8>;
    let after_post_ent: &[u8] = if flags & CHUNK_FLAG_POST_ENTROPY != 0 {
        post_ent = entropy_decode(data)?;
        &post_ent
    } else {
        data
    };

    let lz77_decoded: Vec<u8>;
    let after_lz77: &[u8] = if flags & CHUNK_FLAG_LZ77 != 0 {
        lz77_decoded = lz77_decompress(after_post_ent)?;
        &lz77_decoded
    } else {
        after_post_ent
    };

    let mut current: &[u8] = after_lz77;

    // Strip deentropy meta if present
    let de_meta = if flags & CHUNK_FLAG_DEENTROPY != 0 {
        let (meta, consumed) = DeentropyMeta::deserialize(current)?;
        current = &current[consumed..];
        Some(meta)
    } else {
        None
    };

    // 3. EGAP decode if present
    let mut owned = if flags & CHUNK_FLAG_EGAP != 0 {
        entropy_decode(current)?
    } else {
        current.to_vec()
    };

    // 4. Inverse deentropy if applied
    if let Some(meta) = de_meta {
        owned = inverse_deentropy_transform(&owned, &meta);
    }

    Ok(owned)
}

/// Reverse a single dictionary stage
fn decode_dict_stage(
    data: &[u8],
    escape_byte: u8,
    code_map: &HashMap<u8, Vec<u8>>,
) -> io::Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(data.len() * 2);
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        i += 1;
        if b == escape_byte {
            if i >= data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Unexpected end after escape byte",
                ));
            }
            let code = data[i];
            i += 1;
            if code == 0 {
                out.push(escape_byte);
            } else {
                match code_map.get(&code) {
                    Some(chunk) => out.extend_from_slice(chunk),
                    None => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Unknown code {}", code),
                        ))
                    }
                }
            }
        } else {
            out.push(b);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() {
    println!("Created by SavageDuck26\n");

    print!("Enter the file path: ");
    io::stdout().flush().unwrap();

    let stdin = io::stdin();
    let mut file_path = String::new();
    stdin.lock().read_line(&mut file_path).unwrap();
    let file_path = file_path.trim().trim_matches('"');

    if !Path::new(file_path).exists() {
        eprintln!("File not found: {}", file_path);
        return;
    }

    print!("Compress or Decompress '{}'? (c/d): ", file_path);
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    stdin.lock().read_line(&mut choice).unwrap();
    let choice = choice.trim().to_lowercase();

    match choice.as_str() {
        "c" => {
            let result = if Path::new(file_path).is_dir() {
                compress_folder(file_path)
            } else {
                compress_file(file_path)
            };
            if let Err(e) = result {
                eprintln!("Compression error: {}", e);
            }
        }
        "d" => {
            if let Err(e) = decompress_duckomp(file_path) {
                eprintln!("Decompression error: {}", e);
            }
        }
        _ => eprintln!("Invalid choice. Please enter 'c' or 'd'."),
    }
}

fn main() {
    // Catch panics so the window stays open even if something crashes
    let result = std::panic::catch_unwind(run);

    if let Err(e) = result {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown error".to_string()
        };
        eprintln!("\nFatal error: {}", msg);
    }

    println!();
    print!("Press Enter to exit...");
    io::stdout().flush().unwrap();
    let _ = io::stdin().lock().read_line(&mut String::new());
}
