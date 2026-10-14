mod dp;
mod packed;
mod solver;

use num_bigint::BigUint;
use num_traits::Zero;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};

// ---------------------------------------------------------------------------
// Tunables — edit these directly and rebuild.
// ---------------------------------------------------------------------------
/// Greedy peels before the leftover becomes the magic number.
/// The coefficient budget auto-sizes so the seed folds into this many terms
/// with a magic that fits a u32.
const MAX_TERMS: usize = 32;
/// Magic must not exceed this many bits (u32).
const MAGIC_BITS: u64 = 32;
/// Highest base tried in the efficiency sweep.
const MAX_BASE: u64 = 65536;
/// Parallel worker threads (rayon pool size).
const THREADS: usize = 6;
/// Print the sweep + per-step diagnostics to stderr.
const VERBOSE: bool = false;

fn read_data(path: &str) -> io::Result<Vec<u128>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if let Ok(n) = line.trim().parse::<u128>() {
            data.push(n);
        }
    }

    Ok(data)
}

/// Render the PoC output exactly as it reads in output/PCS.txt: the plain
/// offset data beside what it costs when packed.
fn build_output(
    data: &[u128],
    diffs: &[u128],
    low: u128,
    high: u128,
    base: u128,
    seed: &BigUint,
    result: &solver::SolverResult,
) -> String {
    let offset_text = data.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|");
    let diff_text = diffs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|");

    let seed_dec = seed.to_string();
    let seed_bytes = packed::packed_len(seed);
    let seed_hex = packed::packed_hex(seed);

    let entropy = packed::entropy_bits(diffs);
    let huffman = packed::huffman_bits(diffs);
    let fixed_payload = diffs.len() as f64 * (base as f64).log2();
    let ratio = seed_dec.len() as f64 / seed_bytes as f64;
    let magic_sign = if result.magic_sign < 0 && !result.magic.is_zero() { "-" } else { "" };

    // Total encoded size of each representation, overheads included, so the
    // "smallest" comparison is apples-to-apples:
    //   fixed-radix = payload + 16-bit header (base + length)
    //   Huffman     = payload + codebook + 16-bit header
    //   DP chunks   = per-chunk payload + 16 bits per chunk
    let hdr = packed::PACK_HEADER_BITS as f64;
    let fixed_total = fixed_payload + hdr;
    let huffman_total = huffman as f64 + packed::huffman_codebook_bits(diffs) as f64 + hdr;
    let plan = dp::plan(diffs);
    let dp_total = plan.total_bits();

    // Smallest wins; ties keep the earlier line.
    let mut best = fixed_total;
    let mut winner = 0u8;
    if huffman_total < best {
        best = huffman_total;
        winner = 1;
    }
    if dp_total < best {
        winner = 2;
    }
    let fixed_mark = if winner == 0 { " [WINNER]" } else { "" };
    let huffman_mark = if winner == 1 { " [WINNER]" } else { "" };
    let dp_mark = if winner == 2 { " [WINNER]" } else { "" };
    let chunk_word = if plan.chunks.len() == 1 { "chunk" } else { "chunks" };
    let fixed_bytes = (fixed_total / 8.0).ceil() as u64;
    let huffman_bytes = (huffman_total / 8.0).ceil() as u64;
    let dp_bytes = (dp_total / 8.0).ceil() as u64;

    let mut out = String::new();
    out.push_str(&format!("Offsets ({}):\n{offset_text}\n\n", data.len()));
    out.push_str(&format!("Offset gaps ({}):\n{diff_text}\n\n", diffs.len()));
    out.push_str(&format!("Offset Low: {low}\n"));
    out.push_str(&format!("Offset High: {high}\n\n"));
    out.push_str(&format!(
        "Pack radix: {base} (Largest gap - smallest gap + 1)\n\n"
    ));
    out.push_str(&format!(
        "Gap entropy: {entropy:.2} bits per gap (Fewest bits each gap takes, on avg)\n"
    ));
    out.push_str(&format!(
        "Fixed-radix pack: {fixed_total:.0} bits, {fixed_bytes} bytes (payload + 16-bit header){fixed_mark}\n"
    ));
    out.push_str(&format!(
        "Huffman pack: {huffman_total:.0} bits, {huffman_bytes} bytes (payload + codebook + header){huffman_mark}\n"
    ));
    out.push_str(&format!(
        "DP chunks: {} {chunk_word}, {dp_total:.0} bits, {dp_bytes} bytes (payload + 16 bits per chunk){dp_mark}\n\n",
        plan.chunks.len()
    ));
    out.push_str(&format!("Seed:\n{seed_dec}\n\n"));
    out.push_str(&format!(
        "Seed as bytes: {seed_bytes} bytes, {ratio:.2}x smaller than {} bytes of text (Stored as raw bytes)\n",
        seed_dec.len()
    ));
    out.push_str(&format!("Seed as hex: {seed_hex}\n\n"));
    out.push_str(&format!("Solver Equation:\n{}\n", result.equation));
    out.push_str(&format!("Magic: {magic_sign}{}\n", result.magic));
    out.push_str(&format!("Verify: {}\n", result.ok));
    out
}

fn main() -> io::Result<()> {
    let data = read_data("sample.txt")?;
    if data.len() < 2 {
        println!("Need at least 2 values in sample.txt.");
        return Ok(());
    }

    // Adjacent offsets -> diffs, plus the low/high bounds of the diff range.
    let mut low = u128::MAX;
    let mut high = u128::MIN;
    let mut diffs = Vec::with_capacity(data.len() - 1);
    for w in data.windows(2) {
        let diff = w[0].abs_diff(w[1]);
        diffs.push(diff);
        if diff < low {
            low = diff;
        }
        if diff > high {
            high = diff;
        }
    }

    // The seed: diffs packed as a fixed-length base-(high-low+1) number.
    let base = high - low + 1;
    let mut seed = BigUint::zero();
    for &d in &diffs {
        seed *= BigUint::from(base);
        seed += BigUint::from(d - low);
    }

    // Build the rayon pool with the configured worker count, then solve.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS.max(1))
        .build_global();

    let result = solver::solve(&seed, MAX_TERMS, MAX_BASE, VERBOSE);

    // One PoC file: plain data beside its packed form.
    let out = build_output(&data, &diffs, low, high, base, &seed, &result);

    print!("{out}");
    fs::create_dir_all("output")?;
    fs::write("output/PCS.txt", &out)?;
    eprintln!(
        "(written to output/PCS.txt; {} terms, magic {} bits)",
        result.terms,
        result.magic.bits()
    );
    if result.magic.bits() > MAGIC_BITS {
        eprintln!(
            "WARNING: magic exceeds u32 ({} bits > {} bits)",
            result.magic.bits(),
            MAGIC_BITS
        );
    }

    Ok(())
}
