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

    let mut out = String::new();
    out.push_str(&format!("Offset Low: {low}\n"));
    out.push_str(&format!("Offset High: {high}\n"));
    out.push_str(&format!("Count: {}\n", data.len()));
    out.push_str("Seed:\n");
    out.push_str(&seed.to_string());
    out.push_str("\nSolver Equation:\n");
    out.push_str(&result.equation);
    out.push_str("\n");
    if !result.magic.is_zero() {
        if result.magic_sign < 0 {
            out.push_str(&format!("Magic: -{}\n", result.magic));
        } else {
            out.push_str(&format!("Magic: {}\n", result.magic));
        }
    }
    out.push_str(&format!("Verify: {}\n", result.ok));

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
