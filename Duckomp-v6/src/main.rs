use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};

struct Offsets {
    low: u128,
    high: u128,
    count: u128,
}

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

fn calculate_offsets(data: Vec<u128>) -> Offsets {
    let mut low = u128::MAX;
    let mut high = u128::MIN;

    for window in data.windows(2) {
        let diff = window[0].abs_diff(window[1]);
        if diff < low { low = diff; }
        if diff > high { high = diff; }
    }

    Offsets {
        low,
        high,
        count: data.len() as u128,
    }
}

fn build_pond(offsets: Offsets) -> io::Result<()> {
    let low = offsets.low;
    let high = offsets.high;
    let count = offsets.count;

    if count == 0 {
        eprintln!("count is 0 — nothing to generate");
        return Ok(());
    }

    let base = high - low + 1;

    let total_seeds = match base.checked_pow(count as u32) {
        Some(v) => v,
        None => {
            eprintln!("total_seeds overflowed u128 — count/base too large to enumerate");
            return Ok(());
        }
    };

    fs::create_dir_all("output")?;

    let mut times_added: u128 = 0;
    let mut current_first_digit: Option<u128> = None;
    let mut writer: Option<BufWriter<File>> = None;

    for seed in 0..total_seeds {
        let mut combo = vec![0u128; count as usize];
        let mut n = seed;

        for i in (0..count as usize).rev() {
            combo[i] = (n % base) + low;
            n /= base;
        }

        let first_digit = combo[0];

        // First digit changed (or this is the very first seed) — swap to a new file.
        if current_first_digit != Some(first_digit) {
            if let Some(mut w) = writer.take() {
                w.flush()?;
            }

            let path = format!("output/offsets_{}.txt", first_digit);
            let file = File::create(&path)?;
            writer = Some(BufWriter::new(file));
            current_first_digit = Some(first_digit);
        }

        if let Some(w) = writer.as_mut() {
            let line = combo
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(w, "{}: [{}]", seed, line)?;
        }

        times_added += 1;
    }

    if let Some(mut w) = writer.take() {
        w.flush()?;
    }

    println!("Times added: {}", times_added);

    Ok(())
}

fn main() -> io::Result<()> {
    let data = read_data("sample.txt")?;
    let offsets = calculate_offsets(data);
    build_pond(offsets)?;

    Ok(())
}