use std::io::{self, Write, Read, BufReader};
use std::path::Path;
use std::collections::HashMap;
use std::fs::File;

use aho_corasick::AhoCorasick;

// === Tuning Constants ===
const MIN_SUBSTR_LEN: usize = 14;
const MAX_SUBSTR_LEN: usize = 16;
const MIN_SUBSTR_COUNT: usize = 24;
const COEFF_SUBSTR_MIN_LEN: usize = 2;
const COEFF_SUBSTR_MAX_LEN: usize = 6;
const SAMPLING_RATE: usize = 100;
const TOP_CANDIDATES: usize = 50;
const CHUNK_SIZE: usize = 64 * 1024 * 1024; // 64MB — keeps RSS low on multi-GB files

/// Hex-encode a byte slice for display.
fn hex_fmt(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "<empty>".to_string();
    }
    // Try to show as ASCII if printable, otherwise hex
    let printable = bytes.iter().all(|&b| b.is_ascii_graphic() || b == b' ');
    if printable {
        String::from_utf8_lossy(bytes).into_owned()
    } else {
        let max_show = bytes.len().min(64);
        let mut s = String::with_capacity(2 + max_show * 2);
        s.push_str("0x");
        for &b in &bytes[..max_show] {
            s.push_str(&format!("{:02x}", b));
        }
        if bytes.len() > max_show {
            s.push_str("..");
        }
        s
    }
}

/// Stream a file, collecting offsets for a set of patterns (Aho-Corasick) in a single pass.
/// Patterns are raw byte vectors — works on any binary data.
/// Pass `store_for` to store full offset lists (expensive), pass `count_only` to only count.
/// Returns (name_to_offsets, name_to_count).
fn stream_find_offsets(
    path: &Path,
    patterns: &[Vec<u8>],
    store_indices: &[bool],
) -> io::Result<(HashMap<usize, Vec<usize>>, HashMap<usize, usize>)> {
    // AhoCorasick::new accepts patterns as &[impl AsRef<[u8]>]; Vec<u8> implements that.
    let ac = AhoCorasick::new(patterns).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(CHUNK_SIZE, file);
    let mut buf = vec![0u8; CHUNK_SIZE + MAX_SUBSTR_LEN];
    let mut file_pos: usize = 0;
    let mut carry_over = 0usize;

    let mut offsets: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for (i, &store) in store_indices.iter().enumerate() {
        if store { offsets.insert(i, Vec::new()); }
        counts.insert(i, 0usize);
    }

    loop {
        // Read into carry-over area first, then rest
        let target = &mut buf[carry_over..];
        let n = reader.read(target)?;
        if n == 0 && carry_over == 0 { break; }
        let filled = carry_over + n;
        let data = &buf[..filled];
        if n == 0 { carry_over = 0; break; }

        // Scan for patterns in this chunk
        for m in ac.find_iter(data) {
            let abs_pos = file_pos + m.start();
            let pat_idx = m.pattern().as_usize();
            *counts.get_mut(&pat_idx).unwrap() += 1;
            if store_indices[pat_idx] {
                offsets.get_mut(&pat_idx).unwrap().push(abs_pos);
            }
        }

        // Advance file position, keep overlap for cross-boundary matches
        let advance = if n < CHUNK_SIZE { filled } else { filled - MAX_SUBSTR_LEN };
        file_pos += advance;
        // Copy overlap bytes to front
        let overlap = filled - advance;
        buf.copy_within(advance..filled, 0);
        carry_over = overlap;
    }

    Ok((offsets, counts))
}

/// Streaming sample: read file in chunks, sample at SAMPLING_RATE intervals, build candidate counts.
/// Uses raw byte vectors as keys — works on any binary data.
fn stream_sample(path: &Path) -> io::Result<HashMap<Vec<u8>, usize>> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(CHUNK_SIZE, file);
    let mut buf = vec![0u8; CHUNK_SIZE + MAX_SUBSTR_LEN];
    let mut file_pos: usize = 0;
    let mut carry_over = 0usize;
    let mut sample_counts: HashMap<Vec<u8>, usize> = HashMap::new();

    loop {
        let target = &mut buf[carry_over..];
        let n = reader.read(target)?;
        if n == 0 && carry_over == 0 { break; }
        let filled = carry_over + n;
        let data = &buf[..filled];
        if n == 0 { break; }

        let chunk_start = file_pos;
        for i in (0..data.len().saturating_sub(MIN_SUBSTR_LEN)).step_by(SAMPLING_RATE) {
            let abs_i = chunk_start + i;
            if abs_i % SAMPLING_RATE != 0 { continue; }
            let end = (i + MAX_SUBSTR_LEN).min(data.len());
            for len in MIN_SUBSTR_LEN..=end - i {
                let key = data[i..i + len].to_vec();
                *sample_counts.entry(key).or_insert(0) += 1;
            }
        }

        let advance = if n < CHUNK_SIZE { filled } else { filled - MAX_SUBSTR_LEN };
        file_pos += advance;
        let overlap = filled - advance;
        buf.copy_within(advance..filled, 0);
        carry_over = overlap;
    }

    Ok(sample_counts)
}

/// Find the best repeated substring and its offsets using streaming.
/// Works entirely on raw bytes — no UTF-8 assumptions.
fn find_common_substrings_streaming(
    path: &Path,
    min_len: usize,
    max_len: usize,
) -> io::Result<(Vec<(Vec<u8>, usize, usize, usize)>, Vec<usize>)> {
    // Step 1: Sample the file to find candidate substrings (raw bytes)
    let sample_counts = stream_sample(path)?;

    // Step 2: Score and rank candidates
    let mut scored: Vec<(Vec<u8>, usize, usize)> = sample_counts.into_iter()
        .map(|(s, c)| {
            let l = s.len();
            (s, c, c * l)
        })
        .collect();
    scored.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.0.len().cmp(&a.0.len())));

    let candidates: Vec<Vec<u8>> = scored.into_iter()
        .take(TOP_CANDIDATES)
        .map(|(s, _, _)| s)
        .collect();
    if candidates.is_empty() { return Ok((Vec::new(), Vec::new())); }

    // Step 3: Count each candidate in a single streaming pass via Aho-Corasick
    let store_all: Vec<bool> = candidates.iter().map(|_| false).collect();
    // Pass raw byte patterns directly — no String conversion needed
    let (_, candidate_counts) = stream_find_offsets(path, &candidates, &store_all)?;

    // Keep only candidates with >= MIN_SUBSTR_COUNT occurrences
    let valid: Vec<(Vec<u8>, usize)> = candidates.into_iter().enumerate()
        .filter(|(i, _)| *candidate_counts.get(i).unwrap_or(&0) >= MIN_SUBSTR_COUNT)
        .map(|(i, s)| (s, *candidate_counts.get(&i).unwrap()))
        .collect();
    if valid.is_empty() { return Ok((Vec::new(), Vec::new())); }

    // Find the best candidate (highest score = count * len)
    let best_idx = valid.iter().enumerate()
        .max_by_key(|(_, (s, c))| c * s.len())
        .map(|(i, _)| i)
        .unwrap();

    // Step 4: Try to extend the best candidate using raw bytes
    let mut best_sub = valid[best_idx].0.clone();

    // Extend bidirectionally until no common byte is found (or 128-char max)
    // Use a single mmap for the whole extension phase
    let file = File::open(path)?;
    let file_len = file.metadata()?.len() as usize;
    let mmap_check = unsafe { memmap2::Mmap::map(&file)? };
    let data = &mmap_check[..file_len];

    for _ext_attempt in 0..128 {
        // Collect offsets for current best_sub
        let ext_patterns = vec![best_sub.clone()];
        let store_ext = vec![true];
        let (mut ext_offsets_map, _) = stream_find_offsets(path, &ext_patterns, &store_ext)?;
        let mut best_offsets = ext_offsets_map.remove(&0).unwrap_or_default();
        let best_count = best_offsets.len();
        if best_count <= MIN_SUBSTR_COUNT { break; }

        // Right extension check
        let mut ext_byte: Option<u8> = None;
        let mut can_extend = true;
        for &off in &best_offsets {
            let pos = off + best_sub.len();
            if pos >= data.len() { can_extend = false; break; }
            let b = data[pos];
            match ext_byte {
                None => ext_byte = Some(b),
                Some(c) if c != b => { can_extend = false; break; }
                _ => {}
            }
        }
        if can_extend && ext_byte.is_some() {
            best_sub.push(ext_byte.unwrap());
            continue;
        }

        // Left extension check
        ext_byte = None;
        can_extend = true;
        for &off in &best_offsets {
            if off == 0 { can_extend = false; break; }
            let b = data[off - 1];
            match ext_byte {
                None => ext_byte = Some(b),
                Some(c) if c != b => { can_extend = false; break; }
                _ => {}
            }
        }
        if can_extend && ext_byte.is_some() {
            best_sub.insert(0, ext_byte.unwrap());
            continue;
        }

        drop(mmap_check);
        // Can't extend further — collect final offsets
        let store_final = vec![true];
        let (mut final_map, _) = stream_find_offsets(path, &[best_sub.clone()], &store_final)?;
        best_offsets = final_map.remove(&0).unwrap_or_default();

        // Step 5: Count prefixes/suffixes for display (single streaming pass)
        let tl = best_sub.len();
        let mut display_results: Vec<(Vec<u8>, usize, usize, usize)> = Vec::new();
        let best_count = best_offsets.len();
        let best_score = best_count * tl;
        display_results.push((best_sub.clone(), best_count, tl, best_score));

        // Build patterns for all prefixes/suffixes (works on raw bytes)
        let mut sub_patterns: Vec<Vec<u8>> = Vec::new();
        let mut sub_map: Vec<(usize, Vec<u8>, bool)> = Vec::new(); // (idx, bytes, is_prefix)
        let mut seen = std::collections::HashSet::new();
        seen.insert(best_sub.clone());

        for sub_len in min_len..tl {
            let prefix = best_sub[..sub_len].to_vec();
            if !seen.contains(&prefix) {
                sub_map.push((sub_patterns.len(), prefix.clone(), true));
                sub_patterns.push(prefix);
            }

            let suffix = best_sub[tl - sub_len..].to_vec();
            if !seen.contains(&suffix) {
                sub_map.push((sub_patterns.len(), suffix.clone(), false));
                sub_patterns.push(suffix.clone());
                seen.insert(suffix);
            }
        }

        // Dedup sub_patterns and rebuild sub_map
        // (We already dedup via 'seen', but need to ensure sub_map indices match sub_patterns)
        // Actually, the original code only inserted unique prefixes/suffixes, but shared patterns
        // between prefix and suffix could cause an off-by-one. Let's just rebuild clean.
        let mut clean_patterns: Vec<Vec<u8>> = Vec::new();
        let mut clean_map: Vec<(usize, Vec<u8>, bool)> = Vec::new();
        let mut clean_seen = std::collections::HashSet::new();
        for sub_len in min_len..tl {
            let prefix = best_sub[..sub_len].to_vec();
            if !clean_seen.contains(&prefix) {
                clean_map.push((clean_patterns.len(), prefix.clone(), true));
                clean_patterns.push(prefix.clone());
                clean_seen.insert(prefix);
            }

            let suffix = best_sub[tl - sub_len..].to_vec();
            if !clean_seen.contains(&suffix) {
                clean_map.push((clean_patterns.len(), suffix.clone(), false));
                clean_patterns.push(suffix.clone());
                clean_seen.insert(suffix);
            }
        }

        if !clean_patterns.is_empty() {
            let store_subs: Vec<bool> = vec![false; clean_patterns.len()];
            let (_, sub_counts) = stream_find_offsets(path, &clean_patterns, &store_subs)?;

            for &(pat_idx, ref sub_bytes, _is_prefix) in &clean_map {
                let count = *sub_counts.get(&pat_idx).unwrap_or(&0);
                if count > MIN_SUBSTR_COUNT {
                    let sub_score = count * sub_bytes.len();
                    display_results.push((sub_bytes.clone(), count, sub_bytes.len(), sub_score));
                }
            }
        }

        // Sort and deduplicate display results
        display_results.sort_by(|a, b| {
            b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0))
        });
        let mut dedup: Vec<(Vec<u8>, usize, usize, usize)> = Vec::new();
        for r in display_results {
            if !dedup.iter().any(|d| d.0 == r.0) {
                dedup.push(r);
            }
        }

        return Ok((dedup, best_offsets));
    }

    // Fallback — use best_offsets from the extension loop
    let fallback_offsets = {
        let store_final = vec![true];
        let (mut final_map, _) = stream_find_offsets(path, &[best_sub.clone()], &store_final)?;
        final_map.remove(&0).unwrap_or_default()
    };
    let best_count = fallback_offsets.len();
    let best_score = best_count * best_sub.len();
    let display = vec![(best_sub.clone(), best_count, best_sub.len(), best_score)];
    Ok((display, fallback_offsets))
}

/// Format a coefficient value into a string.
fn format_coeff(v: f64) -> String {
    let raw = if v >= 10000.0 { format!("{:.4}", v) }
        else if v >= 1.0 { format!("{:.6}", v) }
        else if v >= 0.001 { format!("{:.8}", v) }
        else { format!("{:.4e}", v) };
    if raw.contains('.') {
        let trimmed = raw.trim_end_matches('0');
        trimmed.trim_end_matches('.').to_string()
    } else { raw }
}

/// Build frequency maps from all segments' coefficients in a single pass.
fn build_coeff_freqs(valid_segs: &[&Seg]) -> (HashMap<String, usize>, HashMap<String, usize>) {
    let mut dec_counts: HashMap<String, usize> = HashMap::new();
    let mut whole_counts: HashMap<String, usize> = HashMap::new();

    for seg in valid_segs {
        let deg = seg.coeffs.len() - 3;
        for d in (0..=deg).rev() {
            let c = seg.coeffs[2 + d];
            let v = c.abs();
            if v < 1e-12 && d > 0 { continue; }
            let vs = format_coeff(v);

            if let Some(pos) = vs.find('.') {
                let frac = &vs[pos + 1..];
                if frac.len() >= 2 {
                    *dec_counts.entry(frac.to_string()).or_insert(0) += 1;
                }
            }

            let full = if let Some(pos) = vs.find('.') { &vs[..pos] } else { vs.as_str() };
            for start in 0..full.len() {
                for len in COEFF_SUBSTR_MIN_LEN..=(full.len() - start).min(COEFF_SUBSTR_MAX_LEN) {
                    let sub = &full[start..start + len];
                    *dec_counts.entry(sub.to_string()).or_insert(0) += 1;
                }
            }

            *whole_counts.entry(vs).or_insert(0) += 1;
        }
    }

    (dec_counts, whole_counts)
}

/// Compress a number string using shorthand variables.
fn apply_dec_shorthand(num: &str, suffix_map: &[(String, String)]) -> String {
    let mut result = num.to_string();
    let mut changed = true;
    while changed {
        changed = false;
        for (suffix, var) in suffix_map.iter() {
            let dot_pattern = format!(".{}", suffix);
            if result.find(&dot_pattern).is_some() {
                if let Some(dot_pos) = result.find('.') {
                    let after_dot = &result[dot_pos..];
                    if after_dot.ends_with(&dot_pattern) {
                        let before = &result[..dot_pos];
                        let keep = &after_dot[1..after_dot.len() - suffix.len()];
                        result = if keep.is_empty() {
                            format!("{}.{}", before, var)
                        } else {
                            format!("{}.{}{}", before, keep, var)
                        };
                        changed = true;
                        break;
                    }
                }
            }
            if let Some(idx) = result.find(suffix) {
                if !result.contains('.') || idx < result.find('.').unwrap() {
                    let prev_ok = idx == 0 || result.as_bytes()[idx-1].is_ascii_digit() || result.as_bytes()[idx-1].is_ascii_lowercase();
                    let next_ok = idx + suffix.len() >= result.len()
                        || result.as_bytes()[idx + suffix.len()].is_ascii_digit()
                        || result.as_bytes()[idx + suffix.len()] == b'.';
                    if prev_ok && next_ok {
                        let mut new_str = String::with_capacity(result.len());
                        new_str.push_str(&result[..idx]);
                        new_str.push_str(var);
                        new_str.push_str(&result[idx + suffix.len()..]);
                        result = new_str;
                        changed = true;
                        break;
                    }
                }
            }
        }
    }
    result
}

// === Polynomial math (unchanged) ===

fn fit_poly(ys: &[f64], x_start: f64, degree: usize) -> Vec<f64> {
    let n = ys.len();
    let m = degree + 1;
    if n < 2 || n < m {
        let mut c = vec![0.0; m]; if n > 0 { c[0] = ys[0]; }
        let mut r = vec![x_start, 1.0]; r.extend(c); return r;
    }

    let x_mean = x_start + (n - 1) as f64 / 2.0;
    let x_scale = (n - 1) as f64 / 2.0;
    if x_scale < 1e-15 {
        let mut c = vec![0.0; m]; c[0] = ys[0];
        let mut r = vec![x_start, 1.0]; r.extend(c); return r;
    }

    let mut s = vec![vec![0.0; m]; m];
    let mut t = vec![0.0; m];

    for i in 0..n {
        let u = (x_start + i as f64 - x_mean) / x_scale;
        let y = ys[i];
        let mut upow = vec![1.0; 2 * m];
        for p in 1..2 * m { upow[p] = upow[p - 1] * u; }

        for j in 0..m {
            t[j] += y * upow[j];
            for k in 0..m {
                s[j][k] += upow[j + k];
            }
        }
    }

    let mut aug = vec![vec![0.0; m + 1]; m];
    for i in 0..m { for j in 0..m { aug[i][j] = s[i][j]; } aug[i][m] = t[i]; }

    for col in 0..m {
        let mut mr = col; let mut mv = aug[col][col].abs();
        for r in (col + 1)..m { let v = aug[r][col].abs(); if v > mv { mv = v; mr = r; } }
        if mv < 1e-15 { continue; }
        aug.swap(col, mr); let pv = aug[col][col];
        for r in (col + 1)..m { let f = aug[r][col] / pv;
            for k in col..=m { aug[r][k] -= f * aug[col][k]; } }
    }

    let mut c = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = aug[i][m];
        for j in (i + 1)..m { sum -= aug[i][j] * c[j]; }
        if aug[i][i].abs() > 1e-15 { c[i] = sum / aug[i][i]; }
    }

    let mut result = vec![x_mean, x_scale];
    result.extend(c);
    result
}

fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    let u = (x - coeffs[0]) / coeffs[1];
    let mut r = 0.0;
    for &c in coeffs[2..].iter().rev() { r = r * u + c; }
    r
}

fn eval_rounded(coeffs: &[f64], x: f64) -> i64 { (eval_poly(coeffs, x) + 0.5).floor() as i64 }

struct Seg { sx: usize, ex: usize, coeffs: Vec<f64> }

fn segment_data(xs: &[f64], ys: &[f64]) -> Vec<Seg> {
    let n = xs.len(); let mut segs = Vec::new();
    if n < 3 { return segs; }
    let mut pos = 0;
    while pos < n - 2 {
        let mut be = pos + 1; let mut bc = Vec::new(); let mut bh = 0; let mut bt = 0;
        for w in 4..=15 {
            let e = (pos + w).min(n); if e - pos < 3 { continue; }
            let sl = e - pos; let d = 3.min(sl - 1).max(1);
            let slice = &ys[pos..e];
            let coeffs = fit_poly(slice, xs[pos], d);
            if coeffs.len() < 3 { continue; }
            let mut h = 0;
            for i in pos..e { if eval_rounded(&coeffs, xs[i]) == ys[i] as i64 { h += 1; } }
            let better = if h == sl { true } else if bh == bt { false } else { sl > bt || (sl == bt && h > bh) };
            if better { be = e; bc = coeffs; bh = h; bt = sl; }
        }
        if bh == 0 {
            let e = (pos + 4).min(n);
            let slice = &ys[pos..e];
            bc = fit_poly(slice, xs[pos], 3.min(e - pos - 1).max(1)); be = e;
        }
        segs.push(Seg { sx: xs[pos] as usize, ex: xs[be - 1] as usize, coeffs: bc }); pos = be;
    }
    // Handle trailing points that didn't get covered by the main loop
    while pos < n {
        let e = (pos + 4).min(n);
        let rem = e - pos;
        let d = if rem < 3 { rem.saturating_sub(1).max(0) } else { 3.min(rem - 1).max(1) };
        let slice = &ys[pos..e];
        let coeffs = fit_poly(slice, xs[pos], d);
        segs.push(Seg { sx: xs[pos] as usize, ex: xs[e - 1] as usize, coeffs });
        pos = e;
    }
    segs
}

fn main() -> std::io::Result<()> {
    let file = Path::new("/home/savageduck26/Documents/Coding/Rust/Duckomp/Duckomp-v4/src/sample.txt");

    // === PHASE 1: Streaming file analysis (never loads whole file into RAM) ===
    println!("Analyzing all substrings (streaming)...\n");

    let (results, best_offsets) = find_common_substrings_streaming(file, MIN_SUBSTR_LEN, MAX_SUBSTR_LEN)?;

    let n = best_offsets.len();

    if n == 0 {
        println!("No repeated substrings found.");
        return Ok(());
    }

    println!("Repeated substrings:\n");
    println!("  {:30} {:>7} {:>6} {:>8}", "Substring", "Count", "Length", "Score");
    println!("  {}", "-".repeat(53));
    for (substr, count, len, score) in &results {
        let display = hex_fmt(substr);
        // Truncate display for table layout
        let disp = if display.len() > 28 {
            format!("{}..", &display[..26])
        } else {
            display
        };
        println!("  \"{disp:28}\" {count:7} {len:6} {score:8}");
    }

    // === PHASE 2: Output generation (no file access needed) ===
    let csv_path = file.parent().unwrap_or(Path::new(".")).join("duckomp_output.csv");
    let mut csv = File::create(&csv_path)?;
    let mut writer = io::BufWriter::new(&mut csv);

    let eq_path = file.parent().unwrap_or(Path::new(".")).join("duckomp_equations.txt");
    let mut eq_file = File::create(&eq_path)?;
    let mut eq_writer = io::BufWriter::new(&mut eq_file);

    let (_best_sub, _best_count, _best_len, _best_score) = &results[0];
    writeln!(writer, "offset")?;
    for &o in &best_offsets { writeln!(writer, "{o}")?; }

    let xs: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let ys: Vec<f64> = best_offsets.iter().map(|&o| o as f64).collect();
    drop(best_offsets);

    let segs = segment_data(&xs, &ys);

    let mut total_hits = 0; let mut misses = Vec::new();
    for seg in &segs {
        for i in (seg.sx - 1)..seg.ex {
            let p = eval_rounded(&seg.coeffs, xs[i]);
            if p == ys[i] as i64 { total_hits += 1; } else { misses.push(i + 1); }
        }
    }
    let hit_rate = total_hits as f64 / n as f64 * 100.0;

    println!();
    println!("=== Segmentation ===");
    println!("{n} pts → {} equations, {}/{} hits ({:.1}%)", segs.len(), total_hits, n, hit_rate);
    if !misses.is_empty() { println!("Missed: {}", misses.len()); }
    println!();

    let i0 = if segs.is_empty() { 0.0 } else { segs[0].coeffs[0] };
    let s_val = if segs.is_empty() { 1.0 } else { segs[0].coeffs[1] };

    let valid_segs: Vec<&Seg> = segs.iter().enumerate()
        .filter(|(si, seg)| {
            let expected_xm = i0 + *si as f64 * 4.0;
            let xm_match = (seg.coeffs[0] - expected_xm).abs() < 1e-9;
            let scale_match = (seg.coeffs[1] - s_val).abs() < 1e-9;
            xm_match && scale_match
        })
        .map(|(_, seg)| seg)
        .collect();

    let n_dropped = segs.len() - valid_segs.len();
    let valid_n: usize = valid_segs.iter().map(|seg| seg.ex - seg.sx + 1).sum();
    let valid_hits = valid_segs.iter().map(|seg| {
        let mut h = 0;
        for i in (seg.sx - 1)..seg.ex {
            if eval_rounded(&seg.coeffs, xs[i]) == ys[i] as i64 { h += 1; }
        }
        h
    }).sum::<usize>();
    let valid_hit_rate = if valid_n > 0 { valid_hits as f64 / valid_n as f64 * 100.0 } else { 0.0 };

    // Write best substring as raw bytes to equations file header
    eq_writer.write_all(&results[0].0)?;
    writeln!(eq_writer)?;
    writeln!(eq_writer, "{}", n)?;
    writeln!(eq_writer, "{}", segs.len())?;
    writeln!(eq_writer, "{}", i0)?;
    writeln!(eq_writer, "{}", s_val)?;
    writeln!(eq_writer)?;

    let (dec_counts, whole_counts) = build_coeff_freqs(&valid_segs);

    let var_names = ["d", "f", "h", "k", "m", "p", "q", "r", "t", "v", "w", "y", "z", "b", "c", "g", "j", "l", "a", "o"];
    let mut scored: Vec<_> = dec_counts.into_iter()
        .filter(|(s, c)| {
            let min_count = if s.len() == 2 { 3 } else { 2 };
            *c >= min_count && s.len() >= 2
        })
        .map(|(s, c)| (s.clone(), c, c * s.len()))
        .collect();
    scored.sort_by(|a, b| b.2.cmp(&a.2));
    let mut suffix_map: Vec<(String, String)> = Vec::new();
    for (suffix, _count, _savings) in &scored {
        if suffix_map.len() < var_names.len() {
            suffix_map.push((suffix.clone(), var_names[suffix_map.len()].to_string()));
        }
    }
    suffix_map.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    // Reserved symbols for polynomial power suffixes (u3, u2, u) — always present
    let power_var_syms = [("u3", "@"), ("u2", "`")];
    let power_reserved: std::collections::HashSet<&str> =
        power_var_syms.iter().map(|(_, s)| *s).collect();

    // Build a large symbol pool: symbols first, then uppercase A-Z
    let sym_base: Vec<&str> = vec!["#", "$", "%", "&", "*", "(", ")", "[", "]", "{", "}", ":", ";", "'", "\"", "<", ">", "?", "/", "!"];
    let upper: Vec<String> = (b'A'..=b'Z').map(|c| (c as char).to_string()).collect();
    let uppper_refs: Vec<&str> = upper.iter().map(|s| s.as_str()).collect();
    let mut all_syms: Vec<&str> = Vec::with_capacity(sym_base.len() + uppper_refs.len());
    all_syms.extend(&sym_base);
    all_syms.extend(&uppper_refs);

    let mut whole_scored: Vec<_> = whole_counts.into_iter()
        .filter(|(s, c)| s.len() >= 2 && *c >= 3)
        .map(|(s, c)| (s.clone(), c, c * (s.len() - 1)))
        .collect();
    whole_scored.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.0.len().cmp(&a.0.len())));
    let mut whole_map: Vec<(String, String)> = Vec::new();
    for (val, _count, _savings) in &whole_scored {
        if whole_map.len() < all_syms.len() {
            let conflicts_with_sub = suffix_map.iter().any(|(s, _v)| s == val)
                || power_reserved.contains(val.as_str());
            if !conflicts_with_sub {
                whole_map.push((val.clone(), all_syms[whole_map.len()].to_string()));
            }
        }
    }
    whole_map.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut substr_sym_map: Vec<(String, String)> = Vec::new();
    for (suffix, _count, _savings) in &scored {
        if substr_sym_map.len() >= all_syms.len() { break; }
        let already_mapped = suffix_map.iter().any(|(s, _)| s == suffix)
            || whole_map.iter().any(|(s, _)| s == suffix)
            || power_reserved.contains(suffix.as_str());
        if !already_mapped {
            substr_sym_map.push((suffix.clone(), all_syms[substr_sym_map.len()].to_string()));
        }
    }
    substr_sym_map.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut combined_shorthand: Vec<(String, String)> = suffix_map.clone();
    combined_shorthand.extend(substr_sym_map.iter().cloned());
    combined_shorthand.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    // Track used symbols to avoid duplicates
    let mut used_symbols: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_, sym) in &suffix_map { used_symbols.insert(sym.clone()); }
    for (_, sym) in &power_var_syms { used_symbols.insert(sym.to_string()); }

    // Filter whole_map to only use unused symbols
    let whole_map_dedup: Vec<(String, String)> = whole_map.into_iter()
        .filter(|(_, sym)| used_symbols.insert(sym.clone()))
        .collect();
    let whole_map = whole_map_dedup;

    // Filter substr_sym_map to only use unused symbols
    let substr_sym_map_dedup: Vec<(String, String)> = substr_sym_map.into_iter()
        .filter(|(_, sym)| used_symbols.insert(sym.clone()))
        .collect();
    let substr_sym_map = substr_sym_map_dedup;

    for (suffix, var) in &suffix_map { writeln!(eq_writer, "{}={}", var, suffix)?; }
    for (substr, sym) in &substr_sym_map { writeln!(eq_writer, "{}={}", sym, substr)?; }
    for (val, sym) in &whole_map { writeln!(eq_writer, "{}={}", sym, val)?; }
    for (var, sym) in &power_var_syms { writeln!(eq_writer, "{}={}", sym, var)?; }
    if !suffix_map.is_empty() || !substr_sym_map.is_empty() || !whole_map.is_empty() { writeln!(eq_writer)?; }

    for (_si, seg) in valid_segs.iter().enumerate() {
        let deg = seg.coeffs.len() - 3;
        let mut first = true;
        for d in (0..=deg).rev() {
            let c = seg.coeffs[2 + d];
            if c.abs() < 1e-12 && !first { continue; }
            if c.abs() >= 1e-12 || first {
                let sign = if first { if c < 0.0 { "-" } else { "" } } else { if c >= 0.0 { "+" } else { "-" } };
                first = false;
                let v = c.abs();
                let vs = format_coeff(v);
                let vs_sym = whole_map.iter()
                    .find(|(val, _)| vs == *val)
                    .map(|(_, sym)| sym.clone())
                    .unwrap_or(vs);
                let vs_final = apply_dec_shorthand(&vs_sym, &combined_shorthand);
                let var_part = if d == 0 { String::new() } else {
                    let key = if d == 1 { "u".to_string() } else { format!("u{}", d) };
                    power_var_syms.iter()
                        .find(|(pv, _)| *pv == key)
                        .map(|(_, sym)| sym.to_string())
                        .unwrap_or(key)
                };
                if d > 0 && (v - 1.0).abs() < 1e-12 {
                    write!(eq_writer, "{}{}", sign, var_part)?;
                } else {
                    write!(eq_writer, "{}{}{}", sign, vs_final, var_part)?;
                }
            }
        }
        if first { write!(eq_writer, "0")?; }
        writeln!(eq_writer)?;
    }
    eq_writer.flush()?;

    if !misses.is_empty() {
        println!("Missed: {}/{} points", misses.len(), n);
    }

    writeln!(writer)?;
    writeln!(writer, "# Fit — {n} pts → {} eqs, {}/{} hits ({:.1}%)", segs.len(), total_hits, n, hit_rate)?;
    writeln!(writer, "#")?;
    for (si, seg) in segs.iter().enumerate() {
        writeln!(writer, "# Eq {si}: x∈[{},{}]", seg.sx, seg.ex)?;
        writeln!(writer, "#   x_mean={:.4}, x_scale={:.4}", seg.coeffs[0], seg.coeffs[1])?;
        for d in 0..(seg.coeffs.len() - 2) { writeln!(writer, "#   c{d}={:.15e}", seg.coeffs[2 + d])?; }
    }
    writeln!(writer, "#")?;
    writeln!(writer, "index,offset,predicted,correct")?;
    for i in 0..n {
        let mut p = 0i64;
        for seg in &segs { if i + 1 >= seg.sx && i + 1 <= seg.ex { p = eval_rounded(&seg.coeffs, xs[i]); break; } }
        let correct = if p == ys[i] as i64 { "true" } else { "false" };
        writeln!(writer, "{},{},{},{}", i+1, ys[i] as i64, p, correct)?;
    }
    writer.flush()?;
    println!("CSV: {}", csv_path.display());
    println!("Equations: {}", eq_path.display());
    Ok(())
}