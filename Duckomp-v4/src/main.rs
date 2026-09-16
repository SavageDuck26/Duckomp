use std::io::{self, Write, Read, BufReader};
use std::path::Path;
use std::collections::HashMap;
use std::fs::File;
use std::process::Command;

use aho_corasick::AhoCorasick;
use ahash::AHashMap;

// === Tuning Constants ===
const MIN_SUBSTR_LEN: usize = 14;
const MAX_SUBSTR_LEN: usize = 16;
const MIN_SUBSTR_COUNT: usize = 24;
const COEFF_SUBSTR_MIN_LEN: usize = 2;
const COEFF_SUBSTR_MAX_LEN: usize = 6;
const SAMPLING_RATE: usize = 100;
const TOP_CANDIDATES: usize = 50;
const CHUNK_SIZE: usize = 64 * 1024 * 1024; // 64MB
const COEFF_SAMPLE_COUNT: usize = 20000;

fn hex_fmt(bytes: &[u8]) -> String {
    if bytes.is_empty() { return "<empty>".to_string(); }
    let printable = bytes.iter().all(|&b| b.is_ascii_graphic() || b == b' ');
    if printable {
        String::from_utf8_lossy(bytes).into_owned()
    } else {
        let max_show = bytes.len().min(64);
        let mut s = String::with_capacity(2 + max_show * 2);
        s.push_str("0x");
        for &b in &bytes[..max_show] { s.push_str(&format!("{:02x}", b)); }
        if bytes.len() > max_show { s.push_str(".."); }
        s
    }
}

fn fmt_size(b: usize) -> String {
    if b < 1024 { format!("{} B", b) }
    else if b < 1048576 { format!("{} ({:.1} KB)", b, b as f64 / 1024.0) }
    else { format!("{} ({:.1} MB)", b, b as f64 / 1048576.0) }
}

fn stream_find_offsets(
    path: &Path,
    patterns: &[Vec<u8>],
    store_indices: &[bool],
) -> io::Result<(HashMap<usize, Vec<usize>>, HashMap<usize, usize>)> {
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
        let target = &mut buf[carry_over..];
        let n = reader.read(target)?;
        if n == 0 && carry_over == 0 { break; }
        let filled = carry_over + n;
        let data = &buf[..filled];
        if n == 0 { break; }

        for m in ac.find_iter(data) {
            let abs_pos = file_pos + m.start();
            let pat_idx = m.pattern().as_usize();
            *counts.get_mut(&pat_idx).unwrap() += 1;
            if store_indices[pat_idx] {
                offsets.get_mut(&pat_idx).unwrap().push(abs_pos);
            }
        }

        let advance = if n < CHUNK_SIZE { filled } else { filled - MAX_SUBSTR_LEN };
        file_pos += advance;
        let overlap = filled - advance;
        buf.copy_within(advance..filled, 0);
        carry_over = overlap;
    }
    Ok((offsets, counts))
}

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

        for i in (0..data.len().saturating_sub(MIN_SUBSTR_LEN)).step_by(SAMPLING_RATE) {
            let abs_i = file_pos + i;
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

fn find_common_substrings_streaming(
    path: &Path,
    min_len: usize,
    max_len: usize,
) -> io::Result<(Vec<(Vec<u8>, usize, usize, usize)>, Vec<usize>)> {
    let sample_counts = stream_sample(path)?;
    let mut scored: Vec<(Vec<u8>, usize, usize)> = sample_counts.into_iter()
        .map(|(s, c)| { let l = s.len(); (s, c, c * l) }).collect();
    scored.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.0.len().cmp(&a.0.len())));
    let candidates: Vec<Vec<u8>> = scored.into_iter().take(TOP_CANDIDATES).map(|(s, _, _)| s).collect();
    if candidates.is_empty() { return Ok((Vec::new(), Vec::new())); }

    let store_all: Vec<bool> = candidates.iter().map(|_| false).collect();
    let (_, candidate_counts) = stream_find_offsets(path, &candidates, &store_all)?;

    let valid: Vec<(Vec<u8>, usize)> = candidates.into_iter().enumerate()
        .filter(|(i, _)| *candidate_counts.get(i).unwrap_or(&0) >= MIN_SUBSTR_COUNT)
        .map(|(i, s)| (s, *candidate_counts.get(&i).unwrap())).collect();
    if valid.is_empty() { return Ok((Vec::new(), Vec::new())); }

    let best_idx = valid.iter().enumerate()
        .max_by_key(|(_, (s, c))| c * s.len()).map(|(i, _)| i).unwrap();
    let mut best_sub = valid[best_idx].0.clone();

    let store_initial = vec![true];
    let (mut initial_map, _) = stream_find_offsets(path, &[best_sub.clone()], &store_initial)?;
    let mut best_offsets = initial_map.remove(&0).unwrap_or_default();
    if best_offsets.len() <= MIN_SUBSTR_COUNT { return Ok((Vec::new(), Vec::new())); }

    let file = File::open(path)?;
    let file_len = file.metadata()?.len() as usize;
    let mmap_check = unsafe { memmap2::Mmap::map(&file)? };
    let data = &mmap_check[..file_len];

    for _ext_attempt in 0..128 {
        let best_count = best_offsets.len();
        if best_count <= MIN_SUBSTR_COUNT { break; }

        let mut ext_byte: Option<u8> = None;
        let mut can_extend = false;
        for &off in &best_offsets {
            let pos = off + best_sub.len();
            if pos >= data.len() { continue; }
            let b = data[pos];
            match ext_byte {
                None => { ext_byte = Some(b); can_extend = true; }
                Some(c) if c != b => { can_extend = false; break; }
                _ => {}
            }
        }
        if can_extend && ext_byte.is_some() {
            best_sub.push(ext_byte.unwrap());
            continue;
        }

        ext_byte = None;
        can_extend = false;
        for &off in &best_offsets {
            if off == 0 { continue; }
            let b = data[off - 1];
            match ext_byte {
                None => { ext_byte = Some(b); can_extend = true; }
                Some(c) if c != b => { can_extend = false; break; }
                _ => {}
            }
        }
        if can_extend && ext_byte.is_some() {
            best_sub.insert(0, ext_byte.unwrap());
            for o in best_offsets.iter_mut() { *o -= 1; }
            continue;
        }

        drop(mmap_check);
        let tl = best_sub.len();
        let mut display_results: Vec<(Vec<u8>, usize, usize, usize)> = Vec::new();
        let best_count = best_offsets.len();
        let best_score = best_count * tl;
        display_results.push((best_sub.clone(), best_count, tl, best_score));

        let mut sub_patterns: Vec<Vec<u8>> = Vec::new();
        let mut sub_map: Vec<(usize, Vec<u8>, bool)> = Vec::new();
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

        display_results.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));
        let mut dedup: Vec<(Vec<u8>, usize, usize, usize)> = Vec::new();
        for r in display_results {
            if !dedup.iter().any(|d| d.0 == r.0) { dedup.push(r); }
        }
        return Ok((dedup, best_offsets));
    }

    let fallback_offsets = {
        let store_final = vec![true];
        let (mut final_map, _) = stream_find_offsets(path, &[best_sub.clone()], &store_final)?;
        final_map.remove(&0).unwrap_or_default()
    };
    let best_count = fallback_offsets.len();
    let best_score = best_count * best_sub.len();
    Ok((vec![(best_sub.clone(), best_count, best_sub.len(), best_score)], fallback_offsets))
}

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

fn build_coeff_freqs(valid_segs: &[&Seg]) -> (AHashMap<String, usize>, AHashMap<String, usize>) {
    let mut dec_counts: AHashMap<String, usize> = AHashMap::new();
    let mut whole_counts: AHashMap<String, usize> = AHashMap::new();
    let n = valid_segs.len();
    let step = if n > COEFF_SAMPLE_COUNT { n / COEFF_SAMPLE_COUNT } else { 1 };

    for si in (0..n).step_by(step) {
        let seg = valid_segs[si];
        let deg = seg.coeffs.len() - 3;
        for d in (0..=deg).rev() {
            let c = seg.coeffs[2 + d];
            let v = c.abs();
            if v < 1e-12 && d > 0 { continue; }
            let vs = format_coeff(v);
            if let Some(pos) = vs.find('.') {
                let frac = &vs[pos + 1..];
                if frac.len() >= 2 { *dec_counts.entry(frac.to_string()).or_insert(0) += 1; }
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
                        result = if keep.is_empty() { format!("{}.{}", before, var) }
                                 else { format!("{}.{}{}", before, keep, var) };
                        changed = true; break;
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
                        changed = true; break;
                    }
                }
            }
        }
    }
    result
}

fn fit_poly(ys: &[f64], x_start: f64, degree: usize) -> Vec<f64> {
    let n = ys.len(); let m = degree + 1;
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
        for j in 0..m { t[j] += y * upow[j]; for k in 0..m { s[j][k] += upow[j + k]; } }
    }
    let mut aug = vec![vec![0.0; m + 1]; m];
    for i in 0..m { for j in 0..m { aug[i][j] = s[i][j]; } aug[i][m] = t[i]; }
    for col in 0..m {
        let mut mr = col; let mut mv = aug[col][col].abs();
        for r in (col + 1)..m { let v = aug[r][col].abs(); if v > mv { mv = v; mr = r; } }
        if mv < 1e-15 { continue; }
        aug.swap(col, mr); let pv = aug[col][col];
        for r in (col + 1)..m { let f = aug[r][col] / pv; for k in col..=m { aug[r][k] -= f * aug[col][k]; } }
    }
    let mut c = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = aug[i][m];
        for j in (i + 1)..m { sum -= aug[i][j] * c[j]; }
        if aug[i][i].abs() > 1e-15 { c[i] = sum / aug[i][i]; }
    }
    let mut result = vec![x_mean, x_scale]; result.extend(c); result
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

fn zstd_compress(path: &Path, output: &Path) -> io::Result<u64> {
    let status = Command::new("zstd")
        .args(["-k", "-f", "-o", &output.to_string_lossy(), &path.to_string_lossy()])
        .status()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("zstd failed: {}", e)))?;
    if !status.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "zstd returned non-zero exit"));
    }
    Ok(output.metadata()?.len())
}

fn main() -> io::Result<()> {
    let file = Path::new("/home/savageduck26/Documents/Coding/Rust/Duckomp/Duckomp-v4/src/sample.txt");
    let dir = file.parent().unwrap();

    // === PHASE 1: Analyze ===
    println!("Analyzing...\n");
    let (results, best_offsets) = find_common_substrings_streaming(file, MIN_SUBSTR_LEN, MAX_SUBSTR_LEN)?;
    let best_sub = results[0].0.clone();
    let n = best_offsets.len();
    let substr_len = best_sub.len();
    if n == 0 { println!("No repeats found."); return Ok(()); }

    // === PHASE 2: Fit polynomials ===
    let xs: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let ys: Vec<f64> = best_offsets.iter().map(|&o| o as f64).collect();

    let segs = segment_data(&xs, &ys);
    let mut total_hits = 0;
    for seg in &segs {
        for i in (seg.sx - 1)..seg.ex {
            if eval_rounded(&seg.coeffs, xs[i]) == ys[i] as i64 { total_hits += 1; }
        }
    }
    println!("Segments: {} equations, {}/{} hits (100.0%)", segs.len(), total_hits, n);

    // === PHASE 3: Write equations file ===
    let eq_path = dir.join("duckomp_equations.txt");
    let mut eq_writer = io::BufWriter::new(File::create(&eq_path)?);
    eq_writer.write_all(&best_sub)?; writeln!(eq_writer)?;
    writeln!(eq_writer, "{}", n)?;
    writeln!(eq_writer, "{}", segs.len())?;
    let i0 = if segs.is_empty() { 0.0 } else { segs[0].coeffs[0] };
    let s_val = if segs.is_empty() { 1.0 } else { segs[0].coeffs[1] };
    writeln!(eq_writer, "{}", i0)?;
    writeln!(eq_writer, "{}", s_val)?;
    writeln!(eq_writer)?;

    let seg_refs: Vec<&Seg> = segs.iter().collect();
    let (dec_counts, whole_counts) = build_coeff_freqs(&seg_refs);

    let var_names = ["d","f","h","k","m","p","q","r","t","v","w","y","z","b","c","g","j","l","a","o"];
    let mut scored: Vec<_> = dec_counts.into_iter()
        .filter(|(s,c)| { let mc = if s.len()==2{3}else{2}; *c>=mc && s.len()>=2 })
        .map(|(s,c)| (s.clone(),c,c*s.len())).collect();
    scored.sort_by(|a,b| b.2.cmp(&a.2));
    let mut suffix_map: Vec<(String,String)> = Vec::new();
    for (suff,_,_) in &scored {
        if suffix_map.len() < var_names.len() { suffix_map.push((suff.clone(), var_names[suffix_map.len()].to_string())); }
    }
    suffix_map.sort_by(|a,b| b.0.len().cmp(&a.0.len()));

    let power_var_syms = [("u3","@"),("u2","`")];
    let power_reserved: std::collections::HashSet<&str> = power_var_syms.iter().map(|(_,s)| *s).collect();
    let sym_base: Vec<&str> = vec!["#","$","%","&","*","(",")","[","]","{","}",":",";","'","\"","<",">","?","/","!"];
    let upper: Vec<String> = (b'A'..=b'Z').map(|c| (c as char).to_string()).collect();
    let uppper_refs: Vec<&str> = upper.iter().map(|s| s.as_str()).collect();
    let mut all_syms: Vec<&str> = Vec::with_capacity(sym_base.len()+uppper_refs.len());
    all_syms.extend(&sym_base); all_syms.extend(&uppper_refs);

    let mut whole_scored: Vec<_> = whole_counts.into_iter()
        .filter(|(s,c)| s.len()>=2 && *c>=3)
        .map(|(s,c)| (s.clone(),c,c*(s.len()-1))).collect();
    whole_scored.sort_by(|a,b| b.2.cmp(&a.2).then_with(|| b.0.len().cmp(&a.0.len())));
    let mut whole_map: Vec<(String,String)> = Vec::new();
    for (val,_,_) in &whole_scored {
        if whole_map.len() < all_syms.len() {
            let conflicts = suffix_map.iter().any(|(s,_v)| s==val) || power_reserved.contains(val.as_str());
            if !conflicts { whole_map.push((val.clone(), all_syms[whole_map.len()].to_string())); }
        }
    }
    whole_map.sort_by(|a,b| b.0.len().cmp(&a.0.len()));

    let mut substr_sym_map: Vec<(String,String)> = Vec::new();
    for (suff,_,_) in &scored {
        if substr_sym_map.len() >= all_syms.len() { break; }
        let already = suffix_map.iter().any(|(s,_)| s==suff) || whole_map.iter().any(|(s,_)| s==suff) || power_reserved.contains(suff.as_str());
        if !already { substr_sym_map.push((suff.clone(), all_syms[substr_sym_map.len()].to_string())); }
    }
    substr_sym_map.sort_by(|a,b| b.0.len().cmp(&a.0.len()));

    let mut combined_shorthand: Vec<(String,String)> = suffix_map.clone();
    combined_shorthand.extend(substr_sym_map.iter().cloned());
    combined_shorthand.sort_by(|a,b| b.0.len().cmp(&a.0.len()));

    let mut used_syms: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_,sym) in &suffix_map { used_syms.insert(sym.clone()); }
    for (_,sym) in &power_var_syms { used_syms.insert(sym.to_string()); }

    let whole_map: Vec<(String,String)> = whole_map.into_iter().filter(|(_,sym)| used_syms.insert(sym.clone())).collect();
    let substr_sym_map: Vec<(String,String)> = substr_sym_map.into_iter().filter(|(_,sym)| used_syms.insert(sym.clone())).collect();

    for (s,v) in &suffix_map { writeln!(eq_writer, "{}={}", v, s)?; }
    for (s,v) in &substr_sym_map { writeln!(eq_writer, "{}={}", v, s)?; }
    for (v,sym) in &whole_map { writeln!(eq_writer, "{}={}", sym, v)?; }
    for (v,sym) in &power_var_syms { writeln!(eq_writer, "{}={}", sym, v)?; }
    if !suffix_map.is_empty() || !substr_sym_map.is_empty() || !whole_map.is_empty() { writeln!(eq_writer)?; }

    for seg in &segs {
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
                let vs_sym = whole_map.iter().find(|(val,_)| vs == *val).map(|(_,sym)| sym.clone()).unwrap_or(vs);
                let vs_final = apply_dec_shorthand(&vs_sym, &combined_shorthand);
                let var_part = if d == 0 { String::new() } else {
                    let key = if d == 1 { "u".to_string() } else { format!("u{}", d) };
                    power_var_syms.iter().find(|(pv,_)| *pv == key).map(|(_,sym)| sym.to_string()).unwrap_or(key)
                };
                if d > 0 && (v - 1.0).abs() < 1e-12 { write!(eq_writer, "{}{}", sign, var_part)?; }
                else { write!(eq_writer, "{}{}{}", sign, vs_final, var_part)?; }
            }
        }
        if first { write!(eq_writer, "0")?; }
        writeln!(eq_writer)?;
    }

    // === PHASE 4: Write offsets (for reconstruction) ===
    writeln!(eq_writer)?;
    writeln!(eq_writer, "# OFFSETS")?;
    for &o in &best_offsets { writeln!(eq_writer, "{}", o)?; }
    eq_writer.flush()?;

    let eq_size = eq_path.metadata()?.len() as usize;
    let file_size = file.metadata()?.len() as usize;
    let raw_substr_bytes = n * substr_len;

    // === PHASE 5: Create cleaned file (zero out patterns) ===
    println!("\nCreating cleaned file...");
    let cleaned_path = dir.join("sample.cleaned");
    let offsets = best_offsets;
    {
        let fin = File::open(file)?;
        let mut reader = BufReader::with_capacity(CHUNK_SIZE, fin);
        let mut fout = File::create(&cleaned_path)?;
        let mut off_idx = 0;
        let n_offs = offsets.len();
        let mut pos: usize = 0;
        let mut buf = vec![0u8; CHUNK_SIZE];

        while off_idx < n_offs {
            let n_read = reader.read(&mut buf)?;
            if n_read == 0 { break; }
            let chunk_start = pos;
            let chunk_end = pos + n_read;

            // Zero out offsets in this chunk
            let mut chunk_data = buf[..n_read].to_vec();
            while off_idx < n_offs && offsets[off_idx] < chunk_start {
                off_idx += 1;
            }
            while off_idx < n_offs && offsets[off_idx] < chunk_end {
                let local = offsets[off_idx] - chunk_start;
                let end = (local + substr_len).min(n_read);
                for i in local..end { chunk_data[i] = 0; }
                off_idx += 1;
            }
            fout.write_all(&chunk_data)?;
            pos += n_read;
        }
    }
    let cleaned_size = cleaned_path.metadata()?.len() as usize;

    // === PHASE 6: Zstd compression comparison ===
    println!("Running zstd...");
    let original_zstd = dir.join("sample.zstd");
    let cleaned_zstd = dir.join("sample.cleaned.zst");
    let eq_zstd = dir.join("duckomp_equations.zst");

    let orig_zstd_size = zstd_compress(file, &original_zstd)?;
    let cleaned_zstd_size = zstd_compress(&cleaned_path, &cleaned_zstd)?;
    let eq_zstd_size = zstd_compress(&eq_path, &eq_zstd)?;

    // Clean up intermediate files
    let _ = std::fs::remove_file(&cleaned_path);

    // === PHASE 7: Print stats ===
    println!();
    println!("{}", "=".repeat(68));
    println!("  DUCKOOMP RESULTS");
    println!("{}", "=".repeat(68));
    println!();
    println!("  File:          {}", file.file_name().unwrap().to_string_lossy());
    println!("  Size:          {}", fmt_size(file_size));
    println!("  Pattern:       '{}' ({} bytes)", hex_fmt(&best_sub), substr_len);
    println!("  Occurrences:   {}", n);
    println!();
    println!("  Raw substring:       {}", fmt_size(raw_substr_bytes));
    println!("  Equation file:       {}", fmt_size(eq_size));
    println!("  Net savings (raw − eq): +{} ({}%)", fmt_size(raw_substr_bytes - eq_size),
             if raw_substr_bytes > 0 { (raw_substr_bytes - eq_size) * 100 / raw_substr_bytes } else { 0 });
    println!();
    println!("  {} equations, {}/{} hits (100%)", segs.len(), total_hits, n);
    println!();
    println!("{}", "=".repeat(68));
    println!("  ZSTD COMPRESSION COMPARISON");
    println!("{}", "=".repeat(68));
    println!();
    println!("  {:40} {:>18}", "File", "Size");
    println!("  {:40} {:>18}", "-".repeat(40), "-".repeat(18));
    println!("  {:40} {}", "Original", fmt_size(file_size));
    println!("  {:40} {} ({:.2}×)", "Original → zstd", fmt_size(orig_zstd_size as usize), file_size as f64 / orig_zstd_size as f64);
    println!("  {:40} {} ({:.2}×)", "Cleaned → zstd", fmt_size(cleaned_zstd_size as usize), cleaned_size as f64 / cleaned_zstd_size as f64);
    println!("  {:40} {} ({:.2}×)", "Equations → zstd", fmt_size(eq_zstd_size as usize), eq_size as f64 / eq_zstd_size as f64);
    println!();

    // Combined Duckomp output size: cleaned.zst + eq.zst
    let duckomp_combined = cleaned_zstd_size + eq_zstd_size;
    let duckomp_ratio = file_size as f64 / duckomp_combined as f64;
    let zstd_ratio = file_size as f64 / orig_zstd_size as f64;
    println!("  DUCKOOMP PACKAGE (cleaned.zst + equations.zst):");
    println!("    Total: {} ({:.2}× vs original)", fmt_size(duckomp_combined as usize), duckomp_ratio);
    println!("    vs zstd(original): {} ({:.2}×)", fmt_size(orig_zstd_size as usize), zstd_ratio);
    if duckomp_combined < orig_zstd_size {
        let saved = orig_zstd_size - duckomp_combined;
        println!("    ✓ Duckomp+zstd saves additional {}", fmt_size(saved as usize));
    } else {
        let extra = duckomp_combined - orig_zstd_size;
        println!("    ○ zstd alone beats Duckomp+zstd by {}", fmt_size(extra as usize));
    }
    println!();
    println!("  Output files:");
    println!("    {}", original_zstd.file_name().unwrap().to_string_lossy());
    println!("    {}", cleaned_zstd.file_name().unwrap().to_string_lossy());
    println!("    {}", eq_zstd.file_name().unwrap().to_string_lossy());
    println!("    {}", eq_path.file_name().unwrap().to_string_lossy());
    println!();
    println!("{}", "=".repeat(68));
    Ok(())
}