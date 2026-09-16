//! Duckomp-v7 equation solver (module, used by main.rs)
//!
//! Brute-force search for a short, precise equation that reproduces one giant
//! integer (the "seed"):
//!
//!     seed = Σ σᵢ · cᵢ·nᵢ^eᵢ  +  magic
//!
//! Each term uses a base nᵢ ∈ [2, MAX_BASE], a coefficient cᵢ < nᵢ, a sign
//! σᵢ ∈ {+1, −1}, and `magic` is a correction number that absorbs whatever is
//! left over. Every emitted equation is replayed against the seed exactly.
//!
//! What it does:
//!   1. single-power sweep — for every base 2..=MAX_BASE, find the closest
//!      c·n^e to the seed and rank them ("what lands close").
//!   2. greedy multi-term  — repeatedly peel the closest-fitting term off the
//!      residual. Power caches make each step cheap.
//!   3. verify             — replay the equation against the seed, exact.
//!
//! Diagnostics (sweep table, per-step peeling) go to stderr so the stdout
//! output stays clean.

use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use std::sync::Mutex;

/// How many top-ranked bases get an exact (BigUint) evaluation per phase.
const TOP_EXACT: usize = 10;
/// Coefficient bit budget: a term's coefficient may hold up to this many bits
/// (multi-limb chunk), so each term removes ~COEF_BITS bits instead of one
/// base-digit. 0 disables the window (single-limb, old behaviour).
const COEF_BITS: u32 = 60;

// ---------------------------------------------------------------------------
// Candidate bases
// ---------------------------------------------------------------------------

/// Dense small bases plus log-spaced high bases, so raising MAX_BASE stays cheap.
fn base_candidates(max_base: u64) -> Vec<u64> {
    let mut v: Vec<u64> = (2..=1024.min(max_base)).collect();
    let mut p = 2048u64;
    while p <= max_base {
        for c in [p, p / 2, p / 4, p + p / 4, p + p / 2] {
            if c >= 2 && c <= max_base {
                v.push(c);
            }
        }
        p *= 2;
    }
    let mut r = 2000u64;
    while r <= max_base {
        for c in [r, r + r / 2] {
            if c >= 2 && c <= max_base {
                v.push(c);
            }
        }
        r *= 10;
    }
    v.sort_unstable();
    v.dedup();
    v
}

// ---------------------------------------------------------------------------
// Term-cost helpers (for the size-aware greedy)
// ---------------------------------------------------------------------------

/// Decimal digit count of a u64.
fn digits(x: u64) -> usize {
    x.to_string().len()
}


// ---------------------------------------------------------------------------
// Log / power helpers
// ---------------------------------------------------------------------------

/// Fast f64 approximation of log2(x), accurate to < 2^-60.
fn approx_log2(x: &BigUint) -> f64 {
    if x.is_zero() {
        return 0.0;
    }
    let digits = x.to_u64_digits();
    let top = *digits.last().unwrap_or(&0) as f64;
    let limbs = digits.len() as f64;
    64.0 * (limbs - 1.0) + top.log2()
}

/// base^exp. Powers of two become bit shifts (instant); everything else uses
/// exponentiation by squaring via num-bigint.
fn fast_pow(base: u64, exp: u32) -> BigUint {
    let l2n = (base as f64).log2();
    if l2n.fract().abs() < 1e-12 {
        let shift = (exp as f64 * l2n).round() as usize;
        BigUint::from(1u64) << shift
    } else {
        BigUint::from(base).pow(exp)
    }
}

/// Caches base^e so the greedy loop can step e up/down cheaply instead of
/// redoing a full exponentiation every step.
struct PowCache {
    base: u64,
    e: u32,
    p: BigUint,
    seeded: bool,
}

impl PowCache {
    fn new(base: u64) -> Self {
        PowCache {
            base,
            e: 0,
            p: BigUint::from(1u64),
            seeded: false,
        }
    }

    /// First-time sync from scratch: estimate e, compute base^e, fix up.
    fn seed(&mut self, value: &BigUint) {
        let l2v = approx_log2(value);
        let l2n = (self.base as f64).log2();
        self.e = (l2v / l2n) as u32;
        self.p = fast_pow(self.base, self.e);
        self.seeded = true;
        self.sync(value);
    }

    /// Step p = base^e to the largest power of `base` that is <= value.
    fn sync(&mut self, value: &BigUint) {
        let n = BigUint::from(self.base);
        while &self.p > value {
            self.e -= 1;
            self.p = &self.p / &n; // exact: p was base^(e+1)
        }
        while &self.p * &n <= *value {
            self.e += 1;
            self.p *= &n;
        }
    }
}


// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Term {
    base: u64,
    exp: u32,
    coef: BigUint,
    /// true  -> term = +coef·base^exp but it *overshoots* the residual
    ///          (the residual flips sign in the telescope)
    /// false -> term = +coef·base^exp and it fits under the residual
    ceil: bool,
}

struct Ranked {
    base: u64,
    e: u32,
    a_digits: usize,
    score_est: f64,
}

/// Insert `r` into the top-k list (kept sorted by score_est, descending).
fn insert_top_k(best: &mut Vec<Ranked>, r: Ranked, k: usize) {
    if best.len() < k {
        let idx = best
            .iter()
            .position(|x| x.score_est < r.score_est)
            .unwrap_or(best.len());
        best.insert(idx, r);
    } else if r.score_est > best[k - 1].score_est {
        let idx = best
            .iter()
            .position(|x| x.score_est < r.score_est)
            .unwrap_or(k);
        best.insert(idx, r);
        best.truncate(k);
    }
}

/// Per-base efficiency estimate (pure f64, cheap). Searches the multi-limb
/// exponent window [e_max - COEF_BITS/log2(base), e_max] using a fractional
/// remainder recurrence, and returns the best (exponent, coefficient) score.
fn estimate_ranked(base: u64, l2v: f64, coef_bits: u32) -> Ranked {
    let l2n = (base as f64).log2();
    let mut e = (l2v / l2n) as u32;
    let mut d = l2v - e as f64 * l2n; // log2(value / base^e), in [0, l2n)
    if d < 0.0 {
        if e > 0 {
            e -= 1;
        }
        d += l2n;
    }
    if d >= l2n {
        e += 1;
        d -= l2n;
    }
    let w = (coef_bits as f64 / l2n).floor() as u32;
    let ratio = d.exp2(); // value / base^e, in [1, base)
    let a_f = ratio.floor();
    let mut frac = ratio - a_f;
    let mut a_digits = digits(a_f as u64);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_e = e;
    let mut best_a_digits = a_digits;
    for _ in 0..=w {
        if e > 0 {
            let m = frac.min(1.0 - frac).max(1e-12);
            let gap_bits = e as f64 * l2n + m.log2();
            let bits_removed = (l2v - gap_bits).max(0.0);
            if bits_removed > best_score {
                best_score = bits_removed;
                best_e = e;
                best_a_digits = a_digits;
            }
        }
        if e == 0 {
            break;
        }
        // Step down one limb: a' = base*a + floor(base*frac), frac' = frac(base*frac).
        let scaled = frac * base as f64;
        let k = scaled.floor();
        a_digits += digits(base);
        frac = scaled - k;
        e -= 1;
    }

    Ranked {
        base,
        e: best_e,
        a_digits: best_a_digits,
        score_est: best_score,
    }
}

/// Rank the best `k` bases out of the candidate list by estimated efficiency
/// (bits removed per equation character). The scan is split across all
/// available threads; each chunk keeps a local top-k, then the chunks merge.
fn rank_bases(value: &BigUint, max_base: u64, k: usize, coef_bits: u32) -> Vec<Ranked> {
    let l2v = approx_log2(value);
    let candidates = base_candidates(max_base);
    candidates
        .into_par_iter()
        .fold(Vec::new, |mut acc: Vec<Ranked>, base| {
            insert_top_k(&mut acc, estimate_ranked(base, l2v, coef_bits), k);
            acc
        })
        .reduce(Vec::new, |mut a, b| {
            for r in b {
                insert_top_k(&mut a, r, k);
            }
            a
        })
}


/// Build the floor/ceil fit at the maintained (base^e, coefficient, remainder).
/// Floor: term = a·p <= value, gap = r.  Ceil: term = (a+1)·p, gap = p − r.
/// Picks the larger removal (smaller gap); the score is the bits removed,
/// which is what the fold contract maximizes.
fn build_fit(
    base: u64,
    e: u32,
    a: &BigUint,
    p: &BigUint,
    r: &BigUint,
    value_bits: u64,
) -> Option<(Term, BigUint, f64)> {
    let hi = p - r;
    if r.bits() <= hi.bits() {
        let bits_removed = value_bits.saturating_sub(r.bits()) as f64;
        Some((
            Term {
                base,
                exp: e,
                coef: a.clone(),
                ceil: false,
            },
            r.clone(),
            bits_removed,
        ))
    } else {
        let bits_removed = value_bits.saturating_sub(hi.bits()) as f64;
        Some((
            Term {
                base,
                exp: e,
                coef: a + 1u64,
                ceil: true,
            },
            hi,
            bits_removed,
        ))
    }
}

/// Best multi-limb fit of c·base^e to `value`. The exponent window
/// [e_max − w, e_max] is walked via exact base-n digit extraction
/// (no bigint division): value = a·base^e + r, and stepping down one limb
/// peels the next digit. Coefficients are arbitrary-size BigUints.
fn best_fit_cached(
    value: &BigUint,
    base: u64,
    cache: &mut PowCache,
    coef_bits: u32,
) -> (Term, BigUint) {
    if cache.seeded {
        cache.sync(value);
    } else {
        cache.seed(value);
    }
    let l2n = (base as f64).log2();
    let w = (coef_bits as f64 / l2n).floor() as u32;

    let mut p = cache.p.clone(); // base^e_max
    let mut e = cache.e;
    // Top coefficient a0 = floor(value / p), a0 in [1, base).
    let mut a0 = {
        let est = (approx_log2(value) - e as f64 * l2n).exp2() as u64;
        est.max(1).min(base - 1)
    };
    while &p * a0 > *value {
        a0 -= 1;
    }
    while &p * (a0 + 1) <= *value {
        a0 += 1;
    }
    let mut a = BigUint::from(a0);
    let mut r = value - &p * a0; // < p

    let mut best: Option<(Term, BigUint)> = None;
    let mut best_score = f64::NEG_INFINITY;
    for _ in 0..=w {
        if let Some((t, gap, score)) = build_fit(base, e, &a, &p, &r, value.bits()) {
            if score > best_score {
                best_score = score;
                best = Some((t, gap));
            }
        }
        if e == 0 {
            break;
        }
        // Step down one limb: p' = p/base, k = floor(r/p') in [0, base),
        // a' = a·base + k, r' = r − k·p'.
        let p_next = &p / BigUint::from(base); // exact
        let mut k_lo = 0u64;
        let mut k_hi = base; // exclusive
        while k_lo + 1 < k_hi {
            let mid = k_lo + (k_hi - k_lo) / 2;
            if &p_next * mid <= r {
                k_lo = mid;
            } else {
                k_hi = mid;
            }
        }
        r = r - &p_next * k_lo;
        a = &a * base + k_lo;
        p = p_next;
        e -= 1;
    }
    best.expect("at least one exponent fit")
}

// ---------------------------------------------------------------------------
// Greedy multi-term peel
// ---------------------------------------------------------------------------

fn greedy(
    seed: &BigUint,
    caches: &[Mutex<PowCache>],
    max_terms: usize,
    max_base: u64,
    coef_bits: u32,
    verbose: bool,
) -> (Vec<(i8, Term)>, i8, BigUint) {
    let mut residual = seed.clone();
    let mut rsign: i8 = 1; // sign of the residual inside the telescoping equation
    let mut terms = Vec::new();
    let show_all = max_terms <= 32;

    for step in 0..max_terms {
        if residual.is_zero() {
            break;
        }
        // Size-aware selection: rank bases by estimated bits-per-character,
        // then evaluate the top-ranked bases in parallel (each base owns its
        // own pow cache) and pick the term with the best exact score.
        let ranked = rank_bases(&residual, max_base, TOP_EXACT, coef_bits);
        let best = ranked
            .par_iter()
            .map(|r| {
                let mut cache = caches[r.base as usize].lock().unwrap();
                let (t, gap) = best_fit_cached(&residual, r.base, &mut cache, coef_bits);
                let bits_removed = residual.bits().saturating_sub(gap.bits()) as f64;
                (rsign, t, gap, bits_removed)
            })
            .max_by(|a, b| a.3.total_cmp(&b.3))
            .expect("at least one base to try");
        let (out_sign, t, gap, _score) = best;

        let show = verbose
            && (show_all || step < 20 || (step + 1) % 1000 == 0 || (step + 1) == max_terms);
        if show {
            let op = if out_sign > 0 { '+' } else { '-' };
            eprintln!(
                "  step {:>6}: {}{:<22}  residual {} -> {} bits",
                step + 1,
                op,
                fmt_term(&t),
                residual.bits(),
                gap.bits()
            );
        }

        if t.ceil {
            rsign = -rsign;
        }
        residual = gap;
        terms.push((out_sign, t));
    }

    (terms, rsign, residual)
}

// ---------------------------------------------------------------------------
// Verification + formatting
// ---------------------------------------------------------------------------

/// Replay the equation telescope against the seed exactly.
fn verify_equation(seed: &BigUint, terms: &[(i8, Term)], magic_sign: i8, magic: &BigUint) -> bool {
    let mut r = seed.clone();
    let mut s: i8 = 1;
    for (_sigma, t) in terms {
        let val = &t.coef * BigUint::from(t.base).pow(t.exp);
        if t.ceil {
            if val < r {
                return false;
            }
            r = val - r;
            s = -s;
        } else {
            if val > r {
                return false;
            }
            r = r - val;
        }
    }
    r == *magic && s == magic_sign
}

fn fmt_term(t: &Term) -> String {
    if t.coef == BigUint::from(1u64) {
        format!("{}^{}", t.base, t.exp)
    } else {
        format!("{}·{}^{}", t.coef, t.base, t.exp)
    }
}

fn fmt_equation(terms: &[(i8, Term)], magic_sign: i8, magic: &BigUint) -> String {
    let mut out = String::from("seed = ");
    for (i, (sign, t)) in terms.iter().enumerate() {
        if *sign > 0 {
            if i > 0 {
                out.push_str(" + ");
            }
        } else {
            out.push_str(" - ");
        }
        out.push_str(&fmt_term(t));
    }
    if !magic.is_zero() {
        if magic_sign < 0 {
            out.push_str(" - ");
        } else {
            out.push_str(" + ");
        }
        out.push_str(&magic.to_string());
    }
    out
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of a solve: the formatted equation plus the raw correction.
pub struct SolverResult {
    pub equation: String,
    pub magic: BigUint,
    pub magic_sign: i8,
    pub terms: usize,
    pub ok: bool,
}

/// Find a precise `seed = Σ σᵢ·cᵢ·nᵢ^eᵢ + magic` for the given seed.
///
/// The coefficient budget auto-sizes so the seed folds into at most `max_terms`
/// terms with a magic small enough to fit a u32. Diagnostics (the single power
/// sweep and per-step peeling) are written to stderr when `verbose`.
pub fn solve(seed: &BigUint, max_terms: usize, max_base: u64, verbose: bool) -> SolverResult {
    let max_base = max_base.max(2);
    // Each term's coefficient must hold enough bits to cover the seed in
    // `max_terms` steps. The +32 margin absorbs per-step fit losses so the
    // magic always lands within a u32 (usually zero).
    let target = (seed.bits() as usize + max_terms - 1) / max_terms + 32;
    let coef_bits = (COEF_BITS as usize).max(target).min(u32::MAX as usize) as u32;
    let caches: Vec<Mutex<PowCache>> = (0..=max_base).map(|b| Mutex::new(PowCache::new(b))).collect();

    // ---- [1] single-power sweep (diagnostic) ------------------------------
    if verbose {
        eprintln!(
            "[1] single-power sweep — best {} bases out of 2..={} by efficiency (coef budget {})",
            TOP_EXACT, max_base, coef_bits
        );
        eprintln!("  rank  base    exp   coef_digits  est-removed  exact-removed  term");
        for (i, r) in rank_bases(seed, max_base, TOP_EXACT, coef_bits)
            .iter()
            .enumerate()
        {
            let mut cache = caches[r.base as usize].lock().unwrap();
            let (t, gap) = best_fit_cached(seed, r.base, &mut cache, coef_bits);
            let exact_removed = seed.bits().saturating_sub(gap.bits()) as f64;
            eprintln!(
                "  {:>4}  {:>4}  {:>6}  {:>11}  {:>11.1}  {:>13.1}  {}",
                i + 1,
                r.base,
                r.e,
                r.a_digits,
                r.score_est,
                exact_removed,
                fmt_term(&t)
            );
        }
    }

    // ---- [2] greedy multi-term --------------------------------------------
    if verbose {
        eprintln!();
        eprintln!(
            "[2] greedy multi-term — peel the closest c·n^e off the residual ({} steps)",
            max_terms
        );
    }
    let (terms, magic_sign, magic) = greedy(seed, &caches, max_terms, max_base, coef_bits, verbose);

    // ---- [3] verify + format ----------------------------------------------
    let ok = verify_equation(seed, &terms, magic_sign, &magic);
    let equation = fmt_equation(&terms, magic_sign, &magic);
    if verbose {
        eprintln!();
        eprintln!("[3] final equation ({} terms + magic)", terms.len());
    }

    SolverResult {
        equation,
        magic,
        magic_sign,
        terms: terms.len(),
        ok,
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    fn big(x: u128) -> BigUint {
        BigUint::from(x)
    }

    #[test]
    fn approx_log2_is_accurate() {
        // Powers of two must be exact.
        for k in 0..300u64 {
            let x = BigUint::from(1u64) << k;
            let l = approx_log2(&x);
            assert!((l - k as f64).abs() < 1e-9, "k={k} l={l}");
        }
        // Small integers against f64::log2.
        for x in 1..2000u64 {
            let l = approx_log2(&big(x as u128));
            let expect = (x as f64).log2();
            assert!((l - expect).abs() < 1e-9, "x={x} l={l}");
        }
    }

    #[test]
    fn fast_pow_matches_biguint_pow() {
        for base in 2u64..=200 {
            for e in 0u32..=30 {
                assert_eq!(
                    fast_pow(base, e),
                    BigUint::from(base).pow(e),
                    "base={base} e={e}"
                );
            }
        }
    }

    #[test]
    fn pow_cache_is_consistent() {
        for base in 2u64..=64 {
            let mut c = PowCache::new(base);
            for &v in &[1u64, 2, 7, 63, 64, 100, 999, 1_000_000, 1_000_000_000] {
                let v = big(v as u128);
                c.seed(&v);
                assert!(&c.p <= &v, "base={base} v={v}");
                assert!(&c.p * BigUint::from(base) > v, "base={base} v={v}");
                assert_eq!(c.p, fast_pow(c.base, c.e), "base={base}");
            }
        }
    }


    #[test]
    fn best_fit_is_consistent() {
        // The returned gap must always equal |value − term| exactly, and the
        // coefficient must stay within the configured bit budget.
        for base in 2u64..=20 {
            let mut c = PowCache::new(base);
            for v in 1u64..=1000 {
                let val = big(v as u128);
                let (t, gap) = best_fit_cached(&val, base, &mut c, 10);
                // gap must equal |val - term|
                let term = &t.coef * BigUint::from(base).pow(t.exp);
                let expect = if term > val { term - &val } else { &val - term };
                assert_eq!(gap, expect, "base={base} v={v}");
                // coefficient must fit the budget (10 bits + one base limb)
                assert!(
                    t.coef.bits() <= 10 + (base as f64).log2() as u64 + 1,
                    "base={base} v={v} coef too big: {} bits",
                    t.coef.bits()
                );
            }
        }
    }

    #[test]
    fn greedy_zeroes_seed_and_verifies() {
        let seeds = [
            big(42),
            big(123_456_789),
            big(1_776_131_265_209_184_858_631_281_479_792_477_464), // user's small example
            BigUint::from(u64::MAX) * BigUint::from(u64::MAX),      // 128-bit
            (BigUint::from(1u64) << 150) + big(999),                // 151-bit
        ];
        for (i, seed) in seeds.iter().enumerate() {
            for &max_base in &[99u64, 1024, 65536] {
                let caches: Vec<Mutex<PowCache>> = (0..=max_base)
                    .map(|b| Mutex::new(PowCache::new(b)))
                    .collect();
                let (terms, magic_sign, magic) = greedy(seed, &caches, 4000, max_base, COEF_BITS, false);
                assert!(
                    magic.is_zero(),
                    "seed {i} base {max_base}: magic left with {} bits",
                    magic.bits()
                );
                assert!(
                    verify_equation(seed, &terms, magic_sign, &magic),
                    "seed {i} base {max_base}: replay failed"
                );
            }
        }
    }

    #[test]
    fn higher_bases_cannot_hurt_best_score() {
        // More candidate bases = superset, so the best single-term efficiency
        // can only stay equal or improve.
        let seed = (BigUint::from(1u64) << 500) + big(0xDEAD_BEEF);
        let s99 = rank_bases(&seed, 99, 1, COEF_BITS)[0].score_est;
        let s1024 = rank_bases(&seed, 1024, 1, COEF_BITS)[0].score_est;
        let s65536 = rank_bases(&seed, 65536, 1, COEF_BITS)[0].score_est;
        assert!(s1024 >= s99 - 1e-9, "1024 should be >= 99: {s1024} vs {s99}");
        assert!(
            s65536 >= s1024 - 1e-9,
            "65536 should be >= 1024: {s65536} vs {s1024}"
        );
    }

    #[test]
    fn rank_bases_returns_sorted_top_k() {
        let seed = (BigUint::from(1u64) << 500) + big(12_345);
        let r = rank_bases(&seed, 1024, 10, COEF_BITS);
        assert_eq!(r.len(), 10, "must fill k slots");
        for w in r.windows(2) {
            assert!(
                w[0].score_est >= w[1].score_est,
                "must be sorted by score descending"
            );
        }
        assert!(r.iter().all(|x| x.base >= 2 && x.base <= 1024));
    }

    #[test]
    fn solve_returns_verified_equation() {
        let seed = big(1_776_131_265_209_184_858_631_281_479_792_477_464);
        let r = solve(&seed, 200, 1024, false);
        assert!(r.ok, "verify must pass");
        assert!(r.magic.is_zero(), "small seed should be fully consumed");
        assert!(r.equation.starts_with("seed = "));
        // magic == 0 means no magic tail is printed at all.
        assert!(!r.equation.contains(" + 0") && !r.equation.contains(" - 0"));
    }

    #[test]
    fn solve_respects_max_terms_cap() {
        // A pseudo-random 400-bit seed with max_terms=5: the coefficient budget
        // auto-sizes so 5 terms fold it, leaving a magic that fits a u32.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut seed = BigUint::zero();
        for _ in 0..7 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            seed = (seed << 64) | BigUint::from(state);
        }
        let mask = (BigUint::from(1u64) << 400) - BigUint::from(1u64);
        seed &= mask;
        assert_eq!(seed.bits(), 400, "seed should be a full 400-bit number");
        let r = solve(&seed, 5, 1024, false);
        assert!(r.ok, "verify must pass");
        assert!(r.terms <= 5, "must fold into at most 5 terms, got {}", r.terms);
        assert!(
            r.magic.bits() <= 32,
            "magic must fit a u32, got {} bits",
            r.magic.bits()
        );
    }

    #[test]
    fn folds_into_24_terms_with_u32_magic() {
        // The headline contract: any seed folds into <= 24 terms whose magic
        // fits a u32, and the equation verifies exactly.
        let mut state = 0xdead_beef_cafe_f00du64;
        let mut seed = BigUint::zero();
        for _ in 0..64 {
            state = state
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            seed = (seed << 32) | BigUint::from(state & 0xFFFF_FFFF);
        }
        // ~2048-bit seed (force the top bit so the length is guaranteed)
        seed |= BigUint::from(1u64) << 2047;
        assert_eq!(seed.bits(), 2048, "seed should be a full 2048-bit number");
        let r = solve(&seed, 24, 65536, false);
        assert!(r.ok, "verify must pass");
        assert!(r.terms <= 24, "must fold into at most 24 terms, got {}", r.terms);
        assert!(
            r.magic.bits() <= 32,
            "magic must fit a u32, got {} bits",
            r.magic.bits()
        );
    }
}

