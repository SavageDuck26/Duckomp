//! Duckomp-v7 packed view (module, used by main.rs)
//!
//! Measurement helpers for the PoC: what each representation of the offset data
//! costs. All display/formatting lives in main.rs.

use num_bigint::BigUint;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

/// Order-0 Shannon entropy (bits per symbol) of a symbol stream.
pub fn entropy_bits(symbols: &[u128]) -> f64 {
    if symbols.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<u128, usize> = HashMap::new();
    for &s in symbols {
        *counts.entry(s).or_insert(0) += 1;
    }
    let n = symbols.len() as f64;
    -counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            p * p.log2()
        })
        .sum::<f64>()
}

/// Total Huffman cost (bits) for a symbol multiset. A stream with a single
/// distinct symbol costs 0 bits — its value is already known.
pub fn huffman_bits(symbols: &[u128]) -> u128 {
    if symbols.is_empty() {
        return 0;
    }
    let mut counts: HashMap<u128, usize> = HashMap::new();
    for &s in symbols {
        *counts.entry(s).or_insert(0) += 1;
    }
    if counts.len() == 1 {
        return 0;
    }
    let mut heap: BinaryHeap<Reverse<u128>> =
        counts.values().map(|&c| Reverse(c as u128)).collect();
    let mut total: u128 = 0;
    while heap.len() > 1 {
        let Reverse(a) = heap.pop().unwrap();
        let Reverse(b) = heap.pop().unwrap();
        total += a + b;
        heap.push(Reverse(a + b));
    }
    total
}

/// Byte length of a big integer's packed (big-endian) representation. Zero
/// still occupies one byte so the length is self-describing.
pub fn packed_len(seed: &BigUint) -> usize {
    let bits = seed.bits();
    if bits == 0 {
        1
    } else {
        ((bits + 7) / 8) as usize
    }
}

/// Lowercase hex of a big integer's packed bytes (full, big-endian).
pub fn packed_hex(seed: &BigUint) -> String {
    let bytes = if seed.bits() == 0 {
        vec![0u8]
    } else {
        seed.to_bytes_be()
    };
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Header bits for a self-describing pack: the base/radix and the length.
pub const PACK_HEADER_BITS: u64 = 16;

/// Bits to store a Huffman codebook for `symbols`: every distinct symbol's
/// value plus a 4-bit code length.
pub fn huffman_codebook_bits(symbols: &[u128]) -> u128 {
    if symbols.is_empty() {
        return 0;
    }
    let mut counts: HashMap<u128, usize> = HashMap::new();
    for &s in symbols {
        *counts.entry(s).or_insert(0) += 1;
    }
    let max = *symbols.iter().max().unwrap();
    let value_bits = (128 - max.leading_zeros()) as u128;
    counts.len() as u128 * (value_bits + 4)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_uniform_binary_is_one() {
        assert!((entropy_bits(&[0, 1]) - 1.0).abs() < 1e-12);
        assert!((entropy_bits(&[0, 1, 0, 1]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_single_symbol_is_zero() {
        assert_eq!(entropy_bits(&[7, 7, 7, 7]), 0.0);
        assert_eq!(entropy_bits(&[]), 0.0);
    }

    #[test]
    fn huffman_single_symbol_costs_nothing() {
        assert_eq!(huffman_bits(&[5, 5, 5, 5]), 0);
    }

    #[test]
    fn huffman_four_uniform_symbols_is_two_bits_each() {
        // merge weights: 1+1=2, 1+1=2, 2+2=4 -> total weighted path = 8.
        assert_eq!(huffman_bits(&[1, 2, 3, 4]), 8);
    }

    #[test]
    fn huffman_never_beats_entropy_floor() {
        let syms = [0u128, 0, 0, 1, 1, 2, 2, 2, 2, 2];
        let floor = entropy_bits(&syms) * syms.len() as f64;
        assert!(huffman_bits(&syms) as f64 >= floor - 1e-9);
    }

    #[test]
    fn packed_len_and_hex_roundtrip() {
        let seed = BigUint::from(4686045276332411031518314426086636103u128);
        assert_eq!(packed_len(&seed), 16);
        let hex = packed_hex(&seed);
        assert_eq!(BigUint::parse_bytes(hex.as_bytes(), 16).unwrap(), seed);
        assert_eq!(packed_len(&BigUint::from(0u64)), 1);
    }

    #[test]
    fn codebook_bits_counts_values_and_lengths() {
        // 3 distinct values, each <= 3 (2 bits) + a 4-bit length = 6 bits each.
        assert_eq!(huffman_codebook_bits(&[1u128, 2, 3, 1, 2, 3]), 3 * (2 + 4));
        assert_eq!(huffman_codebook_bits(&[]), 0);
    }
}
