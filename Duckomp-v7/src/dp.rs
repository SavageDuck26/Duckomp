//! Duckomp-v7 optimal chunking (module, used by main.rs)
//!
//! Cuts the gap stream into consecutive chunks, each packed at its own
//! fixed-radix base, choosing the cut points that minimise total bits:
//!
//!     total = Σ chunk_payload_bits + CHUNK_HEADER_BITS · chunks
//!
//! Solved exactly by dynamic programming over prefix costs:
//!
//!     best[0] = 0
//!     best[j] = min over i<j of ( best[i] + payload(gaps[i..j]) + header )
//!
//! Since "one chunk" (i = 0) is always an option, the result is never worse
//! than a single global base — on a stationary stream it just stays one chunk.

/// Bits charged per chunk for its stored base and length.
pub const CHUNK_HEADER_BITS: u64 = 16;

/// One chunk of the gap stream: the half-open range `[start, end)`.
pub struct Chunk {
    pub start: usize,
    pub end: usize,
}

/// The chosen partition of the gap stream plus its payload cost.
pub struct ChunkPlan {
    pub chunks: Vec<Chunk>,
    pub payload_bits: f64,
}

impl ChunkPlan {
    /// Payload bits plus one header per chunk.
    pub fn total_bits(&self) -> f64 {
        self.payload_bits + CHUNK_HEADER_BITS as f64 * self.chunks.len() as f64
    }
}

/// Fixed-radix payload bits for `gaps[start..end]`.
fn chunk_payload_bits(gaps: &[u128], start: usize, end: usize) -> f64 {
    let seg = &gaps[start..end];
    let lo = *seg.iter().min().expect("non-empty chunk");
    let hi = *seg.iter().max().expect("non-empty chunk");
    seg.len() as f64 * ((hi - lo + 1) as f64).log2()
}

/// Exact optimal partition of the gap stream (dynamic programming, O(n²)).
pub fn plan(gaps: &[u128]) -> ChunkPlan {
    let n = gaps.len();
    let mut best = vec![f64::INFINITY; n + 1];
    let mut cut = vec![0usize; n + 1];
    best[0] = 0.0;

    for j in 1..=n {
        // Grow the candidate last chunk backwards from j, tracking its running
        // min/max so payload(i..j) costs O(1) per step.
        let mut lo = u128::MAX;
        let mut hi = u128::MIN;
        for i in (0..j).rev() {
            lo = lo.min(gaps[i]);
            hi = hi.max(gaps[i]);
            let payload = (j - i) as f64 * ((hi - lo + 1) as f64).log2();
            let cost = best[i] + payload + CHUNK_HEADER_BITS as f64;
            if cost < best[j] {
                best[j] = cost;
                cut[j] = i;
            }
        }
    }

    // Walk the cuts backwards, then flip into start..end order.
    let mut chunks = Vec::new();
    let mut j = n;
    while j > 0 {
        let i = cut[j];
        chunks.push(Chunk { start: i, end: j });
        j = i;
    }
    chunks.reverse();

    let payload_bits = chunks
        .iter()
        .map(|c| chunk_payload_bits(gaps, c.start, c.end))
        .sum();
    ChunkPlan {
        chunks,
        payload_bits,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn one_chunk_payload(g: &[u128]) -> f64 {
        let lo = *g.iter().min().unwrap();
        let hi = *g.iter().max().unwrap();
        g.len() as f64 * ((hi - lo + 1) as f64).log2()
    }

    #[test]
    fn flat_stream_stays_one_chunk() {
        let p = plan(&[7u128, 7, 7, 7, 7]);
        assert_eq!(p.chunks.len(), 1);
        assert_eq!(p.payload_bits, 0.0); // radix 1 -> 0 bits per gap
    }

    #[test]
    fn two_regimes_split_and_beat_one_chunk() {
        let mut g = vec![2u128; 50];
        g.extend(std::iter::repeat(900u128).take(50));
        let p = plan(&g);
        assert!(p.chunks.len() >= 2, "should split at the regime change");
        let one = one_chunk_payload(&g) + CHUNK_HEADER_BITS as f64;
        assert!(p.total_bits() < one, "{} vs {one}", p.total_bits());
    }

    #[test]
    fn never_worse_than_single_chunk() {
        let g: Vec<u128> = (0..200u128).map(|i| (i * 37 + 11) % 91 + 5).collect();
        let p = plan(&g);
        let one = one_chunk_payload(&g) + CHUNK_HEADER_BITS as f64;
        assert!(p.total_bits() <= one + 1e-9, "{} vs {one}", p.total_bits());
    }

    #[test]
    fn chunks_cover_the_stream_in_order() {
        let g: Vec<u128> = vec![1, 1, 1, 50, 50, 50, 2, 2, 2];
        let p = plan(&g);
        assert_eq!(p.chunks.first().unwrap().start, 0);
        assert_eq!(p.chunks.last().unwrap().end, g.len());
        for w in p.chunks.windows(2) {
            assert_eq!(w[0].end, w[1].start, "chunks must be contiguous");
        }
    }

    #[test]
    fn every_gap_is_covered_exactly_once() {
        let g: Vec<u128> = vec![3, 9, 9, 1, 4, 4, 4, 2, 8, 8, 8, 8, 7];
        let p = plan(&g);
        let covered: usize = p.chunks.iter().map(|c| c.end - c.start).sum();
        assert_eq!(covered, g.len());
    }

    #[test]
    fn empty_stream_is_safe() {
        let p = plan(&[]);
        assert_eq!(p.chunks.len(), 0);
        assert_eq!(p.total_bits(), 0.0);
    }
}
