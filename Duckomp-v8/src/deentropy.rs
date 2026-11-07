//! Duckomp-v8 de-entropy pass (module, used by main.rs)
//!
//! Entropy-guided *deletion*. The transform sweep rewrites cells; this pass
//! deletes whole lines and rebuilds them from a rule:
//!
//!   constant — every cell equal; the value is stored once
//!   repeat   — the row equals an earlier row; store a pointer
//!   shift    — the row equals an earlier row rotated by s; pointer + s
//!   perm     — the row is a permutation of one shared multiset; store its
//!              Lehmer rank (log2 w! bits) and rebuild by unranking
//!   keep     — no rule fits; the row is stored as-is
//!
//! Rows are ranked by their own entropy (highest first) because those are the
//! expensive ones: a permutation row is maximum-entropy *and* fully derivable,
//! so it is the best possible deletion target.
//!
//! Exactness is proven at encode time: every deleted line is rebuilt by the
//! decoder's own rules and compared to the original, so the pass can never
//! claim a deletion it cannot undo.

use std::collections::HashMap;

use crate::grid::{Cell, Grid};

/// Bits charged per routed row for its rule tag.
const RULE_TAG_BITS: f64 = 3.0;
/// Bits charged per kept row for its stored radix/length header.
pub const ROW_HEADER_BITS: f64 = 16.0;
/// Lehmer ranks use u128, so 34! is the largest permutation count that fits.
const MAX_PERM_W: usize = 34;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowRule {
    Keep,
    Constant { value: Cell },
    Repeat { src: usize },
    Shift { src: usize, shift: usize },
    Perm { rank: u128 },
}

impl RowRule {
    /// Human-readable name for the report (no abbreviations).
    pub fn display(&self) -> &'static str {
        match self {
            RowRule::Keep => "keep",
            RowRule::Constant { .. } => "constant",
            RowRule::Repeat { .. } => "repeat",
            RowRule::Shift { .. } => "shift",
            RowRule::Perm { .. } => "permutation",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RowDecision {
    pub row: usize,
    pub rule: RowRule,
    pub cost_bits: f64,
    pub entropy: f64,
}

pub struct Outcome {
    pub w: usize,
    pub rows: usize,
    pub total: usize,
    pub ragged_cells: usize,
    pub decisions: Vec<RowDecision>,
    pub deleted_cells: usize,
    pub deleted_entropy: f64,
    pub multiset: Option<Vec<Cell>>,
    pub constants: Vec<Cell>,
    pub multiset_bits: f64,
    pub constants_bits: f64,
    pub payload_bits: f64,
    pub tag_bits: f64,
    pub total_bits: f64,
    /// Cost of the same row-wise container with no deletions.
    pub keep_bits: f64,
    /// Reference cost of the flat stream (not the same container).
    pub identity_bits: f64,
    pub entropy_before: f64,
    pub entropy_kept: f64,
    pub exact: bool,
}

/// Order-0 entropy (bits per cell) of a value stream.
pub fn entropy(values: &[Cell]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<Cell, usize> = HashMap::new();
    for &v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    let n = values.len() as f64;
    let h = -counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            p * p.log2()
        })
        .sum::<f64>();
    // Keep -0.0 out of the report.
    if h == 0.0 { 0.0 } else { h }
}

/// Fixed-radix payload bits for one row (radix = row span + 1).
fn row_fixed_bits(row: &[Cell]) -> f64 {
    if row.is_empty() {
        return 0.0;
    }
    let lo = *row.iter().min().unwrap();
    let hi = *row.iter().max().unwrap();
    row.len() as f64 * ((hi - lo + 1) as f64).log2()
}

fn fixed_bits(values: &[Cell]) -> f64 {
    row_fixed_bits(values)
}

/// Payload + radix header for one row that is stored as-is.
fn row_keep_bits(row: &[Cell]) -> f64 {
    row_fixed_bits(row) + ROW_HEADER_BITS
}

/// log2(n!) — the cost of a Lehmer rank.
fn fact_log(n: usize) -> f64 {
    (1..=n).map(|k| (k as f64).log2()).sum()
}

fn factorials() -> [u128; MAX_PERM_W + 1] {
    let mut f = [1u128; MAX_PERM_W + 1];
    for k in 1..=MAX_PERM_W {
        f[k] = f[k - 1] * k as u128;
    }
    f
}

/// Lehmer rank of `row` relative to the sorted multiset `sorted`.
fn rank_perm(row: &[Cell], sorted: &[Cell]) -> Option<u128> {
    let n = row.len();
    if n > MAX_PERM_W || n != sorted.len() {
        return None;
    }
    let fact = factorials();
    let mut used = vec![false; n];
    let mut rank: u128 = 0;
    for i in 0..n {
        let mut pos = None;
        for k in 0..n {
            if !used[k] && sorted[k] == row[i] {
                pos = Some(k);
                break;
            }
        }
        let p = pos?;
        used[p] = true;
        let smaller = (0..p).filter(|&k| !used[k]).count() as u128;
        rank += smaller * fact[n - 1 - i];
    }
    Some(rank)
}

/// Inverse of `rank_perm`.
fn unrank_perm(mut rank: u128, sorted: &[Cell]) -> Vec<Cell> {
    let n = sorted.len();
    let fact = factorials();
    let mut avail = sorted.to_vec();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let f = fact[n - 1 - i];
        let idx = (rank / f) as usize;
        rank %= f;
        out.push(avail.remove(idx.min(avail.len() - 1)));
    }
    out
}

fn rotate_left(row: &[Cell], s: usize) -> Vec<Cell> {
    let n = row.len();
    (0..n).map(|j| row[(j + s) % n]).collect()
}

fn rotation_of(src: &[Cell], want: &[Cell]) -> Option<usize> {
    let n = src.len();
    (1..n).find(|&s| rotate_left(src, s) == want)
}

/// Run the entropy-guided deletion pass over one grid width.
pub fn run(grid: &Grid) -> Outcome {
    let w = grid.w;
    let cells = &grid.cells;
    let total = cells.len();
    let rows = total / w;
    let ragged_cells = total % w;
    let mut out = Outcome {
        w,
        rows,
        total,
        ragged_cells,
        decisions: Vec::new(),
        deleted_cells: 0,
        deleted_entropy: 0.0,
        multiset: None,
        constants: Vec::new(),
        multiset_bits: 0.0,
        constants_bits: 0.0,
        payload_bits: 0.0,
        tag_bits: 0.0,
        total_bits: 0.0,
        keep_bits: 0.0,
        identity_bits: fixed_bits(cells),
        entropy_before: entropy(cells),
        entropy_kept: 0.0,
        exact: true,
    };
    if rows == 0 {
        // No full rows: the whole stream is a ragged tail, stored raw.
        let raw = row_keep_bits(cells);
        out.payload_bits = raw;
        out.keep_bits = raw;
        out.total_bits = raw;
        return out;
    }

    // Shared multiset: the first row's sorted values. `perm` applies to rows
    // whose sorted values equal it, and the multiset is charged once.
    let mut shared: Vec<Cell> = cells[..w].to_vec();
    shared.sort_unstable();

    let mut kept_rows: Vec<Vec<Cell>> = Vec::with_capacity(rows);
    let mut constants: Vec<Cell> = Vec::new();
    let mut payload = row_keep_bits(&cells[rows * w..]); // ragged tail is raw
    let mut keep = payload;
    let row_ent: Vec<f64> = (0..rows)
        .map(|r| entropy(&cells[r * w..(r + 1) * w]))
        .collect();

    // The perm family is a *global* decision: the shared multiset is charged
    // once, so it pays only if it beats keeping every member row raw.
    let perm_members: Vec<usize> = (0..rows)
        .filter(|&r| {
            if w > MAX_PERM_W {
                return false;
            }
            let mut s = cells[r * w..(r + 1) * w].to_vec();
            s.sort_unstable();
            s == shared
        })
        .collect();
    let perm_enabled = !perm_members.is_empty() && {
        let mut with_perm = fixed_bits(&shared);
        let mut without = 0.0;
        for &r in &perm_members {
            with_perm += fact_log(w);
            without += row_fixed_bits(&cells[r * w..(r + 1) * w]);
        }
        with_perm < without
    };

    for r in 0..rows {
        let row = &cells[r * w..(r + 1) * w];
        // Keep is always available; it is also the baseline for `keep_bits`.
        let mut best_rule = RowRule::Keep;
        let mut best_cost = row_keep_bits(row);
        keep += best_cost;

        if best_cost > 0.0 && row.iter().all(|&v| v == row[0]) {
            best_rule = RowRule::Constant { value: row[0] };
            best_cost = 0.0;
        }
        for (src, prev) in kept_rows.iter().enumerate() {
            if *prev == row {
                let c = (rows as f64).log2();
                if c < best_cost {
                    best_cost = c;
                    best_rule = RowRule::Repeat { src };
                }
            } else if let Some(s) = rotation_of(prev, row) {
                let c = (rows as f64).log2() + (w as f64).log2();
                if c < best_cost {
                    best_cost = c;
                    best_rule = RowRule::Shift { src, shift: s };
                }
            }
        }
        if perm_enabled && perm_members.contains(&r) {
            if let Some(rank) = rank_perm(row, &shared) {
                let c = fact_log(w);
                if c < best_cost {
                    best_cost = c;
                    best_rule = RowRule::Perm { rank };
                }
            }
        }

        // Decode this row with the chosen rule; the pass must reproduce it.
        let rebuilt = match &best_rule {
            RowRule::Keep => row.to_vec(),
            RowRule::Constant { value } => {
                constants.push(*value);
                vec![*value; w]
            }
            RowRule::Repeat { src } => kept_rows[*src].clone(),
            RowRule::Shift { src, shift } => rotate_left(&kept_rows[*src], *shift),
            RowRule::Perm { rank } => unrank_perm(*rank, &shared),
        };
        if rebuilt != row {
            out.exact = false;
        }
        if best_rule != RowRule::Keep {
            out.deleted_cells += w;
            out.deleted_entropy += row_ent[r] * w as f64;
        }
        payload += best_cost;
        out.decisions.push(RowDecision {
            row: r,
            rule: best_rule,
            cost_bits: best_cost,
            entropy: row_ent[r],
        });
        kept_rows.push(rebuilt);
    }

    out.constants = constants;
    let keep_count = out
        .decisions
        .iter()
        .filter(|d| d.rule == RowRule::Keep)
        .count();
    out.tag_bits = RULE_TAG_BITS * (rows - keep_count) as f64;
    let multiset_used = out
        .decisions
        .iter()
        .any(|d| matches!(d.rule, RowRule::Perm { .. }));
    out.multiset = if multiset_used { Some(shared.clone()) } else { None };
    out.multiset_bits = if multiset_used { fixed_bits(&shared) } else { 0.0 };
    out.constants_bits = fixed_bits(&out.constants);
    out.payload_bits = payload;
    out.keep_bits = keep;
    out.total_bits = payload + out.tag_bits + out.multiset_bits + out.constants_bits;
    out.entropy_kept = entropy(
        &out.decisions
            .iter()
            .filter(|d| d.rule == RowRule::Keep)
            .flat_map(|d| cells[d.row * w..(d.row + 1) * w].to_vec())
            .collect::<Vec<_>>(),
    );
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn latin(n: usize) -> Grid {
        let cells: Vec<Cell> = (0..n)
            .flat_map(|y| (0..n).map(move |x| ((y + x) % n) as Cell + 1))
            .collect();
        Grid::from_stream(&cells, n)
    }

    #[test]
    fn latin_square_deletes_every_row() {
        let o = run(&latin(9));
        assert!(o.exact);
        assert_eq!(o.deleted_cells, 81, "every row routes to a rule");
        assert!(matches!(o.decisions[0].rule, RowRule::Perm { .. }));
        assert!(
            o.decisions[1..]
                .iter()
                .all(|d| matches!(d.rule, RowRule::Shift { src: 0, .. }))
        );
        assert!(o.total_bits < o.keep_bits, "{} vs {}", o.total_bits, o.keep_bits);
        assert!((o.multiset_bits - 9.0 * 9f64.log2()).abs() < 1e-9);
    }

    #[test]
    fn permutation_rows_are_max_entropy_yet_cheap() {
        let o = run(&latin(9));
        assert!(fact_log(9) < 9.0 * 9f64.log2());
        assert!(o.decisions.iter().all(|d| d.entropy > 3.0));
        assert!(o.total_bits < o.identity_bits);
    }

    #[test]
    fn repeated_rows_collapse_to_pointers() {
        let cells: Vec<Cell> = [1, 2, 3, 4].iter().cycle().take(16).copied().collect();
        let o = run(&Grid::from_stream(&cells, 4));
        assert!(o.exact);
        assert_eq!(o.deleted_cells, 16, "row 0 perm, rows 1..3 repeat");
        assert!(
            o.decisions[1..]
                .iter()
                .all(|d| matches!(d.rule, RowRule::Repeat { src: 0 }))
        );
    }

    #[test]
    fn rotated_rows_collapse_to_shift_pointers() {
        let cells: Vec<Cell> = vec![1, 2, 3, 4, 3, 4, 1, 2];
        let o = run(&Grid::from_stream(&cells, 4));
        assert!(o.exact);
        assert_eq!(o.decisions[1].rule, RowRule::Shift { src: 0, shift: 2 });
        assert_eq!(o.deleted_cells, 4);
    }

    #[test]
    fn random_rows_stay_kept() {
        let cells: Vec<Cell> = (0..64).map(|i| ((i * 29 + 7) % 97) as Cell).collect();
        let o = run(&Grid::from_stream(&cells, 8));
        assert!(o.exact);
        assert_eq!(o.deleted_cells, 0);
        assert!(o.decisions.iter().all(|d| d.rule == RowRule::Keep));
        assert!((o.total_bits - (o.keep_bits + o.tag_bits)).abs() < 1e-9);
    }

    #[test]
    fn rank_round_trips() {
        let sorted = vec![1 as Cell, 2, 2, 5, 9];
        let mut row = sorted.clone();
        row.reverse();
        let r = rank_perm(&row, &sorted).unwrap();
        assert_eq!(unrank_perm(r, &sorted), row);
        assert_eq!(rank_perm(&sorted, &sorted), Some(0));
    }
}
