//! Duckomp-v8 grid rules (module, used by main.rs)
//!
//! The gap stream is laid out as a W-wide grid and an invertible "rule"
//! mutates every cell:
//!
//!     grid --forward--> mutated grid --inverse--> grid
//!
//! Nothing is deleted and nothing is solved: exactness is a round-trip
//! property (`inverse(forward(x)) == x`), checked in the tests below and again
//! in every report. The mutated stream is then scored with the same optimal
//! chunker v7 uses (dp.rs), so a rule only earns its bytes if it makes the
//! data cheaper to pack.
//!
//! Rules (all total and invertible):
//!   identity   — control, no mutation
//!   row_delta  — diff each row left-to-right (predictive)
//!   col_delta  — diff each column top-to-bottom (predictive)
//!   haar1d     — integer Haar lifting per row, multi-level
//!   haar2d     — Haar lifting over rows then columns
//!   row_rotate — per-row cyclic shift fitted to the previous row
//!
//! The Haar steps are integer-exact (`d = b - a`, `s = a + (d >> 1)`), the
//! same reversible lifting JPEG2000/FLAC use, so the inverse never searches
//! and never needs the original.

/// One grid cell. Gaps arrive as u128; the transforms need signed values.
pub type Cell = i128;

/// Bits charged for the fixed part of any rule's parameters (4-bit rule id,
/// 8-bit width, 3-bit Haar depth).
pub const RULE_HEADER_BITS: f64 = 4.0 + 8.0 + 3.0;

/// The rule families the sweep tries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleKind {
    Identity,
    RowDelta,
    ColDelta,
    Haar1d,
    Haar2d,
    RowRotate,
}

impl RuleKind {
    /// Every rule, in report order.
    pub const ALL: [RuleKind; 6] = [
        RuleKind::Identity,
        RuleKind::RowDelta,
        RuleKind::ColDelta,
        RuleKind::Haar1d,
        RuleKind::Haar2d,
        RuleKind::RowRotate,
    ];

    /// Human-readable name for the report (no abbreviations).
    pub fn display(self) -> &'static str {
        match self {
            RuleKind::Identity => "none",
            RuleKind::RowDelta => "row differences",
            RuleKind::ColDelta => "column differences",
            RuleKind::Haar1d => "1D Haar lifting",
            RuleKind::Haar2d => "2D Haar lifting",
            RuleKind::RowRotate => "row rotation",
        }
    }
}

/// A W-wide, row-major grid over the gap stream (the last row may be ragged).
pub struct Grid {
    pub w: usize,
    pub cells: Vec<Cell>,
}

/// A grid after a rule has been applied, plus whatever the inverse needs.
pub struct Fitted {
    pub kind: RuleKind,
    pub levels: u32,
    /// Cyclic shift per full row (empty unless `kind == RowRotate`).
    pub shifts: Vec<u32>,
    pub cells: Vec<Cell>,
}

impl Grid {
    pub fn from_stream(stream: &[Cell], w: usize) -> Grid {
        assert!(w > 0, "grid width must be positive");
        Grid {
            w,
            cells: stream.to_vec(),
        }
    }

    /// Number of complete rows (the ones column rules act on).
    pub fn full_rows(&self) -> usize {
        self.cells.len() / self.w
    }

    /// Global fixed-radix payload bits (radix = max - min + 1).
    pub fn fixed_bits(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let lo = *self.cells.iter().min().unwrap();
        let hi = *self.cells.iter().max().unwrap();
        (self.cells.len() as f64) * ((hi - lo + 1) as f64).log2()
    }

    /// Per-row fixed-radix payload bits, plus one 16-bit header per row.
    pub fn row_bits(&self) -> f64 {
        self.cells
            .chunks(self.w)
            .map(|row| {
                let lo = *row.iter().min().unwrap();
                let hi = *row.iter().max().unwrap();
                (row.len() as f64) * ((hi - lo + 1) as f64).log2() + 16.0
            })
            .sum()
    }

    /// Apply `kind` to a copy of the grid.
    pub fn fit(&self, kind: RuleKind, levels: u32) -> Fitted {
        let mut cells = self.cells.clone();
        let mut shifts = Vec::new();
        match kind {
            RuleKind::Identity => {}
            RuleKind::RowDelta => {
                for row in cells.chunks_mut(self.w) {
                    row_delta(row);
                }
            }
            RuleKind::ColDelta => col_delta(&mut cells, self.w, self.full_rows()),
            RuleKind::Haar1d => {
                for row in cells.chunks_mut(self.w) {
                    haar_forward_row(row, levels);
                }
            }
            RuleKind::Haar2d => {
                for row in cells.chunks_mut(self.w) {
                    haar_forward_row(row, levels);
                }
                haar_columns(&mut cells, self.w, self.full_rows(), levels, true);
            }
            RuleKind::RowRotate => {
                shifts = fit_rotations(&mut cells, self.w, self.full_rows());
            }
        }
        Fitted {
            kind,
            levels,
            shifts,
            cells,
        }
    }
}

impl Fitted {
    /// Undo the rule exactly; no search, no original needed.
    pub fn inverse(&self, w: usize) -> Grid {
        let mut cells = self.cells.clone();
        let full = cells.len() / w;
        match self.kind {
            RuleKind::Identity => {}
            RuleKind::RowDelta => {
                for row in cells.chunks_mut(w) {
                    row_undelta(row);
                }
            }
            RuleKind::ColDelta => col_undelta(&mut cells, w, full),
            RuleKind::Haar1d => {
                for row in cells.chunks_mut(w) {
                    haar_inverse_row(row, self.levels);
                }
            }
            RuleKind::Haar2d => {
                haar_columns(&mut cells, w, full, self.levels, false);
                for row in cells.chunks_mut(w) {
                    haar_inverse_row(row, self.levels);
                }
            }
            RuleKind::RowRotate => unrotate_rows(&mut cells, w, full, &self.shifts),
        }
        Grid { w, cells }
    }

    /// Bits to store the rule's parameters (header + fitted shifts).
    pub fn params_bits(&self, w: usize) -> f64 {
        let mut bits = RULE_HEADER_BITS;
        if self.kind == RuleKind::RowRotate {
            bits += self.shifts.len() as f64 * (w as f64).log2().ceil();
        }
        bits
    }
}

// ---------------------------------------------------------------------------
// Rule steps (each pair is an exact inverse)
// ---------------------------------------------------------------------------

fn row_delta(row: &mut [Cell]) {
    for j in (1..row.len()).rev() {
        row[j] -= row[j - 1];
    }
}

fn row_undelta(row: &mut [Cell]) {
    for j in 1..row.len() {
        row[j] += row[j - 1];
    }
}

fn col_delta(cells: &mut [Cell], w: usize, rows: usize) {
    if rows == 0 {
        return;
    }
    for col in 0..w {
        let mut prev = cells[col];
        for r in 1..rows {
            let i = r * w + col;
            let cur = cells[i];
            cells[i] = cur - prev;
            prev = cur;
        }
    }
}

fn col_undelta(cells: &mut [Cell], w: usize, rows: usize) {
    for col in 0..w {
        for r in 1..rows {
            let i = r * w + col;
            cells[i] += cells[i - w];
        }
    }
}

/// In-place multi-level integer Haar lifting on one row.
fn haar_forward_row(row: &mut [Cell], levels: u32) {
    let mut size = row.len();
    for _ in 0..levels {
        if size < 2 {
            break;
        }
        let half = size / 2;
        let s_len = size - half;
        let tmp = row[..size].to_vec();
        for k in 0..half {
            let a = tmp[2 * k];
            let b = tmp[2 * k + 1];
            let d = b - a;
            row[k] = a + (d >> 1);
            row[s_len + k] = d;
        }
        if size % 2 == 1 {
            row[s_len - 1] = tmp[size - 1];
        }
        size = s_len;
    }
}

/// Exact inverse of `haar_forward_row`.
fn haar_inverse_row(row: &mut [Cell], levels: u32) {
    let mut sizes = vec![row.len()];
    let mut size = row.len();
    for _ in 0..levels {
        if size < 2 {
            break;
        }
        size = size.div_ceil(2);
        sizes.push(size);
    }
    for lv in (0..sizes.len().saturating_sub(1)).rev() {
        let size = sizes[lv];
        let half = size / 2;
        let s_len = size - half;
        let tmp = row[..size].to_vec();
        for k in 0..half {
            let s = tmp[k];
            let d = tmp[s_len + k];
            let a = s - (d >> 1);
            row[2 * k] = a;
            row[2 * k + 1] = d + a;
        }
        if size % 2 == 1 {
            row[size - 1] = tmp[s_len - 1];
        }
    }
}

/// Haar lifting over whole columns (full rows only; ragged tail untouched).
fn haar_columns(cells: &mut [Cell], w: usize, rows: usize, levels: u32, forward: bool) {
    if rows == 0 {
        return;
    }
    let mut col = vec![0 as Cell; rows];
    for c in 0..w {
        for r in 0..rows {
            col[r] = cells[r * w + c];
        }
        if forward {
            haar_forward_row(&mut col, levels);
        } else {
            haar_inverse_row(&mut col, levels);
        }
        for r in 0..rows {
            cells[r * w + c] = col[r];
        }
    }
}

/// Fit one cyclic shift per full row, aligning it to the previous row.
fn fit_rotations(cells: &mut [Cell], w: usize, rows: usize) -> Vec<u32> {
    let orig = cells.to_vec();
    let mut shifts = Vec::with_capacity(rows);
    for r in 0..rows {
        if r == 0 {
            shifts.push(0);
            continue;
        }
        let prev = &orig[(r - 1) * w..r * w];
        let cur = &orig[r * w..(r + 1) * w];
        let mut best_s = 0usize;
        let mut best_hits = 0usize;
        for s in 0..w {
            let hits = (0..w).filter(|&j| cur[(j + s) % w] == prev[j]).count();
            if hits > best_hits {
                best_hits = hits;
                best_s = s;
            }
        }
        shifts.push(best_s as u32);
        let row = &mut cells[r * w..(r + 1) * w];
        let copy = row.to_vec();
        for j in 0..w {
            row[j] = copy[(j + best_s) % w];
        }
    }
    shifts
}

/// Exact inverse of `fit_rotations` (needs the fitted shifts).
fn unrotate_rows(cells: &mut [Cell], w: usize, rows: usize, shifts: &[u32]) {
    for r in 1..rows {
        let s = shifts[r] as usize % w;
        let row = &mut cells[r * w..(r + 1) * w];
        let copy = row.to_vec();
        for j in 0..w {
            row[(j + s) % w] = copy[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_grids() -> Vec<Grid> {
        vec![
            Grid::from_stream(&[5, 1, 9, 3, 7, 2, 8, 4, 6, 0, 11, 13], 3),
            Grid::from_stream(
                &(0..23).map(|i| (i * 37 % 101) as Cell - 50).collect::<Vec<_>>(),
                5,
            ),
            Grid::from_stream(&[7], 4),
            Grid::from_stream(&vec![42i128; 16], 4),
        ]
    }

    #[test]
    fn every_rule_round_trips() {
        for g in sample_grids() {
            for kind in RuleKind::ALL {
                let fitted = g.fit(kind, 3);
                let back = fitted.inverse(g.w);
                assert_eq!(back.cells, g.cells, "{} w={}", kind.display(), g.w);
            }
        }
    }

    #[test]
    fn haar_known_row() {
        let mut row = [1, 2, 3, 4];
        haar_forward_row(&mut row, 3);
        assert_eq!(row, [2, 2, 1, 1]);
        haar_inverse_row(&mut row, 3);
        assert_eq!(row, [1, 2, 3, 4]);
    }

    #[test]
    fn row_delta_known() {
        let mut row = [4, 7, 7, 2];
        row_delta(&mut row);
        assert_eq!(row, [4, 3, 0, -5]);
        row_undelta(&mut row);
        assert_eq!(row, [4, 7, 7, 2]);
    }

    #[test]
    fn constant_grid_has_zero_payload() {
        let g = Grid::from_stream(&vec![9i128; 12], 4);
        assert_eq!(g.fixed_bits(), 0.0);
        assert_eq!(g.row_bits(), 3.0 * 16.0);
        // A constant row deltas to one value followed by zeros.
        let d = g.fit(RuleKind::RowDelta, 3);
        assert_eq!(d.cells, vec![9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0]);
    }

    #[test]
    fn rotate_finds_alignment() {
        let g = Grid::from_stream(&[1, 2, 3, 4, 3, 4, 1, 2], 4);
        let f = g.fit(RuleKind::RowRotate, 3);
        assert_eq!(f.shifts, vec![0, 2]);
        assert_eq!(f.cells, vec![1, 2, 3, 4, 1, 2, 3, 4]);
        assert_eq!(f.inverse(4).cells, g.cells);
    }

    #[test]
    fn ragged_tail_is_untouched_by_column_rules() {
        let g = Grid::from_stream(&[1, 2, 3, 4, 5, 6, 7], 3); // rows: 3, 3, 1
        for kind in [RuleKind::ColDelta, RuleKind::Haar2d] {
            let f = g.fit(kind, 3);
            assert_eq!(
                f.cells[6], 7,
                "{} must leave the ragged cell alone",
                kind.display()
            );
            assert_eq!(f.inverse(3).cells, g.cells);
        }
    }
}
