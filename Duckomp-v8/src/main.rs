//! Duckomp-v8 — grid-rule PoC driver.
//!
//! Reads offsets from sample.txt, turns them into gaps, lays the gaps out as a
//! W-wide grid, and sweeps {width × rule}. Every rule is invertible, so
//! exactness is a round-trip check (never a search), and every candidate
//! stream is scored with the same optimal chunker v7 uses (dp.rs).
//!
//! The report ranks rule/width pairs by DP bytes + rule-parameter bytes and
//! compares the winner against the identity baseline.

mod deentropy;
mod dp;
mod grid;

use grid::{Grid, RuleKind};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};

// ---------------------------------------------------------------------------
// Tunables — edit these directly and rebuild.
// ---------------------------------------------------------------------------
/// Widths tried in the grid sweep (cells per row).
const WIDTHS: [usize; 16] = [
    2, 3, 4, 5, 6, 8, 9, 12, 16, 20, 24, 32, 48, 64, 96, 128,
];
/// Haar lifting depth used by haar1d / haar2d.
const HAAR_LEVELS: u32 = 3;
/// DP is O(n²); run it on this many best row-cost finalists (identity included).
const DP_FINALISTS: usize = 6;

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

/// One (width, rule) candidate after forward-transforming the gap grid.
struct Candidate {
    w: usize,
    kind: RuleKind,
    fixed_bits: f64,
    row_bits: f64,
    params_bits: f64,
    dp_bits: Option<f64>,
    exact: bool,
    cells: Vec<grid::Cell>,
}

/// DP total for a signed stream: shift by the minimum (radix costs are
/// shift-invariant), then run the u128 chunker.
fn dp_total(cells: &[grid::Cell]) -> f64 {
    let min = *cells.iter().min().expect("non-empty stream");
    let shifted: Vec<u128> = cells.iter().map(|&v| (v - min) as u128).collect();
    dp::plan(&shifted).total_bits()
}

fn main() -> io::Result<()> {
    let data = read_data("sample.txt")?;
    if data.len() < 2 {
        println!("Need at least 2 offsets in sample.txt.");
        return Ok(());
    }

    // Adjacent offsets -> gaps, plus the low/high bounds.
    let mut low = u128::MAX;
    let mut high = u128::MIN;
    let mut diffs = Vec::with_capacity(data.len() - 1);
    for pair in data.windows(2) {
        let diff = pair[0].abs_diff(pair[1]);
        diffs.push(diff);
        low = low.min(diff);
        high = high.max(diff);
    }

    let gaps: Vec<grid::Cell> = diffs.iter().map(|&d| d as grid::Cell).collect();

    // Forward-transform every (width, rule) pair and prove it round-trips.
    let mut cands: Vec<Candidate> = Vec::with_capacity(WIDTHS.len() * RuleKind::ALL.len());
    for &w in &WIDTHS {
        let base = Grid::from_stream(&gaps, w);
        for kind in RuleKind::ALL {
            let fitted = base.fit(kind, HAAR_LEVELS);
            let exact = fitted.inverse(w).cells == base.cells;
            let transformed = Grid::from_stream(&fitted.cells, w);
            cands.push(Candidate {
                w,
                kind,
                fixed_bits: transformed.fixed_bits(),
                row_bits: transformed.row_bits(),
                params_bits: fitted.params_bits(w),
                dp_bits: None,
                exact,
                cells: fitted.cells,
            });
        }
    }

    // DP is O(n²): score the best row-cost finalists plus an identity control.
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by(|&a, &b| cands[a].row_bits.total_cmp(&cands[b].row_bits));
    let mut scored: Vec<usize> = order.iter().copied().take(DP_FINALISTS).collect();
    if let Some(id) = order
        .iter()
        .copied()
        .find(|&i| cands[i].kind == RuleKind::Identity)
    {
        if !scored.contains(&id) {
            scored.push(id);
        }
    }
    for &i in &scored {
        cands[i].dp_bits = Some(dp_total(&cands[i].cells));
    }

    // Winner = smallest DP + params; identity is always in `scored`, so the
    // baseline is always in the comparison.
    let total = |i: usize| cands[i].dp_bits.unwrap() + cands[i].params_bits;
    let winner = scored
        .iter()
        .copied()
        .min_by(|&a, &b| total(a).total_cmp(&total(b)))
        .expect("at least one scored candidate");
    let identity = scored
        .iter()
        .copied()
        .find(|&i| cands[i].kind == RuleKind::Identity)
        .expect("identity is always scored");

    // De-entropy pass: delete lines a rule can rebuild; best width wins.
    let pass = WIDTHS
        .iter()
        .map(|&w| deentropy::run(&Grid::from_stream(&gaps, w)))
        .min_by(|a, b| a.total_bits.total_cmp(&b.total_bits))
        .expect("at least one width");

    let out = build_output(
        &data, &diffs, low, high, &cands, &scored, winner, identity, &pass,
    );
    print!("{out}");
    fs::create_dir_all("output")?;
    fs::write("output/PCS.txt", &out)?;
    if cands[winner].kind == RuleKind::Identity {
        eprintln!("(written to output/PCS.txt; no rule beat the flat baseline)");
    } else {
        eprintln!(
            "(written to output/PCS.txt; winner {} at width {}, {:.0} bytes)",
            cands[winner].kind.display(),
            cands[winner].w,
            total(winner) / 8.0
        );
    }
    Ok(())
}

/// Render the PoC report: terse summary at the top, the sweep table, the
/// de-entropy pass with every row listed, and the raw data in an appendix.
fn build_output(
    data: &[u128],
    diffs: &[u128],
    low: u128,
    high: u128,
    cands: &[Candidate],
    scored: &[usize],
    winner: usize,
    identity: usize,
    pass: &deentropy::Outcome,
) -> String {
    let win = &cands[winner];
    let chunk_bits = win.dp_bits.unwrap();
    let total = chunk_bits + win.params_bits;
    let identity_chunks = cands[identity].dp_bits.unwrap();
    let identity_total = identity_chunks + cands[identity].params_bits;

    let count = |kind: &str| {
        pass.decisions
            .iter()
            .filter(|d| d.rule.display() == kind)
            .count()
    };
    let deleted_rows = pass.rows - count("keep");
    let keep_total = pass.keep_bits;
    let saved = keep_total - pass.total_bits;
    let saved_percent = if keep_total > 0.0 {
        100.0 * saved / keep_total
    } else {
        0.0
    };
    let pass_is_best = pass.total_bits < identity_total;

    let mut out = String::new();
    out.push_str("Duckomp-v8 report\n=================\n");
    out.push_str(&format!(
        "data        : {} offsets, {} gaps, range {low}..{high}\n",
        data.len(),
        diffs.len()
    ));
    out.push_str(&format!(
        "grid sweep  : {} widths × {} rules, Haar depth {HAAR_LEVELS}\n",
        WIDTHS.len(),
        RuleKind::ALL.len()
    ));
    if win.kind == RuleKind::Identity {
        out.push_str(&format!(
            "rule sweep  : no rule beat flat; baseline {:.0} bytes = {:.0} bits\n",
            identity_total / 8.0,
            identity_total
        ));
    } else {
        out.push_str(&format!(
            "rule sweep  : {} at width {}; {:.0} bytes = {:.0} bits\n",
            win.kind.display(),
            win.w,
            total / 8.0,
            total
        ));
    }
    out.push_str(&format!(
        "de-entropy  : width {}; {deleted_rows}/{} rows deleted ({} of {} cells); {:.0} bits vs {:.0} kept ({saved:+.0} bits, {saved_percent:+.1}%); exact {}{}\n\n",
        pass.w,
        pass.rows,
        pass.deleted_cells,
        pass.total,
        pass.total_bits,
        keep_total,
        if pass.exact { "yes" } else { "no" },
        if pass_is_best { "  <- best overall" } else { "" }
    ));

    out.push_str("Rule sweep\n----------\n");
    out.push_str("All sizes are bytes. \"Rule parameters\" is what the rule's own settings cost.\n");
    out.push_str("Every rule/width pair is rebuilt and checked; the optimal chunker runs on\n");
    out.push_str("the best per-row candidates only, because a full search is quadratic.\n\n");
    out.push_str(&format!(
        "{:>7}  {:<20}  {:>15}  {:>11}  {:>13}  {:>14}  {:>16}\n",
        "Width",
        "Rule",
        "Rule parameters",
        "Fixed radix",
        "Per-row radix",
        "Optimal chunks",
        "Rebuilds exactly"
    ));
    for (i, c) in cands.iter().enumerate() {
        let chunks = match c.dp_bits {
            Some(bits) => format!("{:.0}", bits / 8.0),
            None => "not scored".to_string(),
        };
        let mark = if i == winner { "  [WINNER]" } else { "" };
        out.push_str(&format!(
            "{:>7}  {:<20}  {:>15.0}  {:>11.0}  {:>13.0}  {:>14}  {:>16}{mark}\n",
            c.w,
            c.kind.display(),
            c.params_bits / 8.0,
            c.fixed_bits / 8.0,
            c.row_bits / 8.0,
            chunks,
            if c.exact { "yes" } else { "no" }
        ));
    }

    out.push('\n');
    if win.kind == RuleKind::Identity {
        out.push_str(&format!(
            "Winner: none — no rule beat the flat baseline ({:.0} bytes = {:.0} bits).\n",
            identity_total / 8.0,
            identity_total
        ));
    } else {
        out.push_str(&format!(
            "Winner: {} at width {} — {:.0} bytes = {:.0} bits ({:.0} chunk bits + {:.0} parameter bits).\n",
            win.kind.display(),
            win.w,
            total / 8.0,
            total,
            chunk_bits,
            win.params_bits
        ));
    }
    out.push_str(&format!(
        "Identity baseline: {:.0} bytes = {:.0} bits ({:.0} chunk bits + {:.0} parameter bits).\n",
        identity_total / 8.0,
        identity_total,
        identity_chunks,
        cands[identity].params_bits
    ));
    out.push_str(&format!(
        "Rebuild check: {} of {} rule/width pairs rebuild exactly ({} of {} were scored).\n",
        cands.iter().filter(|c| c.exact).count(),
        cands.len(),
        scored.len(),
        cands.len()
    ));
    if total < identity_total {
        out.push_str(&format!(
            "Rule gain: saves {:.0} bytes ({:.1}%) against keeping the data flat.\n",
            (identity_total - total) / 8.0,
            100.0 * (identity_total - total) / identity_total
        ));
    } else {
        out.push_str("Rule gain: none — keeping the data flat is cheaper.\n");
    }

    // --- de-entropy pass ---------------------------------------------------
    let verdict = if pass.total_bits < keep_total {
        format!("saves {saved:.0} bits ({saved_percent:.1}%)")
    } else {
        format!("no gain — it costs {:.0} bits more", pass.total_bits - keep_total)
    };

    out.push_str(&format!(
        "\nDe-entropy pass (best width {})\n------------------------------\n",
        pass.w
    ));
    out.push_str(&format!(
        "Entropy before: {:.2} bits per cell. Entropy after, kept rows only: {:.2} bits per cell.\n",
        pass.entropy_before, pass.entropy_kept
    ));
    out.push_str(&format!(
        "Rows deleted: {deleted_rows} of {} ({} of {} cells; {} ragged tail cells). Rebuilt exactly: {}.\n",
        pass.rows,
        pass.deleted_cells,
        pass.total,
        pass.ragged_cells,
        if pass.exact { "yes" } else { "no" }
    ));
    out.push_str(&format!(
        "Rules used: permutation {}, repeat {}, shift {}, constant {}, keep {}.\n",
        count("permutation"),
        count("repeat"),
        count("shift"),
        count("constant"),
        count("keep")
    ));
    out.push_str(
        "Note: rows are ranked by entropy, but only rows a rule can rebuild are deleted,\n\
         so the kept rows can be denser than the input average.\n",
    );
    out.push_str(&format!(
        "Entropy deleted: {:.0} bits of {:.0} bits total.\n",
        pass.deleted_entropy,
        pass.entropy_before * pass.total as f64
    ));
    out.push_str(&format!(
        "Cost: payload {:.0} bits + shared multiset {:.0} bits + constant values {:.0} bits + rule tags {:.0} bits = {:.0} bits ({:.0} bytes).\n",
        pass.payload_bits,
        pass.multiset_bits,
        pass.constants_bits,
        pass.tag_bits,
        pass.total_bits,
        pass.total_bits / 8.0
    ));
    out.push_str(&format!(
        "Against keeping every row: {:.0} bits ({:.0} bytes) -> {verdict}.\n",
        keep_total,
        keep_total / 8.0
    ));
    out.push_str(&format!(
        "Reference: flat fixed radix {:.0} bits; flat optimal chunks {:.0} bits.\n",
        pass.identity_bits, identity_chunks
    ));
    out.push_str(&format!(
        "Against the best rule-sweep result: {:.0} bits vs {:.0} bits ({:+.0} bits).\n",
        pass.total_bits,
        total,
        pass.total_bits - total
    ));
    if let Some(m) = &pass.multiset {
        out.push_str(&format!("Shared multiset ({} values):\n", m.len()));
        for chunk in m.chunks(16) {
            out.push_str(&format!(
                "  {}\n",
                chunk
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        }
    }
    out.push_str(&format!(
        "\nRows at width {} ({} total; [Deleted] rows are rebuilt by the rule shown):\n",
        pass.w, pass.rows
    ));
    for d in &pass.decisions {
        match &d.rule {
            deentropy::RowRule::Keep => out.push_str(&format!(
                "  row {:>4} [Kept]   : keep; row cost {:.2} bits, entropy {:.2} bits.\n",
                d.row, d.cost_bits, d.entropy
            )),
            rule => {
                let detail = match rule {
                    deentropy::RowRule::Keep => String::new(),
                    deentropy::RowRule::Constant { value } => format!("value {value}"),
                    deentropy::RowRule::Repeat { src } => format!("source row {src}"),
                    deentropy::RowRule::Shift { src, shift } => {
                        format!("source row {src}, shift {shift}")
                    }
                    deentropy::RowRule::Perm { rank } => format!("rank {rank}"),
                };
                out.push_str(&format!(
                    "  row {:>4} [Deleted]: {}, {}; rule cost {:.2} bits, entropy {:.2} bits.\n",
                    d.row,
                    rule.display(),
                    detail,
                    d.cost_bits,
                    d.entropy
                ));
            }
        }
    }

    out.push('\n');
    out
}
