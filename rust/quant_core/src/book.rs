//! Batch order-book features.
//!
//! One row is `(depth,)` bid/ask prices and sizes. A row that is non-finite,
//! non-positive, or crossed/locked (`best_bid >= best_ask`) is fail-closed:
//! every feature on that row is NaN. Depth-shape fields are NaN when `depth < 2`.
//! This is the array form of `book_metrics_from_snapshot`, without raising,
//! so a panel can be scored in one call.

pub const N_FIELDS: usize = 13;

fn ols_log_slope(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    for &value in values {
        if !value.is_finite() || value <= 0.0 {
            return f64::NAN;
        }
    }
    let n_f = n as f64;
    let denom = (n_f * (n_f * n_f - 1.0)) / 12.0;
    if denom <= 1e-18 {
        return f64::NAN;
    }
    let center = 0.5 * (n_f - 1.0);
    let mut dot = 0.0;
    for (i, &value) in values.iter().enumerate() {
        let x = i as f64 - center;
        dot += x * value.ln();
    }
    dot / denom
}

fn mean_log_tick_spacing(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let gaps = prices.len() - 1;
    for i in 1..prices.len() {
        if !prices[i].is_finite() || !prices[i - 1].is_finite() {
            return f64::NAN;
        }
        let gap = (prices[i] - prices[i - 1]).abs();
        if !gap.is_finite() || gap <= 0.0 {
            return f64::NAN;
        }
        sum += gap.ln();
    }
    sum / gaps as f64
}

fn row_valid(bid_px: &[f64], bid_sz: &[f64], ask_px: &[f64], ask_sz: &[f64]) -> bool {
    if bid_px.is_empty() {
        return false;
    }
    for i in 0..bid_px.len() {
        let values = [bid_px[i], bid_sz[i], ask_px[i], ask_sz[i]];
        for value in values {
            if !value.is_finite() || value <= 0.0 {
                return false;
            }
        }
    }
    bid_px[0] < ask_px[0]
}

fn push_nan(cols: &mut [Vec<f64>]) {
    for col in cols {
        col.push(f64::NAN);
    }
}

pub fn book_features(
    bid_px: &[f64],
    bid_sz: &[f64],
    ask_px: &[f64],
    ask_sz: &[f64],
    rows: usize,
    depth: usize,
) -> [Vec<f64>; N_FIELDS] {
    let mut cols: [Vec<f64>; N_FIELDS] = std::array::from_fn(|_| Vec::with_capacity(rows));
    if depth == 0 {
        return cols;
    }
    for row in 0..rows {
        let start = row * depth;
        let end = start + depth;
        let bp = &bid_px[start..end];
        let bs = &bid_sz[start..end];
        let ap = &ask_px[start..end];
        let az = &ask_sz[start..end];
        if !row_valid(bp, bs, ap, az) {
            push_nan(&mut cols);
            continue;
        }
        let mut bid_depth = 0.0;
        let mut ask_depth = 0.0;
        for i in 0..depth {
            bid_depth += bs[i];
            ask_depth += az[i];
        }
        let top_bid = bs[0];
        let top_ask = az[0];
        let mid = 0.5 * (bp[0] + ap[0]);
        let spread = ap[0] - bp[0];
        let denom = top_bid + top_ask;
        let microprice = if denom <= 0.0 || !denom.is_finite() {
            mid
        } else {
            (ap[0] * top_bid + bp[0] * top_ask) / denom
        };
        let imbalance_top = (top_bid - top_ask) / denom;
        let depth_sum = bid_depth + ask_depth;
        let imbalance_depth = if depth_sum > 0.0 {
            (bid_depth - ask_depth) / depth_sum
        } else {
            0.0
        };
        let values = [
            mid,
            spread,
            microprice,
            imbalance_top,
            imbalance_depth,
            bid_depth,
            ask_depth,
            ols_log_slope(bs),
            ols_log_slope(az),
            ols_log_slope(bp),
            ols_log_slope(ap),
            mean_log_tick_spacing(bp),
            mean_log_tick_spacing(ap),
        ];
        for (col, value) in cols.iter_mut().zip(values) {
            col.push(value);
        }
    }
    cols
}
