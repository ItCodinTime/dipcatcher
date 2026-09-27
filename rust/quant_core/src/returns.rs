//! Simple returns, wealth, and L1 turnover.
//!
//! `wealth_index` multiplies in order, matching `numpy.cumprod`. Turnover sums
//! absolute weight changes left to right. NumPy's pairwise `sum` can differ
//! by a few ulps once the vector is long; the Python tests allow that.

const PRICE_EPS: f64 = 1e-12;

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn step_return(prev: f64, price: f64) -> f64 {
    // `np.maximum(prev, 1e-12)` propagates NaN and replaces every other
    // non-greater value (including -inf) with the floor.
    let denom = if prev.is_nan() {
        f64::NAN
    } else if prev > PRICE_EPS {
        prev
    } else {
        PRICE_EPS
    };
    finite_or_zero(price / denom - 1.0)
}

pub fn simple_returns_1d(prices: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; prices.len()];
    for i in 1..prices.len() {
        out[i] = step_return(prices[i - 1], prices[i]);
    }
    out
}

pub fn simple_returns_2d(prices: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows.saturating_mul(cols)];
    if rows == 0 || cols == 0 {
        return out;
    }
    for col in 0..cols {
        for row in 1..rows {
            let prev = prices[(row - 1) * cols + col];
            let price = prices[row * cols + col];
            out[row * cols + col] = step_return(prev, price);
        }
    }
    out
}

pub fn wealth_index(returns: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(returns.len());
    let mut acc = 0.0;
    for (i, &ret) in returns.iter().enumerate() {
        if i == 0 {
            acc = 1.0 + ret;
        } else {
            acc *= 1.0 + ret;
        }
        out.push(acc);
    }
    out
}

pub fn turnover(weights: &[f64], prev: &[f64]) -> f64 {
    if weights.len() != prev.len() {
        return f64::NAN;
    }
    if weights.is_empty() {
        return 0.0;
    }
    for i in 0..weights.len() {
        if !weights[i].is_finite() || !prev[i].is_finite() {
            return f64::NAN;
        }
    }
    let mut total = 0.0;
    for i in 0..weights.len() {
        total += (weights[i] - prev[i]).abs();
    }
    total
}

pub fn turnover_series(weights: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows];
    if rows == 0 {
        return out;
    }
    out[0] = 0.0;
    for (row, slot) in out.iter_mut().enumerate().skip(1) {
        let start = row * cols;
        let prev = start - cols;
        *slot = turnover(&weights[start..start + cols], &weights[prev..prev + cols]);
    }
    out
}
