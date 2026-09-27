//! Causal rolling reductions.
//!
//! Prefix sums are left-to-right `f64` additions, the same order as
//! `numpy.cumsum`. That is what makes `rolling_mean` / `rolling_std` bit-for-bit
//! with `quant_fund.lightspeed.ema` on IEEE-754 inputs, including NaN and inf.

/// Population rolling standard deviation does not apply for a window under 2.
const MIN_STD_WINDOW: i64 = 2;

fn nan_vec(n: usize) -> Vec<f64> {
    vec![f64::NAN; n]
}

pub fn rolling_mean_1d(series: &[f64], window: i64) -> Vec<f64> {
    let n = series.len();
    if window <= 0 {
        return nan_vec(n);
    }
    let Ok(width) = usize::try_from(window) else {
        return nan_vec(n);
    };
    if n < width {
        return nan_vec(n);
    }
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0);
    let mut acc = 0.0;
    for &value in series {
        acc += value;
        prefix.push(acc);
    }
    let mut out = nan_vec(n);
    let width_f = width as f64;
    for i in (width - 1)..n {
        let sum = prefix[i + 1] - prefix[i + 1 - width];
        out[i] = sum / width_f;
    }
    out
}

pub fn rolling_std_1d(series: &[f64], window: i64) -> Vec<f64> {
    let n = series.len();
    if window < MIN_STD_WINDOW {
        return nan_vec(n);
    }
    let Ok(width) = usize::try_from(window) else {
        return nan_vec(n);
    };
    if n < width {
        return nan_vec(n);
    }
    let mut prefix = Vec::with_capacity(n + 1);
    let mut prefix_sq = Vec::with_capacity(n + 1);
    prefix.push(0.0);
    prefix_sq.push(0.0);
    let mut acc = 0.0;
    let mut acc_sq = 0.0;
    for &value in series {
        acc += value;
        acc_sq += value * value;
        prefix.push(acc);
        prefix_sq.push(acc_sq);
    }
    let mut out = nan_vec(n);
    let width_f = width as f64;
    for i in (width - 1)..n {
        let sum = prefix[i + 1] - prefix[i + 1 - width];
        let sum_sq = prefix_sq[i + 1] - prefix_sq[i + 1 - width];
        let mean = sum / width_f;
        // `<` is false for NaN, so a non-finite window stays NaN. A tiny
        // negative from cancellation clamps to 0, matching `np.maximum(..., 0)`.
        let mut var = sum_sq / width_f - mean * mean;
        if var < 0.0 {
            var = 0.0;
        }
        out[i] = var.sqrt();
    }
    out
}

pub fn rolling_rows(
    series_rows: &[f64],
    rows: usize,
    cols: usize,
    window: i64,
    std: bool,
) -> Vec<f64> {
    let mut out = vec![f64::NAN; rows.saturating_mul(cols)];
    if rows == 0 || cols == 0 {
        return out;
    }
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let rolled = if std {
            rolling_std_1d(&series_rows[start..end], window)
        } else {
            rolling_mean_1d(&series_rows[start..end], window)
        };
        out[start..end].copy_from_slice(&rolled);
    }
    out
}

/// SMA-seeded EMA. `alpha = 2 / (window + 1)`. Seed is the sequential mean of
/// the first `window` samples (NumPy's pairwise mean can differ by an ulp).
pub fn ema(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = nan_vec(n);
    if window == 0 || n < window {
        return out;
    }
    let mut seed = 0.0;
    for value in values.iter().take(window) {
        seed += *value;
    }
    out[window - 1] = seed / window as f64;
    let alpha = 2.0 / (window as f64 + 1.0);
    let beta = 1.0 - alpha;
    for i in window..n {
        out[i] = alpha * values[i] + beta * out[i - 1];
    }
    out
}

fn wilder(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = nan_vec(n);
    if window == 0 || n < window {
        return out;
    }
    let mut seed = 0.0;
    for value in values.iter().take(window) {
        seed += *value;
    }
    out[window - 1] = seed / window as f64;
    let width = window as f64;
    for i in window..n {
        out[i] = (out[i - 1] * (width - 1.0) + values[i]) / width;
    }
    out
}

/// Wilder RSI. Inputs are finite; the Python wrapper rejects NaN/inf first.
pub fn rsi(close: &[f64], window: usize) -> Vec<f64> {
    let n = close.len();
    let mut up = vec![0.0; n];
    let mut down = vec![0.0; n];
    for i in 1..n {
        let delta = close[i] - close[i - 1];
        if delta > 0.0 {
            up[i] = delta;
        } else if delta < 0.0 {
            down[i] = -delta;
        }
    }
    let avg_up = wilder(&up, window);
    let avg_down = wilder(&down, window);
    let mut out = nan_vec(n);
    for i in 0..n {
        let rs = avg_up[i] / avg_down[i];
        out[i] = 100.0 - 100.0 / (1.0 + rs);
        if avg_down[i] == 0.0 {
            out[i] = if avg_up[i] > 0.0 { 100.0 } else { 50.0 };
        }
    }
    out
}

/// Bollinger bands. `mid` is the cumsum SMA. `sd` is a two-pass population
/// standard deviation on each window (NumPy's reduction can differ by an ulp).
pub fn bollinger(close: &[f64], window: usize, num_sd: f64) -> [Vec<f64>; 5] {
    let n = close.len();
    let mid = if window == 0 {
        nan_vec(n)
    } else {
        rolling_mean_1d(close, window as i64)
    };
    let mut sd = nan_vec(n);
    if window >= 1 && n >= window {
        for (i, sd_i) in sd.iter_mut().enumerate().skip(window - 1) {
            let start = i + 1 - window;
            let mut mean = 0.0;
            for value in close.iter().take(i + 1).skip(start) {
                mean += *value;
            }
            mean /= window as f64;
            let mut var = 0.0;
            for value in close.iter().take(i + 1).skip(start) {
                let delta = *value - mean;
                var += delta * delta;
            }
            *sd_i = (var / window as f64).sqrt();
        }
    }
    let mut upper = nan_vec(n);
    let mut lower = nan_vec(n);
    let mut pct_b = nan_vec(n);
    let mut bandwidth = nan_vec(n);
    for i in 0..n {
        upper[i] = mid[i] + num_sd * sd[i];
        lower[i] = mid[i] - num_sd * sd[i];
        let span = upper[i] - lower[i];
        pct_b[i] = (close[i] - lower[i]) / span;
        bandwidth[i] = span / mid[i];
    }
    [mid, upper, lower, pct_b, bandwidth]
}
