//! Optional native kernels for the dipcatcher research harness.
//!
//! Safe Rust only: this crate contains no `unsafe` blocks. Order routing and
//! live-trading paths are intentionally absent.

mod book;
mod hashing;
mod returns;
mod rolling;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

fn as_slice<'a>(values: &'a PyReadonlyArray1<'a, f64>) -> PyResult<&'a [f64]> {
    values.as_slice().map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "expected a contiguous float64 buffer ({err})"
        ))
    })
}

fn check_len(flat: &[f64], rows: usize, cols: usize, name: &str) -> PyResult<()> {
    let expected = rows.saturating_mul(cols);
    if flat.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} length {got} != {rows} * {cols}",
            got = flat.len()
        )));
    }
    Ok(())
}

#[pyfunction]
fn rolling_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&values)?;
    Ok(PyArray1::from_vec(
        py,
        rolling::rolling_mean_1d(series, window),
    ))
}

#[pyfunction]
fn rolling_std<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&values)?;
    Ok(PyArray1::from_vec(
        py,
        rolling::rolling_std_1d(series, window),
    ))
}

#[pyfunction]
fn rolling_mean_2d<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    rows: usize,
    cols: usize,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = as_slice(&values)?;
    check_len(flat, rows, cols, "rolling_mean_2d")?;
    Ok(PyArray1::from_vec(
        py,
        rolling::rolling_rows(flat, rows, cols, window, false),
    ))
}

#[pyfunction]
fn rolling_std_2d<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    rows: usize,
    cols: usize,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = as_slice(&values)?;
    check_len(flat, rows, cols, "rolling_std_2d")?;
    Ok(PyArray1::from_vec(
        py,
        rolling::rolling_rows(flat, rows, cols, window, true),
    ))
}

#[pyfunction]
fn ema<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&values)?;
    Ok(PyArray1::from_vec(py, rolling::ema(series, window)))
}

#[pyfunction]
fn rsi<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&values)?;
    Ok(PyArray1::from_vec(py, rolling::rsi(series, window)))
}

#[pyfunction]
fn bollinger<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: usize,
    num_sd: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let series = as_slice(&values)?;
    let [mid, upper, lower, pct_b, bandwidth] = rolling::bollinger(series, window, num_sd);
    let out = PyDict::new(py);
    out.set_item("mid", PyArray1::from_vec(py, mid))?;
    out.set_item("upper", PyArray1::from_vec(py, upper))?;
    out.set_item("lower", PyArray1::from_vec(py, lower))?;
    out.set_item("pct_b", PyArray1::from_vec(py, pct_b))?;
    out.set_item("bandwidth", PyArray1::from_vec(py, bandwidth))?;
    Ok(out)
}

#[pyfunction]
fn simple_returns<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&prices)?;
    Ok(PyArray1::from_vec(py, returns::simple_returns_1d(series)))
}

#[pyfunction]
fn simple_returns_2d<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = as_slice(&prices)?;
    check_len(flat, rows, cols, "simple_returns_2d")?;
    Ok(PyArray1::from_vec(
        py,
        returns::simple_returns_2d(flat, rows, cols),
    ))
}

#[pyfunction]
fn wealth_index<'py>(
    py: Python<'py>,
    rets: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&rets)?;
    Ok(PyArray1::from_vec(py, returns::wealth_index(series)))
}

#[pyfunction]
fn turnover(weights: PyReadonlyArray1<'_, f64>, prev: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let w = as_slice(&weights)?;
    let p = as_slice(&prev)?;
    if w.len() != p.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "turnover weight length mismatch: {} vs {}",
            w.len(),
            p.len()
        )));
    }
    Ok(returns::turnover(w, p))
}

#[pyfunction]
fn turnover_series<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = as_slice(&weights)?;
    check_len(flat, rows, cols, "turnover_series")?;
    Ok(PyArray1::from_vec(
        py,
        returns::turnover_series(flat, rows, cols),
    ))
}

#[pyfunction]
fn book_features<'py>(
    py: Python<'py>,
    bid_px: PyReadonlyArray1<'py, f64>,
    bid_sz: PyReadonlyArray1<'py, f64>,
    ask_px: PyReadonlyArray1<'py, f64>,
    ask_sz: PyReadonlyArray1<'py, f64>,
    rows: usize,
    depth: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let bp = as_slice(&bid_px)?;
    let bs = as_slice(&bid_sz)?;
    let ap = as_slice(&ask_px)?;
    let az = as_slice(&ask_sz)?;
    check_len(bp, rows, depth, "bid_px")?;
    check_len(bs, rows, depth, "bid_sz")?;
    check_len(ap, rows, depth, "ask_px")?;
    check_len(az, rows, depth, "ask_sz")?;
    let cols = book::book_features(bp, bs, ap, az, rows, depth);
    let names = [
        "mid",
        "spread",
        "microprice",
        "imbalance_top",
        "imbalance_depth",
        "bid_depth",
        "ask_depth",
        "bid_log_size_slope",
        "ask_log_size_slope",
        "bid_log_price_slope",
        "ask_log_price_slope",
        "bid_mean_log_tick_spacing",
        "ask_mean_log_tick_spacing",
    ];
    let out = PyDict::new(py);
    for (name, values) in names.iter().zip(cols) {
        out.set_item(name, PyArray1::from_vec(py, values))?;
    }
    Ok(out)
}

#[pyfunction]
fn hash_bytes(data: &[u8]) -> String {
    hashing::sha256_hex(data)
}

fn push_ascii(out: &mut Vec<[u8; 64]>, item: &Bound<'_, PyAny>) -> PyResult<()> {
    let blob = item.downcast::<PyBytes>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("hash_many expects a sequence of bytes")
    })?;
    out.push(hashing::sha256_hex_ascii(blob.as_bytes()));
    Ok(())
}

fn py_list_from_ascii<'py>(py: Python<'py>, hexes: &[[u8; 64]]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(
        py,
        hexes.iter().map(|hex| {
            std::str::from_utf8(hex).unwrap_or_else(|_| unreachable!("hex digits are ASCII"))
        }),
    )
}

#[pyfunction]
fn hash_many<'py>(py: Python<'py>, chunks: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
    // Hex stays in one `[u8; 64]` buffer per digest. Building a Rust `String`
    // per chunk allocated twice (the `String`, then the Python `str`) and lost
    // the 1.5× floor against hashlib on SHA-NI hosts.
    if let Ok(list) = chunks.downcast::<PyList>() {
        let mut hexes = Vec::with_capacity(list.len());
        for index in 0..list.len() {
            push_ascii(&mut hexes, &list.get_item(index)?)?;
        }
        return py_list_from_ascii(py, &hexes);
    }
    if let Ok(tuple) = chunks.downcast::<PyTuple>() {
        let mut hexes = Vec::with_capacity(tuple.len());
        for index in 0..tuple.len() {
            push_ascii(&mut hexes, &tuple.get_item(index)?)?;
        }
        return py_list_from_ascii(py, &hexes);
    }
    let mut hexes = Vec::new();
    for item in chunks.try_iter()? {
        push_ascii(&mut hexes, &item?)?;
    }
    py_list_from_ascii(py, &hexes)
}

#[pymodule]
fn quant_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_2d, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_2d, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger, m)?)?;
    m.add_function(wrap_pyfunction!(simple_returns, m)?)?;
    m.add_function(wrap_pyfunction!(simple_returns_2d, m)?)?;
    m.add_function(wrap_pyfunction!(wealth_index, m)?)?;
    m.add_function(wrap_pyfunction!(turnover, m)?)?;
    m.add_function(wrap_pyfunction!(turnover_series, m)?)?;
    m.add_function(wrap_pyfunction!(book_features, m)?)?;
    m.add_function(wrap_pyfunction!(hash_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(hash_many, m)?)?;
    Ok(())
}
