//! Host numerical-model examples: suite-driven validators and JSON V&V.
//!
//! ## Column-major matrix literals
//!
//! `ArrayStorage::from_array` and `Owned::from_array` take **column-major**
//! literals: `[[col0], [col1], …]` where each inner array is one column of
//! length `R`. A NumPy row-major `[[a, b], [c, d]]` is written
//! `[[a, c], [b, d]]` in Rust.

#![allow(
    clippy::print_stdout,
    clippy::uninlined_format_args,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::large_stack_arrays,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    missing_docs
)]

pub mod compare;
pub mod matrix;
pub mod polynomial;
pub mod state_space;
pub mod suite;
pub mod tensor;
pub mod transfer_function;

use std::time::Instant;

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::Owned;
use serde_json::{Value, json};

/// Absolute error bound for `f64` tutorial keys (umbrella §6.3).
pub const ABS_F64: f64 = 1e-12;
/// Absolute error bound for `f32` tutorial / curved-grid keys.
pub const ABS_F32: f32 = 1e-6;
/// Linear-solve ∞-norm residual-ratio bound (umbrella §6.3, $\tau_{\mathrm{res}} = 20$).
pub const SOLVE_RESIDUAL_TAU: f64 = 20.0;
/// ZOH `A_d` residual-ratio bound versus SciPy $e^{A T_s}$ (state-space §6.3).
pub const ZOH_AD_TAU: f64 = 20.0;
/// Condition-scaled forward-error factor $\tau$ in $\tau\kappa\varepsilon$.
pub const TAU_KAPPA: f64 = 20.0;

/// Higham $\gamma_k = k\varepsilon / (1 - k\varepsilon)$.
#[must_use]
pub fn gamma(k: f64) -> f64 {
    let ke = k * f64::EPSILON;
    if ke >= 1.0 {
        f64::INFINITY
    } else {
        ke / (1.0 - ke)
    }
}

/// Warmup once, then return the minimum duration of `iters` calls in nanoseconds.
///
/// The closure must return a value that depends on the kernel output. Release
/// builds otherwise dead-code-eliminate pure kernels (Bode, Horner, ZOH).
pub fn time_kernel<T, F: FnMut() -> T>(iters: u32, mut f: F) -> u64 {
    let _ = core::hint::black_box(f());
    let mut best = u64::MAX;
    for _ in 0..iters {
        let start = Instant::now();
        let _ = core::hint::black_box(f());
        let ns = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        if ns < best {
            best = ns;
        }
    }
    best
}

/// JSON object `{ "iters": …, "ns": … }`.
#[must_use]
pub fn timing_entry(iters: u32, ns: u64) -> Value {
    json!({ "iters": iters, "ns": ns })
}

/// $N$ log-spaced points $10^{a}$ … $10^{b}$.
#[must_use]
pub fn logspace(start_log10: f64, stop_log10: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![10.0_f64.powf(stop_log10)];
    }
    let den = (n - 1) as f64;
    (0..n)
        .map(|i| {
            10.0_f64.powf(
                start_log10 + (stop_log10 - start_log10) * (i as f64) / den,
            )
        })
        .collect()
}

/// Prints a dense `f64` matrix in row-major visual layout.
pub fn print_matrix<const R: usize, const C: usize>(
    name: &str,
    m: &Owned<f64, R, C>,
) where
    Const<R>: Dim,
    Const<C>: Dim,
{
    eprintln!("{name}:");
    for i in 0..R {
        eprint!("  [");
        for j in 0..C {
            if j > 0 {
                eprint!(", ");
            }
            eprint!("{:12.6}", m.get(i, j).copied().unwrap_or(0.0));
        }
        eprintln!("]");
    }
}

/// ∞-norm of a dense `f64` matrix (maximum absolute row sum).
pub fn inf_norm_mat<const R: usize, const C: usize>(m: &Owned<f64, R, C>) -> f64
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    let mut best = 0.0_f64;
    for i in 0..R {
        let mut s = 0.0_f64;
        for j in 0..C {
            s += m.get(i, j).copied().unwrap_or(0.0).abs();
        }
        if s > best {
            best = s;
        }
    }
    best
}

/// ∞-norm of a dense `f64` column vector.
pub fn inf_norm_col<const N: usize>(m: &Owned<f64, N, 1>) -> f64
where
    Const<N>: Dim,
{
    let mut best = 0.0_f64;
    for i in 0..N {
        let v = m.get(i, 0).copied().unwrap_or(0.0).abs();
        if v > best {
            best = v;
        }
    }
    best
}

/// Frobenius norm of a dense `f64` matrix.
pub fn frobenius<const R: usize, const C: usize>(m: &Owned<f64, R, C>) -> f64
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    let mut s = 0.0_f64;
    for i in 0..R {
        for j in 0..C {
            let v = m.get(i, j).copied().unwrap_or(0.0);
            s += v * v;
        }
    }
    s.sqrt()
}

/// Residual test ratio $\lVert Ax-b\rVert_\infty / (\lVert A\rVert_\infty \lVert x\rVert_\infty \varepsilon)$.
pub fn solve_residual_ratio<const N: usize>(
    a: &Owned<f64, N, N>,
    x: &Owned<f64, N, 1>,
    b: &Owned<f64, N, 1>,
) -> f64
where
    Const<N>: Dim,
{
    let mut ax = Owned::<f64, N, 1>::zero();
    a.mul_into(x, &mut ax);
    let mut num = 0.0_f64;
    for i in 0..N {
        num = num.max(
            (ax.get(i, 0).copied().unwrap_or(0.0)
                - b.get(i, 0).copied().unwrap_or(0.0))
            .abs(),
        );
    }
    let den = inf_norm_mat(a) * inf_norm_col(x) * f64::EPSILON;
    if den == 0.0 { 0.0 } else { num / den }
}

/// Row-major JSON array from a dense `Owned` matrix.
pub fn owned_to_rows<const R: usize, const C: usize>(
    m: &Owned<f64, R, C>,
) -> Value
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    let mut rows = Vec::with_capacity(R);
    for i in 0..R {
        let mut row = Vec::with_capacity(C);
        for j in 0..C {
            row.push(m.get(i, j).copied().unwrap_or(0.0));
        }
        rows.push(Value::Array(row.into_iter().map(Value::from).collect()));
    }
    Value::Array(rows)
}

/// Column of an `Owned` matrix as a flat `Vec`.
pub fn col0<const N: usize>(m: &Owned<f64, N, 1>) -> Vec<f64>
where
    Const<N>: Dim,
{
    (0..N)
        .map(|i| m.get(i, 0).copied().unwrap_or(0.0))
        .collect()
}

/// Wrap native `values` / `series` / `metrics` / `timings` as a V&V artifact.
pub fn native_artifact(
    slug: &str,
    values: Value,
    series: Value,
    metrics: Value,
    timings: Value,
) -> Value {
    json!({
        "slug": slug,
        "source": "rust",
        "values": values,
        "series": series,
        "metrics": metrics,
        "timings": timings,
    })
}
