//! Host numerical-model examples and JSON file-based V&V tests.
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
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    missing_docs
)]

pub mod emit;
pub mod host_scale_emit;
pub mod matrix;
pub mod polynomial;
pub mod state_space;
pub mod tensor;
pub mod transfer_function;

#[cfg(test)]
mod host_scale;

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::Owned;

/// Absolute error bound for `f64` equivalence tables (umbrella §6.3).
pub const ABS_F64: f64 = 1e-12;
/// Absolute error bound for `f32` equivalence tables (umbrella §6.3).
pub const ABS_F32: f32 = 1e-6;
/// Linear-solve ∞-norm residual-ratio bound (umbrella §6.3, $\tau_{\mathrm{res}} = 20$).
pub const SOLVE_RESIDUAL_TAU: f64 = 20.0;
/// ZOH `A_d` residual-ratio bound versus SciPy `e^{A T_s}` (state-space §6.3).
pub const ZOH_AD_TAU: f64 = 20.0;

#[cfg(test)]
pub(crate) fn assert_f32(left: f32, right: f32, what: &str) {
    control_rs::assert_almost_eq!(left, right, ABS_F32, "{}", what);
}

#[cfg(test)]
pub(crate) fn assert_f64(left: f64, right: f64, what: &str) {
    control_rs::assert_almost_eq!(left, right, ABS_F64, "{}", what);
}

/// Prints a dense `f64` matrix in row-major visual layout.
pub fn print_matrix<const R: usize, const C: usize>(
    name: &str,
    m: &Owned<f64, R, C>,
) where
    Const<R>: Dim,
    Const<C>: Dim,
{
    println!("{name}:");
    for i in 0..R {
        print!("  [");
        for j in 0..C {
            if j > 0 {
                print!(", ");
            }
            print!("{:12.6}", m.get(i, j).copied().unwrap_or(0.0));
        }
        println!("]");
    }
}

/// ∞-norm of a dense `f64` matrix (maximum absolute row sum).
pub fn inf_norm_mat<const R: usize, const C: usize>(
    m: &Owned<f64, R, C>,
) -> f64
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

/// ∞-norm of a square row slice (host-scale / state-space tests).
pub(crate) fn inf_norm_rows<const N: usize>(rows: &[[f64; N]; N]) -> f64 {
    let mut best = 0.0_f64;
    for row in rows {
        let mut s = 0.0_f64;
        for &v in row {
            s += v.abs();
        }
        if s > best {
            best = s;
        }
    }
    best
}
