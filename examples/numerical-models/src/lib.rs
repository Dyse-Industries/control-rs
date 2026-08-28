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

pub mod matrix;
pub mod polynomial;
pub mod state_space;
pub mod tensor;
pub mod transfer_function;

use std::path::Path;

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::Owned;
use serde_json::{Value, json};

/// Absolute error bound for `f64` equivalence tables (umbrella §6.3).
pub const ABS_F64: f64 = 1e-12;
/// Absolute error bound for `f32` equivalence tables (umbrella §6.3).
pub const ABS_F32: f32 = 1e-6;
/// Linear-solve ∞-norm residual-ratio bound (umbrella §6.3, $\tau_{\mathrm{res}} = 20$).
pub const SOLVE_RESIDUAL_TAU: f64 = 20.0;
/// ZOH `A_d` residual-ratio bound versus SciPy `e^{A T_s}` (state-space §6.3).
pub const ZOH_AD_TAU: f64 = 20.0;

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

/// Write a JSON artifact under the crate root (creates parent dirs).
pub fn save(rel: impl AsRef<Path>, doc: &Value) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel.as_ref());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create results dir");
    }
    let text = serde_json::to_string_pretty(doc).expect("serialize json");
    std::fs::write(&path, format!("{text}\n")).expect("write artifact");
    eprintln!("wrote {}", path.display());
}

/// Wrap native `values` / `series` as a V&V artifact.
pub fn native_artifact(slug: &str, values: Value, series: Value) -> Value {
    json!({
        "slug": slug,
        "source": "rust",
        "values": values,
        "series": series,
    })
}
