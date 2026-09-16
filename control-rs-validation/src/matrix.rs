//! Matrix suite: ill-conditioned dense linear algebra.
//!
//! Every kernel here is chosen because double precision *loses* digits on it:
//! a Hilbert solve at $\kappa_2 \approx 10^{13}$, a monomial Vandermonde solve
//! on equispaced nodes, an explicit inverse residual, and Gram-Schmidt
//! orthogonality loss on a near-rank-deficient matrix. Agreement with the
//! SciPy oracle on these inputs is evidence that `control-rs` accumulates
//! rounding the same way LAPACK does; agreement on well-conditioned inputs
//! (what `examples/matrix.rs` demonstrates) is not.
//!
//! Timing lives in `benches/numerical_models.rs`, not here.

use serde_json::{Value, json};

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::{LuDecomposition, Owned, Symmetric};

/// Datasets gated by the comparator; every other key is context only.
pub const GATED_PATHS: &[&str] = &[
    "hilbert_solve/x",
    "hilbert_solve/residual",
    "vandermonde_solve/x",
    "inverse_residual/matrix",
    "cholesky_spread/x",
    "qr_orthogonality/residual",
];

/// Order of the Hilbert system, $\kappa_2(H_{10}) \approx 1.6 \times 10^{13}$.
const HILBERT_N: usize = 10;

/// Order of the Vandermonde system on equispaced nodes in $[-1, 1]$.
const VANDERMONDE_N: usize = 8;

/// Order of the inverse-residual and Cholesky systems.
const SPREAD_N: usize = 8;

/// Hilbert matrix $H_{ij} = 1 / (i + j + 1)$.
fn hilbert<const N: usize>() -> Owned<f64, N, N>
where
    Const<N>: Dim,
{
    Owned::<f64, N, N>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64))
}

/// Monomial Vandermonde $V_{ij} = x_i^j$ on $N$ equispaced nodes in $[-1, 1]$.
fn vandermonde<const N: usize>() -> Owned<f64, N, N>
where
    Const<N>: Dim,
{
    Owned::<f64, N, N>::from_fn(|i, j| {
        let x = -1.0 + 2.0 * (i as f64) / ((N - 1) as f64);
        let power = i32::try_from(j).expect("column index fits in i32");
        x.powi(power)
    })
}

/// Row-major nesting of matrix entries.
type Rows = Vec<Vec<f64>>;

/// Row-major nesting of a matrix for JSON export.
fn rows<const R: usize, const C: usize>(m: &Owned<f64, R, C>) -> Rows
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    (0..R)
        .map(|i| (0..C).map(|j| *m.get(i, j).expect("in-bounds")).collect())
        .collect()
}

/// Column of a single-column matrix.
fn column<const R: usize>(x: &Owned<f64, R, 1>) -> Vec<f64>
where
    Const<R>: Dim,
{
    (0..R).map(|i| *x.get(i, 0).expect("in-bounds")).collect()
}

/// Solve $H_{10} x = b$ with $b = H_{10} \mathbf{1}$, exact $x = \mathbf{1}$.
///
/// The residual $\lVert H x - b \rVert_\infty$ stays near machine epsilon
/// while $x$ itself loses most of its digits: the pair separates backward
/// stability from forward accuracy, and both are compared.
fn hilbert_solve() -> Value {
    let h = hilbert::<HILBERT_N>();
    let ones = Owned::<f64, HILBERT_N, 1>::from_fn(|_, _| 1.0);
    let b = &h * &ones;

    let lu = LuDecomposition::decompose(h).expect("Hilbert LU");
    let mut x = b;
    lu.solve_mut(&mut x).expect("Hilbert solve");

    let hx = &h * &x;
    let residual = (0..HILBERT_N)
        .map(|i| {
            (*hx.get(i, 0).expect("in-bounds")
                - *b.get(i, 0).expect("in-bounds"))
            .abs()
        })
        .fold(0.0_f64, f64::max);

    json!({ "x": column(&x), "residual": residual, "order": HILBERT_N })
}

/// Solve $V c = y$ for the interpolating coefficients of $y_i = 1 / (1 + 25 x_i^2)$.
///
/// The Runge function on equispaced nodes makes the monomial basis the worst
/// available choice, which is the point: the coefficients are dominated by
/// cancellation and any difference in pivot order shows up immediately.
fn vandermonde_solve() -> Value {
    let v = vandermonde::<VANDERMONDE_N>();
    let y = Owned::<f64, VANDERMONDE_N, 1>::from_fn(|i, _| {
        let x = -1.0 + 2.0 * (i as f64) / ((VANDERMONDE_N - 1) as f64);
        1.0 / (1.0 + 25.0 * x * x)
    });

    let lu = LuDecomposition::decompose(v).expect("Vandermonde LU");
    let mut c = y;
    lu.solve_mut(&mut c).expect("Vandermonde solve");

    json!({ "x": column(&c), "order": VANDERMONDE_N })
}

/// Explicit inverse of $H_8$, reported as the residual $H_8 H_8^{-1} - I$.
///
/// Forming an inverse is the operation every numerics text warns against;
/// the residual is where the two implementations are allowed to differ.
fn inverse_residual() -> Value {
    let h = hilbert::<SPREAD_N>();
    let lu = LuDecomposition::decompose(h).expect("Hilbert LU (inverse)");
    let inv = lu.inverse().expect("Hilbert inverse");
    let product = &h * &inv;
    let eye = Owned::<f64, SPREAD_N, SPREAD_N>::identity();
    let residual = &product - &eye;

    json!({ "matrix": rows(&residual), "order": SPREAD_N })
}

/// Cholesky solve of an SPD matrix with a $10^{10}$ eigenvalue spread.
///
/// $A = Q \Lambda Q^\top$ is built directly from a graded diagonal plus a
/// rank-one term, so the small eigenvalues are the ones that decide the
/// answer.
fn cholesky_spread() -> Value {
    let a = Owned::<f64, SPREAD_N, SPREAD_N>::from_fn(|i, j| {
        let scale = |k: usize| 10f64.powf(-10.0 * (k as f64) / 7.0);
        if i == j {
            scale(i) + 1e-3
        } else {
            1e-3 * (scale(i) * scale(j)).sqrt()
        }
    });
    let spd = Symmetric::<f64, SPREAD_N>::from_owned(a)
        .expect("graded matrix is symmetric");
    let chol = spd
        .into_cholesky()
        .expect("graded matrix is positive definite");

    let mut x = Owned::<f64, SPREAD_N, 1>::from_fn(|_, _| 1.0);
    chol.solve_mut(&mut x).expect("Cholesky solve");

    json!({ "x": column(&x), "order": SPREAD_N })
}

/// Orthogonality loss of a QR factorization of a near-rank-deficient matrix.
///
/// Columns 3 and 4 differ by $10^{-9}$, so $\lVert Q^\top Q - I \rVert_F$ is
/// a direct measurement of how the factorization degrades.
fn qr_orthogonality() -> Value {
    const N: usize = 6;
    let mut a = Owned::<f64, N, N>::from_fn(|i, j| {
        let base = 1.0 / ((i + j + 1) as f64);
        if j == 4 {
            1.0 / ((i + 3 + 1) as f64) + 1e-9
        } else {
            base
        }
    });
    let mut q = Owned::<f64, N, N>::zero();
    a.qr_decompose_mut(&mut q);

    let qt_q = &q.transpose() * &q;
    let eye = Owned::<f64, N, N>::identity();
    let diff = &qt_q - &eye;
    let residual = (0..N)
        .flat_map(|i| {
            (0..N).map(move |j| {
                let v = *diff.get(i, j).expect("in-bounds");
                v * v
            })
        })
        .sum::<f64>()
        .sqrt();

    json!({ "residual": residual, "r": rows(&a), "order": N })
}

/// Assemble the full matrix-suite payload.
#[must_use]
pub fn payload() -> Value {
    json!({
        "hilbert_solve": hilbert_solve(),
        "vandermonde_solve": vandermonde_solve(),
        "inverse_residual": inverse_residual(),
        "cholesky_spread": cholesky_spread(),
        "qr_orthogonality": qr_orthogonality(),
    })
}

/// Emit `results/matrix.rust.h5`.
pub fn run() {
    println!("matrix: ill-conditioned linear algebra");
    crate::write_rust_container("matrix", &payload(), GATED_PATHS);
}
