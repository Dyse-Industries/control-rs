//! Polynomial suite: cancellation-dominated evaluation and factorization.
//!
//! The kernels are the classical ill-conditioned cases: Wilkinson's
//! $W(x) = \prod_{k=1}^{20} (x - k)$ evaluated at its own roots, Horner on a
//! polynomial with a 16-fold root at $1.01$, Euclidean division by a factor
//! that nearly divides, and a product of coefficient vectors spanning
//! $10^{16}$. None of them restate `examples/polynomial.rs`, whose job is to
//! show the API on inputs where the arithmetic is uninteresting.
//!
//! Timing lives in `benches/numerical_models.rs`, not here.

use serde_json::{Value, json};

use control_rs::math::num_types::Const;
use control_rs::math::storage::ArrayStorage;
use control_rs::polynomial::Polynomial;

/// Dense array-backed polynomial of capacity `N` (degree `N - 1`).
type Poly<const N: usize> = Polynomial<f64, Const<N>, ArrayStorage<f64, N, 1>>;

/// Single-precision counterpart used to expose the conditioning directly.
type Poly32<const N: usize> =
    Polynomial<f32, Const<N>, ArrayStorage<f32, N, 1>>;

/// Datasets gated by the comparator; every other key is context only.
pub const GATED_PATHS: &[&str] = &[
    "wilkinson/residual_f64",
    "wilkinson/residual_f32",
    "clustered_horner/values",
    "clustered_division/quot",
    "clustered_division/rem",
    "scaled_product/coeffs",
    "companion/matrix",
];

/// Multiplicity of the clustered root.
const CLUSTER_ORDER: usize = 16;

/// Location of the clustered root.
const CLUSTER_ROOT: f64 = 1.01;

/// $W(x) = \prod_{k=1}^{20}(x - k)$, evaluated at each root in both precisions.
///
/// The exact value is zero everywhere. What comes back is the expansion error
/// of the coefficients, which reaches $10^{10}$ near $k = 16$ in `f64` and
/// destroys `f32` entirely: a strict test of *how* the two implementations
/// expand the product, not merely of what they evaluate.
fn wilkinson() -> Value {
    let mut roots_64 = [0.0_f64; 21];
    let mut roots_32 = [0.0_f32; 21];
    for i in 0..20 {
        roots_64[i] = (i + 1) as f64;
        roots_32[i] = (i + 1) as f32;
    }

    let poly_64 = Poly::<21>::from_roots(roots_64);
    let poly_32 = Poly32::<21>::from_roots(roots_32);

    let root_indices: [usize; 20] = core::array::from_fn(|i| i + 1);
    let mut residual_f64 = [0.0_f64; 20];
    let mut residual_f32 = [0.0_f64; 20];
    for (idx, &k) in root_indices.iter().enumerate() {
        residual_f64[idx] = poly_64.evaluate(k as f64).abs();
        residual_f32[idx] = f64::from(poly_32.evaluate(k as f32).abs());
    }

    json!({
        "root_indices": root_indices,
        "residual_f64": residual_f64,
        "residual_f32": residual_f32,
        "coefficients": poly_64.as_slice(),
    })
}

/// Polynomial with a root of multiplicity 16 at `CLUSTER_ROOT`.
fn clustered() -> Poly<17> {
    let mut roots = [0.0; 17];
    for root in roots.iter_mut().take(CLUSTER_ORDER) {
        *root = CLUSTER_ROOT;
    }
    Poly::<17>::from_roots(roots)
}

/// Horner evaluation of the clustered polynomial across $[0, 2]$.
///
/// Near $x = 1.01$ every term cancels against the next, so the computed value
/// is pure rounding noise. Two implementations agreeing there agree on their
/// summation order.
fn clustered_horner() -> Value {
    let poly = clustered();
    let xs: Vec<f64> = (0..128).map(|i| 2.0 * (i as f64) / 127.0).collect();
    let values: Vec<f64> = xs.iter().map(|&x| poly.evaluate(x)).collect();

    json!({ "x": xs, "values": values })
}

/// Divide the clustered polynomial by $(x - 1.01)$, a factor it *nearly* has.
///
/// The remainder should be zero and is not; the quotient carries the error
/// forward into 16 coefficients.
fn clustered_division() -> Value {
    let dividend = clustered();
    let divisor = Poly::<2>::from_coefficients([-CLUSTER_ROOT, 1.0]);
    let (quot, rem) = dividend
        .div_rem::<2, 16, 1>(&divisor)
        .expect("division by a monic linear factor");

    json!({ "quot": quot.as_slice(), "rem": rem.as_slice() })
}

/// Product of two coefficient vectors whose magnitudes span $10^{16}$.
///
/// The convolution adds $10^{-8}$ terms to $10^{8}$ terms, so the small
/// contributions survive only if both implementations accumulate in the same
/// order.
fn scaled_product() -> Value {
    let a = Poly::<3>::from_coefficients([1e-8, 1.0, 1e8]);
    let b = Poly::<2>::from_coefficients([1e8, -1.0]);
    let product = a.mul_poly::<2, 4>(&b);

    json!({ "coeffs": product.as_slice() })
}

/// Companion matrix of $(s^2 + 2s + 5)(s^2 + 4s + 5)$, a complex-pole pair.
fn companion() -> Value {
    let poly = Poly::<5>::from_coefficients([25.0, 30.0, 18.0, 6.0, 1.0]);
    let matrix = poly
        .companion_matrix::<4>()
        .expect("monic companion construction");

    json!({ "matrix": matrix.to_rows() })
}

/// Newton iteration onto the 16-fold root from a range of starting offsets.
///
/// Convergence is linear, not quadratic, and the achievable accuracy is
/// bounded by $\epsilon^{1/16}$: the iterate stalls roughly $10^{-1}$ away
/// from the true root, at a point that depends on the last bits of every
/// evaluation. It is reported as context, not gated: the stalling point is
/// chaotic in the inputs and carries no cross-implementation bound.
fn newton_clustered() -> Value {
    const DISTANCES: [f64; 6] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0];
    const MAX_ITERS: usize = 100;

    let poly = clustered();
    let derivative = poly.derivative();

    let mut iterations = [0usize; 6];
    let mut final_x = [0.0_f64; 6];
    let mut final_residual = [0.0_f64; 6];

    for (idx, &distance) in DISTANCES.iter().enumerate() {
        let mut x = CLUSTER_ROOT + distance;
        let mut iters = 0;
        while iters < MAX_ITERS {
            let fx = poly.evaluate(x);
            let fpx = derivative.evaluate(x);
            if fpx.abs() < 1e-300 {
                break;
            }
            let next = x - fx / fpx;
            iters += 1;
            if (next - x).abs() < 1e-12 {
                x = next;
                break;
            }
            x = next;
        }
        iterations[idx] = iters;
        final_x[idx] = x;
        final_residual[idx] = poly.evaluate(x).abs();
    }

    json!({
        "distances": DISTANCES,
        "iterations": iterations,
        "final_x": final_x,
        "final_residual": final_residual,
    })
}

/// Assemble the full polynomial-suite payload.
#[must_use]
pub fn payload() -> Value {
    json!({
        "wilkinson": wilkinson(),
        "clustered_horner": clustered_horner(),
        "clustered_division": clustered_division(),
        "scaled_product": scaled_product(),
        "companion": companion(),
        "newton_clustered": newton_clustered(),
    })
}

/// Emit `results/polynomial.rust.h5`.
pub fn run() {
    println!("polynomial: cancellation-dominated evaluation");
    crate::write_rust_container("polynomial", &payload(), GATED_PATHS);
}
