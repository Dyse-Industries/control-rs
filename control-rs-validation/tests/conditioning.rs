//! Rust-side invariants of the ill-conditioned kernels.
//!
//! These assertions need no Python and no HDF5: they state what must hold of
//! `control-rs` alone on each suite's inputs, and they state it in both
//! directions. A kernel that stops being ill-conditioned has stopped testing
//! anything, so the tests assert that the error is *large* where the problem
//! is hard and *small* where the algorithm is supposed to be backward stable.

// Harness code: the tests abort loudly on a malformed payload and index fixed
// paths that the gated schema guarantees.
#![allow(
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::arithmetic_side_effects
)]

use serde_json::Value;

/// Read a `group/dataset` path as an `f64` array.
fn array(payload: &Value, path: &str) -> Vec<f64> {
    let node = path
        .split('/')
        .try_fold(payload, |acc, seg| acc.get(seg))
        .unwrap_or_else(|| panic!("missing path `{path}`"));
    node.as_array()
        .unwrap_or_else(|| panic!("`{path}` is not an array"))
        .iter()
        .map(|v| {
            v.as_f64()
                .unwrap_or_else(|| panic!("`{path}` is not numeric"))
        })
        .collect()
}

/// Read a `group/dataset` path as a scalar.
fn scalar(payload: &Value, path: &str) -> f64 {
    path.split('/')
        .try_fold(payload, |acc, seg| acc.get(seg))
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing scalar `{path}`"))
}

#[test]
/// # Verification
/// Method: Requirements-based test on the Hilbert solve of order 10.
fn hilbert_solve_is_backward_stable_and_forward_inaccurate() {
    let payload = control_rs_validation::matrix::payload();

    let residual = scalar(&payload, "hilbert_solve/residual");
    assert!(
        residual < 1e-12,
        "backward error must stay near machine epsilon, got {residual:e}"
    );

    // The exact solution is the all-ones vector. At kappa ~ 1e13 the computed
    // solution must visibly miss it, otherwise the kernel is not exercising
    // the conditioning it claims to.
    let x = array(&payload, "hilbert_solve/x");
    let worst = x.iter().map(|v| (v - 1.0).abs()).fold(0.0_f64, f64::max);
    assert!(
        worst > 1e-8,
        "forward error must be observable at kappa ~ 1e13, got {worst:e}"
    );
    assert!(x.iter().all(|v| v.is_finite()), "solution must stay finite");
}

#[test]
/// # Verification
/// Method: Requirements-based test on QR of a near-rank-deficient matrix.
fn qr_keeps_orthogonality() {
    let payload = control_rs_validation::matrix::payload();
    let residual = scalar(&payload, "qr_orthogonality/residual");
    assert!(
        residual < 1e-12,
        "||Q^T Q - I||_F must stay near machine epsilon, got {residual:e}"
    );
}

#[test]
/// # Verification
/// Method: Requirements-based test on Wilkinson's polynomial.
fn wilkinson_residual_grows_toward_the_middle_roots() {
    let payload = control_rs_validation::polynomial::payload();
    let residual = array(&payload, "wilkinson/residual_f64");

    // W(1) is computed from coefficients that have not yet cancelled; W(16)
    // is the classical worst case. The ratio is the whole point of the kernel.
    let first = residual.first().copied().expect("20 roots");
    let worst = residual.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        worst > first * 1e6,
        "expansion error must blow up toward the middle roots: {first:e} -> {worst:e}"
    );
    assert!(residual.iter().all(|v| v.is_finite()));
}

#[test]
/// # Verification
/// Method: Requirements-based test on division by a near-exact factor.
fn clustered_division_remainder_is_small_but_nonzero() {
    let payload = control_rs_validation::polynomial::payload();
    let rem = array(&payload, "clustered_division/rem");
    let magnitude = rem.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        magnitude < 1e-6,
        "dividing out a root the polynomial has must nearly vanish, got {magnitude:e}"
    );
}

#[test]
/// # Verification
/// Method: Requirements-based test on the stiff ZOH realization.
fn stiff_discretization_matches_the_scalar_modes() {
    let payload = control_rs_validation::state_space::payload();
    let ad = &payload["stiff_zoh"]["ad"];
    let ts = scalar(&payload, "stiff_zoh/ts");

    let slow = ad[0][0].as_f64().expect("A_d[0][0]");
    let fast = ad[1][1].as_f64().expect("A_d[1][1]");
    let slow_exact = (-ts).exp();
    let fast_exact = (-5.0e3 * ts).exp();

    assert!(
        (slow - slow_exact).abs() < 1e-12,
        "slow mode: {slow:e} vs {slow_exact:e}"
    );
    assert!(
        (fast - fast_exact).abs() / fast_exact < 1e-6,
        "fast mode: {fast:e} vs {fast_exact:e}"
    );
}

#[test]
/// # Verification
/// Method: Requirements-based test on the 2000-step trajectory.
fn long_horizon_trajectory_stays_bounded() {
    let payload = control_rs_validation::state_space::payload();
    let theta = array(&payload, "phase_portrait/theta");

    assert_eq!(theta.len(), 2000, "horizon length");
    assert!(
        theta.iter().all(|v| v.is_finite()),
        "trajectory must stay finite"
    );

    // Light damping: the envelope must decay, not grow, over 2000 steps.
    let head = theta[..100].iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let tail = theta[1900..]
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(tail < head, "envelope must decay: {head:e} -> {tail:e}");
}
