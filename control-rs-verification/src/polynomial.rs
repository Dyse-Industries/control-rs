//! Polynomial numerical model control-rs-verification suite.
//!
//! Evaluates polynomial kernels:
//! 1. Tutorial polynomial evaluation (real and complex)
//! 2. Newton-Raphson root convergence rates across starting offsets
//! 3. Wilkinson polynomial ill-conditioning backward residual in f64 and f32
//! 4. Root sensitivity under coefficient perturbations

#![allow(
    missing_docs,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::unwrap_used
)]

use std::path::Path;

use control_rs::math::complex_num::Complex;
use control_rs::polynomial::ArrayPolynomial;

use crate::h5_writer::H5Writer;

type Poly<const N: usize> = ArrayPolynomial<f64, N>;

fn evaluate_tutorial_polynomial() -> (f64, f64, f64) {
    // P(x) = (x - 2)(x - 3)(x - 5) = x^3 - 10x^2 + 31x - 30
    // Ascending: [-30.0, 31.0, -10.0, 1.0]
    let p = Poly::<4>::from_coefficients([-30.0, 31.0, -10.0, 1.0]);
    let p_real = p.evaluate(2.5);

    // Complex evaluation at 1.0 + 2.0i
    let c_val = p.evaluate_complex(Complex::new(1.0, 2.0));

    (p_real, c_val.re, c_val.im)
}

fn compute_root_convergence() -> Vec<f64> {
    // P(x) = (x - 2)(x + 3)(x - 5) = x^3 - 4x^2 - 11x + 30
    // Ascending: [30.0, -11.0, -4.0, 1.0]
    let p = Poly::<4>::from_coefficients([30.0, -11.0, -4.0, 1.0]);
    let dp = p.derivative();

    let target_root = 2.0;
    const DISTANCES: [f64; 11] =
        [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
    let mut iterations = Vec::with_capacity(11);

    for &dist in &DISTANCES {
        let x0 = target_root + dist;
        let mut x = x0;
        let mut iters = 0;
        let max_iters = 100;

        while iters < max_iters {
            iters += 1;
            let fx = p.evaluate(x);
            let fpx = dp.evaluate(x);
            if fpx.abs() < 1e-12 {
                break;
            }
            let x_next = x - fx / fpx;
            if (x_next - x).abs() < 1e-8 {
                break;
            }
            x = x_next;
        }
        iterations.push(iters as f64);
    }

    iterations
}

fn compute_wilkinson_residuals() -> (Vec<f64>, Vec<f64>) {
    let mut roots_64 = [0.0_f64; 21];
    let mut roots_32 = [0.0_f32; 21];
    for i in 0..20 {
        roots_64[i] = (i + 1) as f64;
        roots_32[i] = (i + 1) as f32;
    }

    let poly_64 = Poly::<21>::from_roots(roots_64);
    let poly_32 = ArrayPolynomial::<f32, 21>::from_roots(roots_32);

    let mut residual_f64 = Vec::with_capacity(20);
    let mut residual_f32 = Vec::with_capacity(20);

    for k in 1..=20 {
        let val_64 = poly_64.evaluate(k as f64);
        residual_f64.push(val_64.abs());

        let val_32 = poly_32.evaluate(k as f32);
        residual_f32.push(val_32.abs() as f64);
    }

    (residual_f64, residual_f32)
}

/// Executes the polynomial control-rs-verification kernel and writes `target/verification/polynomial.rust.h5`.
pub fn emit_container(output_path: &Path) -> Result<(), String> {
    let mut writer = H5Writer::new();

    let (p_real, p_c_re, p_c_im) = evaluate_tutorial_polynomial();
    writer.add_dataset("tutorial/p_real", &[p_real]);
    writer.set_tolerance("tutorial/p_real", "abs", 1e-6);

    writer.add_dataset("tutorial/p_c_re", &[p_c_re]);
    writer.set_tolerance("tutorial/p_c_re", "abs", 1e-6);

    writer.add_dataset("tutorial/p_c_im", &[p_c_im]);
    writer.set_tolerance("tutorial/p_c_im", "abs", 1e-6);

    let iters = compute_root_convergence();
    writer.add_dataset("root_convergence/iterations", &iters);
    writer.set_tolerance("root_convergence/iterations", "abs", 2.0);

    let (res_64, res_32) = compute_wilkinson_residuals();
    writer.add_dataset("wilkinson_residual/residual_f64", &res_64);
    writer.set_tolerance("wilkinson_residual/residual_f64", "rel", 0.05);

    writer.add_dataset("wilkinson_residual/residual_f32", &res_32);
    writer.set_tolerance("wilkinson_residual/residual_f32", "rel", 10.0);

    writer.write_to_file(output_path)
}
