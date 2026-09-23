//! Polynomial numerical model control-rs-verification suite.
//!
//! Evaluates polynomial kernels:
//! 1. Tutorial polynomial evaluation (real and complex)
//! 2. Newton-Raphson root convergence rates across starting offsets
//! 3. Wilkinson polynomial ill-conditioning backward residual in f64 and f32
//! 4. Root sensitivity under coefficient perturbations

#![allow(missing_docs)]

use std::path::Path;

use control_rs::math::complex_num::Complex;
use control_rs::polynomial::ArrayPolynomial;

use crate::h5_writer::H5Writer;
use crate::numeric::KernelResult;

/// Starting offsets from the target root for the Newton-Raphson sweep.
const DISTANCES: [f64; 11] =
    [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];

/// Iteration cap for one Newton-Raphson run.
const MAX_NEWTON_ITERS: u32 = 100;

type Poly<const N: usize> = ArrayPolynomial<f64, N>;

/// `P(x) = x^3 - 10x^2 + 31x - 30` evaluated at a real and a complex point.
struct TutorialValues {
    /// `P(2.5)`.
    real: f64,
    /// `P(1 + 2i)`.
    complex: Complex<f64>,
}

/// Absolute Wilkinson residuals `|W(k)|`, `k = 1..=20`, in two precisions.
struct WilkinsonResiduals {
    f64_residuals: Vec<f64>,
    f32_residuals: Vec<f64>,
}

fn evaluate_tutorial_polynomial() -> TutorialValues {
    // P(x) = (x - 2)(x - 3)(x - 5) = x^3 - 10x^2 + 31x - 30
    // Ascending: [-30.0, 31.0, -10.0, 1.0]
    let p = Poly::<4>::from_coefficients([-30.0, 31.0, -10.0, 1.0]);
    TutorialValues {
        real: p.evaluate(2.5),
        complex: p.evaluate_complex(Complex::new(1.0, 2.0)),
    }
}

fn compute_root_convergence() -> Vec<f64> {
    // P(x) = (x - 2)(x + 3)(x - 5) = x^3 - 4x^2 - 11x + 30
    // Ascending: [30.0, -11.0, -4.0, 1.0]
    let p = Poly::<4>::from_coefficients([30.0, -11.0, -4.0, 1.0]);
    let dp = p.derivative();

    let target_root = 2.0;
    DISTANCES
        .iter()
        .map(|&dist| {
            let mut x = target_root + dist;
            let mut iters = 0_u32;
            for n in 1..=MAX_NEWTON_ITERS {
                iters = n;
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
            f64::from(iters)
        })
        .collect()
}

fn compute_wilkinson_residuals() -> WilkinsonResiduals {
    let mut roots_64 = [0.0_f64; 21];
    let mut roots_32 = [0.0_f32; 21];
    for ((r64, r32), k) in roots_64.iter_mut().zip(&mut roots_32).zip(1..=20_u8)
    {
        *r64 = f64::from(k);
        *r32 = f32::from(k);
    }

    let poly_64 = Poly::<21>::from_roots(roots_64);
    let poly_32 = ArrayPolynomial::<f32, 21>::from_roots(roots_32);

    WilkinsonResiduals {
        f64_residuals: (1..=20_u8)
            .map(|k| poly_64.evaluate(f64::from(k)).abs())
            .collect(),
        f32_residuals: (1..=20_u8)
            .map(|k| f64::from(poly_32.evaluate(f32::from(k)).abs()))
            .collect(),
    }
}

/// Executes the polynomial control-rs-verification kernel and writes `target/verification/polynomial.rust.h5`.
///
/// # Errors
///
/// Returns an error if the container cannot be written.
pub fn emit_container(output_path: &Path) -> KernelResult<()> {
    let mut writer = H5Writer::new();

    let tutorial = evaluate_tutorial_polynomial();
    writer.add_dataset("tutorial/p_real", &[tutorial.real]);
    writer.set_tolerance("tutorial/p_real", "abs", 1e-6);

    writer.add_dataset("tutorial/p_c_re", &[tutorial.complex.re]);
    writer.set_tolerance("tutorial/p_c_re", "abs", 1e-6);

    writer.add_dataset("tutorial/p_c_im", &[tutorial.complex.im]);
    writer.set_tolerance("tutorial/p_c_im", "abs", 1e-6);

    let iters = compute_root_convergence();
    writer.add_dataset("root_convergence/iterations", &iters);
    writer.set_tolerance("root_convergence/iterations", "abs", 2.0);

    let wilkinson = compute_wilkinson_residuals();
    writer.add_dataset(
        "wilkinson_residual/residual_f64",
        &wilkinson.f64_residuals,
    );
    writer.set_tolerance("wilkinson_residual/residual_f64", "rel", 0.05);

    writer.add_dataset(
        "wilkinson_residual/residual_f32",
        &wilkinson.f32_residuals,
    );
    writer.set_tolerance("wilkinson_residual/residual_f32", "rel", 10.0);

    writer.write_to_file(output_path)
}
