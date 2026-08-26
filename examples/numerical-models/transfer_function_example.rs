//! Transfer Function Numerical Model Example
//!
//! Demonstrates continuous-time transfer function modeling, frequency response,
//! Bode analysis, series cascade interconnection, and realization into controllable canonical state-space form.

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
    clippy::doc_markdown
)]

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::Owned;
use control_rs::transfer_function::ArrayTransferFunction;

fn print_matrix<const R: usize, const C: usize>(
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

fn main() {
    println!("=== Transfer Function Numerical Model Example ===");

    // 1. Continuous 2nd-Order Low-Pass Butterworth Filter
    // H(s) = \omega_c^2 / (s^2 + \sqrt{2}\omega_c s + \omega_c^2)
    // \omega_c = 10.0 rad/s
    let omega_c = 10.0;
    let w_c2 = omega_c * omega_c;
    let sqrt2_wc = core::f64::consts::SQRT_2 * omega_c;

    let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
        [w_c2],
        [w_c2, sqrt2_wc, 1.0],
    );

    println!("\n--- Butterworth Filter (omega_c = {omega_c} rad/s) ---");
    println!("Numerator (ascending): {:?}", tf.num_slice());
    println!("Denominator (ascending): {:?}", tf.den_slice());

    let test_freqs = [0.1, 1.0, 10.0, 100.0];
    println!("\n--- Frequency Evaluation ---");
    println!(
        "{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}",
        "omega (rad/s)", "Real", "Imag", "Mag (abs)", "Mag (dB)", "Phase (deg)"
    );

    for &w in &test_freqs {
        let resp = tf.eval_frequency(w);
        let (mag, phase_rad) = tf.bode_point(w);
        let mag_db = 20.0 * libm::log10(mag);
        let phase_deg = phase_rad * (180.0 / core::f64::consts::PI);

        println!(
            "{:<16.2}{:<16.8}{:<16.8}{:<16.8}{:<16.8}{:<16.8}",
            w, resp.re, resp.im, mag, mag_db, phase_deg
        );
    }

    // 2. Series Cascade Interconnection: H1 * H2
    let h1 = ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [2.0, 1.0]);
    let h2 = ArrayTransferFunction::<f64, 1, 2>::continuous([5.0], [5.0, 1.0]);
    let h_series = h1.series::<1, 2, 1, 3>(&h2);

    println!("\n--- Series Cascade H1 * H2 ---");
    println!("H1(s): {:?} / {:?}", h1.num_slice(), h1.den_slice());
    println!("H2(s): {:?} / {:?}", h2.num_slice(), h2.den_slice());
    println!(
        "H_series: {:?} / {:?}",
        h_series.num_slice(),
        h_series.den_slice()
    );

    // 3. Controllable Canonical State-Space Realization
    // H(s) = (2 + 3s) / (4 + 5s + s^2)
    let tf_realize = ArrayTransferFunction::<f64, 2, 3>::continuous(
        [2.0, 3.0],
        [4.0, 5.0, 1.0],
    );
    if let Ok(ss) = tf_realize.to_controllable_canonical_form::<2>() {
        println!("\n--- Controllable Canonical Realization ---");
        println!("H(s) = (2 + 3s) / (4 + 5s + s^2)");
        print_matrix("A", ss.a());
        print_matrix("B", ss.b());
        print_matrix("C", ss.c());
        print_matrix("D", ss.d());
    }
}
