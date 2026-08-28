//! Transfer-function demo. Copy this file and point `Store` / `Dsp` / `Blas` at your backends.
//!
//! Numerator and denominator coefficients are in ascending power order.

use crate::{ABS_F64, native_artifact, owned_to_rows, print_matrix, save};
use control_rs::math::dsp::DefaultDsp;
use control_rs::math::num_types::Const;
use control_rs::math::storage::ArrayStorage;
use control_rs::math::subprograms::DefaultBlas;
use control_rs::transfer_function::TransferFunction;
use serde_json::json;

/// Swap these for custom coefficient backends.
type StoreNum<const N: usize> = ArrayStorage<f64, N, 1>;
type StoreDen<const D: usize> = ArrayStorage<f64, D, 1>;
/// Swap this for a hardware convolution backend.
type Dsp = DefaultDsp;
/// Swap this for a hardware BLAS (`Scal` / `Axpy`) used by CCF.
type Blas = DefaultBlas;
type Tf<const N: usize, const D: usize> =
    TransferFunction<f64, Const<N>, Const<D>, StoreNum<N>, StoreDen<D>>;

pub fn main() {
    println!("=== Transfer Function Numerical Model Example ===");

    let omega_c = 10.0;
    let w_c2 = omega_c * omega_c;
    let sqrt2_wc = core::f64::consts::SQRT_2 * omega_c;

    let tf = Tf::<1, 3>::from_storage(
        StoreNum::from_column([w_c2]),
        StoreDen::from_column([w_c2, sqrt2_wc, 1.0]),
        None,
    );

    println!("\n--- Butterworth Filter (omega_c = {omega_c} rad/s) ---");
    println!("Numerator (ascending): {:?}", tf.num_slice());
    println!("Denominator (ascending): {:?}", tf.den_slice());

    let test_freqs = [0.1, 1.0, 10.0, 100.0];
    let mut h_re = [0.0_f64; 4];
    let mut h_im = [0.0_f64; 4];
    let mut mag = [0.0_f64; 4];
    println!("\n--- Frequency Evaluation ---");
    println!(
        "{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}",
        "omega (rad/s)", "Real", "Imag", "Mag (abs)", "Mag (dB)", "Phase (deg)"
    );

    for (idx, &w) in test_freqs.iter().enumerate() {
        let resp = tf.eval_frequency(w);
        let (mag_pt, phase_rad) = tf.bode_point(w);
        let mag_db = 20.0 * libm::log10(mag_pt);
        let phase_deg = phase_rad * (180.0 / core::f64::consts::PI);
        h_re[idx] = resp.re;
        h_im[idx] = resp.im;
        mag[idx] = mag_pt;
        println!(
            "{:<16.2}{:<16.8}{:<16.8}{:<16.8}{:<16.8}{:<16.8}",
            w, resp.re, resp.im, mag_pt, mag_db, phase_deg
        );
        if (w - omega_c).abs() <= ABS_F64 {
            let expected_mag = core::f64::consts::FRAC_1_SQRT_2;
            assert!(
                (mag_pt - expected_mag).abs() <= ABS_F64,
                "Butterworth |H(jω_c)|: {mag_pt} vs {expected_mag}"
            );
        }
    }

    let h1 = Tf::<1, 2>::from_storage(
        StoreNum::from_column([2.0]),
        StoreDen::from_column([2.0, 1.0]),
        None,
    );
    let h2 = Tf::<1, 2>::from_storage(
        StoreNum::from_column([5.0]),
        StoreDen::from_column([5.0, 1.0]),
        None,
    );
    let h_series = h1.series_with::<Dsp, 1, 2, 1, 3>(&h2);

    println!("\n--- Series Cascade H1 * H2 ---");
    println!("H1(s): {:?} / {:?}", h1.num_slice(), h1.den_slice());
    println!("H2(s): {:?} / {:?}", h2.num_slice(), h2.den_slice());
    println!(
        "H_series: {:?} / {:?}",
        h_series.num_slice(),
        h_series.den_slice()
    );
    assert_eq!(h_series.num_slice().len(), 1);
    assert_eq!(h_series.den_slice().len(), 3);
    assert!(
        (h_series.num_slice()[0] - 10.0).abs() <= ABS_F64,
        "series numerator"
    );
    let den = h_series.den_slice();
    assert!((den[0] - 10.0).abs() <= ABS_F64, "series den s^0");
    assert!((den[1] - 7.0).abs() <= ABS_F64, "series den s^1");
    assert!((den[2] - 1.0).abs() <= ABS_F64, "series den s^2");

    let tf_realize = Tf::<2, 3>::from_storage(
        StoreNum::from_column([2.0, 3.0]),
        StoreDen::from_column([4.0, 5.0, 1.0]),
        None,
    );
    let ss = tf_realize
        .to_controllable_canonical_form_with::<Blas, 2>()
        .expect("CCF");
    println!("\n--- Controllable Canonical Realization ---");
    println!("H(s) = (2 + 3s) / (4 + 5s + s^2)");
    print_matrix("A", &ss.a());
    print_matrix("B", &ss.b());
    print_matrix("C", &ss.c());
    print_matrix("D", &ss.d());
    assert_eq!(ss.a().rows(), 2);
    assert_eq!(ss.a().cols(), 2);
    assert_eq!(ss.b().rows(), 2);
    assert_eq!(ss.c().cols(), 2);

    let values = json!({
        "H_RE": h_re,
        "H_IM": h_im,
        "NUM_SER": [h_series.num_slice()[0]],
        "DEN_SER": [den[0], den[1], den[2]],
        "CCF_A": owned_to_rows(&ss.a()),
        "CCF_B": owned_to_rows(&ss.b()),
        "CCF_C": owned_to_rows(&ss.c()),
        "CCF_D": owned_to_rows(&ss.d()),
    });
    let series = json!({
        "bode_mag": { "x": test_freqs, "y": mag },
    });
    save(
        "results/transfer_function/native.json",
        &native_artifact("transfer_function", values, series),
    );
}
