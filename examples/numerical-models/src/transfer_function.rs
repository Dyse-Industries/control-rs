//! Transfer-function demo. Copy this file and point `Store` / `Dsp` / `Blas` at your backends.
//!
//! Numerator and denominator coefficients are in ascending power order.

use crate::suite::{
    case_inputs, col_array, emit_stdout, json_f64, json_usize, require_usize,
};
use crate::{
    ABS_F64, logspace, native_artifact, owned_to_rows, print_matrix,
    time_kernel, timing_entry,
};
use control_rs::math::dsp::DefaultDsp;
use control_rs::math::num_types::Const;
use control_rs::math::storage::ArrayStorage;
use control_rs::math::subprograms::DefaultBlas;
use control_rs::polynomial::Polynomial;
use control_rs::transfer_function::TransferFunction;
use serde_json::{Value, json};

/// Swap these for custom coefficient backends.
type StoreNum<const N: usize> = ArrayStorage<f64, N, 1>;
type StoreDen<const D: usize> = ArrayStorage<f64, D, 1>;
/// Swap this for a hardware convolution backend.
type Dsp = DefaultDsp;
/// Swap this for a hardware BLAS (`Scal` / `Axpy`) used by CCF.
type Blas = DefaultBlas;
type Tf<const N: usize, const D: usize> =
    TransferFunction<f64, Const<N>, Const<D>, StoreNum<N>, StoreDen<D>>;

const BODE_N: usize = 128;

fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut v = 1.0_f64;
    for i in 0..k {
        v = v * ((n - i) as f64) / ((i + 1) as f64);
    }
    v
}

/// $(s - (-a))^n = (s+a)^n$ ascending, $n=4$, five coefficients.
fn poly_s_plus_a_4(a: f64) -> [f64; 5] {
    let n = 4_usize;
    let mut c = [0.0_f64; 5];
    for k in 0..=n {
        c[k] = binomial(n, k) * a.powi((n - k) as i32);
    }
    c
}

pub fn run(suite: &Value) {
    eprintln!("=== Transfer Function Numerical Model Example ===");

    let pair = case_inputs(suite, "transfer_function.host.complex_pair");
    require_usize(&pair["freqs"], "n", BODE_N);
    let omega_n = json_f64(&pair["omega_n"]);
    let zeta = json_f64(&pair["zeta"]);
    let num = col_array::<1>(&pair["num"]);
    let den_c = col_array::<3>(&pair["den"]);

    let tf = Tf::<1, 3>::from_storage(
        StoreNum::from_column(num),
        StoreDen::from_column(den_c),
        None,
    );

    eprintln!(
        "\n--- Underdamped complex pair (ωn = {omega_n} rad/s, ζ = {zeta}) ---"
    );
    eprintln!("Numerator (ascending): {:?}", tf.num_slice());
    eprintln!("Denominator (ascending): {:?}", tf.den_slice());

    let start_log10 = json_f64(&pair["freqs"]["start_log10"]);
    let stop_log10 = json_f64(&pair["freqs"]["stop_log10"]);
    let freqs = logspace(start_log10, stop_log10, BODE_N);
    let mut h_re = vec![0.0_f64; BODE_N];
    let mut h_im = vec![0.0_f64; BODE_N];
    let mut mag = vec![0.0_f64; BODE_N];
    let mut phase = vec![0.0_f64; BODE_N];
    eprintln!("\n--- Frequency Evaluation (logspace -2..3, {BODE_N} pts) ---");
    let expected_wn = 1.0 / (2.0 * zeta);
    for (idx, &w) in freqs.iter().enumerate() {
        let resp = tf.eval_frequency(w);
        let (mag_pt, phase_rad) = tf.bode_point(w);
        h_re[idx] = resp.re;
        h_im[idx] = resp.im;
        mag[idx] = mag_pt;
        phase[idx] = phase_rad;
    }
    let h_at_wn = tf.bode_point(omega_n).0;
    eprintln!("H(jωn) mag={h_at_wn:.6} (1/(2ζ)={expected_wn:.6})");
    assert!(
        (h_at_wn - expected_wn).abs() <= 1e-6,
        "complex-pair |H(jωn)|: {h_at_wn} vs {expected_wn}"
    );

    let ser = case_inputs(suite, "transfer_function.host.series");
    let h1 = Tf::<1, 2>::from_storage(
        StoreNum::from_column(col_array(&ser["H1"]["num"])),
        StoreDen::from_column(col_array(&ser["H1"]["den"])),
        None,
    );
    let h2 = Tf::<1, 2>::from_storage(
        StoreNum::from_column(col_array(&ser["H2"]["num"])),
        StoreDen::from_column(col_array(&ser["H2"]["den"])),
        None,
    );
    let h_series = h1.series_with::<Dsp, 1, 2, 1, 3>(&h2);

    eprintln!("\n--- Series Cascade H1 * H2 ---");
    eprintln!("H1(s): {:?} / {:?}", h1.num_slice(), h1.den_slice());
    eprintln!("H2(s): {:?} / {:?}", h2.num_slice(), h2.den_slice());
    eprintln!(
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

    let ccf = case_inputs(suite, "transfer_function.host.ccf");
    let tf_realize = Tf::<2, 3>::from_storage(
        StoreNum::from_column(col_array(&ccf["num"])),
        StoreDen::from_column(col_array(&ccf["den"])),
        None,
    );
    let ss = tf_realize
        .to_controllable_canonical_form_with::<Blas, 2>()
        .expect("CCF");
    eprintln!("\n--- Controllable Canonical Realization ---");
    eprintln!("H(s) = (2 + 3s) / (4 + 5s + s^2)");
    print_matrix("A", &ss.a());
    print_matrix("B", &ss.b());
    print_matrix("C", &ss.c());
    print_matrix("D", &ss.d());
    assert_eq!(ss.a().rows(), 2);
    assert_eq!(ss.a().cols(), 2);

    let cl = case_inputs(suite, "transfer_function.host.clustered_poles");
    require_usize(&cl["freqs"], "n", BODE_N);
    eprintln!("\n--- Clustered-pole H(s) = 1/[(s+1)^4 (s+1.01)^4] ---");
    let d1 = Polynomial::<f64, Const<5>, StoreDen<5>>::from_storage(
        StoreDen::from_column(poly_s_plus_a_4(1.0)),
    );
    let d2 = Polynomial::<f64, Const<5>, StoreDen<5>>::from_storage(
        StoreDen::from_column(poly_s_plus_a_4(1.01)),
    );
    let den_c = d1.mul_poly_with::<Dsp, 5, 9>(&d2);
    let mut den_arr = [0.0_f64; 9];
    for i in 0..9 {
        den_arr[i] = den_c.get(i).copied().unwrap_or(0.0);
    }
    let tf_c = Tf::<1, 9>::from_storage(
        StoreNum::from_column(col_array(&cl["num"])),
        StoreDen::from_column(den_arr),
        None,
    );
    let mut c_re = vec![0.0_f64; BODE_N];
    let mut c_im = vec![0.0_f64; BODE_N];
    let mut c_mag = vec![0.0_f64; BODE_N];
    for (idx, &w) in freqs.iter().enumerate() {
        let resp = tf_c.eval_frequency(w);
        c_re[idx] = resp.re;
        c_im[idx] = resp.im;
        c_mag[idx] = (resp.re * resp.re + resp.im * resp.im).sqrt();
    }
    let bode_iters = json_usize(&pair["iters"]).unwrap_or(50) as u32;
    let bode_ns = time_kernel(bode_iters, || {
        let mut acc = 0.0_f64;
        for &w in &freqs {
            acc += tf.eval_frequency(w).re;
        }
        acc
    });
    eprintln!("Bode min ns ({bode_iters} sweeps): {bode_ns}");

    let values = json!({
        "H_RE": h_re,
        "H_IM": h_im,
        "NUM_SER": [h_series.num_slice()[0]],
        "DEN_SER": [den[0], den[1], den[2]],
        "CCF_A": owned_to_rows(&ss.a()),
        "CCF_B": owned_to_rows(&ss.b()),
        "CCF_C": owned_to_rows(&ss.c()),
        "CCF_D": owned_to_rows(&ss.d()),
        "CLUSTER_H_RE": c_re,
        "CLUSTER_H_IM": c_im,
        "FREQS": freqs,
        "OMEGA_N": omega_n,
        "ZETA": zeta,
    });
    let series = json!({
        "bode_mag": { "x": freqs, "y": mag },
        "bode_phase": { "x": freqs, "y": phase },
        "cluster_mag": { "x": freqs, "y": c_mag },
    });
    let metrics = json!({});
    let timings = json!({
        "bode": timing_entry(bode_iters, bode_ns),
    });
    emit_stdout(&native_artifact(
        "transfer_function",
        values,
        series,
        metrics,
        timings,
    ));
}
