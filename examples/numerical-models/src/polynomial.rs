//! Polynomial demo. Copy this file and point `Store` / `Dsp` at your backends.
//!
//! Coefficients are stored in ascending power order (constant term first).

use crate::suite::{
    case_inputs, col_array, emit_stdout, json_f64, json_usize, require_usize,
};
use crate::{
    native_artifact, owned_to_rows, print_matrix, time_kernel, timing_entry,
    ABS_F64,
};
use control_rs::math::complex_num::Complex;
use control_rs::math::dsp::DefaultDsp;
use control_rs::math::num_types::{Const, Dim};
use control_rs::math::storage::ArrayStorage;
use control_rs::polynomial::Polynomial;
use serde_json::{json, Value};

/// Swap this for a custom coefficient backend.
type Store<const N: usize> = ArrayStorage<f64, N, 1>;
/// Swap this for a hardware convolution backend.
type Dsp = DefaultDsp;
type Poly<const N: usize> = Polynomial<f64, Const<N>, Store<N>>;

const SWEEP_N: usize = 128;

fn first_n<const N: usize>(p: &Poly<N>, n: usize) -> Vec<f64>
where
    Const<N>: Dim,
{
    (0..n).map(|i| p.get(i).copied().unwrap_or(0.0)).collect()
}

fn abs_poly_eval(coeffs: &[f64], x_abs: f64) -> f64 {
    let mut acc = 0.0_f64;
    for &c in coeffs.iter().rev() {
        acc = acc * x_abs + c.abs();
    }
    acc
}

pub fn run(suite: &Value) {
    eprintln!("=== Polynomial Numerical Model Example ===");

    let tut = case_inputs(suite, "polynomial.host.tutorial");
    let p =
        Poly::<5>::from_storage(Store::from_column(col_array(&tut["coeffs"])));
    eprintln!("\n--- Polynomial Evaluation & Calculus ---");
    eprintln!("Coefficients (ascending): {:?}", p.as_slice());

    let x_test = json_f64(&tut["x_real"]);
    let val_real = p.evaluate(x_test);
    eprintln!("p({x_test}) = {val_real:.10}");

    let s_test = Complex::new(
        json_f64(&tut["z_complex"]["re"]),
        json_f64(&tut["z_complex"]["im"]),
    );
    let val_complex = p.evaluate_complex(s_test);
    eprintln!(
        "p({:.1} + {:.1}j) = {:.10} + {:.10}j",
        s_test.re, s_test.im, val_complex.re, val_complex.im
    );

    let dp = p.derivative();
    eprintln!("p'(x) coefficients: {:?}", dp.as_slice());
    let dp_val = dp.evaluate(x_test);
    eprintln!("p'({x_test}) = {:.10}", dp_val);

    let c0 = json_f64(&tut["integral_c0"]);
    let integ = p.integral(c0);
    eprintln!("int p(x) dx (c0={c0}) coefficients: {:?}", integ.as_slice());
    let integ_val = integ.evaluate(x_test);
    eprintln!("int_0^{x_test} p(t) dt + {c0} = {:.10}", integ_val);

    let p1 = Poly::<2>::from_storage(Store::from_column(col_array(&tut["p1"])));
    let p2 = Poly::<2>::from_storage(Store::from_column(col_array(&tut["p2"])));
    let prod = p1.mul_poly_with::<Dsp, 2, 3>(&p2);
    eprintln!("\n--- Polynomial Multiplication & Division ---");
    eprintln!("(1 + 2x) * (3 + 4x) = {:?}", prod.as_slice());

    let (quot, rem) = prod.div_rem::<2, 2, 1>(&p1).expect("div_rem");
    eprintln!(
        "Quotient of ({:?}) / ({:?}): {:?}",
        prod.as_slice(),
        p1.as_slice(),
        quot.as_slice()
    );
    eprintln!("Remainder: {:?}", rem.as_slice());

    let recon = quot.mul_poly_with::<Dsp, 2, 3>(&p1);
    for i in 0..3 {
        let mut got = recon.get(i).copied().unwrap();
        if i == 0 {
            got += rem.get(0).copied().unwrap();
        }
        let expected = prod.get(i).copied().unwrap();
        assert!(
            (got - expected).abs() <= ABS_F64,
            "div_rem reconstruct coeff {i}: {got} vs {expected}"
        );
    }

    let p_monic =
        Poly::<13>::from_storage(Store::from_column(col_array(&tut["monic"])));
    require_usize(tut, "monic_degree", 12);
    eprintln!("\n--- Monic Companion Matrix (degree 12) ---");
    let ncoeff = p_monic.as_slice().len();
    eprintln!("Monic p(x) = (x+1)^12, {ncoeff} coefficients");
    assert!(p_monic.is_monic());

    let comp = p_monic.companion_matrix::<12>().expect("companion");
    eprintln!("Companion Matrix C:");
    print_matrix("C", &comp);

    let cluster_in = case_inputs(suite, "polynomial.host.clustered_horner");
    require_usize(&cluster_in["sweep"], "n", SWEEP_N);
    require_usize(cluster_in, "degree", 16);
    let sweep_start = json_f64(&cluster_in["sweep"]["start"]);
    let sweep_stop = json_f64(&cluster_in["sweep"]["stop"]);
    let timed_x = json_f64(&cluster_in["timed_x"]);
    let horner_iters =
        json_usize(&cluster_in["iters"]).unwrap_or(10_000) as u32;
    eprintln!("\n--- Clustered-root Horner (x-1)^8 (x-1.01)^8 ---");
    // let left = Poly::<9>::from_storage(Store::from_column(linear_pow8(1.0)));
    // let right = Poly::<9>::from_storage(Store::from_column(linear_pow8(1.01)));
    // let cluster = left.mul_poly_with::<Dsp, 9, 17>(&right);
    let cluster = Poly::<17>::from_storage(Store::from_column([
        20922789888000.0,
        -70734282393600.0,
        102992244837120.0,
        -87077748875904.0,
        48366009233424.0,
        -18861567058880.0,
        5374523477960.0,
        -1146901283528.0,
        185953177553.0,
        -23057159840.0,
        2185031420.0,
        -156952432.0,
        8394022.0,
        -323680.0,
        8500.0,
        -136.0,
        1.0, // 1.0,
             // -136.0,
             // 8500.0,
             // -323680.0,
             // 8394022.0,
             // -156952432.0,
             // 2185031420.0,
             // -23057159840.0,
             // 185953177553.0,
             // -1146901283528.0,
             // 5374523477960.0,
             // -18861567058880.0,
             // 48366009233424.0,
             // -87077748875904.0,
             // 102992244837120.0,
             // -70734282393600.0,
             // 20922789888000.0,
    ]));
    let cluster_coeffs: Vec<f64> = first_n(&cluster, 17);
    let sweep_x: Vec<f64> = (0..SWEEP_N)
        .map(|i| {
            sweep_start
                + (sweep_stop - sweep_start) * (i as f64)
                    / ((SWEEP_N - 1) as f64)
        })
        .collect();
    let cluster_y: Vec<f64> =
        sweep_x.iter().map(|&x| cluster.evaluate(x)).collect();
    let mut max_rel = 0.0_f64;
    for (&x, &y) in sweep_x.iter().zip(cluster_y.iter()) {
        let tilde = abs_poly_eval(&cluster_coeffs, x.abs());
        let bound = crate::gamma(32.0) * tilde;
        let rel = if y.abs() == 0.0 { 0.0 } else { bound / y.abs() };
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let horner_ns = time_kernel(horner_iters, || {
        cluster.evaluate(core::hint::black_box(timed_x))
    });
    eprintln!("Horner at x={timed_x}: {}", cluster.evaluate(timed_x));
    eprintln!("Horner min ns ({horner_iters} iters): {horner_ns}");

    let values = json!({
        "P_REAL": val_real,
        "P_C_RE": val_complex.re,
        "P_C_IM": val_complex.im,
        "DERIV": first_n(&dp, 5),
        "P_DERIV": dp_val,
        "INTEG": first_n(&integ, 5),
        "P_INTEG": integ_val,
        "PROD": first_n(&prod, 3),
        "QUOT": first_n(&quot, 2),
        "REM": rem.get(0).copied().unwrap(),
        "COMPANION": owned_to_rows(&comp),
        "CLUSTER_COEFFS": cluster_coeffs,
        "CLUSTER_X": sweep_x,
        "CLUSTER_Y": cluster_y,
    });
    let series = json!({
        "horner": { "x": sweep_x, "y": cluster_y },
    });
    let metrics = json!({
        "horner_bound_scale": max_rel,
    });
    let timings = json!({
        "horner": timing_entry(horner_iters, horner_ns),
    });
    emit_stdout(&native_artifact(
        "polynomial",
        "rust",
        values,
        series,
        metrics,
        timings,
    ));
}