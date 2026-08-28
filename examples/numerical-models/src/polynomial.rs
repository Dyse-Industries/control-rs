//! Polynomial demo. Copy this file and point `Store` / `Dsp` at your backends.
//!
//! Coefficients are stored in ascending power order (constant term first).

use crate::{
    ABS_F64, native_artifact, owned_to_rows, print_matrix, save, time_kernel,
    timing_entry,
};
use control_rs::math::complex_num::Complex;
use control_rs::math::dsp::DefaultDsp;
use control_rs::math::num_types::{Const, Dim};
use control_rs::math::storage::ArrayStorage;
use control_rs::polynomial::Polynomial;
use serde_json::json;

/// Swap this for a custom coefficient backend.
type Store<const N: usize> = ArrayStorage<f64, N, 1>;
/// Swap this for a hardware convolution backend.
type Dsp = DefaultDsp;
type Poly<const N: usize> = Polynomial<f64, Const<N>, Store<N>>;

const SWEEP_N: usize = 128;
const HORNER_ITERS: u32 = 10_000;
const CLUSTER_X: f64 = 1.005;

fn first_n<const N: usize>(p: &Poly<N>, n: usize) -> Vec<f64>
where
    Const<N>: Dim,
{
    (0..n).map(|i| p.get(i).copied().unwrap_or(0.0)).collect()
}

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

/// Coefficients of $(x - a)^8$, ascending.
fn linear_pow8(a: f64) -> [f64; 9] {
    let n = 8_usize;
    let mut c = [0.0_f64; 9];
    for k in 0..=n {
        let sign = if (n - k) % 2 == 0 { 1.0 } else { -1.0 };
        c[k] = binomial(n, k) * sign * a.powi((n - k) as i32);
    }
    c
}

fn abs_poly_eval(coeffs: &[f64], x_abs: f64) -> f64 {
    let mut acc = 0.0_f64;
    for &c in coeffs.iter().rev() {
        acc = acc * x_abs + c.abs();
    }
    acc
}

pub fn main() {
    println!("=== Polynomial Numerical Model Example ===");

    let p =
        Poly::<5>::from_storage(Store::from_column([2.0, -3.0, 4.0, 1.0, 0.0]));
    println!("\n--- Polynomial Evaluation & Calculus ---");
    println!("Coefficients (ascending): {:?}", p.as_slice());

    let x_test = 2.5;
    let val_real = p.evaluate(x_test);
    println!("p({x_test}) = {val_real:.10}");

    let s_test = Complex::new(1.0, 2.0);
    let val_complex = p.evaluate_complex(s_test);
    println!(
        "p({:.1} + {:.1}j) = {:.10} + {:.10}j",
        s_test.re, s_test.im, val_complex.re, val_complex.im
    );

    let dp = p.derivative();
    println!("p'(x) coefficients: {:?}", dp.as_slice());
    let dp_val = dp.evaluate(x_test);
    println!("p'({x_test}) = {:.10}", dp_val);

    let integ = p.integral(5.0);
    println!("int p(x) dx (c0=5) coefficients: {:?}", integ.as_slice());
    let integ_val = integ.evaluate(x_test);
    println!("int_0^{x_test} p(t) dt + 5.0 = {:.10}", integ_val);

    let p1 = Poly::<2>::from_storage(Store::from_column([1.0, 2.0]));
    let p2 = Poly::<2>::from_storage(Store::from_column([3.0, 4.0]));
    let prod = p1.mul_poly_with::<Dsp, 2, 3>(&p2);
    println!("\n--- Polynomial Multiplication & Division ---");
    println!("(1 + 2x) * (3 + 4x) = {:?}", prod.as_slice());

    let (quot, rem) = prod.div_rem::<2, 2, 1>(&p1).expect("div_rem");
    println!(
        "Quotient of ({:?}) / ({:?}): {:?}",
        prod.as_slice(),
        p1.as_slice(),
        quot.as_slice()
    );
    println!("Remainder: {:?}", rem.as_slice());

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
        Poly::<3>::from_storage(Store::from_column([-6.0, -5.0, 1.0]));
    println!("\n--- Monic Companion Matrix ---");
    println!("Monic p(x) coefficients: {:?}", p_monic.as_slice());
    assert!(p_monic.is_monic());

    let comp = p_monic.companion_matrix::<2>().expect("companion");
    println!("Companion Matrix C:");
    print_matrix("C", &comp);

    println!("\n--- Clustered-root Horner (x-1)^8 (x-1.01)^8 ---");
    let left = Poly::<9>::from_storage(Store::from_column(linear_pow8(1.0)));
    let right = Poly::<9>::from_storage(Store::from_column(linear_pow8(1.01)));
    let cluster = left.mul_poly_with::<Dsp, 9, 17>(&right);
    let cluster_coeffs: Vec<f64> = first_n(&cluster, 17);
    let sweep_x: Vec<f64> = (0..SWEEP_N)
        .map(|i| 0.9 + 0.2 * (i as f64) / ((SWEEP_N - 1) as f64))
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
    let horner_ns = time_kernel(HORNER_ITERS, || {
        cluster.evaluate(core::hint::black_box(CLUSTER_X))
    });
    println!("Horner at x={CLUSTER_X}: {}", cluster.evaluate(CLUSTER_X));
    println!("Horner min ns ({HORNER_ITERS} iters): {horner_ns}");

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
        "horner": timing_entry(HORNER_ITERS, horner_ns),
    });
    save(
        "results/polynomial/native.json",
        &native_artifact("polynomial", values, series, metrics, timings),
    );
}
