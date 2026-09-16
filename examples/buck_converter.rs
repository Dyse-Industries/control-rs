//! Pedagogical buck-converter demo: plant, Routh, margins, and lead design.
//!
//! Averaged CCM control-to-output model:
//! G_vd(s) = (V_in / (L C)) / (s² + s/(R_L C) + 1/(L C)).
//!
//! Run with `cargo run --example buck_converter`.

// Pedagogical ordering: constants follow the physical derivation and helpers
// precede `main`, so the source-order lint is relaxed for this example.
#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::suboptimal_flops
)]

use control_rs::classical_tools::compensators::lead;
use control_rs::classical_tools::margins::{Margins, stability_margins};
use control_rs::classical_tools::realization::DirectForm2T;
use control_rs::classical_tools::routh::stability;
use control_rs::math::num_types::{Const, Dim};
use control_rs::polynomial::ArrayPolynomial;
use control_rs::transfer_function::ArrayTransferFunction;

const V_IN: f64 = 12.0;
const V_OUT: f64 = 5.0;
const L: f64 = 100e-6;
const C: f64 = 100e-6;
const R_L: f64 = 1.0;
const N_OMEGA: usize = 50;
const ROUTH_EPS: f64 = 1e-12;
const TS: f64 = 1e-5;
// Plant constants are duplicated from `control-rs-validation/src/buck_converter/` for
// pedagogical self-containment. Verification-grade values live there.

fn plant() -> ArrayTransferFunction<f64, 1, 3> {
    ArrayTransferFunction::continuous(
        [V_IN / (L * C)],
        [1.0 / (L * C), 1.0 / (R_L * C), 1.0],
    )
}

fn logspace(start: f64, stop: f64) -> [f64; N_OMEGA] {
    let mut omegas = [0.0; N_OMEGA];
    let log_start = start.log10();
    let log_span = stop.log10() - log_start;
    let denom = (N_OMEGA - 1) as f64;
    for (i, omega) in omegas.iter_mut().enumerate() {
        *omega = 10.0_f64.powf(log_start + log_span * (i as f64) / denom);
    }
    omegas
}

fn print_tf<const N: usize, const D: usize>(
    name: &str,
    tf: &ArrayTransferFunction<f64, N, D>,
) where
    Const<N>: Dim,
    Const<D>: Dim,
{
    println!("{name} num (ascending) = {:?}", tf.num_slice());
    println!("{name} den (ascending) = {:?}", tf.den_slice());
}

fn print_margins(label: &str, m: &Margins<f64>) {
    println!("{label}");
    match m.gain_crossover_freq {
        Some(w) => println!("  ω_gc = {w:.4} rad/s"),
        None => println!("  ω_gc not found in the sweep"),
    }
    match m.phase_margin {
        Some(pm) => println!("  Φ_m  = {:.2} deg", pm.to_degrees()),
        None => println!("  Φ_m not found in the sweep"),
    }
    match m.gain_margin {
        Some(gm) => println!("  K_g  = {gm:.4} ({:.2} dB)", 20.0 * gm.log10()),
        None => println!("  K_g  infinite (no −180° crossing)"),
    }
}

struct LeadDesign {
    k: f64,
    t: f64,
    alpha: f64,
    tf: ArrayTransferFunction<f64, 2, 2>,
}

fn synthesize_lead(
    plant: &ArrayTransferFunction<f64, 1, 3>,
    target_wc: f64,
    target_pm_deg: f64,
) -> LeadDesign {
    let g = plant.eval_frequency(target_wc);
    let plant_mag = g.magnitude();
    let uncomp_pm = core::f64::consts::PI + g.arg();
    let target_pm = target_pm_deg.to_radians();
    let phi = (target_pm - uncomp_pm + 5.0_f64.to_radians()).clamp(0.1, 1.4);
    let sin_phi = phi.sin();
    let alpha = (1.0 - sin_phi) / (1.0 + sin_phi);
    let t = 1.0 / (target_wc * alpha.sqrt());
    let k = 1.0 / (alpha.sqrt() * plant_mag);
    LeadDesign {
        k,
        t,
        alpha,
        tf: lead(k, t, alpha).expect("lead α < 1"),
    }
}

fn lead_to_df2t(
    discrete: &ArrayTransferFunction<f64, 2, 2>,
) -> DirectForm2T<f64, 1> {
    let num = discrete.num_slice();
    let den = discrete.den_slice();
    let a0 = *den.last().expect("lead den has two coefficients");
    let b0 = *num.last().expect("lead num has two coefficients") / a0;
    let b1 = *num.first().expect("lead num has two coefficients") / a0;
    let a1 = *den.first().expect("lead den has two coefficients") / a0;
    DirectForm2T::new(b0, [b1], [a1])
}

fn main() {
    let wn = 1.0 / (L * C).sqrt();
    let zeta = (L / C).sqrt() / (2.0 * R_L);
    let plant = plant();

    println!("Buck converter (averaged CCM, duty-to-output)");
    println!(
        "  V_in = {V_IN} V,  V_out = {V_OUT} V,  D = {:.4}",
        V_OUT / V_IN
    );
    println!("  L = {L:.1e} H,  C = {C:.1e} F,  R_L = {R_L} Ω");
    println!("  ω_n = {wn:.4} rad/s,  ζ = {zeta:.4}");
    print_tf("G_vd(s)", &plant);

    let den = ArrayPolynomial::<f64, 3>::from_coefficients([
        wn * wn,
        2.0 * zeta * wn,
        1.0,
    ]);
    let rhp = stability(&den, ROUTH_EPS).expect("Routh on plant denominator");
    println!("Routh-Hurwitz RHP count (plant den) = {rhp}");

    let omegas = logspace(1e2, 1e6);
    let plant_margins = stability_margins(&plant, &omegas);
    print_margins("Uncompensated plant margins:", &plant_margins);

    let target_wc = 3.0 * wn;
    let design = synthesize_lead(&plant, target_wc, 45.0);
    println!(
        "Lead C(s) targeting ω_c = 3 ω_n ({target_wc:.4} rad/s), PM = 45°"
    );
    println!(
        "  K = {:.6},  T = {:.6} s,  α = {:.6}",
        design.k, design.t, design.alpha
    );
    print_tf("C(s)", &design.tf);

    let loop_tf = design.tf.series::<1, 3, 2, 4>(&plant);
    print_tf("L(s) = C(s) G_vd(s)", &loop_tf);
    let loop_margins = stability_margins(&loop_tf, &omegas);
    print_margins("Compensated loop margins:", &loop_margins);

    let discrete = design.tf.to_discrete_tustin(TS, None);
    print_tf("C(z) Tustin Ts=1e-5", &discrete);
    let df2t = lead_to_df2t(&discrete);
    let b0 = df2t.b0;
    let b1 = *df2t.b.first().expect("order-1 numerator");
    let a1 = *df2t.a.first().expect("order-1 denominator");
    println!("DirectForm2T<1>  b0 = {b0:.6}, b1 = {b1:.6}, a1 = {a1:.6}");
}
