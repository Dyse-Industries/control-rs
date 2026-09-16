//! Pedagogical DC-motor position servo: plant, Routh, lead, and PID steps.
//!
//! Run with `cargo run --example dc_motor`.

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
use control_rs::classical_tools::pid::Pid;
use control_rs::classical_tools::routh::stability;
use control_rs::math::num_types::{Const, Dim};
use control_rs::polynomial::ArrayPolynomial;
use control_rs::transfer_function::ArrayTransferFunction;

const R_A: f64 = 2.0;
const L_A: f64 = 0.5e-3;
const K_T: f64 = 0.05;
const K_B: f64 = 0.05;
const J: f64 = 2e-4;
const B: f64 = 1e-4;
const V_MAX: f64 = 12.0;
const ROUTH_EPS: f64 = 1e-12;
// Plant constants are duplicated from `control-rs-validation/src/dc_motor/` for
// pedagogical self-containment. Verification-grade values live there.
const TS: f64 = 5e-4;

fn plant_position() -> ArrayTransferFunction<f64, 1, 4> {
    let d0 = 0.0;
    let d1 = R_A * B + K_T * K_B;
    let d2 = R_A * J + L_A * B;
    let d3 = L_A * J;
    ArrayTransferFunction::continuous([K_T], [d0, d1, d2, d3])
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

struct LeadDesign {
    k: f64,
    t: f64,
    alpha: f64,
    tf: ArrayTransferFunction<f64, 2, 2>,
}

fn synthesize_lead(
    plant: &ArrayTransferFunction<f64, 1, 4>,
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

fn main() {
    let d0 = 0.0;
    let d1 = R_A * B + K_T * K_B;
    let d2 = R_A * J + L_A * B;
    let d3 = L_A * J;
    let plant = plant_position();

    println!("DC motor armature position servo");
    println!("  R_a = {R_A} Ω,  L_a = {L_A:.1e} H");
    println!("  K_t = {K_T} N·m/A,  K_b = {K_B} V·s/rad");
    println!("  J = {J:.1e} kg·m²,  b = {B:.1e} N·m·s/rad");
    println!("  V_max = {V_MAX} V");
    print_tf("G_vθ(s)", &plant);
    println!("Denominator a0 = {d0} (free integrator at s = 0 is expected).");

    let full = ArrayPolynomial::<f64, 4>::from_coefficients([d0, d1, d2, d3]);
    match stability(&full, ROUTH_EPS) {
        Ok(n) => println!("Routh RHP count on 4-coeff den = {n}"),
        Err(err) => {
            println!("Routh on the 4-coeff den failed ({err}).");
            println!("Factoring out the integrator and testing the rest:");
            let rest =
                ArrayPolynomial::<f64, 3>::from_coefficients([d1, d2, d3]);
            match stability(&rest, ROUTH_EPS) {
                Ok(n) => println!("  remaining cubic RHP count = {n}"),
                Err(err2) => println!("  remaining cubic also failed: {err2}"),
            }
        }
    }

    let target_wc = 50.0;
    let design = synthesize_lead(&plant, target_wc, 50.0);
    println!("Lead C(s) targeting ω_c = {target_wc} rad/s, PM = 50°");
    println!(
        "  K = {:.6},  T = {:.6} s,  α = {:.6}",
        design.k, design.t, design.alpha
    );
    print_tf("C(s)", &design.tf);

    let kp = 10.0;
    let ki = 30.0;
    let kd = 0.3;
    let tf = 0.002;
    let mut pid = Pid::new(kp, ki, kd, tf, -V_MAX, V_MAX, 1.0);
    println!("Discrete PID  Kp={kp}, Ki={ki}, Kd={kd}, Tf={tf} s, Ts={TS} s");
    println!("  u clamped to ±{V_MAX} V, gamma = 1 (freeze integrator)");

    let setpoint = 1.0;
    let measurements = [0.0, 0.02, 0.05, 0.08, 0.12];
    for (step, &y) in measurements.iter().enumerate() {
        let u = pid.step(setpoint, y, TS);
        println!("  k={step}:  θ={y:.2} rad  u={u:.4} V");
    }
}
