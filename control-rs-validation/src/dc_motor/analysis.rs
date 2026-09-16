//! src/analysis.rs
//!
//! Classical control analysis and compensator synthesis for the DC motor position servo:
//! - Frequency-domain stability margin extraction on uncompensated motor plant.
//! - Analytical delay phase loss calculation ($\Delta \phi = -\omega \tau_d$).
//! - Analytical lead compensator synthesis restoring phase margin lost to delay.
//! - Closed-loop Routh-Hurwitz stability verification.
//! - Root locus closed-loop pole trajectory sweeps.

use control_rs::classical_tools::compensators::{CompensatorError, lead};
use control_rs::classical_tools::margins::{Margins, stability_margins};
use control_rs::classical_tools::root_locus::{RootLocusError, sweep_adaptive};
use control_rs::classical_tools::routh::{RouthError, stability};
use control_rs::math::complex_num::Complex;
use control_rs::polynomial::ArrayPolynomial;
use control_rs::transfer_function::ArrayTransferFunction;

/// Control-to-position plant $G_{v\theta}(s)$.
pub type PlantTf = ArrayTransferFunction<f64, 1, 4>;
/// First-order lead $C(s)$.
pub type LeadTf = ArrayTransferFunction<f64, 2, 2>;
/// Compensated loop $L(s) = C(s)G_{v\theta}(s)$.
pub type LoopTf = ArrayTransferFunction<f64, 2, 5>;
/// Four closed-loop poles at one locus gain.
pub type ClosedLoopPoles = [Complex<f64>; 4];
/// Real or imaginary parts of those poles along a gain sweep.
pub type LocusPoleParts = Vec<[f64; 4]>;
/// Cartesian $(re, im)$ pairs of the locus terminal poles.
pub type RectPoles = Vec<[f64; 2]>;
/// Closed-loop pole trajectory versus gain.
pub type RootLocusTrace = Vec<RootLocusPoint>;

/// Compensated loop transfer function and its stability margins.
pub struct CompensatedLoop {
    /// $L(s) = C(s)G_{v\theta}(s)$.
    pub tf: LoopTf,
    /// Frequency-domain margins of `tf`.
    pub margins: Margins<f64>,
}

/// Generates a logarithmically spaced frequency vector with `n` points between
/// $10^{\text{start\_log10}}$ and $10^{\text{stop\_log10}}$ rad/s.
#[must_use]
pub fn logspace_omegas(
    start_log10: f64,
    stop_log10: f64,
    n: usize,
) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![10.0_f64.powf(stop_log10)];
    }
    let den = (n - 1) as f64;
    (0..n)
        .map(|i| {
            10.0_f64.powf(
                start_log10 + (stop_log10 - start_log10) * (i as f64) / den,
            )
        })
        .collect()
}

/// Evaluates frequency-domain stability margins of the uncompensated plant.
#[must_use]
pub fn analyze_plant_margins(plant: &PlantTf, omegas: &[f64]) -> Margins<f64> {
    stability_margins(plant, omegas)
}

/// Calculates the phase loss in radians introduced by pure transport delay $\tau_d$ at frequency $\omega$:
/// $$\Delta \phi(\omega) = -\omega \tau_d$$
#[must_use]
pub fn delay_phase_loss_rad(omega_rad_s: f64, total_delay_s: f64) -> f64 {
    -omega_rad_s * total_delay_s
}

/// Synthesized lead compensator parameters and continuous rational transfer function.
#[derive(Debug, Clone)]
pub struct LeadDesign {
    /// Compensator gain $K$.
    pub k: f64,
    /// Time constant $T$ (seconds).
    pub t: f64,
    /// Attenuation factor $\alpha < 1$.
    pub alpha: f64,
    /// Center frequency $\omega_m = 1 / (T\sqrt{\alpha})$ (rad/s).
    pub omega_m: f64,
    /// Phase lead at center frequency $\phi_{\max}$ (radians).
    pub max_phase_lead_rad: f64,
    /// Rational transfer function $C(s) = K \frac{s + 1/T}{s + 1/(\alpha T)}$.
    pub compensator_tf: LeadTf,
}

/// Synthesizes an analytical lead compensator that places crossover at `target_crossover_rad_s`
/// and explicitly offsets the phase loss $-\omega_{gc} \tau_d$ from loop delay, achieving `target_pm_deg`.
///
/// # Errors
/// Returns [`CompensatorError`] if required lead $\alpha \ge 1$.
pub fn synthesize_delay_compensated_lead(
    plant: &PlantTf,
    target_crossover_rad_s: f64,
    target_pm_deg: f64,
    total_loop_delay_s: f64,
) -> Result<LeadDesign, CompensatorError> {
    // 1. Evaluate plant magnitude and phase at target crossover frequency
    let g_resp = plant.eval_frequency(target_crossover_rad_s);
    let plant_mag = g_resp.magnitude();
    let plant_phase_rad = g_resp.arg();

    // 2. Compute delay phase loss: Delta phi = -omega * tau_d
    let delay_phase_rad =
        delay_phase_loss_rad(target_crossover_rad_s, total_loop_delay_s);
    let effective_plant_phase = plant_phase_rad + delay_phase_rad;

    // 3. Uncompensated phase margin with delay: PM = pi + effective_phase
    let uncomp_pm_rad = core::f64::consts::PI + effective_plant_phase;
    let target_pm_rad = target_pm_deg * core::f64::consts::PI / 180.0;

    // 4. Required phase lead boost (with 5° safety buffer to offset crossover shifting)
    let safety_buffer_rad = 5.0 * core::f64::consts::PI / 180.0;
    let phi_lead_rad =
        (target_pm_rad - uncomp_pm_rad + safety_buffer_rad).clamp(0.1, 1.45);

    // 5. Attenuation factor alpha: sin(phi_max) = (1 - alpha) / (1 + alpha)
    let sin_phi = phi_lead_rad.sin();
    let alpha = (1.0 - sin_phi) / (1.0 + sin_phi);

    // 6. Center frequency at target crossover: omega_m = 1 / (T * sqrt(alpha))
    let omega_m = target_crossover_rad_s;
    let t = 1.0 / (omega_m * alpha.sqrt());

    // 7. Unity loop gain at omega_m: |C(j omega_m)| * |G(j omega_m)| = 1
    // |C(j omega_m)| = K * sqrt(alpha) -> K = 1 / (sqrt(alpha) * plant_mag)
    let k = 1.0 / (alpha.sqrt() * plant_mag);

    let compensator_tf = lead(k, t, alpha)?;

    Ok(LeadDesign {
        k,
        t,
        alpha,
        omega_m,
        max_phase_lead_rad: phi_lead_rad,
        compensator_tf,
    })
}

/// Evaluates stability margins for the compensated open loop $L(s) = C(s) G_{v\theta}(s)$.
#[must_use]
pub fn analyze_compensated_loop_margins(
    compensator: &LeadTf,
    plant: &PlantTf,
    omegas: &[f64],
) -> CompensatedLoop {
    let tf = compensator.series::<1, 4, 2, 5>(plant);
    let margins = stability_margins(&tf, omegas);
    CompensatedLoop { tf, margins }
}

/// Evaluates Routh-Hurwitz stability of the closed-loop characteristic polynomial
/// $P_{cl}(s) = D_L(s) + N_L(s)$ for $L(s) = C(s) G_{v\theta}(s)$.
///
/// # Errors
/// Returns [`RouthError`] if leading coefficient is zero or degeneracy is unresolvable.
pub fn analyze_closed_loop_stability(
    loop_tf: &LoopTf,
) -> Result<usize, RouthError> {
    let num = loop_tf.num_slice();
    let den = loop_tf.den_slice();

    // P_cl(s) = D_L(s) + N_L(s)
    let p0 = den[0] + num[0];
    let p1 = den[1] + num[1];
    let p2 = den[2];
    let p3 = den[3];
    let p4 = den[4];

    let poly =
        ArrayPolynomial::<f64, 5>::from_coefficients([p0, p1, p2, p3, p4]);
    stability(&poly, 1e-12)
}

/// Point on the root locus pole trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootLocusPoint {
    /// Feedback loop gain multiplier $K$.
    pub gain: f64,
    /// Closed-loop poles (4 roots for 4th-order compensated system).
    pub poles: ClosedLoopPoles,
}

/// Sweeps closed-loop pole trajectories adaptively over the gain range $0 \le K \le 2 K_{\text{nom}}$,
/// bounding consecutive pole displacements to `max_displacement`.
///
/// # Errors
/// Returns [`RootLocusError`] if pole extraction fails.
pub fn analyze_root_locus(
    plant: &PlantTf,
    lead_design: &LeadDesign,
    max_displacement: f64,
    capacity: usize,
) -> Result<RootLocusTrace, RootLocusError> {
    let unscaled_lead = lead(1.0, lead_design.t, lead_design.alpha)
        .map_err(|_| RootLocusError::ImproperSystem)?;
    let unscaled_loop = unscaled_lead.series::<1, 4, 2, 5>(plant);

    let max_gain = 2.0 * lead_design.k;
    let mut gains_buf = vec![0.0; capacity];
    let mut roots_buf = vec![Complex::new(0.0, 0.0); capacity * 4];

    let count = sweep_adaptive(
        &unscaled_loop,
        (0.0, max_gain),
        max_displacement,
        &mut gains_buf,
        &mut roots_buf,
    )?;

    let mut points = Vec::with_capacity(count);
    for (i, &gain) in gains_buf.iter().enumerate().take(count) {
        let base = i * 4;
        points.push(RootLocusPoint {
            gain,
            poles: [
                roots_buf[base],
                roots_buf[base + 1],
                roots_buf[base + 2],
                roots_buf[base + 3],
            ],
        });
    }

    Ok(points)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dc_motor::motor::plant_position;

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_delay_phase_loss_calculation() {
        let delay_s = 0.001; // 1 ms
        let omega = 100.0; // 100 rad/s
        let loss = delay_phase_loss_rad(omega, delay_s);
        // Delta phi = -100 * 0.001 = -0.1 rad (~ -5.73 deg)
        assert!((loss - (-0.1)).abs() < 1e-12);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_synthesize_delay_compensated_lead() {
        let plant = plant_position();
        let target_wc = 50.0; // 50 rad/s crossover
        let target_pm = 50.0; // 50 degrees phase margin
        let total_delay_s = 0.001; // 1 ms loop delay

        let design = synthesize_delay_compensated_lead(
            &plant,
            target_wc,
            target_pm,
            total_delay_s,
        )
        .expect("Compensator synthesis should succeed");

        assert!(
            design.alpha > 0.0 && design.alpha < 1.0,
            "alpha must be in (0, 1)"
        );
        assert!(design.k > 0.0, "Gain must be positive");
        assert_eq!(design.omega_m, target_wc);

        // Verify compensated loop phase margin meets target
        let omegas = logspace_omegas(1.0, 3.0, 500);
        let compensated = analyze_compensated_loop_margins(
            &design.compensator_tf,
            &plant,
            &omegas,
        );
        let loop_tf = compensated.tf;
        let margins = compensated.margins;

        assert!(margins.phase_margin.is_some());
        let pm_deg =
            margins.phase_margin.unwrap() * 180.0 / core::f64::consts::PI;
        assert!(
            pm_deg >= 45.0,
            "Compensated phase margin should meet or exceed 45 deg, got: {:.2}",
            pm_deg
        );

        // Verify closed-loop stability via Routh-Hurwitz
        let rhp = analyze_closed_loop_stability(&loop_tf)
            .expect("Routh check should succeed");
        assert_eq!(rhp, 0, "Closed loop must have 0 RHP poles");
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_root_locus_sweep_produces_valid_poles() {
        let plant = plant_position();
        let design =
            synthesize_delay_compensated_lead(&plant, 50.0, 50.0, 0.001)
                .expect("Lead synthesis should succeed");

        let locus = analyze_root_locus(&plant, &design, 1.0, 1000)
            .expect("Root locus should succeed");
        assert!(locus.len() > 100);

        for pt in &locus {
            for pole in &pt.poles {
                // At nominal gains, closed loop poles must remain in open LHP
                if pt.gain > 0.0 && pt.gain <= design.k {
                    assert!(
                        pole.re <= 1e-6,
                        "Pole should be in LHP, got re={:.4} at gain={:.4}",
                        pole.re,
                        pt.gain
                    );
                }
            }
        }
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_sweep_adaptive_dc_motor_continuity() {
        let plant = plant_position();
        let design =
            synthesize_delay_compensated_lead(&plant, 50.0, 50.0, 0.001)
                .expect("Lead synthesis should succeed");

        let locus = analyze_root_locus(&plant, &design, 1.0, 1000)
            .expect("Root locus should succeed");

        assert!(locus.len() > 100);
        let max_gain = 2.0 * design.k;
        assert!((locus.last().unwrap().gain - max_gain).abs() < 1e-9);

        // Verify z-plane continuity across every branch
        let ts = 0.0005_f64;
        let mut max_dz = [0.0_f64; 4];
        for i in 1..locus.len() {
            for (b, max_dz_val) in max_dz.iter_mut().enumerate() {
                let p_prev = locus[i - 1].poles[b];
                let p_curr = locus[i].poles[b];

                let mag_prev = (p_prev.re * ts).exp();
                let z_prev_re = mag_prev * (p_prev.im * ts).cos();
                let z_prev_im = mag_prev * (p_prev.im * ts).sin();

                let mag_curr = (p_curr.re * ts).exp();
                let z_curr_re = mag_curr * (p_curr.im * ts).cos();
                let z_curr_im = mag_curr * (p_curr.im * ts).sin();

                let dz = ((z_curr_re - z_prev_re).powi(2)
                    + (z_curr_im - z_prev_im).powi(2))
                .sqrt();
                if dz > *max_dz_val {
                    *max_dz_val = dz;
                }
            }
        }

        for &dz in &max_dz {
            assert!(
                dz < 0.001,
                "Branch jump in z-plane too large: {dz} (must be < 0.001)"
            );
        }
    }
}
