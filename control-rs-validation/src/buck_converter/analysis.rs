//! src/analysis.rs
//!
//! Classical control analysis module for the buck converter validation example
//! (`documentation/control-toolboxes/classical-tools-design.md`, §6.6):
//! - Open-loop plant stability verification via Routh-Hurwitz.
//! - Frequency-domain stability margin extraction on the uncompensated plant.
//! - Rational lead compensator synthesis targeting bandwidth $\omega_{gc} \approx 3\omega_n$ and phase margin $\Phi_m \ge 45^\circ$.
//! - Compensated open-loop stability margins confirmation.
//! - Parameterized root locus closed-loop pole trajectory sweep.
//! - Closed-loop characteristic polynomial Routh-Hurwitz stability verification.

use control_rs::classical_tools::compensators::{CompensatorError, lead};
use control_rs::classical_tools::margins::{Margins, stability_margins};
use control_rs::classical_tools::root_locus::{RootLocusError, sweep};
use control_rs::classical_tools::routh::{RouthError, stability};
use control_rs::math::complex_num::Complex;
use control_rs::polynomial::ArrayPolynomial;
use control_rs::transfer_function::ArrayTransferFunction;

/// Averaged CCM plant $G_{vd}(s)$.
pub type PlantTf = ArrayTransferFunction<f64, 1, 3>;
/// First-order lead $C(s)$.
pub type LeadTf = ArrayTransferFunction<f64, 2, 2>;
/// Compensated loop $L(s) = C(s)G_{vd}(s)$.
pub type LoopTf = ArrayTransferFunction<f64, 2, 4>;
/// Three closed-loop poles at one locus gain.
pub type ClosedLoopPoles = [Complex<f64>; 3];
/// Real or imaginary parts of those poles along a gain sweep.
pub type LocusPoleParts = Vec<[f64; 3]>;
/// Closed-loop pole trajectory versus gain.
pub type RootLocusTrace = Vec<RootLocusPoint>;

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

/// Evaluates Routh-Hurwitz stability of the uncompensated buck converter plant.
///
/// Characteristic polynomial: $s^2 + 10^4 s + 10^8$ (ascending `[1e8, 1e4, 1.0]`).
///
/// # Errors
/// Returns [`RouthError`] if leading coefficient is zero or degeneracy is unresolvable.
pub fn analyze_plant_stability(plant: &PlantTf) -> Result<usize, RouthError> {
    let den = plant.den_slice();
    let poly =
        ArrayPolynomial::<f64, 3>::from_coefficients([den[0], den[1], den[2]]);
    stability(&poly, 1e-12)
}

/// Sweeps frequency response of the uncompensated plant and extracts stability margins.
///
/// Two-pole strictly proper plant never reaches $-180^\circ$ at finite frequency,
/// so `phase_crossover_freq` is expected to be `None`.
#[must_use]
pub fn analyze_plant_margins(plant: &PlantTf, omegas: &[f64]) -> Margins<f64> {
    stability_margins(plant, omegas)
}

/// Synthesized lead compensator parameters and rational transfer function.
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

/// Synthesizes a lead compensator placing the loop crossover near $3\omega_n$ ($30\,\text{krad/s}$)
/// with target phase margin $\ge 45^\circ$.
///
/// # Errors
/// Returns [`CompensatorError`] if $\alpha \ge 1$.
pub fn synthesize_lead_compensator(
    plant: &PlantTf,
    target_crossover_rad_s: f64,
    target_pm_deg: f64,
) -> Result<LeadDesign, CompensatorError> {
    // 1. Evaluate plant frequency response at the target crossover frequency.
    let g_resp = plant.eval_frequency(target_crossover_rad_s);
    let plant_mag = g_resp.magnitude();
    let plant_phase_rad = g_resp.arg();

    // 2. Uncompensated phase margin at target crossover:
    let uncomp_pm_rad = core::f64::consts::PI + plant_phase_rad;
    let target_pm_rad = target_pm_deg * core::f64::consts::PI / 180.0;

    // Required phase lead with 5° margin buffer for crossover shift:
    let phi_lead_rad = (target_pm_rad - uncomp_pm_rad
        + (5.0 * core::f64::consts::PI / 180.0))
        .clamp(0.1, 1.4);

    // 3. alpha from sin(phi_max) = (1 - alpha) / (1 + alpha)
    let sin_phi = phi_lead_rad.sin();
    let alpha = (1.0 - sin_phi) / (1.0 + sin_phi);

    // 4. Center frequency placed exactly at target crossover:
    let omega_m = target_crossover_rad_s;
    let t = 1.0 / (omega_m * alpha.sqrt());

    // 5. Compensator magnitude at omega_m is K * sqrt(alpha).
    // Unity loop gain |C(j omega_m)| * |G(j omega_m)| = 1 -> K = 1 / (sqrt(alpha) * plant_mag)
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

/// Compensated loop transfer function and its stability margins.
pub struct CompensatedLoop {
    /// $L(s) = C(s)G_{vd}(s)$.
    pub tf: LoopTf,
    /// Frequency-domain margins of `tf`.
    pub margins: Margins<f64>,
}

/// Evaluates stability margins for the compensated open-loop $L(s) = C(s) G_{vd}(s)$.
#[must_use]
pub fn analyze_compensated_loop_margins(
    compensator: &LeadTf,
    plant: &PlantTf,
    omegas: &[f64],
) -> CompensatedLoop {
    let tf = compensator.series::<1, 3, 2, 4>(plant);
    let margins = stability_margins(&tf, omegas);
    CompensatedLoop { tf, margins }
}

/// Root locus trajectory data point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootLocusPoint {
    /// Feedback gain $K$.
    pub gain: f64,
    /// Three closed-loop poles at this gain.
    pub poles: ClosedLoopPoles,
}

/// Sweeps feedback gain over the compensated loop and returns closed-loop pole trajectories.
///
/// # Errors
/// Returns [`RootLocusError`] if dimensions or root solving fail.
pub fn analyze_root_locus(
    plant: &PlantTf,
    lead_design: &LeadDesign,
    num_gain_steps: usize,
) -> Result<RootLocusTrace, RootLocusError> {
    // Unscaled lead compensator C0(s) with K = 1.0:
    let c0 = lead(1.0, lead_design.t, lead_design.alpha)
        .map_err(|_| RootLocusError::ImproperSystem)?;
    let l0 = c0.series::<1, 3, 2, 4>(plant);

    let max_gain = 2.0 * lead_design.k;
    let gains: Vec<f64> = (0..num_gain_steps)
        .map(|i| max_gain * (i as f64) / ((num_gain_steps.max(2) - 1) as f64))
        .collect();

    let mut out = vec![Complex::new(0.0, 0.0); gains.len() * 3];
    sweep(&l0, &gains, &mut out)?;

    let mut points = Vec::with_capacity(gains.len());
    for (i, &gain) in gains.iter().enumerate() {
        let p0 = out[i * 3];
        let p1 = out[i * 3 + 1];
        let p2 = out[i * 3 + 2];
        points.push(RootLocusPoint {
            gain,
            poles: [p0, p1, p2],
        });
    }

    Ok(points)
}

/// Evaluates Routh-Hurwitz stability of the closed-loop characteristic polynomial:
/// $P_{cl}(s) = D_L(s) + N_L(s)$.
///
/// # Errors
/// Returns [`RouthError`] if leading coefficient is zero or degeneracy occurs.
pub fn analyze_closed_loop_stability(
    loop_tf: &LoopTf,
) -> Result<usize, RouthError> {
    let num = loop_tf.num_slice();
    let den = loop_tf.den_slice();

    // Ascending polynomial addition:
    // P_cl(s) = (den[0] + num[0]) + (den[1] + num[1]) s + den[2] s^2 + den[3] s^3
    let cl_coeffs = [den[0] + num[0], den[1] + num[1], den[2], den[3]];
    let poly = ArrayPolynomial::<f64, 4>::from_coefficients(cl_coeffs);
    stability(&poly, 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buck_converter::circuit::plant;

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1
    /// Method: Requirements-based test
    fn test_uncompensated_plant_routh_is_zero() {
        let p = plant();
        let rhp_roots =
            analyze_plant_stability(&p).expect("Routh failed on plant");
        assert_eq!(rhp_roots, 0, "Uncompensated plant must have 0 RHP poles");
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1
    /// Method: Requirements-based test
    fn test_uncompensated_plant_has_no_phase_crossover() {
        let p = plant();
        let omegas = logspace_omegas(2.0, 6.0, 500);
        let margins = analyze_plant_margins(&p, &omegas);

        assert!(
            margins.phase_crossover_freq.is_none(),
            "2-pole all-pole plant should not have finite phase crossover"
        );
        assert!(margins.gain_margin.is_none());

        let wc = margins.gain_crossover_freq.expect("Gain crossover exists");
        assert!((wc - 3.6e4).abs() < 5.0e3, "wc = {wc}");

        let pm = margins.phase_margin.expect("Phase margin exists");
        let pm_deg = pm * 180.0 / core::f64::consts::PI;
        assert!(pm_deg < 25.0, "Uncompensated plant has poor PM: {pm_deg}°");
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1
    /// Method: Requirements-based test
    fn test_lead_compensator_design_achieves_targets() {
        let p = plant();
        let target_wc = 3.0e4; // 3 * omega_n
        let design = synthesize_lead_compensator(&p, target_wc, 50.0)
            .expect("Lead synthesis failed");

        assert!(design.alpha < 1.0);
        assert!(design.alpha > 0.1);
        assert!(design.k > 0.0);

        let omegas = logspace_omegas(2.0, 6.0, 1000);
        let compensated = analyze_compensated_loop_margins(
            &design.compensator_tf,
            &p,
            &omegas,
        );
        let loop_tf = compensated.tf;
        let margins = compensated.margins;

        let wc = margins
            .gain_crossover_freq
            .expect("Compensated loop must have gain crossover");
        let pm = margins
            .phase_margin
            .expect("Compensated loop must have phase margin");
        let pm_deg = pm * 180.0 / core::f64::consts::PI;

        // Verify §6.6 targets: wc ~ 3 omega_n (30 krad/s), PM >= 45 deg
        assert!(
            (wc - target_wc).abs() < 4.0e3,
            "Crossover {wc} near target {target_wc}"
        );
        assert!(pm_deg >= 45.0, "Compensated PM {pm_deg}° must be >= 45°");

        // Minimum-phase 2-pole plant + 1-pole/1-zero lead network has total phase > -180° for all w,
        // so phase crossover is None (infinite gain margin)
        assert!(margins.phase_crossover_freq.is_none());
        assert!(margins.gain_margin.is_none());

        // Verify closed-loop stability via Routh
        let cl_rhp = analyze_closed_loop_stability(&loop_tf)
            .expect("Closed-loop Routh failed");
        assert_eq!(cl_rhp, 0, "Closed loop must be strictly stable");
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1
    /// Method: Requirements-based test
    fn test_root_locus_poles_remain_in_lhp() {
        let p = plant();
        let design = synthesize_lead_compensator(&p, 3.0e4, 50.0)
            .expect("Lead synthesis failed");
        let locus = analyze_root_locus(&p, &design, 20)
            .expect("Root locus sweep failed");

        assert_eq!(locus.len(), 20);
        for pt in &locus {
            for pole in &pt.poles {
                assert!(
                    pole.re < -100.0,
                    "Pole {:?} at gain {} must be strictly in open LHP",
                    pole,
                    pt.gain
                );
            }
        }
    }
}
