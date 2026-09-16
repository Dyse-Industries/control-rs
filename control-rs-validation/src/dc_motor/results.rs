//! src/results.rs
//!
//! Single definition of the study configuration and of the analysis half of
//! the emitted payload: plant parameters, peripheral parameters, margins,
//! lead synthesis, frequency sweep and root locus.
//!
//! `main` and the cross-validation gate both build from here, so the payload
//! the gate checks is the payload the binary writes.

use control_rs::transfer_function::ArrayTransferFunction;
use serde_json::{Value, json};

use crate::dc_motor::analysis::{
    LocusPoleParts, RectPoles, analyze_closed_loop_stability,
    analyze_compensated_loop_margins, analyze_plant_margins,
    analyze_root_locus, delay_phase_loss_rad, logspace_omegas,
    synthesize_delay_compensated_lead,
};
use crate::dc_motor::motor::{B, J, K_B, K_T, L_A, R_A, V_MAX, plant_position};
use crate::dc_motor::peripherals::{
    ENCODER_BITS, ENCODER_COUNTS, ENCODER_QUANTUM,
};

/// Processor sample period $T_s$ (seconds), a 2 kHz loop rate.
pub const TS_S: f64 = 0.0005;
/// Actuator transport delay in sample periods.
pub const ACT_DELAY_STEPS: usize = 1;
/// Sensor transport delay in sample periods.
pub const SENS_DELAY_STEPS: usize = 1;
/// Actuator voltage noise standard deviation (V RMS).
pub const ACT_NOISE_SIGMA_V: f64 = 0.05;
/// Encoder angle noise standard deviation (rad RMS).
pub const SENS_NOISE_SIGMA_RAD: f64 = 1.0e-4;
/// Target loop gain crossover frequency (rad/s).
pub const TARGET_WC_RAD_S: f64 = 50.0;
/// Target phase margin (degrees).
pub const TARGET_PM_DEG: f64 = 50.0;
/// Frequency grid, shared with `python3/dc_motor_oracle.py`.
pub const OMEGA_LOG10_START: f64 = 0.0;
/// Upper decade of the frequency grid.
pub const OMEGA_LOG10_STOP: f64 = 4.0;
/// Number of frequency grid points.
pub const OMEGA_POINTS: usize = 1000;
/// Maximum complex-plane pole displacement between consecutive root locus samples.
pub const LOCUS_MAX_DISPLACEMENT: f64 = 1.0;
/// Maximum capacity of the adaptive root locus sweep buffer.
pub const LOCUS_CAPACITY: usize = 1000;

/// Total loop transport delay $\tau_d$ (seconds).
#[must_use]
pub fn total_delay_s() -> f64 {
    ((ACT_DELAY_STEPS + SENS_DELAY_STEPS) as f64) * TS_S
}

/// Outcome of the analysis phase.
pub struct AnalysisStudy {
    /// Discrete Tustin realization of the compensator at `TS_S`.
    pub compensator_discrete: ArrayTransferFunction<f64, 2, 2>,
    /// `plant`, `peripherals` and `analysis` subtrees of the emitted payload.
    pub payload: Value,
}

fn frequency_traces(
    plant: &crate::dc_motor::analysis::PlantTf,
    loop_tf: &crate::dc_motor::analysis::LoopTf,
    omegas: Vec<f64>,
    total_delay: f64,
) -> serde_json::Value {
    let mut plant_mag_db = Vec::with_capacity(omegas.len());
    let mut plant_phase_deg = Vec::with_capacity(omegas.len());
    let mut loop_mag_db = Vec::with_capacity(omegas.len());
    let mut loop_phase_deg = Vec::with_capacity(omegas.len());
    let mut delay_phase_deg = Vec::with_capacity(omegas.len());
    for &w in &omegas {
        let (m1, p1) = plant.bode_point(w);
        plant_mag_db.push(20.0 * m1.log10());
        plant_phase_deg.push(p1.to_degrees());
        let (m2, p2) = loop_tf.bode_point(w);
        loop_mag_db.push(20.0 * m2.log10());
        loop_phase_deg.push(p2.to_degrees());
        delay_phase_deg.push(delay_phase_loss_rad(w, total_delay).to_degrees());
    }
    json!({
        "omegas": omegas,
        "plant_mag_db": plant_mag_db,
        "plant_phase_deg": plant_phase_deg,
        "loop_mag_db": loop_mag_db,
        "loop_phase_deg": loop_phase_deg,
        "delay_phase_deg": delay_phase_deg,
    })
}

fn root_locus_payload(
    plant: &crate::dc_motor::analysis::PlantTf,
    lead_design: &crate::dc_motor::analysis::LeadDesign,
) -> serde_json::Value {
    let locus_points = analyze_root_locus(
        plant,
        lead_design,
        LOCUS_MAX_DISPLACEMENT,
        LOCUS_CAPACITY,
    )
    .expect("Root locus failed");
    let gains: Vec<f64> = locus_points.iter().map(|p| p.gain).collect();
    let poles_re: LocusPoleParts = locus_points
        .iter()
        .map(|p| [p.poles[0].re, p.poles[1].re, p.poles[2].re, p.poles[3].re])
        .collect();
    let poles_im: LocusPoleParts = locus_points
        .iter()
        .map(|p| [p.poles[0].im, p.poles[1].im, p.poles[2].im, p.poles[3].im])
        .collect();
    let mut final_poles: RectPoles = poles_re
        .last()
        .zip(poles_im.last())
        .map(|(re_row, im_row)| {
            re_row
                .iter()
                .zip(im_row.iter())
                .map(|(&re, &im)| [re, im])
                .collect::<RectPoles>()
        })
        .unwrap_or_default();
    final_poles
        .sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    json!({
        "gains": gains,
        "poles_re": poles_re,
        "poles_im": poles_im,
        "final_poles": final_poles,
    })
}

/// Runs the frequency-domain analysis and builds the analysis payload.
///
/// # Panics
/// Panics if lead synthesis, Routh analysis or the locus sweep fails, all of
/// which indicate a malformed plant rather than a runtime condition.
#[must_use]
pub fn run_analysis() -> AnalysisStudy {
    let plant = plant_position();
    let total_delay = total_delay_s();
    let omegas =
        logspace_omegas(OMEGA_LOG10_START, OMEGA_LOG10_STOP, OMEGA_POINTS);
    let plant_margins = analyze_plant_margins(&plant, &omegas);

    let lead_design = synthesize_delay_compensated_lead(
        &plant,
        TARGET_WC_RAD_S,
        TARGET_PM_DEG,
        total_delay,
    )
    .expect("Lead compensator synthesis failed");

    let compensated = analyze_compensated_loop_margins(
        &lead_design.compensator_tf,
        &plant,
        &omegas,
    );
    let loop_tf = compensated.tf;
    let loop_margins = compensated.margins;
    let cl_rhp = analyze_closed_loop_stability(&loop_tf)
        .expect("Routh stability analysis failed");
    assert_eq!(cl_rhp, 0, "compensated loop must have zero RHP poles");

    let frequency_sweep =
        frequency_traces(&plant, &loop_tf, omegas, total_delay);
    let root_locus = root_locus_payload(&plant, &lead_design);

    let compensator_discrete = lead_design
        .compensator_tf
        .to_discrete_tustin(TS_S, Some(TARGET_WC_RAD_S));
    let df2t =
        crate::dc_motor::controllers::lead_to_df2t(&compensator_discrete);

    let gain_margin_reason = |gm: Option<f64>| {
        if gm.is_none() {
            Some("no phase crossover (unbounded)")
        } else {
            None
        }
    };

    let payload = json!({
        "plant": {
            "r_a": R_A,
            "l_a": L_A,
            "k_t": K_T,
            "k_b": K_B,
            "j": J,
            "b": B,
            "v_max": V_MAX,
            "num_ascending": plant.num_slice(),
            "den_ascending": plant.den_slice(),
        },
        "peripherals": {
            "ts_s": TS_S,
            "actuator_delay_steps": ACT_DELAY_STEPS,
            "actuator_delay_s": (ACT_DELAY_STEPS as f64) * TS_S,
            "actuator_noise_sigma_v": ACT_NOISE_SIGMA_V,
            "encoder_bits": ENCODER_BITS,
            "encoder_counts": ENCODER_COUNTS,
            "encoder_quantum_rad": ENCODER_QUANTUM,
            "sensor_delay_steps": SENS_DELAY_STEPS,
            "sensor_delay_s": (SENS_DELAY_STEPS as f64) * TS_S,
            "sensor_noise_sigma_rad": SENS_NOISE_SIGMA_RAD,
            "total_loop_delay_s": total_delay,
        },
        "analysis": {
            "uncompensated_margins": {
                "gain_crossover_rad_s": plant_margins.gain_crossover_freq,
                "phase_margin_deg": plant_margins.phase_margin.map(f64::to_degrees),
                "phase_crossover_rad_s": plant_margins.phase_crossover_freq,
                "gain_margin_db": plant_margins.gain_margin.map(|gm| 20.0 * gm.log10()),
                "gain_margin_reason": gain_margin_reason(plant_margins.gain_margin),
            },
            "lead_design": {
                "gain_k": lead_design.k,
                "time_constant_t_s": lead_design.t,
                "attenuation_alpha": lead_design.alpha,
                "center_freq_rad_s": lead_design.omega_m,
                "max_phase_lead_deg": lead_design.max_phase_lead_rad.to_degrees(),
                "df2t_b0": df2t.b0,
                "df2t_b1": df2t.b[0],
                "df2t_a1": df2t.a[0],
            },
            "compensated_margins": {
                "gain_crossover_rad_s": loop_margins.gain_crossover_freq,
                "phase_margin_deg": loop_margins.phase_margin.map(f64::to_degrees),
                "phase_crossover_rad_s": loop_margins.phase_crossover_freq,
                "gain_margin_db": loop_margins.gain_margin.map(|gm| 20.0 * gm.log10()),
                "gain_margin_reason": gain_margin_reason(loop_margins.gain_margin),
                "delay_margin_us": loop_margins.delay_margin.map(|dm| dm * 1e6),
                "closed_loop_rhp_poles": cl_rhp,
            },
            "frequency_sweep": frequency_sweep,
            "root_locus": root_locus,
        },
    });

    AnalysisStudy {
        compensator_discrete,
        payload,
    }
}
