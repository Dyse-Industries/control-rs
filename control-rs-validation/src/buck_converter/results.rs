//! src/results.rs
//!
//! Single definition of the buck converter study configuration, analysis,
//! and simulations.
//!
//! `main` and the cross-validation gate both build from here, so the payload
//! the gate checks is the payload the binary writes.

use serde::Serialize;
use serde_json::Value;

use crate::buck_converter::analysis::{
    LocusPoleParts, analyze_closed_loop_stability,
    analyze_compensated_loop_margins, analyze_plant_margins,
    analyze_plant_stability, analyze_root_locus, logspace_omegas,
    synthesize_lead_compensator,
};
use crate::buck_converter::circuit::{
    CAPACITANCE, INDUCTANCE, LOAD_RESISTANCE, V_IN, V_OUT, damping_ratio,
    natural_frequency, operating_duty_cycle, plant,
};
use crate::buck_converter::simulation::{
    LoadStep, LoadTrajectory, SampledHorizon, SimulationTrajectory,
    VoltageStep, controller_to_biquad, controller_to_df2t,
    discretize_compensator, simulate_linear_closed_loop,
    simulate_load_transient, simulate_nonlinear_circuit,
};

/// Processor sample period $T_s$ in microseconds (10 us, 100 kHz rate).
pub const TS_US: f64 = 10.0;
/// Processor sample period $T_s$ in seconds ($10 \times 10^{-6}$ s).
pub const TS_S: f64 = TS_US * 1.0e-6;
/// Target loop gain crossover frequency (rad/s) = 3 * omega_n = 30 krad/s.
pub const TARGET_WC_RAD_S: f64 = 3.0e4;
/// Target phase margin (degrees).
pub const TARGET_PM_DEG: f64 = 50.0;
/// Frequency grid start decade (10^2 rad/s).
pub const OMEGA_LOG10_START: f64 = 2.0;
/// Frequency grid stop decade (10^6 rad/s).
pub const OMEGA_LOG10_STOP: f64 = 6.0;
/// Number of frequency grid points.
pub const OMEGA_POINTS: usize = 1000;
/// Number of gains in the root locus sweep.
pub const LOCUS_POINTS: usize = 100;

/// Simulation duration for step response (seconds).
pub const SIM_T_FINAL_S: f64 = 0.003;
/// Step disturbance / change injection time (seconds).
pub const SIM_STEP_TIME_S: f64 = 0.0005;
/// Step setpoint voltage change $\Delta V$ (Volts).
pub const SIM_STEP_DELTA_V: f64 = 0.5;
/// Load transient disturbance injection time (seconds).
pub const SIM_LOAD_STEP_TIME_S: f64 = 0.001;
/// Load transient resistance step $\Delta R$ (Ohms).
pub const SIM_LOAD_DELTA_R: f64 = 2.0;

/// Strongly-typed results structure matching `buck_converter.json`.
#[derive(Serialize)]
pub struct BuckResults {
    pub plant: PlantData,
    pub analysis: AnalysisData,
    pub margins: MarginsSetData,
    pub compensator: LeadDesignData,
    pub frequency_sweep: FrequencySweepData,
    pub root_locus: RootLocusData,
    pub simulation: SimulationData,
}

#[derive(Clone, Serialize)]
pub struct PlantData {
    pub v_in: f64,
    pub inductance_h: f64,
    pub capacitance_f: f64,
    pub load_resistance_ohm: f64,
    pub v_out_v: f64,
    pub operating_duty_cycle: f64,
    pub natural_frequency_rad_s: f64,
    pub damping_ratio: f64,
    pub num_slice: Vec<f64>,
    pub den_slice: Vec<f64>,
    pub open_loop_rhp_poles: usize,
    #[serde(rename = "routh_rhp_roots")]
    pub routh_rhp_roots_alias: usize,
}

#[derive(Clone, Serialize)]
pub struct AnalysisData {
    pub uncompensated_margins: MarginsData,
    pub lead_design: LeadDesignData,
    pub compensated_margins: MarginsData,
    pub frequency_sweep: FrequencySweepData,
    pub root_locus: RootLocusData,
}

#[derive(Clone, Serialize)]
pub struct MarginsData {
    pub gain_crossover_rad_s: Option<f64>,
    pub phase_margin_deg: Option<f64>,
    pub phase_crossover_rad_s: Option<f64>,
    pub gain_margin_db: Option<f64>,
    pub gain_margin_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delay_margin_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub closed_loop_rhp_poles: Option<usize>,
}

#[derive(Clone, Serialize)]
pub struct MarginsSetData {
    pub uncompensated: MarginsData,
    pub compensated: MarginsData,
}

#[derive(Clone, Serialize)]
pub struct LeadDesignData {
    pub gain_k: f64,
    pub time_constant_t_s: f64,
    pub attenuation_alpha: f64,
    pub zero_rad_s: f64,
    pub pole_rad_s: f64,
    pub center_freq_rad_s: f64,
    pub max_phase_lead_deg: f64,
    pub num_slice: Vec<f64>,
    pub den_slice: Vec<f64>,
    pub discrete_ts_us: f64,
    pub df2t_b0: f64,
    pub df2t_b1: f64,
    pub df2t_a1: f64,
}

#[derive(Clone, Serialize)]
pub struct FrequencySweepData {
    pub omegas: Vec<f64>,
    pub uncomp_mag_db: Vec<f64>,
    pub uncomp_phase_deg: Vec<f64>,
    pub comp_mag_db: Vec<f64>,
    pub comp_phase_deg: Vec<f64>,
    // Aliases for unified schema parity with dc-motor:
    pub plant_mag_db: Vec<f64>,
    pub plant_phase_deg: Vec<f64>,
    pub loop_mag_db: Vec<f64>,
    pub loop_phase_deg: Vec<f64>,
}

#[derive(Clone, Serialize)]
pub struct RootLocusData {
    pub gains: Vec<f64>,
    pub poles_re: LocusPoleParts,
    pub poles_im: LocusPoleParts,
    pub poles_re_sorted: LocusPoleParts,
    pub poles_im_sorted: LocusPoleParts,
}

#[derive(Clone, Serialize)]
pub struct SimulationData {
    pub linear_step: SimulationTrajectory,
    pub nonlinear_step: SimulationTrajectory,
    pub load_transient: LoadTrajectory,
    #[serde(rename = "load_disturbance")]
    pub load_disturbance_alias: LoadTrajectory,
}

/// Outcome of the analysis and simulation study.
pub struct AnalysisStudy {
    /// Strongly-typed results structure matching `buck_converter.json`.
    pub results: BuckResults,
    /// Emitted payload as a `serde_json::Value`.
    pub payload: Value,
}

fn frequency_traces(
    plant: &crate::buck_converter::analysis::PlantTf,
    loop_tf: &crate::buck_converter::analysis::LoopTf,
    omegas: Vec<f64>,
) -> FrequencySweepData {
    let mut uncomp_mag_db = Vec::with_capacity(omegas.len());
    let mut uncomp_phase_deg = Vec::with_capacity(omegas.len());
    let mut comp_mag_db = Vec::with_capacity(omegas.len());
    let mut comp_phase_deg = Vec::with_capacity(omegas.len());
    for &w in &omegas {
        let (m1, p1) = plant.bode_point(w);
        uncomp_mag_db.push(20.0 * m1.log10());
        uncomp_phase_deg.push(p1 * 180.0 / core::f64::consts::PI);
        let (m2, p2) = loop_tf.bode_point(w);
        comp_mag_db.push(20.0 * m2.log10());
        comp_phase_deg.push(p2 * 180.0 / core::f64::consts::PI);
    }
    FrequencySweepData {
        omegas,
        uncomp_mag_db: uncomp_mag_db.clone(),
        uncomp_phase_deg: uncomp_phase_deg.clone(),
        comp_mag_db: comp_mag_db.clone(),
        comp_phase_deg: comp_phase_deg.clone(),
        plant_mag_db: uncomp_mag_db,
        plant_phase_deg: uncomp_phase_deg,
        loop_mag_db: comp_mag_db,
        loop_phase_deg: comp_phase_deg,
    }
}

fn root_locus_data(
    plant: &crate::buck_converter::analysis::PlantTf,
    lead_design: &crate::buck_converter::analysis::LeadDesign,
) -> RootLocusData {
    let locus_full = analyze_root_locus(plant, lead_design, LOCUS_POINTS)
        .expect("Root locus sweep failed");
    let gains: Vec<f64> = locus_full.iter().map(|p| p.gain).collect();
    let poles_re: LocusPoleParts = locus_full
        .iter()
        .map(|p| [p.poles[0].re, p.poles[1].re, p.poles[2].re])
        .collect();
    let poles_im: LocusPoleParts = locus_full
        .iter()
        .map(|p| [p.poles[0].im, p.poles[1].im, p.poles[2].im])
        .collect();
    let mut poles_re_sorted = poles_re.clone();
    let mut poles_im_sorted = poles_im.clone();
    for (re, im) in poles_re_sorted.iter_mut().zip(poles_im_sorted.iter_mut()) {
        let mut idx = [0usize, 1, 2];
        idx.sort_by(|&a, &b| {
            re[a]
                .partial_cmp(&re[b])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(
                    im[a]
                        .partial_cmp(&im[b])
                        .unwrap_or(core::cmp::Ordering::Equal),
                )
        });
        let re_s = [re[idx[0]], re[idx[1]], re[idx[2]]];
        let im_s = [im[idx[0]], im[idx[1]], im[idx[2]]];
        *re = re_s;
        *im = im_s;
    }
    RootLocusData {
        gains,
        poles_re,
        poles_im,
        poles_re_sorted,
        poles_im_sorted,
    }
}

fn margins_data(
    margins: &control_rs::classical_tools::margins::Margins<f64>,
    delay_margin_us: Option<f64>,
    closed_loop_rhp_poles: Option<usize>,
) -> MarginsData {
    MarginsData {
        gain_crossover_rad_s: margins.gain_crossover_freq,
        phase_margin_deg: margins
            .phase_margin
            .map(|pm| pm * 180.0 / core::f64::consts::PI),
        phase_crossover_rad_s: margins.phase_crossover_freq,
        gain_margin_db: margins.gain_margin.map(|gm| 20.0 * gm.log10()),
        gain_margin_reason: if margins.gain_margin.is_none() {
            Some("no phase crossover (unbounded)")
        } else {
            None
        },
        delay_margin_us,
        closed_loop_rhp_poles,
    }
}

/// Runs the classical control analysis and closed-loop simulations, building
/// the emitted study results and payload.
///
/// # Panics
/// Panics if lead synthesis, Routh analysis, or the root locus sweep fails,
/// all of which indicate a malformed plant rather than a runtime condition.
#[must_use]
pub fn run_analysis() -> AnalysisStudy {
    let plant = plant();
    let plant_rhp = analyze_plant_stability(&plant)
        .expect("Routh analysis failed on plant");
    assert_eq!(plant_rhp, 0);

    let omegas =
        logspace_omegas(OMEGA_LOG10_START, OMEGA_LOG10_STOP, OMEGA_POINTS);
    let plant_margins = analyze_plant_margins(&plant, &omegas);

    let lead_design =
        synthesize_lead_compensator(&plant, TARGET_WC_RAD_S, TARGET_PM_DEG)
            .expect("Compensator synthesis failed");

    let compensated = analyze_compensated_loop_margins(
        &lead_design.compensator_tf,
        &plant,
        &omegas,
    );
    let loop_tf = compensated.tf;
    let loop_margins = compensated.margins;
    let delay_margin_us = loop_margins.delay_margin.unwrap_or(0.0) * 1.0e6;

    let cl_rhp = analyze_closed_loop_stability(&loop_tf)
        .expect("Closed loop Routh check failed");
    assert_eq!(cl_rhp, 0);

    let c_discrete = discretize_compensator(
        &lead_design.compensator_tf,
        TS_S,
        Some(TARGET_WC_RAD_S),
    );

    let df2t = controller_to_df2t(&c_discrete);
    let _ = controller_to_biquad(&c_discrete);

    let horizon = SampledHorizon {
        total_time_s: SIM_T_FINAL_S,
        sample_time_s: TS_S,
    };
    let volt_step = VoltageStep {
        time_s: SIM_STEP_TIME_S,
        delta_v: SIM_STEP_DELTA_V,
    };
    let sim_lin = simulate_linear_closed_loop(
        &lead_design.compensator_tf,
        horizon,
        volt_step,
    );
    let sim_nonlin = simulate_nonlinear_circuit(
        &lead_design.compensator_tf,
        horizon,
        volt_step,
        None,
    );
    let sim_load = simulate_load_transient(
        &lead_design.compensator_tf,
        horizon,
        LoadStep {
            time_s: SIM_LOAD_STEP_TIME_S,
            resistance_ohm: SIM_LOAD_DELTA_R,
        },
    );

    let freq_data = frequency_traces(&plant, &loop_tf, omegas);
    let locus_data = root_locus_data(&plant, &lead_design);
    let uncomp_margins_data = margins_data(&plant_margins, None, None);
    let comp_margins_data =
        margins_data(&loop_margins, Some(delay_margin_us), Some(cl_rhp));

    let lead_design_data = LeadDesignData {
        gain_k: lead_design.k,
        time_constant_t_s: lead_design.t,
        attenuation_alpha: lead_design.alpha,
        zero_rad_s: 1.0 / lead_design.t,
        pole_rad_s: 1.0 / (lead_design.alpha * lead_design.t),
        center_freq_rad_s: lead_design.omega_m,
        max_phase_lead_deg: lead_design.max_phase_lead_rad * 180.0
            / core::f64::consts::PI,
        num_slice: lead_design.compensator_tf.num_slice().to_vec(),
        den_slice: lead_design.compensator_tf.den_slice().to_vec(),
        discrete_ts_us: TS_US,
        df2t_b0: df2t.b0,
        df2t_b1: df2t.b[0],
        df2t_a1: df2t.a[0],
    };

    let sim_data = SimulationData {
        linear_step: sim_lin,
        nonlinear_step: sim_nonlin,
        load_transient: sim_load.clone(),
        load_disturbance_alias: sim_load,
    };

    let plant_data = PlantData {
        v_in: V_IN,
        inductance_h: INDUCTANCE,
        capacitance_f: CAPACITANCE,
        load_resistance_ohm: LOAD_RESISTANCE,
        v_out_v: V_OUT,
        operating_duty_cycle: operating_duty_cycle(),
        natural_frequency_rad_s: natural_frequency(),
        damping_ratio: damping_ratio(),
        num_slice: plant.num_slice().to_vec(),
        den_slice: plant.den_slice().to_vec(),
        open_loop_rhp_poles: plant_rhp,
        routh_rhp_roots_alias: plant_rhp,
    };

    let results = BuckResults {
        plant: plant_data,
        analysis: AnalysisData {
            uncompensated_margins: uncomp_margins_data.clone(),
            lead_design: lead_design_data.clone(),
            compensated_margins: comp_margins_data.clone(),
            frequency_sweep: freq_data.clone(),
            root_locus: locus_data.clone(),
        },
        margins: MarginsSetData {
            uncompensated: uncomp_margins_data,
            compensated: comp_margins_data,
        },
        compensator: lead_design_data,
        frequency_sweep: freq_data,
        root_locus: locus_data,
        simulation: sim_data,
    };

    let payload = serde_json::to_value(&results)
        .expect("Failed to serialize BuckResults to Value");

    AnalysisStudy { results, payload }
}
