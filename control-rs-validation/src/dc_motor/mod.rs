//! src/main.rs
//!
//! DC motor armature position servo classical control example with:
//! - Peripheral delay ($\tau_{\text{act}} = 1 \cdot T_s$)
//! - Noisy actuator (voltage noise + saturation)
//! - 16-bit absolute position encoder sensor with delay ($\tau_{\text{sens}} = 1 \cdot T_s$) and measurement noise
//! - Processor executing dual discrete controllers:
//!   - Controller A: Analytical Lead Compensator realized via [`DirectForm2T`]
//!   - Controller B: Discrete PID with Filtered Derivative via [`Pid`]

mod analysis;
mod controllers;
mod motor;
mod oracle;
mod peripherals;
mod results;
mod simulation;

use controllers::{
    LeadDf2tController, MotorController, PidGains, PidMotorController,
};
use motor::{B, J, K_B, K_T, L_A, R_A, V_MAX};
use peripherals::{
    Actuator, ENCODER_BITS, ENCODER_COUNTS, ENCODER_QUANTUM, EncoderSensor,
};
use simulation::{
    LoadDisturbance, PositionStep, SampledHorizon, ServoScenario,
    simulate_ideal_baseline, simulate_motor_system,
};

use std::path::{Path, PathBuf};

/// Run the DC-motor servo suite and emit `results/dc-motor.rust.h5`.
pub fn run() {
    // -------------------------------------------------------------------------
    // 1. Startup Information
    // -------------------------------------------------------------------------
    println!("Plant & Circuit Parameters:");
    println!("  Armature Resistance (R_a)    : {:.2} Ohm", R_A);
    println!(
        "  Armature Inductance (L_a)    : {:.1e} H ({:.1} mH)",
        L_A,
        L_A * 1e3
    );
    println!("  Torque Constant (K_t)        : {:.4} N*m/A", K_T);
    println!("  Back-EMF Constant (K_b)      : {:.4} V*s/rad", K_B);
    println!("  Rotor Inertia (J)            : {:.1e} kg*m^2", J);
    println!("  Viscous Damping (b)          : {:.1e} N*m*s/rad", B);
    println!("  Terminal Supply Limit (V_max): {:.1} V\n", V_MAX);

    let ts_s = results::TS_S;
    let act_delay_steps = results::ACT_DELAY_STEPS;
    let sens_delay_steps = results::SENS_DELAY_STEPS;
    let total_delay_s = results::total_delay_s();
    let act_noise_sigma_v = results::ACT_NOISE_SIGMA_V;
    let sens_noise_sigma_rad = results::SENS_NOISE_SIGMA_RAD;

    println!("Embedded System & Peripheral Parameters:");
    println!(
        "  Processor Sampling Rate (f_s): {:.1} kHz (T_s = {:.0} us)",
        1.0 / ts_s / 1e3,
        ts_s * 1e6
    );
    println!(
        "  Actuator Delay (tau_act)     : {} step ({:.0} us)",
        act_delay_steps,
        (act_delay_steps as f64) * ts_s * 1e6
    );
    println!(
        "  Actuator Noise (sigma_u)     : {:.3} V RMS (clamped to +/- {:.1} V)",
        act_noise_sigma_v, V_MAX
    );
    println!(
        "  Sensor Type                  : Absolute Position Encoder ({} bits)",
        ENCODER_BITS
    );
    println!(
        "  Encoder Resolution           : {:.0} counts/turn (LSB = {:.3e} rad / {:.4}°)",
        ENCODER_COUNTS,
        ENCODER_QUANTUM,
        ENCODER_QUANTUM * 180.0 / core::f64::consts::PI
    );
    println!(
        "  Sensor Delay (tau_sens)      : {} step ({:.0} us)",
        sens_delay_steps,
        (sens_delay_steps as f64) * ts_s * 1e6
    );
    println!(
        "  Sensor Noise (sigma_theta)   : {:.1e} rad RMS",
        sens_noise_sigma_rad
    );
    println!(
        "  Total Loop Transport Delay   : {:.3} ms (phase penalty = -omega * tau_d)\n",
        total_delay_s * 1e3
    );

    // -------------------------------------------------------------------------
    // 2. Frequency-Domain Analysis & Compensator Synthesis
    // -------------------------------------------------------------------------
    println!(
        "Executing Rust classical control analysis and compensator synthesis..."
    );

    let study = results::run_analysis();
    let c_discrete = study.compensator_discrete;
    let mut ctrl_a = LeadDf2tController::new(&c_discrete, V_MAX);

    // Controller B: Discrete PID with filtered derivative
    let kp = 10.0;
    let ki = 30.0;
    let kd = 0.3;
    let tf_filter = 0.002;
    let mut ctrl_b = PidMotorController::new(
        PidGains {
            kp,
            ki,
            kd,
            tf: tf_filter,
        },
        V_MAX,
    );

    // -------------------------------------------------------------------------
    // 3. Closed-Loop Simulations
    // -------------------------------------------------------------------------
    let t_total = 1.2;
    let step_time = 0.05;
    let step_rad = 1.0;
    let dist_time = 0.45;
    let dist_torque = 0.02; // 0.02 N*m load torque step

    let scenario = ServoScenario {
        horizon: SampledHorizon {
            total_time_s: t_total,
            sample_time_s: ts_s,
        },
        step: PositionStep {
            time_s: step_time,
            angle_rad: step_rad,
        },
        disturbance: LoadDisturbance {
            time_s: dist_time,
            torque_nm: dist_torque,
        },
    };

    // Ideal baseline
    let mut ctrl_ideal = ctrl_a.clone();
    let traj_ideal = simulate_ideal_baseline(&mut ctrl_ideal, scenario);

    // Controller A (Lead DirectForm2T) with real noisy/delayed peripherals
    println!(
        "Simulating Controller A (Lead via DirectForm2T) with noisy/delayed peripherals..."
    );
    let mut act_a =
        Actuator::new(act_delay_steps, act_noise_sigma_v, V_MAX, 101);
    let mut sens_a =
        EncoderSensor::new(sens_delay_steps, sens_noise_sigma_rad, 202);
    let traj_a =
        simulate_motor_system(&mut ctrl_a, &mut act_a, &mut sens_a, scenario);

    // Controller B (Discrete PID) with real noisy/delayed peripherals
    println!(
        "Simulating Controller B (Discrete PID via Pid) with noisy/delayed peripherals..."
    );
    let mut act_b =
        Actuator::new(act_delay_steps, act_noise_sigma_v, V_MAX, 303);
    let mut sens_b =
        EncoderSensor::new(sens_delay_steps, sens_noise_sigma_rad, 404);
    let traj_b =
        simulate_motor_system(&mut ctrl_b, &mut act_b, &mut sens_b, scenario);

    // -------------------------------------------------------------------------
    // 4. Export JSON Dataset
    // -------------------------------------------------------------------------
    let mut json_payload = study.payload;
    json_payload["controllers"] = serde_json::json!({
        "controller_a_lead": {
            "name": ctrl_a.name(),
            "trajectory": traj_a,
        },
        "controller_b_pid": {
            "name": ctrl_b.name(),
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "tf": tf_filter,
            "trajectory": traj_b,
        },
        "ideal_baseline": {
            "name": "Ideal Baseline (Lead, 0-delay, 0-noise)",
            "trajectory": traj_ideal,
        }
    });

    let results_dir =
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            PathBuf::from(manifest_dir).join("results")
        } else if Path::new("control-rs-validation").is_dir() {
            PathBuf::from("control-rs-validation/results")
        } else {
            PathBuf::from("results")
        };
    std::fs::create_dir_all(&results_dir).ok();
    let h5_path = results_dir.join("dc-motor.rust.h5");

    let legacy_oracle = results_dir.join("dc_motor_oracle.json");
    if legacy_oracle.exists() {
        let _ = std::fs::remove_file(&legacy_oracle);
    }
    let legacy_json = results_dir.join("dc_motor.json");
    if legacy_json.exists() {
        let _ = std::fs::remove_file(&legacy_json);
    }

    println!("Writing results/dc-motor.rust.h5 ...");
    control_rs_ci::H5Container::write_variant_file(
        &h5_path,
        &json_payload,
        oracle::SCIPY_GATED_PATHS,
    )
    .unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {e}", h5_path.display());
        std::process::exit(1);
    });

    println!("Exported HDF5 to {}", h5_path.display());
}
