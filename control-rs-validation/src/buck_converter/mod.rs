//! src/main.rs
//!
//! `classical_tools` validation example
//! (`documentation/control-toolboxes/classical-tools-design.md`, §6.6):
//! exercises Routh, root locus, margins, compensator design, and discrete
//! firmware execution against a synchronous buck converter's averaged small-signal
//! duty-cycle-to-output-voltage model.

mod analysis;
mod circuit;
mod oracle;
mod results;
mod simulation;

use std::path::{Path, PathBuf};

/// Run the buck-converter suite and emit `results/buck-converter.rust.h5`.
pub fn run() {
    // -------------------------------------------------------------------------
    // 1. Plant & Circuit Specification
    // -------------------------------------------------------------------------
    println!("Plant & Circuit Parameters:");
    println!("  Input Rail Voltage (V_in)    : {:.1} V", circuit::V_IN);
    println!(
        "  Filter Inductance (L)        : {:.1e} H (100 uH)",
        circuit::INDUCTANCE
    );
    println!(
        "  Filter Capacitance (C)       : {:.1e} F (100 uF)",
        circuit::CAPACITANCE
    );
    println!(
        "  Nominal Load Resistance (R_L): {:.2} Ohm (Nominal I_out = 5.0 A)",
        circuit::LOAD_RESISTANCE
    );
    println!("  Target Regulated Voltage     : {:.2} V", circuit::V_OUT);
    println!(
        "  Operating Duty Cycle (D0)    : {:.4} (5/12)",
        circuit::operating_duty_cycle()
    );
    println!(
        "  Natural Frequency (omega_n)  : {:.4e} rad/s ({:.2} Hz)",
        circuit::natural_frequency(),
        circuit::natural_frequency() / (2.0 * core::f64::consts::PI)
    );
    println!(
        "  Filter Damping Ratio (zeta)  : {:.4}",
        circuit::damping_ratio()
    );

    let plant = circuit::plant();
    println!(
        "  Control-to-Output G_vd(s)    : {:.1e} / (s^2 + {:.1e} s + {:.1e})\n",
        plant.num_slice()[0],
        plant.den_slice()[1],
        plant.den_slice()[0]
    );

    // -------------------------------------------------------------------------
    // 2. Classical Control Analysis & Simulations
    // -------------------------------------------------------------------------
    println!("Executing Rust classical control analysis and simulations...");
    let study = results::run_analysis();

    // -------------------------------------------------------------------------
    // 3. HDF5 (Rust variant file). Oracles and compare are CI commands.
    // -------------------------------------------------------------------------
    println!("Writing results/buck-converter.rust.h5 ...");
    let results_dir =
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            PathBuf::from(manifest_dir).join("results")
        } else if Path::new("control-rs-validation").is_dir() {
            PathBuf::from("control-rs-validation/results")
        } else {
            PathBuf::from("results")
        };
    std::fs::create_dir_all(&results_dir).ok();
    let h5_path = results_dir.join("buck-converter.rust.h5");

    let mut payload = study.payload;
    let step_final = study
        .results
        .simulation
        .linear_step
        .v_out_v
        .last()
        .copied()
        .unwrap_or(circuit::V_OUT);
    let load_restored = study.results.simulation.load_transient.restored_v;
    payload["circuit"] = serde_json::json!({
        "ac_plant_dc_gain": 20.0 * circuit::V_IN.log10(),
        "step_final_voltage": step_final,
        "load_restored_voltage": load_restored,
    });

    control_rs_ci::H5Container::write_variant_file(
        &h5_path,
        &payload,
        oracle::SCIPY_GATED_PATHS,
    )
    .unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {e}", h5_path.display());
        std::process::exit(1);
    });

    println!("Exported HDF5 to {}", h5_path.display());

    let legacy_json = results_dir.join("buck_converter.json");
    if legacy_json.exists() {
        let _ = std::fs::remove_file(legacy_json);
    }
    let legacy_oracle = results_dir.join("buck_converter_oracle.json");
    if legacy_oracle.exists() {
        let _ = std::fs::remove_file(legacy_oracle);
    }
}
