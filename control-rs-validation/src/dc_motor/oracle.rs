//! src/oracle.rs
//!
//! Host-side HDF5 write helpers for the DC motor example.
//! Root-locus comparison uses the sorted final-pole dataset
//! `analysis/root_locus/final_poles`.

/// Compared dataset paths written at `/` in `results/dc-motor.*.h5`.
pub const SCIPY_GATED_PATHS: &[&str] = &[
    "plant/num_ascending",
    "plant/den_ascending",
    "analysis/uncompensated_margins/gain_crossover_rad_s",
    "analysis/uncompensated_margins/phase_margin_deg",
    "analysis/lead_design/gain_k",
    "analysis/lead_design/time_constant_t_s",
    "analysis/lead_design/attenuation_alpha",
    "analysis/lead_design/max_phase_lead_deg",
    "analysis/lead_design/df2t_b0",
    "analysis/lead_design/df2t_b1",
    "analysis/lead_design/df2t_a1",
    "analysis/compensated_margins/gain_crossover_rad_s",
    "analysis/compensated_margins/phase_margin_deg",
    "analysis/compensated_margins/delay_margin_us",
    "analysis/compensated_margins/closed_loop_rhp_poles",
    "analysis/frequency_sweep/loop_mag_db",
    "analysis/frequency_sweep/loop_phase_deg",
    "analysis/root_locus/final_poles",
];

/// Signal-stem to tolerance-key pairs for the HDF5 gate.
#[cfg(test)]
type ManifestKeys = Vec<(String, String)>;

/// Manifest signal → tolerance key mappings covering every `dc_motor.*` row.
#[cfg(test)]
pub fn scipy_manifest_keys() -> ManifestKeys {
    [
        ("plant/num_ascending", "dc_motor.plant.num"),
        ("plant/den_ascending", "dc_motor.plant.den"),
        (
            "analysis/uncompensated_margins/gain_crossover_rad_s",
            "dc_motor.margins.uncompensated.gain_crossover",
        ),
        (
            "analysis/uncompensated_margins/phase_margin_deg",
            "dc_motor.margins.uncompensated.phase_margin",
        ),
        ("analysis/lead_design/gain_k", "dc_motor.compensator.gain_k"),
        (
            "analysis/lead_design/time_constant_t_s",
            "dc_motor.compensator.time_constant_t_s",
        ),
        (
            "analysis/lead_design/attenuation_alpha",
            "dc_motor.compensator.attenuation_alpha",
        ),
        (
            "analysis/lead_design/max_phase_lead_deg",
            "dc_motor.compensator.max_phase_lead_deg",
        ),
        (
            "analysis/lead_design/df2t_b0",
            "dc_motor.compensator.df2t_b0",
        ),
        (
            "analysis/lead_design/df2t_b1",
            "dc_motor.compensator.df2t_b1",
        ),
        (
            "analysis/lead_design/df2t_a1",
            "dc_motor.compensator.df2t_a1",
        ),
        (
            "analysis/compensated_margins/gain_crossover_rad_s",
            "dc_motor.margins.compensated.gain_crossover",
        ),
        (
            "analysis/compensated_margins/phase_margin_deg",
            "dc_motor.margins.compensated.phase_margin",
        ),
        (
            "analysis/compensated_margins/delay_margin_us",
            "dc_motor.margins.compensated.delay_margin",
        ),
        (
            "analysis/compensated_margins/closed_loop_rhp_poles",
            "dc_motor.stability.rhp_poles",
        ),
        (
            "analysis/frequency_sweep/loop_mag_db",
            "dc_motor.frequency_sweep.mag_db",
        ),
        (
            "analysis/frequency_sweep/loop_phase_deg",
            "dc_motor.frequency_sweep.phase_deg",
        ),
        (
            "analysis/root_locus/final_poles",
            "dc_motor.root_locus.poles",
        ),
    ]
    .into_iter()
    .map(|(s, k)| (s.to_string(), k.to_string()))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use control_rs_ci::ToleranceTable;

    fn load_tolerance_table() -> ToleranceTable {
        let candidates = [
            PathBuf::from("tolerances/dc_motor.toml"),
            PathBuf::from("control-rs-validation/tolerances/dc_motor.toml"),
        ];
        for p in &candidates {
            if let Ok(table) = ToleranceTable::from_file(p) {
                return table;
            }
        }
        ToleranceTable::from_toml(include_str!(
            "../../tolerances/dc_motor.toml"
        ))
        .expect("Failed to parse embedded dc_motor.toml")
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2, classical-tools-examples#FR-3
    /// Method: Requirements-based test
    fn scipy_manifest_covers_dc_motor_keys() {
        let table = load_tolerance_table();
        for (_, key) in scipy_manifest_keys() {
            assert!(
                table.get(&key).is_some(),
                "missing key {key} in classical_tools.toml"
            );
        }
    }
}
