//! src/oracle.rs
//!
//! Host-side HDF5 write helpers for the buck converter example.

/// Compared dataset paths written at `/` in `results/buck-converter.*.h5`.
pub const SCIPY_GATED_PATHS: &[&str] = &[
    "plant/natural_frequency_rad_s",
    "plant/damping_ratio",
    "margins/uncompensated/gain_crossover_rad_s",
    "margins/uncompensated/phase_margin_deg",
    "compensator/gain_k",
    "compensator/time_constant_t_s",
    "compensator/attenuation_alpha",
    "compensator/zero_rad_s",
    "compensator/pole_rad_s",
    "compensator/max_phase_lead_deg",
    "compensator/df2t_b0",
    "compensator/df2t_b1",
    "compensator/df2t_a1",
    "margins/compensated/gain_crossover_rad_s",
    "margins/compensated/phase_margin_deg",
    "margins/compensated/delay_margin_us",
    "frequency_sweep/uncomp_mag_db",
    "frequency_sweep/comp_mag_db",
    "frequency_sweep/uncomp_phase_deg",
    "frequency_sweep/comp_phase_deg",
    "root_locus/poles_re_sorted",
    "root_locus/poles_im_sorted",
];

/// Signal-stem to tolerance-key pairs for the HDF5 gate.
#[cfg(test)]
type ManifestKeys = Vec<(String, String)>;

/// Manifest signal → tolerance key mappings covering every `buck.*` row.
#[cfg(test)]
pub fn scipy_manifest_keys() -> ManifestKeys {
    [
        (
            "plant/natural_frequency_rad_s",
            "buck.plant.natural_frequency",
        ),
        ("plant/damping_ratio", "buck.plant.damping_ratio"),
        (
            "margins/uncompensated/gain_crossover_rad_s",
            "buck.margins.uncompensated.gain_crossover",
        ),
        (
            "margins/uncompensated/phase_margin_deg",
            "buck.margins.uncompensated.phase_margin",
        ),
        ("compensator/gain_k", "buck.compensator.gain_k"),
        (
            "compensator/time_constant_t_s",
            "buck.compensator.time_constant_t_s",
        ),
        (
            "compensator/attenuation_alpha",
            "buck.compensator.attenuation_alpha",
        ),
        ("compensator/zero_rad_s", "buck.compensator.zero_rad_s"),
        ("compensator/pole_rad_s", "buck.compensator.pole_rad_s"),
        (
            "compensator/max_phase_lead_deg",
            "buck.compensator.max_phase_lead_deg",
        ),
        ("compensator/df2t_b0", "buck.compensator.df2t_b0"),
        ("compensator/df2t_b1", "buck.compensator.df2t_b1"),
        ("compensator/df2t_a1", "buck.compensator.df2t_a1"),
        (
            "margins/compensated/gain_crossover_rad_s",
            "buck.margins.compensated.gain_crossover",
        ),
        (
            "margins/compensated/phase_margin_deg",
            "buck.margins.compensated.phase_margin",
        ),
        (
            "margins/compensated/delay_margin_us",
            "buck.margins.compensated.delay_margin",
        ),
        (
            "frequency_sweep/uncomp_mag_db",
            "buck.frequency_sweep.uncomp_mag_db",
        ),
        (
            "frequency_sweep/comp_mag_db",
            "buck.frequency_sweep.comp_mag_db",
        ),
        (
            "frequency_sweep/uncomp_phase_deg",
            "buck.frequency_sweep.uncomp_phase_deg",
        ),
        (
            "frequency_sweep/comp_phase_deg",
            "buck.frequency_sweep.comp_phase_deg",
        ),
        ("root_locus/poles_re_sorted", "buck.root_locus.poles_re"),
        ("root_locus/poles_im_sorted", "buck.root_locus.poles_im"),
    ]
    .into_iter()
    .map(|(s, k)| (s.to_string(), k.to_string()))
    .collect()
}

/// ngspice-gated signal mappings. Included only when `/ngspice` is present.
#[cfg(test)]
pub fn ngspice_manifest_keys() -> ManifestKeys {
    [
        ("circuit/ac_plant_dc_gain", "buck.ngspice.ac_plant_dc_gain"),
        (
            "circuit/step_final_voltage",
            "buck.ngspice.step_final_voltage",
        ),
        (
            "simulation/linear_step/metrics/steady_state_error_v",
            "buck.ngspice.step_steady_state_error",
        ),
        (
            "circuit/load_restored_voltage",
            "buck.ngspice.load_restored_voltage",
        ),
        ("circuit/switched_ripple", "buck.ngspice.switched_ripple"),
    ]
    .into_iter()
    .map(|(s, k)| (s.to_string(), k.to_string()))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use control_rs_ci::ToleranceTable;

    fn load_tolerance_table() -> Result<ToleranceTable, String> {
        let candidates = [
            "tolerances/buck_converter.toml",
            "control-rs-validation/tolerances/buck_converter.toml",
        ];
        for path_str in candidates {
            let p = Path::new(path_str);
            if p.exists() {
                return ToleranceTable::from_file(p);
            }
        }
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let p =
                Path::new(&manifest_dir).join("tolerances/buck_converter.toml");
            if p.exists() {
                return ToleranceTable::from_file(&p);
            }
        }
        Err("Failed to locate buck_converter.toml".to_string())
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1, classical-tools-examples#FR-3
    /// Method: Requirements-based test
    fn scipy_manifest_covers_buck_scipy_keys() {
        let table = load_tolerance_table().expect("tolerance table");
        for (_, key) in scipy_manifest_keys() {
            assert!(
                table.get(&key).is_some(),
                "missing scipy key {key} in classical_tools.toml"
            );
        }
        for (_, key) in ngspice_manifest_keys() {
            assert!(
                table.get(&key).is_some(),
                "missing ngspice key {key} in classical_tools.toml"
            );
        }
    }
}
