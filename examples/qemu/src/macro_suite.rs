//! Integration test suite verifying ETS procedural macro transformation,
//! dynamic parameter reflection, and on-target test dispatch.

use control_rs_macros::ets_suite;

#[ets_suite]
/// Multi-target ETS procedural macro and dynamic settings verification suite.
pub mod macro_verification_suite {
    use control_rs_ets::settings::{Setting, SettingValue};

    /// Target controller loop frequency in Hertz.
    pub static LOOP_FREQUENCY_HZ: u32 = 1000;

    /// Cutoff frequency for low-pass sensor filter in Hertz.
    pub static FILTER_CUTOFF_HZ: u32 = 200;

    /// Master enable flag for on-target actuator output.
    pub static ACTUATOR_ENABLED: bool = true;

    /// Verifies that typed dynamic parameters are reflected accurately with default values.
    fn test_parameter_reflection_and_values() {
        let freq = match LOOP_FREQUENCY_HZ.get() {
            SettingValue::U32(v) => v,
            _ => 0,
        };
        assert_eq!(freq, 1000, "Expected loop frequency to default to 1000 Hz");

        let cutoff = match FILTER_CUTOFF_HZ.get() {
            SettingValue::U32(v) => v,
            _ => 0,
        };
        assert_eq!(cutoff, 200, "Expected filter cutoff to default to 200 Hz");

        let enabled = match ACTUATOR_ENABLED.get() {
            SettingValue::Bool(b) => b,
            _ => false,
        };
        assert!(enabled, "Expected actuator master flag to default to true");
    }

    /// Internal helper function with leading underscore; must not be registered as a test case.
    fn _private_helper_function() -> u32 {
        42
    }

    /// Internal helper function with parameters; must not be registered as a test case.
    fn _parameterized_helper(val: u32) -> u32 {
        val.wrapping_add(1)
    }

    /// Verifies dynamic parameter mutation and querying through the Setting interface.
    fn test_parameter_mutation_and_query() {
        assert!(LOOP_FREQUENCY_HZ.set(SettingValue::U32(2000)).is_ok());
        assert_eq!(LOOP_FREQUENCY_HZ.get(), SettingValue::U32(2000));
        // Restore default
        assert!(LOOP_FREQUENCY_HZ.set(SettingValue::U32(1000)).is_ok());
        assert_eq!(LOOP_FREQUENCY_HZ.get(), SettingValue::U32(1000));
    }

    /// Verifies that private helper functions and parameterized functions are excluded from EXECUTABLES.
    fn test_ignored_helper_functions() {
        assert_eq!(_private_helper_function(), 42);
        assert_eq!(_parameterized_helper(10), 11);
        for exec in EXECUTABLES {
            assert!(
                !exec.name.starts_with('_'),
                "Helper function must not be registered as an executable test case"
            );
        }
    }

    /// Verifies parameter descriptor metadata reflection (name and doc comments).
    fn test_parameter_descriptor_metadata() {
        assert_eq!(LOOP_FREQUENCY_HZ.name(), "LOOP_FREQUENCY_HZ");
        assert!(
            !LOOP_FREQUENCY_HZ.description().is_empty(),
            "Setting doc comment must be preserved"
        );

        assert_eq!(ACTUATOR_ENABLED.name(), "ACTUATOR_ENABLED");
    }
}
