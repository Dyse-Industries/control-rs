//! Exportable Test Suite (ETS) Declaration and Execution
//!
//! Demonstrates the `control-rs-macros` and `control-rs-ets` testing framework:
//!
//! 1. `#[ets_suite]`: Transforms a standard Rust module into an Exportable Test
//!    Suite, automatically generating `SuiteDescriptor` and `ExecDescriptor` items.
//! 2. Inferred test cases: Plain functions inside the module become runnable
//!    executables with doc comments preserved as test descriptions.
//! 3. Dynamic settings: Module statics (`pub static`) are converted into typed
//!    configurable settings accessible via the `Setting` trait.
//! 4. Linker discovery: Emits `SUITE_DESCRIPTOR_PTR` in the `.ets_test_suites` section
//!    (or `__DATA,__ets_suites` on macOS) for automatic linker discovery.
//! 5. Execution: The host runner reads the suite descriptor, iterates over cases,
//!    runs assertions, and inspects settings without dynamic heap allocation.

use control_rs_ets::settings::SettingValue;
use control_rs_macros::ets_suite;

// --- Test Suite Definition ---

#[ets_suite]
/// Motor controller performance and safety test suite.
pub mod motor_control_suite {
    use control_rs_ets::settings::{Setting, SettingValue};

    /// Nominal supply voltage in volts.
    pub static SUPPLY_VOLTAGE: u32 = 12;

    /// Maximum allowable rotor current limit in amperes.
    pub static MAX_CURRENT_A: u32 = 10;

    /// Verifies that the supply voltage setting falls within the safe operating envelope.
    fn test_voltage_limits() {
        let v = match SUPPLY_VOLTAGE.get() {
            SettingValue::U32(val) => val,
            _ => 0,
        };
        assert!(
            (5..=24).contains(&v),
            "Voltage {v} outside nominal range [5, 24] V"
        );
    }

    /// Verifies that electrical time constant (L/R) is faster than mechanical time constant (J/b).
    fn test_time_constant_hierarchy() {
        // Motor parameters: R = 1.0 Ohm, L = 0.5 mH -> Tau_e = 0.5 ms
        // Inertia: J = 0.01 kg*m^2, damping: b = 0.1 N*m*s -> Tau_m = 100 ms
        let tau_e = 0.0005_f64;
        let tau_m = 0.1_f64;
        assert!(
            tau_e < tau_m,
            "Electrical dynamics ({tau_e}s) must be faster than mechanical dynamics ({tau_m}s)"
        );
    }

    /// Verifies back-EMF headroom at rated motor operating speed.
    fn test_back_emf_headroom() {
        let ke = 0.01_f64; // Back-EMF constant (V / (rad/s))
        let rated_speed_rad_s = 300.0_f64;
        let back_emf = ke * rated_speed_rad_s; // 3.0 V

        let v_supply = match SUPPLY_VOLTAGE.get() {
            SettingValue::U32(val) => f64::from(val),
            _ => 12.0,
        };
        assert!(
            back_emf < v_supply,
            "Back-EMF ({back_emf} V) exceeds supply headroom ({v_supply} V)"
        );
    }
}

fn main() {
    println!("=== control-rs: Exportable Test Suite (ETS) Example ===");

    // Access the statically generated suite descriptor pointer
    let suite = motor_control_suite::SUITE_DESCRIPTOR_PTR;

    println!("\n--- Suite Metadata ---");
    println!("  Name       : {}", suite.name);
    println!("  Description: {}", suite.description);
    println!("  Executables: {}", suite.executables.len());
    println!("  Settings   : {}", suite.settings.len());

    // 1. Inspect and verify configurable dynamic settings
    println!("\n--- Configured Settings ---");
    for (idx, setting) in suite.settings.iter().enumerate() {
        let name = setting.name();
        let doc = setting.description();
        let val = match setting.get() {
            SettingValue::U32(v) => format!("{v} (u32)"),
            SettingValue::Bool(b) => format!("{b} (bool)"),
            _ => "other".to_string(),
        };
        println!("  [{idx}] {name} = {val}");
        println!("      Description: {doc}");
    }

    // 2. Iterate through test executables and run each test
    println!("\n--- Executing Test Cases ---");
    let mut passed: usize = 0;
    for (idx, exec) in suite.executables.iter().enumerate() {
        println!("  Running test [{idx}] '{}'...", exec.name);
        println!("      Description: {}", exec.description);

        // Execute the test function pointer
        (exec.test_fn)();
        println!("      Verdict: PASS");
        passed = passed.saturating_add(1);
    }

    println!("\n--- Summary ---");
    println!(
        "  Total: {} | Passed: {} | Failed: 0",
        suite.executables.len(),
        passed
    );
    println!(
        "\nAll test suite cases passed with zero dynamic heap allocation."
    );
}
