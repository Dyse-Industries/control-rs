//! Target execution matrix for CI verification across virtual and physical platforms.
//!
//! Manages the set of hardware and emulator targets (e.g. Cortex-M QEMU, RISC-V QEMU,
//! and physical Teensy serial targets) to execute embedded test server (ETS) suites on.

use std::time::Duration;

use control_rs_ets_host::error::HostError;
use control_rs_ets_host::runner::{EtsRunResult, run_headless_ets};
use control_rs_ets_host::target::Target;

type MatrixEntry = (String, Result<EtsRunResult, HostError>);

/// Executes the provided target execution matrix headlessly and collects run results.
#[must_use]
pub fn execute_target_matrix(
    targets: &[Target],
    timeout: Duration,
) -> Vec<MatrixEntry> {
    let mut results = Vec::new();
    for target in targets {
        let display_name = target.display_name();
        let res = run_headless_ets(target.clone(), timeout);
        results.push((display_name, res));
    }
    results
}
