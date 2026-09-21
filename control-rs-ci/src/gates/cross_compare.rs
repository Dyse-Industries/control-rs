//! Quality gate executing cross-comparison and HDF5 validation (`cargo compare`).

use std::process::Command;

use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Quality gate running host-side cross-comparison.
#[derive(Debug, Clone, Default)]
pub struct CrossCompareGate;

impl QualityGate for CrossCompareGate {
    fn name(&self) -> &str {
        "cross-compare"
    }

    fn description(&self) -> &'static str {
        "Executes multi-language reference oracles and verifies HDF5 tolerance bounds"
    }

    fn command_display(&self) -> String {
        "cargo compare --config compare.toml".to_string()
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(&ctx.workspace_root);
        cmd.args([
            "run",
            "--package",
            "control-rs-compare",
            "--bin",
            "compare",
            "--",
            "--config",
            "compare.toml",
            "--bypass-gate",
        ]);

        let log_path = ctx.log_path(self.name());
        let (status, duration) =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout)?;

        let verdict = if status.success() {
            Verdict::Pass
        } else {
            Verdict::Fail
        };

        let outcome = GateOutcome {
            gate: self.name().to_string(),
            verdict,
            exit_code: status.code(),
            duration_secs: duration,
            summary: Some(if status.success() {
                "HDF5 cross-comparison passed".to_string()
            } else {
                "HDF5 cross-comparison discrepancies or execution failures detected"
                    .to_string()
            }),
            log_file: format!("{}.log", self.name()),
            raw_artifact: Some("cross-val-report.json".to_string()),
        };

        let _ = outcome.save_to_dir(&ctx.out_dir)?;
        Ok(outcome)
    }
}
