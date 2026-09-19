//! Code coverage analysis gate using cargo-tarpaulin.

use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Minimal representation of cargo-tarpaulin raw coverage JSON output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TarpaulinRawReport {
    /// Covered line count.
    #[serde(default)]
    pub covered: usize,
    /// Coverable line count.
    #[serde(default)]
    pub coverable: usize,
    /// Percentage covered (0.0 - 100.0).
    #[serde(default)]
    pub coverage_percent: f64,
}

/// Built-in code coverage gate.
#[derive(Debug, Clone, Default)]
pub struct CoverageGate;

impl QualityGate for CoverageGate {
    fn name(&self) -> &str {
        "coverage"
    }

    fn description(&self) -> &'static str {
        "Measures workspace line coverage using cargo-tarpaulin"
    }

    fn command_display(&self) -> String {
        "`cargo tarpaulin --verbose --workspace --color never --out Json`"
            .to_string()
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let log_path = ctx.log_path(self.name());
        let raw_artifact_path = ctx.raw_artifact_path("tarpaulin");

        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut cmd = Command::new("cargo");
        cmd.current_dir(&ctx.workspace_root);
        cmd.args([
            "tarpaulin",
            "--verbose",
            "--workspace",
            "--color",
            "never",
            "--out",
            "Json",
            "--output-dir",
        ]);
        cmd.arg(&ctx.out_dir);

        let spawn_res =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout);

        match spawn_res {
            Ok((status, duration)) => {
                let exit_code = status.code();
                let verdict = if status.success() {
                    Verdict::Pass
                } else {
                    Verdict::Warn
                };

                // Check if tarpaulin generated tarpaulin.json or tarpaulin-report.json in out_dir
                let tarpaulin_json = ctx.out_dir.join("tarpaulin.json");
                let report_json = ctx.out_dir.join("tarpaulin-report.json");
                let mut raw_artifact = None;

                if tarpaulin_json.exists() {
                    let _ = fs::rename(&tarpaulin_json, &raw_artifact_path);
                    raw_artifact = Some("tarpaulin-raw.json".to_string());
                } else if report_json.exists() {
                    let _ = fs::rename(&report_json, &raw_artifact_path);
                    raw_artifact = Some("tarpaulin-raw.json".to_string());
                }

                let summary = if status.success() {
                    Some("Coverage analysis completed successfully".to_string())
                } else {
                    Some(format!(
                        "Tarpaulin exited with code {}",
                        exit_code.unwrap_or(-1)
                    ))
                };

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict,
                    exit_code,
                    duration_secs: duration,
                    summary,
                    log_file: "coverage.log".to_string(),
                    raw_artifact,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(GateError::Spawn { .. }) => {
                // Graceful degradation when cargo-tarpaulin is not installed
                let mut log_file = File::create(&log_path)?;
                writeln!(
                    log_file,
                    "Warning: 'cargo-tarpaulin' is not installed on host.\n\
                     To install: cargo install cargo-tarpaulin\n\
                     Skipping coverage analysis (degraded)."
                )?;
                let duration = start.elapsed().as_secs_f64();

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict: Verdict::Warn,
                    exit_code: None,
                    duration_secs: duration,
                    summary: Some(
                        "cargo-tarpaulin is uninstalled (install with: cargo install cargo-tarpaulin); coverage skipped (degraded)".to_string(),
                    ),
                    log_file: "coverage.log".to_string(),
                    raw_artifact: None,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(e) => Err(e),
        }
    }
}
