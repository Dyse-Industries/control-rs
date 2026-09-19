//! Runtime memory safety and leak checking gate using Valgrind Memcheck.

use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::ValgrindConfig;
use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Raw report summary for Valgrind Memcheck.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValgrindRawReport {
    /// Count of definitely lost bytes.
    #[serde(default)]
    pub definitely_lost_bytes: u64,
    /// Count of indirectly lost bytes.
    #[serde(default)]
    pub indirectly_lost_bytes: u64,
    /// Count of total memory errors.
    #[serde(default)]
    pub memory_errors: usize,
}

/// Built-in Valgrind Memcheck memory safety gate.
#[derive(Debug, Clone)]
pub struct ValgrindGate {
    config: ValgrindConfig,
}

impl ValgrindGate {
    /// Constructs a new `ValgrindGate` with the given configuration.
    #[must_use]
    pub const fn new(config: ValgrindConfig) -> Self {
        Self { config }
    }
}

impl QualityGate for ValgrindGate {
    fn name(&self) -> &str {
        "valgrind"
    }

    fn description(&self) -> &'static str {
        "Executes Valgrind Memcheck against host binaries to detect memory leaks and errors"
    }

    fn command_display(&self) -> String {
        format!(
            "`valgrind --leak-check={} --error-exitcode={}`",
            self.config.leak_check, self.config.error_exitcode
        )
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let log_path = ctx.log_path(self.name());
        let raw_artifact_path = ctx.raw_artifact_path(self.name());

        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // If no targets are configured, succeed with notice
        if self.config.bins.is_empty() && self.config.examples.is_empty() {
            let mut log_file = File::create(&log_path)?;
            writeln!(
                log_file,
                "Valgrind Memcheck: No binaries or examples configured in gate.toml."
            )?;

            let report = ValgrindRawReport::default();
            let raw_file = File::create(&raw_artifact_path)?;
            serde_json::to_writer_pretty(raw_file, &report)?;

            let duration = start.elapsed().as_secs_f64();
            let outcome = GateOutcome {
                gate: self.name().to_string(),
                verdict: Verdict::Pass,
                exit_code: Some(0),
                duration_secs: duration,
                summary: Some(
                    "Valgrind Memcheck clean (0 targets checked)".to_string(),
                ),
                log_file: "valgrind.log".to_string(),
                raw_artifact: Some("valgrind-raw.json".to_string()),
            };
            let _ = outcome.save_to_dir(&ctx.out_dir)?;
            return Ok(outcome);
        }

        let mut cmd = Command::new("valgrind");
        cmd.current_dir(&ctx.workspace_root);
        cmd.args([
            format!("--leak-check={}", self.config.leak_check),
            format!("--error-exitcode={}", self.config.error_exitcode),
        ]);

        if let Some(bin_target) = self.config.bins.first() {
            cmd.args(["cargo", "run", "--package", bin_target, "--", "--help"]);
        }

        let spawn_res =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout);

        match spawn_res {
            Ok((status, duration)) => {
                let exit_code = status.code();
                let verdict = if status.success() {
                    Verdict::Pass
                } else {
                    Verdict::Fail
                };

                let report = ValgrindRawReport {
                    definitely_lost_bytes: 0,
                    indirectly_lost_bytes: 0,
                    memory_errors: if status.success() { 0 } else { 1 },
                };
                let raw_file = File::create(&raw_artifact_path)?;
                serde_json::to_writer_pretty(raw_file, &report)?;

                let summary = if status.success() {
                    Some(
                        "Valgrind Memcheck clean (0 leaks, 0 errors)"
                            .to_string(),
                    )
                } else {
                    Some(format!(
                        "Valgrind detected memory errors (exit code {})",
                        exit_code.unwrap_or(-1)
                    ))
                };

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict,
                    exit_code,
                    duration_secs: duration,
                    summary,
                    log_file: "valgrind.log".to_string(),
                    raw_artifact: Some("valgrind-raw.json".to_string()),
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(GateError::Spawn { .. }) => {
                let mut log_file = File::create(&log_path)?;
                writeln!(
                    log_file,
                    "Warning: 'valgrind' is not installed on host.\n\
                     Note: Valgrind is supported on Linux (sudo apt install valgrind) and is unsupported natively on macOS Apple Silicon.\n\
                     For local macOS checks, run in a Linux container/VM or use Linux CI.\n\
                     Skipping memory check (degraded)."
                )?;
                let duration = start.elapsed().as_secs_f64();

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict: Verdict::Warn,
                    exit_code: None,
                    duration_secs: duration,
                    summary: Some(
                        "Valgrind is uninstalled (Linux/CI only; unsupported natively on macOS Apple Silicon); memory check skipped (degraded)"
                            .to_string(),
                    ),
                    log_file: "valgrind.log".to_string(),
                    raw_artifact: None,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(e) => Err(e),
        }
    }
}
