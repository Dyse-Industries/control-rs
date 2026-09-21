//! Unsafe code surface and memory safety audit gate using cargo-geiger.

use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::GeigerConfig;
use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Raw report summary for cargo-geiger unsafe counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeigerRawReport {
    /// Total count of unsafe blocks detected across workspace.
    #[serde(default)]
    pub unsafe_blocks: usize,
    /// Total count of unsafe functions detected.
    #[serde(default)]
    pub unsafe_fns: usize,
}

/// Built-in unsafe memory audit gate.
#[derive(Debug, Clone)]
pub struct GeigerGate {
    config: GeigerConfig,
}

impl GeigerGate {
    /// Constructs a new `GeigerGate` with the given configuration.
    #[must_use]
    pub const fn new(config: GeigerConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration for this gate.
    #[must_use]
    pub const fn config(&self) -> &GeigerConfig {
        &self.config
    }
}

impl QualityGate for GeigerGate {
    fn name(&self) -> &str {
        "geiger"
    }

    fn description(&self) -> &'static str {
        "Scans workspace crates for unsafe code blocks and functions using cargo-geiger"
    }

    fn command_display(&self) -> String {
        "`cargo geiger --output-format Json`".to_string()
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let log_path = ctx.log_path(self.name());
        let raw_artifact_path = ctx.raw_artifact_path(self.name());

        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut cmd = Command::new("cargo");
        cmd.current_dir(&ctx.workspace_root);
        let geiger_target_dir = ctx.out_dir.join("geiger-target");
        cmd.env("CARGO_TARGET_DIR", &geiger_target_dir);
        cmd.args(["geiger", "--output-format", "Json"]);

        let spawn_res =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout);

        match spawn_res {
            Ok((status, duration)) => {
                let exit_code = status.code();

                // If cargo reported missing subcommand, degrade gracefully
                if let Ok(content) = fs::read_to_string(&log_path) {
                    if (content.contains("no such command")
                        || content.contains("no such subcommand"))
                        && content.contains("geiger")
                    {
                        let outcome = GateOutcome {
                            gate: self.name().to_string(),
                            verdict: Verdict::Warn,
                            exit_code: None,
                            duration_secs: duration,
                            summary: Some(
                                "cargo-geiger is uninstalled (install with: cargo install cargo-geiger); unsafe scan skipped (degraded)"
                                    .to_string(),
                            ),
                            log_file: "geiger.log".to_string(),
                            raw_artifact: None,
                        };
                        let _ = outcome.save_to_dir(&ctx.out_dir)?;
                        return Ok(outcome);
                    }

                    if let Some(json_start) = content.rfind("{\"packages\":") {
                        if let Some(json_slice) = content.get(json_start..) {
                            let _ = fs::write(
                                &raw_artifact_path,
                                json_slice.trim(),
                            );
                        }
                    } else if let Some(json_start) = content.rfind('{') {
                        if let Some(json_slice) = content.get(json_start..) {
                            let _ = fs::write(
                                &raw_artifact_path,
                                json_slice.trim(),
                            );
                        }
                    }
                }

                let verdict = if status.success() {
                    Verdict::Pass
                } else {
                    Verdict::Fail
                };

                let summary = if verdict == Verdict::Pass {
                    Some("Geiger unsafe code audit passed".to_string())
                } else {
                    Some(format!(
                        "Geiger check failed (exit code {})",
                        exit_code.unwrap_or(-1)
                    ))
                };

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict,
                    exit_code,
                    duration_secs: duration,
                    summary,
                    log_file: "geiger.log".to_string(),
                    raw_artifact: if raw_artifact_path.exists() {
                        Some("geiger-raw.json".to_string())
                    } else {
                        None
                    },
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(GateError::Spawn { .. }) => {
                let mut log_file = File::create(&log_path)?;
                writeln!(
                    log_file,
                    "Warning: 'cargo-geiger' is not installed on host.\n\
                     To install: cargo install cargo-geiger\n\
                     Skipping unsafe code scan (degraded)."
                )?;
                let duration = start.elapsed().as_secs_f64();

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict: Verdict::Warn,
                    exit_code: None,
                    duration_secs: duration,
                    summary: Some("cargo-geiger is uninstalled (install with: cargo install cargo-geiger); unsafe scan skipped (degraded)".to_string()),
                    log_file: "geiger.log".to_string(),
                    raw_artifact: None,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(e) => Err(e),
        }
    }
}
