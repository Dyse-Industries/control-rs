//! Semantic versioning and public API stability gate using cargo-semver-checks.

use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use crate::config::SemverConfig;
use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Built-in SemVer public API compatibility gate.
#[derive(Debug, Clone)]
pub struct SemverGate {
    config: SemverConfig,
}

impl SemverGate {
    /// Constructs a new `SemverGate` with the given configuration.
    #[must_use]
    pub const fn new(config: SemverConfig) -> Self {
        Self { config }
    }
}

impl QualityGate for SemverGate {
    fn name(&self) -> &str {
        "semver"
    }

    fn description(&self) -> &'static str {
        "Verifies public API stability against baseline ref using cargo-semver-checks"
    }

    fn command_display(&self) -> String {
        format!(
            "`cargo semver-checks check-release --baseline-rev {}`",
            self.config.baseline_ref
        )
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let log_path = ctx.log_path(self.name());

        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut cmd = Command::new("cargo");
        cmd.current_dir(&ctx.workspace_root);
        cmd.args(["semver-checks", "check-release", "--baseline-rev"]);
        cmd.arg(&self.config.baseline_ref);

        let spawn_res =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout);

        match spawn_res {
            Ok((status, duration)) => {
                let exit_code = status.code();

                // If cargo reported missing subcommand, degrade gracefully
                if let Ok(content) = fs::read_to_string(&log_path) {
                    if (content.contains("no such command")
                        || content.contains("no such subcommand"))
                        && content.contains("semver-checks")
                    {
                        let outcome = GateOutcome {
                            gate: self.name().to_string(),
                            verdict: Verdict::Warn,
                            exit_code: None,
                            duration_secs: duration,
                            summary: Some(
                                "cargo-semver-checks is uninstalled (install with: cargo install cargo-semver-checks --locked); SemVer check skipped (degraded)"
                                    .to_string(),
                            ),
                            log_file: "semver.log".to_string(),
                            raw_artifact: None,
                        };
                        let _ = outcome.save_to_dir(&ctx.out_dir)?;
                        return Ok(outcome);
                    }
                }

                let verdict = if status.success() {
                    Verdict::Pass
                } else {
                    Verdict::Fail
                };

                let summary = if status.success() {
                    Some(format!(
                        "Public API compatible with baseline {}",
                        self.config.baseline_ref
                    ))
                } else {
                    Some(format!(
                        "cargo-semver-checks reported breaking changes (exit code {})",
                        exit_code.unwrap_or(-1)
                    ))
                };

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict,
                    exit_code,
                    duration_secs: duration,
                    summary,
                    log_file: "semver.log".to_string(),
                    raw_artifact: None,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(GateError::Spawn { .. }) => {
                let mut log_file = File::create(&log_path)?;
                writeln!(
                    log_file,
                    "Warning: 'cargo-semver-checks' is not installed on host.\n\
                     To install: cargo install cargo-semver-checks --locked\n\
                     Skipping SemVer check (degraded)."
                )?;
                let duration = start.elapsed().as_secs_f64();

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict: Verdict::Warn,
                    exit_code: None,
                    duration_secs: duration,
                    summary: Some("cargo-semver-checks is uninstalled (install with: cargo install cargo-semver-checks --locked); SemVer check skipped (degraded)".to_string()),
                    log_file: "semver.log".to_string(),
                    raw_artifact: None,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(e) => Err(e),
        }
    }
}
