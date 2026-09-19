//! Mutation testing quality gate using cargo-mutants.

use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use crate::config::MutantsConfig;
use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Built-in mutation testing quality gate.
#[derive(Debug, Clone)]
pub struct MutantsGate {
    config: MutantsConfig,
}

impl MutantsGate {
    /// Constructs a new `MutantsGate` with the given configuration.
    #[must_use]
    pub const fn new(config: MutantsConfig) -> Self {
        Self { config }
    }
}

impl QualityGate for MutantsGate {
    fn name(&self) -> &str {
        "mutants"
    }

    fn description(&self) -> &'static str {
        "Mutates ASTs to verify test fault-injection rigor using cargo-mutants"
    }

    fn command_display(&self) -> String {
        let jobs = self.config.jobs.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
        });
        format!(
            "`cargo mutants --jobs {} --timeout-multiplier {} --json`",
            jobs, self.config.timeout_multiplier
        )
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
        cmd.arg("mutants");

        let jobs = self.config.jobs.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
        });
        cmd.args(["--jobs", &jobs.to_string()]);
        cmd.args([
            "--timeout-multiplier",
            &self.config.timeout_multiplier.to_string(),
        ]);
        cmd.arg("--json");

        for excl in &self.config.exclude {
            cmd.args(["--exclude", excl]);
        }

        if let Some(ref excl_re) = self.config.exclude_re {
            cmd.args(["--exclude-re", excl_re]);
        }

        for file_pattern in &self.config.files {
            cmd.args(["--file", file_pattern]);
        }

        let spawn_res =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout);

        match spawn_res {
            Ok((status, duration)) => {
                let exit_code = status.code();

                // If cargo reported missing subcommand, degrade gracefully
                if let Ok(content) = fs::read_to_string(&log_path) {
                    if (content.contains("no such command")
                        || content.contains("no such subcommand"))
                        && content.contains("mutants")
                    {
                        let outcome = GateOutcome {
                            gate: self.name().to_string(),
                            verdict: Verdict::Warn,
                            exit_code: None,
                            duration_secs: duration,
                            summary: Some(
                                "cargo-mutants is uninstalled (install with: cargo install cargo-mutants); mutation testing skipped (degraded)"
                                    .to_string(),
                            ),
                            log_file: "mutants.log".to_string(),
                            raw_artifact: None,
                        };
                        let _ = outcome.save_to_dir(&ctx.out_dir)?;
                        return Ok(outcome);
                    }

                    if content.trim_start().starts_with('{') {
                        let _ = fs::write(&raw_artifact_path, &content);
                    }
                }

                let verdict = if status.success() {
                    Verdict::Pass
                } else {
                    Verdict::Warn
                };

                let summary = if status.success() {
                    Some(
                        "All injected mutations were caught by test suite"
                            .to_string(),
                    )
                } else {
                    Some(format!(
                        "Mutation testing completed with exit code {}",
                        exit_code.unwrap_or(-1)
                    ))
                };

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict,
                    exit_code,
                    duration_secs: duration,
                    summary,
                    log_file: "mutants.log".to_string(),
                    raw_artifact: if raw_artifact_path.exists() {
                        Some("mutants-raw.json".to_string())
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
                    "Warning: 'cargo-mutants' is not installed on host.\n\
                     To install: cargo install cargo-mutants\n\
                     Skipping mutation testing (degraded)."
                )?;
                let duration = start.elapsed().as_secs_f64();

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict: Verdict::Warn,
                    exit_code: None,
                    duration_secs: duration,
                    summary: Some("cargo-mutants is uninstalled (install with: cargo install cargo-mutants); mutation testing skipped (degraded)".to_string()),
                    log_file: "mutants.log".to_string(),
                    raw_artifact: None,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(e) => Err(e),
        }
    }
}
