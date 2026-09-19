//! Supply chain security and advisory audit gate using cargo-deny.

use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Built-in supply chain security and license audit gate.
#[derive(Debug, Clone, Default)]
pub struct DenyGate;

impl QualityGate for DenyGate {
    fn name(&self) -> &str {
        "deny"
    }

    fn description(&self) -> &'static str {
        "Audits dependencies for security advisories and license compliance using cargo-deny"
    }

    fn command_display(&self) -> String {
        "`cargo deny check`".to_string()
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let log_path = ctx.log_path(self.name());

        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut cmd = Command::new("cargo");
        cmd.current_dir(&ctx.workspace_root);
        cmd.args(["deny", "check"]);

        let spawn_res =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout);

        match spawn_res {
            Ok((status, duration)) => {
                // If cargo reported missing subcommand, degrade gracefully
                if let Ok(content) = fs::read_to_string(&log_path) {
                    if (content.contains("no such command")
                        || content.contains("no such subcommand"))
                        && content.contains("deny")
                    {
                        let outcome = GateOutcome {
                            gate: self.name().to_string(),
                            verdict: Verdict::Warn,
                            exit_code: None,
                            duration_secs: duration,
                            summary: Some(
                                "cargo-deny is uninstalled (install with: cargo install cargo-deny --locked); supply chain audit skipped (degraded)"
                                    .to_string(),
                            ),
                            log_file: "deny.log".to_string(),
                            raw_artifact: None,
                        };
                        let _ = outcome.save_to_dir(&ctx.out_dir)?;
                        return Ok(outcome);
                    }
                }

                let exit_code = status.code();
                let verdict = if status.success() {
                    Verdict::Pass
                } else {
                    Verdict::Fail
                };

                let summary = if status.success() {
                    Some(
                        "Supply chain licenses and advisories verified clean"
                            .to_string(),
                    )
                } else {
                    Some(format!(
                        "cargo-deny reported violations (exit code {})",
                        exit_code.unwrap_or(-1)
                    ))
                };

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict,
                    exit_code,
                    duration_secs: duration,
                    summary,
                    log_file: "deny.log".to_string(),
                    raw_artifact: None,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(GateError::Spawn { .. }) => {
                let mut log_file = File::create(&log_path)?;
                writeln!(
                    log_file,
                    "Warning: 'cargo-deny' is not installed on host.\n\
                     To install: cargo install cargo-deny --locked\n\
                     Skipping supply chain audit (degraded)."
                )?;
                let duration = start.elapsed().as_secs_f64();

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict: Verdict::Warn,
                    exit_code: None,
                    duration_secs: duration,
                    summary: Some("cargo-deny is uninstalled (install with: cargo install cargo-deny --locked); supply chain audit skipped (degraded)".to_string()),
                    log_file: "deny.log".to_string(),
                    raw_artifact: None,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(e) => Err(e),
        }
    }
}
