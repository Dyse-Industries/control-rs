//! Prose and documentation style linting gate using Vale.

use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use crate::config::ValeConfig;
use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Built-in Vale documentation and prose style linting gate.
#[derive(Debug, Clone)]
pub struct ValeGate {
    config: ValeConfig,
}

impl ValeGate {
    /// Constructs a new `ValeGate` with the given configuration.
    #[must_use]
    pub const fn new(config: ValeConfig) -> Self {
        Self { config }
    }
}

impl QualityGate for ValeGate {
    fn name(&self) -> &str {
        "vale"
    }

    fn description(&self) -> &'static str {
        "Lints documentation and doc comments for prose style conformity"
    }

    fn command_display(&self) -> String {
        format!(
            "`vale --config={} {}`",
            self.config.config,
            self.config.paths.join(" ")
        )
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let log_path = ctx.log_path(self.name());
        let raw_artifact_path = ctx.raw_artifact_path(self.name());

        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let config_file_path = ctx.workspace_root.join(&self.config.config);
        if !config_file_path.exists() {
            let mut log_file = File::create(&log_path)?;
            writeln!(
                log_file,
                "Warning: Vale configuration file '{}' does not exist at workspace root.\n\
                 Ensure '{}' is created with valid style rules or update [vale].config in gate.toml.",
                self.config.config, self.config.config
            )?;
            let duration = start.elapsed().as_secs_f64();
            let outcome = GateOutcome {
                gate: self.name().to_string(),
                verdict: Verdict::Warn,
                exit_code: None,
                duration_secs: duration,
                summary: Some(format!(
                    "Vale config file '{}' not found; prose linting skipped (degraded)",
                    self.config.config
                )),
                log_file: "vale.log".to_string(),
                raw_artifact: None,
            };
            let _ = outcome.save_to_dir(&ctx.out_dir)?;
            return Ok(outcome);
        }

        let mut cmd = Command::new("vale");
        cmd.current_dir(&ctx.workspace_root);
        cmd.arg(format!("--config={}", self.config.config));
        cmd.arg("--output=JSON");

        for path in &self.config.paths {
            cmd.arg(path);
        }

        let spawn_res =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout);

        match spawn_res {
            Ok((status, duration)) => {
                let exit_code = status.code();
                let log_content =
                    fs::read_to_string(&log_path).unwrap_or_default();
                let trimmed = log_content.trim_start();

                let mut total_alerts = 0;
                let mut affected_files = 0;
                let mut runtime_error: Option<String> = None;

                if trimmed.starts_with('{') {
                    let _ = fs::write(&raw_artifact_path, &log_content);
                    if let Ok(val) =
                        serde_json::from_str::<serde_json::Value>(trimmed)
                    {
                        if let Some(code) =
                            val.get("Code").and_then(|c| c.as_str())
                        {
                            let text = val
                                .get("Text")
                                .and_then(|t| t.as_str())
                                .unwrap_or("Runtime error");
                            runtime_error = Some(format!(
                                "{}: {}",
                                code,
                                text.lines().next().unwrap_or(text)
                            ));
                        } else if let Some(map) = val.as_object() {
                            for (_file, alerts) in map {
                                if let Some(arr) = alerts.as_array() {
                                    if !arr.is_empty() {
                                        total_alerts += arr.len();
                                        affected_files += 1;
                                    }
                                }
                            }
                        }
                    }
                }

                let verdict = if runtime_error.is_some() {
                    Verdict::Warn
                } else if status.success() && total_alerts == 0 {
                    Verdict::Pass
                } else {
                    Verdict::Warn
                };

                let summary = if let Some(err) = runtime_error {
                    Some(format!("Vale runtime error ({})", err))
                } else if verdict == Verdict::Pass {
                    Some(
                        "Documentation prose style check passed cleanly"
                            .to_string(),
                    )
                } else if total_alerts > 0 {
                    Some(format!(
                        "Vale reported {} style alert(s) across {} file(s)",
                        total_alerts, affected_files
                    ))
                } else {
                    Some(format!(
                        "Vale reported style warnings (exit code {})",
                        exit_code.unwrap_or(-1)
                    ))
                };

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict,
                    exit_code,
                    duration_secs: duration,
                    summary,
                    log_file: "vale.log".to_string(),
                    raw_artifact: if raw_artifact_path.exists() {
                        Some("vale-raw.json".to_string())
                    } else {
                        None
                    },
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(GateError::Spawn { .. }) => {
                // Graceful degradation when Vale is uninstalled
                let mut log_file = File::create(&log_path)?;
                writeln!(
                    log_file,
                    "Warning: 'vale' is not installed on host.\n\
                     To install: brew install vale (macOS) or see https://vale.sh/docs/vale-cli/installation/\n\
                     Skipping prose linting (degraded)."
                )?;
                let duration = start.elapsed().as_secs_f64();

                let outcome = GateOutcome {
                    gate: self.name().to_string(),
                    verdict: Verdict::Warn,
                    exit_code: None,
                    duration_secs: duration,
                    summary: Some(
                        "Vale is uninstalled (install with: brew install vale / https://vale.sh); prose linting skipped (degraded)"
                            .to_string(),
                    ),
                    log_file: "vale.log".to_string(),
                    raw_artifact: None,
                };
                let _ = outcome.save_to_dir(&ctx.out_dir)?;
                Ok(outcome)
            }
            Err(e) => Err(e),
        }
    }
}
