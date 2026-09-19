//! Quality gate trait and outcome data models.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::GateError;

/// Outcome verdict classification for a quality gate execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Gate succeeded completely without warnings or errors.
    Pass,
    /// Gate completed with non-blocking warnings or degraded capability.
    Warn,
    /// Gate failed or violated quality bounds.
    Fail,
    /// Gate execution was skipped by configuration or user flag.
    Skipped,
}

/// Standardized metadata record written to `<gate>.result.json` after execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateOutcome {
    /// Unique identifier / slug of the gate (e.g. "fmt", "valgrind").
    pub gate: String,
    /// High-level verdict outcome.
    pub verdict: Verdict,
    /// Process exit status code, if applicable.
    pub exit_code: Option<i32>,
    /// Total wall-clock execution duration in seconds.
    pub duration_secs: f64,
    /// Concise human-readable summary of the gate result.
    pub summary: Option<String>,
    /// Relative filename of the captured log output (e.g. "fmt.log").
    pub log_file: String,
    /// Optional relative filename of the raw tool JSON data (e.g. "metrics-raw.json").
    pub raw_artifact: Option<String>,
}

impl GateOutcome {
    /// Writes this gate outcome to `<out_dir>/<gate>.result.json`.
    ///
    /// # Errors
    /// Returns `GateError::Io` or `GateError::Json` on failure.
    pub fn save_to_dir(&self, out_dir: &Path) -> Result<PathBuf, GateError> {
        std::fs::create_dir_all(out_dir)?;
        let result_path = out_dir.join(format!("{}.result.json", self.gate));
        let file = File::create(&result_path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(result_path)
    }

    /// Loads a gate outcome record from a JSON file.
    ///
    /// # Errors
    /// Returns `GateError::Io` or `GateError::Json` on failure.
    pub fn load_from_file(path: &Path) -> Result<Self, GateError> {
        let file = File::open(path)?;
        let outcome = serde_json::from_reader(file)?;
        Ok(outcome)
    }
}

/// Execution context passed to quality gate runners.
#[derive(Debug, Clone)]
pub struct GateContext {
    /// Absolute path to the workspace root directory.
    pub workspace_root: PathBuf,
    /// Output directory for logs and JSON outcome artifacts.
    pub out_dir: PathBuf,
    /// Default wall-clock execution timeout.
    pub default_timeout: Duration,
}

impl GateContext {
    /// Returns the target log path for the given gate name.
    #[must_use]
    pub fn log_path(&self, gate_name: &str) -> PathBuf {
        self.out_dir.join(format!("{gate_name}.log"))
    }

    /// Returns the raw tool artifact path for the given gate name.
    #[must_use]
    pub fn raw_artifact_path(&self, gate_name: &str) -> PathBuf {
        self.out_dir.join(format!("{gate_name}-raw.json"))
    }
}

/// Core interface for quality gates in the continuous integration pipeline.
pub trait QualityGate: Send + Sync {
    /// Unique identifier for this gate.
    fn name(&self) -> &str;

    /// Short technical description of what this gate verifies.
    fn description(&self) -> &'static str;

    /// Exact command line string being executed for cargo-style output display.
    fn command_display(&self) -> String;

    /// Executes the gate logic, captures logs, and produces a `GateOutcome`.
    ///
    /// # Errors
    /// Returns `GateError` if execution or outcome persistence fails.
    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError>;

    /// Spawns the process, redirects standard output and error directly to `log_path`,
    /// bounds execution by `timeout`, and captures process exit status and duration.
    ///
    /// # Errors
    /// Returns `GateError::Spawn`, `GateError::Timeout`, or `GateError::Io` on failure.
    fn spawn_and_log(
        &self,
        cmd: &mut Command,
        log_path: &Path,
        timeout: Duration,
    ) -> Result<(ExitStatus, f64), GateError> {
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let log_file = File::create(log_path)?;
        let err_file = log_file.try_clone()?;

        cmd.stdout(Stdio::from(log_file));
        cmd.stderr(Stdio::from(err_file));

        let start = Instant::now();
        let mut child = cmd.spawn().map_err(|e| GateError::Spawn {
            gate: self.name().to_string(),
            message: e.to_string(),
        })?;

        let poll_interval = Duration::from_millis(50);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    return Ok((status, elapsed));
                }
                Ok(None) => {
                    if start.elapsed() > timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(GateError::Timeout {
                            gate: self.name().to_string(),
                            timeout_secs: timeout.as_secs_f64(),
                        });
                    }
                    thread::sleep(poll_interval);
                }
                Err(e) => {
                    let _ = child.kill();
                    return Err(GateError::Spawn {
                        gate: self.name().to_string(),
                        message: e.to_string(),
                    });
                }
            }
        }
    }
}
