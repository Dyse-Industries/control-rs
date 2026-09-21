//! Generic quality gate model, execution engine, and outcome data types.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::config::{GateConfig, GateDefinition, GatePolicy};
use crate::error::GateError;

/// Outcome verdict classification for a quality gate execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Gate succeeded cleanly without warnings or errors.
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
    /// Unique identifier / slug of the gate (for example, "fmt", "valgrind").
    pub gate: String,
    /// High-level verdict outcome.
    pub verdict: Verdict,
    /// Process exit status code, if applicable.
    pub exit_code: Option<i32>,
    /// Total wall-clock execution duration in seconds.
    pub duration_secs: f64,
    /// Concise human-readable summary of the gate result.
    pub summary: Option<String>,
    /// Relative filename of the captured log output (for example, "fmt.log").
    pub log_file: String,
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
}

/// The single, concrete quality gate type used for all gates.
#[derive(Debug, Clone)]
pub struct Gate {
    /// Gate identifier (e.g. "clean", "fmt", "clippy").
    pub name: String,
    /// Base command string (e.g. "cargo clean", "cargo fmt", "vale").
    pub command: String,
    /// Additional arguments passed to the command.
    pub args: Vec<String>,
    /// Optional description for status displays and reports.
    pub description: Option<String>,
    /// Environment variable overrides.
    pub env: HashMap<String, String>,
}

impl Gate {
    /// Constructs a new `Gate`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        command: impl Into<String>,
        args: Vec<String>,
        description: Option<String>,
        env: HashMap<String, String>,
    ) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args,
            description,
            env,
        }
    }

    /// Constructs a `Gate` from a parsed `GateDefinition`.
    #[must_use]
    pub fn from_definition(
        name: impl Into<String>,
        def: &GateDefinition,
    ) -> Self {
        Self {
            name: name.into(),
            command: def.command.clone(),
            args: def.args.clone(),
            description: def.description.clone(),
            env: def.env.clone(),
        }
    }

    /// Returns the name of this gate.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the optional description of this gate.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Formatted command line string for display.
    #[must_use]
    pub fn command_display(&self) -> String {
        if self.args.is_empty() {
            format!("`{}`", self.command)
        } else {
            format!("`{} {}`", self.command, self.args.join(" "))
        }
    }

    /// Executes the gate command, captures logs, bounds execution by timeout,
    /// evaluates verdict and writes `<gate>.result.json`.
    ///
    /// # Errors
    /// Returns `GateError` on process spawn failure or I/O error.
    pub fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let log_path = ctx.log_path(&self.name);
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Decompose compound command string into binary + initial arguments (e.g. "cargo clean")
        let mut words = self.command.split_whitespace();
        let program = words.next().unwrap_or(&self.command);
        let mut initial_args: Vec<String> = words.map(String::from).collect();
        initial_args.extend(self.args.clone());

        let mut cmd = Command::new(program);
        cmd.current_dir(&ctx.workspace_root);
        cmd.args(&initial_args);
        for (k, v) in &self.env {
            cmd.env(k, v);
        }

        let (status, duration) =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout)?;

        let exit_code = status.code();
        let verdict = if status.success() {
            Verdict::Pass
        } else {
            Verdict::Fail
        };

        let summary = if status.success() {
            Some(format!("{} succeeded cleanly", self.name))
        } else {
            Some(format!(
                "{} failed with exit code {}",
                self.name,
                exit_code.unwrap_or(-1)
            ))
        };

        let outcome = GateOutcome {
            gate: self.name.clone(),
            verdict,
            exit_code,
            duration_secs: duration,
            summary,
            log_file: format!("{}.log", self.name),
        };

        let _ = outcome.save_to_dir(&ctx.out_dir)?;
        Ok(outcome)
    }

    fn spawn_and_log(
        &self,
        cmd: &mut Command,
        log_path: &Path,
        timeout: Duration,
    ) -> Result<(ExitStatus, f64), GateError> {
        let log_file = File::create(log_path)?;
        let err_file = log_file.try_clone()?;

        cmd.stdout(Stdio::from(log_file));
        cmd.stderr(Stdio::from(err_file));

        let start = Instant::now();
        let mut child = cmd.spawn().map_err(|e| GateError::Spawn {
            gate: self.name.clone(),
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
                            gate: self.name.clone(),
                            timeout_secs: timeout.as_secs_f64(),
                        });
                    }
                    thread::sleep(poll_interval);
                }
                Err(e) => {
                    let _ = child.kill();
                    return Err(GateError::Spawn {
                        gate: self.name.clone(),
                        message: e.to_string(),
                    });
                }
            }
        }
    }
}

/// Instantiates all configured quality gates according to the workspace configuration.
///
/// Gates are gathered in deterministic pipeline order:
/// 1. Grouped gates from `[execution.groups]` in sorted group key order and member sequence.
/// 2. Exclusive gates from `exclusive_gates`.
/// 3. Any additional gates defined in `[gates]`.
/// 4. Any remaining gate definitions in `gate_definitions`.
///
/// # Errors
/// Returns `GateError::Config` if an active gate is missing its `[<gate>]` definition table in `gate.toml`.
pub fn build_all_gates(
    config: &GateConfig,
) -> Result<Vec<Arc<Gate>>, GateError> {
    let mut names = Vec::new();

    let mut sorted_group_keys: Vec<_> =
        config.execution.groups.keys().collect();
    sorted_group_keys.sort();
    for grp in sorted_group_keys {
        if let Some(members) = config.execution.groups.get(grp) {
            for m in members {
                if !names.contains(m) {
                    names.push(m.clone());
                }
            }
        }
    }

    for m in &config.execution.exclusive_gates {
        if !names.contains(m) {
            names.push(m.clone());
        }
    }

    let mut other_gates: Vec<_> = config.gates.keys().collect();
    other_gates.sort();
    for g in other_gates {
        if !names.contains(g) {
            names.push(g.clone());
        }
    }

    let mut remaining_defs: Vec<_> = config.gate_definitions.keys().collect();
    remaining_defs.sort();
    for g in remaining_defs {
        if !names.contains(g) {
            names.push(g.clone());
        }
    }

    let mut gates = Vec::new();
    for name in names {
        if let Some(def) = config.gate_def(&name) {
            gates.push(Arc::new(Gate::from_definition(name, def)));
        } else if config.policy_for(&name) != GatePolicy::Skip {
            return Err(GateError::Config {
                path: PathBuf::from("gate.toml"),
                message: format!("Missing [gate] definition for gate '{name}'"),
            });
        }
    }

    Ok(gates)
}
