//! Configuration data models for `gate.toml`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::GateError;

/// Execution policy for a quality gate.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum GatePolicy {
    /// Gate failure blocks the entire pipeline.
    #[default]
    Fail,
    /// Gate executes; failure reports a warning but does not block.
    Warn,
    /// Gate execution is skipped entirely.
    Skip,
}

/// Type alias for [`GatePolicy`] matching declarative `gate-mode` terminology.
pub type GateMode = GatePolicy;

/// Runner-wide execution settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerConfig {
    /// Pipeline or workspace title.
    #[serde(default = "default_title")]
    pub title: String,
    /// Output artifact directory path.
    #[serde(default = "default_out_dir")]
    pub out_dir: PathBuf,
    /// Default execution timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

fn default_title() -> String {
    "control-rs".to_string()
}

fn default_out_dir() -> PathBuf {
    PathBuf::from("target/ci-artifacts")
}

const fn default_timeout_secs() -> u64 {
    90
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            title: default_title(),
            out_dir: default_out_dir(),
            timeout_secs: default_timeout_secs(),
        }
    }
}

const fn default_true() -> bool {
    true
}

/// Execution schedule, group partitioning, and concurrency configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// If true, executes independent groups concurrently across worker threads.
    #[serde(default = "default_true")]
    pub parallel: bool,
    /// List of gate names assigned to exclusive execution.
    #[serde(default)]
    pub exclusive_gates: Vec<String>,
    /// Declarative execution groups: mapping group_name -> list of gate names.
    #[serde(default, alias = "lanes", alias = "threads")]
    pub groups: HashMap<String, Vec<String>>,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            exclusive_gates: Vec::new(),
            groups: HashMap::new(),
        }
    }
}

impl ExecutionConfig {
    /// Normalizes configuration by merging any `groups["exclusive"]` entries into `exclusive_gates`.
    pub fn normalize(&mut self) {
        if let Some(mut excl) = self.groups.remove("exclusive") {
            for g in excl.drain(..) {
                if !self.exclusive_gates.contains(&g) {
                    self.exclusive_gates.push(g);
                }
            }
        }
    }
}

/// Generic declarative gate definition parsed from `gate.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateDefinition {
    /// Command string to execute (e.g. `"cargo clean"`, `"cargo fmt"`, `"vale"`).
    pub command: String,
    /// Optional additional command arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Optional human-readable description for reports.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional environment variable overrides.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Optional execution mode / policy for this gate (e.g. "fail", "warn", "skip").
    #[serde(default)]
    pub mode: Option<GatePolicy>,
}

impl GateDefinition {
    /// Constructs a new `GateDefinition`.
    #[must_use]
    pub fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
            description: None,
            env: HashMap::new(),
            mode: None,
        }
    }

    /// Resolves the execution mode/policy for this gate, defaulting to [`GatePolicy::Fail`].
    #[must_use]
    pub fn mode(&self) -> GatePolicy {
        self.mode.unwrap_or(GatePolicy::Fail)
    }
}

/// Top-level workspace quality gate configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateConfig {
    /// General runner configuration.
    #[serde(default)]
    pub runner: RunnerConfig,
    /// Multi-lane parallel scheduling and authority configuration.
    #[serde(default)]
    pub execution: ExecutionConfig,
    /// Execution policies mapped by gate name (for example, `fmt = "fail"`).
    #[serde(default)]
    pub gates: HashMap<String, GatePolicy>,
    /// Declarative gate definitions (e.g. `[clean]`, `[fmt]`, `[clippy]`, etc.).
    #[serde(flatten)]
    pub gate_definitions: HashMap<String, GateDefinition>,
}

impl GateConfig {
    /// Normalizes configuration by synchronizing execution groups and populating gate policies.
    pub fn normalize(&mut self) {
        self.execution.normalize();
        for (name, def) in &self.gate_definitions {
            let mode = def.mode.unwrap_or_else(|| {
                self.gates.get(name).copied().unwrap_or(GatePolicy::Fail)
            });
            self.gates.insert(name.clone(), mode);
        }
    }

    /// Loads a `GateConfig` from a TOML file path. If the file does not exist,
    /// returns default configuration.
    ///
    /// # Errors
    /// Returns `GateError::Config` if reading or parsing TOML fails.
    pub fn load_from_path(path: &Path) -> Result<Self, GateError> {
        if !path.exists() {
            return Ok(Self::default());
        }

        let content =
            fs::read_to_string(path).map_err(|e| GateError::Config {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        let mut config: Self =
            toml::from_str(&content).map_err(|e| GateError::Config {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
        config.normalize();

        Ok(config)
    }

    /// Gets the policy for a named gate, defaulting to `GatePolicy::Fail`.
    #[must_use]
    pub fn policy_for(&self, gate_name: &str) -> GatePolicy {
        if let Some(def) = self.gate_definitions.get(gate_name) {
            if let Some(mode) = def.mode {
                return mode;
            }
        }
        self.gates
            .get(gate_name)
            .copied()
            .unwrap_or(GatePolicy::Fail)
    }

    /// Resolves the gate definition for a named quality gate from `gate.toml`.
    ///
    /// Returns `None` if the corresponding `[<gate_name>]` table is missing.
    #[must_use]
    pub fn gate_def(&self, gate_name: &str) -> Option<&GateDefinition> {
        self.gate_definitions.get(gate_name)
    }
}
