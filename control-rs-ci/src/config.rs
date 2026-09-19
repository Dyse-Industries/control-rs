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

/// Codebase metrics gate settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Maximum allowed source code lines across tracked directories.
    pub max_source_lines: Option<usize>,
    /// List of tracked workspace directories to scan.
    #[serde(default)]
    pub tracked_dirs: Vec<String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            max_source_lines: Some(100_000),
            tracked_dirs: vec!["src".to_string()],
        }
    }
}

/// Git repository hygiene gate settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitConfig {
    /// If true, working tree must have no untracked or modified files.
    #[serde(default)]
    pub require_clean_working_tree: bool,
    /// If true, commit messages on branch must follow Conventional Commits.
    #[serde(default = "default_true")]
    pub enforce_conventional_commits: bool,
    /// Disallowed commit summary substrings (e.g. "wip", "temp").
    #[serde(default)]
    pub disallowed_patterns: Vec<String>,
    /// Maximum character length for commit subject lines.
    #[serde(default = "default_max_header_length")]
    pub max_header_length: usize,
}

const fn default_true() -> bool {
    true
}

const fn default_max_header_length() -> usize {
    72
}

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            require_clean_working_tree: false,
            enforce_conventional_commits: true,
            disallowed_patterns: vec![
                "wip".to_string(),
                "temp".to_string(),
                "asdf".to_string(),
                "fix typo".to_string(),
            ],
            max_header_length: 72,
        }
    }
}

/// Vale prose linter gate settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValeConfig {
    /// Path to the Vale configuration file.
    #[serde(default = "default_vale_config")]
    pub config: String,
    /// Paths or directories to lint.
    #[serde(default)]
    pub paths: Vec<String>,
}

fn default_vale_config() -> String {
    ".vale.ini".to_string()
}

impl Default for ValeConfig {
    fn default() -> Self {
        Self {
            config: default_vale_config(),
            paths: vec!["documentation".to_string(), "src".to_string()],
        }
    }
}

/// Geiger unsafe memory audit settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeigerConfig {
    /// Maximum allowed unsafe blocks across workspace crates.
    #[serde(default)]
    pub max_unsafe_blocks: usize,
}

/// SemVer API compatibility gate settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemverConfig {
    /// Baseline git ref to compare public API against.
    #[serde(default = "default_baseline_ref")]
    pub baseline_ref: String,
}

fn default_baseline_ref() -> String {
    "origin/main".to_string()
}

impl Default for SemverConfig {
    fn default() -> Self {
        Self {
            baseline_ref: default_baseline_ref(),
        }
    }
}

/// Mutation testing gate settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutantsConfig {
    /// Test execution timeout multiplier for mutated runs.
    #[serde(default = "default_timeout_multiplier")]
    pub timeout_multiplier: f64,
    /// Number of concurrent cargo build/test jobs (defaults to available logical cores).
    pub jobs: Option<usize>,
    /// Globs for files or paths to exclude from mutation (e.g. tests, examples, benches).
    #[serde(default = "default_mutants_exclude")]
    pub exclude: Vec<String>,
    /// Regex pattern for mutations to exclude.
    #[serde(default = "default_mutants_exclude_re")]
    pub exclude_re: Option<String>,
    /// File globs to examine for mutation (e.g. `src/**`).
    #[serde(default = "default_mutants_files")]
    pub files: Vec<String>,
}

const fn default_timeout_multiplier() -> f64 {
    2.0
}

fn default_mutants_exclude() -> Vec<String> {
    vec![
        "tests/**".to_string(),
        "examples/**".to_string(),
        "benches/**".to_string(),
    ]
}

fn default_mutants_exclude_re() -> Option<String> {
    Some("(^|::)(test|tests|example|examples|bench|benches)".to_string())
}

fn default_mutants_files() -> Vec<String> {
    vec!["src/**".to_string()]
}

impl Default for MutantsConfig {
    fn default() -> Self {
        Self {
            timeout_multiplier: default_timeout_multiplier(),
            jobs: None,
            exclude: default_mutants_exclude(),
            exclude_re: default_mutants_exclude_re(),
            files: default_mutants_files(),
        }
    }
}

/// Valgrind Memcheck gate settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValgrindConfig {
    /// Leak check level (e.g. "full").
    #[serde(default = "default_leak_check")]
    pub leak_check: String,
    /// Exit status returned on memory leak or error.
    #[serde(default = "default_error_exitcode")]
    pub error_exitcode: i32,
    /// List of host binary targets to check.
    #[serde(default)]
    pub bins: Vec<String>,
    /// List of workspace example targets to check.
    #[serde(default)]
    pub examples: Vec<String>,
}

fn default_leak_check() -> String {
    "full".to_string()
}

const fn default_error_exitcode() -> i32 {
    1
}

impl Default for ValgrindConfig {
    fn default() -> Self {
        Self {
            leak_check: default_leak_check(),
            error_exitcode: 1,
            bins: vec![],
            examples: vec![],
        }
    }
}

/// Top-level workspace quality gate configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateConfig {
    /// General runner configuration.
    #[serde(default)]
    pub runner: RunnerConfig,
    /// Execution policies mapped by gate name (e.g. `fmt = "fail"`).
    #[serde(default)]
    pub gates: HashMap<String, GatePolicy>,
    /// Codebase metrics settings.
    #[serde(default)]
    pub metrics: MetricsConfig,
    /// Git hygiene settings.
    #[serde(default)]
    pub git: GitConfig,
    /// Vale prose settings.
    #[serde(default)]
    pub vale: ValeConfig,
    /// Geiger unsafe scanner settings.
    #[serde(default)]
    pub geiger: GeigerConfig,
    /// SemVer stability settings.
    #[serde(default)]
    pub semver: SemverConfig,
    /// Mutants mutation testing settings.
    #[serde(default)]
    pub mutants: MutantsConfig,
    /// Valgrind memory safety settings.
    #[serde(default)]
    pub valgrind: ValgrindConfig,
}

impl GateConfig {
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

        let config: Self =
            toml::from_str(&content).map_err(|e| GateError::Config {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        Ok(config)
    }

    /// Gets the policy for a named gate, defaulting to `GatePolicy::Fail`.
    #[must_use]
    pub fn policy_for(&self, gate_name: &str) -> GatePolicy {
        self.gates
            .get(gate_name)
            .copied()
            .unwrap_or(GatePolicy::Fail)
    }
}
