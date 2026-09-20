//! Unified modular configuration schema and recursive loader for `compare.toml`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::HarnessError;

/// Backward-compatible type alias for `CompareConfigFile`.
pub type OracleConfigFile = CompareConfigFile;

/// Backward-compatible type alias for `CompareGeneralConfig`.
pub type OracleGeneralConfig = CompareGeneralConfig;

/// Root or suite-level `compare.toml` configuration document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompareConfigFile {
    /// General orchestrator settings.
    #[serde(default, alias = "oracle")]
    pub compare: CompareGeneralConfig,

    /// Optional child suite directory paths to load recursively.
    #[serde(default)]
    pub suites: Vec<String>,

    /// Inline suite definitions declared in this configuration file.
    #[serde(default, rename = "suite")]
    pub inlined_suites: Vec<SuiteConfig>,
}

/// Global orchestrator and execution parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareGeneralConfig {
    /// Human-readable title for the validation run.
    #[serde(default = "default_title")]
    pub title: String,

    /// Target output directory for result `.h5` containers and reports.
    #[serde(default = "default_out_dir")]
    pub out_dir: String,

    /// Default process execution timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// If true, fail-closed policy exits with non-zero status on discrepancy.
    #[serde(default = "default_true")]
    pub strict: bool,

    /// Optional child suite directories specified under the [compare] table.
    #[serde(default)]
    pub suites: Vec<String>,

    /// Optional thread count override for parallel chunked evaluation.
    #[serde(default)]
    pub threads: Option<usize>,
}

/// Declaration of a single validation suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteConfig {
    /// Canonical suite name (for example, "`buck_converter`", "matrix").
    pub name: String,

    /// Name of the reference oracle variant (default: "scipy").
    #[serde(default = "default_oracle")]
    pub true_oracle: String,

    /// Optional path to an external single-source TOML tolerance table.
    #[serde(default)]
    pub tolerance_table: Option<String>,

    /// Optional explicitly provided list of signals/datasets to verify.
    #[serde(default)]
    pub signals: Option<Vec<String>>,

    /// List of execution variants configured for this suite.
    #[serde(default)]
    pub variants: Vec<VariantConfig>,
}

/// Single-source TOML tolerance table containing per-signal numerical tolerances.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToleranceTable {
    /// Tolerances mapped by signal path (for example, "matrix/a" or "transient/v_out").
    #[serde(default)]
    pub tolerances: BTreeMap<String, SignalToleranceConfig>,

    /// Alternative table key `[signals]`.
    #[serde(default)]
    pub signals: BTreeMap<String, SignalToleranceConfig>,
}

/// Tolerance configuration for a specific signal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalToleranceConfig {
    /// Target subject/suite name (optional filter).
    pub subject: Option<String>,

    /// Target signal path (optional if table key is signal path).
    pub signal: Option<String>,

    /// Single method name ("abs", "rel", "rms", "matrix_norm", etc.).
    pub method: Option<String>,

    /// Numerical bound for single method.
    pub bound: Option<f64>,

    /// Multi-method list.
    #[serde(default)]
    pub methods: Vec<ToleranceMethodConfig>,

    /// Satisfaction policy ("all_of" or "any_of").
    #[serde(default = "default_policy")]
    pub policy: String,

    /// Peer-specific bound overrides (e.g. `peer_bounds.ngspice = 1e-2`).
    #[serde(default)]
    pub peer_bounds: BTreeMap<String, f64>,
}

/// A single tolerance method configuration in a multi-method list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToleranceMethodConfig {
    /// Method type name ("abs", "rel", "rms", "matrix_norm", "exact_match", etc.).
    #[serde(rename = "type", alias = "method")]
    pub r#type: String,

    /// Numerical threshold or tolerance bound.
    pub bound: f64,
}

impl ToleranceTable {
    /// Loads a `tolerance_table.toml` file from disk.
    ///
    /// # Errors
    /// Returns `HarnessError::Config` if reading or parsing TOML fails.
    pub fn load_from_file(path: &Path) -> Result<Self, HarnessError> {
        let content =
            fs::read_to_string(path).map_err(|e| HarnessError::Config {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        let parsed: Self =
            toml::from_str(&content).map_err(|e| HarnessError::Config {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        Ok(parsed)
    }

    /// Finds the tolerance configuration for a given signal path.
    #[must_use]
    pub fn find_signal(&self, signal: &str) -> Option<&SignalToleranceConfig> {
        if let Some(cfg) = self.tolerances.get(signal) {
            return Some(cfg);
        }
        if let Some(cfg) = self.signals.get(signal) {
            return Some(cfg);
        }
        self.tolerances
            .values()
            .chain(self.signals.values())
            .find(|cfg| cfg.signal.as_deref() == Some(signal))
    }
}

fn default_policy() -> String {
    "all_of".to_string()
}

/// Execution specification for a single language or model variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantConfig {
    /// Variant identifier (for example, "rust", "scipy", "jax", "ngspice").
    pub name: String,

    /// Execution engine type ("`rust_bin`", "`python_script`", or "`command`").
    pub r#type: String,

    /// Cargo manifest path for `rust_bin` variants.
    #[serde(default)]
    pub manifest_path: Option<String>,

    /// Target binary name for `rust_bin` variants.
    #[serde(default)]
    pub bin: Option<String>,

    /// Script path for `python_script` variants.
    #[serde(default)]
    pub script: Option<String>,

    /// Raw command string for `command` variants.
    #[serde(default)]
    pub command: Option<String>,

    /// Relative path where this variant deposits its `.h5` output container.
    pub output_file: String,

    /// If true, variant failure produces a warning rather than blocking gate failure.
    #[serde(default)]
    pub optional: bool,
}

/// Resolved master execution plan aggregating all discovery and configuration files.
#[derive(Debug, Clone)]
pub struct MasterPlan {
    /// Global runner settings.
    pub general: CompareGeneralConfig,

    /// All discovered and validated suite configurations.
    pub suites: Vec<SuiteConfig>,
}

impl Default for CompareGeneralConfig {
    fn default() -> Self {
        Self {
            title: default_title(),
            out_dir: default_out_dir(),
            timeout_secs: default_timeout_secs(),
            strict: default_true(),
            suites: Vec::new(),
            threads: None,
        }
    }
}

impl CompareConfigFile {
    /// Loads a `compare.toml` configuration file from disk.
    ///
    /// # Errors
    /// Returns `HarnessError::Config` if reading or parsing TOML fails.
    pub fn load_from_file(path: &Path) -> Result<Self, HarnessError> {
        let content =
            fs::read_to_string(path).map_err(|e| HarnessError::Config {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        let parsed: Self =
            toml::from_str(&content).map_err(|e| HarnessError::Config {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        Ok(parsed)
    }

    /// Recursively resolves all child suites and builds a normalized `MasterPlan`.
    ///
    /// # Errors
    /// Returns `HarnessError::Config` if loading any referenced child suite fails.
    pub fn resolve_master_plan(
        &self,
        config_path: &Path,
    ) -> Result<MasterPlan, HarnessError> {
        let base_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
        let mut master_suites = Vec::new();

        // 1. Process referenced child suite directories
        let mut all_suite_refs = self.suites.clone();
        for s in &self.compare.suites {
            if !all_suite_refs.contains(s) {
                all_suite_refs.push(s.clone());
            }
        }

        for suite_rel_path in &all_suite_refs {
            let child_suite_dir = base_dir.join(suite_rel_path);
            let child_config_path = if child_suite_dir.is_file() {
                child_suite_dir.clone()
            } else if child_suite_dir.join("compare.toml").exists() {
                child_suite_dir.join("compare.toml")
            } else {
                child_suite_dir.join("oracle.toml")
            };

            if !child_config_path.exists() {
                return Err(HarnessError::Config {
                    path: child_config_path,
                    message: format!(
                        "Referenced suite configuration '{suite_rel_path}' not found"
                    ),
                });
            }

            let child_file = Self::load_from_file(&child_config_path)?;
            let child_base =
                child_config_path.parent().unwrap_or_else(|| Path::new("."));

            for mut suite in child_file.inlined_suites {
                normalize_suite_paths(&mut suite, child_base);
                master_suites.push(suite);
            }
        }

        // 2. Process inlined suites from the target config (these take precedence)
        for mut inline_suite in self.inlined_suites.clone() {
            normalize_suite_paths(&mut inline_suite, base_dir);
            // Replace existing suite if already present, or push
            if let Some(existing) = master_suites
                .iter_mut()
                .find(|s| s.name == inline_suite.name)
            {
                *existing = inline_suite;
            } else {
                master_suites.push(inline_suite);
            }
        }

        Ok(MasterPlan {
            general: self.compare.clone(),
            suites: master_suites,
        })
    }
}

fn default_title() -> String {
    "control-rs Cross-Comparison Suite".to_string()
}

fn default_out_dir() -> String {
    "results".to_string()
}

const fn default_timeout_secs() -> u64 {
    120
}

const fn default_true() -> bool {
    true
}

fn default_oracle() -> String {
    "scipy".to_string()
}

fn normalize_suite_paths(suite: &mut SuiteConfig, base_dir: &Path) {
    if let Some(tol) = &suite.tolerance_table {
        let path = PathBuf::from(tol);
        if path.is_relative() {
            suite.tolerance_table =
                Some(clean_path(&base_dir.join(&path)).display().to_string());
        }
    }

    for variant in &mut suite.variants {
        if let Some(manifest) = &variant.manifest_path {
            let path = PathBuf::from(manifest);
            if path.is_relative() {
                variant.manifest_path = Some(
                    clean_path(&base_dir.join(&path)).display().to_string(),
                );
            }
        }
        if let Some(script) = &variant.script {
            let path = PathBuf::from(script);
            if path.is_relative() {
                variant.script = Some(
                    clean_path(&base_dir.join(&path)).display().to_string(),
                );
            }
        }
        let out_path = PathBuf::from(&variant.output_file);
        if out_path.is_relative() {
            variant.output_file =
                clean_path(&base_dir.join(&out_path)).display().to_string();
        }
    }
}

fn clean_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for comp in path.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if components.last().is_some_and(|last| {
                    *last != std::path::Component::ParentDir
                }) {
                    components.pop();
                    continue;
                }
                components.push(comp);
            }
            _ => components.push(comp),
        }
    }
    components.into_iter().collect()
}
