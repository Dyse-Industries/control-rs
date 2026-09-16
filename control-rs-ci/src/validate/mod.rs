//! Multi-target cross-comparison and oracle verification subsystem.
//!
//! Each suite names a cargo `bin`, a true-oracle variant, and external
//! `commands`. The runner starts cargo, then one subprocess per command,
//! globs `results/<name>.*.h5`, and 1:1-compares peers against the true oracle.

use std::fmt::Write as _;
use std::path::Path;

pub use comparator::{
    CompareOutcome, ComparisonRecord, CrossValidation, PeerPath,
    ValidationResult, compare_h5_files,
};
pub use envelope::{
    ContainerValidationReport, EnvelopeValidationReport, FreshnessPolicy,
    ResultEnvelope,
};
pub use h5::{
    DatasetRead, DatasetTolerance, DiscoveredDataset, DiscoveredDatasets,
    H5Container, JsonObject, PeerBound, ShapeAndData, SuiteH5File,
    discover_suite_h5_files, project_json_paths, retain_numeric_json,
    validate_suite_files,
};
pub use runner::{
    CommandArgv, CommandList, ExampleRunOutcome, ExampleTargetConfig,
    run_example, run_multiple_examples,
};
pub use tolerance::{Interval, LoadedTable, ToleranceBound, ToleranceTable};

pub mod comparator;
pub mod envelope;
pub mod h5;
pub mod runner;
pub mod tolerance;

/// Maximum discrepancy rows rendered in the CI report before truncation.
const MAX_ERROR_ROWS: usize = 15;

/// Lines of example output rendered when a failure produced no findings.
const OUTPUT_TAIL_LINES: usize = 20;

/// Summary of all cross-comparison runs across $N$ examples.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CrossComparisonSummary {
    /// Total number of examples executed.
    pub total_examples: usize,
    /// Number of examples that passed completely.
    pub passed_examples: usize,
    /// Number of examples that failed or breached tolerances.
    pub failed_examples: usize,
    /// Total execution time in seconds.
    pub total_duration_secs: f32,
    /// Detailed outcomes per example.
    pub outcomes: Vec<ExampleRunOutcome>,
}

/// Parsed structure of `validate.toml`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidateConfigFile {
    /// General validation settings.
    #[serde(default)]
    pub validate: ValidateGeneralConfig,
    /// Configured validation suites.
    #[serde(default)]
    pub suites: Vec<ValidateSuiteConfig>,
}

/// General validation options.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidateGeneralConfig {
    /// Report title.
    #[serde(default = "default_title")]
    pub title: String,
    /// Output directory for artifacts.
    #[serde(default = "default_out_dir")]
    pub out_dir: String,
    /// Per-example execution timeout.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Strict tolerance gating flag.
    #[serde(default = "default_true")]
    pub strict: bool,
}

/// Type alias for validation suite configuration.
pub type ValidateSuiteConfig = ExampleTargetConfig;

impl Default for ValidateGeneralConfig {
    fn default() -> Self {
        Self {
            title: default_title(),
            out_dir: default_out_dir(),
            timeout_secs: default_timeout(),
            strict: default_true(),
        }
    }
}

fn default_out_dir() -> String {
    "target/ci".to_string()
}

const fn default_timeout() -> u64 {
    90
}

fn default_title() -> String {
    "control-rs".to_string()
}

const fn default_true() -> bool {
    true
}

/// Load `validate.toml` from a path.
#[must_use]
pub fn load_validate_config(path: &Path) -> Option<ValidateConfigFile> {
    if path.exists()
        && let Ok(content) = std::fs::read_to_string(path)
        && let Ok(cfg) = toml::from_str::<ValidateConfigFile>(&content)
    {
        return Some(cfg);
    }
    None
}

/// Render a high-level executive brief suitable for inclusion in `ci-report.md`.
#[must_use]
pub fn render_cross_comparison_brief(
    summary: &CrossComparisonSummary,
) -> String {
    let status_str = if summary.failed_examples == 0 {
        "Pass"
    } else {
        "Discrepancies Found"
    };

    let mut out = String::new();
    out.push_str("### Cross-Comparison & Oracle Validation Brief\n\n");
    let _ = write!(
        out,
        "| Metric | Value |\n\
         | :--- | :--- |\n\
         | **Status** | {status_str} |\n\
         | **Examples Verified** | {} / {} passed ({:.2}s total) |\n\
         | **HDF5 Attributes** | {} |\n\
         | **Regression Artifacts** | See `cross-val-report.json` |\n\n",
        summary.passed_examples,
        summary.total_examples,
        summary.total_duration_secs,
        tolerance_provenance(summary),
    );
    render_outcome_rows(summary, &mut out);
    render_failure_rows(summary, &mut out);
    render_no_findings(summary, &mut out);
    out
}

fn render_failure_rows(summary: &CrossComparisonSummary, out: &mut String) {
    let mut failures = Vec::new();
    for outcome in &summary.outcomes {
        for env in &outcome.envelopes {
            for err in &env.errors {
                failures.push((
                    outcome.name.as_str(),
                    env.subject.as_str(),
                    err.as_str(),
                ));
            }
        }
    }

    if failures.is_empty() {
        return;
    }

    let total = failures.len();
    out.push_str("#### Validation Errors & Tolerance Breaches\n\n");
    out.push_str("| Example | Subject | Discrepancy / Breach |\n");
    out.push_str("| :--- | :--- | :--- |\n");
    for (ex, subj, err) in failures.iter().take(MAX_ERROR_ROWS) {
        let escaped = err.replace('|', "\\|");
        let _ = writeln!(out, "| `{ex}` | `{subj}` | {escaped} |");
    }
    if total > MAX_ERROR_ROWS {
        let _ = write!(
            out,
            "\n*... and {} further findings in `cross-val-report.json`*\n",
            total.saturating_sub(MAX_ERROR_ROWS)
        );
    }
    out.push('\n');
}

fn render_no_findings(summary: &CrossComparisonSummary, out: &mut String) {
    for outcome in summary.outcomes.iter().filter(|o| !o.success) {
        let has_findings =
            outcome.envelopes.iter().any(|e| !e.errors.is_empty());
        if has_findings {
            continue;
        }

        let lines: Vec<&str> = outcome.output.lines().collect();
        let start = lines.len().saturating_sub(OUTPUT_TAIL_LINES);
        let _ = write!(
            out,
            "#### `{}` failed without producing findings\n\n",
            outcome.name
        );
        out.push_str("```\n");
        if let Some(tail) = lines.get(start..) {
            for line in tail {
                out.push_str(line);
                out.push('\n');
            }
        }
        out.push_str("```\n\n");
    }
}

fn render_outcome_rows(summary: &CrossComparisonSummary, out: &mut String) {
    out.push_str("| Example | Status | Duration | Result Envelopes |\n");
    out.push_str("| :--- | :--- | :--- | :--- |\n");

    for outcome in &summary.outcomes {
        let status = if outcome.success { "Pass" } else { "**FAIL**" };
        let env_count = outcome.envelopes.len();
        let cmp_count: usize =
            outcome.envelopes.iter().map(|e| e.comparisons.len()).sum();
        let _ = writeln!(
            out,
            "| `{}` | {} | {:.2}s | {} envelope(s), {} comparison(s) |",
            outcome.name, status, outcome.duration_secs, env_count, cmp_count
        );
    }
    out.push('\n');
}

/// Run cross-comparison verification over given example targets.
#[must_use]
pub fn run_cross_comparison(
    example_targets: &[ExampleTargetConfig],
    repo_root: &Path,
    default_timeout_secs: u64,
) -> CrossComparisonSummary {
    let outcomes =
        run_multiple_examples(example_targets, repo_root, default_timeout_secs);

    let total_examples = outcomes.len();
    let mut passed_examples: usize = 0;
    let mut failed_examples: usize = 0;
    let mut total_duration_secs = 0.0f32;

    for out in &outcomes {
        total_duration_secs += out.duration_secs;
        if out.success {
            passed_examples = passed_examples.saturating_add(1);
        } else {
            failed_examples = failed_examples.saturating_add(1);
        }
    }

    CrossComparisonSummary {
        total_examples,
        passed_examples,
        failed_examples,
        total_duration_secs,
        outcomes,
    }
}

/// Describe the tolerance tables the verdicts were re-derived against.
fn tolerance_provenance(summary: &CrossComparisonSummary) -> String {
    let loaded =
        summary
            .outcomes
            .iter()
            .find(|o| o.tolerance_keys > 0)
            .map(|o| {
                format!("{} ({} bounds)", o.tolerance_source, o.tolerance_keys)
            });

    loaded.unwrap_or_else(|| "**none loaded**".to_string())
}
