//! Structured reporting (`cross-val-report.json`) and Executive Summary Markdown generation (`cross-val-report.md`).

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::HarnessError;

/// High-level summary of the overall cross-validation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total number of suites evaluated.
    pub total_suites: usize,

    /// Number of suites that passed all checks.
    pub passed_suites: usize,

    /// Number of suites with discrepancies or failures.
    pub failed_suites: usize,

    /// Total wall-clock duration across all suites.
    pub total_duration_secs: f64,

    /// High-level verdict string ("Pass" or "Fail").
    pub verdict: String,
}

/// Comprehensive cross-validation outcome report document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Summary metrics.
    pub summary: ValidationSummary,

    /// Detailed per-suite results.
    pub suites: Vec<SuiteReport>,
}

/// Per-suite validation findings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuiteReport {
    /// Canonical suite name.
    pub name: String,

    /// Suite verdict ("Pass" or "Fail").
    pub status: String,

    /// Execution and comparison duration in seconds.
    pub duration_secs: f64,

    /// Comparison results for all evaluated signals.
    pub comparisons: Vec<ComparisonFinding>,

    /// Optional failure explanation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

/// Evaluation record for a single dataset / signal comparison pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonFinding {
    /// Unique comparison key (for example, "`buck_converter.transient/v_out.rust`").
    pub key: String,

    /// Variant pair evaluated (for example, ("scipy", "rust")).
    pub pair: (String, String),

    /// Canonical signal path.
    pub signal: String,

    /// Satisfaction policy applied ("`all_of`" or "`any_of`").
    pub policy: String,

    /// Verdict for this signal comparison ("pass" or "fail").
    pub verdict: String,

    /// Individual method evaluation findings.
    pub methods: Vec<MethodFinding>,
}

/// Method-level evaluation details.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodFinding {
    /// Method type name (for example, "`abs`", "`rms`", "`regex_match`").
    pub r#type: String,

    /// Configured threshold or tolerance bound.
    pub bound: f64,

    /// Computed numerical score or error residual.
    pub observed: f64,

    /// Verdict for this method ("pass" or "fail").
    pub verdict: String,

    /// Optional failure details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl ValidationReport {
    /// Writes `cross-val-report.json` and `cross-val-report.md` to the specified directory.
    ///
    /// # Errors
    /// Returns `HarnessError` if serialization or file writing fails.
    pub fn save_reports(&self, out_dir: &Path) -> Result<(), HarnessError> {
        std::fs::create_dir_all(out_dir)?;

        // 1. JSON Report
        let json_path = out_dir.join("cross-val-report.json");
        let json_file = File::create(&json_path)?;
        serde_json::to_writer_pretty(json_file, self)?;

        // 2. Markdown Report
        let md_path = out_dir.join("cross-val-report.md");
        let md_content = self.render_markdown();
        std::fs::write(&md_path, md_content)?;

        Ok(())
    }

    /// Renders the Executive Summary Markdown document (`cross-val-report.md`).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("### Cross-Comparison & Oracle Validation Brief\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("| :--- | :--- |\n");
        let _ = writeln!(md, "| **Status** | {} |", self.summary.verdict);
        let _ = writeln!(
            md,
            "| **Suites Verified** | {} / {} passed ({:.2}s total) |",
            self.summary.passed_suites,
            self.summary.total_suites,
            self.summary.total_duration_secs
        );
        md.push_str(
            "| **Regression Artifacts** | `cross-val-report.json`, `cross-val-report.md` |\n\n",
        );

        md.push_str(
            "| Suite | Status | Duration | Comparisons | Methods Evaluated |\n",
        );
        md.push_str("| :--- | :--- | :--- | :--- | :--- |\n");

        for suite in &self.suites {
            let mut unique_methods = BTreeSet::new();
            for comp in &suite.comparisons {
                for m in &comp.methods {
                    unique_methods.insert(format!("`{}`", m.r#type));
                }
            }

            let methods_str = if unique_methods.is_empty() {
                "-".to_string()
            } else {
                let list: Vec<String> = unique_methods.into_iter().collect();
                list.join(", ")
            };

            let _ = writeln!(
                md,
                "| `{}` | {} | {:.2}s | {} signal(s) | {} |",
                suite.name,
                suite.status,
                suite.duration_secs,
                suite.comparisons.len(),
                methods_str
            );
        }

        // Include failure diagnostics if any suites failed
        let has_failures = self.suites.iter().any(|s| s.status == "Fail");
        if has_failures {
            md.push_str("\n### Discrepancy & Diagnostic Details\n\n");
            for suite in &self.suites {
                if suite.status == "Fail" {
                    let _ = writeln!(md, "#### Suite: `{}`\n", suite.name);
                    if let Some(reason) = &suite.failure_reason {
                        let _ = writeln!(md, "- **Error**: {reason}\n");
                    }
                    for comp in &suite.comparisons {
                        if comp.verdict == "fail" {
                            let _ = writeln!(
                                md,
                                "- **Signal `{}`** (Pair: `{}` vs `{}`):",
                                comp.signal, comp.pair.0, comp.pair.1
                            );
                            for m in &comp.methods {
                                if m.verdict == "fail" {
                                    let _ = writeln!(
                                        md,
                                        "  - Method `{}`: observed={:.3e}, bound={:.3e} ({})",
                                        m.r#type,
                                        m.observed,
                                        m.bound,
                                        m.details
                                            .as_deref()
                                            .unwrap_or("exceeded bound")
                                    );
                                }
                            }
                        }
                    }
                    md.push('\n');
                }
            }
        }

        md
    }
}
