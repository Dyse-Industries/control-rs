//! Report aggregator and Markdown report generator (`ci-report.md`).

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::config::{GateConfig, GatePolicy};
use crate::error::GateError;
use crate::gates::metrics::MetricsReport;
use crate::gates::valgrind::ValgrindRawReport;
use crate::quality_gate::{GateOutcome, Verdict};

/// Maximum budgeted size for `ci-report.md` (64 KiB).
pub const MAX_REPORT_BYTES: usize = 64 * 1024;

/// Decentralized artifact aggregator and report generator.
#[derive(Debug, Clone)]
pub struct ReportAggregator {
    artifacts_dir: PathBuf,
    workspace_root: PathBuf,
}

impl ReportAggregator {
    /// Constructs a new `ReportAggregator`.
    #[must_use]
    pub fn new(artifacts_dir: PathBuf, workspace_root: PathBuf) -> Self {
        Self {
            artifacts_dir,
            workspace_root,
        }
    }

    /// Ingests all `*.result.json` files found in the artifacts directory.
    ///
    /// # Errors
    /// Returns `GateError` if directory scanning fails.
    pub fn load_outcomes(
        &self,
    ) -> Result<BTreeMap<String, GateOutcome>, GateError> {
        let mut outcomes = BTreeMap::new();
        if !self.artifacts_dir.exists() {
            return Ok(outcomes);
        }

        let entries = fs::read_dir(&self.artifacts_dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with(".result.json") {
                        if let Ok(outcome) = GateOutcome::load_from_file(&path)
                        {
                            outcomes.insert(outcome.gate.clone(), outcome);
                        }
                    }
                }
            }
        }
        Ok(outcomes)
    }

    /// Evaluates fail-closed gate policy across ingested outcomes.
    /// If `subset` is provided, only gates in the subset are required to be present.
    /// Returns true if all required `fail`-policy gates passed.
    #[must_use]
    pub fn is_passing(
        &self,
        config: &GateConfig,
        outcomes: &BTreeMap<String, GateOutcome>,
        subset: Option<&[String]>,
    ) -> bool {
        if let Some(active_subset) = subset {
            for gate_name in active_subset {
                if config.policy_for(gate_name) == GatePolicy::Fail {
                    if let Some(outcome) = outcomes.get(gate_name) {
                        if outcome.verdict == Verdict::Fail {
                            return false;
                        }
                    }
                }
            }
            true
        } else {
            for (gate_name, policy) in &config.gates {
                if *policy == GatePolicy::Fail {
                    match outcomes.get(gate_name) {
                        Some(outcome) => {
                            if outcome.verdict == Verdict::Fail {
                                return false;
                            }
                        }
                        None => {
                            // Missing fail-closed gate fails aggregator (FR-12)
                            return false;
                        }
                    }
                }
            }
            true
        }
    }

    /// Generates the complete `ci-report.md` Markdown content.
    ///
    /// # Errors
    /// Returns `GateError` if file reading or writing fails.
    pub fn generate_markdown(
        &self,
        config: &GateConfig,
        outcomes: &BTreeMap<String, GateOutcome>,
        subset: Option<&[String]>,
    ) -> Result<String, GateError> {
        let mut md = String::new();
        let is_pass = self.is_passing(config, outcomes, subset);

        md.push_str("# Continuous Integration & Verification Report\n\n");
        if is_pass {
            md.push_str("![Overall Status: Pass](https://img.shields.io/badge/CI_Status-Pass-brightgreen)\n\n");
        } else {
            md.push_str("![Overall Status: Fail](https://img.shields.io/badge/CI_Status-Fail-red)\n\n");
        }

        md.push_str("### Executive Summary Matrix\n\n");
        md.push_str("| Gate | Verdict | Duration | Exit Code | Summary |\n");
        md.push_str("|:---|:---|:---|:---|:---|\n");

        for outcome in outcomes.values() {
            if let Some(active_subset) = subset {
                if !active_subset.iter().any(|g| g == &outcome.gate) {
                    continue;
                }
            }

            let verdict_badge = match outcome.verdict {
                Verdict::Pass => "**Pass**",
                Verdict::Warn => "*Warn*",
                Verdict::Fail => "**FAIL**",
                Verdict::Skipped => "Skipped",
            };
            let exit_str =
                outcome.exit_code.map_or("-".to_string(), |c| c.to_string());
            let summary_str = outcome.summary.as_deref().unwrap_or("-");

            md.push_str(&format!(
                "| `{}` | {} | {:.2}s | {} | {} |\n",
                outcome.gate,
                verdict_badge,
                outcome.duration_secs,
                exit_str,
                summary_str
            ));
        }

        // Lightweight Built-in Extraction Section
        let metrics_json = self.artifacts_dir.join("metrics-raw.json");
        let valgrind_json = self.artifacts_dir.join("valgrind-raw.json");
        let geiger_json = self.artifacts_dir.join("geiger-raw.json");

        if metrics_json.exists()
            || valgrind_json.exists()
            || geiger_json.exists()
        {
            md.push_str("\n### Codebase & Memory Metrics\n\n");

            if metrics_json.exists() {
                if let Ok(file) = File::open(&metrics_json) {
                    if let Ok(report) =
                        serde_json::from_reader::<_, MetricsReport>(file)
                    {
                        md.push_str(&format!(
                            "- **Codebase Lines**: {} code, {} comments, {} blank ({} total across {} files)\n",
                            report.total_lines.code_lines,
                            report.total_lines.comment_lines,
                            report.total_lines.blank_lines,
                            report.total_lines.total_lines,
                            report.total_lines.files
                        ));
                    }
                }
            }

            if geiger_json.exists() {
                if let Ok(content) = fs::read_to_string(&geiger_json) {
                    if let Ok(v) =
                        serde_json::from_str::<serde_json::Value>(&content)
                    {
                        if let Some(packages) =
                            v.get("packages").and_then(|p| p.as_array())
                        {
                            let mut unsafe_fns = 0;
                            let mut unsafe_exprs = 0;
                            for pkg in packages {
                                if let Some(used) = pkg
                                    .get("unsafety")
                                    .and_then(|u| u.get("used"))
                                {
                                    if let Some(fns) = used
                                        .get("functions")
                                        .and_then(|f| f.get("unsafe_"))
                                        .and_then(|u| u.as_u64())
                                    {
                                        unsafe_fns += fns;
                                    }
                                    if let Some(exprs) = used
                                        .get("exprs")
                                        .and_then(|e| e.get("unsafe_"))
                                        .and_then(|u| u.as_u64())
                                    {
                                        unsafe_exprs += exprs;
                                    }
                                }
                            }
                            md.push_str(&format!(
                                "- **Unsafe Code Surface (Geiger)**: {} unsafe functions, {} unsafe expressions scanned across {} packages\n",
                                unsafe_fns, unsafe_exprs, packages.len()
                            ));
                        }
                    }
                }
            }

            if valgrind_json.exists() {
                if let Ok(file) = File::open(&valgrind_json) {
                    if let Ok(report) =
                        serde_json::from_reader::<_, ValgrindRawReport>(file)
                    {
                        md.push_str(&format!(
                            "- **Valgrind Memory Leaks**: {} definitely lost bytes, {} indirectly lost bytes ({} memory errors)\n",
                            report.definitely_lost_bytes,
                            report.indirectly_lost_bytes,
                            report.memory_errors
                        ));
                    }
                }
            }
        }

        // Embed Diagnostics for Failed / Warned Gates
        let mut has_diagnostics = false;
        for outcome in outcomes.values() {
            if outcome.verdict == Verdict::Fail
                || outcome.verdict == Verdict::Warn
            {
                if !has_diagnostics {
                    md.push_str("\n### Failure & Diagnostic Logs\n\n");
                    has_diagnostics = true;
                }

                let log_path = self.artifacts_dir.join(&outcome.log_file);
                let log_tail = read_trailing_lines(&log_path, 30);

                md.push_str(&format!(
                    "<details>\n<summary><b>Gate: {} ({:?})</b> - {}</summary>\n\n```text\n{}\n```\n</details>\n\n",
                    outcome.gate,
                    outcome.verdict,
                    outcome.summary.as_deref().unwrap_or(""),
                    log_tail
                ));
            }
        }

        // Check size budget
        if md.len() > MAX_REPORT_BYTES {
            md.truncate(MAX_REPORT_BYTES.saturating_sub(100));
            md.push_str("\n\n*Note: Report truncated to meet 64 KiB platform size budget.*");
        }

        Ok(md)
    }

    /// Renders `ci-report.md` and writes it to both the artifacts directory and workspace root.
    ///
    /// # Errors
    /// Returns `GateError` if writing fails.
    pub fn write_report(
        &self,
        config: &GateConfig,
        subset: Option<&[String]>,
    ) -> Result<(bool, PathBuf), GateError> {
        let outcomes = self.load_outcomes()?;
        let md = self.generate_markdown(config, &outcomes, subset)?;
        let is_pass = self.is_passing(config, &outcomes, subset);

        let report_path = self.workspace_root.join("ci-report.md");
        fs::write(&report_path, &md)?;

        let artifact_report_path = self.artifacts_dir.join("ci-report.md");
        if let Some(parent) = artifact_report_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let _ = fs::write(artifact_report_path, &md);

        Ok((is_pass, report_path))
    }
}

fn read_trailing_lines(path: &Path, max_lines: usize) -> String {
    let Ok(file) = File::open(path) else {
        return "(Log file unavailable)".to_string();
    };
    let reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        lines.push(line);
    }

    let start_idx = lines.len().saturating_sub(max_lines);
    lines
        .get(start_idx..)
        .map(|slice| slice.join("\n"))
        .unwrap_or_default()
}
