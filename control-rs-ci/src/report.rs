//! Report aggregator and Markdown report generator (`ci-report.md`).

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::config::{GateConfig, GatePolicy};
use crate::error::GateError;
use crate::gate::{GateOutcome, Verdict};

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

    /// Returns the configured artifacts directory.
    #[must_use]
    pub fn artifacts_dir(&self) -> &Path {
        &self.artifacts_dir
    }

    /// Returns the workspace root directory.
    #[must_use]
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
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
                            // Missing fail-closed gate fails aggregator (FR-10)
                            return false;
                        }
                    }
                }
            }
            true
        }
    }

    fn determine_ordered_gate_names(
        &self,
        config: &GateConfig,
        outcomes: &BTreeMap<String, GateOutcome>,
    ) -> Vec<String> {
        let mut ordered = Vec::new();

        // 1. User-defined groups in sorted group key order and group element sequence
        let mut sorted_group_keys: Vec<_> =
            config.execution.groups.keys().collect();
        sorted_group_keys.sort();
        for grp in sorted_group_keys {
            if let Some(members) = config.execution.groups.get(grp) {
                for m in members {
                    if !ordered.contains(m) {
                        ordered.push(m.clone());
                    }
                }
            }
        }

        // 2. Exclusive gates
        for m in &config.execution.exclusive_gates {
            if !ordered.contains(m) {
                ordered.push(m.clone());
            }
        }

        // 3. Other configured gates
        let mut other_gates: Vec<_> = config.gates.keys().collect();
        other_gates.sort();
        for g in other_gates {
            if !ordered.contains(g) {
                ordered.push(g.clone());
            }
        }

        // 4. Any remaining outcomes found on disk
        for g in outcomes.keys() {
            if !ordered.contains(g) {
                ordered.push(g.clone());
            }
        }

        ordered
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

        let ordered_names = self.determine_ordered_gate_names(config, outcomes);

        for gate_name in &ordered_names {
            let Some(outcome) = outcomes.get(gate_name) else {
                continue;
            };

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

        // Embed Diagnostics for Failed / Warned Gates
        let mut has_diagnostics = false;
        for gate_name in &ordered_names {
            let Some(outcome) = outcomes.get(gate_name) else {
                continue;
            };

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

    /// Renders `ci-report.md` and writes it to the artifacts directory.
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

        let artifact_report_path = self.artifacts_dir.join("ci-report.md");
        if let Some(parent) = artifact_report_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&artifact_report_path, &md)?;

        Ok((is_pass, artifact_report_path))
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
