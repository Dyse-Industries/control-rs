//! Report aggregator and Markdown report generator (`ci-report.md`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::GateFilter;
use crate::config::{GateConfig, GatePolicy};
use crate::error::GateResult;
use crate::gate::{GateOutcome, RESULT_SUFFIX, Verdict};

/// Maximum budgeted size for `ci-report.md` (64 KiB).
pub const MAX_REPORT_BYTES: usize = 64 * 1024;

/// Gate outcomes keyed by gate name.
pub type Outcomes = BTreeMap<String, GateOutcome>;

/// Decentralized artifact aggregator and report generator.
#[derive(Debug, Clone)]
pub struct ReportAggregator {
    artifacts_dir: PathBuf,
    workspace_root: PathBuf,
}

/// A rendered `ci-report.md` on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenReport {
    /// Whether every required fail-closed gate passed.
    pub pass: bool,
    /// Location of the written report.
    pub path: PathBuf,
}

impl ReportAggregator {
    /// Constructs a new `ReportAggregator`.
    #[must_use]
    pub const fn new(artifacts_dir: PathBuf, workspace_root: PathBuf) -> Self {
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
    pub fn load_outcomes(&self) -> GateResult<Outcomes> {
        let mut outcomes = BTreeMap::new();
        if !self.artifacts_dir.exists() {
            return Ok(outcomes);
        }

        for entry in fs::read_dir(&self.artifacts_dir)? {
            let path = entry?.path();
            if path.is_file()
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|name| name.ends_with(RESULT_SUFFIX))
                && let Ok(outcome) = GateOutcome::load_from_file(&path)
            {
                outcomes.insert(outcome.gate.clone(), outcome);
            }
        }
        Ok(outcomes)
    }

    /// Evaluates fail-closed gate policy across ingested outcomes.
    /// Without `subset`, every `fail` gate must have a non-failing result.
    /// With `subset`, only the `fail` gates in the subset are required.
    /// Returns true if all required `fail`-policy gates passed.
    #[must_use]
    pub fn is_passing(
        &self,
        config: &GateConfig,
        outcomes: &Outcomes,
        subset: GateFilter<'_>,
    ) -> bool {
        let failed = |gate_name: &String| {
            outcomes
                .get(gate_name)
                .is_some_and(|o| o.verdict == Verdict::Fail)
        };
        subset.map_or_else(
            || {
                // A missing fail-closed gate fails the aggregator (FR-10).
                config.gates.iter().all(|(gate_name, policy)| {
                    *policy != GatePolicy::Fail
                        || outcomes.contains_key(gate_name)
                            && !failed(gate_name)
                })
            },
            |active_subset| {
                // A selected fail-closed gate without a result fails (FR-15).
                active_subset.iter().all(|gate_name| {
                    config.policy_for(gate_name) != GatePolicy::Fail
                        || outcomes.contains_key(gate_name)
                            && !failed(gate_name)
                })
            },
        )
    }

    /// Generates the complete `ci-report.md` Markdown content.
    ///
    /// # Errors
    /// Returns `GateError` if file reading or writing fails.
    pub fn generate_markdown(
        &self,
        config: &GateConfig,
        outcomes: &Outcomes,
        subset: GateFilter<'_>,
    ) -> GateResult<String> {
        let mut md = String::new();

        md.push_str("# Continuous Integration & Verification Report\n\n");
        if self.is_passing(config, outcomes, subset) {
            md.push_str("![Overall Status: Pass](https://img.shields.io/badge/CI_Status-Pass-brightgreen)\n\n");
        } else {
            md.push_str("![Overall Status: Fail](https://img.shields.io/badge/CI_Status-Fail-red)\n\n");
        }

        let ordered_names = ordered_gate_names(config, outcomes);
        push_summary_matrix(&mut md, &ordered_names, outcomes, subset);
        self.push_diagnostics(&mut md, &ordered_names, outcomes);

        // Check size budget
        if md.len() > MAX_REPORT_BYTES {
            md.truncate(MAX_REPORT_BYTES.saturating_sub(100));
            md.push_str("\n\n*Note: Report truncated to meet 64 KiB platform size budget.*");
        }

        Ok(md)
    }

    /// Appends the log tails of failed and warned gates.
    fn push_diagnostics(
        &self,
        md: &mut String,
        ordered_names: &[String],
        outcomes: &Outcomes,
    ) {
        let flagged = ordered_names
            .iter()
            .filter_map(|name| outcomes.get(name))
            .filter(|o| matches!(o.verdict, Verdict::Fail | Verdict::Warn));
        for (idx, outcome) in flagged.enumerate() {
            if idx == 0 {
                md.push_str("\n### Failure & Diagnostic Logs\n\n");
            }
            let log_path = self.artifacts_dir.join(&outcome.log_file);
            let log_tail = read_trailing_lines(&log_path, 30);
            let _ = write!(
                md,
                "<details>\n<summary><b>Gate: {} ({:?})</b> - {}</summary>\n\n```text\n{}\n```\n</details>\n\n",
                outcome.gate,
                outcome.verdict,
                outcome.summary.as_deref().unwrap_or(""),
                log_tail
            );
        }
    }

    /// Renders `ci-report.md` and writes it to the artifacts directory.
    ///
    /// # Errors
    /// Returns `GateError` if writing fails.
    pub fn write_report(
        &self,
        config: &GateConfig,
        subset: GateFilter<'_>,
    ) -> GateResult<WrittenReport> {
        let outcomes = self.load_outcomes()?;
        let md = self.generate_markdown(config, &outcomes, subset)?;
        let pass = self.is_passing(config, &outcomes, subset);

        let path = self.artifacts_dir.join("ci-report.md");
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &md)?;

        Ok(WrittenReport { pass, path })
    }
}

/// Gate names in report order: grouped gates (sorted group keys, member
/// order), exclusive gates, other configured gates, then any other outcome
/// found on disk.
fn ordered_gate_names(config: &GateConfig, outcomes: &Outcomes) -> Vec<String> {
    let mut ordered = Vec::new();
    let mut push = |name: &String| {
        if !ordered.contains(name) {
            ordered.push(name.clone());
        }
    };

    let mut sorted_group_keys: Vec<_> =
        config.execution.groups.keys().collect();
    sorted_group_keys.sort();
    for grp in sorted_group_keys {
        if let Some(members) = config.execution.groups.get(grp) {
            members.iter().for_each(&mut push);
        }
    }
    config.execution.exclusive_gates.iter().for_each(&mut push);

    let mut other_gates: Vec<_> = config.gates.keys().collect();
    other_gates.sort();
    other_gates.into_iter().for_each(&mut push);

    outcomes.keys().for_each(&mut push);
    ordered
}

/// Appends the executive summary table, limited to `subset` when given.
fn push_summary_matrix(
    md: &mut String,
    ordered_names: &[String],
    outcomes: &Outcomes,
    subset: GateFilter<'_>,
) {
    md.push_str("### Executive Summary Matrix\n\n");
    md.push_str("| Gate | Verdict | Duration | Exit Code | Summary |\n");
    md.push_str("|:---|:---|:---|:---|:---|\n");

    for outcome in ordered_names.iter().filter_map(|name| outcomes.get(name)) {
        if subset
            .is_some_and(|active| !active.iter().any(|g| g == &outcome.gate))
        {
            continue;
        }

        let verdict_badge = match outcome.verdict {
            Verdict::Pass => "**Pass**",
            Verdict::Warn => "*Warn*",
            Verdict::Fail => "**FAIL**",
            Verdict::Skipped => "Skipped",
        };
        let exit_str = outcome
            .exit_code
            .map_or_else(|| "-".to_string(), |c| c.to_string());
        let summary_str = outcome.summary.as_deref().unwrap_or("-");

        let _ = writeln!(
            md,
            "| `{}` | {} | {:.2}s | {} | {} |",
            outcome.gate,
            verdict_badge,
            outcome.duration_secs,
            exit_str,
            summary_str
        );
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
