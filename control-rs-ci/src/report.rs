//! Report aggregator and Markdown report generator (`ci-report.md`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::GateFilter;
use crate::config::{GateConfig, GatePolicy};
use crate::error::GateResult;
use crate::gate::{GateOutcome, OUTCOME_SCHEMA, RESULT_SUFFIX, Verdict};

/// Lines of each flagged gate's log embedded in the report.
const LOG_TAIL_LINES: usize = 30;

/// Maximum budgeted size for `ci-report.md` (64 KiB, C-1).
pub const MAX_REPORT_BYTES: usize = 64 * 1024;

/// A log tail shorter than this is omitted rather than embedded.
const MIN_TAIL_BYTES: usize = 256;

/// Bytes kept free for the truncation notice.
const NOTICE_RESERVE: usize = 256;

/// Gate outcomes keyed by gate name.
pub type Outcomes = BTreeMap<String, GateOutcome>;

/// Outcome records read from an artifact directory.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadedOutcomes {
    /// Valid records keyed by gate name.
    pub outcomes: Outcomes,
    /// One diagnostic per `*.result.json` that could not be used: unreadable,
    /// malformed or of another schema version.
    pub rejected: Vec<String>,
}

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

    /// Ingests every `*.result.json` in the artifacts directory. A record
    /// that cannot be read, parsed or has another schema version is listed
    /// in `rejected` and counts as missing.
    ///
    /// # Errors
    /// Returns `GateError` if directory scanning fails.
    pub fn load_outcomes(&self) -> GateResult<LoadedOutcomes> {
        let mut loaded = LoadedOutcomes::default();
        if !self.artifacts_dir.exists() {
            return Ok(loaded);
        }

        let mut paths: Vec<PathBuf> = fs::read_dir(&self.artifacts_dir)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        paths.sort();
        for path in paths {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !path.is_file() || !name.ends_with(RESULT_SUFFIX) {
                continue;
            }
            match GateOutcome::load_from_file(&path) {
                Ok(outcome) if outcome.schema == OUTCOME_SCHEMA => {
                    loaded.outcomes.insert(outcome.gate.clone(), outcome);
                }
                Ok(outcome) => loaded.rejected.push(format!(
                    "{name}: schema {} (expected {OUTCOME_SCHEMA})",
                    outcome.schema
                )),
                Err(e) => loaded.rejected.push(format!("{name}: {e}")),
            }
        }
        Ok(loaded)
    }

    /// Evaluates fail-closed gate policy across ingested outcomes.
    /// Without `subset`, every required `fail` gate must have a non-failing
    /// result; with `subset`, only the required `fail` gates in the subset.
    /// A `fail` gate declared `required = false` may have no result, but a
    /// recorded `Fail` still fails the report.
    /// Returns true if all `fail`-policy gates passed.
    #[must_use]
    pub fn is_passing(
        &self,
        config: &GateConfig,
        outcomes: &Outcomes,
        subset: GateFilter<'_>,
    ) -> bool {
        required_gates(config, subset).iter().all(|name| {
            outcomes
                .get(*name)
                .is_some_and(|o| o.verdict != Verdict::Fail)
        }) && fail_gates(config, subset).iter().all(|name| {
            outcomes
                .get(*name)
                .is_none_or(|o| o.verdict != Verdict::Fail)
        })
    }

    /// Generates the complete `ci-report.md` Markdown content, at most
    /// [`MAX_REPORT_BYTES`] long.
    #[must_use]
    pub fn generate_markdown(
        &self,
        config: &GateConfig,
        loaded: &LoadedOutcomes,
        subset: GateFilter<'_>,
    ) -> String {
        let outcomes = &loaded.outcomes;
        let mut md = String::new();

        md.push_str("# Continuous Integration & Verification Report\n\n");
        if self.is_passing(config, outcomes, subset) {
            md.push_str("![Overall Status: Pass](https://img.shields.io/badge/CI_Status-Pass-brightgreen)\n\n");
        } else {
            md.push_str("![Overall Status: Fail](https://img.shields.io/badge/CI_Status-Fail-red)\n\n");
        }

        let ordered_names = ordered_gate_names(config, outcomes);
        let required = required_gates(config, subset);
        let rows: Vec<&str> = ordered_names
            .iter()
            .copied()
            .filter(|name| {
                subset.is_none_or(|active| active.iter().any(|g| g == name))
            })
            .collect();
        push_summary_matrix(&mut md, &rows, outcomes, &required);
        push_rejected(&mut md, &loaded.rejected);
        self.push_diagnostics(&mut md, &ordered_names, outcomes);

        if md.len() > MAX_REPORT_BYTES {
            let cut = floor_char_boundary(
                &md,
                MAX_REPORT_BYTES.saturating_sub(NOTICE_RESERVE),
            );
            md.truncate(cut);
            md.push_str("\n\n*Note: Report truncated to meet the 64 KiB size budget.*\n");
        }
        md
    }

    /// Appends the log tails of failed and warned gates, shortening or
    /// omitting tails so the report stays within [`MAX_REPORT_BYTES`].
    fn push_diagnostics(
        &self,
        md: &mut String,
        ordered_names: &[&str],
        outcomes: &Outcomes,
    ) {
        let flagged: Vec<&GateOutcome> = ordered_names
            .iter()
            .filter_map(|name| outcomes.get(*name))
            .filter(|o| matches!(o.verdict, Verdict::Fail | Verdict::Warn))
            .collect();
        if flagged.is_empty() {
            return;
        }
        md.push_str("\n### Failure & Diagnostic Logs\n\n");
        let budget = MAX_REPORT_BYTES.saturating_sub(NOTICE_RESERVE);
        let mut omitted = Vec::new();
        for outcome in flagged {
            let head = format!(
                "<details>\n<summary><b>Gate: {} ({:?})</b> - {}</summary>\n\n```text\n",
                outcome.gate,
                outcome.verdict,
                escape_cell(outcome.summary.as_deref().unwrap_or("")),
            );
            let foot = "\n```\n</details>\n\n";
            let log_tail = read_trailing_lines(
                &self.artifacts_dir.join(&outcome.log_file),
                LOG_TAIL_LINES,
            );
            let room = budget
                .saturating_sub(md.len())
                .saturating_sub(head.len())
                .saturating_sub(foot.len());
            if room < MIN_TAIL_BYTES.min(log_tail.len()) {
                omitted.push(outcome.gate.as_str());
                continue;
            }
            md.push_str(&head);
            md.push_str(tail_bytes(&log_tail, room));
            md.push_str(foot);
        }
        if !omitted.is_empty() {
            let _ = writeln!(
                md,
                "*Logs omitted for the size budget: {}. See `<gate>.log` in the CI artifacts.*",
                omitted.join(", ")
            );
        }
    }

    /// Renders `ci-report.md` and writes it to the artifacts' directory.
    ///
    /// # Errors
    /// Returns `GateError` if reading the artifacts or writing the report fails.
    pub fn write_report(
        &self,
        config: &GateConfig,
        subset: GateFilter<'_>,
    ) -> GateResult<WrittenReport> {
        let loaded = self.load_outcomes()?;
        let md = self.generate_markdown(config, &loaded, subset);
        let pass = self.is_passing(config, &loaded.outcomes, subset);

        let path = self.artifacts_dir.join("ci-report.md");
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &md)?;

        Ok(WrittenReport { pass, path })
    }
}

/// Every `fail` gate, or with `subset` the `fail` gates in it.
fn fail_gates<'a>(
    config: &'a GateConfig,
    subset: GateFilter<'a>,
) -> Vec<&'a str> {
    let is_fail = |name: &&str| config.policy_for(name) == GatePolicy::Fail;
    subset.map_or_else(
        || {
            config
                .gate_definitions
                .keys()
                .map(String::as_str)
                .filter(is_fail)
                .collect()
        },
        |names| names.iter().map(String::as_str).filter(is_fail).collect(),
    )
}

/// Gates that must have a non-failing result: every `fail` gate not declared
/// `required = false`, or with `subset` those in it (FR-10, FR-15).
fn required_gates<'a>(
    config: &'a GateConfig,
    subset: GateFilter<'a>,
) -> Vec<&'a str> {
    fail_gates(config, subset)
        .into_iter()
        .filter(|name| config.gate_def(name).is_none_or(|g| g.required))
        .collect()
}

/// Gate names in report order: [`GateConfig::pipeline_order`], then any other
/// outcome found on disk.
fn ordered_gate_names<'a>(
    config: &'a GateConfig,
    outcomes: &'a Outcomes,
) -> Vec<&'a str> {
    let mut ordered = config.pipeline_order();
    for name in outcomes.keys() {
        if !ordered.contains(&name.as_str()) {
            ordered.push(name);
        }
    }
    ordered
}

/// Appends the executive summary table over `rows`. A required gate without
/// a result gets a `**MISSING**` row.
fn push_summary_matrix(
    md: &mut String,
    rows: &[&str],
    outcomes: &Outcomes,
    required: &[&str],
) {
    md.push_str("### Executive Summary Matrix\n\n");
    md.push_str("| Gate | Verdict | Duration | Exit Code | Summary |\n");
    md.push_str("|:---|:---|:---|:---|:---|\n");

    for &name in rows {
        let Some(outcome) = outcomes.get(name) else {
            if required.contains(&name) {
                let _ = writeln!(
                    md,
                    "| `{name}` | **MISSING** | - | - | no result recorded |"
                );
            }
            continue;
        };

        let verdict_badge = match outcome.verdict {
            Verdict::Pass => "**Pass**",
            Verdict::Warn => "*Warn*",
            Verdict::Fail => "**FAIL**",
            Verdict::Skipped => "Skipped",
        };
        let exit_str = outcome
            .exit_code
            .map_or_else(|| "-".to_string(), |c| c.to_string());
        let summary_str =
            escape_cell(outcome.summary.as_deref().unwrap_or("-"));

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

/// Appends the list of result records that could not be used.
fn push_rejected(md: &mut String, rejected: &[String]) {
    if rejected.is_empty() {
        return;
    }
    md.push_str("\n### Rejected Result Records\n\n");
    for line in rejected {
        let _ = writeln!(md, "- {}", escape_cell(line));
    }
}

/// Makes `text` safe inside one Markdown table cell.
fn escape_cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\n', '\r'], " ")
}

/// Largest char boundary of `text` at or below `index`.
fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut cut = index.min(text.len());
    while !text.is_char_boundary(cut) {
        cut = cut.saturating_sub(1);
    }
    cut
}

/// The last `max_bytes` bytes of `text`, starting on a char boundary.
fn tail_bytes(text: &str, max_bytes: usize) -> &str {
    let mut start = text.len().saturating_sub(max_bytes);
    while !text.is_char_boundary(start) {
        start = start.saturating_add(1);
    }
    text.get(start..).unwrap_or_default()
}

fn read_trailing_lines(path: &Path, max_lines: usize) -> String {
    let Ok(file) = File::open(path) else {
        return "(Log file unavailable)".to_string();
    };
    let mut lines = std::collections::VecDeque::with_capacity(max_lines);
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if lines.len() == max_lines {
            lines.pop_front();
        }
        lines.push_back(line);
    }
    lines.into_iter().collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_land_on_char_boundaries() {
        let text = "aµb";
        assert_eq!(floor_char_boundary(text, 2), 1);
        assert_eq!(tail_bytes(text, 2), "b");
        assert_eq!(tail_bytes(text, 3), "µb");
        assert_eq!(tail_bytes(text, 99), "aµb");
    }

    #[test]
    fn cells_escape_pipes_and_newlines() {
        assert_eq!(escape_cell("a|b\nc"), "a\\|b c");
    }
}
