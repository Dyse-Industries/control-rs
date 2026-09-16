//! Utility functions and report generation structures for CI quality gates.

use std::env;
use std::fmt::{self, Write as _};
use std::fs;
use std::process::Command;
use std::time::Duration;

use anstream::ColorChoice;
use anstyle::AnsiColor;
use control_rs_ets::comms::TestState;
use control_rs_ets_host::runner::TestOutcome;

use crate::gates::TarpaulinSummary;

const CARGO_STATUS_WIDTH: usize = 12;

/// Cargo `ERROR`: bright red bold diagnostic prefix.
pub const ERROR: anstyle::Style = AnsiColor::BrightRed.on_default().bold();
/// Cargo `HEADER`: bright green bold status verb.
pub const HEADER: anstyle::Style = AnsiColor::BrightGreen.on_default().bold();
/// Help arg style.
pub const HELP_ARG: anstyle::Style = AnsiColor::BrightYellow.on_default();
/// Help flag style.
pub const HELP_FLAG: anstyle::Style = AnsiColor::Cyan.on_default();
/// Help header style.
pub const HELP_HEADER: anstyle::Style =
    AnsiColor::BrightGreen.on_default().bold();
/// Cargo-style warning verb (bright yellow bold).
pub const WARNING: anstyle::Style = AnsiColor::BrightYellow.on_default().bold();

type EtsTableResult = Result<Vec<TestOutcome>, String>;
type EtsTriple = (usize, usize, usize);
type TargetEtsRef<'a> = (&'a str, EtsTableResult);

/// Bitmask of CI report sections omitted from the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CiSkip(u8);

/// Options controlling CI validation execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CiOptions {
    /// Apply formatting instead of checking (`cargo fmt --all`).
    pub fmt: bool,
    /// Sections skipped in this run.
    pub skip: CiSkip,
}

/// Parameters provided to [`build_report`] to construct `ci-report.md`.
pub struct CiReportParams<'a> {
    /// Report title (e.g. "control-rs" or example name).
    pub title: &'a str,
    /// CI options passed to the pipeline.
    pub options: CiOptions,
    /// Formatting gate verdict.
    pub fmt: GateVerdict,
    /// Raw output of formatting command.
    pub fmt_output: &'a str,
    /// Time in seconds taken by formatting check.
    pub fmt_time: f32,
    /// Clippy gate verdict.
    pub clippy: GateVerdict,
    /// Raw output of `cargo clippy`.
    pub clippy_output: &'a str,
    /// Time in seconds taken by clippy.
    pub clippy_time: f32,
    /// Check gate verdict.
    pub check: GateVerdict,
    /// Raw output of `cargo check`.
    pub check_output: &'a str,
    /// Time in seconds taken by `cargo check`.
    pub check_time: f32,
    /// Build gate verdict.
    pub build: GateVerdict,
    /// Raw output of `cargo build`.
    pub build_output: &'a str,
    /// Time in seconds taken by `cargo build`.
    pub build_time: f32,
    /// Clean gate verdict.
    pub clean: GateVerdict,
    /// Raw output of `cargo clean`.
    pub clean_output: &'a str,
    /// Time in seconds taken by `cargo clean`.
    pub clean_time: f32,
    /// Unit-test gate verdict.
    pub test_cmd: GateVerdict,
    /// Raw output of `cargo test`.
    pub test_cmd_output: &'a str,
    /// Time in seconds taken by `cargo test`.
    pub test_cmd_time: f32,
    /// Coverage gate verdict.
    pub coverage: GateVerdict,
    /// Summary of tarpaulin coverage and test counts.
    pub tarp_summary: &'a TarpaulinSummary,
    /// Raw output of tarpaulin.
    pub tarp_output: &'a str,
    /// Time in seconds taken by tests and coverage.
    pub test_time: f32,
    /// Off-the-shelf host tools (mutants, miri, valgrind, fuzz, lockbud, deny, audit, kani).
    pub host_tools: &'a [HostToolRow<'a>],
    /// Canonical gate names that failed under `warn` mode.
    pub warned_gates: &'a [&'a str],
    /// Standard gates omitted from `[gates]` or listed as `skip`.
    pub skipped_standard: &'a [&'a str],
    /// List of ETS results per target: `(target_name, result)`.
    pub target_ets_results: &'a [TargetEtsRef<'a>],
    /// Total time taken by virtual ETS runs.
    pub ets_time: f32,
    /// Traceability gate verdict.
    pub trace: GateVerdict,
    /// Traceability matrix summary, if run.
    pub trace_summary: Option<&'a crate::trace::TraceMatrixSummary>,
    /// Time taken by traceability analysis.
    pub trace_time: f32,
    /// Cross-comparison gate verdict.
    pub cross_val: GateVerdict,
    /// Cross-comparison summary, if run.
    pub cross_val_summary: Option<&'a crate::validate::CrossComparisonSummary>,
    /// Time taken by cross-comparison runs.
    pub cross_val_time: f32,
}

/// Pipeline / host-tool row verdict for `ci-report.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GateVerdict {
    /// Gate succeeded.
    #[default]
    Pass,
    /// Gate failed the pipeline.
    Fail,
    /// Gate was not run.
    Skip,
    /// Gate failed under `warn` mode.
    Warn,
}

/// One off-the-shelf host verification tool row in `ci-report.md`.
#[derive(Debug, Clone, Copy)]
pub struct HostToolRow<'a> {
    /// Display name (e.g. `Mutants`).
    pub name: &'a str,
    /// Command shown in the details column.
    pub command: &'a str,
    /// Pass / fail / skip / warn.
    pub verdict: GateVerdict,
    /// One-line verdict.
    pub details: &'a str,
    /// Raw tool output.
    pub output: &'a str,
    /// Runtime in seconds.
    pub time: f32,
}

struct EtsCounts {
    total: usize,
    passed: usize,
    failed: usize,
}

struct TaskRow<'a> {
    name: &'a str,
    status: &'a str,
    details: &'a str,
    time: f32,
}

impl CiSkip {
    /// Skip coverage / tarpaulin.
    pub const COV: Self = Self(1 << 0);
    /// Skip requirement traceability.
    pub const TRACE: Self = Self(1 << 1);
    /// Skip example cross-comparison.
    pub const EXAMPLES: Self = Self(1 << 2);
    /// Skip ETS target matrix.
    pub const ETS: Self = Self(1 << 3);

    /// True when `flag` is set.
    #[must_use]
    pub const fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 != 0
    }

    /// Set `flag` on this mask.
    pub const fn set(&mut self, flag: Self) {
        self.0 |= flag.0;
    }

    /// Combined mask with `flag` set.
    #[must_use]
    pub const fn union(self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }
}

impl GateVerdict {
    /// Markdown status cell for the summary table.
    #[must_use]
    pub const fn as_cell(self) -> &'static str {
        match self {
            Self::Pass => "Pass",
            Self::Fail => "**FAIL**",
            Self::Skip => "Skipped",
            Self::Warn => "Warn",
        }
    }

    /// True when the gate failed the pipeline.
    #[must_use]
    pub const fn is_fail(self) -> bool {
        matches!(self, Self::Fail)
    }
}

/// Apply cargo-compatible color choice to anstream's global default.
///
/// `CARGO_TERM_COLOR` is cargo-specific and is not read by `anstream` itself.
/// `always` / `never` override; any other value (including unset) is `auto`,
/// which honors `NO_COLOR`, `CLICOLOR`, `TERM=dumb`, and TTY detection.
pub fn init_color() {
    cargo_color_choice().write_global();
}

fn cargo_color_choice() -> ColorChoice {
    match env::var("CARGO_TERM_COLOR").ok().as_deref() {
        Some("always") => ColorChoice::Always,
        Some("never") => ColorChoice::Never,
        _ => ColorChoice::Auto,
    }
}

/// Prints a cargo-style status line to stderr (green bold verb when color is on).
pub fn status(status: &str, msg: impl fmt::Display) {
    anstream::eprintln!(
        "{HEADER}{status:>CARGO_STATUS_WIDTH$}{HEADER:#} {msg}"
    );
}

/// Prints a cargo-style warning status line (yellow bold verb when color is on).
pub fn status_warn(status: &str, msg: impl fmt::Display) {
    anstream::eprintln!(
        "{WARNING}{status:>CARGO_STATUS_WIDTH$}{WARNING:#} {msg}"
    );
}

/// Prints a cargo-style error diagnostic to stderr (red bold `error` when color is on).
pub fn error(msg: impl fmt::Display) {
    anstream::eprintln!("{ERROR}error{ERROR:#}: {msg}");
}

/// Formats an integer with comma thousands separators (e.g. `1,234,567`).
#[must_use]
pub fn format_number(val: u64) -> String {
    let s = val.to_string();
    let bytes = s.as_bytes();
    let mut result = String::new();
    let len = bytes.len();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && len.saturating_sub(i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

/// Formats an elapsed duration in cargo style (e.g. `0.09s`, `3.12s`, `1m 23s`).
#[must_use]
pub fn format_elapsed(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs >= 3600 {
        format!("{}h {}m {:02}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    } else if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{:.2}s", duration.as_secs_f64())
    }
}

/// Collects the total size of the Rust codebase (file count and lines of code).
#[must_use]
pub fn get_codebase_size() -> String {
    let output = Command::new("git").args(["ls-files", "*.rs"]).output().ok();
    if let Some(out) = output
        && out.status.success()
    {
        let files = String::from_utf8_lossy(&out.stdout);
        let mut total_lines = 0usize;
        let mut file_count = 0usize;
        for line in files.lines() {
            let path = line.trim();
            if !path.is_empty() {
                file_count = file_count.saturating_add(1);
                if let Ok(content) = fs::read_to_string(path) {
                    total_lines =
                        total_lines.saturating_add(content.lines().count());
                }
            }
        }
        format!(
            "{} Rust files ({} lines)",
            format_number(file_count as u64),
            format_number(total_lines as u64)
        )
    } else {
        "Unknown".to_string()
    }
}

/// Executes a single CLI command and returns its trimmed stdout/stderr.
#[must_use]
pub fn get_cmd_output(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd).args(args).output().map_or_else(
        |e| format!("Failed to run {cmd}: {e}"),
        |o| {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let stderr = String::from_utf8_lossy(&o.stderr);
            format!("{stdout}{stderr}").trim().to_string()
        },
    )
}

fn ets_metric_cell(value: Option<u64>, suffix: &str) -> String {
    value.map_or_else(
        || "N/A".to_string(),
        |v| format!("{}{suffix}", format_number(v)),
    )
}

fn append_ets_test_rows(s: &mut String, tests: &[TestOutcome]) {
    for t in tests {
        let status_str = match t.state {
            TestState::Passed => "Pass",
            TestState::Failed => "**FAIL**",
            TestState::Running => "Running",
            TestState::Pending => "Pending",
        };
        let cycles_str = ets_metric_cell(t.cycles, "c");
        let time_str = ets_metric_cell(t.time_us, "µs");
        let stack_str = t.stack_peak.map_or_else(
            || "N/A".to_string(),
            |sp| format!("{}B", format_number(u64::from(sp))),
        );
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} |",
            t.suite_name,
            t.test_name,
            status_str,
            cycles_str,
            time_str,
            stack_str
        );
    }
}

/// Formats a single target's ETS results as a dedicated Markdown table wrapped in a collapsible block.
#[must_use]
pub fn format_target_ets_table(
    target_name: &str,
    results: &EtsTableResult,
) -> String {
    let mut s = format!("#### Target: `{target_name}`\n\n");
    match results {
        Ok(tests) if tests.is_empty() => {
            s.push_str("No ETS tests executed.\n\n");
        }
        Ok(tests) => {
            let total = tests.len();
            let passed = tests
                .iter()
                .filter(|t| matches!(t.state, TestState::Passed))
                .count();
            let failed = total.saturating_sub(passed);
            let tally_text = if failed == 0 {
                format!("{passed}/{total} passed")
            } else {
                format!("{failed} failed, {passed}/{total} passed")
            };
            s.push_str("<details>\n");
            let _ =
                write!(s, "<summary>Test Results ({tally_text})</summary>\n\n");
            s.push_str(
                "| Suite | Test | Status | Cycles | Duration | Stack Peak |\n",
            );
            s.push_str("| :--- | :--- | :--- | :--- | :--- | :--- |\n");
            append_ets_test_rows(&mut s, tests);
            s.push_str("\n</details>\n\n");
        }
        Err(e) => {
            let _ = writeln!(s, "**ERROR**: {e}");
            s.push('\n');
        }
    }
    s
}

fn cell(
    verdict: GateVerdict,
    name: &str,
    params: &CiReportParams<'_>,
) -> &'static str {
    if params.skipped_standard.contains(&name) {
        "Skipped"
    } else if verdict == GateVerdict::Pass {
        "Pass"
    } else if params.warned_gates.contains(&name)
        || verdict == GateVerdict::Warn
    {
        "Warn"
    } else if verdict == GateVerdict::Skip {
        "Skipped"
    } else {
        "**FAIL**"
    }
}

fn warned(params: &CiReportParams<'_>, name: &str) -> bool {
    params.warned_gates.contains(&name)
}

fn collect_standard_failures(params: &CiReportParams<'_>) -> Vec<&'static str> {
    let mut failed_tasks = Vec::new();
    if params.clean.is_fail() && !warned(params, "clean") {
        failed_tasks.push("clean");
    }
    if params.fmt.is_fail() && !warned(params, "fmt") {
        failed_tasks.push("formatting");
    }
    if params.clippy.is_fail() && !warned(params, "clippy") {
        failed_tasks.push("clippy");
    }
    if params.check.is_fail() && !warned(params, "check") {
        failed_tasks.push("check");
    }
    if params.build.is_fail() && !warned(params, "build") {
        failed_tasks.push("build");
    }
    if params.test_cmd.is_fail() && !warned(params, "test") {
        failed_tasks.push("test");
    }
    failed_tasks
}

fn collect_failed_tasks<'a>(
    params: &'a CiReportParams<'a>,
    ets_total: usize,
    ets_failed: usize,
) -> Vec<&'a str> {
    let mut failed_tasks = collect_standard_failures(params);
    if !params.options.skip.contains(CiSkip::COV)
        && (params.coverage != GateVerdict::Pass
            || params.tarp_summary.failed > 0)
        && !warned(params, "coverage")
    {
        failed_tasks.push("coverage");
    }
    for tool in params.host_tools {
        if tool.verdict.is_fail() {
            failed_tasks.push(tool.name);
        }
    }
    if !params.options.skip.contains(CiSkip::TRACE)
        && params.trace.is_fail()
        && !warned(params, "trace")
    {
        failed_tasks.push("traceability");
    }
    if !params.options.skip.contains(CiSkip::EXAMPLES)
        && params.cross_val.is_fail()
        && !warned(params, "validate")
    {
        failed_tasks.push("cross-validation");
    }
    if ets_failed > 0 && !warned(params, "ets") {
        failed_tasks.push("virtual-ets");
    }
    if !params.options.skip.contains(CiSkip::ETS)
        && ets_total == 0
        && ets_failed == 0
        && !warned(params, "ets")
    {
        failed_tasks.push("virtual-ets");
    }
    failed_tasks
}

fn count_ets(results: &[TargetEtsRef<'_>]) -> EtsTriple {
    let mut ets_total = 0usize;
    let mut ets_passed = 0usize;
    let mut ets_failed = 0usize;
    for (_, res) in results {
        match res {
            Ok(tests) => {
                for t in tests {
                    ets_total = ets_total.saturating_add(1);
                    if matches!(t.state, TestState::Passed) {
                        ets_passed = ets_passed.saturating_add(1);
                    } else {
                        ets_failed = ets_failed.saturating_add(1);
                    }
                }
            }
            Err(_) => {
                ets_failed = ets_failed.saturating_add(1);
            }
        }
    }
    (ets_total, ets_passed, ets_failed)
}

fn total_duration(params: &CiReportParams<'_>) -> f32 {
    params.clean_time
        + params.fmt_time
        + params.clippy_time
        + params.check_time
        + params.build_time
        + params.test_cmd_time
        + params.test_time
        + params.host_tools.iter().map(|t| t.time).sum::<f32>()
        + params.ets_time
        + params.trace_time
        + params.cross_val_time
}

fn push_task_row(report: &mut String, row: &TaskRow<'_>) {
    let _ = writeln!(
        report,
        "| **{}** | {} | {} | {:.2}s |",
        row.name, row.status, row.details, row.time
    );
}

fn append_standard_rows(report: &mut String, params: &CiReportParams<'_>) {
    push_task_row(
        report,
        &TaskRow {
            name: "Clean",
            status: cell(params.clean, "clean", params),
            details: "`cargo clean`",
            time: params.clean_time,
        },
    );
    let fmt_cmd = if params.options.fmt {
        "`cargo fmt --all`"
    } else {
        "`cargo fmt --all -- --check`"
    };
    push_task_row(
        report,
        &TaskRow {
            name: "Formatting",
            status: cell(params.fmt, "fmt", params),
            details: fmt_cmd,
            time: params.fmt_time,
        },
    );
    push_task_row(
        report,
        &TaskRow {
            name: "Clippy Lints",
            status: cell(params.clippy, "clippy", params),
            details: "`cargo clippy --workspace --all-targets --all-features -- -D warnings`",
            time: params.clippy_time,
        },
    );
    push_task_row(
        report,
        &TaskRow {
            name: "Check",
            status: cell(params.check, "check", params),
            details: "`cargo check --workspace`",
            time: params.check_time,
        },
    );
    push_task_row(
        report,
        &TaskRow {
            name: "Build",
            status: cell(params.build, "build", params),
            details: "`cargo build --workspace`",
            time: params.build_time,
        },
    );
    push_task_row(
        report,
        &TaskRow {
            name: "Unit Tests",
            status: cell(params.test_cmd, "test", params),
            details: "`cargo test --workspace`",
            time: params.test_cmd_time,
        },
    );
}

fn append_coverage_row(report: &mut String, params: &CiReportParams<'_>) {
    if params.options.skip.contains(CiSkip::COV) {
        let _ = writeln!(
            report,
            "| **Coverage** | Skipped | skipped (not in pipeline) | {:.2}s |",
            params.test_time
        );
        return;
    }
    let cov_ok =
        params.coverage == GateVerdict::Pass && params.tarp_summary.failed == 0;
    let _ = writeln!(
        report,
        "| **Coverage** | {} | {} passed, {} failed, {} ignored (`{}%` coverage, {}/{} lines) | {:.2}s |",
        cell(
            if cov_ok {
                GateVerdict::Pass
            } else {
                params.coverage
            },
            "coverage",
            params,
        ),
        params.tarp_summary.passed,
        params.tarp_summary.failed,
        params.tarp_summary.ignored,
        params.tarp_summary.coverage_percent,
        format_number(params.tarp_summary.covered_lines as u64),
        format_number(params.tarp_summary.total_lines as u64),
        params.test_time
    );
}

fn append_host_tool_rows(report: &mut String, params: &CiReportParams<'_>) {
    for tool in params.host_tools {
        if tool.verdict == GateVerdict::Skip {
            let _ = writeln!(
                report,
                "| **{}** | Skipped | {} | {:.2}s |",
                tool.name, tool.details, tool.time
            );
        } else {
            let details = if tool.details.is_empty() {
                tool.command
            } else {
                tool.details
            };
            let _ = writeln!(
                report,
                "| **{}** | {} | {} | {:.2}s |",
                tool.name,
                tool.verdict.as_cell(),
                details,
                tool.time
            );
        }
    }
}

fn append_optional_rows(
    report: &mut String,
    params: &CiReportParams<'_>,
    ets: &EtsCounts,
) {
    let ets_status = if params.options.skip.contains(CiSkip::ETS) {
        "Skipped"
    } else if ets.total == 0 {
        cell(GateVerdict::Fail, "ets", params)
    } else {
        cell(
            if ets.failed == 0 {
                GateVerdict::Pass
            } else {
                GateVerdict::Fail
            },
            "ets",
            params,
        )
    };
    let _ = writeln!(
        report,
        "| **Virtual ETS** | {ets_status} | {}/{} tests passed across {} targets | {:.2}s |",
        ets.passed,
        ets.total,
        params.target_ets_results.len(),
        params.ets_time
    );
    if let Some(trace) = params.trace_summary {
        let _ = writeln!(
            report,
            "| **Traceability** | {} | {}/{} requirements traced ({} findings) | {:.2}s |",
            cell(params.trace, "trace", params),
            trace.verified_count,
            trace.total_requirements,
            trace.defects.len(),
            params.trace_time
        );
    }
    if let Some(cv) = params.cross_val_summary {
        let _ = writeln!(
            report,
            "| **Cross-Validation** | {} | {}/{} examples passed | {:.2}s |",
            cell(params.cross_val, "validate", params),
            cv.passed_examples,
            cv.total_examples,
            params.cross_val_time
        );
    }
}

fn append_briefs(report: &mut String, params: &CiReportParams<'_>) {
    if let Some(cv) = params.cross_val_summary {
        report.push_str(&crate::validate::render_cross_comparison_brief(cv));
    }
    if let Some(trace) = params.trace_summary {
        report.push_str(&crate::trace::render_trace_brief(trace));
    }
}

fn truncate_middle(
    s: &str,
    head_chars: usize,
    tail_chars: usize,
    marker: &str,
) -> String {
    if s.len() <= 16_000 {
        return s.to_string();
    }
    let head_idx = s
        .char_indices()
        .nth(head_chars)
        .map_or(head_chars, |(i, _)| i);
    let tail_idx = s.char_indices().rev().nth(tail_chars).map_or(0, |(i, _)| i);
    let head = s.get(..head_idx).unwrap_or(s);
    let tail = s.get(tail_idx..).unwrap_or("");
    format!("{head}\n\n{marker}\n\n{tail}")
}

fn append_env_logs(report: &mut String) {
    report.push_str(
        "<details>\n<summary>Git & Toolchain Information</summary>\n\n```text\n",
    );
    let git_log = get_cmd_output(
        "git",
        &[
            "log",
            "-1",
            "--format=Commit:  %H%nAuthor:  %an <%ae>%nDate:    %ad%nSubject: %s",
        ],
    );
    let git_status = get_cmd_output("git", &["status", "-s"]);
    let rustc_ver = get_cmd_output("rustc", &["--version"]);
    let cargo_ver = get_cmd_output("cargo", &["--version"]);
    let _ = writeln!(
        report,
        "{git_log}\n\nDirty files:\n{git_status}\n\nToolchains:\n{rustc_ver}\n{cargo_ver}"
    );
    report.push_str("```\n\n</details>\n\n");
    let dep_tree = get_cmd_output("cargo", &["tree", "--workspace"]);
    let dep_tree_trimmed = truncate_middle(
        &dep_tree,
        8000,
        4000,
        "[... dependency tree truncated for length limits ...]",
    );
    report.push_str(
        "<details>\n<summary>Dependency Tree (cargo tree)</summary>\n\n```text\n",
    );
    report.push_str(&dep_tree_trimmed);
    report.push_str("\n```\n\n</details>\n\n");
}

fn append_fmt_clippy_logs(report: &mut String, params: &CiReportParams<'_>) {
    report.push_str(
        "<details>\n<summary>Raw Output: Formatting & Clippy</summary>\n\n```text\n",
    );
    if !params.fmt_output.is_empty() {
        let header = if params.options.fmt {
            "--- cargo fmt --all ---\n"
        } else {
            "--- cargo fmt --check ---\n"
        };
        report.push_str(header);
        report.push_str(params.fmt_output);
        report.push('\n');
    }
    if !params.clippy_output.is_empty() {
        report.push_str("--- cargo clippy ---\n");
        report.push_str(params.clippy_output);
        report.push('\n');
    }
    report.push_str("```\n\n</details>\n\n");
}

fn append_build_test_logs(report: &mut String, params: &CiReportParams<'_>) {
    report.push_str(
        "<details>\n<summary>Raw Output: Clean, Check, Build & Test</summary>\n\n```text\n",
    );
    if !params.clean_output.is_empty() {
        report.push_str("--- cargo clean ---\n");
        report.push_str(params.clean_output);
        report.push('\n');
    }
    if !params.check_output.is_empty() {
        report.push_str("--- cargo check ---\n");
        report.push_str(params.check_output);
        report.push('\n');
    }
    if !params.build_output.is_empty() {
        report.push_str("--- cargo build ---\n");
        report.push_str(params.build_output);
        report.push('\n');
    }
    if !params.test_cmd_output.is_empty() {
        report.push_str("--- cargo test ---\n");
        let trimmed = truncate_middle(
            params.test_cmd_output,
            4000,
            8000,
            "[... test output truncated for length limits ...]",
        );
        report.push_str(&trimmed);
        report.push('\n');
    }
    report.push_str("```\n\n</details>\n\n");
}

fn append_tarpaulin_logs(report: &mut String, params: &CiReportParams<'_>) {
    report.push_str(
        "<details>\n<summary>Raw Output: Tests & Tarpaulin</summary>\n\n```text\n",
    );
    if params.options.skip.contains(CiSkip::COV) {
        report.push_str("Tarpaulin test & coverage was skipped.\n");
    } else {
        let trimmed = truncate_middle(
            params.tarp_output,
            4000,
            8000,
            "[... tarpaulin output truncated for length limits ...]",
        );
        report.push_str(&trimmed);
    }
    report.push_str("\n```\n\n</details>\n");
}

fn append_host_tool_logs(report: &mut String, params: &CiReportParams<'_>) {
    if !params
        .host_tools
        .iter()
        .any(|t| !t.output.is_empty() || t.verdict == GateVerdict::Skip)
    {
        return;
    }
    report.push_str(
        "\n<details>\n<summary>Raw Output: Host Verification Tools</summary>\n\n```text\n",
    );
    for tool in params.host_tools {
        let _ = writeln!(report, "--- {} ---", tool.name);
        if tool.verdict == GateVerdict::Skip {
            report.push_str(tool.details);
            report.push('\n');
        } else {
            let trimmed = truncate_middle(
                tool.output,
                4000,
                8000,
                "[... output truncated for length limits ...]",
            );
            report.push_str(&trimmed);
        }
        report.push('\n');
    }
    report.push_str("```\n\n</details>\n");
}

/// Assembles the complete Markdown report with high-density briefs, stats, and collapsible raw logs.
#[must_use]
pub fn build_report(params: &CiReportParams<'_>) -> String {
    let title = if params.title.is_empty() {
        "control-rs"
    } else {
        params.title
    };
    let mut report = format!("## `{title}` Quality Report\n\n");
    let (ets_total, ets_passed, ets_failed) =
        count_ets(params.target_ets_results);
    let failed_tasks = collect_failed_tasks(params, ets_total, ets_failed);
    let failure_summary = if failed_tasks.is_empty() {
        "0 failures (clean run)".to_string()
    } else {
        format!(
            "{} failed ({})",
            failed_tasks.len(),
            failed_tasks.join(", ")
        )
    };
    let codebase_size = get_codebase_size();
    let duration = total_duration(params);
    report.push_str("### Summary & Stats\n\n");
    report.push_str("| Metric | Value |\n| :--- | :--- |\n");
    let _ = writeln!(report, "| **Codebase Size** | {codebase_size} |");
    let _ = writeln!(report, "| **Failures** | {failure_summary} |");
    let _ = writeln!(report, "| **Total Duration** | {duration:.2}s |");
    report.push('\n');
    report.push_str("| Task | Status | Details | Duration |\n| :--- | :--- | :--- | :--- |\n");
    append_standard_rows(&mut report, params);
    append_coverage_row(&mut report, params);
    append_host_tool_rows(&mut report, params);
    append_optional_rows(
        &mut report,
        params,
        &EtsCounts {
            total: ets_total,
            passed: ets_passed,
            failed: ets_failed,
        },
    );
    report.push('\n');
    append_briefs(&mut report, params);
    if !params.target_ets_results.is_empty() {
        report.push_str("### Virtual ETS Results\n\n");
        for (target_name, res) in params.target_ets_results {
            report.push_str(&format_target_ets_table(target_name, res));
        }
    }
    report.push_str("### Detailed Logs & Environment\n\n");
    append_env_logs(&mut report);
    append_fmt_clippy_logs(&mut report, params);
    append_build_test_logs(&mut report, params);
    append_tarpaulin_logs(&mut report, params);
    append_host_tool_logs(&mut report, params);
    report
}

/// Saves the report string to a file.
///
/// # Errors
///
/// Returns an I/O error when `path` cannot be written.
pub fn save_report(path: &str, content: &str) -> std::io::Result<()> {
    fs::write(path, content)
}

#[cfg(test)]
fn format_status(status: &str, msg: impl fmt::Display) -> String {
    format!("{status:>CARGO_STATUS_WIDTH$} {msg}")
}

#[cfg(test)]
fn format_error(msg: impl fmt::Display) -> String {
    format!("error: {msg}")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PassingSeed<'a> {
        title: &'a str,
        options: CiOptions,
        tarp: &'a TarpaulinSummary,
        tarp_output: &'a str,
        fmt_output: &'a str,
        target_ets: &'a [TargetEtsRef<'a>],
        fmt_time: f32,
        test_time: f32,
        ets_time: f32,
    }

    fn passing_params<'a>(seed: &'a PassingSeed<'a>) -> CiReportParams<'a> {
        CiReportParams {
            title: seed.title,
            options: seed.options,
            fmt: GateVerdict::Pass,
            fmt_output: seed.fmt_output,
            fmt_time: seed.fmt_time,
            clippy: GateVerdict::Pass,
            clippy_output: "",
            clippy_time: 0.5,
            check: GateVerdict::Pass,
            check_output: "",
            check_time: 0.2,
            build: GateVerdict::Pass,
            build_output: "",
            build_time: 0.3,
            clean: GateVerdict::Pass,
            clean_output: "",
            clean_time: 0.05,
            test_cmd: GateVerdict::Pass,
            test_cmd_output: "",
            test_cmd_time: 0.4,
            coverage: GateVerdict::Pass,
            tarp_summary: seed.tarp,
            tarp_output: seed.tarp_output,
            test_time: seed.test_time,
            host_tools: &[],
            warned_gates: &[],
            skipped_standard: &[],
            target_ets_results: seed.target_ets,
            ets_time: seed.ets_time,
            trace: GateVerdict::Pass,
            trace_summary: None,
            trace_time: 0.0,
            cross_val: GateVerdict::Pass,
            cross_val_summary: None,
            cross_val_time: 0.0,
        }
    }

    fn assert_report_headings(report: &str) {
        assert!(report.contains("## `control-rs` Quality Report"));
        assert!(report.contains("### Summary & Stats"));
        assert!(report.contains("| **Codebase Size** |"));
        assert!(report.contains("| **Failures** | 0 failures (clean run) |"));
        assert!(report.contains("`cargo fmt --all -- --check`"));
        assert!(report.contains(
            "`cargo clippy --workspace --all-targets --all-features -- -D warnings`"
        ));
        assert!(report.contains("`cargo check --workspace`"));
        assert!(report.contains("`cargo build --workspace`"));
        assert!(report.contains("`cargo clean`"));
        assert!(report.contains("`cargo test --workspace`"));
        assert!(report.contains("### Virtual ETS Results"));
        assert!(report.contains("#### Target: `ARM HF`"));
        assert!(report.contains(
            "<details>\n<summary>Git & Toolchain Information</summary>"
        ));
        assert!(report.contains(
            "<details>\n<summary>Dependency Tree (cargo tree)</summary>"
        ));
    }

    #[test]
    fn cargo_status_is_12_column_and_error_is_not_justified() {
        assert_eq!(
            format_status("Checking", "formatting"),
            "    Checking formatting"
        );
        assert_eq!(
            format_status("Running", "tarpaulin"),
            "     Running tarpaulin"
        );
        assert_eq!(
            format_error("CI pipeline failed"),
            "error: CI pipeline failed"
        );
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1_234_567), "1,234,567");
    }

    #[test]
    fn test_format_elapsed() {
        assert_eq!(format_elapsed(Duration::from_millis(90)), "0.09s");
        assert_eq!(format_elapsed(Duration::from_millis(3120)), "3.12s");
        assert_eq!(format_elapsed(Duration::from_millis(13610)), "13.61s");
        assert_eq!(format_elapsed(Duration::from_secs(65)), "1m 05s");
        assert_eq!(format_elapsed(Duration::from_secs(3665)), "1h 1m 05s");
    }

    #[test]
    fn test_format_target_ets_table() {
        let results = Ok(vec![
            TestOutcome {
                suite_name: "SuiteA".to_string(),
                test_name: "Test1".to_string(),
                state: TestState::Passed,
                cycles: Some(100),
                time_us: Some(10),
                stack_peak: Some(256),
            },
            TestOutcome {
                suite_name: "SuiteA".to_string(),
                test_name: "Test2".to_string(),
                state: TestState::Failed,
                cycles: None,
                time_us: None,
                stack_peak: None,
            },
        ]);
        let table = format_target_ets_table("ARM HF", &results);
        assert!(table.contains("#### Target: `ARM HF`"));
        assert!(table.contains("<details>"));
        assert!(table.contains(
            "<summary>Test Results (1 failed, 1/2 passed)</summary>"
        ));
        assert!(table.contains("</details>"));
        assert!(
            table.contains("| SuiteA | Test1 | Pass | 100c | 10µs | 256B |")
        );
        assert!(
            table.contains("| SuiteA | Test2 | **FAIL** | N/A | N/A | N/A |")
        );
    }

    #[test]
    fn test_build_report_structure() {
        let tarp = TarpaulinSummary {
            passed: 10,
            failed: 0,
            ignored: 1,
            coverage_percent: 92.5,
            covered_lines: 925,
            total_lines: 1000,
        };
        let ets_res = Ok(vec![TestOutcome {
            suite_name: "Suite1".to_string(),
            test_name: "TestA".to_string(),
            state: TestState::Passed,
            cycles: Some(50),
            time_us: Some(5),
            stack_peak: Some(128),
        }]);
        let target_ets = [("ARM HF", ets_res)];
        let seed = PassingSeed {
            title: "control-rs",
            options: CiOptions::default(),
            tarp: &tarp,
            tarp_output: "10 passed",
            fmt_output: "",
            target_ets: &target_ets,
            fmt_time: 0.1,
            test_time: 1.0,
            ets_time: 2.0,
        };
        let params = passing_params(&seed);
        let report = build_report(&params);
        assert_report_headings(&report);
    }

    #[test]
    fn test_build_report_ci_options() {
        let tarp = TarpaulinSummary {
            passed: 0,
            failed: 0,
            ignored: 0,
            coverage_percent: f64::NAN,
            covered_lines: 0,
            total_lines: 0,
        };
        let options = CiOptions {
            fmt: true,
            skip: CiSkip::COV.union(CiSkip::ETS),
        };
        let seed = PassingSeed {
            title: "numerical-models",
            options,
            tarp: &tarp,
            tarp_output: "",
            fmt_output: "reformatted file.rs\n",
            target_ets: &[],
            fmt_time: 0.2,
            test_time: 0.0,
            ets_time: 0.0,
        };
        let params = passing_params(&seed);
        let report = build_report(&params);
        assert!(report.contains("## `numerical-models` Quality Report"));
        assert!(report.contains("`cargo fmt --all`"));
        assert!(report.contains("--- cargo fmt --all ---"));
        assert!(report.contains("| **Formatting** | Pass |"));
        assert!(report.contains("0 failures (clean run)"));
        assert!(report.contains("skipped (not in pipeline)"));
        assert!(report.contains("Tarpaulin test & coverage was skipped."));
    }
}
