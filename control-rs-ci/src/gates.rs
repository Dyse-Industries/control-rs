//! Subprocess invocation wrappers for `cargo` host verification gates.

use regex::Regex;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use crate::report;

static CAPTURE_ONLY: AtomicBool = AtomicBool::new(false);

type EnvPair<'a> = (&'a str, &'a str);
type GateTriple<T> = (bool, T, String);
type JoinBytes = JoinHandle<Vec<u8>>;
type MutantsParse = (Result<MutantsSummary, MutantsError>, String);
type TarpaulinParse = (Result<TarpaulinSummary, TarpaulinError>, String);

/// Compact status for an off-the-shelf host verification tool.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HostToolSummary {
    /// Canonical gate / tool name (e.g. `mutants`, `deny`).
    pub tool: String,
    /// Whether the tool exited successfully.
    pub success: bool,
    /// Whether the gate was skipped (missing harness, disabled, etc.).
    pub skipped: bool,
    /// One-line verdict details for `ci-report.md`.
    pub details: String,
}

/// Failure to meet `cargo-mutants` kill criteria or parse `outcomes.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutantsError {
    /// JSON missing or unparseable, or missed/timeout/unviable criteria failed.
    Unsuccessful,
}

/// Summary of a `cargo-mutants` run parsed from `mutants.out/outcomes.json`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MutantsSummary {
    /// Total mutants generated.
    pub total_mutants: usize,
    /// Mutants caught by tests.
    pub caught: usize,
    /// Surviving mutants.
    pub missed: usize,
    /// Mutants that timed out.
    pub timeout: usize,
    /// Mutants that failed to build.
    pub unviable: usize,
}

#[derive(serde::Deserialize)]
struct MutantsOutcomesJson {
    #[serde(default)]
    total_mutants: usize,
    #[serde(default)]
    missed: usize,
    #[serde(default)]
    caught: usize,
    #[serde(default)]
    timeout: usize,
    #[serde(default)]
    unviable: usize,
}

/// Mirrors child stdout/stderr onto the parent while retaining a capture buffer.
pub(crate) struct StdioTee {
    stdout: JoinBytes,
    stderr: JoinBytes,
}

/// Failure of a tarpaulin invocation or its parsed test results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TarpaulinError {
    /// The command failed or at least one test failed.
    Unsuccessful,
}

/// Summary of cargo tarpaulin test execution and line coverage.
#[derive(Debug, Clone, Default)]
pub struct TarpaulinSummary {
    /// Number of tests passed.
    pub passed: usize,
    /// Number of tests failed.
    pub failed: usize,
    /// Number of tests ignored.
    pub ignored: usize,
    /// The formatted coverage percentage (e.g., "85.20").
    pub coverage_percent: f64,
    /// Number of lines covered.
    pub covered_lines: usize,
    /// Total number of coverable lines.
    pub total_lines: usize,
}

#[derive(serde::Deserialize)]
struct TarpaulinFile {
    covered: usize,
    coverable: usize,
}

#[derive(serde::Deserialize)]
struct TarpaulinReportJson {
    files: Vec<TarpaulinFile>,
}

impl StdioTee {
    pub(crate) fn attach(child: &mut Child) -> Self {
        let live = !capture_only();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdout_h = thread::spawn(move || {
            stdout.map_or_else(Vec::new, |mut src| {
                copy_stream(&mut src, &mut io::stdout(), live)
            })
        });
        let stderr_h = thread::spawn(move || {
            stderr.map_or_else(Vec::new, |mut src| {
                copy_stream(&mut src, &mut io::stderr(), live)
            })
        });
        Self {
            stdout: stdout_h,
            stderr: stderr_h,
        }
    }

    pub(crate) fn join(self) -> String {
        let stdout = self.stdout.join().ok().unwrap_or_default();
        let stderr = self.stderr.join().ok().unwrap_or_default();
        strip_ansi(&format!(
            "{}{}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        ))
    }
}

/// Coverage percent from covered/total lines without a `usize as f64` cast.
///
/// Computes integer millipercent (`covered * 10_000 / total`) then converts
/// the whole and fractional parts through `u32` → `f64`, which is exact.
#[must_use]
pub fn coverage_percent(covered: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let covered = u64::try_from(covered).unwrap_or(u64::MAX);
    let total = u64::try_from(total).unwrap_or(u64::MAX);
    let milli = covered
        .saturating_mul(10_000)
        .checked_div(total)
        .unwrap_or(0);
    let whole = u32::try_from(milli / 100).unwrap_or(u32::MAX);
    let frac = u32::try_from(milli % 100).unwrap_or(0);
    f64::from(whole) + f64::from(frac) / 100.0
}

/// Capture child stdio without mirroring it to the parent terminal (`--quiet`).
pub fn set_capture_only(quiet: bool) {
    CAPTURE_ONLY.store(quiet, Ordering::Relaxed);
}

pub(crate) fn capture_only() -> bool {
    CAPTURE_ONLY.load(Ordering::Relaxed)
}

/// Color mode for teed children. Pipes are never a TTY, so live runs must
/// force `always` or cargo/tarpaulin emit colorless output.
pub(crate) fn cargo_color() -> &'static str {
    if capture_only() { "never" } else { "always" }
}

/// Honor `--quiet`. Live tees force color; quiet captures stay plain.
pub(crate) fn apply_cargo_term_color(cmd: &mut Command) {
    let color = cargo_color();
    cmd.env("CARGO_TERM_COLOR", color);
    if color == "always" {
        cmd.env("CLICOLOR_FORCE", "1");
        cmd.env_remove("NO_COLOR");
    } else {
        cmd.env_remove("CLICOLOR_FORCE");
        cmd.env("NO_COLOR", "1");
    }
}

fn copy_stream(
    src: &mut impl Read,
    dest: &mut impl Write,
    live: bool,
) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        match src.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let Some(slice) = tmp.get(..n) else {
                    break;
                };
                buf.extend_from_slice(slice);
                if live {
                    let _ = dest.write_all(slice);
                    let _ = dest.flush();
                }
            }
        }
    }
    buf
}

/// Strip ANSI CSI sequences from cargo tool output.
#[must_use]
pub fn strip_ansi(input: &str) -> String {
    let Ok(ansi_escape) = Regex::new(r"\x1B\[[0-9;]*[mK]|\x1B\(B") else {
        return input.to_string();
    };
    ansi_escape.replace_all(input, "").into_owned()
}

fn run_logged_command(
    label: &str,
    program: &str,
    args: &[&str],
    envs: &[EnvPair<'_>],
) -> (bool, String) {
    report::status("Running", label);
    let mut cmd = Command::new(program);
    if program == "cargo" {
        cmd.arg("--color").arg(cargo_color());
    }
    cmd.args(args);
    apply_cargo_term_color(&mut cmd);
    for (key, value) in envs {
        cmd.env(key, value);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    match cmd.spawn() {
        Ok(mut child) => {
            let tee = StdioTee::attach(&mut child);
            match child.wait() {
                Ok(status) => (status.success(), tee.join()),
                Err(e) => (false, format!("Failed to execute {label}: {e}")),
            }
        }
        Err(e) => (false, format!("Failed to execute {label}: {e}")),
    }
}

fn sum_count_before_word(haystack: &str, word: &str) -> usize {
    let mut total = 0usize;
    let mut rest = haystack;
    let needle = format!(" {word}");
    while let Some(idx) = rest.find(&needle) {
        let before = rest.get(..idx).unwrap_or("");
        let digits = before
            .rsplit(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap_or("");
        if let Ok(n) = digits.parse::<usize>() {
            total = total.saturating_add(n);
        }
        rest = rest.get(idx.saturating_add(needle.len())..).unwrap_or("");
    }
    total
}

/// Parse tarpaulin logs and optional JSON coverage report.
pub fn parse_tarpaulin_output(
    tarp_str: &str,
    json_content: Option<&str>,
    cmd_success: bool,
) -> TarpaulinParse {
    let clean_tarp_str = strip_ansi(tarp_str);
    let passed = sum_count_before_word(&clean_tarp_str, "passed");
    let failed = sum_count_before_word(&clean_tarp_str, "failed");
    let ignored = sum_count_before_word(&clean_tarp_str, "ignored");

    let mut covered_lines = 0usize;
    let mut total_lines = 0usize;
    if let Some(json_content) = json_content
        && let Ok(report) =
            serde_json::from_str::<TarpaulinReportJson>(json_content)
    {
        for file in report.files {
            covered_lines = covered_lines.saturating_add(file.covered);
            total_lines = total_lines.saturating_add(file.coverable);
        }
    }

    let summary = TarpaulinSummary {
        passed,
        failed,
        ignored,
        coverage_percent: coverage_percent(covered_lines, total_lines),
        covered_lines,
        total_lines,
    };

    let success = cmd_success && failed == 0;
    let res = if success {
        Ok(summary)
    } else {
        Err(TarpaulinError::Unsuccessful)
    };
    (res, clean_tarp_str)
}

/// Task to run tarpaulin test & coverage.
pub fn run_tarpaulin() -> TarpaulinParse {
    let color = cargo_color();
    let (ok, tarp_str) = run_logged_command(
        "`cargo tarpaulin --workspace --out Html --out Json`",
        "cargo",
        &[
            "tarpaulin",
            "--workspace",
            "--color",
            color,
            "--out",
            "Html",
            "--out",
            "Json",
        ],
        &[],
    );
    let json_content = fs::read_to_string("tarpaulin-report.json").ok();
    parse_tarpaulin_output(&tarp_str, json_content.as_deref(), ok)
}

/// Task to run requirement traceability analysis.
#[must_use]
pub fn run_traceability(
    repo_root: &std::path::Path,
    cargo_test_output: Option<&str>,
    ets_results_json: Option<&str>,
) -> GateTriple<crate::trace::TraceMatrixSummary> {
    report::status("Running", "`cargo trace`");
    let summary = crate::trace::run_traceability_analysis(
        repo_root,
        cargo_test_output,
        ets_results_json,
    );
    let silence = summary.total_documents == 0
        || (cargo_test_output.is_some_and(|s| !s.trim().is_empty())
            && summary.recorded_outcomes_count == 0);
    let success = !silence
        && summary.approved_missing_count == 0
        && summary.approved_unresolved_count == 0
        && summary.failed_count == 0;
    let brief = crate::trace::render_trace_brief(&summary);
    (success, summary, brief)
}

/// Task to run cross-comparison and oracle verification over N examples.
#[must_use]
pub fn run_cross_comparison(
    examples: &[crate::validate::ExampleTargetConfig],
    repo_root: &std::path::Path,
    timeout_secs: u64,
) -> GateTriple<crate::validate::CrossComparisonSummary> {
    report::status("Running", "`cargo validate`");
    let summary = crate::validate::run_cross_comparison(
        examples,
        repo_root,
        timeout_secs,
    );
    let success = summary.failed_examples == 0;
    let brief = crate::validate::render_cross_comparison_brief(&summary);
    (success, summary, brief)
}

/// Parse `cargo-mutants` `outcomes.json`.
///
/// # Errors
///
/// Returns [`MutantsError::Unsuccessful`] when the JSON is missing or
/// unparseable, the command failed, mutants were missed or timed out, or
/// no viable mutants were tested.
pub fn parse_mutants_outcomes(
    json_content: Option<&str>,
    cmd_success: bool,
) -> Result<MutantsSummary, MutantsError> {
    let parsed = json_content.and_then(|json| {
        serde_json::from_str::<MutantsOutcomesJson>(json).ok()
    });
    let Some(parsed) = parsed else {
        return Err(MutantsError::Unsuccessful);
    };
    let summary = MutantsSummary {
        total_mutants: parsed.total_mutants,
        caught: parsed.caught,
        missed: parsed.missed,
        timeout: parsed.timeout,
        unviable: parsed.unviable,
    };
    let viable = summary.total_mutants.saturating_sub(summary.unviable);
    let success = cmd_success
        && summary.missed == 0
        && summary.timeout == 0
        && viable > 0;
    if success {
        Ok(summary)
    } else {
        Err(MutantsError::Unsuccessful)
    }
}

/// Task to run `cargo mutants` against library unit tests.
pub fn run_mutants(out_dir: &Path) -> MutantsParse {
    let out_dir_str = out_dir.to_string_lossy().to_string();
    let label = format!("`cargo mutants --output {}`", out_dir.display());
    let _ = fs::create_dir_all(out_dir);
    let (ok, output) = run_logged_command(
        &label,
        "cargo",
        &["mutants", "--output", &out_dir_str],
        &[],
    );
    let json_path = out_dir.join("mutants.out").join("outcomes.json");
    let json_content = fs::read_to_string(json_path).ok();
    (parse_mutants_outcomes(json_content.as_deref(), ok), output)
}

/// Task to run Miri on library unit tests.
#[must_use]
pub fn run_miri() -> (bool, String) {
    run_logged_command(
        "`cargo miri test -p control-rs --lib`",
        "cargo",
        &["miri", "test", "-p", "control-rs", "--lib"],
        &[],
    )
}

/// Reason a host tool cannot apply on this host, or `None` when it can.
///
/// A gate that cannot execute here reports Skip rather than Pass: a green
/// verdict has to mean the check ran.
#[must_use]
pub fn host_tool_inapplicable(gate: crate::runner::Gate) -> Option<String> {
    if gate == crate::runner::Gate::Valgrind && !valgrind_applicable() {
        return Some(format!(
            "not applicable on {}-{}: the valgrind gate drives \
             CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER, which only \
             takes effect on x86_64-unknown-linux-gnu",
            std::env::consts::ARCH,
            std::env::consts::OS,
        ));
    }
    None
}

/// Whether the valgrind gate's cargo runner override applies to this host.
#[must_use]
pub const fn valgrind_applicable() -> bool {
    cfg!(all(target_arch = "x86_64", target_os = "linux"))
}

/// Task to run host unit tests under Valgrind.
#[must_use]
pub fn run_valgrind() -> (bool, String) {
    if Command::new("valgrind").arg("--version").output().is_err() {
        return (
            false,
            "Failed to execute `valgrind`: tool not found on PATH".to_string(),
        );
    }
    run_logged_command(
        "`valgrind cargo test -p control-rs --lib`",
        "cargo",
        &["test", "-p", "control-rs", "--lib"],
        &[(
            "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER",
            "valgrind --error-exitcode=1 --leak-check=full --quiet",
        )],
    )
}

/// Whether a cargo-fuzz workspace is present in `dir`.
#[must_use]
pub fn fuzz_workspace_present(dir: &Path) -> bool {
    dir.join("fuzz").join("Cargo.toml").exists()
}

/// Task to run time-bounded `cargo fuzz` campaigns.
#[must_use]
pub fn run_fuzz(dir: &Path, timeout_secs: u64) -> (bool, String) {
    if !fuzz_workspace_present(dir) {
        return (false, "no fuzz/ workspace".to_string());
    }
    let (list_ok, list_out) = run_logged_command(
        "`cargo fuzz list`",
        "cargo",
        &["fuzz", "list"],
        &[],
    );
    if !list_ok {
        return (false, list_out);
    }
    let targets: Vec<String> = list_out
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(String::from)
        .collect();
    if targets.is_empty() {
        return (
            false,
            format!("cargo fuzz list returned no targets\n{list_out}"),
        );
    }
    let timeout = timeout_secs.to_string();
    let mut combined = list_out;
    for target in targets {
        let label =
            format!("`cargo fuzz run {target} -- -max_total_time={timeout}`");
        let (ok, output) = run_logged_command(
            &label,
            "cargo",
            &[
                "fuzz",
                "run",
                target.as_str(),
                "--",
                "-max_total_time",
                &timeout,
            ],
            &[],
        );
        combined.push('\n');
        combined.push_str(&output);
        if !ok {
            return (false, combined);
        }
    }
    (true, combined)
}

/// Task to run lockbud deadlock / atomicity analysis.
#[must_use]
pub fn run_lockbud() -> (bool, String) {
    run_logged_command(
        "`cargo lockbud -k all`",
        "cargo",
        &["lockbud", "-k", "all"],
        &[],
    )
}

/// Task to run `cargo deny check`.
#[must_use]
pub fn run_deny() -> (bool, String) {
    run_logged_command(
        "`cargo deny --all-features check`",
        "cargo",
        &["deny", "--all-features", "check"],
        &[],
    )
}

/// Task to run `cargo audit`.
#[must_use]
pub fn run_audit() -> (bool, String) {
    if !Path::new("Cargo.lock").exists() {
        let (ok, output) = run_logged_command(
            "`cargo generate-lockfile`",
            "cargo",
            &["generate-lockfile"],
            &[],
        );
        if !ok {
            return (false, output);
        }
    }
    run_logged_command(
        "`cargo audit --deny warnings`",
        "cargo",
        &["audit", "--deny", "warnings"],
        &[],
    )
}

/// Task to run Kani proofs.
#[must_use]
pub fn run_kani() -> (bool, String) {
    run_logged_command("`cargo kani`", "cargo", &["kani"], &[])
}

/// Format a one-line mutants verdict.
#[must_use]
pub fn mutants_details(summary: &MutantsSummary) -> String {
    format!(
        "{} caught, {} missed, {} timeout, {} unviable ({} tested)",
        summary.caught,
        summary.missed,
        summary.timeout,
        summary.unviable,
        summary.total_mutants
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tarpaulin_sums_logs_and_json() {
        let log = "3 passed; 1 failed; 2 ignored";
        let json = r#"{"files":[{"covered":9,"coverable":10}]}"#;
        let (err, clean) = parse_tarpaulin_output(log, Some(json), true);
        assert!(err.is_err());
        assert!(clean.contains("3 passed"));

        let ok_log = "4 passed; 0 failed; 0 ignored";
        let (ok, _) = parse_tarpaulin_output(ok_log, Some(json), true);
        let summary = ok.unwrap();
        assert_eq!(summary.passed, 4);
        assert_eq!(summary.covered_lines, 9);
        assert_eq!(summary.total_lines, 10);
        assert!((summary.coverage_percent - 90.0).abs() < 1e-9);

        let (zero, _) = parse_tarpaulin_output("0 passed", None, true);
        assert!(zero.unwrap().coverage_percent.abs() < 1e-9);
    }

    #[test]
    fn parse_mutants_outcomes_requires_kills() {
        let json = r#"{"total_mutants":10,"caught":8,"missed":0,"timeout":0,"unviable":2}"#;
        let ok = parse_mutants_outcomes(Some(json), true).unwrap();
        assert_eq!(ok.caught, 8);
        assert_eq!(ok.total_mutants, 10);
        assert_eq!(
            mutants_details(&ok),
            "8 caught, 0 missed, 0 timeout, 2 unviable (10 tested)"
        );

        let missed = r#"{"total_mutants":10,"caught":7,"missed":1,"timeout":0,"unviable":2}"#;
        assert!(parse_mutants_outcomes(Some(missed), true).is_err());
        assert!(parse_mutants_outcomes(None, true).is_err());
        let empty = r#"{"total_mutants":0,"caught":0,"missed":0,"timeout":0,"unviable":0}"#;
        assert!(parse_mutants_outcomes(Some(empty), true).is_err());
    }

    #[test]
    fn copy_stream_buffers_and_mirrors() {
        let mut src = io::Cursor::new(b"hello");
        let mut dest = Vec::new();
        let buf = copy_stream(&mut src, &mut dest, true);
        assert_eq!(buf, b"hello");
        assert_eq!(dest, b"hello");

        let mut src = io::Cursor::new(b"secret");
        let mut dest = Vec::new();
        let buf = copy_stream(&mut src, &mut dest, false);
        assert_eq!(buf, b"secret");
        assert!(dest.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn teed_command_captures_stdout_and_stderr() {
        let (ok, out) = run_logged_command(
            "`sh`",
            "sh",
            &["-c", "printf stdout-tee; printf stderr-tee >&2"],
            &[],
        );
        assert!(ok);
        assert!(out.contains("stdout-tee"));
        assert!(out.contains("stderr-tee"));
    }
    #[test]
    /// A host tool that cannot execute here reports Skip, never Pass.
    ///
    /// # Verification
    /// Trace: ci-design#FR-13
    /// Method: Requirements-based test
    fn test_valgrind_applicability_matches_host() {
        use crate::runner::Gate;

        let applicable = valgrind_applicable();
        assert_eq!(
            applicable,
            cfg!(all(target_arch = "x86_64", target_os = "linux")),
            "the valgrind runner override only takes effect on \
             x86_64-unknown-linux-gnu"
        );
        let reason = host_tool_inapplicable(Gate::Valgrind);
        assert_eq!(
            reason.is_none(),
            applicable,
            "valgrind must report a reason exactly when it cannot run"
        );
        if let Some(text) = reason {
            assert!(
                text.contains("not applicable"),
                "reason must say the gate did not run, got {text}"
            );
        }
        assert!(
            host_tool_inapplicable(Gate::Deny).is_none(),
            "only valgrind is host-restricted today"
        );
    }
}
