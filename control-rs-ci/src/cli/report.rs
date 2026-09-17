//! Logic for the standalone quality report aggregator tool (`cargo report` /
//! `report`). `bin/report.rs` is a thin shell over [`main_impl`].

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::gates::{HostToolSummary, TarpaulinSummary};
use crate::report::{
    self, CiOptions, CiReportParams, CiSkip, GateVerdict, HostToolRow,
};
use crate::trace::TraceMatrixSummary;
use crate::validate::CrossComparisonSummary;
use control_rs_ets::comms::TestState;
use control_rs_ets_host::runner::TestOutcome;

/// Stand-in for a standard gate with no published verdict. A gate the
/// pipeline never reported is `Skip`, not `Pass`.
const UNMEASURED_GATE: StandardGate = StandardGate {
    verdict: GateVerdict::Skip,
    seconds: 0.0,
};

type TargetEtsResult<'a> = (&'a str, Result<Vec<TestOutcome>, String>);
type ParseArgsResult = Result<Option<ReportArgs>, String>;

struct ReportArgs {
    artifacts_dir: PathBuf,
    out_dir: PathBuf,
    title: String,
    /// Gate names whose failure is reported but does not block the
    /// pipeline, lowercased, from `--warn`.
    warn_gates: Vec<String>,
    /// Gate names that must have published a verdict, lowercased, from
    /// `--require`. A required gate with no artifact is a fail, not a skip
    /// (FR-13): a job that died before its upload step cannot be mistaken
    /// for one that passed.
    required_gates: Vec<String>,
}

/// One `[gates]` entry of `gates-report.json`, written by the workflow job
/// that runs the standard gates.
#[derive(serde::Deserialize)]
struct StandardGateEntry {
    gate: String,
    verdict: String,
    #[serde(default)]
    seconds: f32,
}

/// Deserialized `gates-report.json`.
#[derive(serde::Deserialize)]
struct StandardGateReport {
    #[serde(default)]
    gates: Vec<StandardGateEntry>,
}

/// Verdict and wall time of one standard gate.
#[derive(Debug, Clone, Copy)]
struct StandardGate {
    verdict: GateVerdict,
    seconds: f32,
}

/// Verdicts derived from the downloaded summaries rather than from a
/// published gate verdict. Computed once, so the rendered report and the
/// process exit code cannot disagree.
struct Verdicts {
    trace: GateVerdict,
    cross_val: GateVerdict,
}

struct ReportExtras<'a> {
    options: CiOptions,
    host_tools: &'a [HostToolRow<'a>],
    target_ets_results: &'a [TargetEtsResult<'a>],
    warned_gates: &'a [&'a str],
    trace: GateVerdict,
    cross_val: GateVerdict,
}

struct LoadedArtifacts {
    standard_gates: BTreeMap<String, StandardGate>,
    tarp_summary: TarpaulinSummary,
    tarp_exists: bool,
    trace_summary: Option<TraceMatrixSummary>,
    cross_val_summary: Option<CrossComparisonSummary>,
    ets_tests: Vec<TestOutcome>,
    host_tools: Vec<HostToolSummary>,
}

impl Default for ReportArgs {
    fn default() -> Self {
        Self {
            artifacts_dir: PathBuf::from("."),
            out_dir: PathBuf::from("."),
            title: "control-rs".to_string(),
            warn_gates: Vec::new(),
            required_gates: Vec::new(),
        }
    }
}

fn print_help() {
    println!(
        "Usage: cargo report [OPTIONS]\n       report [OPTIONS]\n\n\
        Options:\n  \
          --artifacts-dir <DIR>  Directory containing job JSON artifacts [default: .]\n  \
          --out-dir <DIR>        Directory to write ci-report.md [default: .]\n  \
          --title <TITLE>        Report title [default: control-rs]\n  \
          --warn <LIST>          Comma-separated gates reported but not gating\n  \
          --require <LIST>       Comma-separated gates that must publish a verdict\n  \
          -h, --help             Print help"
    );
}

/// Splits a comma-separated gate list into lowercased, non-empty names.
fn split_gate_list(raw: &str) -> impl Iterator<Item = String> + '_ {
    raw.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_ascii_lowercase)
}

fn handle_flag(
    arg: &str,
    iter: &mut impl Iterator<Item = String>,
    args: &mut ReportArgs,
) -> Result<(), String> {
    match arg {
        "--artifacts-dir" => {
            args.artifacts_dir =
                iter.next().map(PathBuf::from).unwrap_or_default();
        }
        "--out-dir" => {
            args.out_dir = iter.next().map(PathBuf::from).unwrap_or_default();
        }
        "--title" => {
            if let Some(val) = iter.next() {
                args.title.clone_from(&val);
            }
        }
        "--warn" => {
            if let Some(val) = iter.next() {
                args.warn_gates.extend(split_gate_list(&val));
            }
        }
        "--require" => {
            if let Some(val) = iter.next() {
                args.required_gates.extend(split_gate_list(&val));
            }
        }
        "--" => {}
        other => return Err(format!("Unknown argument: {other}")),
    }
    Ok(())
}

fn parse_args(args: &[String]) -> ParseArgsResult {
    let mut parsed = ReportArgs::default();
    let mut raw_args = args.iter().skip(1).cloned().peekable();
    if raw_args.peek().map(String::as_str) == Some("report") {
        raw_args.next();
    }
    while let Some(arg) = raw_args.next() {
        if matches!(arg.as_str(), "-h" | "--help") {
            print_help();
            return Ok(None);
        }
        handle_flag(&arg, &mut raw_args, &mut parsed)?;
    }
    Ok(Some(parsed))
}

fn load_tarpaulin_summary(tarp_path: &Path) -> TarpaulinSummary {
    #[derive(serde::Deserialize)]
    struct TarpFile {
        covered: usize,
        coverable: usize,
    }
    #[derive(serde::Deserialize)]
    struct TarpJson {
        files: Vec<TarpFile>,
    }

    let Ok(content) = fs::read_to_string(tarp_path) else {
        return TarpaulinSummary::default();
    };

    let Ok(report_json) = serde_json::from_str::<TarpJson>(&content) else {
        return TarpaulinSummary::default();
    };

    let mut covered: usize = 0;
    let mut total: usize = 0;
    for f in report_json.files {
        covered = covered.saturating_add(f.covered);
        total = total.saturating_add(f.coverable);
    }
    let pct = crate::gates::coverage_percent(covered, total);

    TarpaulinSummary {
        passed: covered,
        failed: 0,
        ignored: 0,
        coverage_percent: pct,
        covered_lines: covered,
        total_lines: total,
    }
}

fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    fs::read_to_string(path)
        .ok()
        .and_then(|c| serde_json::from_str(&c).ok())
}

/// Maps a published verdict string onto a [`GateVerdict`]. An unrecognized
/// value is `Fail`: the job said something the aggregator cannot read, which
/// is not evidence of a pass.
fn parse_verdict(raw: &str) -> GateVerdict {
    match raw.trim().to_ascii_lowercase().as_str() {
        "pass" | "ok" | "success" => GateVerdict::Pass,
        "warn" => GateVerdict::Warn,
        _ => GateVerdict::Fail,
    }
}

/// Loads the standard-gate verdicts published by the `check-and-test` job.
/// A missing or unreadable file yields an empty map, and every standard gate
/// then renders `Skipped`.
fn load_standard_gates(artifacts_dir: &Path) -> BTreeMap<String, StandardGate> {
    let path = artifacts_dir.join("gates-report.json");
    let Some(report) = load_json::<StandardGateReport>(&path) else {
        return BTreeMap::new();
    };
    report
        .gates
        .into_iter()
        .map(|entry| {
            (
                entry.gate.to_ascii_lowercase(),
                StandardGate {
                    verdict: parse_verdict(&entry.verdict),
                    seconds: entry.seconds,
                },
            )
        })
        .collect()
}

/// Verdict and wall time of one standard gate, or [`UNMEASURED_GATE`].
fn standard_gate(data: &LoadedArtifacts, name: &str) -> StandardGate {
    data.standard_gates
        .get(name)
        .copied()
        .unwrap_or(UNMEASURED_GATE)
}

/// True when `name` is on the `--warn` list.
fn is_warned(args: &ReportArgs, name: &str) -> bool {
    args.warn_gates.iter().any(|warned| warned == name)
}

fn load_artifacts(artifacts_dir: &Path) -> LoadedArtifacts {
    let standard_gates = load_standard_gates(artifacts_dir);
    let tarp_path = artifacts_dir.join("tarpaulin-report.json");
    let tarp_exists = tarp_path.exists();
    let tarp_summary = load_tarpaulin_summary(&tarp_path);
    let trace_path = artifacts_dir.join("trace-report.json");
    let trace_summary: Option<TraceMatrixSummary> = load_json(&trace_path);
    let cross_val_path = artifacts_dir.join("cross-val-report.json");
    let cross_val_summary: Option<CrossComparisonSummary> =
        load_json(&cross_val_path);
    let ets_path = artifacts_dir.join("ets-results.json");
    let ets_tests: Vec<TestOutcome> = load_json(&ets_path).unwrap_or_default();

    let mut host_tools = Vec::new();
    for name in [
        "mutants", "miri", "valgrind", "fuzz", "lockbud", "deny", "audit",
        "kani",
    ] {
        let path = artifacts_dir.join(format!("{name}-report.json"));
        if let Some(summary) = load_json::<HostToolSummary>(&path) {
            host_tools.push(summary);
        } else if name == "mutants" {
            let outcomes =
                artifacts_dir.join("mutants.out").join("outcomes.json");
            if let Ok(json) = fs::read_to_string(&outcomes) {
                if let Ok(summary) =
                    crate::gates::parse_mutants_outcomes(Some(&json), true)
                {
                    host_tools.push(HostToolSummary {
                        tool: "mutants".to_string(),
                        success: true,
                        skipped: false,
                        details: crate::gates::mutants_details(&summary),
                    });
                } else {
                    host_tools.push(HostToolSummary {
                        tool: "mutants".to_string(),
                        success: false,
                        skipped: false,
                        details: "`cargo mutants` failed".to_string(),
                    });
                }
            }
        }
    }

    LoadedArtifacts {
        standard_gates,
        tarp_summary,
        tarp_exists,
        trace_summary,
        cross_val_summary,
        ets_tests,
        host_tools,
    }
}

fn create_ci_options(data: &LoadedArtifacts, ets_empty: bool) -> CiOptions {
    let mut skip = CiSkip::default();
    if data.trace_summary.is_none() {
        skip.set(CiSkip::TRACE);
    }
    if data.cross_val_summary.is_none() {
        skip.set(CiSkip::EXAMPLES);
    }
    if ets_empty {
        skip.set(CiSkip::ETS);
    }
    CiOptions { fmt: false, skip }
}

const fn host_verdict(skipped: bool, success: bool) -> GateVerdict {
    if skipped {
        GateVerdict::Skip
    } else if success {
        GateVerdict::Pass
    } else {
        GateVerdict::Fail
    }
}

const fn gate_from_bool(ok: bool) -> GateVerdict {
    if ok {
        GateVerdict::Pass
    } else {
        GateVerdict::Fail
    }
}

/// Derives the trace and cross-validation verdicts from the summaries.
///
/// An absent summary is a pass here and a skipped section in the report, so a
/// job that did not run cannot fail the aggregate.
fn derive_verdicts(data: &LoadedArtifacts) -> Verdicts {
    let trace_ok = data.trace_summary.as_ref().is_none_or(|summary| {
        summary.approved_missing_count == 0
            && summary.approved_unresolved_count == 0
            && summary.failed_count == 0
    });
    let cross_ok = data
        .cross_val_summary
        .as_ref()
        .is_none_or(|summary| summary.failed_examples == 0);
    Verdicts {
        trace: gate_from_bool(trace_ok),
        cross_val: gate_from_bool(cross_ok),
    }
}

/// Every downloaded verdict that failed and is not on the `--warn` list.
/// A non-empty result is the aggregator's non-zero exit.
fn blocking_failures(
    args: &ReportArgs,
    data: &LoadedArtifacts,
    verdicts: &Verdicts,
) -> Vec<String> {
    let mut failed: Vec<String> = data
        .standard_gates
        .iter()
        .filter(|(name, gate)| gate.verdict.is_fail() && !is_warned(args, name))
        .map(|(name, _)| name.clone())
        .collect();
    failed.extend(
        data.host_tools
            .iter()
            .filter(|tool| {
                !tool.skipped && !tool.success && !is_warned(args, &tool.tool)
            })
            .map(|tool| tool.tool.clone()),
    );
    if verdicts.trace.is_fail() && !is_warned(args, "trace") {
        failed.push("trace".to_string());
    }
    if verdicts.cross_val.is_fail() && !is_warned(args, "validate") {
        failed.push("validate".to_string());
    }
    let ets_failed = data
        .ets_tests
        .iter()
        .any(|test| matches!(test.state, TestState::Failed));
    if ets_failed && !is_warned(args, "ets") {
        failed.push("ets".to_string());
    }
    if coverage_verdict(data).is_fail() && !is_warned(args, "coverage") {
        failed.push("coverage".to_string());
    }
    for name in &args.required_gates {
        if !is_warned(args, name)
            && !published_verdict(data, name)
            && !failed.iter().any(|already| already == name)
        {
            failed.push(name.clone());
        }
    }
    failed
}

/// True when `name` published a verdict this run. A required gate that did
/// not is a fail, not a skip (FR-13).
fn published_verdict(data: &LoadedArtifacts, name: &str) -> bool {
    match name {
        "coverage" => data.tarp_exists,
        "trace" => data.trace_summary.is_some(),
        "validate" => data.cross_val_summary.is_some(),
        "ets" => !data.ets_tests.is_empty(),
        other => {
            data.standard_gates.contains_key(other)
                || data.host_tools.iter().any(|tool| tool.tool == other)
        }
    }
}

const fn coverage_verdict(data: &LoadedArtifacts) -> GateVerdict {
    if data.tarp_exists {
        GateVerdict::Pass
    } else {
        GateVerdict::Fail
    }
}

fn build_ci_report(
    args: &ReportArgs,
    data: &LoadedArtifacts,
    verdicts: &Verdicts,
) -> String {
    let target_ets = if data.ets_tests.is_empty() {
        Vec::new()
    } else {
        vec![("Target Matrix", Ok(data.ets_tests.clone()))]
    };
    let target_ets_refs: Vec<TargetEtsResult<'_>> =
        target_ets.iter().map(|(k, v)| (*k, v.clone())).collect();
    let host_tool_rows: Vec<HostToolRow<'_>> = data
        .host_tools
        .iter()
        .map(|tool| HostToolRow {
            name: tool.tool.as_str(),
            command: tool.tool.as_str(),
            verdict: host_verdict(tool.skipped, tool.success),
            details: tool.details.as_str(),
            output: "",
            time: 0.0,
        })
        .collect();
    let options = create_ci_options(data, target_ets_refs.is_empty());
    let warned: Vec<&str> =
        args.warn_gates.iter().map(String::as_str).collect();
    report::build_report(&aggregator_params(
        args,
        data,
        &ReportExtras {
            options,
            host_tools: &host_tool_rows,
            target_ets_results: &target_ets_refs,
            warned_gates: &warned,
            trace: verdicts.trace,
            cross_val: verdicts.cross_val,
        },
    ))
}

fn aggregator_params<'a>(
    args: &'a ReportArgs,
    data: &'a LoadedArtifacts,
    extras: &ReportExtras<'a>,
) -> CiReportParams<'a> {
    CiReportParams {
        title: &args.title,
        options: extras.options,
        fmt: standard_gate(data, "fmt").verdict,
        fmt_output: "",
        fmt_time: standard_gate(data, "fmt").seconds,
        clippy: standard_gate(data, "clippy").verdict,
        clippy_output: "",
        clippy_time: standard_gate(data, "clippy").seconds,
        check: standard_gate(data, "check").verdict,
        check_output: "",
        check_time: standard_gate(data, "check").seconds,
        build: standard_gate(data, "build").verdict,
        build_output: "",
        build_time: standard_gate(data, "build").seconds,
        clean: standard_gate(data, "clean").verdict,
        clean_output: "",
        clean_time: standard_gate(data, "clean").seconds,
        test_cmd: standard_gate(data, "test").verdict,
        test_cmd_output: "",
        test_cmd_time: standard_gate(data, "test").seconds,
        coverage: coverage_verdict(data),
        tarp_summary: &data.tarp_summary,
        tarp_output: "",
        test_time: 0.0,
        host_tools: extras.host_tools,
        warned_gates: extras.warned_gates,
        skipped_standard: &[],
        target_ets_results: extras.target_ets_results,
        ets_time: 0.0,
        trace: extras.trace,
        trace_summary: data.trace_summary.as_ref(),
        trace_time: 0.0,
        cross_val: extras.cross_val,
        cross_val_summary: data.cross_val_summary.as_ref(),
        cross_val_time: 0.0,
    }
}

fn save_report(out_dir: &Path, content: &str) -> bool {
    if let Err(e) = fs::create_dir_all(out_dir) {
        report::error(format!(
            "Failed to create output directory '{}': {e}",
            out_dir.display()
        ));
    }

    let report_path = out_dir.join("ci-report.md");
    if let Err(e) = fs::write(&report_path, content) {
        report::error(format!(
            "Failed to write ci-report.md to '{}': {e}",
            report_path.display()
        ));
        return false;
    }
    report::status("Saved", format!("report to {}", report_path.display()));
    true
}

/// Aggregates the job artifacts under `--artifacts-dir` into `ci-report.md`.
///
/// Returns the process exit code: non-zero when a downloaded verdict failed
/// and its gate is not on the `--warn` list. Never exits the process itself,
/// so it can be driven from a test.
#[must_use]
pub fn main_impl(args: &[String]) -> i32 {
    report::init_color();
    let start_time = Instant::now();
    let cli_args = match parse_args(args) {
        Ok(Some(cli_args)) => cli_args,
        Ok(None) => return 0,
        Err(e) => {
            report::error(e);
            print_help();
            return 1;
        }
    };
    let artifacts = load_artifacts(&cli_args.artifacts_dir);
    let verdicts = derive_verdicts(&artifacts);
    let report_content = build_ci_report(&cli_args, &artifacts, &verdicts);
    if !save_report(&cli_args.out_dir, &report_content) {
        return 1;
    }

    let failures = blocking_failures(&cli_args, &artifacts, &verdicts);
    let elapsed = report::format_elapsed(start_time.elapsed());
    if failures.is_empty() {
        report::status(
            "Finished",
            format!("aggregated quality report in {elapsed}"),
        );
        return 0;
    }
    report::error(format!(
        "aggregated quality report in {elapsed}: failed gates: {}",
        failures.join(", ")
    ));
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_help_returns_none() {
        assert!(
            parse_args(&["report".to_string(), "--help".to_string()])
                .unwrap()
                .is_none()
        );
        assert!(
            parse_args(&["report".to_string(), "-h".to_string()])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_parse_args_unknown_flag_errors() {
        assert!(
            parse_args(&["report".to_string(), "--bogus".to_string()]).is_err()
        );
    }

    #[test]
    fn test_parse_args_defaults() {
        let parsed = parse_args(&["report".to_string()]).unwrap().unwrap();
        assert_eq!(parsed.artifacts_dir, PathBuf::from("."));
        assert_eq!(parsed.out_dir, PathBuf::from("."));
        assert_eq!(parsed.title, "control-rs");
    }

    #[test]
    fn test_parse_args_overrides() {
        let parsed = parse_args(&[
            "report".to_string(),
            "--artifacts-dir".to_string(),
            "target/ci-artifacts".to_string(),
            "--out-dir".to_string(),
            "target/ci".to_string(),
            "--title".to_string(),
            "my-title".to_string(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(parsed.artifacts_dir, PathBuf::from("target/ci-artifacts"));
        assert_eq!(parsed.out_dir, PathBuf::from("target/ci"));
        assert_eq!(parsed.title, "my-title");
    }

    #[test]
    fn test_load_artifacts_missing_dir_yields_empty_defaults() {
        let temp_dir = std::env::temp_dir()
            .join("control_rs_ci_test_report_cli_load_missing");
        let _ = fs::remove_dir_all(&temp_dir);
        let data = load_artifacts(&temp_dir);
        assert!(!data.tarp_exists);
        assert!(data.trace_summary.is_none());
        assert!(data.cross_val_summary.is_none());
        assert!(data.ets_tests.is_empty());
        assert!(data.host_tools.is_empty());
    }

    #[test]
    fn test_create_ci_options_skips_missing_artifacts() {
        let data = LoadedArtifacts {
            standard_gates: BTreeMap::new(),
            tarp_summary: TarpaulinSummary::default(),
            tarp_exists: false,
            trace_summary: None,
            cross_val_summary: None,
            ets_tests: Vec::new(),
            host_tools: Vec::new(),
        };
        let options = create_ci_options(&data, true);
        assert!(!options.skip.contains(CiSkip::COV));
        assert!(options.skip.contains(CiSkip::TRACE));
        assert!(options.skip.contains(CiSkip::EXAMPLES));
        assert!(options.skip.contains(CiSkip::ETS));
    }

    #[test]
    fn test_host_verdict_mapping() {
        assert_eq!(host_verdict(true, true), GateVerdict::Skip);
        assert_eq!(host_verdict(false, true), GateVerdict::Pass);
        assert_eq!(host_verdict(false, false), GateVerdict::Fail);
    }

    #[test]
    fn test_gate_from_bool() {
        assert_eq!(gate_from_bool(true), GateVerdict::Pass);
        assert_eq!(gate_from_bool(false), GateVerdict::Fail);
    }

    #[test]
    fn test_main_impl_writes_report_from_fixture_artifacts() {
        let temp_dir = std::env::temp_dir()
            .join("control_rs_ci_test_report_cli_main_impl");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let code = main_impl(&[
            "report".to_string(),
            "--artifacts-dir".to_string(),
            temp_dir.to_string_lossy().to_string(),
            "--out-dir".to_string(),
            temp_dir.to_string_lossy().to_string(),
            "--title".to_string(),
            "test-title".to_string(),
        ]);
        assert_eq!(code, 1);
        assert!(temp_dir.join("ci-report.md").exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    fn empty_artifacts() -> LoadedArtifacts {
        LoadedArtifacts {
            standard_gates: BTreeMap::new(),
            tarp_summary: TarpaulinSummary::default(),
            tarp_exists: false,
            trace_summary: None,
            cross_val_summary: None,
            ets_tests: Vec::new(),
            host_tools: Vec::new(),
        }
    }

    fn args_with_warns(warns: &[&str]) -> ReportArgs {
        ReportArgs {
            warn_gates: warns.iter().map(|w| (*w).to_string()).collect(),
            ..ReportArgs::default()
        }
    }

    #[test]
    fn test_parse_verdict_unknown_is_fail() {
        assert_eq!(parse_verdict("pass"), GateVerdict::Pass);
        assert_eq!(parse_verdict(" FAIL "), GateVerdict::Fail);
        assert_eq!(parse_verdict("warn"), GateVerdict::Warn);
        assert_eq!(parse_verdict("nonsense"), GateVerdict::Fail);
    }

    #[test]
    fn test_parse_args_warn_list_is_split_and_lowercased() {
        let parsed = parse_args(&[
            "report".to_string(),
            "--warn".to_string(),
            "Trace, validate".to_string(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(parsed.warn_gates, vec!["trace", "validate"]);
    }

    #[test]
    fn test_unmeasured_standard_gate_is_skipped_not_passed() {
        let data = empty_artifacts();
        assert_eq!(standard_gate(&data, "fmt").verdict, GateVerdict::Skip);
    }

    #[test]
    fn test_load_standard_gates_reads_published_verdicts() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_standard_gates");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        fs::write(
            temp_dir.join("gates-report.json"),
            r#"{"gates":[{"gate":"fmt","verdict":"pass","seconds":1.5},
                         {"gate":"clippy","verdict":"fail","seconds":2.0}]}"#,
        )
        .unwrap();

        let gates = load_standard_gates(&temp_dir);
        assert_eq!(
            gates.get("fmt").map(|gate| gate.verdict),
            Some(GateVerdict::Pass)
        );
        assert_eq!(
            gates.get("clippy").map(|gate| gate.verdict),
            Some(GateVerdict::Fail)
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_blocking_failures_reports_failed_standard_gate() {
        let mut data = empty_artifacts();
        data.tarp_exists = true;
        data.standard_gates.insert(
            "coverage".to_string(),
            StandardGate {
                verdict: GateVerdict::Pass,
                seconds: 1.0,
            },
        );
        data.standard_gates.insert(
            "clippy".to_string(),
            StandardGate {
                verdict: GateVerdict::Fail,
                seconds: 1.0,
            },
        );
        let verdicts = derive_verdicts(&data);
        let failures =
            blocking_failures(&ReportArgs::default(), &data, &verdicts);
        assert_eq!(failures, vec!["clippy".to_string()]);
    }

    #[test]
    /// A required gate that published no verdict blocks, so a job that died
    /// before its upload step cannot be mistaken for a passing one.
    ///
    /// # Verification
    /// Trace: ci-design#FR-13
    /// Method: Requirements-based test
    fn test_blocking_failures_required_gate_without_artifact_is_fail() {
        let mut data = empty_artifacts();
        data.tarp_exists = true;
        let verdicts = derive_verdicts(&data);
        let args = ReportArgs {
            required_gates: vec![
                "deny".to_string(),
                "audit".to_string(),
                "ets".to_string(),
            ],
            ..ReportArgs::default()
        };
        let failures = blocking_failures(&args, &data, &verdicts);
        for name in ["deny", "audit", "ets"] {
            assert!(
                failures.iter().any(|failed| failed == name),
                "{name} published no verdict and must block, got {failures:?}"
            );
        }
    }

    #[test]
    /// An enabled ETS matrix that produced zero cases publishes an empty
    /// result set, which is not a pass.
    ///
    /// # Verification
    /// Trace: ci-design#FR-5
    /// Method: Requirements-based test
    fn test_blocking_failures_empty_ets_is_fail() {
        let mut data = empty_artifacts();
        data.tarp_exists = true;
        data.ets_tests = Vec::new();
        let verdicts = derive_verdicts(&data);
        let args = ReportArgs {
            required_gates: vec!["ets".to_string()],
            ..ReportArgs::default()
        };
        let failures = blocking_failures(&args, &data, &verdicts);
        assert!(
            failures.iter().any(|failed| failed == "ets"),
            "an empty ETS result set must block, got {failures:?}"
        );
    }

    #[test]
    /// Missing coverage is a fail-closed hole, not a skip.
    ///
    /// # Verification
    /// Trace: ci-design#FR-13
    /// Method: Requirements-based test
    fn test_blocking_failures_missing_coverage_is_fail() {
        let data = empty_artifacts();
        let verdicts = derive_verdicts(&data);
        let failures =
            blocking_failures(&ReportArgs::default(), &data, &verdicts);
        assert!(
            failures.iter().any(|name| name == "coverage"),
            "missing tarpaulin artifact must block, got {failures:?}"
        );
    }

    #[test]
    fn test_blocking_failures_honors_warn_list() {
        let mut data = empty_artifacts();
        data.standard_gates.insert(
            "clippy".to_string(),
            StandardGate {
                verdict: GateVerdict::Fail,
                seconds: 1.0,
            },
        );
        data.trace_summary = Some(TraceMatrixSummary {
            approved_missing_count: 12,
            ..TraceMatrixSummary::default()
        });
        let verdicts = derive_verdicts(&data);
        let args = args_with_warns(&["trace", "clippy", "coverage"]);
        assert!(blocking_failures(&args, &data, &verdicts).is_empty());
    }

    #[test]
    fn test_blocking_failures_reports_failed_trace() {
        let mut data = empty_artifacts();
        data.tarp_exists = true;
        data.standard_gates.insert(
            "coverage".to_string(),
            StandardGate {
                verdict: GateVerdict::Pass,
                seconds: 1.0,
            },
        );
        data.trace_summary = Some(TraceMatrixSummary {
            approved_missing_count: 12,
            ..TraceMatrixSummary::default()
        });
        let verdicts = derive_verdicts(&data);
        let failures =
            blocking_failures(&ReportArgs::default(), &data, &verdicts);
        assert_eq!(failures, vec!["trace".to_string()]);
    }

    #[test]
    fn test_main_impl_unknown_flag_returns_one() {
        assert_eq!(main_impl(&["report".to_string(), "--nope".to_string()]), 1);
    }

    #[test]
    fn test_main_impl_help_returns_zero() {
        assert_eq!(main_impl(&["report".to_string(), "--help".to_string()]), 0);
    }
}
