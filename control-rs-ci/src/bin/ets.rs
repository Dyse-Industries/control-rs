//! Headless ETS runner (`cargo ets`).
//!
//! Builds each target's firmware, runs its suites to completion through
//! `control-rs-ets-host`, judges every run with the rule in
//! `control_rs_ci::ets` and writes `ets-results.json`. Exits 0 only when every
//! target passes.
//!
//! ```text
//! cargo ets [--timeout <secs>] [--max-resets <n>] [--out <path>] <targets>...
//! cargo ets --timeout 120 qemu all --release
//! ```
//!
//! Target arguments use the `cargo tui` syntax and are passed unchanged to
//! `control_rs_ets_host::target::parse_targets`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Duration;

use control_rs_ci::ets::{TargetResult, case_line};
use control_rs_ci::ui;
use control_rs_ets::comms::TestState;
use control_rs_ets_host::target::parse_targets;
use control_rs_ets_host::{
    RunOptions, RunRecord, Target, build_target_elf,
    run_headless_ets_with_options,
};

/// Default result file, inside the CI artifact directory.
const DEFAULT_OUT: &str = "target/ci-artifacts/ets-results.json";

/// Options owned by this binary; everything else names targets.
struct EtsArgs {
    /// Session bounds passed to the headless runner.
    options: RunOptions,
    /// Result file path.
    out: PathBuf,
    /// Arguments forwarded to `parse_targets`.
    target_args: Vec<String>,
}

fn print_usage() {
    ui::init_color();
    let h = ui::HELP_HEADER;
    let f = ui::HELP_FLAG;
    let a = ui::HELP_ARG;
    anstream::println!(
        "{h}Usage:{h:#} {f}cargo ets{f:#} {a}[OPTIONS]{a:#} {a}<TARGETS>...{a:#}\n\n\
         {h}Options:{h:#}\n  \
           {f}--timeout{f:#} {a}<secs>{a:#}     Whole-session bound per target [default: 120]\n  \
           {f}--max-resets{f:#} {a}<n>{a:#}     Target resets allowed per session [default: 3]\n  \
           {f}--out{f:#} {a}<path>{a:#}         Result file [default: {DEFAULT_OUT}]\n  \
           {f}-h{f:#}, {f}--help{f:#}           Print help information\n\n\
         {h}Targets:{h:#} same syntax as {f}cargo tui{f:#}, for example\n  \
           {f}cargo ets{f:#} {a}qemu all --release{a:#}\n  \
           {f}cargo ets{f:#} {a}qemu arm riscv32{a:#}\n  \
           {f}cargo ets{f:#} {a}teensy --port /dev/ttyACM0{a:#}"
    );
}

/// Value following `flag`, parsed; exits with a usage error when missing or
/// malformed.
fn flag_value<T: std::str::FromStr>(value: Option<&String>, flag: &str) -> T {
    value.and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        ui::error(format!("{flag} requires a valid value"));
        exit(2);
    })
}

fn parse_args(args: &[String]) -> EtsArgs {
    let mut parsed = EtsArgs {
        options: RunOptions {
            timeout: Duration::from_secs(120),
            ..RunOptions::default()
        },
        out: PathBuf::from(DEFAULT_OUT),
        // `parse_targets` skips a leading program name.
        target_args: vec!["ets".to_string()],
    };
    let mut rest = args.iter().skip(1);
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                exit(0);
            }
            "--timeout" => {
                parsed.options.timeout =
                    Duration::from_secs(flag_value(rest.next(), "--timeout"));
            }
            "--max-resets" => {
                parsed.options.max_resets =
                    flag_value(rest.next(), "--max-resets");
            }
            "--out" => {
                parsed.out = flag_value(rest.next(), "--out");
            }
            _ => parsed.target_args.push(arg.clone()),
        }
    }
    parsed
}

/// Builds (for subprocess targets) and runs one target.
fn run_target(target: &Target, options: RunOptions) -> TargetResult {
    let name = target.display_name();
    if let Target::Subprocess(sub) = target {
        ui::status("Building", &name);
        if let Err(e) = build_target_elf(sub) {
            return TargetResult::from_error(name, e.to_string());
        }
    }
    ui::status("Running", &name);
    match run_headless_ets_with_options(target, options) {
        Ok(record) => TargetResult::from_record(name, record),
        Err(e) => TargetResult::from_error(name, e.to_string()),
    }
}

/// Prints one line per executed case, then the captured target console.
fn print_record(target: &str, record: &RunRecord) {
    for outcome in &record.results {
        let line = case_line(outcome);
        match outcome.state {
            TestState::Passed => ui::status("Pass", line),
            TestState::Failed => ui::failure("Fail", line),
            TestState::Pending | TestState::Running => {
                ui::warning(&format!("{:?}", outcome.state), line);
            }
        }
    }
    if record.console.trim().is_empty() {
        ui::status_info("Console", format!("{target}: no output"));
        return;
    }
    ui::status_info("Console", target);
    for line in record.console.lines() {
        anstream::eprintln!("{line}");
    }
}

/// Prints each case and the console for `result`, then its verdict line.
fn report(result: &TargetResult) {
    if let Some(record) = &result.record {
        print_record(&result.target, record);
    }
    if result.passed {
        ui::status(
            "Passed",
            format!("{}: {} test(s)", result.target, result.passed_tests()),
        );
        return;
    }
    let reason = result.reason.as_deref().unwrap_or("failed");
    ui::failure("Failed", format!("{}: {reason}", result.target));
    if let Some(error) = &result.error {
        ui::error(error);
    }
}

fn write_results(path: &Path, results: &[TargetResult]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json =
        serde_json::to_string_pretty(results).map_err(|e| e.to_string())?;
    fs::write(path, json).map_err(|e| e.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parsed = parse_args(&args);
    let targets = parse_targets(&parsed.target_args).unwrap_or_else(|e| {
        ui::error(e);
        exit(2);
    });
    if targets.is_empty() {
        ui::error("no targets named; nothing would be verified");
        exit(1);
    }

    let results: Vec<TargetResult> = targets
        .iter()
        .map(|t| {
            let result = run_target(t, parsed.options);
            report(&result);
            result
        })
        .collect();

    if let Err(e) = write_results(&parsed.out, &results) {
        ui::error(format!("failed to write {}: {e}", parsed.out.display()));
        exit(1);
    }
    ui::status("Writing", parsed.out.display());

    let failed = results.iter().filter(|r| !r.passed).count();
    if failed > 0 {
        ui::failure(
            "Failed",
            format!("{failed} of {} target(s)", results.len()),
        );
        exit(1);
    }
    ui::status("Finished", format!("{} target(s) passed", results.len()));
}
