//! Headless ETS runner (`cargo ets`).
//!
//! Builds each target's firmware, runs its suites to completion through
//! `control-rs-ets-host`, judges every run with the rule in
//! `control_rs_ci::ets` and writes `ets-results.json`. Exits 0 only when every
//! target passes.
//!
//! ```text
//! cargo ets [--timeout <secs>] [--max-resets <n>] [--out <path>] <targets>... [-- <cargo args>...]
//! cargo ets --timeout 120 qemu all --release
//! ```
//!
//! Target arguments use the `cargo tui` syntax and are passed to
//! `control_rs_ets_host::target::parse_targets`; arguments after `--` reach
//! it verbatim and are forwarded to `cargo build`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Duration;

use control_rs_ci::cli::{USAGE_EXIT, arg_spelling};
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

/// Parsed options, `None` for `--help`, or a usage error.
type ParsedArgs = Result<Option<EtsArgs>, String>;

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
        "{h}Usage:{h:#} {f}cargo ets{f:#} {a}[OPTIONS]{a:#} {a}<TARGETS>...{a:#} [{f}--{f:#} {a}<CARGO_ARGS>...{a:#}]\n\n\
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

/// Value of `flag`, parsed as `T`.
fn parse_number<T: std::str::FromStr>(
    parser: &mut lexopt::Parser,
    flag: &str,
) -> Result<T, String> {
    use lexopt::prelude::*;
    parser
        .value()
        .ok()
        .and_then(|v| v.string().ok())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("{flag} requires a valid value"))
}

/// Parses this binary's options; `Ok(None)` means `--help`.
///
/// Arguments from the first `--` on are forwarded verbatim, `--` included, so
/// `parse_targets` passes them to cargo. Before it, any option this binary
/// does not own is forwarded as the flag followed by its attached value, if
/// any (`--baud=115200` becomes `--baud 115200`).
fn parse_args(args: &[String]) -> ParsedArgs {
    use lexopt::prelude::*;
    let split = args.iter().position(|a| a == "--").unwrap_or(args.len());
    let (own, passthrough) = args.split_at(split);
    let mut parsed = EtsArgs {
        options: RunOptions {
            timeout: Duration::from_secs(120),
            ..RunOptions::default()
        },
        out: PathBuf::from(DEFAULT_OUT),
        // `parse_targets` skips a leading program name.
        target_args: vec!["ets".to_string()],
    };
    let utf8 = |val: std::ffi::OsString| {
        val.string().map_err(|e| format!("Invalid UTF-8: {e:?}"))
    };
    let mut parser = lexopt::Parser::from_iter(own);
    while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
        match arg {
            Short('h') | Long("help") => return Ok(None),
            Long("timeout") => {
                let secs = parse_number::<u64>(&mut parser, "--timeout")?;
                parsed.options.timeout = Duration::from_secs(secs);
            }
            Long("max-resets") => {
                parsed.options.max_resets =
                    parse_number(&mut parser, "--max-resets")?;
            }
            Long("out") => {
                let val = parser
                    .value()
                    .map_err(|_| "--out requires a path".to_string())?;
                parsed.out = PathBuf::from(val);
            }
            Long(_) | Short(_) => {
                parsed.target_args.push(arg_spelling(&arg));
                if let Some(val) = parser.optional_value() {
                    parsed.target_args.push(utf8(val)?);
                }
            }
            Value(val) => parsed.target_args.push(utf8(val)?),
        }
    }
    parsed.target_args.extend_from_slice(passthrough);
    Ok(Some(parsed))
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
    let parsed = match parse_args(&args) {
        Ok(Some(parsed)) => parsed,
        Ok(None) => {
            print_usage();
            exit(0);
        }
        Err(e) => {
            ui::error(e);
            exit(USAGE_EXIT);
        }
    };
    let targets = parse_targets(&parsed.target_args).unwrap_or_else(|e| {
        ui::error(e);
        exit(USAGE_EXIT);
    });
    if targets.is_empty() {
        ui::error("no targets named; nothing would be verified");
        exit(USAGE_EXIT);
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
    if failed != 0 {
        ui::failure(
            "Failed",
            format!("{failed} of {} target(s)", results.len()),
        );
        exit(1);
    }
    ui::status("Finished", format!("{} target(s) passed", results.len()));
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{DEFAULT_OUT, parse_args};

    fn args(list: &[&str]) -> Vec<String> {
        std::iter::once("ets")
            .chain(list.iter().copied())
            .map(ToString::to_string)
            .collect()
    }

    fn targets(list: &[&str]) -> Vec<String> {
        parse_args(&args(list)).unwrap().unwrap().target_args
    }

    #[test]
    fn own_options_are_taken_and_targets_forwarded() {
        let parsed = parse_args(&args(&[
            "--timeout=5",
            "qemu",
            "--max-resets",
            "7",
            "arm",
            "--out",
            "r.json",
            "--release",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(parsed.options.timeout, Duration::from_secs(5));
        assert_eq!(parsed.options.max_resets, 7);
        assert_eq!(parsed.out.to_str(), Some("r.json"));
        assert_eq!(parsed.target_args, ["ets", "qemu", "arm", "--release"]);
    }

    #[test]
    fn defaults_hold_without_options() {
        let parsed = parse_args(&args(&["qemu", "all"])).unwrap().unwrap();
        assert_eq!(parsed.options.timeout, Duration::from_secs(120));
        assert_eq!(parsed.out.to_str(), Some(DEFAULT_OUT));
    }

    #[test]
    fn arguments_after_dash_dash_are_forwarded_verbatim() {
        assert_eq!(
            targets(&["qemu", "arm", "--", "--features", "a,b", "--timeout"]),
            ["ets", "qemu", "arm", "--", "--features", "a,b", "--timeout"]
        );
    }

    #[test]
    fn attached_values_are_split_for_parse_targets() {
        assert_eq!(
            targets(&["teensy", "--baud=115200", "-p/dev/x", "--port", "p"]),
            [
                "ets", "teensy", "--baud", "115200", "-p", "/dev/x", "--port",
                "p"
            ]
        );
    }

    #[test]
    fn help_and_bad_values_are_distinguished() {
        assert!(parse_args(&args(&["--help"])).unwrap().is_none());
        assert!(parse_args(&args(&["-h", "qemu"])).unwrap().is_none());
        for bad in [&["--timeout"][..], &["--timeout", "x"], &["--out"]] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?}");
        }
    }
}
