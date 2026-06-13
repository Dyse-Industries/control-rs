//! Host runner binary task entry points.
//! Implements subcommands for CI linting, QEMU runner execution, and interactive HIL TUI.

// Clippy configuration for control-rs-xtask binary.
#![deny(
    unused,
    clippy::all,
    clippy::todo,
    clippy::style,
    clippy::pedantic,
    clippy::suspicious,
    clippy::complexity,
    clippy::unimplemented,
    clippy::big_endian_bytes,
    clippy::shadow_unrelated,
    clippy::large_stack_arrays,
    clippy::empty_structs_with_brackets,
    missing_docs
)]
#![warn(rust_2018_idioms, clippy::complexity)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::shadow_unrelated,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::wildcard_imports,
    clippy::similar_names,
    clippy::cognitive_complexity,
    clippy::match_wildcard_for_single_variants,
    clippy::type_complexity,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::arbitrary_source_item_ordering,
    clippy::multiple_crate_versions,
    clippy::equatable_if_let,
    clippy::nursery,
    clippy::cargo,
    clippy::collapsible_if,
    clippy::single_match,
    clippy::format_push_string,
    clippy::map_unwrap_or,
    clippy::if_not_else,
    clippy::unreadable_literal,
    clippy::redundant_closure_for_method_calls,
    clippy::single_match_else,
    clippy::items_after_statements
)]

use std::env;
use std::process::exit;
use std::time::Instant;

use control_rs_xtask::bridge;

mod tasks;
mod utils;

/// Main entrypoint. Parses arguments and routes execution to the correct task runner.
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage_and_exit();
    }

    match args[1].as_str() {
        "ci" => {
            let target_str = args
                .get(2)
                .map(std::string::String::as_str)
                .unwrap_or("qemu");
            let target = match target_str {
                "qemu" => bridge::Target::Qemu,
                "teensy" => {
                    let port = args
                        .get(3)
                        .cloned()
                        .unwrap_or_else(|| "/dev/ttyACM0".to_string());
                    let baud = args
                        .get(4)
                        .and_then(|b| b.parse().ok())
                        .unwrap_or(115200);
                    bridge::Target::Serial { port, baud }
                }
                _ => {
                    eprintln!("Unknown target: {target_str}");
                    exit(1);
                }
            };
            run_ci(&target);
        }
        "qemu" => {
            tasks::run_qemu();
        }
        "hil-tui" => {
            let target_str = args
                .get(2)
                .map(std::string::String::as_str)
                .unwrap_or("qemu");
            let target = match target_str {
                "qemu" => bridge::Target::Qemu,
                "teensy" => {
                    let port = args
                        .get(3)
                        .cloned()
                        .unwrap_or_else(|| "/dev/teensy".to_string());
                    let baud = args
                        .get(4)
                        .and_then(|b| b.parse().ok())
                        .unwrap_or(115200);
                    bridge::Target::Serial { port, baud }
                }
                _ => {
                    eprintln!("Unknown target: {target_str}");
                    exit(1);
                }
            };
            tasks::run_hil_tui(&target);
        }
        _ => {
            print_usage_and_exit();
        }
    }
}

/// Prints usage instructions to stderr and exits with error code 1.
fn print_usage_and_exit() -> ! {
    eprintln!("Usage: cargo control-rs-xtask <task> [target] [port] [baud]");
    eprintln!("Tasks: ci, qemu, hil-tui");
    eprintln!("Targets: qemu (default), teensy");
    exit(1);
}

/// Executes the full CI validation pipeline, including formatting, clippy, test coverage, and SIL tests.
fn run_ci(target: &bridge::Target) {
    unsafe {
        env::set_var("RUST_BACKTRACE", "full");
        env::set_var("CARGO_TERM_COLOR", "never");
    }

    let mut ci_success = true;

    // Run format and clippy checks
    let start_lint = Instant::now();
    let (fmt_res, fmt_str) = tasks::run_fmt();
    let (clippy_res, clippy_str) = tasks::run_clippy();
    let lint_time = start_lint.elapsed().as_secs_f32();

    let fmt_errors = match fmt_res {
        Ok(()) => 0,
        Err(e) => {
            ci_success = false;
            e
        }
    };
    let clippy_errors = match clippy_res {
        Ok(()) => 0,
        Err(e) => {
            ci_success = false;
            e
        }
    };

    // Run tarpaulin check
    let start_test = Instant::now();
    let (tarp_res, tarp_str) = tasks::run_tarpaulin();
    let test_time = start_test.elapsed().as_secs_f32();

    let tarp_summary = match tarp_res {
        Ok(s) => s,
        Err(()) => {
            ci_success = false;
            utils::TarpaulinSummary {
                passed: 0,
                failed: 0,
                ignored: 0,
                coverage_percent: "0.00".to_string(),
                covered_lines: 0,
                total_lines: 0,
            }
        }
    };

    // Run SIL tests
    let start_sil = Instant::now();
    let (sil_res, _sil_logs) = tasks::run_headless_sil(target);
    let sil_time = start_sil.elapsed().as_secs_f32();

    match &sil_res {
        Ok(results) => {
            let all_passed = results.iter().all(|r| {
                matches!(r.state, control_rs_hil::comms::TestState::Passed)
            });
            if !all_passed {
                ci_success = false;
            }
        }
        Err(_) => {
            ci_success = false;
        }
    }

    println!("Headless SIL tests completed in {sil_time:.2}s.");

    // Generate markdown report
    let report_content = utils::build_report(
        fmt_errors,
        &fmt_str,
        clippy_errors,
        &clippy_str,
        &tarp_summary,
        &tarp_str,
        &sil_res,
        lint_time,
        test_time,
    );

    // Save report
    if let Err(e) = utils::save_report("ci-report.md", &report_content) {
        eprintln!("Failed to write ci-report.md: {e}");
        exit(1);
    }

    if !ci_success {
        println!("CI pipeline failed. Check ci-report.md for details.");
        exit(1);
    } else {
        println!("CI pipeline passed. Report written to ci-report.md.");
    }
}
