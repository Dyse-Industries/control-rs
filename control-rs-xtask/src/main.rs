//! Host runner binary task entry points.
//! Implements subcommands for CI linting, QEMU runner execution, and interactive HIL TUI.

#![deny(missing_docs)]
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
            let target_str = args.get(2).map(String::as_str).unwrap_or("qemu");
            match target_str {
                "qemu" => {
                    let arch_str =
                        args.get(3).map(String::as_str).unwrap_or("all");
                    match arch_str {
                        "arm" => {
                            let target = bridge::Target::QemuSemihosting {
                                arch: bridge::QemuArch::Arm,
                            };
                            run_ci_single(&target);
                        }
                        "riscv" | "risc-v" => {
                            let target = bridge::Target::QemuSemihosting {
                                arch: bridge::QemuArch::Riscv,
                            };
                            run_ci_single(&target);
                        }
                        _ => {
                            run_ci_all_qemu();
                        }
                    }
                }
                "teensy" => {
                    let port = args
                        .get(3)
                        .cloned()
                        .unwrap_or_else(|| "/dev/ttyACM0".to_string());
                    let baud = args
                        .get(4)
                        .and_then(|b| b.parse().ok())
                        .unwrap_or(115200);
                    let target = bridge::Target::Serial { port, baud };
                    run_ci_single(&target);
                }
                _ => {
                    eprintln!(
                        "\tUnknown target: {target_str}, (choose from `teensy`, `qemu`)"
                    );
                    exit(1);
                }
            }
        }
        "tui" => {
            let target_str = args
                .get(2)
                .map(std::string::String::as_str)
                .unwrap_or("qemu");
            let target = match target_str {
                "qemu" => {
                    let arch_str = args
                        .get(3)
                        .map(std::string::String::as_str)
                        .unwrap_or("arm");
                    let arch = match arch_str {
                        "riscv" | "risc-v" => bridge::QemuArch::Riscv,
                        _ => bridge::QemuArch::Arm,
                    };
                    bridge::Target::QemuSemihosting { arch }
                }
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
                    eprintln!("\tUnknown target: {target_str}");
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
    eprintln!(
        "Usage: cargo control-rs-xtask <task> [target] [port/arch] [baud]"
    );
    eprintln!("\tTasks: ci, tui [qemu|teensy]");
    eprintln!("\tTargets: qemu [arm|risc-v|all], teensy [port] [baud]");
    exit(1);
}

/// Executes the full CI validation pipeline for both ARM and RISC-V targets, aggregating results.
fn run_ci_all_qemu() {
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

    let mut combined_sil_results = Vec::new();
    let mut sil_errors = Vec::new();

    // 1. Run ARM SIL tests
    let arm_target = bridge::Target::QemuSemihosting {
        arch: bridge::QemuArch::Arm,
    };
    let start_arm = Instant::now();
    let (arm_sil_res, _arm_logs) = tasks::run_headless_sil(&arm_target);
    let arm_sil_time = start_arm.elapsed().as_secs_f32();
    println!("\t* ARM SIL tests completed in {arm_sil_time:.2}s.");

    match arm_sil_res {
        Ok(mut results) => {
            for r in &mut results {
                r.suite_name = format!("{} (ARM)", r.suite_name);
            }
            let all_passed = results.iter().all(|r| {
                matches!(r.state, control_rs_hil::comms::TestState::Passed)
            });
            if !all_passed {
                ci_success = false;
            }
            combined_sil_results.extend(results);
        }
        Err(e) => {
            ci_success = false;
            sil_errors.push(format!("ARM failure: {e}"));
        }
    }

    // 2. Run RISC-V SIL tests
    let riscv_target = bridge::Target::QemuSemihosting {
        arch: bridge::QemuArch::Riscv,
    };
    let start_riscv = Instant::now();
    let (riscv_sil_res, _riscv_logs) = tasks::run_headless_sil(&riscv_target);
    let riscv_sil_time = start_riscv.elapsed().as_secs_f32();
    println!("\t* RISC-V SIL tests completed in {riscv_sil_time:.2}s.");

    match riscv_sil_res {
        Ok(mut results) => {
            for r in &mut results {
                r.suite_name = format!("{} (RISC-V)", r.suite_name);
            }
            let all_passed = results.iter().all(|r| {
                matches!(r.state, control_rs_hil::comms::TestState::Passed)
            });
            if !all_passed {
                ci_success = false;
            }
            combined_sil_results.extend(results);
        }
        Err(e) => {
            ci_success = false;
            sil_errors.push(format!("RISC-V failure: {e}"));
        }
    }

    let sil_res = if sil_errors.is_empty() {
        Ok(combined_sil_results)
    } else {
        Err(sil_errors.join("; "))
    };

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
        eprintln!("\tFailed to write ci-report.md: {e}");
        exit(1);
    }

    if !ci_success {
        println!("CI pipeline failed. Check ci-report.md for details.");
        exit(1);
    } else {
        println!("\tCI pipeline passed. Report written to ci-report.md.");
    }
}

/// Executes the full CI validation pipeline for a single target, including formatting, clippy, test coverage, and SIL tests.
fn run_ci_single(target: &bridge::Target) {
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

    println!("\tHeadless SIL tests completed in {sil_time:.2}s.");

    // Generate Markdown report
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
        eprintln!("\tFailed to write ci-report.md: {e}");
        exit(1);
    }

    if !ci_success {
        println!("CI pipeline failed. Check ci-report.md for details.");
        exit(1);
    } else {
        println!("CI pipeline passed. Report written to ci-report.md.");
    }
}
