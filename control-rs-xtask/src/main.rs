//! Entry points for the `xtask` host binary (a `cargo-xtask`-style task runner).
//! Implements subcommands for CI linting, QEMU execution and interactive HIL TUI.

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

/// Main entrypoint. Parses arguments and routes execution to the correct subcommand handler.
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage_and_exit();
    }

    match args[1].as_str() {
        "ci" => match bridge::Target::parse(&args, "all", "/dev/ttyACM0") {
            Ok(Some(target)) => run_ci_single(&target),
            Ok(None) => run_ci_all_qemu(),
            Err(e) => {
                eprintln!("\t{e}");
                exit(1);
            }
        },
        "tui" => match bridge::Target::parse(&args, "arm", "/dev/teensy") {
            Ok(Some(target)) => tasks::run_hil_tui(&target),
            Ok(None) => {
                eprintln!("\tQEMU architecture 'all' is not supported for TUI");
                exit(1);
            }
            Err(e) => {
                eprintln!("\t{e}");
                exit(1);
            }
        },
        "install-hooks" => {
            tasks::install_hooks();
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
    eprintln!("\tTasks: ci, tui [qemu|teensy], install-hooks");
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

    // 1. Run ARM HF SIL tests
    let arm_hf_target = bridge::Target::qemu_arm();
    let start_arm_hf = Instant::now();
    let (arm_hf_sil_res, _arm_hf_logs) =
        tasks::run_headless_sil(&arm_hf_target);
    let arm_hf_sil_time = start_arm_hf.elapsed().as_secs_f32();
    println!("\t* ARM HF SIL tests completed in {arm_hf_sil_time:.2}s.");

    match arm_hf_sil_res {
        Ok(mut results) => {
            for r in &mut results {
                r.suite_name = format!("{} (ARM HF)", r.suite_name);
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
            sil_errors.push(format!("ARM HF failure: {e}"));
        }
    }

    // 2. Run ARM SF SIL tests
    let arm_sf_target = bridge::Target::qemu_arm_soft();
    let start_arm_sf = Instant::now();
    let (arm_sf_sil_res, _arm_sf_logs) =
        tasks::run_headless_sil(&arm_sf_target);
    let arm_sf_sil_time = start_arm_sf.elapsed().as_secs_f32();
    println!("\t* ARM SF SIL tests completed in {arm_sf_sil_time:.2}s.");

    match arm_sf_sil_res {
        Ok(mut results) => {
            for r in &mut results {
                r.suite_name = format!("{} (ARM SF)", r.suite_name);
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
            sil_errors.push(format!("ARM SF failure: {e}"));
        }
    }

    // 3. Run RISC-V 32 SIL tests
    let riscv32_target = bridge::Target::qemu_riscv();
    let start_riscv32 = Instant::now();
    let (riscv32_sil_res, _riscv32_logs) =
        tasks::run_headless_sil(&riscv32_target);
    let riscv32_sil_time = start_riscv32.elapsed().as_secs_f32();
    println!("\t* RISC-V 32 SIL tests completed in {riscv32_sil_time:.2}s.");

    match riscv32_sil_res {
        Ok(mut results) => {
            for r in &mut results {
                r.suite_name = format!("{} (RISC-V 32)", r.suite_name);
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
            sil_errors.push(format!("RISC-V 32 failure: {e}"));
        }
    }

    // 4. Run RISC-V 64 SIL tests
    let riscv64_target = bridge::Target::qemu_riscv64();
    let start_riscv64 = Instant::now();
    let (riscv64_sil_res, _riscv64_logs) =
        tasks::run_headless_sil(&riscv64_target);
    let riscv64_sil_time = start_riscv64.elapsed().as_secs_f32();
    println!("\t* RISC-V 64 SIL tests completed in {riscv64_sil_time:.2}s.");

    match riscv64_sil_res {
        Ok(mut results) => {
            for r in &mut results {
                r.suite_name = format!("{} (RISC-V 64)", r.suite_name);
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
            sil_errors.push(format!("RISC-V 64 failure: {e}"));
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

    // Save sil-results.json if execution succeeded
    if let Ok(ref results) = sil_res {
        if let Ok(json_content) = serde_json::to_string_pretty(results) {
            if let Err(e) =
                utils::save_report("sil-results.json", &json_content)
            {
                eprintln!("\tFailed to write sil-results.json: {e}");
                exit(1);
            }
        }
    }

    if !ci_success {
        println!("CI pipeline failed. Check ci-report.md for details.");
        exit(1);
    }
    println!("CI pipeline passed. Report written to ci-report.md.");
}

/// Executes the full CI validation pipeline for a single target, including formatting, clippy, test coverage and SIL tests.
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
    }
    println!("CI pipeline passed. Results written to ci-report.md.");
}
