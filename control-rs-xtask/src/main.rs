use control_rs_xtask::ci::common::TaskResult;
use control_rs_xtask::ci::{binutil, bloat, common, coverage, lint, test};
use std::env;
use std::fs;
use std::process::exit;

fn main() {
    let task = env::args().nth(1);
    match task.as_deref() {
        Some("ci") => run_ci(),
        Some("hil") => run_hil(),
        Some("bloat-check") => {
            let target_examples =
                common::collect_examples(&["thumbv7em-none-eabihf"]);
            if target_examples
                .values()
                .any(|examples| !examples.is_empty())
            {
                bloat::run(&target_examples);
            } else {
                println!(
                    "No 'examples' found for any target. Skipping bloat check."
                );
            }
        }
        Some("binutil-check") => {
            let target_examples =
                common::collect_examples(&["thumbv7em-none-eabihf"]);
            if target_examples
                .values()
                .any(|examples| !examples.is_empty())
            {
                binutil::run(&target_examples);
            } else {
                println!(
                    "No 'examples' found for any target. Skipping binutils check."
                );
            }
        }
        _ => {
            eprintln!("Usage: cargo xtask <command>");
            eprintln!("Commands:");
            eprintln!(
                "  ci            Runs formatting, clippy, testing, coverage, bloat, and binutil reports."
            );
            eprintln!(
                "  hil           Interactive runner to execute tests and benchmarks on target hardware."
            );
            eprintln!(
                "  bloat-check   Checks for binary size bloat using cargo-bloat."
            );
            eprintln!(
                "  binutil-check Checks for detailed binary size metrics using cargo-binutils."
            );
            exit(1);
        }
    }
}

fn run_hil() {
    let args: Vec<String> = env::args().skip(2).collect();
    println!("Running HIL with args: {:?}", args);
    // TODO: implement HIL runner logic here
    control_rs_xtask::hil::run(args);
}

fn run_ci() {
    unsafe {
        env::set_var("RUST_BACKTRACE", "full");
        env::set_var("CARGO_TERM_COLOR", "never");
    }

    let mut report = String::from(
        "## `control-rs` Quality Report\n\n\
    _This file was automatically generated, do not edit it._\n\n",
    );

    let lint_res = lint::run();
    let test_res = test::run();
    let cov_res = coverage::run();

    let target_examples = common::collect_examples(&["thumbv7em-none-eabihf"]);
    let has_examples = target_examples
        .values()
        .any(|examples| !examples.is_empty());

    if !has_examples {
        println!("No 'examples' found for any target.");
    }

    let bloat_res = if has_examples {
        bloat::run(&target_examples)
    } else {
        TaskResult {
            success: true,
            report: String::from(
                "\n### Bloat Check\n\nNo examples found to analyze.\n",
            ),
            ..Default::default()
        }
    };

    let binutil_res = if has_examples {
        binutil::run(&target_examples)
    } else {
        TaskResult {
            success: true,
            report: String::from(
                "\n### Binutils Size\n\nNo examples found to analyze.\n",
            ),
            ..Default::default()
        }
    };

    let ci_success = lint_res.success
        && test_res.success
        && cov_res.success
        && bloat_res.success
        && binutil_res.success;

    report.push_str(&crate::section_header!(3, "Issue Summary"));
    report.push_str(&crate::table_header!(
        ["Category", "Issues Found"],
        [":---", ":---"]
    ));
    report.push_str(&crate::table_row!("Formatting", lint_res.fmt_errors));
    report
        .push_str(&crate::table_row!("Clippy Errors", lint_res.clippy_errors));
    report.push('\n');

    report.push_str(&crate::section_header!(3, "CI Performance"));
    report.push_str(&crate::table_header!(
        ["Task", "Duration"],
        [":---", ":---"]
    ));
    report.push_str(&crate::table_row!(
        "Lint & Format",
        format!("{:.2}s", lint_res.time)
    ));
    report
        .push_str(&crate::table_row!("Test", format!("{:.2}s", test_res.time)));
    report.push_str(&crate::table_row!(
        "Coverage",
        format!("{:.2}s", cov_res.time)
    ));
    report.push_str(&crate::table_row!(
        "Bloat Check",
        format!("{:.2}s", bloat_res.time)
    ));
    report.push_str(&crate::table_row!(
        "Binutils Check",
        format!("{:.2}s", binutil_res.time)
    ));
    report.push('\n');

    report.push_str(&crate::section_header!(3, "Test Summary"));
    report
        .push_str(&crate::table_header!(["Result", "Count"], [":---", ":---"]));
    report.push_str(&crate::table_row!("Passed", test_res.passed));
    report.push_str(&crate::table_row!("Failed", test_res.failed));
    report.push_str(&crate::table_row!("Ignored / Errored", test_res.ignored));
    report.push('\n');

    report.push_str(&format!("### Coverage Summary: `{}%`\n", cov_res.percent));
    report.push_str(&format!(
        "**Lines Covered:** `{} / {}`\n\n",
        cov_res.cov_lines, cov_res.tot_lines
    ));

    report.push_str(&bloat_res.report);
    report.push_str(&binutil_res.report);

    report.push_str("\n<details>\n<summary>Detailed Logs</summary>\n\n");

    report.push_str(&common::collect_versions());

    if lint_res.fmt_errors > 0 || lint_res.clippy_errors > 0 {
        report.push_str(&crate::section_header!(3, "Cargo fmt & Clippy"));
        report.push_str("```text\n");
        if lint_res.fmt_errors > 0 {
            report.push_str(&lint_res.fmt_str);
        }
        if lint_res.clippy_errors > 0 {
            report.push_str(&lint_res.clippy_str);
        }
        report.push_str("```\n\n");
    }

    report.push_str(&crate::section_header!(3, "Cargo Test Output Log"));
    report.push_str(
        "<details>\n<summary>Click to expand test logs</summary>\n\n```text\n",
    );
    report.push_str(&test_res.log_str);
    report.push_str("\n```\n</details>\n");

    report.push_str(&crate::section_header!(3, "Tarpaulin Output Log"));
    report.push_str("<details>\n<summary>Click to expand coverage logs</summary>\n\n```text\n");
    report.push_str(&cov_res.log_str);
    report.push_str("\n```\n</details>\n");

    report.push_str("</details>\n");

    fs::write("ci-report.md", report).expect("Unable to write ci-report.md");

    if !ci_success {
        println!("CI pipeline failed. Check ci-report.md for details.");
        exit(1);
    } else {
        println!("CI pipeline passed. Report written to ci-report.md.");
    }
}