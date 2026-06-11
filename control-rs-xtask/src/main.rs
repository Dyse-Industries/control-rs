use regex::Regex;
use std::env;
use std::fs;
use std::process::{Command, exit};
use std::time::Instant;

fn main() {
    let task = env::args().nth(1);
    match task.as_deref() {
        Some("ci") => run_ci(),
        Some("qemu") => run_qemu(),
        _ => {
            eprintln!("Usage: cargo control-rs-xtask <task>");
            eprintln!("Tasks: ci, qemu");
            exit(1);
        }
    }
}

fn run_qemu() {
    let clean_status = Command::new("cargo")
        .args([
            "clean",
            "--package",
            "control-rs-hil",
            "--target",
            "thumbv7em-none-eabihf",
            "--profile",
            "qemu",
        ])
        .status()
        .expect("Failed to clean QEMU Sil kernel.");

    if !clean_status.success() {
        eprintln!("Failed to clean QEMU image.");
        exit(1);
    }

    println!("Building QEMU image...");
    let run_status = Command::new("cargo")
        .env("CARGO_TARGET_THUMBV7EM_NONE_EABIHF_RUNNER", "qemu-system-arm -cpu cortex-m7 -machine mps2-an500 -nographic -semihosting-config enable=on,target=native -kernel")
        .env("CARGO_TARGET_THUMBV7EM_NONE_EABIHF_RUSTFLAGS", "-C link-arg=-Tlink.x -C link-arg=-Thil_suites.x")
        .args([
            "run",
            "--package",
            "control-rs-hil",
            "--bin",
            "control-rs-qemu-arm",
            "--target",
            "thumbv7em-none-eabihf",
            "--profile",
            "qemu",
        ])
        .status()
        .expect("Failed to run QEMU Sil kernel.");

    if !run_status.success() {
        eprintln!("QEMU exited with an error.");
        exit(1);
    }
    println!("QEMU run finished successfully.");
}

fn run_ci() {
    unsafe {
        env::set_var("RUST_BACKTRACE", "full");
        env::set_var("CARGO_TERM_COLOR", "never");
    }
    let mut report = String::from("## `control-rs` Quality Report\n\n");
    let mut ci_success = true;

    let start_lint = Instant::now();

    let ansi_escape = Regex::new(r"\x1B\[[0-9;]*[mK]|\x1B\(B").unwrap();

    let fmt_output = Command::new("cargo")
        .args(["fmt", "--all"])
        .output()
        .expect("Failed to run cargo fmt");
    let fmt_str = String::from_utf8_lossy(&fmt_output.stdout);
    let fmt_str = ansi_escape.replace_all(&fmt_str, "");
    let fmt_errors = fmt_str.matches("Diff in").count();

    let clippy_output = Command::new("cargo")
        .args(["clippy", "--", "-D", "warnings"])
        .output()
        .expect("Failed to run cargo clippy");
    let clippy_str = String::from_utf8_lossy(&clippy_output.stderr);
    let clippy_str = ansi_escape.replace_all(&clippy_str, "");
    let clippy_errors = clippy_str.matches("error:").count();

    let lint_time = start_lint.elapsed().as_secs_f32();

    if !fmt_output.status.success() || !clippy_output.status.success() {
        ci_success = false;
    }

    report.push_str("### Issue Summary\n");
    report.push_str("| Category | Issues Found |\n| :--- | :--- |\n");
    report.push_str(&format!("| Formatting | {} |\n", fmt_errors));
    report.push_str(&format!("| Clippy Errors | {} |\n\n", clippy_errors));

    let start_test = Instant::now();
    let tarpaulin_output = Command::new("cargo")
        .args([
            "tarpaulin",
            "--verbose",
            "--color",
            "never",
            "--out",
            "Html",
        ])
        .output()
        .expect("Failed to run cargo tarpaulin");

    // Tarpaulin mixes stdout and stderr, combine them for parsing
    let tarp_str = format!(
        "{}\n{}",
        String::from_utf8_lossy(&tarpaulin_output.stderr),
        String::from_utf8_lossy(&tarpaulin_output.stdout)
    );

    let test_time = start_test.elapsed().as_secs_f32();

    if !tarpaulin_output.status.success() {
        ci_success = false;
    }

    let re_passed = Regex::new(r"(\d+) passed").unwrap();
    let re_failed = Regex::new(r"(\d+) failed").unwrap();
    let re_ignored = Regex::new(r"(\d+) ignored").unwrap();
    let re_coverage =
        Regex::new(r"(\d+\.\d+)% coverage, (\d+)/(\d+) lines covered").unwrap();

    let passed: usize = re_passed
        .captures_iter(&tarp_str)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .sum();
    let failed: usize = re_failed
        .captures_iter(&tarp_str)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .sum();
    let ignored: usize = re_ignored
        .captures_iter(&tarp_str)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .sum();

    let (percent, cov_lines, tot_lines) =
        if let Some(caps) = re_coverage.captures(&tarp_str) {
            (
                caps[1].to_string(),
                caps[2].to_string(),
                caps[3].to_string(),
            )
        } else {
            ("0.00".to_string(), "0".to_string(), "0".to_string())
        };

    report.push_str("### CI Performance\n");
    report.push_str("| Task | Duration |\n| :--- | :--- |\n");
    report.push_str(&format!("| Lint & Format | {:.2}s |\n", lint_time));
    report.push_str(&format!("| Test & Coverage | {:.2}s |\n\n", test_time));

    report.push_str("### Test Summary\n");
    report.push_str("| Result | Count |\n| :--- | :--- |\n");
    report.push_str(&format!("| Passed | {} |\n", passed));
    report.push_str(&format!("| Failed | {} |\n", failed));
    report.push_str(&format!("| Ignored / Errored | {} |\n\n", ignored));

    report.push_str(&format!("### Coverage Summary: `{}%`\n", percent));
    report.push_str(&format!(
        "**Lines Covered:** `{} / {}`\n\n",
        cov_lines, tot_lines
    ));

    report.push_str("<details>\n<summary>Detailed Logs</summary>\n\n");

    report.push_str(&collect_versions());

    if fmt_errors > 0 || clippy_errors > 0 {
        report.push_str(
            "\n<details>\n<summary>Fmt and Clippy logs</summary>\n\n```text\n",
        );
        if fmt_errors > 0 {
            report.push_str(&fmt_str);
            report.push_str("\n");
        }
        if clippy_errors > 0 {
            report.push_str(&clippy_str);
            report.push_str("\n");
        }
        report.push_str("\n```\n\n</details>\n");
    }

    report.push_str(
        "\n<details>\n<summary>Tarpaulin Output Log</summary>\n\n```text\n",
    );
    report.push_str(&tarp_str);
    report.push_str("\n```\n\n</details>\n");

    report.push_str("</details>\n");

    fs::write("ci-report.md", report).expect("Unable to write ci-report.md");

    if !ci_success {
        println!("CI pipeline failed. Check ci-report.md for details.");
        exit(1);
    } else {
        println!("CI pipeline passed. Report written to ci-report.md.");
    }
}

fn collect_versions() -> String {
    let mut section = String::from("\n### System Information\n\n");
    section.push_str("| Component | Version |\n| :--- | :--- |\n");

    let os_info = format!("{} {}", env::consts::OS, env::consts::ARCH);
    section.push_str(&format!("| OS | {} |\n", os_info));

    fn get_version(cmd: &str, args: &[&str]) -> String {
        Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Not found".to_string())
    }

    section.push_str(&format!(
        "| rustc | {} |\n",
        get_version("rustc", &["--version"])
    ));
    section.push_str(&format!(
        "| cargo | {} |\n",
        get_version("cargo", &["--version"])
    ));
    section.push_str(&format!(
        "| rustfmt | {} |\n",
        get_version("cargo", &["fmt", "--version"])
    ));
    section.push_str(&format!(
        "| clippy | {} |\n",
        get_version("cargo", &["clippy", "--version"])
    ));
    section.push_str(&format!(
        "| tarpaulin | {} |\n",
        get_version("cargo", &["tarpaulin", "--version"])
    ));

    section.push_str(
        "\n<details>\n<summary>Dependency Tree</summary>\n\n```text\n",
    );
    section.push_str(&get_version("cargo", &["tree"]));
    section.push_str("\n```\n</details>\n");

    section
}
