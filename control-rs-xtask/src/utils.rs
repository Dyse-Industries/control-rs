use control_rs_hil::comms::TestState;
use std::env;
use std::fs;
use std::process::Command;

#[derive(Debug, Clone)]
pub struct TarpaulinSummary {
    pub passed: usize,
    pub failed: usize,
    pub ignored: usize,
    pub coverage_percent: String,
    pub covered_lines: usize,
    pub total_lines: usize,
}

#[derive(Debug, Clone)]
pub struct HeadlessTestResult {
    pub suite_name: String,
    pub test_name: String,
    pub state: TestState,
    pub cycles: Option<u64>,
    pub time_us: Option<u64>,
}

/// Helper function to format the issue summary section.
pub fn format_issue_summary(fmt_errors: usize, clippy_errors: usize) -> String {
    let mut s = String::new();
    s.push_str("### Issue Summary\n");
    s.push_str("| Category | Issues Found |\n| :--- | :--- |\n");
    s.push_str(&format!("| Formatting | {} |\n", fmt_errors));
    s.push_str(&format!("| Clippy Errors | {} |\n\n", clippy_errors));
    s
}

/// Helper function to format the performance section.
pub fn format_performance(lint_time: f32, test_time: f32) -> String {
    let mut s = String::new();
    s.push_str("### CI Performance\n");
    s.push_str("| Task | Duration |\n| :--- | :--- |\n");
    s.push_str(&format!("| Lint & Format | {:.2}s |\n", lint_time));
    s.push_str(&format!("| Test & Coverage | {:.2}s |\n\n", test_time));
    s
}

/// Helper function to format the test summary section.
pub fn format_test_summary(
    passed: usize,
    failed: usize,
    ignored: usize,
) -> String {
    let mut s = String::new();
    s.push_str("### Test Summary\n");
    s.push_str("| Result | Count |\n| :--- | :--- |\n");
    s.push_str(&format!("| Passed | {} |\n", passed));
    s.push_str(&format!("| Failed | {} |\n", failed));
    s.push_str(&format!("| Ignored / Errored | {} |\n\n", ignored));
    s
}

/// Helper function to format the coverage summary section.
pub fn format_coverage_summary(
    percent: &str,
    covered: usize,
    total: usize,
) -> String {
    let mut s = String::new();
    s.push_str(&format!("### Coverage Summary: `{}%`\n", percent));
    s.push_str(&format!("**Lines Covered:** `{} / {}`\n\n", covered, total));
    s
}

/// Helper function to format the headless SIL test results.
pub fn format_sil_results(results: &[HeadlessTestResult]) -> String {
    let mut s = String::new();
    s.push_str("### Headless SIL Test Results\n\n");
    s.push_str("| Suite | Test | Result | Cycles | Time |\n| :--- | :--- | :--- | :--- | :--- |\n");
    for r in results {
        let res_str = match r.state {
            TestState::Passed => "PASSED",
            _ => "FAILED",
        };
        let cyc_str = r.cycles.map_or("N/A".to_string(), |c| c.to_string());
        let time_str =
            r.time_us.map_or("N/A".to_string(), |t| format!("{}us", t));
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            r.suite_name, r.test_name, res_str, cyc_str, time_str
        ));
    }
    s.push('\n');
    s
}

/// Helper function to collect version info of build tools and cargo dependency tree.
pub fn collect_system_info() -> String {
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
    section.push_str(&get_version("cargo", &["tree", "--workspace"]));
    section.push_str("\n```\n</details>\n");

    section
}

/// Helper function to collect Git information (branch and changes stat summary).
pub fn collect_git_info() -> String {
    let mut s = String::new();
    s.push_str("### Git Information\n\n");
    s.push_str("| Property | Value |\n| :--- | :--- |\n");

    let branch = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    let commit_hash = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    s.push_str(&format!("| Branch | `{}` |\n", branch));
    s.push_str(&format!("| Commit | `{}` |\n\n", commit_hash));

    let git_show = Command::new("git")
        .args(["show", &commit_hash, "--stat"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Not available".to_string());

    s.push_str("<details>\n<summary>Commit Stat Summary (git show HEAD --stat)</summary>\n\n```text\n");
    s.push_str(&git_show);
    s.push_str("\n```\n</details>\n\n");
    s
}

/// Assembles the complete Markdown report from the collected task data.
#[allow(clippy::too_many_arguments)]
pub fn build_report(
    fmt_errors: usize,
    fmt_str: &str,
    clippy_errors: usize,
    clippy_str: &str,
    tarp_summary: &TarpaulinSummary,
    tarp_str: &str,
    sil_results: &Result<Vec<HeadlessTestResult>, String>,
    lint_time: f32,
    test_time: f32,
) -> String {
    let mut report = String::from("## `control-rs` Quality Report\n\n");

    report.push_str(&collect_git_info());
    report.push_str(&format_issue_summary(fmt_errors, clippy_errors));
    report.push_str(&format_performance(lint_time, test_time));
    report.push_str(&format_test_summary(
        tarp_summary.passed,
        tarp_summary.failed,
        tarp_summary.ignored,
    ));
    report.push_str(&format_coverage_summary(
        &tarp_summary.coverage_percent,
        tarp_summary.covered_lines,
        tarp_summary.total_lines,
    ));

    match sil_results {
        Ok(results) => {
            report.push_str(&format_sil_results(results));
        }
        Err(e) => {
            report.push_str(&format!(
                "### Headless SIL Test Results\n\n**ERROR**: Failed to run SIL tests: {}\n\n",
                e
            ));
        }
    }

    report.push_str("<details>\n<summary>Detailed Logs</summary>\n\n");
    report.push_str(&collect_system_info());

    if fmt_errors > 0 || clippy_errors > 0 {
        report.push_str(
            "\n<details>\n<summary>Fmt and Clippy logs</summary>\n\n```text\n",
        );
        if fmt_errors > 0 {
            report.push_str(fmt_str);
            report.push('\n');
        }
        if clippy_errors > 0 {
            report.push_str(clippy_str);
            report.push('\n');
        }
        report.push_str("\n```\n\n</details>\n");
    }

    report.push_str(
        "\n<details>\n<summary>Tarpaulin Output Log</summary>\n\n```text\n",
    );
    report.push_str(tarp_str);
    report.push_str("\n```\n\n</details>\n");

    report.push_str("</details>\n");
    report
}

/// Saves the report string to a file.
pub fn save_report(path: &str, content: &str) -> std::io::Result<()> {
    fs::write(path, content)
}
