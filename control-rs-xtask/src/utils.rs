//! Utility functions and structures for `xtask`'s build and test-execution tasks.
//! Contains formatting, git information collection and report generation helper functions.

use control_rs_ets::comms::TestState;
use regex::Regex;
use std::env;
use std::fs;
use std::process::Command;

/// Summary of cargo tarpaulin test execution and line coverage.
#[derive(Debug, Clone)]
pub struct TarpaulinSummary {
    /// Number of tests passed.
    pub passed: usize,
    /// Number of tests failed.
    pub failed: usize,
    /// Number of tests ignored.
    pub ignored: usize,
    /// The formatted coverage percentage (e.g., "85.20").
    pub coverage_percent: String,
    /// Number of lines covered.
    pub covered_lines: usize,
    /// Total number of coverable lines.
    pub total_lines: usize,
}

/// Results of a CI-driven ETS or virtual ETS run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HeadlessTestResult {
    /// The name of the test suite.
    pub suite_name: String,
    /// The name of the test case.
    pub test_name: String,
    /// The final execution state of the test (Passed or Failed).
    pub state: TestState,
    /// CPU cycles consumed by the test if successful.
    pub cycles: Option<u64>,
    /// Duration of the test in microseconds if successful.
    pub time_us: Option<u64>,
    /// Stack high-water mark in bytes if successful.
    pub stack_peak: Option<u32>,
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

fn format_number(val: u64) -> String {
    let s = val.to_string();
    let bytes = s.as_bytes();
    let mut result = String::new();
    let len = bytes.len();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

fn format_ets_rows(results: &[&HeadlessTestResult]) -> String {
    let re = Regex::new(r"^(.*) \(([^)]+)\)$").unwrap();

    let mut targets = Vec::new();
    let mut suites = Vec::new();
    let mut suite_tests: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    let mut grouped: std::collections::HashMap<
        (&str, &str, &str),
        &HeadlessTestResult,
    > = std::collections::HashMap::new();

    for r in results {
        let (base_suite, target) =
            if let Some(caps) = re.captures(&r.suite_name) {
                (caps.get(1).unwrap().as_str(), caps.get(2).unwrap().as_str())
            } else {
                (r.suite_name.as_str(), "")
            };

        if !target.is_empty() && !targets.contains(&target) {
            targets.push(target);
        }

        if !suites.contains(&base_suite) {
            suites.push(base_suite);
        }

        let tests = suite_tests.entry(base_suite).or_default();
        if !tests.contains(&r.test_name.as_str()) {
            tests.push(r.test_name.as_str());
        }

        grouped.insert((base_suite, r.test_name.as_str(), target), r);
    }

    if targets.is_empty() {
        targets.push("");
    }

    // Determine pruning criteria for each target
    let mut prune_cycles = std::collections::HashMap::new();
    let mut prune_time = std::collections::HashMap::new();
    let mut prune_stack = std::collections::HashMap::new();

    for &t in &targets {
        let target_results: Vec<&&HeadlessTestResult> = results
            .iter()
            .filter(|r| {
                let target_name = if let Some(caps) = re.captures(&r.suite_name)
                {
                    caps.get(2).unwrap().as_str()
                } else {
                    ""
                };
                target_name == t
            })
            .collect();

        let has_cycles = target_results
            .iter()
            .any(|r| r.cycles.is_some_and(|c| c > 0));
        let has_time = target_results
            .iter()
            .any(|r| r.time_us.is_some_and(|time| time > 0));
        let has_stack = target_results
            .iter()
            .any(|r| r.stack_peak.is_some_and(|s| s > 0));

        prune_cycles.insert(t, !has_cycles);
        prune_time.insert(t, !has_time);
        prune_stack.insert(t, !has_stack);
    }

    let format_cell = |r: &HeadlessTestResult, target: &str| -> String {
        let status = match r.state {
            TestState::Passed => "Pass",
            _ => "Fail",
        };
        let mut parts = Vec::new();
        parts.push(status.to_string());
        if !prune_time.get(target).copied().unwrap_or(false) {
            parts.push(r.time_us.map_or("N/A".to_string(), |t| {
                format!("{}us", format_number(t))
            }));
        }
        if !prune_stack.get(target).copied().unwrap_or(false) {
            parts.push(r.stack_peak.map_or("N/A".to_string(), |s| {
                format!("{}B", format_number(u64::from(s)))
            }));
        }
        if !prune_cycles.get(target).copied().unwrap_or(false) {
            parts.push(r.cycles.map_or("N/A".to_string(), |c| {
                format!("{}c", format_number(c))
            }));
        }
        parts.join(" / ")
    };

    let mut s = String::new();

    for suite in suites {
        if !s.is_empty() {
            s.push('\n');
        }
        s.push_str(&format!("### Suite: `{}`\n", suite));

        // Generate table header
        s.push_str("| Test");
        for t in &targets {
            let p_cycles = prune_cycles.get(t).copied().unwrap_or(false);
            let p_time = prune_time.get(t).copied().unwrap_or(false);
            let p_stack = prune_stack.get(t).copied().unwrap_or(false);

            let mut cols = Vec::new();
            cols.push("Status");
            if !p_time {
                cols.push("Time");
            }
            if !p_stack {
                cols.push("Stack");
            }
            if !p_cycles {
                cols.push("Cycles");
            }

            let metrics_label = cols.join(" / ");
            if t.is_empty() {
                s.push_str(&format!(" | {}", metrics_label));
            } else {
                s.push_str(&format!(" | {} ({})", t, metrics_label));
            }
        }
        s.push_str(" |\n| :---");
        for _ in &targets {
            s.push_str(" | :---");
        }
        s.push_str(" |\n");

        // Generate rows
        if let Some(tests) = suite_tests.get(suite) {
            for test in tests {
                s.push_str(&format!("| {}", test));
                for t in &targets {
                    if let Some(r) = grouped.get(&(suite, *test, *t)) {
                        s.push_str(&format!(" | {}", format_cell(r, t)));
                    } else {
                        s.push_str(" | N/A");
                    }
                }
                s.push_str(" |\n");
            }
        }
    }

    s
}

/// Helper function to format the high-level summary of CI ETS results.
pub fn format_ets_summary(results: &[HeadlessTestResult]) -> String {
    let total = results.len();
    let failed_tests: Vec<&HeadlessTestResult> = results
        .iter()
        .filter(|r| !matches!(r.state, TestState::Passed))
        .collect();
    let succeeded = total - failed_tests.len();

    let mut s = String::new();
    s.push_str("### ETS Results\n\n");
    s.push_str(
        "CI frontend; suite names record the backend (virtual ETS vs ETS).\n\n",
    );
    s.push_str(&format!("**Tests Run:** {}\n", total));
    s.push_str(&format!("**Succeeded:** {}\n\n", succeeded));

    if !failed_tests.is_empty() {
        s.push_str("#### Failing Tests\n\n");
        let to_show = failed_tests.len().min(10);
        if failed_tests.len() > 10 {
            s.push_str(&format!(
                "*Showing first 10 of {} failing tests:*\n\n",
                failed_tests.len()
            ));
        }
        s.push_str(&format_ets_rows(&failed_tests[..to_show]));
        s.push('\n');
    }
    s
}

/// Helper function to format the CI ETS results table.
pub fn format_ets_results_table(results: &[HeadlessTestResult]) -> String {
    let refs: Vec<&HeadlessTestResult> = results.iter().collect();
    format_ets_rows(&refs)
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
            .map(|s| s.lines().next().unwrap_or("").trim().to_string())
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
    section.push_str(&format!(
        "| qemu-system-arm | {} |\n",
        get_version("qemu-system-arm", &["--version"])
    ));
    section.push_str(&format!(
        "| qemu-system-riscv32 | {} |\n",
        get_version("qemu-system-riscv32", &["--version"])
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

    let has_origin_main = Command::new("git")
        .args(["rev-parse", "--verify", "origin/main"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let mut git_show = String::new();
    let mut target_used = "HEAD";

    if has_origin_main {
        if let Ok(output) = Command::new("git")
            .args(["show", "origin/main..HEAD", "--stat"])
            .output()
        {
            if output.status.success() {
                if let Ok(stdout_str) = String::from_utf8(output.stdout) {
                    let trimmed = stdout_str.trim().to_string();
                    if !trimmed.is_empty() {
                        git_show = trimmed;
                        target_used = "origin/main..HEAD";
                    }
                }
            }
        }
    }

    if git_show.is_empty() {
        git_show = Command::new("git")
            .args(["show", "HEAD", "--stat"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Not available".to_string());
    }

    s.push_str(&format!(
        "<details>\n<summary>Commit Stat Summary (git show {} --stat)</summary>\n\n```text\n",
        target_used
    ));
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
    ets_results: &Result<Vec<HeadlessTestResult>, String>,
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

    match ets_results {
        Ok(results) => {
            report.push_str(&format_ets_summary(results));

            let artifact_link = if let (Ok(repo), Ok(run_id)) = (
                std::env::var("GITHUB_REPOSITORY"),
                std::env::var("GITHUB_RUN_ID"),
            ) {
                format!(
                    "Detailed ETS results are available in the [ets-results.json](https://github.com/{}/actions/runs/{}) artifact.\n\n",
                    repo, run_id
                )
            } else {
                "Detailed ETS results have been written to `ets-results.json`.\n\n".to_string()
            };
            report.push_str(&artifact_link);
        }
        Err(e) => {
            report.push_str(&format!(
                "### ETS Results\n\n**ERROR**: Failed to run CI ETS: {}\n\n",
                e
            ));
        }
    }

    report.push_str("<details>\n<summary>Detailed Logs</summary>\n\n");
    report.push_str(&collect_system_info());

    if let Ok(results) = ets_results {
        report.push_str("\n<details>\n<summary>ETS Results Log</summary>\n\n");
        report.push_str(&format_ets_results_table(results));
        report.push_str("\n</details>\n");
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use control_rs_ets::comms::TestState;

    #[test]
    fn test_format_ets_rows_passed() {
        let results = [
            HeadlessTestResult {
                suite_name: "SuiteA".to_string(),
                test_name: "Test1".to_string(),
                state: TestState::Passed,
                cycles: Some(100),
                time_us: Some(10),
                stack_peak: Some(256),
            },
            HeadlessTestResult {
                suite_name: "SuiteA".to_string(),
                test_name: "Test2".to_string(),
                state: TestState::Failed,
                cycles: None,
                time_us: None,
                stack_peak: None,
            },
        ];
        let refs: Vec<&HeadlessTestResult> = results.iter().collect();
        let formatted = format_ets_rows(&refs);
        assert!(formatted.contains("SuiteA"));
        assert!(formatted.contains("Test1"));
        assert!(formatted.contains("Pass"));
        assert!(formatted.contains("100c"));
        assert!(formatted.contains("10us"));
        assert!(formatted.contains("256B"));
        assert!(formatted.contains("Fail"));
        assert!(formatted.contains("N/A"));
    }

    #[test]
    fn test_format_ets_summary_all_passed() {
        let results = vec![HeadlessTestResult {
            suite_name: "SuiteA".to_string(),
            test_name: "Test1".to_string(),
            state: TestState::Passed,
            cycles: Some(100),
            time_us: Some(10),
            stack_peak: Some(256),
        }];
        let summary = format_ets_summary(&results);
        assert!(summary.contains("### ETS Results"));
        assert!(summary.contains("**Tests Run:** 1"));
        assert!(summary.contains("**Succeeded:** 1"));
        assert!(!summary.contains("#### Failing Tests"));
    }

    #[test]
    fn test_format_ets_summary_some_failed() {
        let mut results = Vec::new();
        for i in 1..=12 {
            results.push(HeadlessTestResult {
                suite_name: "SuiteA".to_string(),
                test_name: format!("Test{}", i),
                state: if i <= 2 {
                    TestState::Passed
                } else {
                    TestState::Failed
                },
                cycles: None,
                time_us: None,
                stack_peak: None,
            });
        }
        let summary = format_ets_summary(&results);
        assert!(summary.contains("**Tests Run:** 12"));
        assert!(summary.contains("**Succeeded:** 2"));
        assert!(summary.contains("#### Failing Tests"));
        assert!(!summary.contains("*Showing first 10 of 10 failing tests:*")); // Exactly 10 failed, so no truncation text

        // Add one more failure to make it 11 failed
        results.push(HeadlessTestResult {
            suite_name: "SuiteA".to_string(),
            test_name: "Test13".to_string(),
            state: TestState::Failed,
            cycles: None,
            time_us: None,
            stack_peak: None,
        });
        let summary2 = format_ets_summary(&results);
        assert!(summary2.contains("*Showing first 10 of 11 failing tests:*"));
    }

    #[test]
    fn test_format_ets_rows_multiple_targets() {
        let results = [
            HeadlessTestResult {
                suite_name: "SuiteA (ARM)".to_string(),
                test_name: "Test1".to_string(),
                state: TestState::Passed,
                cycles: Some(100),
                time_us: Some(10),
                stack_peak: Some(256),
            },
            HeadlessTestResult {
                suite_name: "SuiteA (RISC-V)".to_string(),
                test_name: "Test1".to_string(),
                state: TestState::Passed,
                cycles: Some(200),
                time_us: Some(20),
                stack_peak: Some(512),
            },
        ];
        let refs: Vec<&HeadlessTestResult> = results.iter().collect();
        let formatted = format_ets_rows(&refs);
        assert!(formatted.contains("SuiteA"));
        assert!(formatted.contains("Test1"));
        assert!(formatted.contains("ARM (Status / Time / Stack / Cycles)"));
        assert!(formatted.contains("RISC-V (Status / Time / Stack / Cycles)"));
        assert!(formatted.contains("100c"));
        assert!(formatted.contains("200c"));
        assert!(formatted.contains("256B"));
        assert!(formatted.contains("512B"));
        let occurrences = formatted.matches("| Test1 | ").count();
        assert_eq!(occurrences, 1);
    }

    #[test]
    fn test_formatting_helpers() {
        let issue_sum = format_issue_summary(1, 2);
        assert!(issue_sum.contains("### Issue Summary"));
        assert!(issue_sum.contains('1'));
        assert!(issue_sum.contains('2'));

        let perf = format_performance(1.23, 4.56);
        assert!(perf.contains("### CI Performance"));
        assert!(perf.contains("1.23s"));
        assert!(perf.contains("4.56s"));

        let test_sum = format_test_summary(10, 0, 1);
        assert!(test_sum.contains("### Test Summary"));
        assert!(test_sum.contains("10"));
        assert!(test_sum.contains('0'));
        assert!(test_sum.contains('1'));

        let cov_sum = format_coverage_summary("85.20", 852, 1000);
        assert!(cov_sum.contains("### Coverage Summary: `85.20%`"));
        assert!(cov_sum.contains("852 / 1000"));

        let number = format_number(1234567);
        assert_eq!(number, "1,234,567");
    }

    #[test]
    fn test_collect_info_helpers() {
        let sys_info = collect_system_info();
        assert!(sys_info.contains("### System Information"));
        assert!(sys_info.contains("rustc"));

        let git_info = collect_git_info();
        assert!(git_info.contains("### Git Information"));
        assert!(git_info.contains("Branch"));
        assert!(git_info.contains("Commit"));
    }

    #[test]
    fn test_build_report_and_save() {
        let tarp_summary = TarpaulinSummary {
            passed: 5,
            failed: 0,
            ignored: 0,
            coverage_percent: "75.00".to_string(),
            covered_lines: 75,
            total_lines: 100,
        };
        let ets_results = Ok(std::vec![HeadlessTestResult {
            suite_name: "SuiteB".to_string(),
            test_name: "TestB".to_string(),
            state: TestState::Passed,
            cycles: Some(100),
            time_us: Some(10),
            stack_peak: Some(256),
        }]);

        let report = build_report(
            0,
            "",
            0,
            "",
            &tarp_summary,
            "tarpaulin logs",
            &ets_results,
            1.0,
            2.0,
        );
        assert!(report.contains("## `control-rs` Quality Report"));
        assert!(report.contains("### CI Performance"));
        assert!(report.contains("tarpaulin logs"));

        let path = "temp_test_report.md";
        let save_res = save_report(path, &report);
        assert!(save_res.is_ok());
        let read_res = std::fs::read_to_string(path).unwrap();
        assert_eq!(read_res, report);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_format_helpers_and_targeted_ets_rows() {
        assert!(format_issue_summary(2, 3).contains("Formatting"));
        assert!(format_performance(1.5, 2.25).contains("1.50s"));
        assert!(format_test_summary(4, 1, 0).contains("Passed"));
        assert!(format_coverage_summary("91.00", 91, 100).contains("91 / 100"));

        let results = [
            HeadlessTestResult {
                suite_name: "SuiteA (ARM HF)".to_string(),
                test_name: "Test1".to_string(),
                state: TestState::Passed,
                cycles: Some(1_234_567),
                time_us: Some(10),
                stack_peak: Some(256),
            },
            HeadlessTestResult {
                suite_name: "SuiteA (RISC-V 32)".to_string(),
                test_name: "Test1".to_string(),
                state: TestState::Failed,
                cycles: None,
                time_us: None,
                stack_peak: None,
            },
        ];
        let refs: Vec<&HeadlessTestResult> = results.iter().collect();
        let formatted = format_ets_rows(&refs);
        assert!(formatted.contains("ARM HF"));
        assert!(formatted.contains("1,234,567"));
    }
}
