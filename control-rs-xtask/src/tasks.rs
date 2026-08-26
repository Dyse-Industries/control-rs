//! Modules defining the test-execution and compilation tasks executed by `xtask`.
//! Includes tasks for formatting check, clippy checks, coverage tracking with tarpaulin,
//! and running CI (virtual ETS / ETS) and interactive TUI tests.

use regex::Regex;
use std::fs;
use std::process::{Command, exit};
use std::time::{Duration, Instant};

use crate::utils::{HeadlessTestResult, TarpaulinSummary};
use control_rs_xtask::{bridge, tui};

/// Representation of a single test case under discovery.
struct TestItem {
    /// Name of the test case.
    name: String,
    /// Current execution state of the test.
    state: control_rs_ets::comms::TestState,
}

/// Representation of a configuration setting in a test suite.
struct SettingItem {
    /// Name of the setting.
    name: String,
    /// Current value of the setting.
    value: control_rs_ets::settings::SettingValue,
}

/// Representation of a test suite containing tests and settings.
struct SuiteItem {
    /// Name of the suite.
    name: String,
    /// Collection of tests inside this suite.
    tests: Vec<TestItem>,
    /// Collection of config settings inside this suite.
    settings: Vec<SettingItem>,
}

/// Represents individual file coverage summary from tarpaulin JSON output.
#[derive(serde::Deserialize)]
struct TarpaulinFile {
    /// Number of covered lines.
    covered: usize,
    /// Number of coverable lines.
    coverable: usize,
}

/// Represents the top-level structure of tarpaulin JSON output report.
#[derive(serde::Deserialize)]
struct TarpaulinReportJson {
    /// List of file coverage statistics.
    files: Vec<TarpaulinFile>,
}

/// Helper function to build the QEMU target ELF.
pub fn build_qemu_elf(arch: bridge::QemuArch) -> String {
    println!("\t* building QEMU target ELF for {:?}...", arch);
    let (target_triple, bin_name) = match arch {
        bridge::QemuArch::Thumbv7emNoneEabihf => (
            "thumbv7em-none-eabihf",
            "control-rs-qemu-thumbv7em-none-eabihf",
        ),
        bridge::QemuArch::Thumbv7emNoneEabi => {
            ("thumbv7em-none-eabi", "control-rs-qemu-thumbv7em-none-eabi")
        }
        bridge::QemuArch::Riscv32imacUnknownNoneElf => (
            "riscv32imac-unknown-none-elf",
            "control-rs-qemu-riscv32imac-unknown-none-elf",
        ),
        bridge::QemuArch::Riscv64gcUnknownNoneElf => (
            "riscv64gc-unknown-none-elf",
            "control-rs-qemu-riscv64gc-unknown-none-elf",
        ),
    };

    let mut command = Command::new("cargo");
    command.current_dir("examples/qemu");
    let build_status = command
        .args([
            "build",
            "--bin",
            bin_name,
            "--target",
            target_triple,
            "--release",
        ])
        .status()
        .expect("Failed to build QEMU target.");

    if !build_status.success() {
        eprintln!("\tFailed to compile QEMU target.");
        exit(1);
    }

    format!(
        "examples/qemu/target/{}/release/{}",
        target_triple, bin_name
    )
}

/// Task to run formatting check.
pub fn run_fmt() -> (Result<(), usize>, String) {
    println!("\t* formatting check...");
    let fmt_output = Command::new("cargo")
        .args(["fmt-check"])
        .output()
        .expect("Failed to run cargo fmt-check");

    let stdout_str = String::from_utf8_lossy(&fmt_output.stdout);
    let stderr_str = String::from_utf8_lossy(&fmt_output.stderr);
    let combined = format!("{}{}", stdout_str, stderr_str);

    let ansi_escape = Regex::new(r"\x1B\[[0-9;]*[mK]|\x1B\(B").unwrap();
    let clean_str = ansi_escape.replace_all(&combined, "").into_owned();
    let fmt_errors = clean_str.matches("Diff in").count();

    let res = if fmt_output.status.success() && fmt_errors == 0 {
        Ok(())
    } else {
        Err(fmt_errors)
    };

    (res, clean_str)
}

/// Task to run clippy check with JSON output formatting.
pub fn run_clippy() -> (Result<(), usize>, String) {
    println!("\t* clippy check...");
    let clippy_output = Command::new("cargo")
        .args(["clippy-ci"])
        .output()
        .expect("Failed to run cargo clippy-json");

    let stdout_str = String::from_utf8_lossy(&clippy_output.stdout);
    let stderr_str = String::from_utf8_lossy(&clippy_output.stderr);

    let mut clippy_errors = 0;
    let mut rendered_logs = String::new();

    for line in stdout_str.lines() {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if val.get("reason").and_then(|r| r.as_str())
                == Some("compiler-message")
            {
                if let Some(msg) = val.get("message") {
                    if msg.get("level").and_then(|l| l.as_str())
                        == Some("error")
                    {
                        clippy_errors += 1;
                    }
                    if let Some(rendered) =
                        msg.get("rendered").and_then(|r| r.as_str())
                    {
                        rendered_logs.push_str(rendered);
                    }
                }
            }
        }
    }

    // Include stderr in clippy logs if compilation failed completely
    if !clippy_output.status.success() && rendered_logs.is_empty() {
        rendered_logs.push_str(&stderr_str);
    }

    let ansi_escape = Regex::new(r"\x1B\[[0-9;]*[mK]|\x1B\(B").unwrap();
    let clean_str = ansi_escape.replace_all(&rendered_logs, "").into_owned();

    let success = clippy_output.status.success() && clippy_errors == 0;
    let res = if success { Ok(()) } else { Err(clippy_errors) };

    (res, clean_str)
}

/// Task to run tarpaulin test & coverage.
pub fn run_tarpaulin() -> (Result<TarpaulinSummary, ()>, String) {
    println!("\t* tarpaulin test & coverage...");
    let tarpaulin_output = Command::new("cargo")
        .args(["coverage-ci"])
        .output()
        .expect("Failed to run cargo coverage");

    let tarp_str = format!(
        "{}\n{}",
        String::from_utf8_lossy(&tarpaulin_output.stderr),
        String::from_utf8_lossy(&tarpaulin_output.stdout)
    );

    let ansi_escape = Regex::new(r"\x1B\[[0-9;]*[mK]|\x1B\(B").unwrap();
    let clean_tarp_str = ansi_escape.replace_all(&tarp_str, "").into_owned();

    // Parse test counts using regex from log
    let re_passed = Regex::new(r"(\d+) passed").unwrap();
    let re_failed = Regex::new(r"(\d+) failed").unwrap();
    let re_ignored = Regex::new(r"(\d+) ignored").unwrap();

    let passed: usize = re_passed
        .captures_iter(&clean_tarp_str)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .sum();
    let failed: usize = re_failed
        .captures_iter(&clean_tarp_str)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .sum();
    let ignored: usize = re_ignored
        .captures_iter(&clean_tarp_str)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .sum();

    // Parse coverage from JSON report
    let mut covered_lines = 0;
    let mut total_lines = 0;
    if let Ok(json_content) = fs::read_to_string("tarpaulin-report.json") {
        if let Ok(report) =
            serde_json::from_str::<TarpaulinReportJson>(&json_content)
        {
            for file in report.files {
                covered_lines += file.covered;
                total_lines += file.coverable;
            }
        }
    }

    let percent = if total_lines > 0 {
        format!("{:.2}", (covered_lines as f64 / total_lines as f64) * 100.0)
    } else {
        "0.00".to_string()
    };

    let summary = TarpaulinSummary {
        passed,
        failed,
        ignored,
        coverage_percent: percent,
        covered_lines,
        total_lines,
    };

    let success = tarpaulin_output.status.success() && failed == 0;
    let res = if success { Ok(summary) } else { Err(()) };

    (res, clean_tarp_str)
}

/// CI frontend: drive ETS or virtual ETS through `ServerBridge` (no TUI).
pub fn run_ci_ets(
    target: &bridge::Target,
) -> (Result<Vec<HeadlessTestResult>, String>, String) {
    use bridge::BridgeMessage;
    use control_rs_ets::comms::{Command as CommCommand, Telemetry, TestState};

    let mut logs = String::new();
    let mut elf_path = String::new();

    let mut bridge = {
        if let bridge::Target::QemuSemihosting { arch } = target {
            elf_path = build_qemu_elf(*arch);
        }
        let elf_opt = if elf_path.is_empty() {
            None
        } else {
            Some(elf_path.as_str())
        };
        match bridge::ServerBridge::new(target.clone(), elf_opt) {
            Ok(b) => b,
            Err(e) => return (Err(e.to_string()), logs),
        }
    };

    let mut log_and_print = |msg: &str| {
        print!("{}", msg);
        logs.push_str(msg);
    };

    let backend = match target {
        bridge::Target::QemuSemihosting { .. } => "virtual ETS",
        bridge::Target::Serial { .. } => "ETS",
    };
    log_and_print(&format!("\t* CI → {backend}...\n"));

    // Initial discovery
    let mut last_send = Instant::now();
    let _ = bridge.send_command(&CommCommand::ListSuites);

    let mut suites: Vec<SuiteItem> = Vec::new();
    let mut run_queue = Vec::new();
    let mut results: Vec<HeadlessTestResult> = Vec::new();
    let mut current_running = None;
    let mut discovery_complete = false;
    let mut exit_loop = false;

    let start_time = Instant::now();
    let timeout = Duration::from_secs(90);

    while !exit_loop {
        if start_time.elapsed() > timeout {
            bridge.kill();
            return (
                Err("CI ETS execution timed out after 90s".to_string()),
                logs,
            );
        }

        if !discovery_complete
            && last_send.elapsed() > Duration::from_millis(500)
        {
            let _ = bridge.send_command(&CommCommand::ListSuites);
            last_send = Instant::now();
        }

        // Poll bridge messages
        while let Ok(msg) = bridge.receiver().try_recv() {
            match msg {
                BridgeMessage::Telemetry(telemetry) => match telemetry {
                    Telemetry::SuiteInfo { suite_id, name, .. } => {
                        let id = suite_id as usize;
                        while suites.len() <= id {
                            suites.push(SuiteItem {
                                name: String::new(),
                                tests: Vec::new(),
                                settings: Vec::new(),
                            });
                        }
                        suites[id].name = name.to_string();
                    }
                    Telemetry::TestInfo {
                        suite_id,
                        test_id,
                        name,
                        ..
                    } => {
                        let s_id = suite_id as usize;
                        let t_id = test_id as usize;
                        while suites[s_id].tests.len() <= t_id {
                            suites[s_id].tests.push(TestItem {
                                name: String::new(),
                                state: TestState::Pending,
                            });
                        }
                        suites[s_id].tests[t_id].name = name.to_string();
                    }
                    Telemetry::SettingInfo {
                        suite_id,
                        setting_id,
                        name,
                        value,
                        ..
                    } => {
                        let s_id = suite_id as usize;
                        let set_id = setting_id as usize;
                        while suites[s_id].settings.len() <= set_id {
                            suites[s_id].settings.push(SettingItem {
                                name: String::new(),
                                value:
                                    control_rs_ets::settings::SettingValue::U8(
                                        0,
                                    ),
                            });
                        }
                        suites[s_id].settings[set_id].name = name.to_string();
                        suites[s_id].settings[set_id].value = value;
                    }
                    Telemetry::DiscoveryComplete => {
                        discovery_complete = true;
                        run_queue.clear();
                        for (s_idx, suite) in suites.iter().enumerate() {
                            for (t_idx, test) in suite.tests.iter().enumerate()
                            {
                                let already_run = results.iter().any(|r| {
                                    r.suite_name == suite.name
                                        && r.test_name == test.name
                                });
                                if !already_run {
                                    run_queue
                                        .push((s_idx as u16, t_idx as u16));
                                }
                            }
                        }
                        if !run_queue.is_empty() {
                            let (next_s, next_t) = run_queue.remove(0);
                            current_running = Some((next_s, next_t));
                            let _ = bridge.send_command(
                                &CommCommand::RunExecutable {
                                    suite_id: next_s,
                                    test_id: next_t,
                                },
                            );
                        } else {
                            exit_loop = true;
                        }
                    }
                    Telemetry::TestStateChange {
                        suite_id,
                        test_id,
                        state: new_state,
                    } => {
                        let s_id = suite_id as usize;
                        let t_id = test_id as usize;
                        suites[s_id].tests[t_id].state = new_state;

                        if new_state == TestState::Failed {
                            results.push(HeadlessTestResult {
                                suite_name: suites[s_id].name.clone(),
                                test_name: suites[s_id].tests[t_id]
                                    .name
                                    .clone(),
                                state: TestState::Failed,
                                cycles: None,
                                time_us: None,
                                stack_peak: None,
                            });

                            current_running = None;
                            if !run_queue.is_empty() {
                                let (next_s, next_t) = run_queue.remove(0);
                                current_running = Some((next_s, next_t));
                                let _ = bridge.send_command(
                                    &CommCommand::RunExecutable {
                                        suite_id: next_s,
                                        test_id: next_t,
                                    },
                                );
                            } else {
                                exit_loop = true;
                            }
                        }
                    }
                    Telemetry::MetricReport {
                        suite_id,
                        test_id,
                        cycles,
                        time_us,
                        stack_peak,
                    } => {
                        let s_id = suite_id as usize;
                        let t_id = test_id as usize;

                        results.push(HeadlessTestResult {
                            suite_name: suites[s_id].name.clone(),
                            test_name: suites[s_id].tests[t_id].name.clone(),
                            state: TestState::Passed,
                            cycles: Some(cycles),
                            time_us: Some(time_us),
                            stack_peak: Some(stack_peak),
                        });

                        current_running = None;
                        if !run_queue.is_empty() {
                            let (next_s, next_t) = run_queue.remove(0);
                            current_running = Some((next_s, next_t));
                            let _ = bridge.send_command(
                                &CommCommand::RunExecutable {
                                    suite_id: next_s,
                                    test_id: next_t,
                                },
                            );
                        } else {
                            exit_loop = true;
                        }
                    }
                    Telemetry::Log(_) => {}
                    Telemetry::TargetPanic {
                        message,
                        file,
                        line,
                    } => {
                        let panic_msg = format!(
                            "Target panicked: '{}' at {}:{}\n",
                            message, file, line
                        );
                        log_and_print(&panic_msg);

                        if let Some((s_id, t_id)) = current_running {
                            let s_idx = s_id as usize;
                            let t_idx = t_id as usize;
                            if s_idx < suites.len()
                                && t_idx < suites[s_idx].tests.len()
                            {
                                suites[s_idx].tests[t_idx].state =
                                    TestState::Failed;
                                results.push(HeadlessTestResult {
                                    suite_name: suites[s_idx].name.clone(),
                                    test_name: suites[s_idx].tests[t_idx]
                                        .name
                                        .clone(),
                                    state: TestState::Failed,
                                    cycles: None,
                                    time_us: None,
                                    stack_peak: None,
                                });
                            }
                        }

                        let _ = bridge.send_command(&CommCommand::TryReset);
                        std::thread::sleep(Duration::from_millis(50));
                        bridge.kill();
                        std::thread::sleep(Duration::from_secs(1));

                        current_running = None;
                        discovery_complete = false;

                        let remaining_to_run = suites.iter().any(|suite| {
                            suite.tests.iter().any(|test| {
                                !results.iter().any(|r| {
                                    r.suite_name == suite.name
                                        && r.test_name == test.name
                                })
                            })
                        });

                        if remaining_to_run {
                            log_and_print(
                                "Restarting target bridge to continue running tests...\n",
                            );
                            let elf_opt = if elf_path.is_empty() {
                                None
                            } else {
                                Some(elf_path.as_str())
                            };
                            bridge = match bridge::ServerBridge::new(
                                target.clone(),
                                elf_opt,
                            ) {
                                Ok(b) => b,
                                Err(e) => return (Err(e.to_string()), logs),
                            };
                            let _ =
                                bridge.send_command(&CommCommand::ListSuites);
                            last_send = Instant::now();
                        } else {
                            exit_loop = true;
                        }
                    }
                },
                BridgeMessage::RawConsole(line) => {
                    log_and_print(&format!("[Target Console] {}\n", line));
                }
            }
        }

        if let Ok(Some(status)) = bridge.try_wait() {
            if !discovery_complete
                || current_running.is_some()
                || !run_queue.is_empty()
            {
                bridge.kill();
                return (
                    Err(format!(
                        "QEMU process exited unexpectedly: {}",
                        status
                    )),
                    logs,
                );
            }
            exit_loop = true;
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    bridge.kill();
    (Ok(results), logs)
}

/// Task to start the interactive ETS TUI.
pub fn run_ets_tui(target: &bridge::Target) {
    let elf_path = match target {
        bridge::Target::QemuSemihosting { arch } => build_qemu_elf(*arch),
        bridge::Target::Serial { .. } => String::new(),
    };
    let elf_opt = if elf_path.is_empty() {
        None
    } else {
        Some(elf_path.as_str())
    };
    let bridge = match bridge::ServerBridge::new(target.clone(), elf_opt) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("\tFailed to start bridge: {}", e);
            exit(1);
        }
    };
    if let Err(e) = tui::run_tui(bridge, target, &elf_path) {
        eprintln!("\tTUI Error: {}", e);
        exit(1);
    }
}

/// Task to install git hooks in the local repository.
pub fn install_hooks() {
    println!("\t* installing pre-commit git hook...");
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .expect("Failed to find workspace root directory");
    let git_dir = workspace_root.join(".git");

    if !git_dir.exists() {
        eprintln!(
            "\tError: .git directory not found at {}. Is this a git repository?",
            git_dir.display()
        );
        exit(1);
    }

    let hooks_dir = git_dir.join("hooks");
    if !hooks_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&hooks_dir) {
            eprintln!(
                "\tError: Failed to create hooks directory {}: {}",
                hooks_dir.display(),
                e
            );
            exit(1);
        }
    }

    let pre_commit_path = hooks_dir.join("pre-commit");

    let hook_content = r#"#!/bin/bash
# Pre-commit hook to format code and run cargo clippy
# Automatically generated by cargo xtask install-hooks

echo "========================================="
echo "Running pre-commit checks..."
echo "========================================="

# 1. Format code
echo "Running cargo fmt --all..."
cargo fmt --all

# Auto-stage any formatting changes on already-staged Rust files
staged_rust_files=$(git diff --name-only --cached --diff-filter=d | grep '\.rs$')
if [ -n "$staged_rust_files" ]; then
    echo "Staging auto-formatted Rust files..."
    for file in $staged_rust_files; do
        git add "$file"
    done
fi

# 2. Clippy check
echo "Running cargo clippy..."
cargo clippy --workspace --all-targets -- -D warnings
clippy_status=$?

if [ $clippy_status -ne 0 ]; then
    echo "========================================="
    echo "Error: cargo clippy failed with status $clippy_status."
    echo "Commit aborted."
    echo "========================================="
    exit 1
fi

echo "========================================="
echo "Pre-commit checks passed successfully!"
echo "========================================="
exit 0
"#;

    if let Err(e) = std::fs::write(&pre_commit_path, hook_content) {
        eprintln!(
            "\tError: Failed to write pre-commit hook file {}: {}",
            pre_commit_path.display(),
            e
        );
        exit(1);
    }

    // Set file permissions to executable on Unix-like operating systems.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(&pre_commit_path) {
            let mut permissions = metadata.permissions();
            permissions.set_mode(0o755); // rwxr-xr-x
            if let Err(e) =
                std::fs::set_permissions(&pre_commit_path, permissions)
            {
                eprintln!(
                    "\tWarning: Failed to set executable permissions on {}: {}",
                    pre_commit_path.display(),
                    e
                );
            }
        }
    }

    println!(
        "\t* pre-commit git hook installed successfully at: {}",
        pre_commit_path.display()
    );
}
