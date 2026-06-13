//! Modules defining the runner and compilation tasks executed by `xtask`.
//! Includes tasks for formatting check, clippy checks, coverage tracking with tarpaulin,
//! and running headless/interactive tests.
//!
//! TODO:
//!   - Remove duplicate code fragments (364-375)(398-401).

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
    state: control_rs_hil::comms::TestState,
}

/// Representation of a configuration setting in a test suite.
struct SettingItem {
    /// Name of the setting.
    name: String,
    /// Current value of the setting.
    value: control_rs_hil::settings::SettingValue,
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
pub fn build_qemu_elf() -> String {
    println!("Building QEMU target ELF...");
    let build_status = Command::new("cargo")
        .env(
            "CARGO_TARGET_THUMBV7EM_NONE_EABIHF_RUSTFLAGS",
            "-C link-arg=-Tlink.x -C link-arg=-Thil_suites.x",
        )
        .args([
            "build",
            "--manifest-path",
            "examples/qemu-example/Cargo.toml",
            "--bin",
            "control-rs-qemu-arm",
            "--target",
            "thumbv7em-none-eabihf",
            "--profile",
            "qemu",
        ])
        .status()
        .expect("Failed to build QEMU target.");

    if !build_status.success() {
        eprintln!("Failed to compile QEMU target.");
        exit(1);
    }

    "examples/qemu-example/target/thumbv7em-none-eabihf/qemu/control-rs-qemu-arm".to_string()
}

/// Task to run formatting check.
pub fn run_fmt() -> (Result<(), usize>, String) {
    println!("Running formatting check...");
    let fmt_output = Command::new("cargo")
        .args(["fmt", "--all", "--", "--check"])
        .output()
        .expect("Failed to run cargo fmt");

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
    println!("Running clippy check...");
    let clippy_output = Command::new("cargo")
        .args([
            "clippy",
            "--workspace",
            "--lib",
            "--bins",
            "--tests",
            "--examples",
            "--benches",
            "--message-format=json",
            "--",
            "-D",
            "warnings",
        ])
        .output()
        .expect("Failed to run cargo clippy");

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
    println!("Running tarpaulin test & coverage...");
    let tarpaulin_output = Command::new("cargo")
        .args([
            "tarpaulin",
            "--verbose",
            "--color",
            "never",
            "--out",
            "Html",
            "--out",
            "Json",
        ])
        .output()
        .expect("Failed to run cargo tarpaulin");

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

/// Task to run headless SIL execution.
pub fn run_headless_sil(
    target: &bridge::Target,
) -> (Result<Vec<HeadlessTestResult>, String>, String) {
    use bridge::BridgeMessage;
    use control_rs_hil::comms::{Command as CommCommand, Telemetry, TestState};

    let mut logs = String::new();
    let mut elf_path = String::new();

    let mut bridge = match target {
        bridge::Target::Qemu => {
            elf_path = build_qemu_elf();
            match bridge::QemuBridge::new(&elf_path, target.clone()) {
                Ok(b) => b,
                Err(e) => return (Err(e.to_string()), logs),
            }
        }
        bridge::Target::Serial { .. } => {
            match bridge::QemuBridge::new("", target.clone()) {
                Ok(b) => b,
                Err(e) => return (Err(e.to_string()), logs),
            }
        }
    };

    let mut log_and_print = |msg: &str| {
        print!("{}", msg);
        logs.push_str(msg);
    };

    log_and_print("Running headlessly executed SIL tests...\n");

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
    let timeout = Duration::from_secs(15);

    while !exit_loop {
        if start_time.elapsed() > timeout {
            bridge.kill();
            return (
                Err("SIL execution timed out after 15s".to_string()),
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
                    } => {
                        let s_id = suite_id as usize;
                        let set_id = setting_id as usize;
                        while suites[s_id].settings.len() <= set_id {
                            suites[s_id].settings.push(SettingItem {
                                name: String::new(),
                                value:
                                    control_rs_hil::settings::SettingValue::U8(
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
                    } => {
                        let s_id = suite_id as usize;
                        let t_id = test_id as usize;

                        results.push(HeadlessTestResult {
                            suite_name: suites[s_id].name.clone(),
                            test_name: suites[s_id].tests[t_id].name.clone(),
                            state: TestState::Passed,
                            cycles: Some(cycles),
                            time_us: Some(time_us),
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
                                });
                            }
                        }

                        let _ = bridge.send_command(&CommCommand::OkToReset);
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
                            bridge = match &target {
                                bridge::Target::Qemu => {
                                    match bridge::QemuBridge::new(
                                        &elf_path,
                                        target.clone(),
                                    ) {
                                        Ok(b) => b,
                                        Err(e) => {
                                            return (Err(e.to_string()), logs);
                                        }
                                    }
                                }
                                bridge::Target::Serial { .. } => {
                                    match bridge::QemuBridge::new(
                                        "",
                                        target.clone(),
                                    ) {
                                        Ok(b) => b,
                                        Err(e) => {
                                            return (Err(e.to_string()), logs);
                                        }
                                    }
                                }
                            };
                            let _ =
                                bridge.send_command(&CommCommand::ListSuites);
                            last_send = Instant::now();
                        } else {
                            exit_loop = true;
                        }
                    }
                },
                BridgeMessage::RawConsole(_) => {}
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

/// Task to clean and run the QEMU Sil kernel.
pub fn run_qemu() {
    let clean_status = Command::new("cargo")
        .args([
            "clean",
            "--manifest-path",
            "examples/qemu-example/Cargo.toml",
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
        .env("CARGO_TARGET_THUMBV7EM_NONE_EABIHF_RUNNER", "qemu-system-arm -cpu cortex-m7 -machine mps2-an500 -nographic -serial none -monitor none -chardev stdio,id=con0 -semihosting-config enable=on,chardev=con0 -kernel")
        .env("CARGO_TARGET_THUMBV7EM_NONE_EABIHF_RUSTFLAGS", "-C link-arg=-Tlink.x -C link-arg=-Thil_suites.x")
        .args([
            "run",
            "--manifest-path",
            "examples/qemu-example/Cargo.toml",
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

/// Task to start the interactive HIL TUI.
pub fn run_hil_tui(target: &bridge::Target) {
    let (bridge, elf_path) = match target {
        bridge::Target::Qemu => {
            let elf_path = build_qemu_elf();
            match bridge::QemuBridge::new(&elf_path, target.clone()) {
                Ok(b) => (b, elf_path),
                Err(e) => {
                    eprintln!("Failed to start bridge: {}", e);
                    exit(1);
                }
            }
        }
        bridge::Target::Serial { .. } => {
            match bridge::QemuBridge::new("", target.clone()) {
                Ok(b) => (b, String::new()),
                Err(e) => {
                    eprintln!("Failed to start bridge: {}", e);
                    exit(1);
                }
            }
        }
    };
    if let Err(e) = tui::run_tui(bridge, target, &elf_path) {
        eprintln!("TUI Error: {}", e);
        exit(1);
    }
}
