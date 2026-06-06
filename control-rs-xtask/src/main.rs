use regex::Regex;
use std::env;
use std::fs;
use std::process::{Command, exit};
use std::time::Instant;

mod bridge;
mod tui;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage_and_exit();
    }

    match args[1].as_str() {
        "ci" => {
            let target_str = args.get(2).map(|s| s.as_str()).unwrap_or("qemu");
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
                    eprintln!("Unknown target: {}", target_str);
                    exit(1);
                }
            };
            run_ci(target);
        }
        "qemu" => {
            run_qemu();
        }
        "hil-tui" => {
            let target_str = args.get(2).map(|s| s.as_str()).unwrap_or("qemu");
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
                    eprintln!("Unknown target: {}", target_str);
                    exit(1);
                }
            };
            run_hil_tui(target);
        }
        _ => {
            print_usage_and_exit();
        }
    }
}

fn print_usage_and_exit() -> ! {
    eprintln!("Usage: cargo control-rs-xtask <task> [target] [port] [baud]");
    eprintln!("Tasks: ci, qemu, hil-tui");
    eprintln!("Targets: qemu (default), teensy");
    exit(1);
}

fn build_qemu_elf() -> String {
    println!("Building QEMU target ELF...");
    let build_status = Command::new("cargo")
        .env(
            "CARGO_TARGET_THUMBV7EM_NONE_EABIHF_RUSTFLAGS",
            "-C link-arg=-Tlink.x -C link-arg=-Thil_suites.x",
        )
        .args([
            "build",
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
        .expect("Failed to build QEMU target.");

    if !build_status.success() {
        eprintln!("Failed to compile QEMU target.");
        exit(1);
    }

    "target/thumbv7em-none-eabihf/qemu/control-rs-qemu-arm".to_string()
}

fn run_hil_tui(target: bridge::Target) {
    let bridge = match &target {
        bridge::Target::Qemu => {
            let elf_path = build_qemu_elf();
            bridge::QemuBridge::new(&elf_path, target)
        }
        bridge::Target::Serial { .. } => bridge::QemuBridge::new("", target),
    };
    if let Err(e) = tui::run_tui(bridge) {
        eprintln!("TUI Error: {}", e);
        exit(1);
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
        .env("CARGO_TARGET_THUMBV7EM_NONE_EABIHF_RUNNER", "qemu-system-arm -cpu cortex-m7 -machine mps2-an500 -nographic -serial none -monitor none -chardev stdio,id=con0 -semihosting-config enable=on,chardev=con0 -kernel")
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

fn run_ci(target: bridge::Target) {
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

    // Headless HIL Runner Execution
    println!("Running headlessly executed HIL tests...");
    let start_hil = Instant::now();
    let hil_results_str = match run_headless_hil(target) {
        Ok(results) => {
            let mut s = String::from("### Headless HIL Test Results\n\n");
            s.push_str("| Suite | Test | Result | Cycles | Time |\n| :--- | :--- | :--- | :--- | :--- |\n");
            let mut all_passed = true;
            for r in &results {
                let res_str = match r.state {
                    control_rs_hil::comms::TestState::Passed => "PASSED",
                    _ => {
                        all_passed = false;
                        "FAILED"
                    }
                };
                let cyc_str =
                    r.cycles.map_or("N/A".to_string(), |c| c.to_string());
                let time_str =
                    r.time_us.map_or("N/A".to_string(), |t| format!("{}us", t));
                s.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    r.suite_name, r.test_name, res_str, cyc_str, time_str
                ));
            }
            s.push_str("\n");
            if !all_passed {
                ci_success = false;
            }
            s
        }
        Err(e) => {
            ci_success = false;
            format!(
                "### Headless HIL Test Results\n\n**ERROR**: Failed to run HIL tests: {}\n\n",
                e
            )
        }
    };
    report.push_str(&hil_results_str);
    println!(
        "Headless HIL tests completed in {:.2}s.",
        start_hil.elapsed().as_secs_f32()
    );

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

struct HeadlessTestResult {
    suite_name: String,
    test_name: String,
    state: control_rs_hil::comms::TestState,
    cycles: Option<u64>,
    time_us: Option<u64>,
}

struct TestItem {
    name: String,
    state: control_rs_hil::comms::TestState,
}

struct SettingItem {
    name: String,
    value: control_rs_hil::hil_test::SettingValue,
}

struct SuiteItem {
    name: String,
    tests: Vec<TestItem>,
    settings: Vec<SettingItem>,
}

fn run_headless_hil(
    target: bridge::Target,
) -> Result<Vec<HeadlessTestResult>, String> {
    use bridge::BridgeMessage;
    use control_rs_hil::comms::{Command, Telemetry, TestState};
    use std::time::Duration;

    let mut bridge = match &target {
        bridge::Target::Qemu => {
            let elf_path = build_qemu_elf();
            bridge::QemuBridge::new(&elf_path, target)
        }
        bridge::Target::Serial { .. } => bridge::QemuBridge::new("", target),
    };

    // Initial discovery
    let mut last_send = Instant::now();
    let _ = bridge.send_command(&Command::ListSuites);

    let mut suites: Vec<SuiteItem> = Vec::new();
    let mut run_queue = Vec::new();
    let mut results = Vec::new();
    let mut current_running = None;
    let mut discovery_complete = false;
    let mut exit_loop = false;

    // We set a safety timeout (e.g. 15 seconds) so CI doesn't hang forever
    let start_time = Instant::now();
    let timeout = Duration::from_secs(15);

    while !exit_loop {
        if start_time.elapsed() > timeout {
            bridge.kill();
            return Err("HIL execution timed out after 15s".to_string());
        }

        if !discovery_complete
            && last_send.elapsed() > Duration::from_millis(500)
        {
            let _ = bridge.send_command(&Command::ListSuites);
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
                                    control_rs_hil::hil_test::SettingValue::U8(
                                        0,
                                    ),
                            });
                        }
                        suites[s_id].settings[set_id].name = name.to_string();
                        suites[s_id].settings[set_id].value = value;
                    }
                    Telemetry::DiscoveryComplete => {
                        discovery_complete = true;
                        // Enqueue all tests
                        for (s_idx, suite) in suites.iter().enumerate() {
                            for (t_idx, _) in suite.tests.iter().enumerate() {
                                run_queue.push((s_idx as u16, t_idx as u16));
                            }
                        }
                        // Start first test
                        if !run_queue.is_empty() {
                            let (next_s, next_t) = run_queue.remove(0);
                            current_running = Some((next_s, next_t));
                            let _ =
                                bridge.send_command(&Command::RunExecutable {
                                    suite_id: next_s,
                                    test_id: next_t,
                                });
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
                            // Record failed test
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
                                    &Command::RunExecutable {
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
                            let _ =
                                bridge.send_command(&Command::RunExecutable {
                                    suite_id: next_s,
                                    test_id: next_t,
                                });
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
                        bridge.kill();
                        return Err(format!(
                            "Target panicked: '{}' at {}:{}",
                            message, file, line
                        ));
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
                return Err(format!(
                    "QEMU process exited unexpectedly: {}",
                    status
                ));
            }
            exit_loop = true;
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    bridge.kill();
    Ok(results)
}
