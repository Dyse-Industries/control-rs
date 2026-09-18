//! Discovery, run queue and session state machine for host-side ETS.

use control_rs_ets::comms::{Command as CommCommand, TestState};
use control_rs_ets::settings::SettingValue;

use crate::bridge::{BridgeMessage, OwnedTelemetry};
use crate::runner::TestOutcome;

/// Representation of a single test case under discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestItem {
    /// Name of the suite containing this test case.
    pub suite_name: String,
    /// Name of the test case.
    pub name: String,
    /// Current execution state of the test.
    pub state: TestState,
    /// CPU cycles consumed during the test run if completed.
    pub cycles: Option<u64>,
    /// Elapsed time of the test run in microseconds if completed.
    pub time_us: Option<u64>,
    /// Peak stack memory usage in bytes if completed.
    pub stack_peak: Option<u32>,
}

/// Representation of a configuration setting in a test suite.
#[derive(Debug, Clone)]
pub struct SettingItem {
    /// Name of the setting.
    pub name: String,
    /// Doc-comment description of the setting.
    pub description: String,
    /// Current value of the setting.
    pub value: SettingValue,
}

/// Representation of a test suite containing tests and settings.
#[derive(Debug, Clone)]
pub struct SuiteItem {
    /// Name of the suite.
    pub name: String,
    /// Collection of tests inside this suite.
    pub tests: Vec<TestItem>,
    /// Collection of config settings inside this suite.
    pub settings: Vec<SettingItem>,
}

/// Host-side ETS session state (discovery, run queue, results).
pub struct SessionState {
    /// Currently executing test (`suite_id`, `test_id`).
    pub current_running: Option<(u16, u16)>,
    /// Whether initial test discovery has finished.
    pub discovery_complete: bool,
    /// Flag signaling that test execution has finished or reached terminal state.
    pub exit_loop: bool,
    /// Buffer of log and diagnostic messages collected from the target.
    pub logs: String,
    /// Collection of test results recorded so far.
    pub results: Vec<TestOutcome>,
    /// Queue of tests pending execution.
    pub run_queue: Vec<(u16, u16)>,
    /// Discovered suites and their tests/settings.
    pub suites: Vec<SuiteItem>,
}

/// Side effects requested by [`SessionState`] while processing bridge messages.
#[derive(Debug)]
pub enum SessionAction {
    /// Request target restart and bridge reconnection.
    PanicRestart,
    /// Send a command packet to the target.
    Send(CommCommand),
}

impl TestItem {
    /// Converts this [`TestItem`] into a [`TestOutcome`].
    #[must_use]
    pub fn into_outcome(self) -> TestOutcome {
        TestOutcome {
            suite_name: self.suite_name,
            test_name: self.name,
            state: self.state,
            cycles: self.cycles,
            time_us: self.time_us,
            stack_peak: self.stack_peak,
        }
    }
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionState {
    /// Creates a new, empty session state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current_running: None,
            discovery_complete: false,
            exit_loop: false,
            logs: String::new(),
            results: Vec::new(),
            run_queue: Vec::new(),
            suites: Vec::new(),
        }
    }

    /// Appends a raw message to the internal log buffer.
    pub fn log(&mut self, msg: &str) {
        self.logs.push_str(msg);
    }

    /// Enqueues all discovered tests across all suites for execution.
    #[allow(clippy::cast_possible_truncation)]
    pub fn enqueue_all(&mut self) -> Option<SessionAction> {
        self.run_queue.clear();
        for (s_idx, suite) in self.suites.iter().enumerate() {
            for (t_idx, _) in suite.tests.iter().enumerate() {
                self.run_queue.push((s_idx as u16, t_idx as u16));
            }
        }
        if self.current_running.is_none() {
            self.start_next_or_exit()
        } else {
            None
        }
    }

    /// Enqueues a specific test for execution.
    pub fn enqueue_test(
        &mut self,
        suite_id: u16,
        test_id: u16,
    ) -> Option<SessionAction> {
        if self.current_running.is_none() {
            self.current_running = Some((suite_id, test_id));
            Some(SessionAction::Send(CommCommand::RunExecutable {
                suite_id,
                test_id,
            }))
        } else {
            self.run_queue.push((suite_id, test_id));
            None
        }
    }

    /// Stops any currently running tests and clears the run queue.
    pub fn stop(&mut self) {
        self.run_queue.clear();
        self.current_running = None;
    }

    /// Dequeues the next test to run, or marks the session loop complete if queue is empty.
    pub fn start_next_or_exit(&mut self) -> Option<SessionAction> {
        self.current_running = None;
        if self.run_queue.is_empty() {
            self.exit_loop = true;
            None
        } else {
            let (next_s, next_t) = self.run_queue.remove(0);
            self.current_running = Some((next_s, next_t));
            Some(SessionAction::Send(CommCommand::RunExecutable {
                suite_id: next_s,
                test_id: next_t,
            }))
        }
    }

    /// Processes an incoming bridge message and updates session state accordingly.
    #[allow(clippy::cast_possible_truncation)]
    pub fn handle_message(&mut self, msg: BridgeMessage) -> Vec<SessionAction> {
        match msg {
            BridgeMessage::RawConsole(line) => {
                let formatted = format!("      [ETS] {line}\n");
                self.log(&formatted);
                Vec::new()
            }
            BridgeMessage::Telemetry(telemetry) => match telemetry {
                OwnedTelemetry::SuiteInfo { suite_id, name, .. } => {
                    let id = suite_id as usize;
                    while self.suites.len() <= id {
                        self.suites.push(SuiteItem {
                            name: String::new(),
                            tests: Vec::new(),
                            settings: Vec::new(),
                        });
                    }
                    self.suites[id].name = name;
                    Vec::new()
                }
                OwnedTelemetry::TestInfo {
                    suite_id,
                    test_id,
                    name,
                    ..
                } => {
                    let s_id = suite_id as usize;
                    let t_id = test_id as usize;
                    while self.suites.len() <= s_id {
                        self.suites.push(SuiteItem {
                            name: String::new(),
                            tests: Vec::new(),
                            settings: Vec::new(),
                        });
                    }
                    let suite_name = self.suites[s_id].name.clone();
                    while self.suites[s_id].tests.len() <= t_id {
                        self.suites[s_id].tests.push(TestItem {
                            suite_name: suite_name.clone(),
                            name: String::new(),
                            state: TestState::Pending,
                            cycles: None,
                            time_us: None,
                            stack_peak: None,
                        });
                    }
                    self.suites[s_id].tests[t_id].suite_name = suite_name;
                    self.suites[s_id].tests[t_id].name = name;
                    Vec::new()
                }
                OwnedTelemetry::SettingInfo {
                    suite_id,
                    setting_id,
                    name,
                    value,
                    description,
                    ..
                } => {
                    let s_id = suite_id as usize;
                    let set_id = setting_id as usize;
                    while self.suites.len() <= s_id {
                        self.suites.push(SuiteItem {
                            name: String::new(),
                            tests: Vec::new(),
                            settings: Vec::new(),
                        });
                    }
                    while self.suites[s_id].settings.len() <= set_id {
                        self.suites[s_id].settings.push(SettingItem {
                            name: String::new(),
                            description: String::new(),
                            value: SettingValue::U8(0),
                        });
                    }
                    self.suites[s_id].settings[set_id].name = name;
                    self.suites[s_id].settings[set_id].description =
                        description;
                    self.suites[s_id].settings[set_id].value = value;
                    Vec::new()
                }
                OwnedTelemetry::DiscoveryComplete => {
                    self.discovery_complete = true;
                    self.run_queue.clear();
                    for (s_idx, suite) in self.suites.iter_mut().enumerate() {
                        for (t_idx, test) in suite.tests.iter_mut().enumerate()
                        {
                            let already_recorded =
                                self.results.iter().find(|r| {
                                    r.suite_name == suite.name
                                        && r.test_name == test.name
                                });
                            if let Some(prev) = already_recorded {
                                test.state = prev.state;
                                test.cycles = prev.cycles;
                                test.time_us = prev.time_us;
                                test.stack_peak = prev.stack_peak;
                            } else {
                                self.run_queue
                                    .push((s_idx as u16, t_idx as u16));
                            }
                        }
                    }
                    self.start_next_or_exit().into_iter().collect()
                }
                OwnedTelemetry::TestStateChange {
                    suite_id,
                    test_id,
                    state: new_state,
                } => {
                    let s_id = suite_id as usize;
                    let t_id = test_id as usize;
                    if s_id < self.suites.len()
                        && t_id < self.suites[s_id].tests.len()
                    {
                        self.suites[s_id].tests[t_id].state = new_state;
                    }

                    if new_state == TestState::Failed {
                        // Target `handle_failure` always emits Failed then
                        // TargetPanic. Advancing the queue here would make the
                        // subsequent TargetPanic attribute the crash to the
                        // next queued test and skip running it.
                        let suite_name = self
                            .suites
                            .get(s_id)
                            .map(|s| s.name.clone())
                            .unwrap_or_default();
                        let test_name = self
                            .suites
                            .get(s_id)
                            .and_then(|s| s.tests.get(t_id))
                            .map(|t| t.name.clone())
                            .unwrap_or_default();
                        let already_recorded = self.results.iter().any(|r| {
                            r.suite_name == suite_name
                                && r.test_name == test_name
                        });
                        if !already_recorded {
                            self.results.push(TestOutcome {
                                suite_name,
                                test_name,
                                state: TestState::Failed,
                                cycles: None,
                                time_us: None,
                                stack_peak: None,
                            });
                        }
                        self.current_running = Some((suite_id, test_id));
                    }
                    Vec::new()
                }
                OwnedTelemetry::MetricReport {
                    suite_id,
                    test_id,
                    cycles,
                    time_us,
                    stack_peak,
                } => {
                    let s_id = suite_id as usize;
                    let t_id = test_id as usize;
                    if s_id < self.suites.len()
                        && t_id < self.suites[s_id].tests.len()
                    {
                        self.suites[s_id].tests[t_id].state = TestState::Passed;
                        self.suites[s_id].tests[t_id].cycles = Some(cycles);
                        self.suites[s_id].tests[t_id].time_us = Some(time_us);
                        self.suites[s_id].tests[t_id].stack_peak =
                            Some(stack_peak);
                    }
                    let suite_name = self
                        .suites
                        .get(s_id)
                        .map(|s| s.name.clone())
                        .unwrap_or_default();
                    let test_name = self
                        .suites
                        .get(s_id)
                        .and_then(|s| s.tests.get(t_id))
                        .map(|t| t.name.clone())
                        .unwrap_or_default();
                    self.results.push(TestOutcome {
                        suite_name,
                        test_name,
                        state: TestState::Passed,
                        cycles: Some(cycles),
                        time_us: Some(time_us),
                        stack_peak: Some(stack_peak),
                    });
                    self.start_next_or_exit().into_iter().collect()
                }
                OwnedTelemetry::Log { .. } => Vec::new(),
                OwnedTelemetry::TargetPanic {
                    message,
                    file,
                    line,
                } => {
                    let panic_str = format!(
                        "target panicked: '{message}' at {file}:{line}\n"
                    );
                    self.log(&panic_str);

                    if let Some((s_id, t_id)) = self.current_running {
                        let s_idx = s_id as usize;
                        let t_idx = t_id as usize;
                        if s_idx < self.suites.len()
                            && t_idx < self.suites[s_idx].tests.len()
                        {
                            self.suites[s_idx].tests[t_idx].state =
                                TestState::Failed;
                            let suite_name = self.suites[s_idx].name.clone();
                            let test_name =
                                self.suites[s_idx].tests[t_idx].name.clone();
                            let already_recorded =
                                self.results.iter().any(|r| {
                                    r.suite_name == suite_name
                                        && r.test_name == test_name
                                });
                            if !already_recorded {
                                self.results.push(TestOutcome {
                                    suite_name,
                                    test_name,
                                    state: TestState::Failed,
                                    cycles: None,
                                    time_us: None,
                                    stack_peak: None,
                                });
                            }
                        }
                    }

                    self.current_running = None;
                    self.discovery_complete = false;

                    let remaining_to_run = self.suites.iter().any(|suite| {
                        suite.tests.iter().any(|test| {
                            !self.results.iter().any(|r| {
                                r.suite_name == suite.name
                                    && r.test_name == test.name
                            })
                        })
                    });

                    if remaining_to_run {
                        self.log("Restarting target bridge to continue running tests\n");
                    } else {
                        self.exit_loop = true;
                    }
                    vec![
                        SessionAction::Send(CommCommand::TryReset),
                        SessionAction::PanicRestart,
                    ]
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_rs_ets::comms::Telemetry;

    fn discover_two_tests(state: &mut SessionState) {
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 2,
                setting_count: 1,
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "t0",
                description: "",
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "t1",
                description: "",
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                name: "gain",
                description: "",
                value: SettingValue::U8(3),
            },
        ));
    }

    #[test]
    fn ci_ets_state_runs_queue_and_records_pass_fail() {
        let mut state = SessionState::new();
        discover_two_tests(&mut state);

        let actions = state.handle_message(BridgeMessage::telemetry(
            Telemetry::DiscoveryComplete,
        ));
        assert!(matches!(
            actions.as_slice(),
            [SessionAction::Send(CommCommand::RunExecutable {
                suite_id: 0,
                test_id: 0
            })]
        ));
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1)]);

        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Running,
            },
        ));
        let actions = state.handle_message(BridgeMessage::telemetry(
            Telemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 10,
                time_us: 20,
                stack_peak: 30,
            },
        ));
        assert!(matches!(
            actions.as_slice(),
            [SessionAction::Send(CommCommand::RunExecutable {
                suite_id: 0,
                test_id: 1
            })]
        ));
        assert_eq!(state.results[0].state, TestState::Passed);

        // Failed alone must not drain the session: TargetPanic owns recovery.
        let actions = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 1,
                state: TestState::Failed,
            },
        ));
        assert!(actions.is_empty());
        assert!(!state.exit_loop);
        assert_eq!(state.current_running, Some((0, 1)));
        assert_eq!(state.results[1].state, TestState::Failed);

        let actions = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TargetPanic {
                message: "assert",
                file: "t1.rs",
                line: 1,
            },
        ));
        assert!(state.exit_loop);
        assert_eq!(state.results.len(), 2);
        assert!(matches!(
            actions.as_slice(),
            [
                SessionAction::Send(CommCommand::TryReset),
                SessionAction::PanicRestart
            ]
        ));

        let _ = state.handle_message(BridgeMessage::telemetry(Telemetry::Log(
            control_rs_ets::comms::LogMessage {
                timestamp_us: 1,
                suite_id: 0,
                test_id: 0,
                payload: "hi",
            },
        )));
        let _ = state
            .handle_message(BridgeMessage::RawConsole("console".to_string()));
        assert!(state.logs.contains("      [ETS] console"));
    }

    #[test]
    fn failed_before_target_panic_does_not_blame_next_test() {
        let mut state = SessionState::new();
        discover_two_tests(&mut state);
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::DiscoveryComplete,
        ));
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1)]);

        // Mirror on-target handle_failure: Failed for the crashing test, then
        // TargetPanic. The next queued test must remain pending for restart.
        let actions = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Failed,
            },
        ));
        assert!(actions.is_empty());
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1)]);
        assert_eq!(state.results.len(), 1);
        assert_eq!(state.results[0].test_name, "t0");

        let actions = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TargetPanic {
                message: "boom",
                file: "t0.rs",
                line: 3,
            },
        ));
        assert_eq!(state.results.len(), 1);
        assert_eq!(state.suites[0].tests[1].state, TestState::Pending);
        assert!(!state.exit_loop);
        assert!(state.logs.contains("Restarting target bridge"));
        assert!(matches!(
            actions.as_slice(),
            [
                SessionAction::Send(CommCommand::TryReset),
                SessionAction::PanicRestart
            ]
        ));
    }

    #[test]
    fn ci_ets_state_empty_discovery_and_panic_restart() {
        let mut empty = SessionState::new();
        let _ = empty.handle_message(BridgeMessage::telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "empty",
                description: "",
                test_count: 0,
                setting_count: 0,
            },
        ));
        let actions = empty.handle_message(BridgeMessage::telemetry(
            Telemetry::DiscoveryComplete,
        ));
        assert!(actions.is_empty());
        assert!(empty.exit_loop);

        let mut state = SessionState::new();
        discover_two_tests(&mut state);
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::DiscoveryComplete,
        ));
        let actions = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TargetPanic {
                message: "boom",
                file: "main.rs",
                line: 9,
            },
        ));
        assert!(state.logs.contains("target panicked"));
        assert!(state.logs.contains("Restarting target bridge"));
        assert!(!state.discovery_complete);
        assert!(matches!(
            actions.as_slice(),
            [
                SessionAction::Send(CommCommand::TryReset),
                SessionAction::PanicRestart
            ]
        ));

        state.results.push(TestOutcome {
            suite_name: "suite".to_string(),
            test_name: "t0".to_string(),
            state: TestState::Failed,
            cycles: None,
            time_us: None,
            stack_peak: None,
        });
        state.results.push(TestOutcome {
            suite_name: "suite".to_string(),
            test_name: "t1".to_string(),
            state: TestState::Failed,
            cycles: None,
            time_us: None,
            stack_peak: None,
        });
        state.current_running = Some((0, 0));
        let actions = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TargetPanic {
                message: "done",
                file: "main.rs",
                line: 1,
            },
        ));
        assert!(state.exit_loop);
        assert!(matches!(
            actions.as_slice(),
            [
                SessionAction::Send(CommCommand::TryReset),
                SessionAction::PanicRestart
            ]
        ));
    }

    #[test]
    fn ci_ets_state_telemetry_metrics_and_rediscovery() {
        let mut state = SessionState::new();
        discover_two_tests(&mut state);

        state.results.push(TestOutcome {
            suite_name: "suite".to_string(),
            test_name: "t0".to_string(),
            state: TestState::Passed,
            cycles: Some(100),
            time_us: Some(10),
            stack_peak: Some(32),
        });

        let actions = state.handle_message(BridgeMessage::telemetry(
            Telemetry::DiscoveryComplete,
        ));
        assert!(!actions.is_empty());
        assert_eq!(state.run_queue.len(), 0);
        assert_eq!(state.current_running, Some((0, 1)));

        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::MetricReport {
                suite_id: 0,
                test_id: 1,
                cycles: 500,
                time_us: 50,
                stack_peak: 64,
            },
        ));

        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 1,
                state: TestState::Passed,
            },
        ));
        assert!(state.exit_loop);
        assert_eq!(state.results.len(), 2);
        assert_eq!(state.suites[0].tests[0].cycles, Some(100));
        assert_eq!(state.suites[0].tests[0].time_us, Some(10));
        assert_eq!(state.suites[0].tests[0].stack_peak, Some(32));
        assert_eq!(state.suites[0].tests[1].cycles, Some(500));
    }

    #[test]
    fn session_enqueue_and_stop_helpers() {
        let mut state = SessionState::new();
        discover_two_tests(&mut state);

        assert_eq!(state.run_queue, vec![]);
        let action = state.enqueue_all();
        assert!(matches!(
            action,
            Some(SessionAction::Send(CommCommand::RunExecutable {
                suite_id: 0,
                test_id: 0
            }))
        ));
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1)]);

        state.stop();
        assert_eq!(state.current_running, None);
        assert_eq!(state.run_queue, vec![]);

        let action_single = state.enqueue_test(0, 1);
        assert!(matches!(
            action_single,
            Some(SessionAction::Send(CommCommand::RunExecutable {
                suite_id: 0,
                test_id: 1
            }))
        ));
        assert_eq!(state.current_running, Some((0, 1)));
    }
}
