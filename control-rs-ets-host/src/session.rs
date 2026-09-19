//! Discovery, run queue and session state machine for host-side ETS.

use std::collections::BTreeMap;

use control_rs_ets::comms::{Command as CommCommand, TestState};
use control_rs_ets::settings::SettingValue;

use crate::bridge::{BridgeMessage, OwnedTelemetry};
use crate::runner::TestOutcome;

/// Pair of `(suite_id, test_id)` identifying a test case.
pub type TestIndex = (u16, u16);

/// Lifecycle phases of a host-side ETS testing session.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum SessionPhase {
    /// Target test discovery in progress. Telemetry is collected into a staging scratch map.
    Discovering,
    /// Discovery has been validated and committed. Discovered tests are actively executing or queued.
    Running,
    /// Target panic or reset recovery in progress while unexecuted tests remain.
    Recovering,
}

/// Representation of a single test case under discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestItem {
    /// Identifier assigned to this test within its suite.
    pub test_id: u16,
    /// Identifier of the suite containing this test case.
    pub suite_id: u16,
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
    /// Identifier assigned to this setting within its suite.
    pub setting_id: u16,
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
    /// Identifier assigned to this suite.
    pub suite_id: u16,
    /// Name of the suite.
    pub name: String,
    /// Collection of tests inside this suite.
    pub tests: Vec<TestItem>,
    /// Collection of config settings inside this suite.
    pub settings: Vec<SettingItem>,
}

/// In-progress test case metadata staged during discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScratchTest {
    /// Name of the test case.
    pub name: String,
    /// Doc-comment description of the test case.
    pub description: String,
}

/// In-progress setting metadata staged during discovery.
#[derive(Debug, Clone)]
pub struct ScratchSetting {
    /// Name of the setting.
    pub name: String,
    /// Doc-comment description of the setting.
    pub description: String,
    /// Current value of the setting.
    pub value: SettingValue,
}

/// In-progress suite metadata staged during discovery.
#[derive(Debug, Clone, Default)]
pub struct ScratchSuite {
    /// Name of the suite.
    pub name: Option<String>,
    /// Doc-comment description of the suite.
    pub description: Option<String>,
    /// Expected number of tests in the suite from `SuiteInfo`.
    pub test_count: Option<u16>,
    /// Expected number of settings in the suite from `SuiteInfo`.
    pub setting_count: Option<u16>,
    /// Staged test cases keyed by `test_id`.
    pub tests: BTreeMap<u16, ScratchTest>,
    /// Staged settings keyed by `setting_id`.
    pub settings: BTreeMap<u16, ScratchSetting>,
}

/// Scratch map for buffering incoming discovery telemetry before atomic validation and commit.
#[derive(Debug, Clone, Default)]
pub struct DiscoveryScratch {
    /// Staged suites keyed by `suite_id`.
    pub suites: BTreeMap<u16, ScratchSuite>,
}

/// Host-side ETS session state (discovery, run queue, results).
pub struct SessionState {
    /// Current lifecycle phase of the session.
    pub phase: SessionPhase,
    /// Staged discovery metadata buffer.
    pub discovery_scratch: DiscoveryScratch,
    /// Currently executing test (`suite_id`, `test_id`).
    pub current_running: Option<TestIndex>,
    /// Flag signaling whether discovery has been committed (kept in sync with phase).
    pub discovery_complete: bool,
    /// Flag signaling that test execution has finished or reached terminal state.
    pub exit_loop: bool,
    /// Buffer of log and diagnostic messages collected from the target.
    pub logs: String,
    /// Collection of test results recorded so far.
    pub results: Vec<TestOutcome>,
    /// Queue of tests pending execution `(suite_id, test_id)`.
    pub run_queue: Vec<TestIndex>,
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
            suite_id: self.suite_id,
            test_id: self.test_id,
            suite_name: self.suite_name,
            test_name: self.name,
            state: self.state,
            cycles: self.cycles,
            time_us: self.time_us,
            stack_peak: self.stack_peak,
        }
    }
}

impl DiscoveryScratch {
    /// Creates a new, empty discovery scratch space.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            suites: BTreeMap::new(),
        }
    }

    /// Clears all staged discovery metadata.
    pub fn clear(&mut self) {
        self.suites.clear();
    }

    /// Validates that the staged discovery state is complete and contiguous.
    ///
    /// Requires:
    /// - Contiguous suite IDs `0..N-1` where `N = suites.len()`.
    /// - For each suite, `SuiteInfo` was received (`name`, `test_count`, `setting_count` present).
    /// - For each suite, `tests.len() == test_count` and all test IDs `0..test_count-1` are present.
    /// - For each suite, `settings.len() == setting_count` and all setting IDs `0..setting_count-1` are present.
    #[must_use]
    pub fn validate(&self) -> bool {
        let suite_count = self.suites.len();
        for s_idx in 0..suite_count {
            let Ok(s_id) = u16::try_from(s_idx) else {
                return false;
            };
            let Some(suite) = self.suites.get(&s_id) else {
                return false;
            };
            let (Some(_name), Some(test_count), Some(setting_count)) =
                (&suite.name, suite.test_count, suite.setting_count)
            else {
                return false;
            };
            if suite.tests.len() != test_count as usize {
                return false;
            }
            for t_id in 0..test_count {
                if !suite.tests.contains_key(&t_id) {
                    return false;
                }
            }
            if suite.settings.len() != setting_count as usize {
                return false;
            }
            for set_id in 0..setting_count {
                if !suite.settings.contains_key(&set_id) {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionState {
    /// Creates a new, empty session state in the discovering phase.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            phase: SessionPhase::Discovering,
            discovery_scratch: DiscoveryScratch::new(),
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

    /// Returns a reference to a recorded outcome matching `(suite_id, test_id)`.
    #[must_use]
    pub fn find_outcome(
        &self,
        suite_id: u16,
        test_id: u16,
    ) -> Option<&TestOutcome> {
        self.results
            .iter()
            .find(|r| r.suite_id == suite_id && r.test_id == test_id)
    }

    /// Returns a mutable reference to a recorded outcome matching `(suite_id, test_id)`.
    pub fn find_outcome_mut(
        &mut self,
        suite_id: u16,
        test_id: u16,
    ) -> Option<&mut TestOutcome> {
        self.results
            .iter_mut()
            .find(|r| r.suite_id == suite_id && r.test_id == test_id)
    }

    /// Records or updates a test outcome keyed by `(suite_id, test_id)`.
    pub fn record_outcome(&mut self, outcome: TestOutcome) {
        if let Some(existing) =
            self.find_outcome_mut(outcome.suite_id, outcome.test_id)
        {
            *existing = outcome;
        } else {
            self.results.push(outcome);
        }
    }

    /// Enqueues all discovered tests across all suites for execution.
    ///
    /// When a case is already in flight (`current_running`), that case is
    /// omitted from the rebuilt queue so the next metric report does not
    /// immediately re-run it.
    pub fn enqueue_all(&mut self) -> Option<SessionAction> {
        if self.phase != SessionPhase::Running {
            return None;
        }
        let in_flight = self.current_running;
        self.run_queue.clear();
        for suite in &self.suites {
            for test in &suite.tests {
                let id = (suite.suite_id, test.test_id);
                if in_flight == Some(id) {
                    continue;
                }
                self.run_queue.push(id);
            }
        }
        if self.current_running.is_none() {
            self.start_next_or_exit()
        } else {
            None
        }
    }

    /// Returns cases that have not finished when a run aborts.
    ///
    /// Includes the in-flight `current_running` case when it is not already
    /// recorded in `results`, then the remaining `run_queue` entries.
    #[must_use]
    pub fn pending_cases(&self) -> Vec<TestIndex> {
        let mut pending = Vec::new();
        if let Some((s_id, t_id)) = self.current_running
            && self.find_outcome(s_id, t_id).is_none()
        {
            pending.push((s_id, t_id));
        }
        for id in &self.run_queue {
            if !pending.contains(id) {
                pending.push(*id);
            }
        }
        pending
    }

    /// Enqueues a specific test for execution.
    pub fn enqueue_test(
        &mut self,
        suite_id: u16,
        test_id: u16,
    ) -> Option<SessionAction> {
        if self.phase != SessionPhase::Running {
            return None;
        }
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

    /// Atomically commits staged discovery metadata into `suites` and dispatches the initial test.
    fn commit_discovery(&mut self) -> Vec<SessionAction> {
        let mut new_suites =
            Vec::with_capacity(self.discovery_scratch.suites.len());
        for (s_id, scratch_suite) in &self.discovery_scratch.suites {
            let suite_name = scratch_suite.name.clone().unwrap_or_default();
            let mut tests = Vec::with_capacity(scratch_suite.tests.len());
            for (t_id, scratch_test) in &scratch_suite.tests {
                let prev = self.find_outcome(*s_id, *t_id);
                let state = prev.map_or(TestState::Pending, |p| p.state);
                let cycles = prev.and_then(|p| p.cycles);
                let time_us = prev.and_then(|p| p.time_us);
                let stack_peak = prev.and_then(|p| p.stack_peak);

                tests.push(TestItem {
                    suite_id: *s_id,
                    test_id: *t_id,
                    suite_name: suite_name.clone(),
                    name: scratch_test.name.clone(),
                    state,
                    cycles,
                    time_us,
                    stack_peak,
                });
            }

            let mut settings = Vec::with_capacity(scratch_suite.settings.len());
            for (set_id, scratch_setting) in &scratch_suite.settings {
                settings.push(SettingItem {
                    setting_id: *set_id,
                    name: scratch_setting.name.clone(),
                    description: scratch_setting.description.clone(),
                    value: scratch_setting.value,
                });
            }

            new_suites.push(SuiteItem {
                suite_id: *s_id,
                name: suite_name,
                tests,
                settings,
            });
        }

        self.suites = new_suites;
        self.discovery_scratch.clear();
        self.run_queue.clear();

        for suite in &self.suites {
            for test in &suite.tests {
                if self.find_outcome(suite.suite_id, test.test_id).is_none() {
                    self.run_queue.push((suite.suite_id, test.test_id));
                }
            }
        }

        self.phase = SessionPhase::Running;
        self.discovery_complete = true;

        self.start_next_or_exit().into_iter().collect()
    }

    /// Processes an incoming bridge message and updates session state accordingly.
    #[allow(clippy::too_many_lines)]
    pub fn handle_message(&mut self, msg: BridgeMessage) -> Vec<SessionAction> {
        match msg {
            BridgeMessage::RawConsole(line) => {
                let formatted = format!("      [ETS] {line}\n");
                self.log(&formatted);
                Vec::new()
            }
            BridgeMessage::Telemetry(telemetry) => match telemetry {
                OwnedTelemetry::SuiteInfo {
                    suite_id,
                    name,
                    description,
                    test_count,
                    setting_count,
                } => {
                    if self.phase == SessionPhase::Running {
                        return Vec::new();
                    }
                    let suite = self
                        .discovery_scratch
                        .suites
                        .entry(suite_id)
                        .or_default();
                    suite.name = Some(name);
                    suite.description = Some(description);
                    suite.test_count = Some(test_count);
                    suite.setting_count = Some(setting_count);
                    Vec::new()
                }
                OwnedTelemetry::TestInfo {
                    suite_id,
                    test_id,
                    name,
                    description,
                } => {
                    if self.phase == SessionPhase::Running {
                        return Vec::new();
                    }
                    let suite = self
                        .discovery_scratch
                        .suites
                        .entry(suite_id)
                        .or_default();
                    suite
                        .tests
                        .insert(test_id, ScratchTest { name, description });
                    Vec::new()
                }
                OwnedTelemetry::SettingInfo {
                    suite_id,
                    setting_id,
                    name,
                    value,
                    description,
                } => {
                    if self.phase == SessionPhase::Running {
                        return Vec::new();
                    }
                    let suite = self
                        .discovery_scratch
                        .suites
                        .entry(suite_id)
                        .or_default();
                    suite.settings.insert(
                        setting_id,
                        ScratchSetting {
                            name,
                            description,
                            value,
                        },
                    );
                    Vec::new()
                }
                OwnedTelemetry::DiscoveryComplete => {
                    if self.phase == SessionPhase::Running {
                        // In-flight duplicate complete while tests already running
                        return Vec::new();
                    }
                    if !self.discovery_scratch.validate() {
                        self.log(
                            "Discovery validation failed (incomplete or non-contiguous slots). Retrying discovery.\n",
                        );
                        self.discovery_scratch.clear();
                        return Vec::new();
                    }
                    self.commit_discovery()
                }
                OwnedTelemetry::TestStateChange {
                    suite_id,
                    test_id,
                    state: new_state,
                } => {
                    let s_idx = suite_id as usize;
                    let t_idx = test_id as usize;
                    if let Some(suite) = self.suites.get_mut(s_idx)
                        && let Some(test) = suite.tests.get_mut(t_idx)
                    {
                        test.state = new_state;
                    }

                    if new_state == TestState::Failed {
                        // Target `handle_failure` always emits Failed then
                        // TargetPanic. Advancing the queue here would make the
                        // subsequent TargetPanic attribute the crash to the
                        // next queued test and skip running it.
                        let suite_name = self
                            .suites
                            .get(s_idx)
                            .map_or_else(String::new, |s| s.name.clone());
                        let test_name = self
                            .suites
                            .get(s_idx)
                            .and_then(|s| s.tests.get(t_idx))
                            .map_or_else(String::new, |t| t.name.clone());
                        if self.find_outcome(suite_id, test_id).is_none() {
                            self.record_outcome(TestOutcome {
                                suite_id,
                                test_id,
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
                    let s_idx = suite_id as usize;
                    let t_idx = test_id as usize;
                    if let Some(suite) = self.suites.get_mut(s_idx)
                        && let Some(test) = suite.tests.get_mut(t_idx)
                    {
                        test.state = TestState::Passed;
                        test.cycles = Some(cycles);
                        test.time_us = Some(time_us);
                        test.stack_peak = Some(stack_peak);
                    }
                    let suite_name = self
                        .suites
                        .get(s_idx)
                        .map_or_else(String::new, |s| s.name.clone());
                    let test_name = self
                        .suites
                        .get(s_idx)
                        .and_then(|s| s.tests.get(t_idx))
                        .map_or_else(String::new, |t| t.name.clone());
                    self.record_outcome(TestOutcome {
                        suite_id,
                        test_id,
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
                        if let Some(suite) = self.suites.get_mut(s_idx)
                            && let Some(test) = suite.tests.get_mut(t_idx)
                        {
                            test.state = TestState::Failed;
                            let suite_name = suite.name.clone();
                            let test_name = test.name.clone();
                            if self.find_outcome(s_id, t_id).is_none() {
                                self.record_outcome(TestOutcome {
                                    suite_id: s_id,
                                    test_id: t_id,
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
                    self.discovery_scratch.clear();

                    let remaining_to_run = match self.phase {
                        SessionPhase::Discovering => true,
                        SessionPhase::Running | SessionPhase::Recovering => {
                            self.suites.iter().any(|suite| {
                                suite.tests.iter().any(|test| {
                                    self.find_outcome(
                                        suite.suite_id,
                                        test.test_id,
                                    )
                                    .is_none()
                                })
                            })
                        }
                    };

                    self.discovery_complete = false;
                    let restart_action = SessionAction::PanicRestart;

                    if remaining_to_run {
                        self.phase = SessionPhase::Recovering;
                    } else {
                        self.exit_loop = true;
                    }
                    vec![
                        SessionAction::Send(CommCommand::TryReset),
                        restart_action,
                    ]
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::indexing_slicing,
        clippy::iter_on_single_items,
        clippy::too_many_lines,
        clippy::unwrap_used
    )]

    use super::*;

    fn make_test_suite_telemetry(
        suite_id: u16,
        suite_name: &str,
        test_count: u16,
        setting_count: u16,
    ) -> Vec<OwnedTelemetry> {
        let mut frames = Vec::new();
        frames.push(OwnedTelemetry::SuiteInfo {
            suite_id,
            name: suite_name.to_string(),
            description: format!("Description for {suite_name}"),
            test_count,
            setting_count,
        });
        for t in 0..test_count {
            frames.push(OwnedTelemetry::TestInfo {
                suite_id,
                test_id: t,
                name: format!("test_{t}"),
                description: format!("Desc for test {t}"),
            });
        }
        for s in 0..setting_count {
            frames.push(OwnedTelemetry::SettingInfo {
                suite_id,
                setting_id: s,
                name: format!("setting_{s}"),
                description: format!("Desc for setting {s}"),
                value: SettingValue::U8(s as u8),
            });
        }
        frames
    }

    #[test]
    fn test_transactional_discovery_buffering_and_validation() {
        let mut state = SessionState::new();
        assert_eq!(state.phase, SessionPhase::Discovering);
        assert!(!state.discovery_complete);
        assert!(state.suites.is_empty());

        let frames = make_test_suite_telemetry(0, "MathSuite", 2, 1);
        for f in frames {
            let actions =
                state.handle_message(BridgeMessage::Telemetry(f.clone()));
            assert!(actions.is_empty());
        }

        // Discovery scratch is populated, suites remains untouched
        assert_eq!(state.discovery_scratch.suites.len(), 1);
        assert!(state.suites.is_empty());
        assert!(!state.discovery_complete);

        // DiscoveryComplete commits atomically
        let actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            SessionAction::Send(CommCommand::RunExecutable {
                suite_id: 0,
                test_id: 0
            })
        ));

        assert_eq!(state.phase, SessionPhase::Running);
        assert!(state.discovery_complete);
        assert_eq!(state.suites.len(), 1);
        assert_eq!(state.suites[0].tests.len(), 2);
        assert_eq!(state.suites[0].settings.len(), 1);
        assert_eq!(state.run_queue, vec![(0, 1)]);
        assert_eq!(state.current_running, Some((0, 0)));
        assert!(state.discovery_scratch.suites.is_empty());
    }

    #[test]
    fn test_validation_rejects_missing_or_non_contiguous_suites() {
        let mut scratch = DiscoveryScratch::new();

        // Suite 1 present, but suite 0 missing -> non-contiguous
        let s1 = ScratchSuite {
            name: Some("S1".to_string()),
            description: Some("desc".to_string()),
            test_count: Some(1),
            setting_count: Some(0),
            tests: [(
                0,
                ScratchTest {
                    name: "t0".to_string(),
                    description: "d".to_string(),
                },
            )]
            .into_iter()
            .collect(),
            settings: BTreeMap::new(),
        };
        scratch.suites.insert(1, s1);
        assert!(!scratch.validate());

        // Now add suite 0 -> contiguous 0..2
        let s0 = ScratchSuite {
            name: Some("S0".to_string()),
            description: Some("desc".to_string()),
            test_count: Some(1),
            setting_count: Some(0),
            tests: [(
                0,
                ScratchTest {
                    name: "t0".to_string(),
                    description: "d".to_string(),
                },
            )]
            .into_iter()
            .collect(),
            settings: BTreeMap::new(),
        };
        scratch.suites.insert(0, s0);
        assert!(scratch.validate());
    }

    #[test]
    fn test_validation_rejects_incomplete_test_slots() {
        let mut scratch = DiscoveryScratch::new();
        let mut s0 = ScratchSuite {
            name: Some("S0".to_string()),
            description: Some("desc".to_string()),
            test_count: Some(3),
            setting_count: Some(0),
            tests: BTreeMap::new(),
            settings: BTreeMap::new(),
        };
        s0.tests.insert(
            0,
            ScratchTest {
                name: "t0".to_string(),
                description: "d".to_string(),
            },
        );
        s0.tests.insert(
            2,
            ScratchTest {
                name: "t2".to_string(),
                description: "d".to_string(),
            },
        ); // missing slot 1!
        scratch.suites.insert(0, s0);

        assert!(!scratch.validate());
    }

    #[test]
    fn test_validation_failure_resets_scratch_and_stays_discovering() {
        let mut state = SessionState::new();

        // Send incomplete suite (expects 2 tests, only send test 0)
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::SuiteInfo {
                suite_id: 0,
                name: "Incomplete".to_string(),
                description: "desc".to_string(),
                test_count: 2,
                setting_count: 0,
            },
        ));
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "t0".to_string(),
                description: "d".to_string(),
            },
        ));

        // DiscoveryComplete fails validation
        let actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));
        assert!(actions.is_empty());
        assert_eq!(state.phase, SessionPhase::Discovering);
        assert!(!state.discovery_complete);
        assert!(state.discovery_scratch.suites.is_empty());
        assert!(state.suites.is_empty());
        assert!(state.logs.contains("Discovery validation failed"));
    }

    #[test]
    fn ci_ets_state_runs_queue_and_records_pass_fail() {
        let mut state = SessionState::new();

        let frames = make_test_suite_telemetry(0, "Suite0", 2, 0);
        for f in frames {
            let _ = state.handle_message(BridgeMessage::Telemetry(f));
        }
        let actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));
        assert_eq!(actions.len(), 1);
        assert_eq!(state.current_running, Some((0, 0)));

        // Test 0 completes with metrics
        let actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 1000,
                time_us: 50,
                stack_peak: 256,
            },
        ));
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            SessionAction::Send(CommCommand::RunExecutable {
                suite_id: 0,
                test_id: 1
            })
        ));
        assert_eq!(state.current_running, Some((0, 1)));
        assert_eq!(state.results.len(), 1);
        assert_eq!(state.results[0].state, TestState::Passed);
        assert_eq!(state.results[0].suite_id, 0);
        assert_eq!(state.results[0].test_id, 0);

        // Test 1 fails
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::TestStateChange {
                suite_id: 0,
                test_id: 1,
                state: TestState::Failed,
            },
        ));
        assert_eq!(state.results.len(), 2);
        assert_eq!(state.results[1].state, TestState::Failed);
        assert_eq!(state.results[1].suite_id, 0);
        assert_eq!(state.results[1].test_id, 1);
        // Failed state does not advance queue until TargetPanic or complete
        assert_eq!(state.current_running, Some((0, 1)));
    }

    #[test]
    fn failed_before_target_panic_does_not_blame_next_test() {
        let mut state = SessionState::new();

        let frames = make_test_suite_telemetry(0, "Suite0", 3, 0);
        for f in frames {
            let _ = state.handle_message(BridgeMessage::Telemetry(f));
        }
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1), (0, 2)]);

        // Case 0 fails
        let actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Failed,
            },
        ));
        assert!(actions.is_empty());
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1), (0, 2)]);

        // TargetPanic arrives immediately after
        let panic_actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::TargetPanic {
                message: "assertion failed".to_string(),
                file: "src/test.rs".to_string(),
                line: 42,
            },
        ));
        assert_eq!(panic_actions.len(), 2);
        assert!(matches!(
            panic_actions[0],
            SessionAction::Send(CommCommand::TryReset)
        ));
        assert!(matches!(panic_actions[1], SessionAction::PanicRestart));

        assert_eq!(state.phase, SessionPhase::Recovering);
        assert_eq!(state.current_running, None);
        assert_eq!(state.results.len(), 1);
        assert_eq!(state.results[0].suite_id, 0);
        assert_eq!(state.results[0].test_id, 0);
        assert_eq!(state.results[0].state, TestState::Failed);
    }

    #[test]
    fn duplicate_discovery_complete_does_not_requeue_in_flight() {
        let mut state = SessionState::new();
        let frames = make_test_suite_telemetry(0, "Suite0", 2, 0);
        for f in frames {
            let _ = state.handle_message(BridgeMessage::Telemetry(f));
        }
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1)]);

        // Second duplicate DiscoveryComplete arriving late while running
        let actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));
        assert!(actions.is_empty());
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1)]);
    }

    #[test]
    fn enqueue_all_while_running_excludes_in_flight_case() {
        let mut state = SessionState::new();
        let frames = make_test_suite_telemetry(0, "Suite0", 3, 0);
        for f in frames {
            let _ = state.handle_message(BridgeMessage::Telemetry(f));
        }
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));
        assert_eq!(state.current_running, Some((0, 0)));

        let _ = state.enqueue_all();
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1), (0, 2)]);
    }

    #[test]
    fn pre_discovery_target_panic_does_not_false_drain() {
        let mut state = SessionState::new();
        let actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::TargetPanic {
                message: "early boot panic".to_string(),
                file: "main.rs".to_string(),
                line: 10,
            },
        ));
        assert!(!state.exit_loop);
        assert_eq!(state.phase, SessionPhase::Recovering);
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn ci_ets_state_telemetry_metrics_and_rediscovery() {
        let mut state = SessionState::new();
        let frames = make_test_suite_telemetry(0, "S0", 3, 0);
        for f in frames {
            let _ = state.handle_message(BridgeMessage::Telemetry(f));
        }
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));

        // Pass case 0
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 100,
                time_us: 10,
                stack_peak: 32,
            },
        ));
        assert_eq!(state.current_running, Some((0, 1)));

        // Panic on case 1 (case 2 remains)
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::TargetPanic {
                message: "boom".to_string(),
                file: "t.rs".to_string(),
                line: 1,
            },
        ));
        assert_eq!(state.phase, SessionPhase::Recovering);
        assert!(!state.exit_loop);

        // Rediscover
        let frames2 = make_test_suite_telemetry(0, "S0", 3, 0);
        for f in frames2 {
            let _ = state.handle_message(BridgeMessage::Telemetry(f));
        }
        let actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));

        // Case 0 and 1 preserved, Case 2 dispatched
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            SessionAction::Send(CommCommand::RunExecutable {
                suite_id: 0,
                test_id: 2
            })
        ));
        assert_eq!(state.phase, SessionPhase::Running);
        assert_eq!(state.current_running, Some((0, 2)));

        // Pass case 2
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::MetricReport {
                suite_id: 0,
                test_id: 2,
                cycles: 200,
                time_us: 20,
                stack_peak: 32,
            },
        ));
        assert!(state.exit_loop);
        assert_eq!(state.results.len(), 3);
        assert_eq!(state.results[0].state, TestState::Passed);
        assert_eq!(state.results[1].state, TestState::Failed);
        assert_eq!(state.results[2].state, TestState::Passed);
    }

    #[test]
    fn pending_cases_includes_in_flight_before_queue() {
        let mut state = SessionState::new();
        let frames = make_test_suite_telemetry(0, "S0", 3, 0);
        for f in frames {
            let _ = state.handle_message(BridgeMessage::Telemetry(f));
        }
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1), (0, 2)]);
        assert_eq!(state.pending_cases(), vec![(0, 0), (0, 1), (0, 2)]);
    }

    #[test]
    fn session_enqueue_and_stop_helpers() {
        let mut state = SessionState::new();
        let frames = make_test_suite_telemetry(0, "S0", 2, 0);
        for f in frames {
            let _ = state.handle_message(BridgeMessage::Telemetry(f));
        }
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));

        state.stop();
        assert_eq!(state.current_running, None);
        assert!(state.run_queue.is_empty());

        let action = state.enqueue_test(0, 1);
        assert_eq!(state.current_running, Some((0, 1)));
        assert!(matches!(
            action,
            Some(SessionAction::Send(CommCommand::RunExecutable {
                suite_id: 0,
                test_id: 1
            }))
        ));
    }
}
