//! Discovery, run queue and session state machine for host-side ETS.

use control_rs_ets::comms::{Command as CommCommand, TestState};
use control_rs_ets::settings::SettingValue;

use crate::bridge::{BridgeMessage, OwnedTelemetry};
use crate::runner::TestOutcome;

/// Flag indicating that all `SettingInfo` items (`0..setting_count-1`) have been received.
pub const SETTINGS_READY: u8 = 0b0000_0100; // 0x04
/// Flag indicating that `SuiteInfo` metadata (name, test/setting counts) has been received.
pub const SUITE_INFO_READY: u8 = 0b0000_0001; // 0x01
/// Complete readiness mask for a test suite (`SUITE_INFO_READY | TESTS_READY | SETTINGS_READY`).
pub const SUITE_READY_MASK: u8 = 0b0000_0111; // 0x07
/// Flag indicating that all `TestInfo` items (`0..test_count-1`) have been received.
pub const TESTS_READY: u8 = 0b0000_0010; // 0x02

/// Pair of `(suite_id, test_id)` identifying a test case.
pub type TestIndex = (u16, u16);

/// Lifecycle phases of a host-side ETS testing session.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum SessionPhase {
    /// Target test discovery in progress. Telemetry is collected directly into suite readiness masks.
    Discovering,
    /// Discovery has been validated. Discovered tests are actively executing or queued.
    Running,
    /// Target panic or reset recovery in progress while unexecuted tests remain.
    Recovering,
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
    /// Doc-comment description of the test case.
    pub description: String,
    /// Current execution state of the test.
    pub state: TestState,
    /// CPU cycles consumed during the test run if completed.
    pub cycles: Option<u64>,
    /// Elapsed time of the test run in microseconds if completed.
    pub time_us: Option<u64>,
    /// Peak stack memory usage in bytes if completed.
    pub stack_peak: Option<u32>,
}

/// Representation of a test suite containing tests and settings.
#[derive(Debug, Clone)]
pub struct SuiteItem {
    /// Identifier assigned to this suite.
    pub suite_id: u16,
    /// Name of the suite.
    pub name: String,
    /// Doc-comment description of the suite.
    pub description: String,
    /// Expected number of tests in the suite from `SuiteInfo`.
    pub test_count: u16,
    /// Expected number of settings in the suite from `SuiteInfo`.
    pub setting_count: u16,
    /// Collection of tests inside this suite.
    pub tests: Vec<TestItem>,
    /// Collection of config settings inside this suite.
    pub settings: Vec<SettingItem>,
    /// Bitmask of initialized components (`SUITE_INFO_READY | TESTS_READY | SETTINGS_READY`).
    pub ready_mask: u8,
    /// Bitmask tracking individual test indices (bit t is set when test t arrives).
    pub test_slots_mask: u64,
    /// Bitmask tracking individual setting indices (bit s is set when setting s arrives).
    pub setting_slots_mask: u64,
}

/// Host-side ETS session state (discovery, run queue, results).
pub struct SessionState {
    /// Current lifecycle phase of the session.
    pub phase: SessionPhase,
    /// Currently executing test (`suite_id`, `test_id`).
    pub current_running: Option<TestIndex>,
    /// Flag signaling whether discovery has been validated (kept in sync with phase).
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

impl SuiteItem {
    /// Creates a new, uninitialized suite descriptor.
    #[must_use]
    pub const fn new(suite_id: u16) -> Self {
        Self {
            suite_id,
            name: String::new(),
            description: String::new(),
            test_count: 0,
            setting_count: 0,
            tests: Vec::new(),
            settings: Vec::new(),
            ready_mask: 0,
            test_slots_mask: 0,
            setting_slots_mask: 0,
        }
    }

    /// Returns true if this suite has received all info, test, and setting frames.
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        (self.ready_mask & SUITE_READY_MASK) == SUITE_READY_MASK
    }

    /// Resets all readiness masks and clears transient items for re-discovery.
    pub fn reset_discovery(&mut self) {
        self.ready_mask = 0;
        self.test_slots_mask = 0;
        self.setting_slots_mask = 0;
        self.tests.clear();
        self.settings.clear();
    }
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

    /// Ensures that `self.suites` contains an entry at index `suite_id`.
    pub fn ensure_suite_slot(&mut self, suite_id: u16) -> &mut SuiteItem {
        let idx = suite_id as usize;
        if self.suites.len() <= idx {
            self.suites.reserve(
                idx.saturating_add(1).saturating_sub(self.suites.len()),
            );
            while self.suites.len() <= idx {
                #[allow(clippy::cast_possible_truncation)]
                let next_id = self.suites.len() as u16;
                self.suites.push(SuiteItem::new(next_id));
            }
        }
        &mut self.suites[idx]
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

    /// Processes an incoming bridge message and updates session state accordingly.
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
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
                    self.ensure_suite_slot(suite_id);
                    let suite = &mut self.suites[suite_id as usize];
                    suite.name = name;
                    suite.description = description;
                    suite.test_count = test_count;
                    suite.setting_count = setting_count;
                    suite.ready_mask |= SUITE_INFO_READY;
                    if test_count == 0 {
                        suite.ready_mask |= TESTS_READY;
                    }
                    if setting_count == 0 {
                        suite.ready_mask |= SETTINGS_READY;
                    }
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
                    self.ensure_suite_slot(suite_id);
                    let s_idx = suite_id as usize;
                    let prev = self.find_outcome(suite_id, test_id);
                    let state = prev.map_or(TestState::Pending, |p| p.state);
                    let cycles = prev.and_then(|p| p.cycles);
                    let time_us = prev.and_then(|p| p.time_us);
                    let stack_peak = prev.and_then(|p| p.stack_peak);

                    let suite = &mut self.suites[s_idx];
                    let t_idx = test_id as usize;
                    if t_idx < 64 {
                        suite.test_slots_mask |= 1u64 << t_idx;
                    }
                    let item = TestItem {
                        suite_id,
                        test_id,
                        suite_name: suite.name.clone(),
                        name,
                        description,
                        state,
                        cycles,
                        time_us,
                        stack_peak,
                    };
                    if let Some(existing) =
                        suite.tests.iter_mut().find(|t| t.test_id == test_id)
                    {
                        *existing = item;
                    } else {
                        suite.tests.push(item);
                    }
                    if suite.test_count > 0
                        && suite.tests.len() == suite.test_count as usize
                    {
                        if suite.test_count <= 64 {
                            let expected = if suite.test_count == 64 {
                                u64::MAX
                            } else {
                                (1u64 << suite.test_count).saturating_sub(1)
                            };
                            if (suite.test_slots_mask & expected) == expected {
                                suite.ready_mask |= TESTS_READY;
                            }
                        } else {
                            suite.ready_mask |= TESTS_READY;
                        }
                    }
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
                    self.ensure_suite_slot(suite_id);
                    let s_idx = suite_id as usize;
                    let suite = &mut self.suites[s_idx];
                    let set_idx = setting_id as usize;
                    if set_idx < 64 {
                        suite.setting_slots_mask |= 1u64 << set_idx;
                    }
                    let item = SettingItem {
                        setting_id,
                        name,
                        description,
                        value,
                    };
                    if let Some(existing) = suite
                        .settings
                        .iter_mut()
                        .find(|s| s.setting_id == setting_id)
                    {
                        *existing = item;
                    } else {
                        suite.settings.push(item);
                    }
                    if suite.setting_count > 0
                        && suite.settings.len() == suite.setting_count as usize
                    {
                        if suite.setting_count <= 64 {
                            let expected = if suite.setting_count == 64 {
                                u64::MAX
                            } else {
                                (1u64 << suite.setting_count).saturating_sub(1)
                            };
                            if (suite.setting_slots_mask & expected) == expected
                            {
                                suite.ready_mask |= SETTINGS_READY;
                            }
                        } else {
                            suite.ready_mask |= SETTINGS_READY;
                        }
                    }
                    Vec::new()
                }
                OwnedTelemetry::DiscoveryComplete => {
                    if self.phase == SessionPhase::Running {
                        return Vec::new();
                    }
                    if self.suites.is_empty()
                        || !self.suites.iter().all(SuiteItem::is_ready)
                    {
                        self.log(
                            "Discovery validation failed (incomplete or non-contiguous slots). Retrying discovery.\n",
                        );
                        for s in &mut self.suites {
                            s.reset_discovery();
                        }
                        return Vec::new();
                    }

                    self.run_queue.clear();
                    for suite in &self.suites {
                        for test in &suite.tests {
                            if self
                                .find_outcome(suite.suite_id, test.test_id)
                                .is_none()
                            {
                                self.run_queue
                                    .push((suite.suite_id, test.test_id));
                            }
                        }
                    }

                    self.phase = SessionPhase::Running;
                    self.discovery_complete = true;

                    self.start_next_or_exit().into_iter().collect()
                }
                OwnedTelemetry::TestStateChange {
                    suite_id,
                    test_id,
                    state: new_state,
                } => {
                    let s_idx = suite_id as usize;
                    if let Some(suite) = self.suites.get_mut(s_idx)
                        && let Some(test) = suite
                            .tests
                            .iter_mut()
                            .find(|t| t.test_id == test_id)
                    {
                        test.state = new_state;
                    }

                    if new_state == TestState::Failed {
                        let suite_name = self
                            .suites
                            .get(s_idx)
                            .map_or_else(String::new, |s| s.name.clone());
                        let test_name = self
                            .suites
                            .get(s_idx)
                            .and_then(|s| {
                                s.tests.iter().find(|t| t.test_id == test_id)
                            })
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
                    if let Some(suite) = self.suites.get_mut(s_idx)
                        && let Some(test) = suite
                            .tests
                            .iter_mut()
                            .find(|t| t.test_id == test_id)
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
                        .and_then(|s| {
                            s.tests.iter().find(|t| t.test_id == test_id)
                        })
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

                    match self.phase {
                        SessionPhase::Discovering => {
                            self.log(
                                "Target panic occurred during discovery phase.\n",
                            );
                        }
                        SessionPhase::Running => {
                            if let Some((s_id, t_id)) = self.current_running {
                                let s_idx = s_id as usize;
                                if let Some(suite) = self.suites.get_mut(s_idx)
                                    && let Some(test) = suite
                                        .tests
                                        .iter_mut()
                                        .find(|t| t.test_id == t_id)
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
                            } else {
                                self.log("Target panic occurred outside test execution in running phase.\n");
                            }
                        }
                        SessionPhase::Recovering => {
                            self.log(
                                "Target panic occurred during recovery phase.\n",
                            );
                        }
                    }

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

                    self.current_running = None;
                    for s in &mut self.suites {
                        s.reset_discovery();
                    }

                    self.discovery_complete = false;
                    self.phase = SessionPhase::Recovering;
                    if !remaining_to_run {
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
    fn test_bitmask_discovery_and_readiness() {
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

        // Suite 0 is fully ready in suites vector
        assert_eq!(state.suites.len(), 1);
        assert!(state.suites[0].is_ready());
        assert_eq!(state.suites[0].ready_mask, SUITE_READY_MASK);
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
    }

    #[test]
    fn test_validation_rejects_incomplete_test_slots() {
        let mut state = SessionState::new();

        // Send SuiteInfo expecting 3 tests
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::SuiteInfo {
                suite_id: 0,
                name: "S0".to_string(),
                description: "desc".to_string(),
                test_count: 3,
                setting_count: 0,
            },
        ));
        // Send test 0 and test 2 (missing test 1)
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "t0".to_string(),
                description: "d".to_string(),
            },
        ));
        let _ = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::TestInfo {
                suite_id: 0,
                test_id: 2,
                name: "t2".to_string(),
                description: "d".to_string(),
            },
        ));

        assert!(!state.suites[0].is_ready());
        assert_eq!(state.suites[0].ready_mask & TESTS_READY, 0);

        // DiscoveryComplete fails validation
        let actions = state.handle_message(BridgeMessage::Telemetry(
            OwnedTelemetry::DiscoveryComplete,
        ));
        assert!(actions.is_empty());
        assert_eq!(state.phase, SessionPhase::Discovering);
        assert!(!state.discovery_complete);
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
