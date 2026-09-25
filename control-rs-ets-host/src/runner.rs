//! Headless execution driver for running ETS tests to completion on a target.

use std::thread;
use std::time::{Duration, Instant};

use control_rs_ets::comms::{
    Command as CommCommand, PROTOCOL_VERSION, TestState,
};

use crate::bridge::ETSBridge;
use crate::error::HostError;
use crate::session::{SessionAction, SessionState, TestIndex};
use crate::target::Target;

/// Execution options controlling timeout and retry parameters for headless runs.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub struct RunOptions {
    /// Maximum wall-clock duration for the entire test session.
    pub timeout: Duration,
    /// Maximum allowed target resets or reconnection attempts before aborting.
    pub max_resets: u32,
}

/// The result and performance telemetry of an individual test case.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TestOutcome {
    /// Identifier of the test suite.
    pub suite_id: u16,
    /// Identifier of the test case within the suite.
    pub test_id: u16,
    /// Suite namespace of the test.
    pub suite_name: String,
    /// Identifier name of the test.
    pub test_name: String,
    /// Final execution state (Passed or Failed).
    pub state: TestState,
    /// CPU cycles consumed if completed.
    pub cycles: Option<u64>,
    /// Elapsed time in microseconds if completed.
    pub time_us: Option<u64>,
    /// Stack peak water-mark in bytes if completed.
    pub stack_peak: Option<u32>,
}

/// Recorded execution summary of an ETS test run session.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RunRecord {
    /// Test outcomes collected during the session.
    pub results: Vec<TestOutcome>,
    /// Tests that remained pending or unexecuted when the run ended `(suite_id, test_id)`.
    pub pending: Vec<TestIndex>,
    /// Number of target resets performed during execution.
    pub resets: u32,
    /// Terminal abort condition, if the run was aborted (`None` when drained).
    pub abort: Option<Completion>,
    /// Total wall-clock time elapsed during the run.
    pub elapsed: Duration,
    /// Captured console and log output from the target.
    pub console: String,
}

/// Terminal outcome condition of an ETS run.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum Completion {
    /// All queued test cases completed execution.
    Drained,
    /// Wall-clock timeout expired before all tests completed.
    TimedOut,
    /// Maximum allowed reset cycles were exhausted.
    ResetBudgetExhausted,
    /// A command could not be written to the target.
    SendFailed,
    /// Panic recovery could not reopen the transport.
    ReconnectFailed,
    /// Target process exited before all queued tests finished.
    TargetExited,
}

/// Next step after `PanicRestart` terminates the current bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterPanicRestart {
    /// Suite finished; leave the headless loop with `abort: None`.
    Drained,
    /// Remaining work exists but the reset budget is spent.
    ResetBudgetExhausted,
    /// Remaining work exists; reopen the transport and rediscover.
    Reconnect,
}

/// Why a headless run stopped, plus an optional note for the console log.
#[derive(Debug, Default)]
struct RunEnd {
    /// Terminal abort condition (`None` when the run drained).
    abort: Option<Completion>,
    /// Line appended to the captured console output.
    note: Option<String>,
}

/// Live state of one headless run: the link, the session and the counters.
struct HeadlessRun<'t> {
    bridge: ETSBridge,
    target: &'t Target,
    options: RunOptions,
    state: SessionState,
    resets: u32,
    start_time: Instant,
    last_send: Instant,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_resets: 3,
        }
    }
}

impl RunRecord {
    /// Returns the terminal completion status.
    #[must_use]
    pub const fn completion(&self) -> Completion {
        match self.abort {
            Some(c) => c,
            None => Completion::Drained,
        }
    }
}

impl<'t> HeadlessRun<'t> {
    /// Connects to `target` and prepares an empty session.
    fn start(
        target: &'t Target,
        options: RunOptions,
    ) -> Result<Self, HostError> {
        Ok(Self {
            bridge: ETSBridge::new(target.clone(), true)?,
            target,
            options,
            state: SessionState::new(),
            resets: 0,
            start_time: Instant::now(),
            last_send: Instant::now(),
        })
    }

    /// Drives discovery and execution until the session drains (`Ok`) or
    /// aborts (`Err` with the reason).
    fn drive(&mut self) -> Result<(), RunEnd> {
        send_discovery(&mut self.bridge)
            .map_err(|e| RunEnd::send_failed(&e))?;

        while !self.state.exit_loop {
            if self.start_time.elapsed() > self.options.timeout {
                return Err(RunEnd::aborted(Completion::TimedOut));
            }
            self.retry_discovery()?;
            while let Ok(msg) = self.bridge.receiver().try_recv() {
                let actions = self.state.handle_message(msg);
                self.apply_actions(actions)?;
            }
            if self.check_target_exit()? {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        Ok(())
    }

    /// Re-sends discovery every 500 ms until the catalog is complete.
    ///
    /// Serial targets that miss the post-panic `TryReset` spin forever in
    /// `handle_failure` and ignore `ListSuites`; rediscovery must keep
    /// offering `TryReset`.
    fn retry_discovery(&mut self) -> Result<(), RunEnd> {
        if !self.state.discovery_complete
            && self.last_send.elapsed() > Duration::from_millis(500)
        {
            send_discovery(&mut self.bridge)
                .map_err(|e| RunEnd::send_failed(&e))?;
            self.last_send = Instant::now();
        }
        Ok(())
    }

    /// Executes the actions produced by one message.
    fn apply_actions(
        &mut self,
        actions: Vec<SessionAction>,
    ) -> Result<(), RunEnd> {
        for action in actions {
            match action {
                SessionAction::Send(cmd) => {
                    self.bridge
                        .send_command(&cmd)
                        .map_err(|e| RunEnd::send_failed(&e))?;
                }
                SessionAction::PanicRestart => {
                    if self.restart_after_panic()? == AfterPanicRestart::Drained
                    {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    /// Kills the target after a panic and reconnects when budget remains.
    ///
    /// `TargetPanic` clears `discovery_complete` and may set `exit_loop`
    /// when no cases remain. Drain is preferred over reset-budget or
    /// `TargetExited` for that path.
    fn restart_after_panic(&mut self) -> Result<AfterPanicRestart, RunEnd> {
        self.resets = self.resets.saturating_add(1);
        thread::sleep(Duration::from_millis(50));
        self.bridge.terminate();
        thread::sleep(Duration::from_secs(1));

        let next = after_panic_restart(
            self.state.exit_loop,
            self.resets,
            self.options.max_resets,
        );
        match next {
            AfterPanicRestart::Drained => {}
            AfterPanicRestart::ResetBudgetExhausted => {
                return Err(RunEnd::aborted(Completion::ResetBudgetExhausted));
            }
            AfterPanicRestart::Reconnect => {
                self.bridge = ETSBridge::new(self.target.clone(), true)
                    .map_err(|e| {
                        RunEnd::aborted_with(
                            Completion::ReconnectFailed,
                            format!("reconnect failed: {e}"),
                        )
                    })?;
                send_discovery(&mut self.bridge)
                    .map_err(|e| RunEnd::send_failed(&e))?;
                self.last_send = Instant::now();
            }
        }
        Ok(next)
    }

    /// Classifies a child exit. Returns `Ok(true)` when the loop should stop
    /// because the session already drained.
    ///
    /// A drained `PanicRestart` already terminated the child. That
    /// intentional kill is not reclassified as `TargetExited`, even though
    /// `TargetPanic` left `discovery_complete` false.
    fn check_target_exit(&mut self) -> Result<bool, RunEnd> {
        let Ok(Some(status)) = self.bridge.try_wait() else {
            return Ok(false);
        };
        if self.state.exit_loop {
            return Ok(true);
        }
        if is_unexpected_target_exit(
            self.state.discovery_complete,
            self.state.current_running.is_some(),
            self.state.run_queue.is_empty(),
        ) {
            return Err(RunEnd::aborted_with(
                Completion::TargetExited,
                format!("process exited unexpectedly: {status}"),
            ));
        }
        self.state.exit_loop = true;
        Ok(false)
    }

    /// Stops the link and converts the session into a [`RunRecord`].
    fn finish(mut self, end: RunEnd) -> RunRecord {
        self.bridge.terminate();
        finish_record(self.state, self.resets, self.start_time, end)
    }
}

impl RunEnd {
    /// The run drained its queue without aborting.
    fn drained() -> Self {
        Self::default()
    }

    /// The run aborted with `abort` and nothing to add to the console.
    const fn aborted(abort: Completion) -> Self {
        Self {
            abort: Some(abort),
            note: None,
        }
    }

    /// The run aborted with `abort`, appending `note` to the console.
    const fn aborted_with(abort: Completion, note: String) -> Self {
        Self {
            abort: Some(abort),
            note: Some(note),
        }
    }

    /// A command could not be delivered to the target.
    fn send_failed(e: &HostError) -> Self {
        Self::aborted_with(Completion::SendFailed, format!("send failed: {e}"))
    }
}

/// Chooses the post-`PanicRestart` step.
///
/// `exit_loop` wins over the reset budget: a draining final panic must not be
/// reported as [`Completion::ResetBudgetExhausted`].
const fn after_panic_restart(
    exit_loop: bool,
    resets: u32,
    max_resets: u32,
) -> AfterPanicRestart {
    if exit_loop {
        AfterPanicRestart::Drained
    } else if resets >= max_resets {
        AfterPanicRestart::ResetBudgetExhausted
    } else {
        AfterPanicRestart::Reconnect
    }
}

/// Returns true when a live child exit should abort as [`Completion::TargetExited`].
///
/// Callers must skip this check when `exit_loop` is already set (drained panic
/// recovery kills the child on purpose while `discovery_complete` is false).
const fn is_unexpected_target_exit(
    discovery_complete: bool,
    has_current_running: bool,
    run_queue_empty: bool,
) -> bool {
    !discovery_complete || has_current_running || !run_queue_empty
}

/// Headless execution loop for driving ETS tests to completion on a target with default options.
///
/// Establishes the transport bridge (`cargo run` builds subprocess firmware),
/// runs test discovery, and executes the suite run-queue under a wall-clock timeout.
///
/// # Errors
///
/// Returns `HostError` if the target cannot be spawned or unexpectedly disconnects
/// before any session record can be produced.
pub fn run_headless_ets(
    target: &Target,
    timeout: Duration,
) -> Result<RunRecord, HostError> {
    run_headless_ets_with_options(
        target,
        RunOptions {
            timeout,
            max_resets: 3,
        },
    )
}

/// Headless execution loop for driving ETS tests to completion on a target with explicit options.
///
/// # Errors
///
/// Returns `HostError` if the target cannot be spawned or unexpectedly disconnects
/// before any session record can be produced.
pub fn run_headless_ets_with_options(
    target: &Target,
    options: RunOptions,
) -> Result<RunRecord, HostError> {
    let mut run = HeadlessRun::start(target, options)?;
    let end = match run.drive() {
        Ok(()) => RunEnd::drained(),
        Err(end) => end,
    };
    // FR-8: a wire-contract mismatch means no result from this session can
    // be trusted, so it is an error rather than a `RunRecord`.
    if let Some(target_version) = run.state.protocol_mismatch {
        run.bridge.terminate();
        return Err(HostError::ProtocolMismatch {
            host: u32::from(PROTOCOL_VERSION),
            target: u32::from(target_version),
        });
    }
    Ok(run.finish(end))
}

/// Sends cooperative reset then suite discovery.
///
/// `handle_failure` on the target waits only for [`CommCommand::TryReset`].
/// After a panic reconnect, [`CommCommand::ListSuites`] alone leaves a serial
/// target spinning forever if the earlier `TryReset` was lost when the link
/// closed. `TryReset` is a no-op in the normal server command loop, so pairing
/// it with every discovery attempt is safe.
fn send_discovery(bridge: &mut ETSBridge) -> Result<(), HostError> {
    bridge.send_command(&CommCommand::TryReset)?;
    bridge.send_command(&CommCommand::ListSuites)
}

/// Builds the [`RunRecord`] for a finished session.
fn finish_record(
    mut state: SessionState,
    resets: u32,
    start_time: Instant,
    end: RunEnd,
) -> RunRecord {
    if let Some(msg) = end.note {
        state.logs.push_str(&msg);
        if !msg.ends_with('\n') {
            state.logs.push('\n');
        }
    }
    let pending = state.pending_cases();
    RunRecord {
        results: state.results,
        pending,
        resets,
        abort: end.abort,
        elapsed: start_time.elapsed(),
        console: state.logs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_rs_ets::comms::Telemetry;
    use control_rs_ets::settings::SettingValue;

    use crate::bridge::BridgeMessage;

    /// A session that already received a matching `TargetInfo` (FR-8).
    fn matched() -> SessionState {
        let mut s = SessionState::new();
        s.target_info = Some(crate::session::TargetInfo {
            protocol_version: control_rs_ets::comms::PROTOCOL_VERSION,
            board_id: 0,
            core_clock_hz: 0,
            fpu_flags: 0,
        });
        s
    }

    fn discover_two(state: &mut SessionState) {
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 2,
                setting_count: 1,
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "t0",
                description: "",
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "t1",
                description: "",
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                name: "gain",
                description: "",
                value: SettingValue::U8(0),
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::DiscoveryComplete,
        ));
    }

    #[test]
    fn finish_record_includes_in_flight_case_in_pending() {
        let mut state = matched();
        discover_two(&mut state);
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1)]);

        let record = finish_record(
            state,
            0,
            Instant::now(),
            RunEnd::aborted(Completion::TimedOut),
        );
        assert_eq!(record.pending, vec![(0, 0), (0, 1)]);
        assert_eq!(record.results, []);
        assert_eq!(record.abort, Some(Completion::TimedOut));
    }

    #[test]
    fn finish_record_omits_in_flight_when_already_recorded() {
        let mut state = matched();
        discover_two(&mut state);
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Failed,
            },
        ));
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.results.len(), 1);

        let record = finish_record(
            state,
            0,
            Instant::now(),
            RunEnd::aborted(Completion::TimedOut),
        );
        assert_eq!(record.pending, vec![(0, 1)]);
        assert_eq!(record.results.len(), 1);
    }

    #[test]
    fn target_exit_mid_run_retains_results_as_abort() {
        let mut state = matched();
        discover_two(&mut state);
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Passed,
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 10,
                time_us: 5,
                stack_peak: 64,
            },
        ));
        assert_eq!(state.current_running, Some((0, 1)));
        assert_eq!(state.results.len(), 1);
        assert_eq!(state.run_queue, []);

        let record = finish_record(
            state,
            0,
            Instant::now(),
            RunEnd::aborted_with(
                Completion::TargetExited,
                "process exited unexpectedly: exit status: 1".to_string(),
            ),
        );
        assert_eq!(record.results.len(), 1);
        assert_eq!(
            record.results.first().map(|r| r.test_name.as_str()),
            Some("t0")
        );
        assert_eq!(
            record.results.first().map(|r| r.state),
            Some(TestState::Passed)
        );
        assert_eq!(record.pending, vec![(0, 1)]);
        assert_eq!(record.abort, Some(Completion::TargetExited));
        assert!(record.console.contains("process exited unexpectedly"));
    }

    #[test]
    fn draining_final_panic_prefers_drained_over_reset_budget() {
        // Suite finished on the Nth panic: exit_loop set, resets at the cap.
        assert_eq!(after_panic_restart(true, 3, 3), AfterPanicRestart::Drained);
        assert_eq!(after_panic_restart(true, 4, 3), AfterPanicRestart::Drained);
    }

    #[test]
    fn mid_suite_panic_at_reset_cap_exhausts_budget() {
        assert_eq!(
            after_panic_restart(false, 3, 3),
            AfterPanicRestart::ResetBudgetExhausted
        );
        assert_eq!(
            after_panic_restart(false, 2, 3),
            AfterPanicRestart::Reconnect
        );
    }

    #[test]
    fn drained_panic_kill_is_not_unexpected_target_exit() {
        // TargetPanic clears discovery_complete before PanicRestart kills the
        // child. With exit_loop already true the runner must not use this
        // predicate (guarded by exit_loop in run_headless_ets).
        assert!(is_unexpected_target_exit(false, false, true));
        assert!(!is_unexpected_target_exit(true, false, true));
        assert!(is_unexpected_target_exit(true, true, true));
        assert!(is_unexpected_target_exit(true, false, false));
    }

    #[test]
    fn last_case_panic_sets_exit_loop_with_discovery_incomplete() {
        let mut state = matched();
        discover_two(&mut state);
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 1,
                time_us: 1,
                stack_peak: 8,
            },
        ));
        assert_eq!(state.current_running, Some((0, 1)));
        let _ = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 1,
                state: TestState::Failed,
            },
        ));
        let actions = state.handle_message(BridgeMessage::telemetry(
            &Telemetry::TargetPanic {
                message: "boom",
                file: "t.rs",
                line: 1,
            },
        ));
        assert!(state.exit_loop);
        assert!(!state.discovery_complete);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, SessionAction::PanicRestart))
        );
        // Without the exit_loop guard, try_wait would see this as TargetExited.
        assert!(is_unexpected_target_exit(
            state.discovery_complete,
            state.current_running.is_some(),
            state.run_queue.is_empty(),
        ));
        assert_eq!(
            after_panic_restart(state.exit_loop, 1, 3),
            AfterPanicRestart::Drained
        );
    }
}
