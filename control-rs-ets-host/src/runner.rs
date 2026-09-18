//! Headless execution driver for running ETS tests to completion on a target.

use std::thread;
use std::time::{Duration, Instant};

use control_rs_ets::comms::{Command as CommCommand, TestState};

use crate::bridge::ETSBridge;
use crate::error::HostError;
use crate::session::{SessionAction, SessionState};
use crate::target::Target;

const MAX_RESETS: u32 = 3;

/// Legacy alias for [`RunRecord`].
pub type EtsRunResult = RunRecord;

/// The result and performance telemetry of an individual test case.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TestOutcome {
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
    pub pending: Vec<(u16, u16)>,
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

/// Headless execution loop for driving ETS tests to completion on a target.
///
/// Establishes the transport bridge (`cargo run` builds subprocess firmware),
/// runs test discovery, and executes the suite run-queue under a wall-clock timeout.
///
/// # Errors
///
/// Returns `HostError` if the target cannot be spawned or unexpectedly disconnects
/// before any session record can be produced.
#[allow(clippy::needless_pass_by_value)]
pub fn run_headless_ets(
    target: Target,
    timeout: Duration,
) -> Result<RunRecord, HostError> {
    let mut bridge = ETSBridge::new(target.clone(), true)?;

    let start_time = Instant::now();
    let mut last_send = Instant::now();
    let mut state = SessionState::new();
    let mut resets = 0u32;

    if let Err(e) = send_discovery(&mut bridge) {
        bridge.terminate();
        return Ok(finish_record(
            state,
            resets,
            Some(Completion::SendFailed),
            start_time,
            Some(format!("send failed: {e}")),
        ));
    }

    while !state.exit_loop {
        if start_time.elapsed() > timeout {
            bridge.terminate();
            return Ok(finish_record(
                state,
                resets,
                Some(Completion::TimedOut),
                start_time,
                None,
            ));
        }

        if !state.discovery_complete
            && last_send.elapsed() > Duration::from_millis(500)
        {
            // Retry TryReset then ListSuites. Serial targets that miss the
            // post-panic TryReset spin forever in handle_failure and ignore
            // ListSuites; rediscovery must keep offering TryReset.
            if let Err(e) = send_discovery(&mut bridge) {
                bridge.terminate();
                return Ok(finish_record(
                    state,
                    resets,
                    Some(Completion::SendFailed),
                    start_time,
                    Some(format!("send failed: {e}")),
                ));
            }
            last_send = Instant::now();
        }

        while let Ok(msg) = bridge.receiver().try_recv() {
            for action in state.handle_message(msg) {
                match action {
                    SessionAction::Send(cmd) => {
                        if let Err(e) = bridge.send_command(&cmd) {
                            bridge.terminate();
                            return Ok(finish_record(
                                state,
                                resets,
                                Some(Completion::SendFailed),
                                start_time,
                                Some(format!("send failed: {e}")),
                            ));
                        }
                    }
                    SessionAction::PanicRestart => {
                        resets = resets.saturating_add(1);
                        thread::sleep(Duration::from_millis(50));
                        bridge.terminate();
                        thread::sleep(Duration::from_secs(1));

                        if resets >= MAX_RESETS {
                            return Ok(finish_record(
                                state,
                                resets,
                                Some(Completion::ResetBudgetExhausted),
                                start_time,
                                None,
                            ));
                        }

                        if !state.exit_loop {
                            match ETSBridge::new(target.clone(), true) {
                                Ok(new_bridge) => {
                                    bridge = new_bridge;
                                    if let Err(e) = send_discovery(&mut bridge)
                                    {
                                        bridge.terminate();
                                        return Ok(finish_record(
                                            state,
                                            resets,
                                            Some(Completion::SendFailed),
                                            start_time,
                                            Some(format!("send failed: {e}")),
                                        ));
                                    }
                                    last_send = Instant::now();
                                }
                                Err(e) => {
                                    return Ok(finish_record(
                                        state,
                                        resets,
                                        Some(Completion::ReconnectFailed),
                                        start_time,
                                        Some(format!("reconnect failed: {e}")),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Ok(Some(status)) = bridge.try_wait() {
            if !state.discovery_complete
                || state.current_running.is_some()
                || !state.run_queue.is_empty()
            {
                bridge.terminate();
                return Ok(finish_record(
                    state,
                    resets,
                    Some(Completion::TargetExited),
                    start_time,
                    Some(format!("process exited unexpectedly: {status}")),
                ));
            }
            state.exit_loop = true;
        }

        thread::sleep(Duration::from_millis(10));
    }

    bridge.terminate();

    Ok(finish_record(state, resets, None, start_time, None))
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

fn finish_record(
    mut state: SessionState,
    resets: u32,
    abort: Option<Completion>,
    start_time: Instant,
    extra: Option<String>,
) -> RunRecord {
    if let Some(msg) = extra {
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
        abort,
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

    fn discover_two(state: &mut SessionState) {
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 2,
                setting_count: 0,
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
                value: SettingValue::U8(0),
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::DiscoveryComplete,
        ));
    }

    #[test]
    fn finish_record_includes_in_flight_case_in_pending() {
        let mut state = SessionState::new();
        discover_two(&mut state);
        assert_eq!(state.current_running, Some((0, 0)));
        assert_eq!(state.run_queue, vec![(0, 1)]);

        let record = finish_record(
            state,
            0,
            Some(Completion::TimedOut),
            Instant::now(),
            None,
        );
        assert_eq!(record.pending, vec![(0, 0), (0, 1)]);
        assert_eq!(record.results, []);
        assert_eq!(record.abort, Some(Completion::TimedOut));
    }

    #[test]
    fn finish_record_omits_in_flight_when_already_recorded() {
        let mut state = SessionState::new();
        discover_two(&mut state);
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TestStateChange {
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
            Some(Completion::TimedOut),
            Instant::now(),
            None,
        );
        assert_eq!(record.pending, vec![(0, 1)]);
        assert_eq!(record.results.len(), 1);
    }

    #[test]
    fn target_exit_mid_run_retains_results_as_abort() {
        let mut state = SessionState::new();
        discover_two(&mut state);
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Passed,
            },
        ));
        let _ = state.handle_message(BridgeMessage::telemetry(
            Telemetry::MetricReport {
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
            Some(Completion::TargetExited),
            Instant::now(),
            Some("process exited unexpectedly: exit status: 1".to_string()),
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
}
