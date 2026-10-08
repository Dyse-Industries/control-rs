//! Headless execution driver for running ETS tests to completion on a target.

use std::thread;
use std::time::{Duration, Instant};

use control_rs_ets::comms::{
    Command as CommCommand, PROTOCOL_VERSION, TestState,
};

use crate::bridge::ETSBridge;
use crate::error::HostError;
use crate::session::{
    LoopRunRecord, LoopStart, SessionAction, SessionState, TestIndex,
};
use crate::sim::BoxedSim;
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
    /// Time to wait for the final loop state after `StopNow` before the
    /// session sends `TryReset` and closes the link.
    pub stop_timeout: Duration,
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
    /// Loop runs the caller selected, in start order. Empty by default.
    pub loops: Vec<LoopRunRecord>,
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
    /// A loop did not acknowledge `StopNow` within the stop timeout.
    StopUnacknowledged,
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
            stop_timeout: crate::session::DEFAULT_STOP_TIMEOUT,
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
        let mut state = SessionState::new();
        state.stop_timeout = options.stop_timeout;
        state.subprocess_link = matches!(target, Target::Subprocess(_));
        Ok(Self {
            bridge: ETSBridge::new(target.clone(), true)?,
            target,
            options,
            state,
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
            let ticks = self.state.tick(Instant::now());
            self.apply_actions(ticks)?;
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
                SessionAction::SendInput {
                    suite_id,
                    test_id,
                    seq,
                    payload,
                } => {
                    self.bridge
                        .send_command(&CommCommand::LoopInput {
                            suite_id,
                            test_id,
                            seq,
                            payload: &payload,
                        })
                        .map_err(|e| RunEnd::send_failed(&e))?;
                }
                SessionAction::CloseLink => {
                    return Err(RunEnd::aborted(
                        Completion::StopUnacknowledged,
                    ));
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
            self.state.current_running.is_some() || self.state.loop_active(),
            self.state.run_queue.is_empty()
                && self.state.pending_loops.is_empty(),
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
            ..RunOptions::default()
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
    run_headless_ets_with_loops(target, options, Vec::new(), None)
}

/// Headless execution loop that also runs the loops the caller selects.
///
/// The loops start one after another once the cases drain. A loop starts with
/// `max_steps = 1` unless its [`LoopStart`] gives another bound. `sim` feeds
/// the input of every loop. No loop runs when `loops` is empty.
///
/// # Errors
///
/// Returns `HostError` if the target cannot be spawned or unexpectedly disconnects
/// before any session record can be produced.
pub fn run_headless_ets_with_loops(
    target: &Target,
    options: RunOptions,
    loops: Vec<LoopStart>,
    sim: Option<BoxedSim>,
) -> Result<RunRecord, HostError> {
    let mut run = HeadlessRun::start(target, options)?;
    let lockstep_allowed = sim.is_some();
    run.state.set_sim(sim);
    run.state.queue_loops(loops.into_iter().map(|mut start| {
        start.lockstep &= lockstep_allowed;
        start
    }));
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
        loops: state.loop_history,
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

    #[cfg(unix)]
    mod headless {
        use std::sync::mpsc::Sender;
        use std::sync::{Arc, Mutex};

        use super::*;
        use crate::bridge::FakeBridge;

        pub(super) type Written = Arc<Mutex<Vec<u8>>>;

        /// A run over a fake link with the handles that drive it.
        pub(super) type FakeRun<'t> =
            (HeadlessRun<'t>, Sender<BridgeMessage>, Written);

        pub(super) fn serial_target() -> Target {
            Target::Serial {
                port: "/dev/none".to_string(),
                baud: 1,
            }
        }

        pub(super) fn fake_run(
            target: &Target,
            timeout: Duration,
            exit_after_polls: Option<usize>,
        ) -> FakeRun<'_> {
            let FakeBridge {
                bridge,
                tx,
                written,
            } = ETSBridge::fake(exit_after_polls);
            let run = HeadlessRun {
                bridge,
                target,
                options: RunOptions {
                    timeout,
                    max_resets: 3,
                    ..RunOptions::default()
                },
                state: SessionState::new(),
                resets: 0,
                start_time: Instant::now(),
                last_send: Instant::now(),
            };
            (run, tx, written)
        }

        pub(super) fn written_len(written: &Written) -> usize {
            written.lock().map_or(usize::MAX, |w| w.len())
        }

        /// Messages that complete discovery of an empty suite.
        fn discover_empty(tx: &Sender<BridgeMessage>) {
            for tel in [
                Telemetry::TargetInfo {
                    protocol_version: PROTOCOL_VERSION,
                    board_id: 0,
                    core_clock_hz: 0,
                    fpu_flags: 0,
                },
                Telemetry::SuiteInfo {
                    suite_id: 0,
                    name: "s",
                    description: "",
                    test_count: 0,
                    setting_count: 0,
                },
                Telemetry::DiscoveryComplete,
            ] {
                tx.send(BridgeMessage::telemetry(&tel)).unwrap();
            }
        }

        #[test]
        fn drive_processes_messages_until_the_session_drains() {
            let target = serial_target();
            // Every message is queued up front, so a healthy run ends within
            // milliseconds. A short limit keeps a broken session failing
            // quickly instead of waiting out a long one.
            let (mut run, tx, written) =
                fake_run(&target, Duration::from_secs(5), None);
            discover_empty(&tx);
            assert!(run.drive().is_ok());
            assert!(run.state.discovery_complete);
            assert!(run.state.exit_loop);
            assert!(written_len(&written) > 0, "discovery was requested");
        }

        #[test]
        fn drive_times_out_when_discovery_never_completes() {
            let target = serial_target();
            let (mut run, _tx, _written) =
                fake_run(&target, Duration::from_millis(60), Some(400));
            let start = Instant::now();
            let end = run.drive().unwrap_err();
            assert_eq!(end.abort, Some(Completion::TimedOut));
            assert!(start.elapsed() < Duration::from_secs(2));
        }

        #[test]
        fn discovery_is_repeated_only_while_incomplete_and_after_half_a_second()
        {
            let target = serial_target();
            let stale = |run: &mut HeadlessRun<'_>| {
                run.last_send = Instant::now()
                    .checked_sub(Duration::from_millis(700))
                    .unwrap();
            };

            // Incomplete and stale: resend and restart the interval.
            let (mut run, _tx, written) =
                fake_run(&target, Duration::from_secs(30), None);
            stale(&mut run);
            let before = run.last_send;
            assert!(run.retry_discovery().is_ok());
            assert!(written_len(&written) > 0);
            assert!(run.last_send > before);

            // Incomplete but recent: wait.
            let (mut run, _tx, written) =
                fake_run(&target, Duration::from_secs(30), None);
            assert!(run.retry_discovery().is_ok());
            assert_eq!(written_len(&written), 0);

            // Complete: never resend, however stale.
            let (mut run, _tx, written) =
                fake_run(&target, Duration::from_secs(30), None);
            run.state.discovery_complete = true;
            stale(&mut run);
            assert!(run.retry_discovery().is_ok());
            assert_eq!(written_len(&written), 0);
        }

        #[test]
        fn a_drained_panic_restart_stops_processing_further_actions() {
            let target = serial_target();
            let (mut run, _tx, written) =
                fake_run(&target, Duration::from_secs(30), None);
            run.state.exit_loop = true;
            let actions = vec![
                SessionAction::PanicRestart,
                SessionAction::Send(CommCommand::TryReset),
            ];
            assert!(run.apply_actions(actions).is_ok());
            assert_eq!(written_len(&written), 0, "actions after the drain ran");
            assert!(run.bridge.is_shut_down());
        }

        #[test]
        fn actions_are_sent_to_the_target_in_order() {
            let target = serial_target();
            let (mut run, _tx, written) =
                fake_run(&target, Duration::from_secs(30), None);
            let actions = vec![SessionAction::Send(CommCommand::TryReset)];
            assert!(run.apply_actions(actions).is_ok());
            assert!(written_len(&written) > 0);
        }
    }

    /// Commands the target received, decoded from the bytes written to the link.
    #[cfg(unix)]
    fn written_commands(written: &headless::Written) -> Vec<String> {
        use control_rs_ets::comms::FrameReader;
        let bytes = written.lock().unwrap().clone();
        let mut reader = FrameReader::new();
        let mut out = Vec::new();
        for b in bytes {
            if let Some(payload) = reader.handle_byte(b)
                && let Ok(cmd) =
                    postcard::from_bytes::<CommCommand<'_>>(payload)
            {
                out.push(format!("{cmd:?}"));
            }
        }
        out
    }

    /// Frames that discover one suite with one loop.
    #[cfg(unix)]
    fn discover_loop(tx: &std::sync::mpsc::Sender<BridgeMessage>) {
        for tel in [
            Telemetry::TargetInfo {
                protocol_version: PROTOCOL_VERSION,
                board_id: 0,
                core_clock_hz: 0,
                fpu_flags: 0,
            },
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "s",
                description: "",
                test_count: 0,
                setting_count: 0,
            },
            Telemetry::LifecycleSuite {
                suite_id: 0,
                loop_count: 1,
            },
            Telemetry::LoopInfo {
                suite_id: 0,
                test_id: 0,
                name: "l",
                description: "",
                input_type: "()",
                output_type: "()",
            },
            Telemetry::DiscoveryComplete,
        ] {
            tx.send(BridgeMessage::telemetry(&tel)).unwrap();
        }
    }

    #[cfg(unix)]
    fn loop_state(
        tx: &std::sync::mpsc::Sender<BridgeMessage>,
        state: control_rs_ets::comms::LoopRunState,
    ) {
        tx.send(BridgeMessage::telemetry(&Telemetry::LoopState {
            suite_id: 0,
            test_id: 0,
            state,
            message: None,
        }))
        .unwrap();
    }

    /// Drives a headless run that discovers one loop and selects `start`;
    /// the target answers with `states` up front.
    #[cfg(unix)]
    fn drive_with_loop<'t>(
        target: &'t Target,
        start: Option<crate::session::LoopStart>,
        states: &[control_rs_ets::comms::LoopRunState],
    ) -> (HeadlessRun<'t>, headless::Written) {
        let (mut run, tx, written) =
            headless::fake_run(target, Duration::from_secs(5), None);
        run.state.queue_loops(start);
        discover_loop(&tx);
        for state in states {
            loop_state(&tx, *state);
        }
        assert!(run.drive().is_ok());
        (run, written)
    }

    /// Like [`drive_with_loop`], but the target answers `Aborted` once
    /// `StopNow` arrives.
    #[cfg(unix)]
    fn drive_until_stopped(
        target: &Target,
        start: crate::session::LoopStart,
    ) -> HeadlessRun<'_> {
        let (mut run, tx, written) =
            headless::fake_run(target, Duration::from_secs(5), None);
        run.state.queue_loops([start]);
        discover_loop(&tx);
        loop_state(&tx, control_rs_ets::comms::LoopRunState::Running);
        let answer = thread::spawn(move || {
            for _ in 0..400 {
                if written_commands(&written)
                    .iter()
                    .any(|c| c.contains("StopNow"))
                {
                    loop_state(
                        &tx,
                        control_rs_ets::comms::LoopRunState::Aborted,
                    );
                    return;
                }
                thread::sleep(Duration::from_millis(5));
            }
        });
        assert!(run.drive().is_ok());
        answer.join().unwrap();
        run
    }

    #[cfg(unix)]
    #[test]
    fn headless_loop_bounded() {
        use crate::session::LoopStart;
        use control_rs_ets::comms::LoopRunState;
        let target = headless::serial_target();
        let last_state = |run: &HeadlessRun<'_>| {
            run.state
                .loop_history
                .first()
                .and_then(|r| r.states.last().map(|s| s.0))
        };

        // By default no loop starts.
        let (_, written) = drive_with_loop(&target, None, &[]);
        let sent = written_commands(&written);
        assert!(!sent.iter().any(|c| c.contains("StartLoop")), "{sent:?}");

        // A selected loop runs a single step by default.
        let single = Some(LoopStart::single_step(0));
        let states = [LoopRunState::Running, LoopRunState::Bounded];
        let (run, written) = drive_with_loop(&target, single, &states);
        let sent = written_commands(&written);
        assert!(
            sent.iter().any(|c| c.contains("StartLoop")
                && c.contains("max_steps: 1")
                && c.contains("lockstep: false")),
            "{sent:?}"
        );
        assert_eq!(last_state(&run), Some(LoopRunState::Bounded));

        // A duration bound sends StopNow when it elapses.
        let timed = LoopStart {
            suite_id: 0,
            max_steps: 0,
            lockstep: false,
            duration: Some(Duration::from_millis(40)),
        };
        let run = drive_until_stopped(&target, timed);
        assert_eq!(last_state(&run), Some(LoopRunState::Aborted));
    }

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

    #[test]
    fn a_note_ends_the_console_with_exactly_one_newline() {
        let note = |text: &str| {
            finish_record(
                SessionState::new(),
                0,
                Instant::now(),
                RunEnd::aborted_with(Completion::SendFailed, text.to_string()),
            )
            .console
        };
        assert_eq!(note("boom"), "boom\n");
        assert_eq!(note("boom\n"), "boom\n");
    }
}
