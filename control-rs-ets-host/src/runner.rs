//! Headless execution driver for running ETS tests to completion on a target.

use std::thread;
use std::time::{Duration, Instant};

use control_rs_ets::comms::{Command as CommCommand, TestState};

use crate::bridge::ETSBridge;
use crate::error::HostError;
use crate::session::{SessionAction, SessionState};
use crate::target::{Target, build_target_elf};

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
/// Builds the target binary (if subprocess), establishes transport bridge,
/// runs test discovery, and executes the suite run-queue under a wall-clock timeout.
///
/// # Errors
///
/// Returns `HostError` if the target cannot be built, spawned, or unexpectedly disconnects.
#[allow(clippy::needless_pass_by_value)]
pub fn run_headless_ets(
    target: Target,
    timeout: Duration,
) -> Result<RunRecord, HostError> {
    // 1. Resolve and build target ELF if running a subprocess (QEMU).
    let elf_path = match &target {
        Target::Subprocess(sub) => build_target_elf(sub)?,
        Target::Serial { .. } => String::new(),
    };

    let elf_opt = if elf_path.is_empty() {
        None
    } else {
        Some(elf_path.as_str())
    };

    // 2. Initialize the transport bridge (subprocess pipes or serial CDC link).
    let mut bridge = ETSBridge::new(target.clone(), elf_opt, true)?;

    let start_time = Instant::now();
    let mut last_send = Instant::now();
    let mut state = SessionState::new();
    let mut resets = 0u32;

    // 3. Initiate discovery by requesting target test suites.
    let _ = bridge.send_command(&CommCommand::ListSuites);

    // 4. Main session polling loop.
    while !state.exit_loop {
        // Enforce total session wall-clock timeout bound.
        if start_time.elapsed() > timeout {
            bridge.kill();
            return Ok(RunRecord {
                results: state.results,
                pending: state.run_queue,
                resets,
                abort: Some(Completion::TimedOut),
                elapsed: start_time.elapsed(),
                console: state.logs,
            });
        }

        // Retry ListSuites periodically (every 500 ms) until target completes discovery.
        if !state.discovery_complete
            && last_send.elapsed() > Duration::from_millis(500)
        {
            let _ = bridge.send_command(&CommCommand::ListSuites);
            last_send = Instant::now();
        }

        // Process incoming bridge messages from target reader thread.
        while let Ok(msg) = bridge.receiver().try_recv() {
            for action in state.handle_message(msg) {
                match action {
                    SessionAction::Send(cmd) => {
                        let _ = bridge.send_command(&cmd);
                    }
                    SessionAction::PanicRestart => {
                        resets = resets.saturating_add(1);
                        thread::sleep(Duration::from_millis(50));
                        bridge.kill();
                        thread::sleep(Duration::from_secs(1));

                        // If reset budget is exhausted, terminate session gracefully.
                        if resets >= MAX_RESETS {
                            return Ok(RunRecord {
                                results: state.results,
                                pending: state.run_queue,
                                resets,
                                abort: Some(Completion::ResetBudgetExhausted),
                                elapsed: start_time.elapsed(),
                                console: state.logs,
                            });
                        }

                        // Reconnect bridge and restart discovery while retaining existing results.
                        if !state.exit_loop {
                            bridge =
                                ETSBridge::new(target.clone(), elf_opt, true)?;
                            let _ =
                                bridge.send_command(&CommCommand::ListSuites);
                            last_send = Instant::now();
                        }
                    }
                }
            }
        }

        // Check if the target subprocess terminated prematurely.
        if let Ok(Some(status)) = bridge.try_wait() {
            if !state.discovery_complete
                || state.current_running.is_some()
                || !state.run_queue.is_empty()
            {
                bridge.kill();
                return Err(HostError::Transport {
                    source: format!("Process exited unexpectedly: {status}")
                        .into(),
                });
            }
            state.exit_loop = true;
        }

        thread::sleep(Duration::from_millis(10));
    }

    // 5. Clean teardown and return drained run record.
    bridge.kill();

    Ok(RunRecord {
        results: state.results,
        pending: state.run_queue,
        resets,
        abort: None,
        elapsed: start_time.elapsed(),
        console: state.logs,
    })
}
