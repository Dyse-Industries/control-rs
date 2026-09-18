//! Headless execution driver for running ETS tests to completion on a target.

use std::thread;
use std::time::{Duration, Instant};

use control_rs_ets::comms::{Command as CommCommand, TestState};

use crate::bridge::ServerBridge;
use crate::error::HostError;
use crate::session::{SessionAction, SessionState};
use crate::target::{Target, build_target_elf};

/// Terminal outcome condition of an ETS run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completion {
    /// All queued test cases completed execution.
    Drained,
    /// Wall-clock timeout expired before all tests completed.
    TimedOut,
    /// Maximum allowed reset cycles were exhausted.
    ResetBudgetExhausted,
}

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

/// Structured summary of an ETS execution session.
#[derive(Debug, Clone)]
pub struct EtsRunResult {
    /// How the session terminated.
    pub completion: Completion,
    /// Vector of test outcomes collected during the session.
    pub results: Vec<TestOutcome>,
    /// Captured console output from the target.
    pub console: String,
    /// Number of target resets performed during execution.
    pub resets: u32,
}

/// Headless execution loop for driving ETS tests to completion on a target.
///
/// # Errors
///
/// Returns `HostError` if the target cannot be built, spawned, or connected to.
pub fn run_headless_ets(
    target: Target,
    timeout: Duration,
) -> Result<EtsRunResult, HostError> {
    let elf_path = match &target {
        Target::Subprocess(sub) => build_target_elf(sub)?,
        Target::Serial { .. } => String::new(),
    };

    let elf_opt = if elf_path.is_empty() {
        None
    } else {
        Some(elf_path.as_str())
    };

    let mut bridge = ServerBridge::new(target.clone(), elf_opt, true)?;

    let start_time = Instant::now();
    let mut last_send = Instant::now();
    let mut state = SessionState::new();
    let mut resets = 0u32;
    const MAX_RESETS: u32 = 3;

    let _ = bridge.send_command(&CommCommand::ListSuites);

    while !state.exit_loop {
        if start_time.elapsed() > timeout {
            bridge.kill();
            return Ok(EtsRunResult {
                completion: Completion::TimedOut,
                results: state.results,
                console: state.logs,
                resets,
            });
        }

        if !state.discovery_complete
            && last_send.elapsed() > Duration::from_millis(500)
        {
            let _ = bridge.send_command(&CommCommand::ListSuites);
            last_send = Instant::now();
        }

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

                        if resets >= MAX_RESETS {
                            return Ok(EtsRunResult {
                                completion: Completion::ResetBudgetExhausted,
                                results: state.results,
                                console: state.logs,
                                resets,
                            });
                        }

                        if !state.exit_loop {
                            bridge = ServerBridge::new(
                                target.clone(),
                                elf_opt,
                                true,
                            )?;
                            let _ =
                                bridge.send_command(&CommCommand::ListSuites);
                            last_send = Instant::now();
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

    bridge.kill();

    Ok(EtsRunResult {
        completion: Completion::Drained,
        results: state.results,
        console: state.logs,
        resets,
    })
}
