//! Headless execution of a target's cases.
//!
//! # Description
//!
//! [`run_headless`] opens a [`Session`] on a target, runs every case to completion under a
//! wall-clock bound and returns a structured [`RunRecord`]. It never runs a task: the quality
//! gates in `control-rs-ci` run cases only.
//!
//! # Core Concepts
//!
//! - **Bounded**: A run that exceeds `RunOptions::timeout` ends with [`Completion::TimedOut`].
//! - **Recovery**: A target panic or silence resets the target and the run continues with the
//!   next case. After `RunOptions::max_resets` resets the run ends with
//!   [`Completion::ResetBudgetExhausted`]; collected results are kept.
//!
//! # Usage
//!
//! ```no_run
//! use control_rs_ets_host::headless::{RunOptions, run_headless};
//! use control_rs_ets_host::target::Target;
//!
//! let record = run_headless(&Target::qemu_arm(), RunOptions::default())?;
//! assert!(record.pending.is_empty());
//! # Ok::<(), control_rs_ets_host::HostError>(())
//! ```
//!
//! # Limitations
//!
//! - **Physical devices**: The serial open path needs a device and is exercised by validation,
//!   not by unit tests.

use std::time::{Duration, Instant};

use crate::bridge::{ETSBridge, TestState};
use crate::catalog::Catalog;
use crate::error::HostError;
use crate::link::HostLink;
use crate::session::{CaseOutcome, CaseStatus, Session, SessionOptions};
use crate::state::TestIndex;
use crate::target::Target;
use control_rs_ets::wire::Verdict;

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

/// Runs every case of `target` to completion under `options`.
///
/// Builds nothing: a subprocess target must already be built (`build_target_elf`).
///
/// # Errors
/// Returns a [`HostError`] if the target cannot be opened, speaks another wire revision, or does
/// not finish discovery within the timeout.
pub fn run_headless(
    target: &Target,
    options: RunOptions,
) -> Result<RunRecord, HostError> {
    let bridge = ETSBridge::new(target.clone(), false)?;
    let mut session = Session::open_with(
        bridge,
        SessionOptions {
            discovery_timeout: options.timeout,
            ..SessionOptions::default()
        },
    )?;
    let record = drive(&mut session, options);
    session.close();
    Ok(record)
}

/// Runs every case of an open session and records the outcome.
pub(crate) fn drive<L: HostLink<Error = HostError>>(
    session: &mut Session<L>,
    options: RunOptions,
) -> RunRecord {
    let start = Instant::now();
    let mut pending = session.catalog().case_ids();
    let mut results = Vec::new();
    let mut abort = None;
    for id in pending.clone() {
        let Some(remaining) = options
            .timeout
            .checked_sub(start.elapsed())
            .filter(|r| !r.is_zero())
        else {
            abort = Some(Completion::TimedOut);
            break;
        };
        session.opts.case_timeout = remaining;
        let Ok(outcome) = session.run_case(id) else {
            abort = Some(if session.link.exited().unwrap_or(true) {
                Completion::TargetExited
            } else {
                Completion::SendFailed
            });
            break;
        };
        pending.retain(|p| *p != id);
        let faulted = !matches!(outcome.status, CaseStatus::Done(_));
        results.push(outcome_of(session.catalog(), &outcome));
        if faulted
            && !pending.is_empty()
            && session.results().resets >= options.max_resets
        {
            abort = Some(Completion::ResetBudgetExhausted);
            break;
        }
    }
    RunRecord {
        results,
        pending,
        resets: session.results().resets,
        abort,
        elapsed: start.elapsed(),
        console: session.console().to_string(),
    }
}

/// The record entry of a case outcome.
fn outcome_of(catalog: &Catalog, outcome: &CaseOutcome) -> TestOutcome {
    let (suite_id, test_id) = outcome.id;
    let suite = catalog.suites.get(usize::from(suite_id));
    TestOutcome {
        suite_id,
        test_id,
        suite_name: suite.map(|s| s.name.clone()).unwrap_or_default(),
        test_name: catalog
            .exec(outcome.id)
            .map(|e| e.name.clone())
            .unwrap_or_default(),
        state: if outcome.status == CaseStatus::Done(Verdict::Pass) {
            TestState::Passed
        } else {
            TestState::Failed
        },
        cycles: outcome.stats.map(|s| s.cycles),
        time_us: outcome.stats.map(|s| s.time_ns / 1000),
        stack_peak: outcome.stats.map(|s| s.stack_peak),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Fake, ScriptLink, quick};

    fn session(fake: Fake) -> Session<ScriptLink> {
        Session::open_with(fake.link(), quick()).unwrap()
    }

    fn options(ms: u64, max_resets: u32) -> RunOptions {
        RunOptions {
            timeout: Duration::from_millis(ms),
            max_resets,
        }
    }

    #[test]
    fn a_suite_drains_with_named_outcomes() {
        let mut s = session(Fake::default());
        let record = drive(&mut s, options(2000, 3));
        assert_eq!(record.completion(), Completion::Drained);
        assert_eq!(record.pending, Vec::<TestIndex>::new());
        let first = record.results.first().unwrap();
        assert_eq!(
            (first.suite_name.as_str(), first.test_name.as_str()),
            ("s", "ok")
        );
        assert_eq!(first.state, TestState::Passed);
        assert_eq!(first.cycles, Some(10));
        assert_eq!(first.time_us, Some(0));
    }

    #[test]
    fn a_panic_resets_and_the_run_continues() {
        let mut s = session(Fake {
            panic_item: Some(0),
            ..Fake::default()
        });
        let record = drive(&mut s, options(2000, 3));
        assert_eq!(record.completion(), Completion::Drained);
        assert_eq!(record.resets, 1);
        assert_eq!(record.results.len(), 2);
        assert_eq!(record.results.first().unwrap().state, TestState::Failed);
        assert_eq!(record.results.get(1).unwrap().state, TestState::Passed);
    }

    #[test]
    fn the_reset_budget_ends_the_run_and_keeps_results() {
        let mut s = session(Fake {
            panic_item: Some(0),
            ..Fake::default()
        });
        let record = drive(&mut s, options(2000, 1));
        assert_eq!(record.completion(), Completion::ResetBudgetExhausted);
        assert_eq!(record.resets, 1);
        assert_eq!(record.results.len(), 1);
        assert_eq!(record.pending, [(0, 1)]);
    }

    #[test]
    fn a_target_that_never_answers_times_out() {
        let mut s = session(Fake {
            hang_on: Some(0),
            ..Fake::default()
        });
        let started = Instant::now();
        let record = drive(&mut s, options(150, 9));
        assert_eq!(record.completion(), Completion::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_ne!(record.pending, Vec::<TestIndex>::new());
    }

    #[test]
    fn a_lost_link_is_a_send_failure_or_an_exit() {
        let mut s = session(Fake::default());
        s.link.mute = true;
        s.link.closed = true;
        let record = drive(&mut s, options(500, 3));
        assert_eq!(record.completion(), Completion::TargetExited);
        assert_eq!(record.pending.len(), 2);
    }
}
