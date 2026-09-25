//! Verdict rule and result records for headless ETS runs (`ets` binary).
//!
//! `control-rs-ets-host` returns a `RunRecord` without a verdict; whether a
//! run passes is the consumer's policy. This module is that policy for CI
//! (FR-19): a target passes only when its run drained, left nothing pending,
//! executed at least one test and every test passed.

use control_rs_ets::comms::TestState;
use control_rs_ets_host::{RunRecord, TestOutcome};
use serde::Serialize;

/// One target's entry in `ets-results.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TargetResult {
    /// Display name of the target.
    pub target: String,
    /// Whether the target satisfied the verdict rule.
    pub passed: bool,
    /// Why the target failed, when it did.
    pub reason: Option<String>,
    /// Build, spawn or transport failure that prevented a session.
    pub error: Option<String>,
    /// The session record, when a session ran.
    pub record: Option<RunRecord>,
}

impl TargetResult {
    /// Judges a finished session.
    #[must_use]
    pub fn from_record(target: impl Into<String>, record: RunRecord) -> Self {
        let reason = failure_reason(&record);
        Self {
            target: target.into(),
            passed: reason.is_none(),
            reason,
            error: None,
            record: Some(record),
        }
    }

    /// Records a target whose session could not start.
    #[must_use]
    pub fn from_error(
        target: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            target: target.into(),
            passed: false,
            reason: Some("target did not run".to_string()),
            error: Some(error.into()),
            record: None,
        }
    }

    /// Number of tests that passed in this target's session.
    #[must_use]
    pub fn passed_tests(&self) -> usize {
        self.record.as_ref().map_or(0, |r| {
            r.results
                .iter()
                .filter(|t| t.state == TestState::Passed)
                .count()
        })
    }
}

/// Result line for one case: `suite::test` and the measurements the target
/// reported (`time_us`, cycles, peak stack).
#[must_use]
pub fn case_line(outcome: &TestOutcome) -> String {
    let name = format!("{}::{}", outcome.suite_name, outcome.test_name);
    let metrics: Vec<String> = [
        outcome.time_us.map(|t| format!("{t} us")),
        outcome.cycles.map(|c| format!("{c} cycles")),
        outcome.stack_peak.map(|b| format!("{b} B stack")),
    ]
    .into_iter()
    .flatten()
    .collect();
    if metrics.is_empty() {
        name
    } else {
        format!("{name} ({})", metrics.join(", "))
    }
}

/// Why `record` fails the verdict rule, or `None` when it passes.
#[must_use]
pub fn failure_reason(record: &RunRecord) -> Option<String> {
    if let Some(abort) = record.abort {
        return Some(format!("run aborted: {abort:?}"));
    }
    if !record.pending.is_empty() {
        return Some(format!("{} test(s) left pending", record.pending.len()));
    }
    if record.results.is_empty() {
        return Some("no tests executed".to_string());
    }
    let failed: Vec<String> = record
        .results
        .iter()
        .filter(|t| t.state != TestState::Passed)
        .map(|t| format!("{}::{} ({:?})", t.suite_name, t.test_name, t.state))
        .collect();
    (!failed.is_empty()).then(|| {
        format!("{} test(s) failed: {}", failed.len(), failed.join(", "))
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use control_rs_ets_host::Completion;

    use super::*;

    fn case(name: &str, state: TestState) -> TestOutcome {
        TestOutcome {
            suite_id: 0,
            test_id: 0,
            suite_name: "math".to_string(),
            test_name: name.to_string(),
            state,
            cycles: Some(100),
            time_us: Some(10),
            stack_peak: Some(64),
        }
    }

    fn record(results: Vec<TestOutcome>) -> RunRecord {
        RunRecord {
            results,
            pending: Vec::new(),
            resets: 0,
            abort: None,
            elapsed: Duration::from_secs(1),
            console: String::new(),
        }
    }

    #[test]
    fn drained_run_with_passing_tests_passes() {
        let r = TargetResult::from_record(
            "arm",
            record(vec![case("a", TestState::Passed)]),
        );
        assert!(r.passed);
        assert_eq!(r.reason, None);
        assert_eq!(r.passed_tests(), 1);
    }

    #[test]
    fn aborted_run_fails() {
        let mut rec = record(vec![case("a", TestState::Passed)]);
        rec.abort = Some(Completion::TimedOut);
        let reason = failure_reason(&rec).unwrap();
        assert!(reason.contains("TimedOut"));
    }

    #[test]
    fn pending_tests_fail() {
        let mut rec = record(vec![case("a", TestState::Passed)]);
        rec.pending = vec![(0, 1)];
        assert!(failure_reason(&rec).unwrap().contains("pending"));
    }

    #[test]
    fn empty_run_fails() {
        assert_eq!(
            failure_reason(&record(Vec::new())).as_deref(),
            Some("no tests executed")
        );
    }

    #[test]
    fn failed_case_fails_and_is_named() {
        let rec = record(vec![
            case("a", TestState::Passed),
            case("b", TestState::Failed),
        ]);
        let reason = failure_reason(&rec).unwrap();
        assert!(reason.contains("math::b"));
    }

    #[test]
    fn error_result_fails_without_record() {
        let r = TargetResult::from_error("riscv32", "qemu not found");
        assert!(!r.passed);
        assert_eq!(r.passed_tests(), 0);
        assert_eq!(r.error.as_deref(), Some("qemu not found"));
    }

    #[test]
    fn case_line_names_case_and_metrics() {
        assert_eq!(
            case_line(&case("a", TestState::Passed)),
            "math::a (10 us, 100 cycles, 64 B stack)"
        );
        let mut bare = case("b", TestState::Failed);
        bare.cycles = None;
        bare.time_us = None;
        bare.stack_peak = None;
        assert_eq!(case_line(&bare), "math::b");
    }
}
