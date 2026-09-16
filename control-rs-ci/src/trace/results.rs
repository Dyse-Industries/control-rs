//! Test execution outcome parser for requirement status verification.

use regex::Regex;
use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;

/// 4-valued execution outcome for a test case.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum TestRunOutcome {
    /// Test passed.
    Passed,
    /// Test failed or panicked.
    Failed,
    /// Test was explicitly ignored or skipped.
    Ignored,
    /// Test is pending execution.
    Pending,
}

#[derive(serde::Deserialize)]
struct EtsEntry {
    test_name: String,
    state: String,
}

type OutcomeMap<S = RandomState> = HashMap<String, TestRunOutcome, S>;

impl TestRunOutcome {
    /// Whether the outcome is a verified pass.
    #[must_use]
    pub const fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }

    /// Whether the outcome is an execution failure.
    #[must_use]
    pub const fn is_failed(&self) -> bool {
        matches!(self, Self::Failed)
    }
}

/// Extract per-test outcome map (test name -> outcome) from cargo test or tarpaulin console output.
#[must_use]
pub fn parse_cargo_test_output(
    output: &str,
) -> HashMap<String, TestRunOutcome> {
    let mut map = HashMap::new();

    // 1. Standard libtest line format:
    // "test path::to::test_function ... ok"
    // "test path::to::test_function ... FAILED"
    // "test path::to::test_function ... ignored"
    if let Ok(re) =
        Regex::new(r"(?m)^\s*test\s+([A-Za-z0-9_:]+)\s+\.\.\.\s+([A-Za-z]+)")
    {
        for cap in re.captures_iter(output) {
            let full_name = cap.get(1).map_or("", |m| m.as_str());
            let outcome_str = cap.get(2).map_or("", |m| m.as_str());
            insert_named_outcome(&mut map, full_name, outcome_str);
        }
    }

    // 2. Rustdoc doctest line format:
    // "test src/math/subprograms.rs - Gemv (line 123) ... ok"
    if let Ok(doc_re) = Regex::new(
        r"(?m)^\s*test\s+([A-Za-z0-9_./\\-]+)\s+-\s+([A-Za-z0-9_:]+)\s+\(line\s+\d+\)\s+\.\.\.\s+([A-Za-z]+)",
    ) {
        for cap in doc_re.captures_iter(output) {
            let path = cap.get(1).map_or("", |m| m.as_str());
            let symbol = cap.get(2).map_or("", |m| m.as_str());
            let outcome_str = cap.get(3).map_or("", |m| m.as_str());
            insert_doctest_outcome(&mut map, path, symbol, outcome_str);
        }
    }

    map
}

fn parse_outcome(outcome_str: &str) -> TestRunOutcome {
    match outcome_str.to_ascii_lowercase().as_str() {
        "ok" | "passed" => TestRunOutcome::Passed,
        "failed" => TestRunOutcome::Failed,
        "ignored" => TestRunOutcome::Ignored,
        _ => TestRunOutcome::Pending,
    }
}

fn insert_named_outcome<S: BuildHasher>(
    map: &mut OutcomeMap<S>,
    full_name: &str,
    outcome_str: &str,
) {
    let outcome = parse_outcome(outcome_str);
    let short_name = full_name.split("::").last().unwrap_or(full_name);
    map.insert(full_name.to_string(), outcome);
    map.insert(short_name.to_string(), outcome);
}

fn insert_doctest_outcome<S: BuildHasher>(
    map: &mut OutcomeMap<S>,
    path: &str,
    symbol: &str,
    outcome_str: &str,
) {
    let outcome = parse_outcome(outcome_str);
    map.insert(format!("{path}::{symbol}"), outcome);
    map.insert(format!("doctest:{path}::{symbol}"), outcome);
    map.insert(symbol.to_string(), outcome);
}

/// Ingest ETS results from serialized JSON (`Vec<TestOutcome>`).
#[must_use]
pub fn parse_ets_results(
    json_content: &str,
) -> HashMap<String, TestRunOutcome> {
    let mut map = HashMap::new();
    if let Ok(entries) = serde_json::from_str::<Vec<EtsEntry>>(json_content) {
        for entry in entries {
            map.insert(entry.test_name.clone(), parse_outcome(&entry.state));
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cargo_test_output() {
        let output = r"
running 3 tests
test classical_tools::tests::test_routh_stability ... ok
test classical_tools::tests::test_root_locus_sweep ... FAILED
test classical_tools::tests::test_pending_feature ... ignored

test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out
";
        let map = parse_cargo_test_output(output);
        assert_eq!(
            map.get("test_routh_stability"),
            Some(&TestRunOutcome::Passed)
        );
        assert_eq!(
            map.get("test_root_locus_sweep"),
            Some(&TestRunOutcome::Failed)
        );
        assert_eq!(
            map.get("test_pending_feature"),
            Some(&TestRunOutcome::Ignored)
        );
    }

    #[test]
    fn test_parse_doctest_output() {
        let output = r"
running 1 test
test src/math/subprograms.rs - Gemv (line 123) ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
";
        let map = parse_cargo_test_output(output);
        assert_eq!(map.get("Gemv"), Some(&TestRunOutcome::Passed));
        assert_eq!(
            map.get("doctest:src/math/subprograms.rs::Gemv"),
            Some(&TestRunOutcome::Passed)
        );
    }
}
