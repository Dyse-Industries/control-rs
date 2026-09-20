//! Integration tests verifying that builtin CI Cargo quality gates fail on invalid code.

use std::error::Error;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::Duration;

use control_rs_ci::gates::cargo::CargoArgvGate;
use control_rs_ci::quality_gate::{GateContext, QualityGate, Verdict};

type TestResult = Result<(), Box<dyn Error>>;

struct TempContext {
    ctx: GateContext,
    out_dir: PathBuf,
}

impl Drop for TempContext {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.out_dir);
    }
}

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn create_temp_context(fixture_name: &str) -> io::Result<TempContext> {
    let out_dir = std::env::temp_dir().join(format!(
        "control_rs_ci_neg_test_{}_{}",
        fixture_name,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir)?;

    let ctx = GateContext {
        workspace_root: fixture_dir(fixture_name),
        out_dir: out_dir.clone(),
        default_timeout: Duration::from_secs(30),
    };

    Ok(TempContext { ctx, out_dir })
}

#[test]
fn test_negative_fmt_gate_fails_on_unformatted_code() -> TestResult {
    let temp = create_temp_context("fail_fmt")?;
    let gate = CargoArgvGate::fmt();

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.out_dir.join("fmt.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(log.contains("Diff in") || outcome.exit_code.is_some());

    Ok(())
}

#[test]
fn test_negative_clippy_gate_fails_on_denied_lint() -> TestResult {
    let temp = create_temp_context("fail_clippy")?;
    let gate = CargoArgvGate::clippy();

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.out_dir.join("clippy.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(
        log.contains("error: used `unwrap()` on an `Option` value")
            || log.contains("clippy")
    );

    Ok(())
}

#[test]
fn test_negative_check_gate_fails_on_type_mismatch() -> TestResult {
    let temp = create_temp_context("fail_check")?;
    let gate = CargoArgvGate::check();

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.out_dir.join("check.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(log.contains("mismatched types") || log.contains("E0308"));

    Ok(())
}

#[test]
fn test_negative_build_gate_fails_on_syntax_error() -> TestResult {
    let temp = create_temp_context("fail_build")?;
    let gate = CargoArgvGate::build();

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.out_dir.join("build.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(log.contains("error: expected") || outcome.exit_code == Some(101));

    Ok(())
}

#[test]
fn test_negative_test_gate_fails_on_assertion_failure() -> TestResult {
    let temp = create_temp_context("fail_test")?;
    let gate = CargoArgvGate::test();

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.out_dir.join("test.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(
        log.contains("deliberate_assertion_failure ... FAILED")
            || log.contains("assertion `left == right` failed")
    );

    Ok(())
}
