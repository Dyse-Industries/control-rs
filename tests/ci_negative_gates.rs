//! Integration tests verifying that builtin CI Cargo quality gates fail on invalid code.

use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use control_rs_ci::gate::{Gate, GateContext, Verdict};

type TestResult = Result<(), Box<dyn Error>>;

#[derive(Debug, Clone, Copy)]
enum NegativeScenario {
    Format,
    Clippy,
    Check,
    Build,
    Test,
}

struct TempContext {
    ctx: GateContext,
    out_dir: PathBuf,
}

impl Drop for TempContext {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.out_dir);
    }
}

fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            if entry.file_name() != "target" {
                copy_dir_all(&entry.path(), &dest_path)?;
            }
        } else {
            fs::copy(entry.path(), &dest_path)?;
        }
    }
    Ok(())
}

fn create_temp_context(scenario: NegativeScenario) -> io::Result<TempContext> {
    let base_fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ci_negatives");

    let out_dir = std::env::temp_dir().join(format!(
        "control_rs_ci_neg_test_{:?}_{}",
        scenario,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir)?;

    let fixture_workspace = out_dir.join("workspace");
    copy_dir_all(&base_fixture, &fixture_workspace)?;

    match scenario {
        NegativeScenario::Format => {
            fs::write(
                fixture_workspace.join("src/lib.rs"),
                "pub fn unformatted( x :i32,y:i32 )-> i32{\nx+y\n}\n",
            )?;
        }
        NegativeScenario::Clippy => {
            fs::write(
                fixture_workspace.join("src/lib.rs"),
                "pub fn trigger() -> i32 {\n    Some(10).unwrap()\n}\n",
            )?;
        }
        NegativeScenario::Check => {
            fs::write(
                fixture_workspace.join("src/lib.rs"),
                "pub fn type_err() -> u32 {\n    \"invalid\"\n}\n",
            )?;
        }
        NegativeScenario::Build => {
            fs::write(
                fixture_workspace.join("src/lib.rs"),
                "pub fn syntax_err() {\n    let x = ;\n}\n",
            )?;
        }
        NegativeScenario::Test => {
            let tests_dir = fixture_workspace.join("tests");
            fs::create_dir_all(&tests_dir)?;
            fs::write(
                tests_dir.join("failing_test.rs"),
                "#[test]\nfn deliberate_assertion_failure() {\n    assert_eq!(1, 2, \"deliberate failure\");\n}\n",
            )?;
        }
    }

    let logs_dir = out_dir.join("logs");
    fs::create_dir_all(&logs_dir)?;

    let ctx = GateContext {
        workspace_root: fixture_workspace,
        out_dir: logs_dir,
        default_timeout: Duration::from_secs(30),
    };

    Ok(TempContext { ctx, out_dir })
}

#[test]
fn test_negative_fmt_gate_fails_on_unformatted_code() -> TestResult {
    let temp = create_temp_context(NegativeScenario::Format)?;
    let gate = Gate::new(
        "fmt",
        "cargo fmt",
        vec!["--all".to_string(), "--".to_string(), "--check".to_string()],
    )
    .with_description("Verifies codebase formatting");

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.ctx.out_dir.join("fmt.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(log.contains("Diff in") || outcome.exit_code.is_some());

    Ok(())
}

#[test]
fn test_negative_clippy_gate_fails_on_denied_lint() -> TestResult {
    let temp = create_temp_context(NegativeScenario::Clippy)?;
    let gate = Gate::new(
        "clippy",
        "cargo clippy",
        vec![
            "--workspace".to_string(),
            "--all-targets".to_string(),
            "--".to_string(),
            "-D".to_string(),
            "warnings".to_string(),
        ],
    )
    .with_description("Executes Clippy linter");

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.ctx.out_dir.join("clippy.log");
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
    let temp = create_temp_context(NegativeScenario::Check)?;
    let gate = Gate::new(
        "check",
        "cargo check",
        vec!["--workspace".to_string(), "--all-targets".to_string()],
    )
    .with_description("Performs compiler type checking");

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.ctx.out_dir.join("check.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(log.contains("mismatched types") || log.contains("E0308"));

    Ok(())
}

#[test]
fn test_negative_build_gate_fails_on_syntax_error() -> TestResult {
    let temp = create_temp_context(NegativeScenario::Build)?;
    let gate = Gate::new(
        "build",
        "cargo build",
        vec!["--workspace".to_string(), "--all-targets".to_string()],
    )
    .with_description("Compiles workspace targets");

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.ctx.out_dir.join("build.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(log.contains("error: expected") || outcome.exit_code == Some(101));

    Ok(())
}

#[test]
fn test_negative_test_gate_fails_on_assertion_failure() -> TestResult {
    let temp = create_temp_context(NegativeScenario::Test)?;
    let gate = Gate::new("test", "cargo test", vec!["--workspace".to_string()])
        .with_description("Executes test suites");

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.ctx.out_dir.join("test.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(
        log.contains("deliberate_assertion_failure ... FAILED")
            || log.contains("assertion `left == right` failed")
    );

    Ok(())
}
