//! Trace negative gate tests.

use std::error::Error;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::Duration;

use control_rs_ci::gate::{Gate, GateContext, Verdict};

type TestResult = Result<(), Box<dyn Error>>;

#[derive(Debug, Clone, Copy)]
enum NegativeScenario {
    Reqs,
    Marks,
    Check,
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

fn setup_scenario_reqs(fixture_workspace: &std::path::Path) -> io::Result<()> {
    let docs_dir = fixture_workspace.join("docs");
    fs::create_dir_all(&docs_dir)?;
    fs::write(
        docs_dir.join("widget-design.md"),
        "# Widget (widget)\n\n- **FR-1**: Size\n- **FR-1**: Size duplicate\n\n| VC-1 | FR-1 | `test` | OK |\n",
    )
}

fn setup_scenario_marks(fixture_workspace: &std::path::Path) -> io::Result<()> {
    let src_dir = fixture_workspace.join("src");
    fs::create_dir_all(&src_dir)?;
    fs::write(
        src_dir.join("lib.rs"),
        format!("#[{}(FR-999)]\nfn dummy() {{}}\n", "req"), // Unqualified ID
    )?;
    let docs_dir = fixture_workspace.join("docs");
    fs::create_dir_all(&docs_dir)?;
    fs::write(
        docs_dir.join("widget-design.md"),
        "# Widget (widget)\n\n- **FR-1**: Size\n\n| VC-1 | FR-1 | `test` | OK |\n",
    )
}

fn setup_scenario_check(fixture_workspace: &std::path::Path) -> io::Result<()> {
    let docs_dir = fixture_workspace.join("docs");
    fs::create_dir_all(&docs_dir)?;
    fs::write(
        docs_dir.join("widget-design.md"),
        "# Widget (widget)\n\n- **FR-1**: Size\n\n| VC-1 | FR-1 | `test` | OK |\n",
    )?;

    fs::create_dir_all(fixture_workspace.join("target/ci-artifacts"))?;
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_trace-reqs"))
        .current_dir(fixture_workspace)
        .args([
            "--config",
            ".cargo/trace/trace.toml",
            "--out",
            "target/ci-artifacts/reqs.jsonl",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "trace-reqs failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    fs::write(
        fixture_workspace.join("target/ci-artifacts/marks.jsonl"),
        "",
    )
}

fn create_temp_context(scenario: NegativeScenario) -> io::Result<TempContext> {
    let out_dir = std::env::temp_dir().join(format!(
        "control_rs_ci_trace_neg_{:?}_{}",
        scenario,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir)?;

    let fixture_workspace = out_dir.join("workspace");
    fs::create_dir_all(&fixture_workspace)?;

    fs::write(
        fixture_workspace.join("Cargo.toml"),
        "[package]\nname = \"dummy\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )?;

    let trace_dir = fixture_workspace.join(".cargo/trace");
    fs::create_dir_all(&trace_dir)?;
    fs::write(
        trace_dir.join("trace.toml"),
        format!(
            "id = 'FR-[0-9]+'\ndoc = '[a-z]+'\nfiles = [\"docs\"]\ndoc_id = '^#\\s+.*\\((?P<doc>[a-z]+)\\)'\ndefinition = '^- \\*\\*FR-'\ncondition = 'VC-[0-9]+'\nverification = '^\\| VC-'\nmethods = [\"test\", \"review\"]\nmarked_methods = [\"test\"]\nretired = []\n[markers]\nfiles = [\"src\"]\nsuffixes = [\".rs\"]\nmarker = \"#[{}(\"\n",
            "req"
        ),
    )?;

    match scenario {
        NegativeScenario::Reqs => setup_scenario_reqs(&fixture_workspace)?,
        NegativeScenario::Marks => setup_scenario_marks(&fixture_workspace)?,
        NegativeScenario::Check => setup_scenario_check(&fixture_workspace)?,
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
fn test_negative_trace_reqs_gate_fails_on_duplicate_def() -> TestResult {
    let temp = create_temp_context(NegativeScenario::Reqs)?;
    let gate = Gate::new(
        "trace-reqs",
        env!("CARGO_BIN_EXE_trace-reqs"),
        vec![
            "--config".to_string(),
            ".cargo/trace/trace.toml".to_string(),
            "--out".to_string(),
            "reqs.jsonl".to_string(),
        ],
    )
    .with_description("Checks requirement definitions");

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.ctx.out_dir.join("trace-reqs.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(
        log.contains("is defined more than once"),
        "Log didn't contain duplicate:
{log}"
    );

    Ok(())
}

#[test]
fn test_negative_trace_check_gate_fails_on_uncovered_condition() -> TestResult {
    let temp = create_temp_context(NegativeScenario::Check)?;
    let gate = Gate::new(
        "trace-check",
        env!("CARGO_BIN_EXE_trace-check"),
        vec![
            "--config".to_string(),
            ".cargo/trace/trace.toml".to_string(),
            "--reqs".to_string(),
            "target/ci-artifacts/reqs.jsonl".to_string(),
            "--marks".to_string(),
            "target/ci-artifacts/marks.jsonl".to_string(),
            "--out".to_string(),
            "target/ci-artifacts/trace-report.json".to_string(),
        ],
    )
    .with_description("Derives condition coverage");

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.ctx.out_dir.join("trace-check.log");
    assert!(log_file.exists());
    let log = fs::read_to_string(&log_file).unwrap_or_default();
    assert!(
        log.contains("widget#VC-1 has no marked test"),
        "Log didn't report the uncovered condition:
{log}"
    );

    Ok(())
}

#[test]
fn test_negative_trace_marks_gate_fails_on_dangling_marker() -> TestResult {
    let temp = create_temp_context(NegativeScenario::Marks)?;
    let gate = Gate::new(
        "trace-marks",
        env!("CARGO_BIN_EXE_trace-marks"),
        vec![
            "--config".to_string(),
            ".cargo/trace/trace.toml".to_string(),
            "--out".to_string(),
            "marks.jsonl".to_string(),
        ],
    )
    .with_description("Checks markers");

    let outcome = gate.execute(&temp.ctx)?;
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert_ne!(outcome.exit_code, Some(0));

    let log_file = temp.ctx.out_dir.join("trace-marks.log");
    assert!(log_file.exists());
    Ok(())
}
