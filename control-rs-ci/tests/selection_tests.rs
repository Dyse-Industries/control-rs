//! Selection vs policy (FR-15), argument passthrough (FR-14), working
//! directory and process-tree termination (FR-16) tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use control_rs_ci::config::GateConfig;
use control_rs_ci::gate::{Gate, GateContext, GateOutcome, Verdict};
use control_rs_ci::report::{Outcomes, ReportAggregator};
use control_rs_ci::{PipelineOptions, run_pipeline};

const CONFIG: &str = r#"
[runner]
out_dir = "artifacts"
timeout_secs = 30

[execution.groups]
main = ["pass", "nightly"]

[pass]
command = "true"

[nightly]
mode = "fail"
default = false
command = "false"

[disabled]
mode = "skip"
command = "touch"
args = ["disabled-ran"]
"#;

const PASSTHROUGH_CONFIG: &str = r#"
[runner]
out_dir = "artifacts"
timeout_secs = 30

[touch]
command = "touch"
args = ["configured"]

[other]
command = "true"
"#;

/// Fresh workspace under the system temp directory with `gate.toml` holding
/// `config`. Returns the workspace root and the config path.
fn workspace(name: &str, config: &str) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("control_rs_ci_sel_{name}"));
    let _ = fs::remove_dir_all(&root);
    let config_path = root.join("gate.toml");
    let written = fs::create_dir_all(&root)
        .and_then(|()| fs::write(&config_path, config));
    assert!(written.is_ok(), "cannot create test workspace: {written:?}");
    (root, config_path)
}

fn outcome(root: &Path, gate: &str) -> Option<GateOutcome> {
    GateOutcome::load_from_file(
        &root.join("artifacts").join(format!("{gate}.result.json")),
    )
    .ok()
}

#[test]
fn unfiltered_run_leaves_out_non_default_gates() {
    let (root, cfg) = workspace("unfiltered", CONFIG);
    let passed =
        run_pipeline(&root, &cfg, &PipelineOptions::default()).unwrap();
    assert!(passed);
    assert_eq!(outcome(&root, "pass").unwrap().verdict, Verdict::Pass);
    assert!(outcome(&root, "nightly").is_none());
    assert!(outcome(&root, "disabled").is_none());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn all_includes_non_default_gates_but_not_disabled_ones() {
    let (root, cfg) = workspace("all", CONFIG);
    let options = PipelineOptions {
        all: true,
        ..PipelineOptions::default()
    };
    let passed = run_pipeline(&root, &cfg, &options).unwrap();
    assert!(!passed);
    assert_eq!(outcome(&root, "nightly").unwrap().verdict, Verdict::Fail);
    assert!(outcome(&root, "disabled").is_none());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn selected_non_default_gate_fails_the_invocation() {
    let (root, cfg) = workspace("selected", CONFIG);
    let only = vec!["nightly".to_string()];
    let options = PipelineOptions {
        only_gates: Some(&only),
        ..PipelineOptions::default()
    };
    let passed = run_pipeline(&root, &cfg, &options).unwrap();
    assert!(!passed, "a selected fail-mode gate must be able to fail");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn selected_disabled_gate_is_recorded_skipped_without_running() {
    let (root, cfg) = workspace("disabled", CONFIG);
    let only = vec!["disabled".to_string()];
    let options = PipelineOptions {
        only_gates: Some(&only),
        ..PipelineOptions::default()
    };
    let passed = run_pipeline(&root, &cfg, &options).unwrap();
    assert!(passed);
    assert_eq!(
        outcome(&root, "disabled").unwrap().verdict,
        Verdict::Skipped
    );
    assert!(!root.join("disabled-ran").exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn passthrough_appends_after_configured_args() {
    let (root, cfg) = workspace("passthrough", PASSTHROUGH_CONFIG);
    let only = vec!["touch".to_string()];
    let extra = vec!["appended".to_string()];
    let options = PipelineOptions {
        only_gates: Some(&only),
        extra_args: &extra,
        ..PipelineOptions::default()
    };
    assert!(run_pipeline(&root, &cfg, &options).unwrap());
    assert!(root.join("configured").exists());
    assert!(root.join("appended").exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn passthrough_with_several_gates_fails_before_running() {
    let (root, cfg) = workspace("passthrough_many", PASSTHROUGH_CONFIG);
    let extra = vec!["appended".to_string()];
    let options = PipelineOptions {
        extra_args: &extra,
        ..PipelineOptions::default()
    };
    assert!(run_pipeline(&root, &cfg, &options).is_err());
    assert!(!root.join("configured").exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn missing_selected_fail_result_fails() {
    let config: GateConfig = toml::from_str(CONFIG).unwrap();
    let aggregator = ReportAggregator::new(PathBuf::new(), PathBuf::new());
    let subset = vec!["pass".to_string()];
    assert!(!aggregator.is_passing(&config, &Outcomes::new(), Some(&subset)));
}

#[test]
fn full_report_requires_non_default_fail_gates() {
    let mut config: GateConfig = toml::from_str(CONFIG).unwrap();
    config.normalize();
    let aggregator = ReportAggregator::new(PathBuf::new(), PathBuf::new());
    let mut outcomes = Outcomes::new();
    outcomes.insert(
        "pass".to_string(),
        GateOutcome {
            gate: "pass".to_string(),
            verdict: Verdict::Pass,
            exit_code: Some(0),
            duration_secs: 0.0,
            summary: None,
            log_file: "pass.log".to_string(),
        },
    );
    assert!(!aggregator.is_passing(&config, &outcomes, None));
}

#[test]
fn gate_runs_in_its_declared_cwd() {
    let (root, cfg) = workspace(
        "cwd",
        r#"
[runner]
out_dir = "artifacts"

[mark]
command = "touch"
args = ["marker"]
cwd = "sub"
"#,
    );
    fs::create_dir_all(root.join("sub")).unwrap();
    assert!(run_pipeline(&root, &cfg, &PipelineOptions::default()).unwrap());
    assert!(root.join("sub/marker").exists());
    let _ = fs::remove_dir_all(&root);
}

/// True when `pid` no longer names a live process (gone or a zombie).
#[cfg(unix)]
fn is_dead(pid: &str) -> bool {
    std::process::Command::new("ps")
        .args(["-o", "stat=", "-p", pid])
        .output()
        .is_ok_and(|out| {
            let stat = String::from_utf8_lossy(&out.stdout);
            let stat = stat.trim();
            stat.is_empty() || stat.starts_with('Z')
        })
}

#[cfg(unix)]
#[test]
fn timeout_kills_descendants() {
    let root = std::env::temp_dir().join("control_rs_ci_sel_tree");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let mut gate = Gate::new(
        "tree",
        "sh",
        vec![
            "-c".to_string(),
            "sleep 60 & echo $! > grandchild.pid; wait".to_string(),
        ],
    );
    gate.timeout = Duration::from_secs(1);
    let ctx = GateContext {
        workspace_root: root.clone(),
        out_dir: root.join("artifacts"),
        default_timeout: Duration::from_secs(1),
    };

    let outcome = gate.execute(&ctx).unwrap();
    assert_eq!(outcome.verdict, Verdict::Fail);
    assert!(outcome.summary.unwrap().contains("timed out"));

    let pid = fs::read_to_string(root.join("grandchild.pid")).unwrap();
    let pid = pid.trim();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !is_dead(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(is_dead(pid), "grandchild {pid} outlived the timed-out gate");
    let _ = fs::remove_dir_all(&root);
}
