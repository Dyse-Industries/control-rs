//! Integration tests for control-rs-ci.

use std::fs;
use std::path::PathBuf;

use control_rs_ci::config::{GateConfig, GatePolicy};
use control_rs_ci::gates::git::GitHygieneGate;
use control_rs_ci::gates::metrics::MetricsGate;
use control_rs_ci::quality_gate::{
    GateContext, GateOutcome, QualityGate, Verdict,
};
use control_rs_ci::report::{MAX_REPORT_BYTES, ReportAggregator};

#[test]
fn test_config_parsing_defaults() {
    let toml_str = r#"
        [runner]
        title = "test-ci"
        out_dir = "target/test-ci"
        timeout_secs = 45

        [gates]
        fmt = "fail"
        metrics = "warn"
        clean = "skip"
    "#;

    let config: GateConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.runner.title, "test-ci");
    assert_eq!(config.runner.out_dir, PathBuf::from("target/test-ci"));
    assert_eq!(config.runner.timeout_secs, 45);
    assert_eq!(config.policy_for("fmt"), GatePolicy::Fail);
    assert_eq!(config.policy_for("metrics"), GatePolicy::Warn);
    assert_eq!(config.policy_for("clean"), GatePolicy::Skip);
    assert_eq!(config.policy_for("unknown"), GatePolicy::Fail);
}

#[test]
fn test_gate_outcome_serialization_roundtrip() {
    let outcome = GateOutcome {
        gate: "valgrind".to_string(),
        verdict: Verdict::Pass,
        exit_code: Some(0),
        duration_secs: 4.25,
        summary: Some("Memcheck clean (0 leaks, 0 errors)".to_string()),
        log_file: "valgrind.log".to_string(),
        raw_artifact: Some("valgrind-raw.json".to_string()),
    };

    let serialized = serde_json::to_string_pretty(&outcome).unwrap();
    let deserialized: GateOutcome = serde_json::from_str(&serialized).unwrap();

    assert_eq!(outcome, deserialized);
}

#[test]
fn test_metrics_gate_execution() {
    let tmp_dir = std::env::temp_dir().join("control_rs_ci_metrics_test");
    let src_dir = tmp_dir.join("src");
    let out_dir = tmp_dir.join("artifacts");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&src_dir).unwrap();
    fs::create_dir_all(&out_dir).unwrap();

    let sample_code = "//! Module doc\n\nfn main() {\n    // comment\n    println!(\"hello\");\n}\n";
    fs::write(src_dir.join("main.rs"), sample_code).unwrap();

    let mut config = GateConfig::default();
    config.metrics.tracked_dirs = vec!["src".to_string()];

    let gate = MetricsGate::new(config.metrics);
    let ctx = GateContext {
        workspace_root: tmp_dir.clone(),
        out_dir: out_dir.clone(),
        default_timeout: std::time::Duration::from_secs(10),
    };

    let outcome = gate.execute(&ctx).unwrap();
    assert_eq!(outcome.verdict, Verdict::Pass);
    assert!(outcome.summary.is_some());
    assert!(out_dir.join("metrics.log").exists());
    assert!(out_dir.join("metrics-raw.json").exists());
    assert!(out_dir.join("metrics.result.json").exists());

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_git_hygiene_commit_auditing() {
    let config = GateConfig::default().git;
    let gate = GitHygieneGate::new(config);

    // Private method test through reflection or execution
    let ctx = GateContext {
        workspace_root: PathBuf::from("."),
        out_dir: std::env::temp_dir().join("control_rs_ci_git_test"),
        default_timeout: std::time::Duration::from_secs(10),
    };
    let _ = fs::create_dir_all(&ctx.out_dir);

    let outcome = gate.execute(&ctx).unwrap();
    assert_eq!(outcome.gate, "git");
    assert!(
        std::path::Path::new(&outcome.log_file)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("log"))
    );

    let _ = fs::remove_dir_all(&ctx.out_dir);
}

#[test]
fn test_report_aggregator_generation() {
    let tmp_dir = std::env::temp_dir().join("control_rs_ci_report_test");
    let artifacts_dir = tmp_dir.join("artifacts");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&artifacts_dir).unwrap();

    let outcome1 = GateOutcome {
        gate: "fmt".to_string(),
        verdict: Verdict::Pass,
        exit_code: Some(0),
        duration_secs: 0.42,
        summary: Some("Formatting clean".to_string()),
        log_file: "fmt.log".to_string(),
        raw_artifact: None,
    };
    outcome1.save_to_dir(&artifacts_dir).unwrap();
    fs::write(artifacts_dir.join("fmt.log"), "rustfmt complete\n").unwrap();

    let outcome2 = GateOutcome {
        gate: "clippy".to_string(),
        verdict: Verdict::Warn,
        exit_code: Some(0),
        duration_secs: 1.25,
        summary: Some("Clippy completed with 1 warning".to_string()),
        log_file: "clippy.log".to_string(),
        raw_artifact: None,
    };
    outcome2.save_to_dir(&artifacts_dir).unwrap();
    fs::write(
        artifacts_dir.join("clippy.log"),
        "warning: unused variable `x`\n",
    )
    .unwrap();

    let mut config = GateConfig::default();
    config.gates.insert("fmt".to_string(), GatePolicy::Fail);
    config.gates.insert("clippy".to_string(), GatePolicy::Warn);

    let aggregator = ReportAggregator::new(artifacts_dir, tmp_dir.clone());
    let (is_pass, report_path) =
        aggregator.write_report(&config, None).unwrap();

    assert!(is_pass);
    assert!(report_path.exists());

    let content = fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("# Continuous Integration & Verification Report"));
    assert!(content.contains("`fmt`"));
    assert!(content.contains("`clippy`"));
    assert!(content.contains("warning: unused variable `x`"));
    assert!(content.len() <= MAX_REPORT_BYTES);

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_report_aggregator_subset_filtering() {
    let tmp_dir = std::env::temp_dir().join("control_rs_ci_report_subset_test");
    let artifacts_dir = tmp_dir.join("artifacts");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&artifacts_dir).unwrap();

    let outcome1 = GateOutcome {
        gate: "fmt".to_string(),
        verdict: Verdict::Pass,
        exit_code: Some(0),
        duration_secs: 0.42,
        summary: Some("Formatting clean".to_string()),
        log_file: "fmt.log".to_string(),
        raw_artifact: None,
    };
    outcome1.save_to_dir(&artifacts_dir).unwrap();

    let outcome2 = GateOutcome {
        gate: "check".to_string(),
        verdict: Verdict::Pass,
        exit_code: Some(0),
        duration_secs: 5.12,
        summary: Some("Check clean".to_string()),
        log_file: "check.log".to_string(),
        raw_artifact: None,
    };
    outcome2.save_to_dir(&artifacts_dir).unwrap();

    let mut config = GateConfig::default();
    config.gates.insert("fmt".to_string(), GatePolicy::Fail);
    config.gates.insert("check".to_string(), GatePolicy::Skip);

    let aggregator = ReportAggregator::new(artifacts_dir, tmp_dir.clone());
    let (is_pass, report_path) = aggregator
        .write_report(&config, Some(&["fmt".to_string()]))
        .unwrap();

    assert!(is_pass);
    let content = fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("`fmt`"));
    assert!(!content.contains("`check`"));

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_cli_args_parsing() {
    use control_rs_ci::cli::parse_args;

    let args = vec![
        "cargo-ci".to_string(),
        "--only".to_string(),
        "fmt,clippy".to_string(),
        "--skip".to_string(),
        "mutants".to_string(),
        "--up-to".to_string(),
        "test".to_string(),
        "--config".to_string(),
        "custom.toml".to_string(),
    ];

    let options = parse_args(&args, "cargo ci");
    assert_eq!(options.only_gates, vec!["fmt", "clippy"]);
    assert_eq!(options.skip_gates, vec!["mutants"]);
    assert_eq!(options.up_to_gate, Some("test".to_string()));
    assert_eq!(options.config_path, Some(PathBuf::from("custom.toml")));

    let eq_args = vec![
        "cargo-ci".to_string(),
        "--only=fmt,clippy".to_string(),
        "--skip=mutants".to_string(),
        "--up-to=test".to_string(),
        "--config=custom.toml".to_string(),
    ];
    let eq_options = parse_args(&eq_args, "cargo ci");
    assert_eq!(eq_options.only_gates, vec!["fmt", "clippy"]);
    assert_eq!(eq_options.skip_gates, vec!["mutants"]);
    assert_eq!(eq_options.up_to_gate, Some("test".to_string()));
    assert_eq!(eq_options.config_path, Some(PathBuf::from("custom.toml")));

    let short_args = vec![
        "cargo-ci".to_string(),
        "-o".to_string(),
        "fmt,clippy".to_string(),
        "-s".to_string(),
        "mutants".to_string(),
        "-u".to_string(),
        "test".to_string(),
    ];
    let short_options = parse_args(&short_args, "cargo ci");
    assert_eq!(short_options.only_gates, vec!["fmt", "clippy"]);
    assert_eq!(short_options.skip_gates, vec!["mutants"]);
    assert_eq!(short_options.up_to_gate, Some("test".to_string()));

    let positional_args =
        vec!["gate".to_string(), "fmt".to_string(), "vale".to_string()];
    let pos_options = parse_args(&positional_args, "cargo gate");
    assert_eq!(pos_options.only_gates, vec!["fmt", "vale"]);
}

#[test]
fn test_cli_multi_token_parsing() {
    use control_rs_ci::cli::parse_args;

    let multi_space_args = vec![
        "cargo-ci".to_string(),
        "--skip".to_string(),
        "fmt".to_string(),
        "clippy".to_string(),
        "check".to_string(),
        "build".to_string(),
        "test".to_string(),
    ];
    let multi_space_options = parse_args(&multi_space_args, "cargo ci");
    assert_eq!(
        multi_space_options.skip_gates,
        vec!["fmt", "clippy", "check", "build", "test"]
    );
    assert_eq!(multi_space_options.only_gates, Vec::<String>::new());
}

#[test]
fn test_render_usage_contains_cargo_styles_and_content() {
    use control_rs_ci::cli::render_usage;

    let help = render_usage("cargo ci");
    assert!(help.contains("Usage:"));
    assert!(help.contains("cargo ci"));
    assert!(help.contains("Options:"));
    assert!(help.contains("--only"));
    assert!(help.contains("--skip"));
    assert!(help.contains("--up-to"));
    assert!(help.contains("--config"));
    assert!(help.contains("--list"));
    assert!(help.contains("--help"));
    assert!(help.contains("Examples:"));
    assert!(help.contains("fmt,clippy"));
}
