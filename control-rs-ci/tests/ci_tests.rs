//! Integration tests for control-rs-ci.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use control_rs_ci::config::{GateConfig, GateDefinition, GatePolicy};
use control_rs_ci::gate::{
    Gate, GateContext, GateOutcome, Verdict, build_all_gates,
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

        [fmt]
        command = "cargo fmt"
        args = ["--all", "--", "--check"]

        [clean]
        command = "cargo clean"
    "#;

    let config: GateConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.runner.title, "test-ci");
    assert_eq!(config.runner.out_dir, PathBuf::from("target/test-ci"));
    assert_eq!(config.runner.timeout_secs, 45);
    assert_eq!(config.policy_for("fmt"), GatePolicy::Fail);
    assert_eq!(config.policy_for("metrics"), GatePolicy::Warn);
    assert_eq!(config.policy_for("clean"), GatePolicy::Skip);
    assert_eq!(config.policy_for("unknown"), GatePolicy::Fail);

    assert_eq!(
        config.gate_def("fmt"),
        Some(&GateDefinition {
            command: "cargo fmt".to_string(),
            args: vec![
                "--all".to_string(),
                "--".to_string(),
                "--check".to_string()
            ],
            description: None,
            env: HashMap::new(),
            mode: None,
        })
    );
    assert_eq!(
        config.gate_def("clean"),
        Some(&GateDefinition {
            command: "cargo clean".to_string(),
            args: vec![],
            description: None,
            env: HashMap::new(),
            mode: None,
        })
    );
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
    };

    let serialized = serde_json::to_string_pretty(&outcome).unwrap();
    let deserialized: GateOutcome = serde_json::from_str(&serialized).unwrap();

    assert_eq!(outcome, deserialized);
}

#[test]
fn test_generic_gate_execution() {
    let tmp_dir = std::env::temp_dir().join("control_rs_ci_generic_gate_test");
    let out_dir = tmp_dir.join("artifacts");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&out_dir).unwrap();

    let gate = Gate::new(
        "test_echo",
        "echo",
        vec!["hello_world".to_string()],
        Some("Echo test gate".to_string()),
        HashMap::new(),
    );

    assert_eq!(gate.name(), "test_echo");
    assert_eq!(gate.description(), Some("Echo test gate"));
    assert_eq!(gate.command_display(), "`echo hello_world`");

    let ctx = GateContext {
        workspace_root: tmp_dir.clone(),
        out_dir: out_dir.clone(),
        default_timeout: std::time::Duration::from_secs(10),
    };

    let outcome = gate.execute(&ctx).unwrap();
    assert_eq!(outcome.verdict, Verdict::Pass);
    assert_eq!(outcome.exit_code, Some(0));
    assert!(outcome.summary.is_some());
    assert!(out_dir.join("test_echo.log").exists());
    assert!(out_dir.join("test_echo.result.json").exists());

    let log_content =
        fs::read_to_string(out_dir.join("test_echo.log")).unwrap();
    assert!(log_content.contains("hello_world"));

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_generic_gate_compound_command() {
    let tmp_dir = std::env::temp_dir().join("control_rs_ci_compound_cmd_test");
    let out_dir = tmp_dir.join("artifacts");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&out_dir).unwrap();

    // Compound command "echo foo" with additional args ["bar"]
    let gate = Gate::new(
        "compound_test",
        "echo foo",
        vec!["bar".to_string()],
        None,
        HashMap::new(),
    );

    assert_eq!(gate.command_display(), "`echo foo bar`");

    let ctx = GateContext {
        workspace_root: tmp_dir.clone(),
        out_dir: out_dir.clone(),
        default_timeout: std::time::Duration::from_secs(10),
    };

    let outcome = gate.execute(&ctx).unwrap();
    assert_eq!(outcome.verdict, Verdict::Pass);

    let log_content =
        fs::read_to_string(out_dir.join("compound_test.log")).unwrap();
    assert!(log_content.contains("foo bar"));

    let _ = fs::remove_dir_all(&tmp_dir);
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

    let aggregator =
        ReportAggregator::new(artifacts_dir.clone(), tmp_dir.clone());
    let (is_pass, report_path) =
        aggregator.write_report(&config, None).unwrap();

    assert!(is_pass);
    assert!(report_path.exists());
    assert_eq!(report_path, artifacts_dir.join("ci-report.md"));
    assert!(!tmp_dir.join("ci-report.md").exists());

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
    };
    outcome1.save_to_dir(&artifacts_dir).unwrap();

    let outcome2 = GateOutcome {
        gate: "check".to_string(),
        verdict: Verdict::Pass,
        exit_code: Some(0),
        duration_secs: 5.12,
        summary: Some("Check clean".to_string()),
        log_file: "check.log".to_string(),
    };
    outcome2.save_to_dir(&artifacts_dir).unwrap();

    let mut config = GateConfig::default();
    config.gates.insert("fmt".to_string(), GatePolicy::Fail);
    config.gates.insert("check".to_string(), GatePolicy::Skip);

    let aggregator =
        ReportAggregator::new(artifacts_dir.clone(), tmp_dir.clone());
    let (is_pass, report_path) = aggregator
        .write_report(&config, Some(&["fmt".to_string()]))
        .unwrap();

    assert!(is_pass);
    assert_eq!(report_path, artifacts_dir.join("ci-report.md"));
    assert!(!tmp_dir.join("ci-report.md").exists());
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
    assert!(help.contains("--clean"));
    assert!(help.contains("--all"));
    assert!(help.contains("--only"));
    assert!(help.contains("--skip"));
    assert!(help.contains("--up-to"));
    assert!(help.contains("--config"));
    assert!(help.contains("--list"));
    assert!(help.contains("--help"));
    assert!(help.contains("Examples:"));
    assert!(help.contains("clean"));
    assert!(help.contains("fmt,clippy"));
}

#[test]
fn test_cli_clean_args_parsing() {
    use control_rs_ci::cli::parse_args;

    // Positional clean
    let clean_args = vec!["cargo-ci".to_string(), "clean".to_string()];
    let opts = parse_args(&clean_args, "cargo ci");
    assert!(opts.clean);
    assert!(opts.only_gates.is_empty());
    assert!(!opts.run_all);

    // Flag --clean
    let flag_args = vec!["cargo-ci".to_string(), "--clean".to_string()];
    let opts = parse_args(&flag_args, "cargo ci");
    assert!(opts.clean);
    assert!(opts.only_gates.is_empty());
    assert!(!opts.run_all);

    // Short flag -X
    let short_args = vec!["cargo-ci".to_string(), "-X".to_string()];
    let opts = parse_args(&short_args, "cargo ci");
    assert!(opts.clean);

    // Clean with all
    let clean_all_args = vec![
        "cargo-ci".to_string(),
        "--clean".to_string(),
        "--all".to_string(),
    ];
    let opts = parse_args(&clean_all_args, "cargo ci");
    assert!(opts.clean);
    assert!(opts.run_all);

    // Clean with specific gates
    let clean_fmt_args = vec![
        "cargo-ci".to_string(),
        "--clean".to_string(),
        "--only".to_string(),
        "fmt".to_string(),
    ];
    let opts = parse_args(&clean_fmt_args, "cargo ci");
    assert!(opts.clean);
    assert_eq!(opts.only_gates, vec!["fmt"]);

    // Positional clean,fmt
    let comma_args = vec!["cargo-ci".to_string(), "clean,fmt".to_string()];
    let opts = parse_args(&comma_args, "cargo ci");
    assert!(opts.clean);
    assert_eq!(opts.only_gates, vec!["fmt"]);
}

#[test]
fn test_clean_artifacts() {
    use control_rs_ci::clean_artifacts;

    let tmp_dir = std::env::temp_dir().join("control_rs_clean_test");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).unwrap();

    let config_path = tmp_dir.join("gate.toml");
    let toml_content = r#"
        [runner]
        title = "test"
        out_dir = "test_artifacts"
    "#;
    fs::write(&config_path, toml_content).unwrap();

    let out_dir = tmp_dir.join("test_artifacts");
    fs::create_dir_all(&out_dir).unwrap();
    fs::write(out_dir.join("fmt.log"), "sample log").unwrap();

    // Create stray root files
    fs::write(tmp_dir.join("ci-report.md"), "stray report").unwrap();
    fs::create_dir_all(tmp_dir.join("mutants.out")).unwrap();

    assert!(out_dir.exists());
    assert!(tmp_dir.join("ci-report.md").exists());
    assert!(tmp_dir.join("mutants.out").exists());

    let cleaned_out_dir = clean_artifacts(&tmp_dir, &config_path).unwrap();
    assert_eq!(cleaned_out_dir, out_dir);
    assert!(!out_dir.exists());
    assert!(!tmp_dir.join("ci-report.md").exists());
    assert!(!tmp_dir.join("mutants.out").exists());

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_gate_definition_parsing() {
    let toml_str = r#"
        [runner]
        title = "test-ci"
        out_dir = "target/test-ci"

        [gates]
        build = "fail"
        custom = "warn"

        [build]
        command = "cargo build"
        args = ["--workspace", "--all-targets"]
        description = "Compiles all targets"

        [custom]
        command = "my-tool"
        args = ["check", "--strict"]
    "#;

    let config: GateConfig = toml::from_str(toml_str).unwrap();
    let build_def = config.gate_def("build");
    assert_eq!(
        build_def,
        Some(&GateDefinition {
            command: "cargo build".to_string(),
            args: vec!["--workspace".to_string(), "--all-targets".to_string(),],
            description: Some("Compiles all targets".to_string()),
            env: HashMap::new(),
            mode: None,
        })
    );

    let custom_def = config.gate_def("custom");
    assert_eq!(
        custom_def,
        Some(&GateDefinition {
            command: "my-tool".to_string(),
            args: vec!["check".to_string(), "--strict".to_string()],
            description: None,
            env: HashMap::new(),
            mode: None,
        })
    );
}

#[test]
fn test_decentralized_gate_mode_parsing() {
    let toml_str = r#"
        [runner]
        title = "test-ci"

        [clean]
        mode = "fail"
        command = "cargo clean"

        [check]
        mode = "skip"
        command = "cargo check"
        args = ["--workspace"]

        [lint]
        mode = "warn"
        command = "cargo clippy"

        [custom]
        command = "my-tool"
    "#;

    let mut config: GateConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.policy_for("clean"), GatePolicy::Fail);
    assert_eq!(config.policy_for("check"), GatePolicy::Skip);
    assert_eq!(config.policy_for("lint"), GatePolicy::Warn);
    assert_eq!(config.policy_for("custom"), GatePolicy::Fail);
    assert_eq!(config.policy_for("unknown"), GatePolicy::Fail);

    // Verify normalize populates config.gates
    config.normalize();
    assert_eq!(config.gates.get("clean"), Some(&GatePolicy::Fail));
    assert_eq!(config.gates.get("check"), Some(&GatePolicy::Skip));
    assert_eq!(config.gates.get("lint"), Some(&GatePolicy::Warn));
    assert_eq!(config.gates.get("custom"), Some(&GatePolicy::Fail));

    // Verify Gate::from_definition inherits mode
    let clean_def = config.gate_def("clean").unwrap();
    assert_eq!(clean_def.mode(), GatePolicy::Fail);
    let gate = Gate::from_definition("clean", clean_def);
    assert_eq!(gate.mode(), GatePolicy::Fail);
}

#[test]
fn test_gate_definition_no_fallbacks() {
    let toml_str = r#"
        [runner]
        title = "test-ci"
        out_dir = "target/test-ci"

        [gates]
        fmt = "fail"
        clippy = "fail"
    "#;

    let config: GateConfig = toml::from_str(toml_str).unwrap();
    // In accordance with no-fallback policy: if not present in TOML table, returns None
    assert_eq!(config.gate_def("fmt"), None);
    assert_eq!(config.gate_def("clippy"), None);
    assert_eq!(config.gate_def("build"), None);
    assert_eq!(config.gate_def("unknown"), None);
}

#[test]
fn test_build_all_gates_ordering_and_missing_check() {
    let toml_str = r#"
        [runner]
        title = "test-ci"
        out_dir = "target/test-ci"

        [execution]
        exclusive_gates = ["valgrind"]

        [execution.groups]
        cargo = ["fmt", "clippy"]

        [gates]
        fmt = "fail"
        clippy = "fail"
        valgrind = "warn"

        [fmt]
        command = "cargo fmt"

        [clippy]
        command = "cargo clippy"

        [valgrind]
        command = "valgrind"
    "#;

    let config: GateConfig = toml::from_str(toml_str).unwrap();
    let gates = build_all_gates(&config).unwrap();
    let names: Vec<&str> = gates.iter().map(|g| g.name()).collect();
    assert_eq!(names, vec!["fmt", "clippy", "valgrind"]);

    // If an active gate is missing definition, build_all_gates returns an error
    let missing_toml = r#"
        [runner]
        title = "test-ci"
        out_dir = "target/test-ci"

        [gates]
        missing_gate = "fail"
    "#;
    let missing_config: GateConfig = toml::from_str(missing_toml).unwrap();
    assert!(build_all_gates(&missing_config).is_err());
}
