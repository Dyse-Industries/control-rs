//! Integration tests for control-rs-ci.

#[cfg(test)]
mod runner {

    use std::collections::HashMap;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::PathBuf;

    use control_rs_ci::config::{GateConfig, GateDefinition, GatePolicy};
    use control_rs_ci::gate::{
        Gate, GateContext, GateOutcome, Verdict, build_all_gates,
    };
    use control_rs_ci::report::{
        MAX_REPORT_BYTES, ReportAggregator, WrittenReport,
    };
    use std::path::Path;

    /// A fresh directory under the system temp dir, unique to this process so
    /// concurrent test runs (for example cargo-mutants jobs) never share it.
    fn temp_workspace(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("control_rs_ci_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn parse_config(text: &str) -> GateConfig {
        GateConfig::parse(text, Path::new("gate.toml")).unwrap()
    }

    fn outcome(gate: &str, verdict: Verdict, summary: &str) -> GateOutcome {
        GateOutcome {
            schema: control_rs_ci::gate::OUTCOME_SCHEMA,
            gate: gate.to_string(),
            verdict,
            exit_code: Some(0),
            duration_secs: 0.5,
            summary: Some(summary.to_string()),
            log_file: format!("{gate}.log"),
        }
    }

    #[test]
    fn test_config_parsing_defaults() {
        let toml_str = r#"
        [runner]
        title = "test-ci"
        out_dir = "target/test-ci"
        timeout_secs = 45

        [fmt]
        command = "cargo fmt"
        args = ["--all", "--", "--check"]

        [metrics]
        mode = "warn"
        command = "tokei"

        [clean]
        mode = "skip"
        command = "cargo clean"
    "#;

        let config = parse_config(toml_str);
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
                default: true,
                cwd: None,
                timeout_secs: None,
                skip_exit_codes: Vec::new(),
            })
        );
        assert_eq!(
            config.gate_def("clean"),
            Some(&GateDefinition {
                command: "cargo clean".to_string(),
                args: vec![],
                description: None,
                env: HashMap::new(),
                mode: Some(GatePolicy::Skip),
                default: true,
                cwd: None,
                timeout_secs: None,
                skip_exit_codes: Vec::new(),
            })
        );
    }

    #[test]
    fn test_gate_outcome_serialization_roundtrip() {
        let outcome = GateOutcome {
            schema: control_rs_ci::gate::OUTCOME_SCHEMA,
            gate: "valgrind".to_string(),
            verdict: Verdict::Pass,
            exit_code: Some(0),
            duration_secs: 4.25,
            summary: Some("Memcheck clean (0 leaks, 0 errors)".to_string()),
            log_file: "valgrind.log".to_string(),
        };

        let serialized = serde_json::to_string_pretty(&outcome).unwrap();
        let deserialized: GateOutcome =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(outcome, deserialized);
    }

    #[test]
    fn test_generic_gate_execution() {
        let tmp_dir = temp_workspace("generic_gate_test");
        let out_dir = tmp_dir.join("artifacts");
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&out_dir).unwrap();

        let gate =
            Gate::new("test_echo", "echo", vec!["hello_world".to_string()])
                .with_description("Echo test gate");

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
        let tmp_dir = temp_workspace("compound_cmd_test");
        let out_dir = tmp_dir.join("artifacts");
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&out_dir).unwrap();

        // Compound command "echo foo" with additional arguments `["bar"]`
        let gate =
            Gate::new("compound_test", "echo foo", vec!["bar".to_string()]);

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
        let tmp_dir = temp_workspace("report_test");
        let artifacts_dir = tmp_dir.join("artifacts");
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&artifacts_dir).unwrap();

        outcome("fmt", Verdict::Pass, "Formatting clean")
            .save_to_dir(&artifacts_dir)
            .unwrap();
        fs::write(artifacts_dir.join("fmt.log"), "rustfmt complete\n").unwrap();

        outcome("clippy", Verdict::Warn, "Clippy completed with 1 warning")
            .save_to_dir(&artifacts_dir)
            .unwrap();
        fs::write(
            artifacts_dir.join("clippy.log"),
            "warning: unused variable `x`\n",
        )
        .unwrap();

        let config = parse_config(
            "[fmt]\ncommand = \"cargo fmt\"\n\
         [clippy]\nmode = \"warn\"\ncommand = \"cargo clippy\"\n",
        );

        let aggregator =
            ReportAggregator::new(artifacts_dir.clone(), tmp_dir.clone());
        let WrittenReport {
            pass: is_pass,
            path: report_path,
        } = aggregator.write_report(&config, None).unwrap();

        assert!(is_pass);
        assert!(report_path.exists());
        assert_eq!(report_path, artifacts_dir.join("ci-report.md"));
        assert!(!tmp_dir.join("ci-report.md").exists());

        let content = fs::read_to_string(&report_path).unwrap();
        assert!(
            content.contains("# Continuous Integration & Verification Report")
        );
        assert!(content.contains("`fmt`"));
        assert!(content.contains("`clippy`"));
        assert!(content.contains("warning: unused variable `x`"));
        assert!(content.len() <= MAX_REPORT_BYTES);

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_report_aggregator_subset_filtering() {
        let tmp_dir = temp_workspace("report_subset_test");
        let artifacts_dir = tmp_dir.join("artifacts");
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&artifacts_dir).unwrap();

        outcome("fmt", Verdict::Pass, "Formatting clean")
            .save_to_dir(&artifacts_dir)
            .unwrap();
        outcome("check", Verdict::Pass, "Check clean")
            .save_to_dir(&artifacts_dir)
            .unwrap();

        let config = parse_config(
            "[fmt]\ncommand = \"cargo fmt\"\n\
         [check]\nmode = \"skip\"\ncommand = \"cargo check\"\n",
        );

        let aggregator =
            ReportAggregator::new(artifacts_dir.clone(), tmp_dir.clone());
        let WrittenReport {
            pass: is_pass,
            path: report_path,
        } = aggregator
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
    fn test_arg_spelling_matches_the_command_line() {
        use control_rs_ci::cli::arg_spelling;
        use lexopt::Arg;
        assert_eq!(arg_spelling(&Arg::Long("config")), "--config");
        assert_eq!(arg_spelling(&Arg::Short('c')), "-c");
        assert_eq!(arg_spelling(&Arg::Value("x".into())), "x");
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

        let group_args = vec![
            "cargo-ci".to_string(),
            "--group".to_string(),
            "lint,audit".to_string(),
        ];
        let group_options = parse_args(&group_args, "cargo ci");
        assert_eq!(group_options.groups, vec!["lint", "audit"]);

        let short_group_args = vec![
            "cargo-ci".to_string(),
            "-g".to_string(),
            "verify".to_string(),
        ];
        let short_group_options = parse_args(&short_group_args, "cargo ci");
        assert_eq!(short_group_options.groups, vec!["verify"]);
    }

    #[test]
    fn test_cli_max_jobs_args_parsing() {
        use control_rs_ci::cli::parse_args;

        let max_jobs_args = vec![
            "cargo-ci".to_string(),
            "--max-jobs".to_string(),
            "4".to_string(),
        ];
        let max_jobs_options = parse_args(&max_jobs_args, "cargo ci");
        assert_eq!(max_jobs_options.max_jobs, Some(4));

        let eq_jobs_args =
            vec!["cargo-ci".to_string(), "--max-jobs=3".to_string()];
        let eq_jobs_options = parse_args(&eq_jobs_args, "cargo ci");
        assert_eq!(eq_jobs_options.max_jobs, Some(3));

        let short_jobs_args =
            vec!["cargo-ci".to_string(), "-j".to_string(), "2".to_string()];
        let short_jobs_options = parse_args(&short_jobs_args, "cargo ci");
        assert_eq!(short_jobs_options.max_jobs, Some(2));
    }

    #[test]
    fn test_cli_passthrough_parsing_and_rules() {
        use control_rs_ci::cli::{
            PASSTHROUGH_BINARY, check_passthrough, parse_args,
        };

        let to_args = |v: &[&str]| -> Vec<String> {
            v.iter().map(|a| (*a).to_string()).collect()
        };
        let config: GateConfig = toml::from_str(
        "[mutants]\ncommand = \"cargo mutants\"\n[fmt]\ncommand = \"cargo fmt\"\n",
    )
    .unwrap();

        let args =
            to_args(&["gate", "--only", "mutants", "--", "--jobs", "8", "-v"]);
        let options = parse_args(&args, PASSTHROUGH_BINARY);
        assert_eq!(options.only_gates, vec!["mutants"]);
        assert!(!options.verbose, "flags after `--` belong to the gate");
        assert_eq!(
            options.passthrough.as_deref(),
            Some(&to_args(&["--jobs", "8", "-v"])[..])
        );
        assert!(
            check_passthrough(PASSTHROUGH_BINARY, &options, &config).is_ok()
        );
        assert!(check_passthrough("cargo ci", &options, &config).is_err());

        let empty =
            parse_args(&to_args(&["gate", "fmt", "--"]), PASSTHROUGH_BINARY);
        assert_eq!(empty.passthrough, Some(Vec::new()));

        let rejected = [
            to_args(&["gate", "--only", "mutants,fmt", "--", "-x"]),
            to_args(&["gate", "--", "-x"]),
            to_args(&["gate", "--all", "mutants", "--", "-x"]),
            to_args(&["gate", "mutants", "--skip", "mutants", "--", "-x"]),
            to_args(&["gate", "missing", "--", "-x"]),
        ];
        for args in &rejected {
            let options = parse_args(args, PASSTHROUGH_BINARY);
            assert!(
                check_passthrough(PASSTHROUGH_BINARY, &options, &config)
                    .is_err(),
                "{args:?} must be rejected"
            );
        }
        let plain = parse_args(&to_args(&["gate", "fmt"]), PASSTHROUGH_BINARY);
        assert!(check_passthrough("cargo ci", &plain, &config).is_ok());
    }

    #[test]
    fn test_cli_unknown_selection_names_warn_and_drop() {
        use control_rs_ci::cli::{parse_args, resolve_selection};

        let config = parse_config(
            "[execution.groups]\nlint = [\"fmt\"]\n\
         [fmt]\ncommand = \"cargo fmt\"\n[test]\ncommand = \"cargo test\"\n",
        );
        let args: Vec<String> = [
            "gate",
            "bench",
            "test",
            "--group",
            "lint,nope",
            "--skip",
            "gone",
            "--up-to",
            "none",
        ]
        .iter()
        .map(|a| (*a).to_string())
        .collect();
        let mut options = parse_args(&args, "cargo gate");
        let warnings = resolve_selection(&mut options, &config);

        assert_eq!(options.only_gates, vec!["fmt", "test"]);
        assert_eq!(
            warnings,
            vec![
                "unknown group 'nope'",
                "unknown gate or group 'bench'",
                "unknown gate 'gone'",
                "unknown gate 'none'",
            ]
        );
    }

    #[test]
    fn test_cli_multi_token_parsing() {
        use control_rs_ci::cli::parse_args;

        let multi_space_args = vec![
            "cargo-ci".to_string(),
            "--skip".to_string(),
            "fmt,clippy,check,build,test".to_string(),
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

        let tmp_dir = temp_workspace("clean_test");
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

        [build]
        command = "cargo build"
        args = ["--workspace", "--all-targets"]
        description = "Compiles all targets"

        [custom]
        command = "my-tool"
        args = ["check", "--strict"]
    "#;

        let config = parse_config(toml_str);
        let build_def = config.gate_def("build");
        assert_eq!(
            build_def,
            Some(&GateDefinition {
                command: "cargo build".to_string(),
                args: vec![
                    "--workspace".to_string(),
                    "--all-targets".to_string(),
                ],
                description: Some("Compiles all targets".to_string()),
                env: HashMap::new(),
                mode: None,
                default: true,
                cwd: None,
                timeout_secs: None,
                skip_exit_codes: Vec::new(),
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
                default: true,
                cwd: None,
                timeout_secs: None,
                skip_exit_codes: Vec::new(),
            })
        );

        // The retired central `[gates]` policy table is no longer accepted.
        assert!(
            GateConfig::parse(
                "[gates]\nfmt = \"fail\"\n",
                Path::new("gate.toml")
            )
            .is_err()
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

        let config = parse_config(toml_str);
        assert_eq!(config.policy_for("clean"), GatePolicy::Fail);
        assert_eq!(config.policy_for("check"), GatePolicy::Skip);
        assert_eq!(config.policy_for("lint"), GatePolicy::Warn);
        assert_eq!(config.policy_for("custom"), GatePolicy::Fail);
        assert_eq!(config.policy_for("unknown"), GatePolicy::Fail);

        // Verify Gate::from_definition inherits mode
        let clean_def = config.gate_def("clean").unwrap();
        assert_eq!(clean_def.mode(), GatePolicy::Fail);
        let gate = Gate::from_definition("clean", clean_def, 30);
        assert_eq!(gate.mode(), GatePolicy::Fail);
        assert_eq!(gate.timeout, std::time::Duration::from_secs(30));
    }

    #[test]
    fn test_build_all_gates_follows_pipeline_order() {
        let toml_str = r#"
        [execution.exclusive]
        pre = ["fetch"]
        post = ["valgrind"]

        [execution.groups]
        cargo = ["fmt", "clippy"]

        [fetch]
        command = "cargo fetch"

        [fmt]
        command = "cargo fmt"

        [clippy]
        command = "cargo clippy"

        [valgrind]
        mode = "warn"
        command = "valgrind"
    "#;

        let gates = build_all_gates(&parse_config(toml_str));
        let names: Vec<&str> = gates.iter().map(|g| g.name()).collect();
        assert_eq!(names, vec!["fetch", "fmt", "clippy", "valgrind"]);

        // A scheduled gate without a definition is a configuration error.
        let missing = GateConfig::parse(
            "[execution.groups]\ncargo = [\"missing_gate\"]\n",
            Path::new("custom/gate.toml"),
        );
        let message = missing.unwrap_err().to_string();
        assert!(message.contains("custom/gate.toml"), "{message}");
        assert!(message.contains("missing_gate"), "{message}");
    }

    #[test]
    fn test_cli_verbose_flag() {
        use control_rs_ci::cli::parse_args;

        let default_args = vec!["cargo-ci".to_string()];
        assert!(!parse_args(&default_args, "cargo ci").verbose);

        let long_args = vec!["cargo-ci".to_string(), "--verbose".to_string()];
        assert!(parse_args(&long_args, "cargo ci").verbose);

        let short_args = vec![
            "cargo-ci".to_string(),
            "-v".to_string(),
            "--group".to_string(),
            "lint".to_string(),
        ];
        let short_options = parse_args(&short_args, "cargo ci");
        assert!(short_options.verbose);
        assert_eq!(short_options.groups, vec!["lint"]);
    }

    #[cfg(unix)]
    #[test]
    fn test_gate_echo_preserves_log_and_verdict() {
        let tmp_dir = temp_workspace("echo_test");
        let out_dir = tmp_dir.join("artifacts");
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&out_dir).unwrap();

        let gate = Gate::new(
            "echo_streams",
            "sh",
            vec![
                "-c".to_string(),
                "echo to_stdout; echo to_stderr 1>&2; exit 3".to_string(),
            ],
        );
        let ctx = GateContext {
            workspace_root: tmp_dir.clone(),
            out_dir: out_dir.clone(),
            default_timeout: std::time::Duration::from_secs(10),
        };

        let outcome =
            gate.execute_with_echo(&ctx, Some("[t] echo | ")).unwrap();
        assert_eq!(outcome.verdict, Verdict::Fail);
        assert_eq!(outcome.exit_code, Some(3));

        let log = fs::read_to_string(out_dir.join("echo_streams.log")).unwrap();
        assert!(
            log.contains("to_stdout\n"),
            "stdout missing from log: {log}"
        );
        assert!(
            log.contains("to_stderr\n"),
            "stderr missing from log: {log}"
        );

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_missing_tool_under_warn_policy_records_warn() {
        let tmp_dir = temp_workspace("degraded_tool");
        let out_dir = tmp_dir.join("artifacts");
        fs::create_dir_all(&out_dir).unwrap();
        let config = parse_config(
            "[absent]\nmode = \"warn\"\ncommand = \"control-rs-ci-no-such-tool --flag\"\n",
        );
        let gate = build_all_gates(&config).remove(0);
        let ctx = GateContext {
            workspace_root: tmp_dir.clone(),
            out_dir: out_dir.clone(),
            default_timeout: std::time::Duration::from_secs(10),
        };

        let outcome = gate.execute(&ctx).unwrap();
        assert_eq!(outcome.verdict, Verdict::Warn);
        assert_eq!(outcome.exit_code, None);
        let log = fs::read_to_string(out_dir.join("absent.log")).unwrap();
        assert!(log.contains("failed to execute"), "{log}");
        assert!(out_dir.join("absent.result.json").exists());

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_oversized_multibyte_logs_stay_within_budget() {
        let tmp_dir = temp_workspace("report_budget");
        let artifacts_dir = tmp_dir.join("artifacts");
        fs::create_dir_all(&artifacts_dir).unwrap();

        // Every log tail alone is close to the budget and every line carries a
        // two-byte character, so any byte-offset cut can split one.
        let line =
            format!("time: [41.1 µs 41.4 µs 41.7 µs] {}\n", "µ".repeat(1200));
        let mut text = String::new();
        for name in ["a", "b", "c", "d"] {
            outcome(name, Verdict::Fail, "regressed")
                .save_to_dir(&artifacts_dir)
                .unwrap();
            fs::write(
                artifacts_dir.join(format!("{name}.log")),
                line.repeat(30),
            )
            .unwrap();
            let _ = write!(text, "[{name}]\ncommand = \"false\"\n");
        }
        let config = parse_config(&text);

        let aggregator = ReportAggregator::new(artifacts_dir, tmp_dir.clone());
        let report = aggregator.write_report(&config, None).unwrap();
        assert!(!report.pass);
        let content = fs::read_to_string(&report.path).unwrap();
        assert!(content.len() <= MAX_REPORT_BYTES, "{}", content.len());
        assert!(content.contains("Gate: a (Fail)"));
        assert!(content.contains("Logs omitted for the size budget"));
        // Every embedded code fence is closed.
        assert_eq!(content.matches("```").count() % 2, 0);

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_rejected_and_missing_results_fail_the_report() {
        let tmp_dir = temp_workspace("report_rejected");
        let artifacts_dir = tmp_dir.join("artifacts");
        fs::create_dir_all(&artifacts_dir).unwrap();
        let mut stale = outcome("fmt", Verdict::Pass, "clean");
        stale.schema = control_rs_ci::gate::OUTCOME_SCHEMA + 1;
        stale.save_to_dir(&artifacts_dir).unwrap();
        fs::write(artifacts_dir.join("clippy.result.json"), "{ not json")
            .unwrap();
        let config = parse_config(
            "[fmt]\ncommand = \"cargo fmt\"\n[clippy]\ncommand = \"cargo clippy\"\n",
        );

        let aggregator = ReportAggregator::new(artifacts_dir, tmp_dir.clone());
        let loaded = aggregator.load_outcomes().unwrap();
        assert!(loaded.outcomes.is_empty());
        assert_eq!(loaded.rejected.len(), 2);

        let report = aggregator.write_report(&config, None).unwrap();
        assert!(!report.pass);
        let content = fs::read_to_string(&report.path).unwrap();
        assert!(content.contains("| `fmt` | **MISSING** |"), "{content}");
        assert!(content.contains("Rejected Result Records"));

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_workspace_gate_toml_mutants_is_executable() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_gate_toml =
            manifest_dir.parent().unwrap().join(".cargo/gate.toml");
        let config = GateConfig::load_from_path(&workspace_gate_toml).unwrap();
        assert_eq!(
            config.policy_for("mutants"),
            control_rs_ci::config::GatePolicy::Fail,
            "mutants gate must be executable (not mode = skip)"
        );
        let mutants_def = config.gate_definitions.get("mutants").unwrap();
        assert!(
            !mutants_def.default,
            "mutants gate must not run by default in unfiltered CI"
        );
    }
}
