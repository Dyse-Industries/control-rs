//! Scheduling tests: exclusive `pre` and `post` stages, concurrent groups in
//! declaration order and the `max_jobs` bound (FR-8, FR-9, FR-13).

#[cfg(test)]
mod scheduling {

    use std::fs;
    use std::path::{Path, PathBuf};

    use control_rs_ci::config::{ExecutionConfig, GateConfig, Stage};
    use control_rs_ci::gate::Verdict;
    use control_rs_ci::report::ReportAggregator;
    use control_rs_ci::{PipelineOptions, run_pipeline};

    /// Tags in start order and the largest number running at once.
    type Stamps = (Vec<String>, usize);

    /// Fresh workspace, unique to this process, holding `gate.toml` = `config`.
    fn workspace(name: &str, config: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir()
            .join(format!("control_rs_ci_sched_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let config_path = root.join("gate.toml");
        fs::write(&config_path, config).unwrap();
        (root, config_path)
    }

    fn parse(text: &str) -> GateConfig {
        GateConfig::parse(text, Path::new("gate.toml")).unwrap()
    }

    /// A gate table whose command appends `S <tag>` to `stamps`, sleeps, then
    /// appends `E <tag>`. Appends of one short line are atomic, so the file
    /// records the order in which gates started and ended.
    fn probe(gate: &str, tag: &str, secs: &str) -> String {
        format!(
            "[{gate}]\ncommand = \"sh\"\nargs = [\"-c\", \"echo S {tag} >> stamps; sleep {secs}; echo E {tag} >> stamps\"]\n"
        )
    }

    /// Start order and the largest number of simultaneously running tags.
    fn read_stamps(root: &Path) -> Stamps {
        let text = fs::read_to_string(root.join("stamps")).unwrap_or_default();
        let mut starts = Vec::new();
        let (mut running, mut peak) = (0_usize, 0_usize);
        for line in text.lines() {
            match line.split_once(' ') {
                Some(("S", tag)) => {
                    starts.push(tag.to_string());
                    running = running.saturating_add(1);
                    peak = peak.max(running);
                }
                Some(("E", _)) => running = running.saturating_sub(1),
                _ => panic!("malformed stamp line {line:?}"),
            }
        }
        (starts, peak)
    }

    /// Three groups declared out of alphabetical order, one probe gate each.
    fn three_groups() -> String {
        let mut text = String::from(
            "[runner]\nout_dir = \"artifacts\"\ntimeout_secs = 30\n\n\
         [execution.groups]\nzeta = [\"z\"]\nalpha = [\"a\"]\nmid = [\"m\"]\n\n",
        );
        for (gate, tag) in [("z", "zeta"), ("a", "alpha"), ("m", "mid")] {
            text.push_str(&probe(gate, tag, "0.4"));
        }
        text
    }

    #[test]
    fn test_execution_config_defaults() {
        let config = ExecutionConfig::default();
        assert!(
            config.exclusive.pre.is_empty(),
            "{:?}",
            config.exclusive.pre
        );
        assert!(
            config.exclusive.post.is_empty(),
            "{:?}",
            config.exclusive.post
        );
        assert!(config.groups.is_empty());
        assert_eq!(config.max_jobs, None);
    }

    #[test]
    fn test_execution_config_toml_deserialization() {
        let config = parse(
            r#"
        [execution]
        max_jobs = 3

        [execution.exclusive]
        pre = ["fetch"]
        post = ["cross-compare", "custom-exclusive"]

        [execution.groups]
        cargo = ["fmt", "clippy"]
        audit = ["deny"]

        [fetch]
        command = "cargo fetch"
        [cross-compare]
        command = "true"
        [custom-exclusive]
        command = "true"
        [fmt]
        command = "true"
        [clippy]
        command = "true"
        [deny]
        command = "true"
    "#,
        );
        let exec = &config.execution;
        assert_eq!(exec.max_jobs, Some(3));
        assert_eq!(exec.exclusive.pre, ["fetch"]);
        assert_eq!(exec.exclusive.post, ["cross-compare", "custom-exclusive"]);
        assert_eq!(exec.groups.names().collect::<Vec<_>>(), ["cargo", "audit"]);
        assert_eq!(
            exec.groups.get("cargo"),
            Some(&vec!["fmt".to_string(), "clippy".to_string()])
        );
    }

    #[test]
    fn test_retired_execution_keys_are_rejected() {
        for text in [
            "[execution]\nexclusive_gates = []\n",
            "[execution.lanes]\ncargo = []\n",
            "[execution]\nparallel = true\n",
        ] {
            assert!(
                GateConfig::parse(text, Path::new("gate.toml")).is_err(),
                "{text}"
            );
        }
    }

    #[test]
    fn test_unassigned_gate_runs_in_post() {
        let config = parse(
            "[execution.groups]\ncargo = [\"fmt\"]\n\
         [fmt]\ncommand = \"true\"\n[loose]\ncommand = \"true\"\n",
        );
        assert_eq!(config.stage_of("fmt"), Stage::Group("cargo"));
        assert_eq!(config.stage_of("loose"), Stage::Post);
    }

    #[test]
    fn test_max_jobs_zero_rejected() {
        let (root, config_path) = workspace(
            "max_jobs_zero",
            "[runner]\nout_dir = \"artifacts\"\ntimeout_secs = 10\n\n[execution.groups]\ng1 = [\"t1\"]\n\n[t1]\ncommand = \"true\"\n",
        );
        let options = PipelineOptions {
            max_jobs: Some(0),
            ..PipelineOptions::default()
        };
        let err = run_pipeline(&root, &config_path, &options).unwrap_err();
        assert!(
            err.to_string()
                .contains("max_jobs must be greater than zero")
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_bounded_group_concurrency_and_declaration_order() {
        for (max_jobs, expected_peak) in [(1, 1), (2, 2), (3, 3)] {
            let (root, config_path) =
                workspace(&format!("bound_{max_jobs}"), &three_groups());
            let options = PipelineOptions {
                max_jobs: Some(max_jobs),
                ..PipelineOptions::default()
            };
            assert!(run_pipeline(&root, &config_path, &options).unwrap());

            let (starts, peak) = read_stamps(&root);
            assert_eq!(peak, expected_peak, "max_jobs = {max_jobs}");
            if max_jobs == 1 {
                // One worker takes the queue strictly in declaration order.
                assert_eq!(starts, ["zeta", "alpha", "mid"]);
            }
            let _ = fs::remove_dir_all(&root);
        }
    }

    #[test]
    fn test_pre_runs_alone_before_groups_and_post_after() {
        let mut text = String::from(
            "[runner]\nout_dir = \"artifacts\"\ntimeout_secs = 30\n\n\
         [execution.exclusive]\npre = [\"first\"]\npost = [\"last\"]\n\n\
         [execution.groups]\ng1 = [\"a\"]\ng2 = [\"b\"]\n\n",
        );
        for (gate, tag) in
            [("first", "first"), ("a", "a"), ("b", "b"), ("last", "last")]
        {
            text.push_str(&probe(gate, tag, "0.2"));
        }
        let (root, config_path) = workspace("stages", &text);
        assert!(
            run_pipeline(&root, &config_path, &PipelineOptions::default())
                .unwrap()
        );

        let stamps = fs::read_to_string(root.join("stamps")).unwrap();
        let lines: Vec<&str> = stamps.lines().collect();
        assert_eq!(lines.first(), Some(&"S first"));
        assert_eq!(lines.get(1), Some(&"E first"));
        assert_eq!(lines.get(lines.len() - 2), Some(&"S last"));
        assert_eq!(lines.last(), Some(&"E last"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_failed_pre_gate_aborts_the_run() {
        let text = "[runner]\nout_dir = \"artifacts\"\ntimeout_secs = 30\n\n\
         [execution.exclusive]\npre = [\"fetch\"]\npost = [\"last\"]\n\n\
         [execution.groups]\ng1 = [\"a\"]\n\n\
         [fetch]\ncommand = \"false\"\n\
         [a]\ncommand = \"touch\"\nargs = [\"a-ran\"]\n\
         [last]\ncommand = \"touch\"\nargs = [\"last-ran\"]\n";
        let (root, config_path) = workspace("pre_abort", text);
        // A result from an earlier run must not stand in for a gate that did not run.
        let artifacts = root.join("artifacts");
        fs::create_dir_all(&artifacts).unwrap();
        fs::write(
        artifacts.join("a.result.json"),
        format!(
            "{{\"schema\":{},\"gate\":\"a\",\"verdict\":\"pass\",\"exit_code\":0,\"duration_secs\":0.0,\"summary\":null,\"log_file\":\"a.log\"}}",
            control_rs_ci::gate::OUTCOME_SCHEMA
        ),
    )
    .unwrap();

        assert!(
            !run_pipeline(&root, &config_path, &PipelineOptions::default())
                .unwrap()
        );
        assert!(!root.join("a-ran").exists());
        assert!(!root.join("last-ran").exists());

        let aggregator = ReportAggregator::new(artifacts.clone(), root.clone());
        let loaded = aggregator.load_outcomes().unwrap();
        assert_eq!(
            loaded.outcomes.get("fetch").map(|o| o.verdict),
            Some(Verdict::Fail)
        );
        assert!(!loaded.outcomes.contains_key("a"));
        let report =
            fs::read_to_string(artifacts.join("ci-report.md")).unwrap();
        assert!(report.contains("| `a` | **MISSING** |"), "{report}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_pre_gate_runs_only_when_selected() {
        let text = "[runner]\nout_dir = \"artifacts\"\n\n\
         [execution.exclusive]\npre = [\"fetch\"]\n\n\
         [execution.groups]\ng1 = [\"a\"]\n\n\
         [fetch]\ncommand = \"touch\"\nargs = [\"fetched\"]\n\
         [a]\ncommand = \"true\"\n";
        let (root, config_path) = workspace("pre_selection", text);
        let only = vec!["a".to_string()];
        let options = PipelineOptions {
            only_gates: Some(&only),
            ..PipelineOptions::default()
        };
        assert!(run_pipeline(&root, &config_path, &options).unwrap());
        assert!(!root.join("fetched").exists());

        assert!(
            run_pipeline(&root, &config_path, &PipelineOptions::default())
                .unwrap()
        );
        assert!(root.join("fetched").exists());
        let _ = fs::remove_dir_all(&root);
    }
}
