//! Process-level tests of the `compare` binary in a scratch workspace.

#![cfg(unix)]

#[cfg(test)]
mod cli_process {
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use hdf5_pure::FileBuilder;

    const BIN: &str = env!("CARGO_BIN_EXE_compare");

    fn container(values: &[f64]) -> Vec<u8> {
        let mut b = FileBuilder::new();
        b.create_dataset("x").with_f64_data(values);
        b.finish().unwrap()
    }

    /// Two suites, `a` and `b`, each with a `ref` oracle and a `rust` peer whose
    /// commands mark that they ran and deposit their fixture into `cfg_out`.
    /// Suite `b`'s peer differs from its oracle when `b_matches` is false.
    fn workspace(name: &str, b_matches: bool) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "control_rs_compare_cli_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("fixtures")).unwrap();
        let mut config = String::from("[compare]\nout_dir = \"cfg_out\"\n");
        for suite in ["a", "b"] {
            let _ = write!(
                config,
                "[[suite]]\nname = \"{suite}\"\ntrue_oracle = \"ref\"\n"
            );
            for variant in ["ref", "rust"] {
                let data = if suite == "b" && variant == "rust" && !b_matches {
                    [1.0, 9.0]
                } else {
                    [1.0, 2.0]
                };
                fs::write(
                    root.join(format!("fixtures/{suite}.{variant}.h5")),
                    container(&data),
                )
                .unwrap();
                let _ = write!(
                    config,
                    "[[suite.variants]]\nname = \"{variant}\"\ntype = \"command\"\n\
                     command = \"touch ran-{suite}-{variant}; mkdir -p cfg_out; \
                     cp fixtures/{suite}.{variant}.h5 cfg_out/{suite}.{variant}.h5\"\n\
                     output_file = \"cfg_out/{suite}.{variant}.h5\"\n"
                );
            }
        }
        fs::write(root.join("compare.toml"), config).unwrap();
        root
    }

    fn run(root: &Path, args: &[&str]) -> Output {
        Command::new(BIN)
            .args(["--config", "compare.toml"])
            .args(args)
            .current_dir(root)
            .output()
            .unwrap()
    }

    fn text(out: &Output) -> String {
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    }

    fn ran(root: &Path, suite: &str) -> bool {
        root.join(format!("ran-{suite}-ref")).exists()
    }

    fn report_in(root: &Path, dir: &str) -> bool {
        root.join(dir).join("cross-val-report.json").exists()
    }

    #[test]
    fn help_prints_usage() {
        let root = workspace("help", true);
        let out = run(&root, &["--help"]);
        assert_eq!(out.status.code(), Some(0));
        assert!(text(&out).contains("USAGE:"));
        assert!(!ran(&root, "a"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_default_run_executes_and_compares_every_suite() {
        let root = workspace("default", true);
        let out = run(&root, &[]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        assert!(ran(&root, "a") && ran(&root, "b"));
        assert!(
            report_in(&root, "cfg_out"),
            "reports follow the plan's out_dir"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_explicit_results_dir_overrides_the_plan() {
        let root = workspace("results_dir", true);
        let out = run(&root, &["--results-dir", "custom"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        assert!(report_in(&root, "custom"));
        assert!(!report_in(&root, "cfg_out"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn run_and_compare_selections_pick_suites() {
        let root = workspace("select", true);
        let out = run(&root, &["--run", "a", "--compare", "a"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        assert!(ran(&root, "a"));
        assert!(!ran(&root, "b"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_skipped_phase_stays_skipped() {
        let root = workspace("skip_run", true);
        let out = run(&root, &["--skip-run", "--run", "all", "--skip-compare"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        assert!(!ran(&root, "a"));
        let _ = fs::remove_dir_all(&root);

        let root = workspace("skip_compare", true);
        let out = run(&root, &["--skip-compare", "--compare", "all"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        assert!(ran(&root, "a"));
        assert!(!report_in(&root, "cfg_out"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_discrepancy_fails_only_the_suites_being_compared() {
        let root = workspace("discrepancy", false);
        let out = run(&root, &[]);
        assert_eq!(out.status.code(), Some(1), "{}", text(&out));
        assert!(
            report_in(&root, "cfg_out"),
            "the report is written even on failure"
        );
        let _ = fs::remove_dir_all(&root);

        let root = workspace("discrepancy_a", false);
        assert_eq!(run(&root, &["--compare", "a"]).status.code(), Some(0));
        let _ = fs::remove_dir_all(&root);

        let root = workspace("discrepancy_b", false);
        assert_eq!(run(&root, &["--compare", "b"]).status.code(), Some(1));
        let _ = fs::remove_dir_all(&root);

        let root = workspace("discrepancy_bypass", false);
        let out = run(&root, &["--bypass-gate"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        assert!(report_in(&root, "cfg_out"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_failing_variant_stops_the_run_unless_the_gate_is_bypassed() {
        let root = workspace("variant_fail", true);
        let broken = fs::read_to_string(root.join("compare.toml"))
            .unwrap()
            .replace("touch ran-a-ref;", "exit 3;");
        fs::write(root.join("compare.toml"), broken).unwrap();
        let out = run(&root, &["--skip-compare"]);
        assert_eq!(out.status.code(), Some(1), "{}", text(&out));
        assert!(text(&out).contains("Execution error"));

        let out = run(&root, &["--skip-compare", "--bypass-gate"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_comparison_error_fails_unless_the_gate_is_bypassed() {
        let root = workspace("compare_error", true);
        let out = run(&root, &["--skip-run", "--results-dir", "absent-dir"]);
        assert_eq!(out.status.code(), Some(1), "{}", text(&out));
        assert!(text(&out).contains("Comparison error"));
        let out = run(
            &root,
            &["--skip-run", "--results-dir", "absent-dir", "--bypass-gate"],
        );
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn quiet_suppresses_progress_and_the_report_but_not_the_verdict() {
        let root = workspace("quiet", true);
        let loud = text(&run(&root, &[]));
        assert!(loud.contains("Running"), "{loud}");
        assert!(
            loud.contains("Cross-Comparison & Oracle Validation Brief"),
            "{loud}"
        );
        let _ = fs::remove_dir_all(&root);

        let root = workspace("quiet_on", true);
        let out = run(&root, &["--quiet"]);
        assert_eq!(out.status.code(), Some(0));
        let quiet = text(&out);
        assert!(!quiet.contains("Running"), "{quiet}");
        assert!(!quiet.contains("Cross-Comparison"), "{quiet}");
        assert!(report_in(&root, "cfg_out"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_configuration_skips_execution_with_a_notice() {
        let root = workspace("no_plan", true);
        let loud = Command::new(BIN)
            .args(["--config", "absent.toml", "--skip-compare"])
            .current_dir(&root)
            .output()
            .unwrap();
        assert_eq!(loud.status.code(), Some(0));
        assert!(
            text(&loud).contains("No master plan loaded"),
            "{}",
            text(&loud)
        );

        let quiet = Command::new(BIN)
            .args(["--config", "absent.toml", "--skip-compare", "--quiet"])
            .current_dir(&root)
            .output()
            .unwrap();
        assert!(!text(&quiet).contains("No master plan loaded"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn unknown_arguments_are_usage_errors() {
        let root = workspace("unknown", true);
        let out = run(&root, &["--frobnicate"]);
        assert_eq!(out.status.code(), Some(2));
        assert!(text(&out).contains("Unrecognized CLI argument"));
        let _ = fs::remove_dir_all(&root);
    }
}
