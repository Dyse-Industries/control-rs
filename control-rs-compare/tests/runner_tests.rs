//! Variant runner: suite filtering, timeouts and output verification.

#![cfg(unix)]

#[cfg(test)]
mod runner {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use control_rs_compare::config::{CompareConfigFile, MasterPlan};
    use control_rs_compare::runner::{RunnerOptions, execute_master_plan};

    fn workspace(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "control_rs_compare_runner_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn plan(text: &str) -> MasterPlan {
        let config: CompareConfigFile = toml::from_str(text).unwrap();
        config
            .resolve_master_plan(Path::new("compare.toml"))
            .unwrap()
    }

    fn options(root: &Path, timeout: Duration) -> RunnerOptions {
        RunnerOptions {
            workspace_root: root.to_path_buf(),
            out_dir: root.join("results"),
            timeout,
            quiet: true,
        }
    }

    fn variant(
        suite: &str,
        command: &str,
        output: &str,
        optional: bool,
    ) -> String {
        format!(
            "[[suite]]\nname = \"{suite}\"\n\
             [[suite.variants]]\nname = \"v\"\ntype = \"command\"\n\
             command = \"{command}\"\noutput_file = \"{output}\"\noptional = {optional}\n"
        )
    }

    #[test]
    fn the_suite_filter_selects_suites_to_run() {
        let root = workspace("filter");
        let text = format!(
            "{}{}",
            variant("a", "touch ran-a", "ran-a", false),
            variant("b", "touch ran-b", "ran-b", false)
        );
        let plan = plan(&text);

        execute_master_plan(
            &plan,
            &options(&root, Duration::from_secs(30)),
            &["a".to_string()],
        )
        .unwrap();
        assert!(root.join("ran-a").exists());
        assert!(!root.join("ran-b").exists());

        execute_master_plan(
            &plan,
            &options(&root, Duration::from_secs(30)),
            &[],
        )
        .unwrap();
        assert!(root.join("ran-b").exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_slow_variant_is_killed_at_the_timeout() {
        let root = workspace("timeout");
        let plan = plan(&variant("slow", "sleep 5", "never", false));
        let start = Instant::now();
        let err = execute_master_plan(
            &plan,
            &options(&root, Duration::from_millis(300)),
            &[],
        )
        .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("timeout exceeded"),
            "{err}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "{:?}",
            start.elapsed()
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_fast_variant_finishes_well_inside_its_timeout() {
        let root = workspace("fast");
        let plan = plan(&variant("fast", "touch out.h5", "out.h5", false));
        execute_master_plan(
            &plan,
            &options(&root, Duration::from_secs(30)),
            &[],
        )
        .unwrap();
        assert!(root.join("out.h5").exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_output_fails_unless_the_variant_is_optional() {
        let root = workspace("output");
        let required = plan(&variant("req", "true", "absent.h5", false));
        let err = execute_master_plan(
            &required,
            &options(&root, Duration::from_secs(30)),
            &[],
        )
        .unwrap_err();
        assert!(err.to_string().contains("was not created"), "{err}");

        let optional = plan(&variant("opt", "true", "absent.h5", true));
        execute_master_plan(
            &optional,
            &options(&root, Duration::from_secs(30)),
            &[],
        )
        .unwrap();

        let present =
            plan(&variant("present", "touch made.h5", "made.h5", false));
        execute_master_plan(
            &present,
            &options(&root, Duration::from_secs(30)),
            &[],
        )
        .unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_failing_optional_variant_does_not_stop_the_plan() {
        let root = workspace("optional_fail");
        let text = format!(
            "{}{}",
            variant("flaky", "exit 3", "x", true),
            variant("after", "touch ran-after", "ran-after", false)
        );
        execute_master_plan(
            &plan(&text),
            &options(&root, Duration::from_secs(30)),
            &[],
        )
        .unwrap();
        assert!(root.join("ran-after").exists());

        let hard = plan(&variant("hard", "exit 3", "x", false));
        assert!(
            execute_master_plan(
                &hard,
                &options(&root, Duration::from_secs(30)),
                &[]
            )
            .is_err()
        );
        let _ = fs::remove_dir_all(&root);
    }
}
