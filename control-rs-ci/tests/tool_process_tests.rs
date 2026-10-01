//! Process-level tests of the `valgrind`, `regression` and `ets` binaries.
//!
//! The external tools they drive (`cargo`, `valgrind`) are replaced by small
//! shell scripts placed first on `PATH`.

#![cfg(unix)]

#[cfg(test)]
mod tool_process {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    const BUDGETS: &str = "[budgets]\n\"x/*\" = \"1 ms\"\n";

    const CLEAN_VALGRIND: &str = "exit 0";

    const ETS_BIN: &str = env!("CARGO_BIN_EXE_ets");

    const REGRESSION_BIN: &str = env!("CARGO_BIN_EXE_regression");

    const VALGRIND_BIN: &str = env!("CARGO_BIN_EXE_valgrind");

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir()
            .join(format!("control_rs_ci_tools_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("bin")).unwrap();
        root
    }

    /// Installs an executable shell script named `name` in `root/bin`.
    fn install(root: &Path, name: &str, body: &str) {
        let path = root.join("bin").join(name);
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// Runs `bin` in `root` with only `root/bin` (and `/usr/bin`-free) on `PATH`.
    fn run(bin: &str, root: &Path, args: &[&str]) -> Output {
        Command::new(bin)
            .args(args)
            .current_dir(root)
            .env("PATH", root.join("bin"))
            .env("NO_COLOR", "1")
            .env_remove("CARGO_TARGET_DIR")
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

    // --- valgrind ---------------------------------------------------------------

    fn valgrind_workspace(name: &str, valgrind_body: Option<&str>) -> PathBuf {
        let root = scratch(name);
        install(&root, "cargo", "exit 0");
        if let Some(body) = valgrind_body {
            install(&root, "valgrind", body);
        }
        let examples = root.join("target/debug/examples");
        fs::create_dir_all(&examples).unwrap();
        fs::write(examples.join("demo"), "").unwrap();
        root
    }

    #[test]
    fn valgrind_help_prints_usage() {
        let root = scratch("vg_help");
        let out = run(VALGRIND_BIN, &root, &["--help"]);
        assert_eq!(out.status.code(), Some(0));
        assert!(text(&out).contains("Usage:"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn valgrind_passes_a_clean_example() {
        let root = valgrind_workspace("vg_clean", Some(CLEAN_VALGRIND));
        let out = run(VALGRIND_BIN, &root, &["--example", "demo"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        assert!(text(&out).contains("clean (0 leaks, 0 errors)"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn valgrind_reports_an_example_with_memory_errors() {
        let root = valgrind_workspace(
            "vg_dirty",
            Some("[ \"$1\" = \"--version\" ] && exit 0\necho leak >&2\nexit 1"),
        );
        let out = run(VALGRIND_BIN, &root, &["--example", "demo"]);
        assert_eq!(out.status.code(), Some(1));
        assert!(text(&out).contains("reported memory errors or leaks"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn valgrind_reports_a_missing_example_binary() {
        let root = valgrind_workspace("vg_missing", Some(CLEAN_VALGRIND));
        let out = run(VALGRIND_BIN, &root, &["--example", "absent"]);
        assert_eq!(out.status.code(), Some(1));
        assert!(text(&out).contains("Example binary not found"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn valgrind_requires_an_installed_working_valgrind() {
        let missing = valgrind_workspace("vg_absent", None);
        let out = run(VALGRIND_BIN, &missing, &["--example", "demo"]);
        assert_eq!(out.status.code(), Some(1));
        assert!(text(&out).contains("not installed"), "{}", text(&out));
        let _ = fs::remove_dir_all(&missing);

        let broken = valgrind_workspace("vg_broken", Some("exit 1"));
        let out = run(VALGRIND_BIN, &broken, &["--example", "demo"]);
        assert_eq!(out.status.code(), Some(1));
        assert!(text(&out).contains("not installed"), "{}", text(&out));
        let _ = fs::remove_dir_all(&broken);
    }

    #[test]
    fn valgrind_stops_when_the_examples_do_not_build() {
        let root = valgrind_workspace("vg_build", Some(CLEAN_VALGRIND));
        install(&root, "cargo", "exit 7");
        let out = run(VALGRIND_BIN, &root, &["--example", "demo"]);
        assert_eq!(out.status.code(), Some(1));
        assert!(text(&out).contains("compiling examples"), "{}", text(&out));
        let _ = fs::remove_dir_all(&root);
    }

    // --- regression -------------------------------------------------------------

    fn regression_workspace(name: &str, criterion_line: &str) -> PathBuf {
        let root = scratch(name);
        fs::write(root.join("budgets.toml"), BUDGETS).unwrap();
        install(
            &root,
            "cargo",
            &format!("echo 'x/bench           time:   [{criterion_line}]'"),
        );
        root
    }

    #[test]
    fn regression_help_prints_usage() {
        let root = scratch("reg_help");
        let out = run(REGRESSION_BIN, &root, &["--help"]);
        assert_eq!(out.status.code(), Some(0));
        assert!(text(&out).contains("Usage:"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn regression_passes_benchmarks_within_budget() {
        let root = regression_workspace("reg_pass", "1.0 µs 2.0 µs 3.0 µs");
        let out = run(REGRESSION_BIN, &root, &["--budgets", "budgets.toml"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        assert!(text(&out).contains("verification PASSED"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn regression_fails_benchmarks_over_budget() {
        let root = regression_workspace("reg_fail", "1.0 ms 2.0 ms 3.0 ms");
        let out = run(REGRESSION_BIN, &root, &["--budgets", "budgets.toml"]);
        assert_eq!(out.status.code(), Some(1), "{}", text(&out));
        let text = text(&out);
        assert!(text.contains("verification FAILED: 1/1"), "{text}");
        assert!(text.contains("exceeded cycle budget"), "{text}");
        let _ = fs::remove_dir_all(&root);
    }

    // --- ets --------------------------------------------------------------------

    #[test]
    fn ets_help_prints_usage() {
        let root = scratch("ets_help");
        let out = run(ETS_BIN, &root, &["--help"]);
        assert_eq!(out.status.code(), Some(0));
        assert!(text(&out).contains("Usage:"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn ets_fails_and_records_a_target_that_cannot_run() {
        let root = scratch("ets_fail");
        let out = run(
            ETS_BIN,
            &root,
            &[
                "--out",
                "results.json",
                "teensy",
                "--port",
                "/dev/control-rs-no-such-port",
            ],
        );
        assert_eq!(out.status.code(), Some(1), "{}", text(&out));
        assert!(text(&out).contains("1 of 1 target(s)"));
        let recorded = fs::read_to_string(root.join("results.json")).unwrap();
        assert!(recorded.contains("\"passed\": false"), "{recorded}");
        let _ = fs::remove_dir_all(&root);
    }
}
