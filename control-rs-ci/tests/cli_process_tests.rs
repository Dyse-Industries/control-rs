//! Process-level tests: the `control-rs-ci` and `gate` binaries run in a
//! scratch workspace, so exit codes, selection and cleaning are observed
//! end to end.

#[cfg(test)]
mod cli_process {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    const CI_BIN: &str = env!("CARGO_BIN_EXE_control-rs-ci");

    const COLOR_CONFIG: &str = r#"
[runner]
out_dir = "artifacts"
timeout_secs = 30

[probe]
command = "sh"
args = ["-c", "printf %s \"$CARGO_TERM_COLOR\" > color.txt"]
"#;

    const CONFIG: &str = r#"
[runner]
out_dir = "artifacts"
timeout_secs = 30

[execution.groups]
main = ["alpha", "beta"]
side = ["gamma"]

[alpha]
command = "touch"
args = ["ran-alpha"]
description = "first gate"

[beta]
command = "touch"
args = ["ran-beta"]

[gamma]
command = "touch"
args = ["ran-gamma"]

[opt-in]
default = false
command = "touch"
args = ["ran-opt-in"]
"#;

    const GATE_BIN: &str = env!("CARGO_BIN_EXE_gate");

    type EnvVar<'a> = (&'a str, &'a str);
    type Case<'a> = (&'a [&'a str], &'a [&'a str]);

    fn workspace(name: &str) -> PathBuf {
        let root = std::env::temp_dir()
            .join(format!("control_rs_ci_cli_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let written = fs::create_dir_all(root.join(".cargo"))
            .and_then(|()| fs::write(root.join(".cargo/gate.toml"), CONFIG));
        assert!(written.is_ok(), "cannot create workspace: {written:?}");
        root
    }

    fn run(bin: &str, root: &Path, args: &[&str]) -> Output {
        Command::new(bin)
            .args(args)
            .current_dir(root)
            .env("NO_COLOR", "1")
            .output()
            .unwrap()
    }

    fn ran(root: &Path, gate: &str) -> bool {
        root.join(format!("ran-{gate}")).exists()
    }

    fn stdout(out: &Output) -> String {
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    fn stderr(out: &Output) -> String {
        String::from_utf8_lossy(&out.stderr).into_owned()
    }

    fn seed_stale_artifact(root: &Path) -> PathBuf {
        let stale = root.join("artifacts/stale.txt");
        fs::create_dir_all(root.join("artifacts")).unwrap();
        fs::write(&stale, "old").unwrap();
        stale
    }

    #[test]
    fn help_flags_print_usage_and_exit_zero() {
        let root = workspace("help");
        for flag in ["--help", "-h"] {
            let out = run(CI_BIN, &root, &[flag]);
            assert_eq!(out.status.code(), Some(0), "{flag}");
            assert!(stdout(&out).contains("Usage:"), "{flag}");
            assert!(!ran(&root, "alpha"));
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn unknown_argument_exits_with_usage_error() {
        let root = workspace("unknown");
        let out = run(CI_BIN, &root, &["--frobnicate"]);
        assert_eq!(out.status.code(), Some(2));
        assert!(stderr(&out).contains("Unknown argument"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn list_prints_registered_gates_and_exits_zero() {
        let root = workspace("list");
        let out = run(CI_BIN, &root, &["--list"]);
        assert_eq!(out.status.code(), Some(0));
        let text = stdout(&out);
        assert!(text.contains("alpha"));
        assert!(text.contains("first gate"));
        assert!(!ran(&root, "alpha"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn default_run_uses_the_working_directory_and_skips_opt_in_gates() {
        let root = workspace("default");
        let out = run(CI_BIN, &root, &[]);
        assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
        assert!(ran(&root, "alpha"));
        assert!(ran(&root, "beta"));
        assert!(ran(&root, "gamma"));
        assert!(!ran(&root, "opt-in"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn clean_alone_removes_artifacts_without_running_gates() {
        for args in [&["--clean"][..], &["clean"][..]] {
            let root = workspace("clean_only");
            let stale = seed_stale_artifact(&root);
            let out = run(CI_BIN, &root, args);
            assert_eq!(out.status.code(), Some(0), "{args:?}");
            assert!(!stale.exists(), "{args:?}");
            assert!(!ran(&root, "alpha"), "{args:?}");
            let _ = fs::remove_dir_all(&root);
        }
    }

    #[test]
    fn clean_with_a_selection_runs_the_selection() {
        let cases: [Case<'_>; 4] = [
            (&["--clean", "--all"], &["alpha", "beta", "gamma", "opt-in"]),
            (&["--clean", "--group", "side"], &["gamma"]),
            (&["--clean", "--only", "beta"], &["beta"]),
            (&["--clean", "--up-to", "alpha"], &["alpha"]),
        ];
        for (args, expected) in cases {
            let root = workspace("clean_run");
            let stale = seed_stale_artifact(&root);
            let out = run(CI_BIN, &root, args);
            assert_eq!(
                out.status.code(),
                Some(0),
                "{args:?}: {}",
                stderr(&out)
            );
            assert!(!stale.exists(), "{args:?}: artifacts are cleaned first");
            for gate in ["alpha", "beta", "gamma", "opt-in"] {
                assert_eq!(
                    ran(&root, gate),
                    expected.contains(&gate),
                    "{args:?}: {gate}"
                );
            }
            let _ = fs::remove_dir_all(&root);
        }
    }

    #[test]
    fn all_runs_opt_in_gates_but_conflicts_with_a_selection() {
        let root = workspace("all");
        let out = run(CI_BIN, &root, &["--all"]);
        assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
        assert!(ran(&root, "opt-in"));
        let _ = fs::remove_dir_all(&root);

        for selection in [["--only", "alpha"], ["--group", "main"]] {
            let root = workspace("all_conflict");
            let mut args = vec!["--all"];
            args.extend(selection);
            let out = run(CI_BIN, &root, &args);
            assert_eq!(out.status.code(), Some(2), "{args:?}");
            assert!(stderr(&out).contains("--all"));
            assert!(!ran(&root, "alpha"));
            let _ = fs::remove_dir_all(&root);
        }
    }

    #[test]
    fn only_group_and_skip_select_gates() {
        let root = workspace("only");
        assert_eq!(
            run(CI_BIN, &root, &["--only", "alpha"]).status.code(),
            Some(0)
        );
        assert!(ran(&root, "alpha"));
        assert!(!ran(&root, "beta"));
        assert!(!ran(&root, "gamma"));
        let _ = fs::remove_dir_all(&root);

        let root = workspace("group");
        assert_eq!(
            run(CI_BIN, &root, &["--group", "main"]).status.code(),
            Some(0)
        );
        assert!(ran(&root, "alpha"));
        assert!(ran(&root, "beta"));
        assert!(!ran(&root, "gamma"));
        let _ = fs::remove_dir_all(&root);

        let root = workspace("skip");
        assert_eq!(
            run(CI_BIN, &root, &["--skip", "beta"]).status.code(),
            Some(0)
        );
        assert!(ran(&root, "alpha"));
        assert!(!ran(&root, "beta"));
        assert!(ran(&root, "gamma"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_selection_of_unknown_names_selects_nothing() {
        let root = workspace("unknown_only");
        let out = run(CI_BIN, &root, &["--only", "nope"]);
        assert_eq!(out.status.code(), Some(1));
        assert!(!ran(&root, "alpha"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn passthrough_binary_documents_passthrough_arguments() {
        let root = workspace("usage");
        let gate_help = stdout(&run(GATE_BIN, &root, &["--help"]));
        assert!(gate_help.contains("[-- <ARGS>...]"));
        let ci_help = stdout(&run(CI_BIN, &root, &["--help"]));
        assert!(!ci_help.contains("[-- <ARGS>...]"));
        let _ = fs::remove_dir_all(&root);
    }

    /// The `CARGO_TERM_COLOR` a gate sees when the runner starts with the given
    /// color-related environment.
    fn gate_color(name: &str, env: &[EnvVar<'_>], unset: &[&str]) -> String {
        let root = workspace(name);
        fs::write(root.join(".cargo/gate.toml"), COLOR_CONFIG).unwrap();
        let mut cmd = Command::new(CI_BIN);
        cmd.current_dir(&root);
        for var in ["CARGO_TERM_COLOR", "NO_COLOR", "TERM", "CLICOLOR"] {
            cmd.env_remove(var);
        }
        for var in unset {
            cmd.env_remove(var);
        }
        let out = cmd.envs(env.iter().copied()).output().unwrap();
        assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
        let seen = fs::read_to_string(root.join("color.txt")).unwrap();
        let _ = fs::remove_dir_all(&root);
        seen
    }

    #[test]
    fn gates_inherit_the_resolved_cargo_color_choice() {
        assert_eq!(
            gate_color("c_always", &[("CARGO_TERM_COLOR", "always")], &[]),
            "always"
        );
        assert_eq!(
            gate_color("c_never", &[("CARGO_TERM_COLOR", "never")], &[]),
            "never"
        );
        assert_eq!(
            gate_color(
                "c_always_over_no_color",
                &[("CARGO_TERM_COLOR", "always"), ("NO_COLOR", "1")],
                &[]
            ),
            "always"
        );
        assert_eq!(
            gate_color("c_no_color", &[("NO_COLOR", "1")], &[]),
            "never"
        );
        assert_eq!(gate_color("c_dumb", &[("TERM", "dumb")], &[]), "never");
        assert_eq!(gate_color("c_xterm", &[("TERM", "xterm")], &[]), "auto");
        assert_eq!(gate_color("c_unset", &[], &[]), "auto");
    }

    #[test]
    fn forced_color_reaches_the_help_output() {
        let root = workspace("forced_color");
        let colored = Command::new(CI_BIN)
            .arg("--help")
            .current_dir(&root)
            .env_remove("NO_COLOR")
            .env("CARGO_TERM_COLOR", "always")
            .output()
            .unwrap();
        assert!(stdout(&colored).contains("\u{1b}["), "no ANSI styling");
        let plain = Command::new(CI_BIN)
            .arg("--help")
            .current_dir(&root)
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap();
        assert!(
            !stdout(&plain).contains("\u{1b}["),
            "unexpected ANSI styling"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn group_tags_are_empty_without_a_name() {
        use control_rs_ci::ui::format_group_tag;
        let style = Some(anstyle::AnsiColor::Red.on_default());
        assert_eq!(format_group_tag(Some(""), style), "");
        assert_eq!(format_group_tag(Some(""), None), "");
        assert_eq!(format_group_tag(None, None), "");
        assert_eq!(format_group_tag(Some("lint"), None), "[lint] ");
        assert!(format_group_tag(Some("lint"), style).contains("[lint]"));
    }
}
