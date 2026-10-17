//! Path-gated skip tests (FR-22 to FR-24): the pure check, then the binary in
//! a scratch git repository.

#[cfg(test)]
mod path_gate {
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use control_rs_ci::diff_trigger::{
        Decision, Triggers, changed_files, gate_var, global_hit, matches,
        override_for, parse_name_status, should_run,
    };

    const CI_BIN: &str = env!("CARGO_BIN_EXE_control-rs-ci");

    const CONFIG: &str = r#"
    [runner]
    out_dir = "artifacts"
    timeout_secs = 30

    [execution.groups]
    main = ["docs", "matrix", "always"]

    [docs]
    command = "touch"
    args = ["ran-docs"]
    diff-trigger = ["documentation/"]

    [matrix]
    command = "touch"
    args = ["ran-matrix"]
    diff-trigger = ["src/matrix/", "Cargo.toml"]

    [always]
    command = "touch"
    args = ["ran-always"]
    "#;

    type Pair<'a> = (&'a str, &'a str);
    type Files<'a> = Option<&'a [&'a str]>;
    type Env = Box<dyn Fn(&str) -> Option<String>>;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(ToString::to_string).collect()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env_of(pairs: &[Pair<'_>]) -> Env {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        Box::new(move |var| map.get(var).cloned())
    }

    fn own(entries: &[String]) -> Triggers<'_> {
        Triggers {
            own: entries,
            global_hit: false,
        }
    }

    fn decide(triggers: &[&str], changed: Files<'_>) -> Decision {
        let changed = changed.map(strings);
        should_run(
            "matrix",
            own(&strings(triggers)),
            changed.as_deref(),
            &no_env,
        )
        .unwrap()
    }

    #[test]
    fn a_gate_runs_when_a_changed_file_equals_or_lies_below_an_entry() {
        assert_eq!(
            decide(&["src/matrix/"], Some(&["src/matrix/a.rs"])),
            Decision::Run
        );
        assert_eq!(
            decide(&["src/matrix"], Some(&["src/matrix/a.rs"])),
            Decision::Run
        );
        assert_eq!(
            decide(&["Cargo.toml"], Some(&["Cargo.toml"])),
            Decision::Run
        );
    }

    #[test]
    fn a_gate_is_skipped_when_nothing_under_its_diff_trigger_changed() {
        assert!(matches!(
            decide(&["src/matrix/"], Some(&["documentation/a.md"])),
            Decision::Skip(_)
        ));
        assert!(matches!(
            decide(&["src/matrix/"], Some(&[])),
            Decision::Skip(_)
        ));
    }

    #[test]
    fn a_sibling_sharing_a_prefix_does_not_match() {
        assert!(!matches("src/matrix", "src/matrix_extra/a.rs"));
        assert!(!matches("Cargo.toml", "Cargo.toml.bak"));
        assert!(!matches("src/*", "src/a.rs"));
        assert!(!matches("", "src/a.rs"));
    }

    #[test]
    fn a_gate_without_a_diff_trigger_always_runs() {
        assert_eq!(decide(&[], Some(&["anything"])), Decision::Run);
        assert_eq!(decide(&[], Some(&[])), Decision::Run);
    }

    #[test]
    fn an_unresolved_diff_skips_no_gate() {
        assert_eq!(decide(&["src/matrix/"], None), Decision::Run);
    }

    #[test]
    fn a_rename_counts_by_both_paths_and_a_delete_by_its_path() {
        let out = "R100\tsrc/matrix/old.rs\tdocumentation/new.rs\nD\tsrc/tensor/x.rs\nM\tCargo.toml\n";
        assert_eq!(
            parse_name_status(out),
            strings(&[
                "src/matrix/old.rs",
                "documentation/new.rs",
                "src/tensor/x.rs",
                "Cargo.toml"
            ])
        );
    }

    #[test]
    fn the_gate_variable_is_upper_case_with_underscores() {
        assert_eq!(gate_var("mutants-ci-0"), "CONTROL_RS_CI_GATE_MUTANTS_CI_0");
    }

    #[test]
    fn the_global_variable_forces_every_gate() {
        let env = env_of(&[("CONTROL_RS_CI_GATES", "skip")]);
        let skip = should_run(
            "matrix",
            own(&strings(&["src/"])),
            Some(&strings(&["src/a.rs"])),
            &env,
        )
        .unwrap();
        assert!(matches!(skip, Decision::Skip(_)));
        let env = env_of(&[("CONTROL_RS_CI_GATES", "run")]);
        let run =
            should_run("matrix", own(&strings(&["src/"])), Some(&[]), &env)
                .unwrap();
        assert_eq!(run, Decision::Run);
    }

    #[test]
    fn the_per_gate_variable_beats_the_global_one() {
        let env = env_of(&[
            ("CONTROL_RS_CI_GATES", "skip"),
            ("CONTROL_RS_CI_GATE_MATRIX", "run"),
        ]);
        assert_eq!(
            should_run("matrix", own(&strings(&["src/"])), Some(&[]), &env)
                .unwrap(),
            Decision::Run
        );
        assert!(matches!(
            should_run("docs", own(&[]), Some(&[]), &env).unwrap(),
            Decision::Skip(_)
        ));
    }

    #[test]
    fn unset_and_empty_variables_defer_and_other_values_are_errors() {
        assert_eq!(override_for("matrix", &no_env).unwrap(), None);
        let empty = env_of(&[("CONTROL_RS_CI_GATES", "")]);
        assert_eq!(override_for("matrix", &empty).unwrap(), None);
        let bad = env_of(&[("CONTROL_RS_CI_GATE_MATRIX", "maybe")]);
        let err = override_for("matrix", &bad).unwrap_err();
        assert!(err.contains("CONTROL_RS_CI_GATE_MATRIX"), "{err}");
    }

    fn git(root: &Path, args: &[&str]) {
        let out = Command::new("git")
            .current_dir(root)
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {out:?}");
    }

    /// A repository on `main` with the config committed, then a `topic` branch
    /// that changes `changed`.
    fn repo(name: &str, changed: &[Pair<'_>]) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "control_rs_ci_diff_trigger_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join(".cargo")).unwrap();
        fs::write(root.join(".cargo/gate.toml"), CONFIG).unwrap();
        git(&root, &["init", "-q", "-b", "main"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "base"]);
        git(&root, &["checkout", "-q", "-b", "topic"]);
        for (path, text) in changed {
            let file = root.join(path);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(file, text).unwrap();
        }
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "topic"]);
        root
    }

    fn run(root: &Path, args: &[&str], env: &[Pair<'_>]) -> Output {
        let mut cmd = Command::new(CI_BIN);
        cmd.args(args).current_dir(root).env("NO_COLOR", "1");
        for var in ["CONTROL_RS_CI_GATES", "CONTROL_RS_CI_GATE_DOCS"] {
            cmd.env_remove(var);
        }
        cmd.envs(env.iter().copied()).output().unwrap()
    }

    fn ran(root: &Path, marker: &str) -> bool {
        root.join(marker).exists()
    }

    #[test]
    fn changed_files_lists_the_branch_diff() {
        let root = repo("diff", &[("documentation/a.md", "x")]);
        let files = changed_files(&root, "main", "topic").unwrap();
        assert_eq!(files, strings(&["documentation/a.md"]));
        assert!(changed_files(&root, "no-such-ref", "topic").is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn changed_files_keeps_non_ascii_paths_unquoted() {
        // Default `core.quotepath` would emit `"src/na\303\257ve.rs"`, which
        // does not match the `src/` prefix and would false-skip path-gated gates.
        let root = repo("quotepath", &[("src/naïve.rs", "x")]);
        let files = changed_files(&root, "main", "topic").unwrap();
        assert_eq!(files, strings(&["src/naïve.rs"]));
        assert!(
            matches("src/", "src/naïve.rs"),
            "unquoted path must match the src/ diff-trigger prefix"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_binary_skips_gates_whose_diff_trigger_did_not_change() {
        let root = repo("skip", &[("documentation/a.md", "x")]);
        let out = run(&root, &["--base", "main", "--head", "topic"], &[]);
        assert!(out.status.success(), "{out:?}");
        assert!(ran(&root, "ran-docs"));
        assert!(!ran(&root, "ran-matrix"));
        assert!(ran(&root, "ran-always"));
        let result =
            fs::read_to_string(root.join("artifacts/matrix.result.json"))
                .unwrap();
        assert!(result.contains("\"skipped\""), "{result}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_head_defaults_to_the_active_branch() {
        let root = repo("head", &[("src/matrix/a.rs", "x")]);
        let out = run(&root, &["--base", "main"], &[]);
        assert!(out.status.success(), "{out:?}");
        assert!(ran(&root, "ran-matrix"));
        assert!(!ran(&root, "ran-docs"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn runner_keys_set_the_range_and_the_flags_override_them() {
        let root = repo("keys", &[("documentation/a.md", "x")]);
        let config = CONFIG.replace(
            "[runner]\n",
            "[runner]\nbase = \"main\"\nhead = \"topic\"\n",
        );
        fs::write(root.join(".cargo/gate.toml"), config).unwrap();
        let out = run(&root, &[], &[]);
        assert!(out.status.success(), "{out:?}");
        assert!(ran(&root, "ran-docs") && !ran(&root, "ran-matrix"));
        let _ = fs::remove_file(root.join("ran-docs"));
        let out = run(&root, &["--head", "main"], &[]);
        assert!(out.status.success(), "{out:?}");
        assert!(!ran(&root, "ran-docs"), "an empty range skips docs");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unresolvable_ref_skips_no_gate() {
        let root = repo("badref", &[("documentation/a.md", "x")]);
        let out = run(&root, &["--base", "no-such-ref"], &[]);
        assert!(out.status.success(), "{out:?}");
        assert!(ran(&root, "ran-docs") && ran(&root, "ran-matrix"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn environment_overrides_replace_the_path_check() {
        let root = repo("env", &[("documentation/a.md", "x")]);
        let out = run(
            &root,
            &["--base", "main"],
            &[("CONTROL_RS_CI_GATES", "run")],
        );
        assert!(out.status.success(), "{out:?}");
        assert!(ran(&root, "ran-matrix"));
        let _ = fs::remove_file(root.join("ran-docs"));
        let out = run(
            &root,
            &["--base", "main"],
            &[
                ("CONTROL_RS_CI_GATES", "run"),
                ("CONTROL_RS_CI_GATE_DOCS", "skip"),
            ],
        );
        assert!(out.status.success(), "{out:?}");
        assert!(!ran(&root, "ran-docs"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_invalid_override_is_a_usage_error() {
        let root = repo("badenv", &[("documentation/a.md", "x")]);
        let out = run(
            &root,
            &["--base", "main"],
            &[("CONTROL_RS_CI_GATES", "maybe")],
        );
        assert_eq!(out.status.code(), Some(2), "{out:?}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_runner_list_extends_only_gates_that_declare_their_own() {
        let root = repo("global", &[("shared/x.txt", "x")]);
        let config = CONFIG
            .replace("[runner]\n", "[runner]\ndiff-trigger = [\"shared/\"]\n");
        fs::write(root.join(".cargo/gate.toml"), config).unwrap();
        let out = run(&root, &["--base", "main"], &[]);
        assert!(out.status.success(), "{out:?}");
        assert!(ran(&root, "ran-docs") && ran(&root, "ran-matrix"));
        assert!(ran(&root, "ran-always"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_runner_list_alone_selects_nothing() {
        let root = repo("globalonly", &[("shared/x.txt", "x")]);
        let out = run(&root, &["--base", "main"], &[]);
        assert!(out.status.success(), "{out:?}");
        assert!(!ran(&root, "ran-docs") && !ran(&root, "ran-matrix"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_runner_list_is_matched_once_and_extends_each_declared_list() {
        let changed = strings(&["shared/x.txt"]);
        let hit = global_hit(&strings(&["shared/"]), &changed);
        assert!(hit);
        assert!(!global_hit(&strings(&["other/"]), &changed));
        let entries = strings(&["src/"]);
        let with_global = Triggers {
            own: &entries,
            global_hit: hit,
        };
        let run =
            should_run("matrix", with_global, Some(&changed), &no_env).unwrap();
        assert_eq!(run, Decision::Run);
        let none = strings(&[]);
        let bare = Triggers {
            own: &none,
            global_hit: false,
        };
        let skip = should_run("matrix", bare, Some(&changed), &no_env).unwrap();
        assert_eq!(skip, Decision::Run, "no own list: always runs");
    }
}
