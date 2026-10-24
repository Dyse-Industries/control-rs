//! Automated CI log result ingestion and reproduction tests (FR-25).

#[cfg(test)]
mod reproduction {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use control_rs_ci::gate::{Gate, GateContext, LOG_TRAILER_PREFIX, Verdict};

    fn context(name: &str) -> (PathBuf, GateContext) {
        let root = std::env::temp_dir()
            .join(format!("control_rs_ci_repro_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("artifacts")).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("docs")).unwrap();
        let ctx = GateContext {
            workspace_root: root.clone(),
            out_dir: root.join("artifacts"),
            default_timeout: Duration::from_secs(30),
        };
        (root, ctx)
    }

    fn shell_with_triggers(
        name: &str,
        script: &str,
        triggers: Vec<String>,
    ) -> Gate {
        let mut gate =
            Gate::new(name, "sh", vec!["-c".to_string(), script.to_string()]);
        gate.diff_trigger = triggers;
        gate
    }

    #[test]
    fn log_trailer_is_appended_and_read_correctly() {
        let (root, ctx) = context("trailer_append");
        let gate = shell_with_triggers(
            "echo_test",
            "echo 'hello world'",
            vec!["src/".to_string()],
        );

        let outcome = gate.execute(&ctx).unwrap();
        assert_eq!(outcome.verdict, Verdict::Pass);

        let log_path = ctx.log_path("echo_test");
        assert!(log_path.is_file());
        let fp = gate.current_fingerprint(&ctx.workspace_root);
        gate.append_outcome_trailer(&log_path, &outcome, &fp)
            .unwrap();

        let content = fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("hello world"));
        assert!(content.contains(LOG_TRAILER_PREFIX));

        let envelope = gate.read_outcome_from_log(&log_path).unwrap();
        assert_eq!(envelope.gate, "echo_test");
        assert_eq!(envelope.verdict, Verdict::Pass);
        assert_eq!(envelope.exit_code, Some(0));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn fresh_passing_gate_is_up_to_date_without_process_execution() {
        let (root, ctx) = context("up_to_date_fresh");
        let gate = shell_with_triggers(
            "counter",
            "echo run; date +%s%N",
            vec!["src/".to_string()],
        );

        // First run: executes live
        let outcome1 = gate.execute(&ctx).unwrap();
        assert_eq!(outcome1.verdict, Verdict::Pass);

        let log1 = fs::read_to_string(ctx.log_path("counter")).unwrap();

        // Second run: up-to-date, should not re-run or change log
        let start = Instant::now();
        let cached = gate.try_read_up_to_date_outcome(&ctx);
        let elapsed = start.elapsed();

        assert!(cached.is_some(), "fresh gate must be up-to-date");
        let outcome2 = cached.unwrap();
        assert_eq!(outcome2.verdict, Verdict::Pass);
        assert_eq!(outcome2.duration_secs, outcome1.duration_secs);
        assert!(
            elapsed < Duration::from_millis(50),
            "up-to-date check should take < 50ms"
        );

        let log2 = fs::read_to_string(ctx.log_path("counter")).unwrap();
        assert_eq!(
            log1, log2,
            "log must not be modified during up-to-date check"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn modifying_trigger_file_invalidates_up_to_date_status() {
        let (root, ctx) = context("content_invalidate");
        let trigger_file = root.join("src/lib.rs");
        fs::write(&trigger_file, "initial").unwrap();

        let gate = shell_with_triggers(
            "builder",
            "echo building",
            vec!["src/".to_string()],
        );
        let outcome = gate.execute(&ctx).unwrap();
        assert_eq!(outcome.verdict, Verdict::Pass);

        // Before modification: up-to-date
        assert!(gate.try_read_up_to_date_outcome(&ctx).is_some());

        // Modify trigger file content: changes content hash
        fs::write(&trigger_file, "modified content").unwrap();

        // After modification: must NOT be up-to-date
        let invalidated = gate.try_read_up_to_date_outcome(&ctx);
        assert!(
            invalidated.is_none(),
            "modifying trigger file must invalidate log"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rewriting_identical_content_preserves_up_to_date_status() {
        let (root, ctx) = context("identical_content");
        let trigger_file = root.join("src/lib.rs");
        fs::write(&trigger_file, "stable content").unwrap();

        let gate = shell_with_triggers(
            "builder",
            "echo building",
            vec!["src/".to_string()],
        );
        let outcome = gate.execute(&ctx).unwrap();
        assert_eq!(outcome.verdict, Verdict::Pass);

        // Re-write identical content (simulating touch or git checkout)
        fs::write(&trigger_file, "stable content").unwrap();

        // Must still be up-to-date because content is identical
        let cached = gate.try_read_up_to_date_outcome(&ctx);
        assert!(
            cached.is_some(),
            "rewriting identical content must preserve up-to-date status"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn modifying_unrelated_file_preserves_up_to_date_status() {
        let (root, ctx) = context("unrelated_file");
        let src_file = root.join("src/lib.rs");
        let doc_file = root.join("docs/readme.md");
        fs::write(&src_file, "code").unwrap();
        fs::write(&doc_file, "docs").unwrap();

        let gate = shell_with_triggers(
            "compiler",
            "echo compiling",
            vec!["src/".to_string()],
        );
        let outcome = gate.execute(&ctx).unwrap();
        assert_eq!(outcome.verdict, Verdict::Pass);

        // Modify file outside diff-trigger
        fs::write(&doc_file, "updated documentation").unwrap();

        // Must still be up-to-date because src/ did not change
        let cached = gate.try_read_up_to_date_outcome(&ctx);
        assert!(
            cached.is_some(),
            "unrelated file change must not invalidate gate"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_gate_is_never_up_to_date() {
        let (root, ctx) = context("fail_no_cache");
        let gate = shell_with_triggers(
            "failing",
            "echo error >&2; exit 1",
            vec!["src/".to_string()],
        );

        let outcome = gate.execute(&ctx).unwrap();
        assert_eq!(outcome.verdict, Verdict::Fail);

        // A failed run must never be treated as up-to-date
        let cached = gate.try_read_up_to_date_outcome(&ctx);
        assert!(cached.is_none(), "failed gate must always be reproduced");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn config_change_invalidates_up_to_date_status() {
        let (root, ctx) = context("config_invalidate");
        let gate1 = shell_with_triggers(
            "task",
            "echo version1",
            vec!["src/".to_string()],
        );
        let outcome = gate1.execute(&ctx).unwrap();
        assert_eq!(outcome.verdict, Verdict::Pass);

        assert!(gate1.try_read_up_to_date_outcome(&ctx).is_some());

        // Alter gate command arguments (simulating gate.toml edit)
        let gate2 = shell_with_triggers(
            "task",
            "echo version2",
            vec!["src/".to_string()],
        );
        let invalidated = gate2.try_read_up_to_date_outcome(&ctx);
        assert!(
            invalidated.is_none(),
            "configuration change must force reproduction"
        );

        let _ = fs::remove_dir_all(&root);
    }
}
