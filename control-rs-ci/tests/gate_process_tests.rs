//! Gate execution edge cases: outcome summaries and output-pump draining.

#[cfg(test)]
mod gate_process {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use control_rs_ci::gate::{Gate, GateContext, Verdict};

    fn context(name: &str) -> (PathBuf, GateContext) {
        let root = std::env::temp_dir()
            .join(format!("control_rs_ci_gate_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let ctx = GateContext {
            workspace_root: root.clone(),
            out_dir: root.join("artifacts"),
            default_timeout: Duration::from_secs(30),
        };
        (root, ctx)
    }

    fn shell(name: &str, script: &str) -> Gate {
        Gate::new(name, "sh", vec!["-c".to_string(), script.to_string()])
    }

    #[test]
    fn outcome_summaries_distinguish_pass_skip_and_failure() {
        let (root, ctx) = context("summaries");

        let pass = shell("pass", "exit 0").execute(&ctx).unwrap();
        assert_eq!(pass.verdict, Verdict::Pass);
        assert_eq!(pass.summary.as_deref(), Some("pass succeeded cleanly"));

        let mut skipping = shell("skip", "exit 3");
        skipping.skip_exit_codes = vec![3];
        let skipped = skipping.execute(&ctx).unwrap();
        assert_eq!(skipped.verdict, Verdict::Skipped);
        assert_eq!(
            skipped.summary.as_deref(),
            Some("skip skipped with exit code 3")
        );

        let fail = shell("fail", "exit 4").execute(&ctx).unwrap();
        assert_eq!(fail.verdict, Verdict::Fail);
        assert_eq!(
            fail.summary.as_deref(),
            Some("fail failed with exit code 4")
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn echoed_output_is_fully_logged_before_the_gate_returns() {
        let (root, ctx) = context("drain_all");
        let gate = shell("flood", "seq 1 30000");
        let outcome =
            gate.execute_with_echo(&ctx, Some("[t] flood | ")).unwrap();
        assert_eq!(outcome.verdict, Verdict::Pass);
        let log = fs::read_to_string(ctx.out_dir.join("flood.log")).unwrap();
        assert!(log.ends_with("29999\n30000\n"), "log was cut short");
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn finished_pumps_do_not_delay_the_gate() {
        let (root, ctx) = context("drain_fast");
        let start = Instant::now();
        let gate = shell("quick", "echo hi");
        gate.execute_with_echo(&ctx, Some("[t] quick | ")).unwrap();
        assert!(
            start.elapsed() < Duration::from_millis(1500),
            "a finished pump must not wait out the drain timeout: {:?}",
            start.elapsed()
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn a_descendant_holding_the_pipes_delays_the_gate_only_up_to_the_drain_timeout()
     {
        let (root, ctx) = context("drain_bound");
        let start = Instant::now();
        let gate = shell("linger", "sleep 8 & echo done");
        let outcome =
            gate.execute_with_echo(&ctx, Some("[t] linger | ")).unwrap();
        let elapsed = start.elapsed();
        assert_eq!(outcome.verdict, Verdict::Pass);
        assert!(
            elapsed >= Duration::from_millis(1800),
            "drain must wait for open pipes: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(6),
            "drain must give up on open pipes: {elapsed:?}"
        );
        let _ = fs::remove_dir_all(&root);
    }
}
