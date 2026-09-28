//! Report size budget, result discovery and log-tail boundary tests.

#[cfg(test)]
mod report_edge {
    use std::fmt::Write as _;
    use std::fs;
    use std::path::PathBuf;

    use control_rs_ci::config::GateConfig;
    use control_rs_ci::gate::{GateOutcome, OUTCOME_SCHEMA, Verdict};
    use control_rs_ci::report::{MAX_REPORT_BYTES, ReportAggregator};

    /// Bytes the report keeps free for its truncation notice.
    const NOTICE_RESERVE: usize = 256;

    fn scratch(name: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "control_rs_ci_report_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let artifacts = root.join("artifacts");
        fs::create_dir_all(&artifacts).unwrap();
        (root, artifacts)
    }

    fn outcome(gate: &str, verdict: Verdict, summary: &str) -> GateOutcome {
        GateOutcome {
            schema: OUTCOME_SCHEMA,
            gate: gate.to_string(),
            verdict,
            exit_code: Some(1),
            duration_secs: 0.5,
            summary: Some(summary.to_string()),
            log_file: format!("{gate}.log"),
        }
    }

    fn config(gates: &[&str]) -> GateConfig {
        let mut text = String::new();
        for gate in gates {
            let _ = writeln!(text, "[{gate}]\ncommand = \"false\"");
        }
        GateConfig::parse(&text, std::path::Path::new("gate.toml")).unwrap()
    }

    #[test]
    fn report_budget_is_sixty_four_kibibytes() {
        assert_eq!(MAX_REPORT_BYTES, 65_536);
    }

    #[test]
    fn result_discovery_ignores_files_and_directories_that_are_not_results() {
        let (root, artifacts) = scratch("discovery");
        outcome("fmt", Verdict::Pass, "clean")
            .save_to_dir(&artifacts)
            .unwrap();
        fs::write(artifacts.join("notes.txt"), "not a result").unwrap();
        fs::create_dir_all(artifacts.join("nested.result.json")).unwrap();

        let aggregator = ReportAggregator::new(artifacts, root.clone());
        let loaded = aggregator.load_outcomes().unwrap();
        assert_eq!(loaded.outcomes.len(), 1);
        assert!(loaded.rejected.is_empty(), "{:?}", loaded.rejected);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_oversized_summary_matrix_is_truncated_with_a_notice() {
        let (root, artifacts) = scratch("oversized");
        outcome("big", Verdict::Fail, &"s".repeat(100_000))
            .save_to_dir(&artifacts)
            .unwrap();
        fs::write(artifacts.join("big.log"), "boom\n").unwrap();
        let aggregator = ReportAggregator::new(artifacts, root.clone());
        let report = aggregator.write_report(&config(&["big"]), None).unwrap();
        let content = fs::read_to_string(report.path).unwrap();
        assert!(content.len() <= MAX_REPORT_BYTES, "{}", content.len());
        assert!(
            content.ends_with("64 KiB size budget.*\n"),
            "no truncation notice"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// Whether gate `b`'s log tail is embedded when gate `a`'s log is `a_len`
    /// bytes long, together with the report length.
    fn b_embedded(a_len: usize) -> (bool, usize) {
        let (root, artifacts) = scratch("boundary");
        outcome("a", Verdict::Fail, "failed")
            .save_to_dir(&artifacts)
            .unwrap();
        outcome("b", Verdict::Fail, "failed")
            .save_to_dir(&artifacts)
            .unwrap();
        fs::write(artifacts.join("a.log"), format!("{}\n", "x".repeat(a_len)))
            .unwrap();
        fs::write(artifacts.join("b.log"), format!("{}\n", "y".repeat(99)))
            .unwrap();
        let aggregator = ReportAggregator::new(artifacts, root.clone());
        let report =
            aggregator.write_report(&config(&["a", "b"]), None).unwrap();
        let content = fs::read_to_string(report.path).unwrap();
        let _ = fs::remove_dir_all(&root);
        (content.contains(&"y".repeat(99)), content.len())
    }

    #[test]
    fn a_log_tail_that_exactly_fills_the_remaining_budget_is_embedded() {
        // Gate `b` fits while `a` is short enough and is omitted beyond that.
        let (mut lo, mut hi) = (0_usize, MAX_REPORT_BYTES);
        assert!(b_embedded(lo).0, "b must fit beside an empty log");
        assert!(!b_embedded(hi).0, "b must not fit beside a huge log");
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if b_embedded(mid).0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // At the last size where `b` fits, its tail used the budget exactly.
        let (embedded, len) = b_embedded(lo);
        assert!(embedded);
        assert_eq!(len, MAX_REPORT_BYTES - NOTICE_RESERVE);
    }

    #[test]
    fn diagnostics_cover_each_flagged_gate_once_including_unconfigured_ones() {
        let (root, artifacts) = scratch("ordered");
        for gate in ["a", "zzz"] {
            outcome(gate, Verdict::Fail, "failed")
                .save_to_dir(&artifacts)
                .unwrap();
            fs::write(artifacts.join(format!("{gate}.log")), "log line\n")
                .unwrap();
        }
        let aggregator = ReportAggregator::new(artifacts, root.clone());
        let report = aggregator.write_report(&config(&["a"]), None).unwrap();
        let content = fs::read_to_string(report.path).unwrap();
        assert_eq!(content.matches("Gate: a (Fail)").count(), 1);
        assert_eq!(content.matches("Gate: zzz (Fail)").count(), 1);
        let _ = fs::remove_dir_all(&root);
    }
}
