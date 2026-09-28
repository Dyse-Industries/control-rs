//! Numerical kernels, tolerance attributes and container ingestion edge cases.

#[cfg(test)]
mod kernel_gaps {
    use std::fs;
    use std::path::{Path, PathBuf};

    use control_rs_compare::compare::{
        ComparatorOptions, PartialChunkStats, ToleranceSpec,
        compare_float_arrays_parallel, compute_chunk_stats,
        reduce_and_evaluate, resolve_signal_tolerances, run_comparison,
    };
    use control_rs_compare::config::CompareConfigFile;
    use control_rs_compare::report::ValidationReport;
    use hdf5_pure::{AttrValue, File, FileBuilder};

    /// One attribute name and value.
    type AttrPair<'a> = (&'a str, AttrValue);

    /// Attributes set on a dataset.
    type AttrList<'a> = Vec<(&'a str, AttrValue)>;

    /// Dataset names with their data.
    type Datasets<'a> = &'a [(&'a str, &'a [f64])];

    /// Expected bound per attribute value.
    type BoundCases = Vec<(AttrValue, f64)>;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "control_rs_compare_kernel_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn spec(method: &str, bound: f64) -> ToleranceSpec {
        ToleranceSpec {
            method: method.to_string(),
            bound,
        }
    }

    fn partial(sum_sq: f64, count: f64) -> PartialChunkStats {
        PartialChunkStats {
            sum_sq,
            count,
            ..PartialChunkStats::default()
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * b.abs().max(1.0)
    }

    // --- chunk statistics and reduction ------------------------------------

    #[test]
    fn chunk_stats_count_elements_and_relative_error_at_zero() {
        let stats = compute_chunk_stats(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
        assert!(close(stats.count, 3.0));
        assert!(close(stats.sum_sq, 14.0));
        assert!(close(stats.max_abs, 3.0));
        // The epsilon keeps the relative error finite and positive at zero.
        assert!(stats.max_rel > 1e15, "{}", stats.max_rel);
    }

    #[test]
    fn rms_pools_squares_and_counts_across_chunks() {
        let finding = reduce_and_evaluate(
            &[partial(4.0, 2.0), partial(4.0, 2.0)],
            &spec("rms", 2.0),
        );
        assert_eq!(finding.r#type, "rms");
        assert!(
            close(finding.observed, 2.0_f64.sqrt()),
            "{}",
            finding.observed
        );
        assert_eq!(finding.verdict, "pass");
    }

    #[test]
    fn rms_of_no_elements_is_zero() {
        let finding = reduce_and_evaluate(
            &[PartialChunkStats::default()],
            &spec("rms", 1.0),
        );
        assert!(close(finding.observed, 0.0));
        assert_eq!(finding.verdict, "pass");
    }

    #[test]
    fn norm_aliases_report_the_frobenius_norm() {
        for method in ["matrix_norm", "norm"] {
            let finding =
                reduce_and_evaluate(&[partial(25.0, 4.0)], &spec(method, 10.0));
            assert_eq!(finding.r#type, "matrix_norm", "{method}");
            assert!(close(finding.observed, 5.0), "{method}");
        }
    }

    #[test]
    fn details_appear_only_for_failed_bounds() {
        let stats = compute_chunk_stats(&[0.0], &[2.0]);
        let pass = reduce_and_evaluate(&[stats], &spec("abs", 3.0));
        assert_eq!(pass.verdict, "pass");
        assert!(pass.details.is_none());
        let fail = reduce_and_evaluate(&[stats], &spec("abs", 1.0));
        assert_eq!(fail.verdict, "fail");
        assert!(
            fail.details
                .as_deref()
                .unwrap()
                .contains("Max absolute difference")
        );
    }

    #[test]
    fn empty_arrays_never_take_the_chunked_path() {
        let finding =
            compare_float_arrays_parallel(&[], &[], &spec("abs", 1.0), 4);
        assert_eq!(finding.verdict, "pass");
    }

    #[test]
    fn large_arrays_are_reduced_chunk_by_chunk() {
        let n = 2 * control_rs_compare::compare::CHUNK_THRESHOLD;
        let oracle = vec![0.0_f64; n];
        let mut peer = vec![1.0_f64; n];
        if let Some(first) = peer.first_mut() {
            *first = 1e8;
        }
        // One sequential pass absorbs every unit square into 1e16; two chunks
        // keep the second chunk's total, so the norms differ.
        let sequential = compare_float_arrays_parallel(
            &oracle,
            &peer,
            &spec("matrix_norm", 1e9),
            1,
        );
        let chunked = compare_float_arrays_parallel(
            &oracle,
            &peer,
            &spec("matrix_norm", 1e9),
            2,
        );
        assert!(close(sequential.observed, 1e8));
        assert!(chunked.observed > 1e8 + 1e-4, "{}", chunked.observed);
    }

    // --- tolerance attributes of every stored type -------------------------

    fn policy_for(attrs: AttrList<'_>) -> (String, f64) {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("sig");
        ds.with_f64_data(&[1.0, 2.0]);
        for (key, value) in attrs {
            ds.set_attr(key, value);
        }
        let file = File::from_bytes(b.finish().unwrap()).unwrap();
        let ds = file.dataset("sig").unwrap();
        let policy = resolve_signal_tolerances(Some(&ds), "sig", "peer", None);
        let first = policy.methods.first().unwrap();
        (first.method.clone(), first.bound)
    }

    #[test]
    fn bound_attributes_convert_from_every_numeric_and_text_type() {
        let cases: BoundCases = vec![
            (AttrValue::F64Array(vec![0.25, 9.0]), 0.25),
            (AttrValue::F32Array(vec![0.5, 9.0]), 0.5),
            (AttrValue::U32(3), 3.0),
            (AttrValue::U64(4), 4.0),
            (AttrValue::String("0.125".to_string()), 0.125),
        ];
        for (value, want) in cases {
            let label = format!("{value:?}");
            let (method, bound) = policy_for(vec![("bound", value)]);
            assert_eq!(method, "abs", "{label}");
            assert!(close(bound, want), "{label}: {bound}");
        }
    }

    #[test]
    fn method_attributes_convert_from_string_arrays() {
        let (method, bound) = policy_for(vec![(
            "method",
            AttrValue::StringArray(vec!["rel".to_string()]),
        )]);
        assert_eq!(method, "rel");
        assert!(close(bound, 1e-4));
    }

    // --- container ingestion and suite comparison --------------------------

    fn container(datasets: Datasets<'_>, attrs: &[AttrPair<'_>]) -> Vec<u8> {
        let mut b = FileBuilder::new();
        for (name, data) in datasets {
            let ds = b.create_dataset(name);
            ds.with_f64_data(data);
            for (key, value) in attrs {
                ds.set_attr(key, value.clone());
            }
        }
        b.finish().unwrap()
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) {
        fs::write(dir.join(name), bytes).unwrap();
    }

    fn compare(dir: &Path) -> ValidationReport {
        let options = ComparatorOptions {
            results_dir: dir.to_path_buf(),
            quiet: true,
            ..ComparatorOptions::default()
        };
        run_comparison(None, &options).unwrap()
    }

    #[test]
    fn only_result_containers_are_ingested() {
        let dir = scratch("ingest");
        let bytes = container(&[("x", &[1.0, 2.0])], &[]);
        write(&dir, "s.scipy.h5", &bytes);
        write(&dir, "s.rust.hdf5", &bytes);
        write(&dir, "notes.txt", b"not a container");
        fs::create_dir_all(dir.join("junk.h5")).unwrap();
        let report = compare(&dir);
        assert_eq!(report.summary.total_suites, 1);
        let suite = report.suites.first().unwrap();
        assert_eq!(suite.name, "s");
        assert_eq!(suite.comparisons.len(), 1);
        assert_eq!(report.summary.verdict, "Pass");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_container_without_signals_passes_with_no_comparisons() {
        let dir = scratch("empty");
        let bytes = FileBuilder::new().finish().unwrap();
        write(&dir, "s.scipy.h5", &bytes);
        write(&dir, "s.rust.h5", &bytes);
        let report = compare(&dir);
        let suite = report.suites.first().unwrap();
        assert!(suite.comparisons.is_empty());
        assert_eq!(suite.status, "Pass");
        let _ = fs::remove_dir_all(&dir);
    }

    fn policy_verdict(policy: Option<&str>, peer: f64) -> String {
        let dir = scratch("policy");
        let methods =
            r#"[{"type":"abs","bound":0.1},{"type":"abs","bound":1.0}]"#;
        let mut attrs =
            vec![("methods", AttrValue::String(methods.to_string()))];
        if let Some(policy) = policy {
            attrs.push(("policy", AttrValue::String(policy.to_string())));
        }
        write(&dir, "s.scipy.h5", &container(&[("x", &[1.0])], &attrs));
        write(&dir, "s.rust.h5", &container(&[("x", &[peer])], &[]));
        let report = compare(&dir);
        let _ = fs::remove_dir_all(&dir);
        report
            .suites
            .first()
            .unwrap()
            .comparisons
            .first()
            .unwrap()
            .verdict
            .clone()
    }

    #[test]
    fn any_of_needs_one_passing_method_and_all_of_needs_every_one() {
        // Peer 1.5 passes the loose bound only; peer 1.05 passes both.
        assert_eq!(policy_verdict(Some("any_of"), 1.5), "pass");
        assert_eq!(policy_verdict(Some("all_of"), 1.5), "fail");
        assert_eq!(policy_verdict(None, 1.5), "fail");
        assert_eq!(policy_verdict(Some("any_of"), 1.05), "pass");
        assert_eq!(policy_verdict(Some("all_of"), 1.05), "pass");
        assert_eq!(policy_verdict(Some("any_of"), 5.0), "fail");
    }

    #[test]
    fn each_suite_uses_its_own_configured_oracle() {
        let dir = scratch("oracle");
        let bytes = container(&[("x", &[1.0])], &[]);
        write(&dir, "b.ref.h5", &bytes);
        write(&dir, "b.peer.h5", &bytes);
        let config: CompareConfigFile = toml::from_str(
            "[[suite]]\nname = \"a\"\ntrue_oracle = \"other\"\n\
             [[suite]]\nname = \"b\"\ntrue_oracle = \"ref\"\n",
        )
        .unwrap();
        let plan = config
            .resolve_master_plan(Path::new("compare.toml"))
            .unwrap();
        let options = ComparatorOptions {
            results_dir: dir.clone(),
            quiet: true,
            ..ComparatorOptions::default()
        };
        let report = run_comparison(Some(&plan), &options).unwrap();
        let suite = report.suites.first().unwrap();
        assert_eq!(suite.name, "b");
        assert_eq!(suite.status, "Pass", "{:?}", suite.failure_reason);
        let _ = fs::remove_dir_all(&dir);
    }

    fn string_container(values: &[&str]) -> Vec<u8> {
        let mut b = FileBuilder::new();
        b.create_dataset("labels").with_strings(values).unwrap();
        b.finish().unwrap()
    }

    #[test]
    fn string_datasets_match_exactly() {
        let dir = scratch("strings");
        write(&dir, "s.scipy.h5", &string_container(&["a", "b"]));
        write(&dir, "s.same.h5", &string_container(&["a", "b"]));
        write(&dir, "s.other.h5", &string_container(&["a", "c"]));
        let report = compare(&dir);
        let suite = report.suites.first().unwrap();
        for comparison in &suite.comparisons {
            let method = comparison.methods.first().unwrap();
            assert_eq!(method.r#type, "exact_match");
            if comparison.pair.1 == "same" {
                assert_eq!(comparison.verdict, "pass");
                assert!(close(method.observed, 1.0));
                assert!(method.details.is_none());
            } else {
                assert_eq!(comparison.verdict, "fail");
                assert!(close(method.observed, 0.0));
                assert_eq!(
                    method.details.as_deref(),
                    Some("String dataset mismatch")
                );
            }
        }
        assert_eq!(suite.comparisons.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }
}
