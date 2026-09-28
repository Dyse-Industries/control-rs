//! Markdown rendering of validation reports.

#[cfg(test)]
mod report_render {
    use control_rs_compare::report::{
        ComparisonFinding, MethodFinding, SuiteReport, ValidationReport,
        ValidationSummary,
    };

    fn method(
        kind: &str,
        verdict: &str,
        details: Option<&str>,
    ) -> MethodFinding {
        MethodFinding {
            r#type: kind.to_string(),
            bound: 0.5,
            observed: 2.0,
            verdict: verdict.to_string(),
            details: details.map(str::to_string),
        }
    }

    fn comparison(
        signal: &str,
        verdict: &str,
        methods: Vec<MethodFinding>,
    ) -> ComparisonFinding {
        ComparisonFinding {
            key: format!("s.{signal}.rust"),
            pair: ("scipy".to_string(), "rust".to_string()),
            signal: signal.to_string(),
            policy: "all_of".to_string(),
            verdict: verdict.to_string(),
            methods,
        }
    }

    fn suite(
        name: &str,
        status: &str,
        reason: Option<&str>,
        comparisons: Vec<ComparisonFinding>,
    ) -> SuiteReport {
        SuiteReport {
            name: name.to_string(),
            status: status.to_string(),
            duration_secs: 0.1,
            comparisons,
            failure_reason: reason.map(str::to_string),
        }
    }

    fn report(suites: Vec<SuiteReport>) -> ValidationReport {
        let failed = suites.iter().filter(|s| s.status == "Fail").count();
        ValidationReport {
            summary: ValidationSummary {
                total_suites: suites.len(),
                passed_suites: suites.len().saturating_sub(failed),
                failed_suites: failed,
                total_duration_secs: 0.2,
                verdict: if failed == 0 { "Pass" } else { "Fail" }.to_string(),
            },
            suites,
        }
    }

    #[test]
    fn a_passing_report_has_no_diagnostics() {
        let md = report(vec![suite(
            "good",
            "Pass",
            None,
            vec![comparison("x", "pass", vec![method("abs", "pass", None)])],
        )])
        .render_markdown();
        assert!(md.contains("| `good` | Pass |"), "{md}");
        assert!(!md.contains("Discrepancy & Diagnostic Details"), "{md}");
    }

    #[test]
    fn diagnostics_list_only_failed_suites_signals_and_methods() {
        let md = report(vec![
            suite(
                "good",
                "Pass",
                None,
                vec![comparison(
                    "fine",
                    "pass",
                    vec![method("abs", "pass", None)],
                )],
            ),
            suite(
                "bad",
                "Fail",
                Some("oracle missing"),
                vec![
                    comparison(
                        "broken",
                        "fail",
                        vec![
                            method("abs", "fail", Some("too far")),
                            method("rel", "fail", None),
                            method("rms", "pass", None),
                        ],
                    ),
                    comparison("ok", "pass", vec![method("abs", "pass", None)]),
                ],
            ),
        ])
        .render_markdown();
        assert!(md.contains("### Discrepancy & Diagnostic Details"), "{md}");
        assert!(md.contains("#### Suite: `bad`"), "{md}");
        assert!(!md.contains("#### Suite: `good`"), "{md}");
        assert!(md.contains("- **Error**: oracle missing"), "{md}");
        assert!(md.contains("**Signal `broken`**"), "{md}");
        assert!(!md.contains("**Signal `ok`**"), "{md}");
        assert!(
            md.contains(
                "Method `abs`: observed=2.000e0, bound=5.000e-1 (too far)"
            ),
            "{md}"
        );
        assert!(md.contains("Method `rel`: observed=2.000e0, bound=5.000e-1 (exceeded bound)"), "{md}");
        assert!(!md.contains("Method `rms`"), "{md}");
    }

    #[test]
    fn a_failed_suite_without_a_reason_omits_the_error_line() {
        let md =
            report(vec![suite("bad", "Fail", None, vec![])]).render_markdown();
        assert!(md.contains("#### Suite: `bad`"), "{md}");
        assert!(!md.contains("**Error**"), "{md}");
    }
}
