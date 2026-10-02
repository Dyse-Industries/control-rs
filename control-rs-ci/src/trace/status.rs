//! Requirement and condition status derivation.
//!
//! Each condition takes a status:
//! - `Pass`: method in `automated_methods`, every target recorded as passed.
//! - `Fail`: method in `automated_methods`, any target recorded as failed (fails gate).
//! - `Unrun`: method in `automated_methods`, any target absent from its result log (fails gate).
//! - `Uncovered`: method in `automated_methods` and no target named (fails gate).
//! - `Review`: method outside `automated_methods`; awaits sign-off.
//!
//! A requirement takes the worst status among its conditions in order:
//! `Fail` > `Unrun` > `Uncovered` > `Review` > `Pass`.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::reqs::TraceConfig;
use super::{
    CONDITION, DEFINITION, Defect, Defects, Row, SCHEMA, read_text, write_file,
};
use crate::error::GateResult;

/// A report and its defects.
pub type Derived = (TraceReport, Defects);

/// Count of conditions per status.
pub type StatusCounts = BTreeMap<Status, usize>;

/// Rows by qualified ID, first occurrence kept.
type ById<'a> = BTreeMap<&'a str, &'a Row>;

/// Rows of one kind by ID, and the IDs in order of first occurrence.
type Indexed<'a> = (ById<'a>, Vec<&'a str>);

/// Satisfied and total cover properties count.
type CoverSummary = (usize, usize);

/// Map of test/harness identifier to pass/fail.
type TestResultMap = HashMap<String, bool>;

/// Map from method name to its parsed result log.
type MethodResultMap = HashMap<String, TestResultMap>;

/// Evaluated conditions and pending reviews.
type ConditionEvalResult = (Vec<ConditionStatus>, Vec<ReviewCondition>);

/// Status of one requirement or condition.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub enum Status {
    /// All targets passed verification.
    Pass,
    /// Method outside `automated_methods`; awaits sign-off.
    Review,
    /// Automated method and no target named; fails the gate.
    Uncovered,
    /// Automated method and a target has no recorded result; fails the gate.
    Unrun,
    /// Automated method and a target failed verification; fails the gate.
    Fail,
}

/// Verification outcome of one target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItemOutcome {
    /// Test or harness identifier named in the condition row.
    pub target: String,
    /// Whether the target passed verification.
    pub passed: bool,
    /// Optional failure or diagnostic message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// One condition's status in `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionStatus {
    /// Qualified Condition ID.
    pub id: String,
    /// Parent requirement IDs.
    pub parents: Vec<String>,
    /// Verification method.
    pub method: String,
    /// Derived status.
    pub status: Status,
    /// Per-target verification outcomes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<ItemOutcome>,
}

/// One requirement's entry in `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementStatus {
    /// Qualified ID.
    pub id: String,
    /// Derived status.
    pub status: Status,
    /// Child verification conditions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<ConditionStatus>,
}

/// A review condition awaiting sign-off.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewCondition {
    /// Qualified condition ID.
    pub id: String,
    /// Parent requirement IDs.
    pub parents: Vec<String>,
    /// Verification method.
    pub method: String,
    /// Path of the defining document.
    pub file: String,
    /// 1-based line number of the condition row.
    pub line: usize,
}

/// Interpreter warning emitted during test analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpreterWarning {
    /// Warning code (`W-2`).
    pub code: String,
    /// Path of the defining document.
    pub file: String,
    /// 1-based line number of the condition row.
    pub line: usize,
    /// Target identifier.
    pub item: String,
    /// Warning description.
    pub message: String,
}

/// Context for condition status derivation across a trace run.
struct StatusCtx<'a> {
    config: &'a TraceConfig,
    result_cache: &'a MethodResultMap,
    miri_results: Option<&'a TestResultMap>,
    defects: &'a mut Defects,
    warnings: &'a mut Vec<InterpreterWarning>,
}

/// `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceReport {
    /// Report format version.
    pub schema: u32,
    /// Number of conditions per status.
    pub counts: StatusCounts,
    /// One entry per requirement, in the order of their first definition.
    pub requirements: Vec<RequirementStatus>,
    /// Review conditions awaiting sign-off.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub review: Vec<ReviewCondition>,
    /// Non-fatal interpreter warnings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<InterpreterWarning>,
}

impl fmt::Display for InterpreterWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}: [{}] {} ({})",
            self.file, self.line, self.code, self.message, self.item
        )
    }
}

impl TraceReport {
    /// Whether no condition/requirement is `Uncovered`, `Unrun`, or `Fail`.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.counts.get(&Status::Fail).copied().unwrap_or(0) == 0
            && self.counts.get(&Status::Unrun).copied().unwrap_or(0) == 0
            && self.counts.get(&Status::Uncovered).copied().unwrap_or(0) == 0
            && self.requirements.iter().all(|r| {
                r.status != Status::Fail
                    && r.status != Status::Unrun
                    && r.status != Status::Uncovered
            })
    }

    /// Writes the report as pretty-printed JSON, creating the parent directory.
    ///
    /// # Errors
    /// `GateError::Io` or `GateError::Json` on failure.
    pub fn write(&self, path: &Path) -> GateResult<()> {
        let mut json = serde_json::to_vec_pretty(self)?;
        json.push(b'\n');
        write_file(path, &json)
    }
}

/// Parses standard libtest line-oriented output (`test <path> ... ok|FAILED`).
#[must_use]
pub fn parse_libtest_results(text: &str) -> TestResultMap {
    let mut results = HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("test ") else {
            continue;
        };
        let Some((name, outcome_part)) = rest.split_once(" ... ") else {
            continue;
        };
        let test_name = name.trim().to_string();
        if outcome_part.starts_with("ok") {
            results.insert(test_name, true);
        } else if outcome_part.starts_with("FAILED") {
            results.insert(test_name, false);
        }
    }
    results
}

/// Parses Kani bounded model checking output.
#[must_use]
pub fn parse_kani_results(text: &str) -> TestResultMap {
    let mut results = HashMap::new();
    let mut current_harness: Option<String> = None;
    let mut harness_lines: Vec<&str> = Vec::new();

    let evaluate_harness = |lines: &[&str]| -> bool {
        let mut has_success = false;
        let mut has_failure = false;
        for line in lines {
            let trimmed = line.trim();
            if trimmed.contains("VERIFICATION:- SUCCESSFUL") {
                has_success = true;
            }
            if trimmed.contains("VERIFICATION:- FAILED")
                || trimmed.contains("- Status: FAILURE")
                || trimmed.contains("- Status: UNSATISFIABLE")
                || parse_cover_summary(trimmed).is_some_and(|(m, n)| m < n)
            {
                has_failure = true;
            }
        }
        has_success && !has_failure
    };

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Checking harness ") {
            if let Some(prev) = current_harness.take() {
                let passed = evaluate_harness(&harness_lines);
                results.insert(prev, passed);
                harness_lines.clear();
            }
            let harness_name = rest.trim_end_matches('.').trim().to_string();
            current_harness = Some(harness_name);
        } else if current_harness.is_some() {
            harness_lines.push(line);
        }
    }
    if let Some(prev) = current_harness {
        let passed = evaluate_harness(&harness_lines);
        results.insert(prev, passed);
    }
    results
}

fn parse_cover_summary(line: &str) -> Option<CoverSummary> {
    let rest = line.strip_prefix("** ")?;
    let idx = rest.find(" cover properties satisfied")?;
    let sub = rest.get(..idx)?;
    let (m_str, n_str) = sub.split_once(" of ")?;
    let m = m_str.trim().parse::<usize>().ok()?;
    let n = n_str.trim().parse::<usize>().ok()?;
    Some((m, n))
}

/// Parses verbose `pytest` output (`<file>::<test> PASSED|FAILED|ERROR`).
#[must_use]
pub fn parse_pytest_results(text: &str) -> TestResultMap {
    let mut results = HashMap::new();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        let (Some(name), Some(outcome)) = (words.next(), words.next()) else {
            continue;
        };
        if !name.contains("::") {
            continue;
        }
        match outcome {
            "PASSED" => results.insert(name.to_string(), true),
            "FAILED" | "ERROR" => results.insert(name.to_string(), false),
            _ => None,
        };
    }
    results
}

/// Parses Google Test output (`[       OK ] Suite.Case`, `[  FAILED  ] Suite.Case`).
#[must_use]
pub fn parse_gtest_results(text: &str) -> TestResultMap {
    let mut results = HashMap::new();
    for line in text.lines() {
        let Some((tag, rest)) =
            line.strip_prefix('[').and_then(|rest| rest.split_once(']'))
        else {
            continue;
        };
        let Some(name) = rest.split_whitespace().next() else {
            continue;
        };
        if !name.contains('.') {
            continue;
        }
        match tag.trim() {
            "OK" => results.insert(name.to_string(), true),
            "FAILED" => results.insert(name.to_string(), false),
            _ => None,
        };
    }
    results
}

/// The outcome recorded for `target`: an exact identifier, or the identifier
/// without its leading crate segment.
fn lookup(results: &TestResultMap, target: &str) -> Option<bool> {
    results.get(target).copied().or_else(|| {
        let (_, rest) = target.split_once("::")?;
        results.get(rest).copied()
    })
}

/// The file name of the result artifact of `method`.
fn artifact_name(config: &TraceConfig, method: &str) -> String {
    config
        .method
        .get(method)
        .and_then(|m| m.result_artifact.as_deref())
        .map_or_else(
            || method.to_string(),
            |path| {
                Path::new(path).file_name().map_or_else(
                    || path.to_string(),
                    |n| n.to_string_lossy().into_owned(),
                )
            },
        )
}

/// The first row of each ID among `rows` of `kind`, and the IDs in order of
/// first occurrence.
fn first_by_id<'a>(rows: &'a [Row], kind: &str) -> Indexed<'a> {
    let mut by_id = ById::new();
    let mut order = Vec::new();
    for row in rows.iter().filter(|r| r.kind == kind) {
        if !by_id.contains_key(row.id.as_str()) {
            order.push(row.id.as_str());
            by_id.insert(row.id.as_str(), row);
        }
    }
    (by_id, order)
}

fn check_miri_log_coverage(
    row: &Row,
    method: &str,
    target: &str,
    ctx: &mut StatusCtx<'_>,
) {
    if method != "libtest" {
        return;
    }
    let Some(miri_map) = ctx.miri_results else {
        return;
    };
    if lookup(miri_map, target).is_none() {
        ctx.warnings.push(InterpreterWarning {
            code: "W-2".to_string(),
            file: row.file.clone(),
            line: row.line,
            item: target.to_string(),
            message:
                "target was executed under test.log but is absent from miri.log"
                    .to_string(),
        });
    }
}

fn evaluate_target(
    row: &Row,
    method: &str,
    target: &str,
    ctx: &mut StatusCtx<'_>,
) -> ItemOutcome {
    let artifact = artifact_name(ctx.config, method);
    let outcome = |passed: bool, message: Option<String>| ItemOutcome {
        target: target.to_string(),
        passed,
        message,
    };
    let Some(results) = ctx.result_cache.get(method) else {
        ctx.defects.push(Defect::at(
            row,
            format!(
                "condition {} target '{target}' not found: {artifact} is missing",
                row.id
            ),
        ));
        return outcome(false, Some("result log missing".to_string()));
    };
    match lookup(results, target) {
        Some(true) => {
            check_miri_log_coverage(row, method, target, ctx);
            outcome(true, None)
        }
        Some(false) => {
            ctx.defects.push(Defect::at(
                row,
                format!(
                    "condition {} target '{target}' failed in {artifact}",
                    row.id
                ),
            ));
            outcome(
                false,
                Some("verification failed in result log".to_string()),
            )
        }
        None => {
            ctx.defects.push(Defect::at(
                row,
                format!(
                    "condition {} target '{target}' not found in {artifact}",
                    row.id
                ),
            ));
            outcome(false, Some("no result recorded in result log".to_string()))
        }
    }
}

fn condition_status(row: &Row, ctx: &mut StatusCtx<'_>) -> ConditionStatus {
    let method = row.method.clone().unwrap_or_default();
    let status = |status: Status, items: Vec<ItemOutcome>| ConditionStatus {
        id: row.id.clone(),
        parents: row.parents.clone(),
        method: method.clone(),
        status,
        items,
    };
    if !ctx.config.automated_methods.contains(&method) {
        return status(Status::Review, Vec::new());
    }
    if row.targets.is_empty() {
        ctx.defects
            .push(Defect::at(row, format!("{} names no target", row.id)));
        return status(Status::Uncovered, Vec::new());
    }

    let items: Vec<ItemOutcome> = row
        .targets
        .iter()
        .map(|target| evaluate_target(row, &method, target, ctx))
        .collect();
    let derived = if items.iter().any(|i| {
        i.message
            .as_deref()
            .is_some_and(|m| m.contains("no result") || m.contains("missing"))
    }) {
        Status::Unrun
    } else if items.iter().any(|i| !i.passed) {
        Status::Fail
    } else {
        Status::Pass
    };
    status(derived, items)
}

/// The worst status of `conditions`; `Uncovered` when there are none.
fn worst(conditions: &[ConditionStatus]) -> Status {
    conditions
        .iter()
        .map(|c| c.status)
        .max()
        .unwrap_or(Status::Uncovered)
}

/// One entry per requirement in `order`, each with its conditions sorted by
/// ID and the worst of their statuses.
fn requirement_statuses(
    order: &[&str],
    evaluated: &[ConditionStatus],
) -> Vec<RequirementStatus> {
    order
        .iter()
        .map(|&id| {
            let mut children: Vec<ConditionStatus> = evaluated
                .iter()
                .filter(|c| c.parents.iter().any(|p| p == id))
                .cloned()
                .collect();
            children.sort_by(|a, b| a.id.cmp(&b.id));
            RequirementStatus {
                id: id.to_string(),
                status: worst(&children),
                conditions: children,
            }
        })
        .collect()
}

fn parse_artifact(method: &str, text: &str) -> Option<TestResultMap> {
    match method {
        "libtest" => Some(parse_libtest_results(text)),
        "kani" => Some(parse_kani_results(text)),
        "pytest" => Some(parse_pytest_results(text)),
        "gtest" => Some(parse_gtest_results(text)),
        _ => None,
    }
}

fn load_method_artifacts(config: &TraceConfig, base: &Path) -> MethodResultMap {
    let mut result_cache = HashMap::new();
    for method in &config.automated_methods {
        let Some(artifact_rel) = config
            .method
            .get(method)
            .and_then(|m| m.result_artifact.as_ref())
        else {
            continue;
        };
        let Ok(text) = read_text(&base.join(artifact_rel)) else {
            continue;
        };
        if let Some(results) = parse_artifact(method, &text) {
            result_cache.insert(method.clone(), results);
        }
    }
    result_cache
}

fn load_miri_log(base: &Path) -> Option<TestResultMap> {
    let miri_path = base.join("target/ci-artifacts/miri.log");
    if miri_path.exists() {
        read_text(&miri_path)
            .ok()
            .map(|t| parse_libtest_results(&t))
    } else {
        None
    }
}

fn initial_status_counts() -> StatusCounts {
    [
        Status::Pass,
        Status::Review,
        Status::Uncovered,
        Status::Unrun,
        Status::Fail,
    ]
    .into_iter()
    .map(|status| (status, 0))
    .collect()
}

fn evaluate_all_conditions(
    cond_idx: &Indexed<'_>,
    ctx: &mut StatusCtx<'_>,
    counts: &mut StatusCounts,
) -> ConditionEvalResult {
    let (conditions, condition_order) = cond_idx;
    let mut evaluated = Vec::new();
    let mut review = Vec::new();

    for row in condition_order.iter().filter_map(|id| conditions.get(id)) {
        let status = condition_status(row, ctx);

        if let Some(count) = counts.get_mut(&status.status) {
            *count = count.saturating_add(1);
        }
        if status.status == Status::Review {
            review.push(ReviewCondition {
                id: row.id.clone(),
                parents: row.parents.clone(),
                method: status.method.clone(),
                file: row.file.clone(),
                line: row.line,
            });
        }
        evaluated.push(status);
    }

    (evaluated, review)
}

/// Derives condition and requirement status from the rows of `trace-reqs`,
/// using the current working directory as base.
#[must_use]
pub fn derive(reqs: &[Row], config: &TraceConfig) -> Derived {
    derive_with_base(reqs, config, Path::new("."))
}

/// Derives condition and requirement status relative to `base`.
#[must_use]
pub fn derive_with_base(
    reqs: &[Row],
    config: &TraceConfig,
    base: &Path,
) -> Derived {
    let (_, order) = first_by_id(reqs, DEFINITION);
    let cond_idx = first_by_id(reqs, CONDITION);

    let mut defects = Vec::new();
    let mut warnings = Vec::new();
    let result_cache = load_method_artifacts(config, base);
    let miri_results = load_miri_log(base);
    let mut counts = initial_status_counts();

    let mut ctx = StatusCtx {
        config,
        result_cache: &result_cache,
        miri_results: miri_results.as_ref(),
        defects: &mut defects,
        warnings: &mut warnings,
    };

    let (evaluated, review) =
        evaluate_all_conditions(&cond_idx, &mut ctx, &mut counts);

    let requirements = requirement_statuses(&order, &evaluated);

    defects.sort_by(|a, b| {
        (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line))
    });
    warnings.sort_by(|a, b| {
        (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line))
    });

    let report = TraceReport {
        schema: SCHEMA,
        counts,
        requirements,
        review,
        warnings,
    };

    (report, defects)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use crate::trace::reqs::MethodConfig;

    /// A condition status and how many conditions have it.
    type Count = (Status, usize);

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "control_rs_ci_status_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("target/ci-artifacts")).unwrap();
        dir
    }

    fn report(counts: &[Count], requirements: &[Status]) -> TraceReport {
        TraceReport {
            schema: 4,
            counts: counts.iter().copied().collect(),
            requirements: requirements
                .iter()
                .enumerate()
                .map(|(n, status)| RequirementStatus {
                    id: format!("widget#FR-{n}"),
                    status: *status,
                    conditions: Vec::new(),
                })
                .collect(),
            review: Vec::new(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn a_report_passes_only_without_fail_unrun_or_uncovered() {
        assert!(
            report(
                &[(Status::Pass, 3), (Status::Review, 1)],
                &[Status::Pass, Status::Review]
            )
            .passes()
        );
        for bad in [Status::Fail, Status::Unrun, Status::Uncovered] {
            assert!(!report(&[(bad, 1)], &[Status::Pass]).passes(), "{bad:?}");
            assert!(!report(&[], &[bad]).passes(), "{bad:?} requirement");
        }
    }

    #[test]
    fn any_kani_failure_marker_fails_a_harness_that_also_succeeded() {
        for marker in [
            "VERIFICATION:- FAILED",
            "- Status: FAILURE",
            "- Status: UNSATISFIABLE",
            "** 1 of 2 cover properties satisfied",
        ] {
            let log = format!(
                "Checking harness h::bad...\n{marker}\nVERIFICATION:- SUCCESSFUL\n"
            );
            assert_eq!(
                parse_kani_results(&log).get("h::bad"),
                Some(&false),
                "{marker}"
            );
        }
        let ok = "Checking harness h::ok...\n** 2 of 2 cover properties satisfied\nVERIFICATION:- SUCCESSFUL\n";
        assert_eq!(parse_kani_results(ok).get("h::ok"), Some(&true));
    }

    #[test]
    fn each_result_format_has_a_parser() {
        let py = parse_artifact("pytest", "tests/a.py::t PASSED\n").unwrap();
        assert_eq!(py.get("tests/a.py::t"), Some(&true));
        let gt = parse_artifact("gtest", "[       OK ] Suite.Case (1 ms)\n")
            .unwrap();
        assert_eq!(gt.get("Suite.Case"), Some(&true));
        assert!(parse_artifact("unknown", "").is_none());
    }

    fn write_log(dir: &Path, name: &str, text: &str) {
        fs::write(dir.join("target/ci-artifacts").join(name), text).unwrap();
    }

    fn default_config() -> TraceConfig {
        let method = ["libtest", "kani", "pytest", "gtest"]
            .into_iter()
            .map(|m| {
                (
                    m.to_string(),
                    MethodConfig {
                        result_artifact: Some(format!(
                            "target/ci-artifacts/{m}.log"
                        )),
                    },
                )
            })
            .collect();
        TraceConfig {
            id: r"(?:FR|NFR|C)-[0-9]+[a-z]?".to_string(),
            condition: r"VC-[0-9]+(?:\.[0-9]+[a-z]?)?".to_string(),
            doc: "[a-z0-9-]+".to_string(),
            files: vec!["docs".to_string()],
            doc_id: r"^#\s+.*\((?P<doc>[a-z0-9-]+)\)".to_string(),
            definition: r"^- \*\*(?:FR|NFR|C)-".to_string(),
            verification: r"^\| *(?:[a-z0-9-]+#)?VC-".to_string(),
            methods: ["libtest", "kani", "pytest", "gtest", "review"]
                .map(String::from)
                .to_vec(),
            automated_methods: ["libtest", "kani", "pytest", "gtest"]
                .map(String::from)
                .to_vec(),
            method,
            retired: vec![],
            decisions: None,
        }
    }

    fn definition() -> Row {
        Row {
            schema: SCHEMA,
            id: "w#FR-1".to_string(),
            kind: DEFINITION.to_string(),
            parents: Vec::new(),
            method: None,
            targets: Vec::new(),
            file: "docs/w-design.md".to_string(),
            line: 1,
            text: "- **FR-1 — A**: Work.".to_string(),
        }
    }

    fn condition(method: &str, targets: &[&str]) -> Row {
        Row {
            schema: SCHEMA,
            id: "w#VC-1.1".to_string(),
            kind: CONDITION.to_string(),
            parents: vec!["w#FR-1".to_string()],
            method: Some(method.to_string()),
            targets: targets.iter().map(ToString::to_string).collect(),
            file: "docs/w-design.md".to_string(),
            line: 3,
            text: String::new(),
        }
    }

    /// Derives the status of one condition against `log` written as `name`.
    fn run(name: &str, log: &str, cond: Row) -> Derived {
        let dir = test_dir(name);
        write_log(&dir, &format!("{}.log", cond.method.clone().unwrap()), log);
        derive_with_base(&[definition(), cond], &default_config(), &dir)
    }

    fn status(derived: &Derived) -> Option<Status> {
        derived.0.requirements.first().map(|r| r.status)
    }

    #[test]
    fn libtest_target_passes_with_crate_qualified_or_bare_name() {
        let log = "test widget::tests::t ... ok\n";
        for target in ["widget::tests::t", "widget_crate::widget::tests::t"] {
            let derived =
                run("libtest_ok", log, condition("libtest", &[target]));
            assert_eq!(status(&derived), Some(Status::Pass), "{target}");
            assert!(derived.0.passes());
            assert!(derived.1.is_empty(), "{:?}", derived.1);
        }
    }

    #[test]
    fn libtest_target_fails_when_the_log_reports_failed() {
        let derived = run(
            "libtest_fail",
            "test widget::tests::t ... FAILED\n",
            condition("libtest", &["widget::tests::t"]),
        );
        assert_eq!(status(&derived), Some(Status::Fail));
        assert!(!derived.0.passes());
        assert_eq!(
            derived.1.first().map(|d| d.message.as_str()),
            Some(
                "condition w#VC-1.1 target 'widget::tests::t' failed in libtest.log"
            )
        );
    }

    #[test]
    fn target_absent_from_the_log_is_unrun() {
        let derived = run(
            "libtest_unrun",
            "test other ... ok\n",
            condition("libtest", &["widget::tests::t"]),
        );
        assert_eq!(status(&derived), Some(Status::Unrun));
        assert_eq!(
            derived.1.first().map(|d| d.message.as_str()),
            Some(
                "condition w#VC-1.1 target 'widget::tests::t' not found in libtest.log"
            )
        );
    }

    #[test]
    fn every_listed_target_must_pass() {
        let log = "test a::t ... ok\ntest a::u ... FAILED\n";
        let derived = run("many", log, condition("libtest", &["a::t", "a::u"]));
        assert_eq!(status(&derived), Some(Status::Fail));
        let outcomes: Vec<_> = derived
            .0
            .requirements
            .iter()
            .flat_map(|r| &r.conditions)
            .flat_map(|c| &c.items)
            .map(|i| i.passed)
            .collect();
        assert_eq!(outcomes, [true, false]);
    }

    #[test]
    fn missing_log_is_unrun() {
        let dir = test_dir("no_log");
        let derived = derive_with_base(
            &[definition(), condition("libtest", &["a::t"])],
            &default_config(),
            &dir,
        );
        assert_eq!(status(&derived), Some(Status::Unrun));
    }

    #[test]
    fn kani_target_passes_and_unsatisfied_cover_fails() {
        let ok = "Checking harness c::proofs::h...\n - Status: SUCCESS\n** 1 of 1 cover properties satisfied\nVERIFICATION:- SUCCESSFUL\n";
        let bad = "Checking harness c::proofs::h...\n - Status: UNSATISFIABLE\n** 0 of 1 cover properties satisfied\nVERIFICATION:- SUCCESSFUL\n";
        let cond = || condition("kani", &["c::proofs::h"]);
        assert_eq!(status(&run("kani_ok", ok, cond())), Some(Status::Pass));
        assert_eq!(status(&run("kani_bad", bad, cond())), Some(Status::Fail));
    }

    #[test]
    fn pytest_and_gtest_logs_are_parsed() {
        let py = "python3/m.py::test_step PASSED [ 50%]\npython3/m.py::test_bad FAILED [100%]\n";
        let results = parse_pytest_results(py);
        assert_eq!(results.get("python3/m.py::test_step"), Some(&true));
        assert_eq!(results.get("python3/m.py::test_bad"), Some(&false));

        let gt = "[ RUN      ] Suite.Case\n[       OK ] Suite.Case (0 ms)\n[  FAILED  ] Suite.Bad (1 ms)\n[==========] 2 tests\n";
        let results = parse_gtest_results(gt);
        assert_eq!(results.get("Suite.Case"), Some(&true));
        assert_eq!(results.get("Suite.Bad"), Some(&false));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn automated_condition_without_targets_is_uncovered() {
        let derived = run("uncovered", "", condition("libtest", &[]));
        assert_eq!(status(&derived), Some(Status::Uncovered));
        assert!(!derived.0.passes());
        assert_eq!(
            derived.1.first().map(|d| d.message.as_str()),
            Some("w#VC-1.1 names no target")
        );
        assert_eq!(derived.0.counts.get(&Status::Uncovered), Some(&1));
    }

    #[test]
    fn review_condition_takes_review_status() {
        let derived = derive(
            &[definition(), condition("review", &[])],
            &default_config(),
        );
        assert_eq!(status(&derived), Some(Status::Review));
        assert!(derived.0.passes());
        assert!(derived.1.is_empty(), "{:?}", derived.1);
        assert_eq!(derived.0.review.len(), 1);
    }

    #[test]
    fn target_missing_from_miri_log_warns() {
        let dir = test_dir("miri_warn");
        write_log(&dir, "libtest.log", "test a::t ... ok\n");
        write_log(&dir, "miri.log", "running 0 tests\n");
        let (report, defects) = derive_with_base(
            &[definition(), condition("libtest", &["a::t"])],
            &default_config(),
            &dir,
        );
        assert!(defects.is_empty(), "{defects:?}");
        assert_eq!(
            report.warnings.first().map(|w| w.code.as_str()),
            Some("W-2")
        );
    }
}
