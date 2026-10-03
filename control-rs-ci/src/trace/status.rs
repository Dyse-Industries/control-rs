//! Requirement and condition status derivation.
//!
//! Each condition takes a status:
//! - `Pass`: method in `automated_methods`, every target recorded as passed.
//! - `Fail`: method in `automated_methods`, any target recorded as failed (fails gate).
//! - `Unrun`: method in `automated_methods`, any target absent from its result log (fails gate).
//! - `Uncovered`: method in `automated_methods` and no target named (fails gate).
//! - `Review`: method outside `automated_methods`; awaits sign-off.
//! - `Deferred`: method `deferred`; verification is planned for a later phase.
//!
//! A requirement takes the worst status among its conditions in order:
//! `Fail` > `Unrun` > `Uncovered` > `Deferred` > `Review` > `Pass`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::reqs::TraceConfig;
use super::{
    CONDITION, DEFINITION, Defect, Defects, Row, SCHEMA, read_text, write_file,
};
use crate::error::GateResult;

/// The method whose conditions are planned but not yet verifiable.
const DEFERRED: &str = "deferred";

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
type MethodResultMap = HashMap<String, ResultLog>;

/// Evaluated conditions, pending reviews and deferred conditions.
type ConditionEvalResult = (
    Vec<ConditionStatus>,
    Vec<ReviewCondition>,
    Vec<ReviewCondition>,
);

/// What a result log says about one target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lookup {
    Passed,
    Failed,
    Ambiguous,
    Absent,
}

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
    /// Method `deferred`; verification is planned for a later phase.
    Deferred,
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

/// A review condition awaiting sign-off, or a deferred condition.
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
    /// Warning code (`W-2` or `W-3`).
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

/// A parsed result log.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResultLog {
    /// Outcome per identifier.
    outcomes: TestResultMap,
    /// Identifiers the log records more than once with different outcomes, or
    /// that name two doctests.
    ambiguous: BTreeSet<String>,
    /// Test binaries the log names in `Running` and `Doc-tests` headers.
    binaries: BTreeSet<String>,
}

/// Context for condition status derivation across a trace run.
struct StatusCtx<'a> {
    config: &'a TraceConfig,
    result_cache: &'a MethodResultMap,
    interpreter_results: Option<&'a ResultLog>,
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
    /// Conditions whose verification is deferred.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deferred: Vec<ReviewCondition>,
    /// Non-fatal interpreter warnings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<InterpreterWarning>,
}

impl From<TestResultMap> for ResultLog {
    fn from(outcomes: TestResultMap) -> Self {
        Self {
            outcomes,
            ..Self::default()
        }
    }
}

impl ResultLog {
    /// Records `passed` for `key`. A repeat is ambiguous when `unique`
    /// forbids repeats or the outcomes differ, and then takes the worse
    /// outcome.
    fn record(&mut self, key: String, passed: bool, unique: bool) {
        match self.outcomes.get(&key).copied() {
            Some(prev) if unique || prev != passed => {
                self.ambiguous.insert(key.clone());
                self.outcomes.insert(key, prev && passed);
            }
            _ => {
                self.outcomes.insert(key, passed);
            }
        }
    }
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

/// The test binary a libtest `Running` header names: the file stem of the
/// parenthesized path without its `-<hash>` suffix.
fn binary_name(header: &str) -> String {
    let path = header
        .rsplit_once('(')
        .and_then(|(_, path)| path.strip_suffix(')'))
        .unwrap_or(header);
    let stem = Path::new(path.trim())
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    match stem.rsplit_once('-') {
        Some((name, hash))
            if !hash.is_empty()
                && hash.bytes().all(|b| b.is_ascii_hexdigit()) =>
        {
            name.to_string()
        }
        _ => stem,
    }
}

/// `name` without the ` (line N)` that libtest appends to a doctest.
fn without_doctest_line(name: &str) -> String {
    let Some(start) = name.find(" (line ") else {
        return name.to_string();
    };
    let tail = name.get(start..).unwrap_or_default();
    tail.find(')').map_or_else(
        || name.to_string(),
        |end| {
            format!(
                "{}{}",
                name.get(..start).unwrap_or_default(),
                tail.get(end.saturating_add(1)..).unwrap_or_default()
            )
        },
    )
}

/// Parses standard libtest line-oriented output (`test <path> ... ok|FAILED`).
///
/// A result is keyed by `<binary>::<path>`, the binary taken from the
/// preceding `Running` or `Doc-tests` header. Lines before any header are
/// keyed by the bare path. A doctest is keyed by its file and item without
/// the line number.
#[must_use]
pub fn parse_libtest_results(text: &str) -> ResultLog {
    let mut log = ResultLog::default();
    let mut binary: Option<String> = None;
    let mut doctests = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(header) = trimmed.strip_prefix("Running ") {
            let name = binary_name(header);
            log.binaries.insert(name.clone());
            binary = Some(name);
            doctests = false;
            continue;
        }
        if let Some(krate) = trimmed.strip_prefix("Doc-tests ") {
            let name = krate.trim().replace('-', "_");
            log.binaries.insert(name.clone());
            binary = Some(name);
            doctests = true;
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("test ") else {
            continue;
        };
        let Some((name, outcome_part)) = rest.split_once(" ... ") else {
            continue;
        };
        let passed = if outcome_part.starts_with("ok") {
            true
        } else if outcome_part.starts_with("FAILED") {
            false
        } else {
            continue;
        };
        let name = if doctests {
            without_doctest_line(name.trim())
        } else {
            name.trim().to_string()
        };
        let key = binary
            .as_deref()
            .map_or_else(|| name.clone(), |b| format!("{b}::{name}"));
        log.record(key, passed, doctests);
    }
    log
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

/// What `log` records for `target`: an exact identifier, or the identifier
/// without its leading crate segment, which only a log without binary
/// headers records.
fn lookup(log: &ResultLog, target: &str) -> Lookup {
    let key = if log.outcomes.contains_key(target) {
        target
    } else {
        match target.split_once("::") {
            Some((_, rest)) if log.outcomes.contains_key(rest) => rest,
            _ => return Lookup::Absent,
        }
    };
    if log.ambiguous.contains(key) {
        Lookup::Ambiguous
    } else if log.outcomes.get(key).copied().unwrap_or(false) {
        Lookup::Passed
    } else {
        Lookup::Failed
    }
}

/// The file name of `path`.
fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned())
}

/// The file name of the result artifact of `method`.
fn artifact_name(config: &TraceConfig, method: &str) -> String {
    config
        .method
        .get(method)
        .and_then(|m| m.result_artifact.as_deref())
        .map_or_else(|| method.to_string(), file_name)
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

/// Warns (W-2) when the interpreter log covers the crate of a passing
/// `libtest` target but does not record the target. Doctests are not run by
/// the interpreter.
fn check_interpreter_coverage(
    row: &Row,
    method: &str,
    target: &str,
    ctx: &mut StatusCtx<'_>,
) {
    if method != "libtest" || target.contains(" - ") {
        return;
    }
    let Some(log) = ctx.interpreter_results else {
        return;
    };
    let covered = target
        .split_once("::")
        .is_some_and(|(krate, _)| log.binaries.contains(krate));
    if covered && lookup(log, target) == Lookup::Absent {
        let artifact = ctx
            .config
            .method
            .get(method)
            .and_then(|m| m.interpreter_artifact.as_deref())
            .map_or_else(String::new, file_name);
        ctx.warnings.push(InterpreterWarning {
            code: "W-2".to_string(),
            file: row.file.clone(),
            line: row.line,
            item: target.to_string(),
            message: format!(
                "target was executed under test.log but is absent from {artifact}"
            ),
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
        Lookup::Passed => {
            check_interpreter_coverage(row, method, target, ctx);
            outcome(true, None)
        }
        Lookup::Ambiguous => {
            ctx.defects.push(Defect::at(
                row,
                format!(
                    "condition {} target '{target}' is ambiguous in {artifact}: it is recorded more than once with different outcomes or names two doctests",
                    row.id
                ),
            ));
            outcome(false, Some("ambiguous result in result log".to_string()))
        }
        Lookup::Failed => {
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
        Lookup::Absent => {
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
    if method == DEFERRED {
        return status(Status::Deferred, Vec::new());
    }
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

fn parse_artifact(method: &str, text: &str) -> Option<ResultLog> {
    match method {
        "libtest" => Some(parse_libtest_results(text)),
        "kani" => Some(parse_kani_results(text).into()),
        "pytest" => Some(parse_pytest_results(text).into()),
        "gtest" => Some(parse_gtest_results(text).into()),
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

/// Warns (W-3) for each result log of an automated method that is older than
/// the most recently modified document `reqs` names: a document edited after
/// the run may name targets the log never ran.
fn check_staleness(
    reqs: &[Row],
    config: &TraceConfig,
    base: &Path,
    warnings: &mut Vec<InterpreterWarning>,
) {
    let modified = |rel: &str| {
        std::fs::metadata(base.join(rel))
            .and_then(|m| m.modified())
            .ok()
    };
    let files: BTreeSet<&str> = reqs.iter().map(|r| r.file.as_str()).collect();
    let newest = files
        .into_iter()
        .filter_map(|file| modified(file).map(|time| (time, file)))
        .max();
    let Some((doc_time, doc)) = newest else {
        return;
    };
    for method in &config.automated_methods {
        let Some(artifact) = config
            .method
            .get(method)
            .and_then(|m| m.result_artifact.as_deref())
        else {
            continue;
        };
        if modified(artifact).is_some_and(|log_time| log_time < doc_time) {
            warnings.push(InterpreterWarning {
                code: "W-3".to_string(),
                file: doc.to_string(),
                line: 1,
                item: artifact.to_string(),
                message: format!(
                    "{method} result log predates the last edit to this document and may be stale; rerun the gate that writes it"
                ),
            });
        }
    }
}

/// The interpreter log (`interpreter_artifact`) of the `libtest` method.
fn load_interpreter_log(
    config: &TraceConfig,
    base: &Path,
) -> Option<ResultLog> {
    let path = config
        .method
        .get("libtest")
        .and_then(|m| m.interpreter_artifact.as_deref())?;
    read_text(&base.join(path))
        .ok()
        .map(|t| parse_libtest_results(&t))
}

fn initial_status_counts() -> StatusCounts {
    [
        Status::Pass,
        Status::Review,
        Status::Deferred,
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
    let mut deferred = Vec::new();

    for row in condition_order.iter().filter_map(|id| conditions.get(id)) {
        let status = condition_status(row, ctx);

        if let Some(count) = counts.get_mut(&status.status) {
            *count = count.saturating_add(1);
        }
        let pending = match status.status {
            Status::Review => Some(&mut review),
            Status::Deferred => Some(&mut deferred),
            _ => None,
        };
        if let Some(pending) = pending {
            pending.push(ReviewCondition {
                id: row.id.clone(),
                parents: row.parents.clone(),
                method: status.method.clone(),
                file: row.file.clone(),
                line: row.line,
            });
        }
        evaluated.push(status);
    }

    (evaluated, review, deferred)
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
    let interpreter_results = load_interpreter_log(config, base);
    let mut counts = initial_status_counts();

    let mut ctx = StatusCtx {
        config,
        result_cache: &result_cache,
        interpreter_results: interpreter_results.as_ref(),
        defects: &mut defects,
        warnings: &mut warnings,
    };

    let (evaluated, review, deferred) =
        evaluate_all_conditions(&cond_idx, &mut ctx, &mut counts);

    let requirements = requirement_statuses(&order, &evaluated);
    check_staleness(reqs, config, base, &mut warnings);

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
        deferred,
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
            deferred: Vec::new(),
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
        assert_eq!(py.outcomes.get("tests/a.py::t"), Some(&true));
        let gt = parse_artifact("gtest", "[       OK ] Suite.Case (1 ms)\n")
            .unwrap();
        assert_eq!(gt.outcomes.get("Suite.Case"), Some(&true));
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
                        interpreter_artifact: (m == "libtest").then(|| {
                            "target/ci-artifacts/miri.log".to_string()
                        }),
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
            methods: [
                "libtest", "kani", "pytest", "gtest", "review", "deferred",
            ]
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
        write_log(
            &dir,
            "miri.log",
            "Running unittests src/lib.rs (target/debug/deps/a-0f1e)\nrunning 0 tests\n",
        );
        write_log(
            &dir,
            "libtest.log",
            "Running unittests src/lib.rs (target/debug/deps/a-9c8d)\ntest t ... ok\n",
        );
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

    /// Statuses of `targets` of `libtest` conditions against `log`.
    fn libtest_statuses(
        name: &str,
        log: &str,
        targets: &[&str],
    ) -> Vec<Status> {
        targets
            .iter()
            .map(|t| {
                status(&run(name, log, condition("libtest", &[t]))).unwrap()
            })
            .collect()
    }

    #[test]
    fn libtest_results_are_keyed_by_binary() {
        let log = "Running unittests src/lib.rs (target/debug/deps/a-1f2e3d)\n\
                   test tests::t ... FAILED\n\
                   Running tests/b.rs (target/debug/deps/b-9a8b7c)\n\
                   test tests::t ... ok\n";
        assert_eq!(
            libtest_statuses(
                "by_binary",
                log,
                &["a::tests::t", "b::tests::t", "c::tests::t", "tests::t"]
            ),
            [Status::Fail, Status::Pass, Status::Unrun, Status::Unrun]
        );
    }

    #[test]
    fn a_path_recorded_with_different_outcomes_is_ambiguous() {
        let log = "Running unittests src/lib.rs (target/debug/deps/a-1f2e3d)\n\
                   test t ... ok\ntest t ... FAILED\ntest u ... ok\ntest u ... ok\n";
        let t = run("ambiguous", log, condition("libtest", &["a::t"]));
        assert_eq!(status(&t), Some(Status::Fail));
        assert!(
            t.1.first()
                .is_some_and(|d| d.message.contains("is ambiguous in")),
            "{:?}",
            t.1
        );
        let u = run("repeat_ok", log, condition("libtest", &["a::u"]));
        assert_eq!(status(&u), Some(Status::Pass));
    }

    #[test]
    fn doctests_match_on_file_and_item_without_the_line() {
        let log = "Doc-tests control-rs\n\
                   test src/a.rs - a::Foo (line 10) ... ok\n\
                   test src/a.rs - a::Bar (line 20) - compile fail ... ok\n\
                   test src/a.rs - a::Dup (line 30) ... ok\n\
                   test src/a.rs - a::Dup (line 40) ... ok\n";
        assert_eq!(
            libtest_statuses(
                "doctests",
                log,
                &[
                    "control_rs::src/a.rs - a::Foo",
                    "control_rs::src/a.rs - a::Bar - compile fail",
                    "control_rs::src/a.rs - a::Dup",
                    "control_rs::src/a.rs - a::Foo (line 10)",
                ]
            ),
            [Status::Pass, Status::Pass, Status::Fail, Status::Unrun]
        );
    }

    #[test]
    fn binary_names_drop_the_extension_and_hash() {
        for (header, name) in [
            (
                "unittests src/lib.rs (target/debug/deps/control_rs-1a2b3c)",
                "control_rs",
            ),
            (
                "tests/x.rs (target/debug/deps/trace_tests-0f.exe)",
                "trace_tests",
            ),
            ("target/debug/deps/plain", "plain"),
            ("unittests (target/debug/deps/no-hash-here)", "no-hash-here"),
        ] {
            assert_eq!(binary_name(header), name, "{header}");
        }
        assert_eq!(without_doctest_line("f - i (line 3"), "f - i (line 3");
    }

    #[test]
    fn interpreter_warnings_cover_only_the_crates_the_log_runs() {
        let dir = test_dir("miri_scope");
        write_log(
            &dir,
            "libtest.log",
            "Running unittests (target/d/a-1)\ntest t ... ok\ntest u ... ok\n\
             Running unittests (target/d/b-2)\ntest t ... ok\n\
             Doc-tests a\ntest src/a.rs - i (line 1) ... ok\n",
        );
        write_log(
            &dir,
            "miri.log",
            "Running unittests (target/m/a-3)\ntest t ... ok\n",
        );
        let (report, defects) = derive_with_base(
            &[
                definition(),
                condition(
                    "libtest",
                    &["a::t", "a::u", "b::t", "a::src/a.rs - i"],
                ),
            ],
            &default_config(),
            &dir,
        );
        assert!(defects.is_empty(), "{defects:?}");
        let items: Vec<_> =
            report.warnings.iter().map(|w| w.item.as_str()).collect();
        assert_eq!(items, ["a::u"]);
        assert!(
            report
                .warnings
                .iter()
                .all(|w| w.message.ends_with("miri.log")),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn no_interpreter_artifact_means_no_w2() {
        let dir = test_dir("no_interpreter");
        write_log(
            &dir,
            "libtest.log",
            "Running unittests (target/d/a-1)\ntest t ... ok\n",
        );
        write_log(&dir, "miri.log", "Running unittests (target/m/a-3)\n");
        let mut config = default_config();
        if let Some(m) = config.method.get_mut("libtest") {
            m.interpreter_artifact = None;
        }
        let (report, _) = derive_with_base(
            &[definition(), condition("libtest", &["a::t"])],
            &config,
            &dir,
        );
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    }

    #[test]
    fn deferred_condition_is_listed_and_does_not_fail() {
        let derived = derive(
            &[definition(), condition("deferred", &[])],
            &default_config(),
        );
        assert_eq!(status(&derived), Some(Status::Deferred));
        assert!(derived.0.passes());
        assert!(derived.1.is_empty(), "{:?}", derived.1);
        assert_eq!(derived.0.deferred.len(), 1);
        assert!(derived.0.review.is_empty());
        assert_eq!(derived.0.counts.get(&Status::Deferred), Some(&1));
        assert_eq!(derived.0.counts.get(&Status::Review), Some(&0));
    }

    /// Sets the modification time of `path` to `secs` after the epoch.
    fn touch(path: &Path, secs: u64) {
        let time = std::time::UNIX_EPOCH
            .checked_add(std::time::Duration::from_secs(secs))
            .unwrap();
        fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(time)
            .unwrap();
    }

    #[test]
    fn result_log_older_than_a_traced_document_warns_stale() {
        let dir = test_dir("stale");
        fs::create_dir_all(dir.join("docs")).unwrap();
        let doc = dir.join("docs/w-design.md");
        let log = dir.join("target/ci-artifacts/libtest.log");
        fs::write(&doc, "# W (w)\n").unwrap();
        fs::write(&log, "test a::t ... ok\n").unwrap();
        let rows = [definition(), condition("libtest", &["a::t"])];

        touch(&log, 1_000);
        touch(&doc, 2_000);
        let (report, _) = derive_with_base(&rows, &default_config(), &dir);
        let stale: Vec<_> =
            report.warnings.iter().filter(|w| w.code == "W-3").collect();
        assert_eq!(stale.len(), 1, "{:?}", report.warnings);
        assert_eq!(
            stale.first().map(|w| (w.file.as_str(), w.item.as_str())),
            Some(("docs/w-design.md", "target/ci-artifacts/libtest.log"))
        );

        touch(&log, 3_000);
        let (fresh, _) = derive_with_base(&rows, &default_config(), &dir);
        assert!(fresh.warnings.iter().all(|w| w.code != "W-3"));

        touch(&log, 2_000);
        let (equal, _) = derive_with_base(&rows, &default_config(), &dir);
        assert!(equal.warnings.iter().all(|w| w.code != "W-3"));
    }
}
