//! Requirement and condition status derivation.
//!
//! Each condition takes a status:
//! - `Pass`: automated method, every target recorded as passed.
//! - `Fail`: automated method, any target recorded as failed (fails gate).
//! - `Unrun`: automated method, any target absent from its result log (fails gate).
//! - `Uncovered`: no target named (fails gate).
//!
//! `doc-status` and `link` conditions take the status of what they name. A
//! requirement takes the worst status among its conditions in order:
//! `Fail` > `Unrun` > `Uncovered` > `Pass`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::link::{
    Acyclic, GraphFormat, LinkGraph, LinkStatuses, RequirementMap, link_graph,
};
use super::reqs::{DecisionRules, RecordStatus, TraceConfig};
use super::select::select;
use super::{
    CONDITION, DEFINITION, Defect, Defects, QUALIFIER, Row, SCHEMA, read_text,
    write_file,
};
use crate::error::GateResult;
use crate::gate::{GateOutcome, RESULT_SUFFIX, Verdict};
use control_rs_compare::ValidationReport;
use control_rs_compare::report::SuiteReport;

/// The method whose conditions verify that quality gates succeeded.
pub(crate) const GATE: TraceCheckMethod = TraceCheckMethod {
    name: "gate",
    check: check_gate,
};

/// The method whose conditions take the status of the link graph.
pub(crate) const LINK: TraceCheckMethod = TraceCheckMethod {
    name: "link",
    check: check_link,
};

/// Every verification method; no other list of method names exists.
pub(crate) const METHODS: [TraceCheckMethod; 7] = [
    TraceCheckMethod {
        name: "libtest",
        check: check_libtest,
    },
    TraceCheckMethod {
        name: "kani",
        check: check_kani,
    },
    TraceCheckMethod {
        name: "compare",
        check: check_compare,
    },
    TraceCheckMethod {
        name: "junit",
        check: check_junit,
    },
    GATE,
    LINK,
    TraceCheckMethod {
        name: "doc-status",
        check: check_doc_status,
    },
];

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

/// Parsed result log of each method whose artifact was read; `None` when the
/// artifact is missing.
type ResultCache = HashMap<&'static str, Option<ResultLog>>;

/// Result of attempting to load a quality gate outcome artifact.
#[derive(Debug, Clone)]
enum GateLoadOutcome {
    Missing,
    Invalid(String),
    Loaded(GateOutcome),
}

/// Cached outcome of quality gates by target slug.
type GateCache = HashMap<String, GateLoadOutcome>;

/// File of each traced document, by document name.
type DocFiles<'a> = BTreeMap<&'a str, &'a str>;

/// Evaluated conditions by parent requirement ID.
type ByParent<'a> = HashMap<&'a str, Vec<ConditionStatus>>;

/// Evaluated conditions and the requirement statuses of the link graph.
type ConditionEvalResult<'a> = (Vec<ConditionStatus>, RequirementMap<'a>);

/// Outcome and status of one `gate` target.
type GateTargetEval = (ItemOutcome, Status);

/// Outcome and status of one `doc-status` target.
type DocTargetEval = (ItemOutcome, Status);

/// The status of a condition row of one method.
type CheckFn = fn(&Row, &mut StatusCtx<'_>) -> ConditionStatus;

/// Parses the text of a result artifact.
type ParseFn = fn(&str) -> ResultLog;

/// Outcomes and statuses of the targets of one `gate` condition.
type GateTargets = (Vec<ItemOutcome>, Vec<Status>);

/// Outcomes and statuses of the targets of one `doc-status` condition.
type DocTargets = (Vec<ItemOutcome>, Vec<Status>);

/// Stack of `JUnit` test suite names.
type SuiteStack = Vec<Option<String>>;

/// Target context when recording a `JUnit` test case.
#[derive(Debug, Clone, Copy)]
struct JunitCaseCtx<'a> {
    name: &'a str,
    class: Option<&'a str>,
    suite: Option<&'a str>,
}

/// State of an active `<testcase>` element being processed.
#[derive(Debug)]
struct ActiveTestCase {
    name: String,
    classname: Option<String>,
    suite: Option<String>,
    has_failure: bool,
    is_skipped: bool,
}

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
    /// No target named; fails the gate.
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

/// Non-fatal warning emitted during status derivation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpreterWarning {
    /// Warning code (`W-2`, `W-3` or `W-4`).
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

/// The parsed artifact of a method, `None` when it is missing.
#[derive(Clone, Copy)]
struct Artifact<'m> {
    method: &'m ArtifactMethod,
    results: Option<&'m ResultLog>,
}

/// A method whose targets are identifiers in its result artifact.
struct ArtifactMethod {
    /// The method name, which keys `[method.<name>]`.
    name: &'static str,
    /// Parses the artifact.
    parse: ParseFn,
    /// Whether a target may match without its first `::` segment.
    strip: bool,
}

/// The fixed pattern that reads a design document's status badge.
struct DocumentMetaPatterns {
    status: Option<regex::Regex>,
}

/// A verification method a condition row may name.
pub(crate) struct TraceCheckMethod {
    /// The method as a condition row writes it.
    pub name: &'static str,
    /// The status of a condition row of this method.
    pub check: CheckFn,
}

/// Context for condition status derivation across a trace run.
pub(crate) struct StatusCtx<'a> {
    config: &'a TraceConfig,
    base: &'a Path,
    results: ResultCache,
    gates: GateCache,
    interpreter_results: Option<&'a ResultLog>,
    documents: DocFiles<'a>,
    decisions: Option<DecisionRules>,
    patterns: DocumentMetaPatterns,
    defects: &'a mut Defects,
    warnings: &'a mut Vec<InterpreterWarning>,
}

/// Resolved target document or decision state for `doc-status`.
struct TargetDocState {
    status: Option<String>,
    passed: bool,
    draft: bool,
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
    /// Non-fatal interpreter warnings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<InterpreterWarning>,
}

impl ResultLog {
    /// Records `passed` for `key`. A repeat is ambiguous when `unique`
    /// forbids repeats or the outcomes differ.
    fn record(&mut self, key: String, passed: bool, unique: bool) {
        match self.outcomes.get(&key).copied() {
            Some(prev) if unique || prev != passed => {
                self.ambiguous.insert(key);
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

/// Parses Kani bounded model checking output into a [`ResultLog`].
///
/// Each `Checking harness <path>...` block maps to one identifier. The harness
/// passes only when Kani reports `VERIFICATION:- SUCCESSFUL` and every evaluated
/// cover property is satisfied. A harness name recorded more than once with
/// different outcomes is ambiguous and becomes a `Fail` defect at lookup.
///
/// # Arguments
/// * `text` - Kani stdout or a saved `kani.log` artifact.
///
/// # Returns
/// A [`ResultLog`] keyed by harness path. Identical repeats keep the first
/// outcome; conflicting repeats mark the key ambiguous.
///
/// # Panics
/// This function does not panic.
#[must_use]
pub fn parse_kani_results(text: &str) -> ResultLog {
    let mut log = ResultLog::default();
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
                log.record(prev, passed, false);
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
        log.record(prev, passed, false);
    }
    log
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

fn record_compare_suite(log: &mut ResultLog, suite: &SuiteReport) {
    let suite_name = &suite.name;
    if !suite_name.is_empty() && !suite.status.is_empty() {
        let passed = suite.status.eq_ignore_ascii_case("pass")
            || suite.status.eq_ignore_ascii_case("passed")
            || suite.status.eq_ignore_ascii_case("ok");
        log.record(suite_name.clone(), passed, false);
    }

    for cmp in &suite.comparisons {
        let passed = cmp.verdict.eq_ignore_ascii_case("pass")
            || cmp.verdict.eq_ignore_ascii_case("passed")
            || cmp.verdict.eq_ignore_ascii_case("ok");

        if !cmp.signal.is_empty() {
            if !suite_name.is_empty() {
                log.record(
                    format!("{suite_name}::{}", cmp.signal),
                    passed,
                    false,
                );
            }
            log.record(cmp.signal.clone(), passed, false);
        }
        if !cmp.key.is_empty() {
            if !suite_name.is_empty() {
                log.record(format!("{suite_name}::{}", cmp.key), passed, false);
            }
            log.record(cmp.key.clone(), passed, false);
        }
    }
}

/// Parses cross-validation comparison results (`cross-val-report.json`) into a [`ResultLog`].
///
/// Records suite names, `<suite>::<signal>`, `<suite>::<key>`, signal names, and comparison keys.
/// A suite passes when its status is `"Pass"`. A comparison passes when its verdict is `"pass"`.
/// Conflicting records for the same key are marked ambiguous.
///
/// # Arguments
/// * `json_str` - JSON string of cross-validation results or a saved `cross-val-report.json` artifact.
///
/// # Returns
/// A [`ResultLog`] keyed by suite and comparison identifiers.
///
/// # Panics
/// This function does not panic.
#[must_use]
pub fn parse_compare_results(json_str: &str) -> ResultLog {
    let mut log = ResultLog::default();
    if let Ok(report) = serde_json::from_str::<ValidationReport>(json_str) {
        for suite in &report.suites {
            record_compare_suite(&mut log, suite);
        }
    } else if let Ok(suites) =
        serde_json::from_str::<Vec<SuiteReport>>(json_str)
    {
        for suite in &suites {
            record_compare_suite(&mut log, suite);
        }
    }
    log
}

fn get_xml_attr(
    e: &quick_xml::events::BytesStart<'_>,
    key: &str,
) -> Option<String> {
    for attr in e.attributes().flatten() {
        if attr.key.local_name().as_ref() == key
            && let Ok(val) =
                attr.normalized_value(quick_xml::XmlVersion::Implicit1_0)
        {
            return Some(val.into_owned());
        }
    }
    None
}

fn record_junit_case(log: &mut ResultLog, ctx: JunitCaseCtx<'_>, passed: bool) {
    log.record(ctx.name.to_string(), passed, false);
    if let Some(cls) = ctx.class {
        log.record(format!("{cls}::{}", ctx.name), passed, false);
        log.record(format!("{cls}.{}", ctx.name), passed, false);
    }
    if let Some(suite) = ctx.suite {
        log.record(format!("{suite}::{}", ctx.name), passed, false);
        if let Some(cls) = ctx.class {
            log.record(format!("{suite}::{cls}::{}", ctx.name), passed, false);
        }
    }
}

fn handle_junit_start(
    e: &quick_xml::events::BytesStart<'_>,
    suite_stack: &mut SuiteStack,
    active_case: &mut Option<ActiveTestCase>,
) {
    match e.local_name().as_ref() {
        "testsuite" => suite_stack.push(get_xml_attr(e, "name")),
        "testcase" => {
            let name = get_xml_attr(e, "name");
            let classname = get_xml_attr(e, "classname");
            let suite = suite_stack
                .iter()
                .rev()
                .find_map(|s| s.as_deref())
                .map(str::to_string);
            if let Some(name) = name {
                *active_case = Some(ActiveTestCase {
                    name,
                    classname,
                    suite,
                    has_failure: false,
                    is_skipped: false,
                });
            }
        }
        "failure" | "error" => {
            if let Some(case) = active_case {
                case.has_failure = true;
            }
        }
        "skipped" => {
            if let Some(case) = active_case {
                case.is_skipped = true;
            }
        }
        _ => {}
    }
}

fn handle_junit_empty(
    e: &quick_xml::events::BytesStart<'_>,
    suite_stack: &SuiteStack,
    active_case: &mut Option<ActiveTestCase>,
    log: &mut ResultLog,
) {
    match e.local_name().as_ref() {
        "testcase" => {
            if let Some(name) = get_xml_attr(e, "name") {
                let classname = get_xml_attr(e, "classname");
                let suite = suite_stack.iter().rev().find_map(|s| s.as_deref());
                record_junit_case(
                    log,
                    JunitCaseCtx {
                        name: &name,
                        class: classname.as_deref(),
                        suite,
                    },
                    true,
                );
            }
        }
        "failure" | "error" => {
            if let Some(case) = active_case {
                case.has_failure = true;
            }
        }
        "skipped" => {
            if let Some(case) = active_case {
                case.is_skipped = true;
            }
        }
        _ => {}
    }
}

fn handle_junit_end(
    tag: &str,
    suite_stack: &mut SuiteStack,
    active_case: &mut Option<ActiveTestCase>,
    log: &mut ResultLog,
) {
    match tag {
        "testsuite" => {
            suite_stack.pop();
        }
        "testcase" => {
            if let Some(case) = active_case.take()
                && !case.is_skipped
            {
                let passed = !case.has_failure;
                record_junit_case(
                    log,
                    JunitCaseCtx {
                        name: &case.name,
                        class: case.classname.as_deref(),
                        suite: case.suite.as_deref(),
                    },
                    passed,
                );
            }
        }
        _ => {}
    }
}

/// Parses `JUnit` XML test results into a [`ResultLog`].
///
/// Scans `<testcase>` elements for `name` and `classname` attributes, judging each
/// case by the presence of `<failure>` or `<error>` child elements. Skipped test cases
/// are ignored (remaining `Unrun`). Conflicting outcomes for the same target are marked
/// ambiguous. XML comments, CDATA sections, and processing instructions are safely
/// skipped by the underlying XML reader.
///
/// # Arguments
/// * `text` - `JUnit` XML string or a saved `junit.xml` artifact.
///
/// # Returns
/// A [`ResultLog`] keyed by case name, `<classname>::<name>`, and `<classname>.<name>`.
///
/// # Panics
/// This function does not panic.
#[must_use]
pub fn parse_junit_results(text: &str) -> ResultLog {
    let mut log = ResultLog::default();
    let mut reader = quick_xml::Reader::from_str(text);
    let mut suite_stack: SuiteStack = Vec::new();
    let mut active_case: Option<ActiveTestCase> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Start(ref e)) => {
                handle_junit_start(e, &mut suite_stack, &mut active_case);
            }
            Ok(quick_xml::events::Event::Empty(ref e)) => {
                handle_junit_empty(e, &suite_stack, &mut active_case, &mut log);
            }
            Ok(quick_xml::events::Event::End(ref e)) => {
                handle_junit_end(
                    e.local_name().as_ref(),
                    &mut suite_stack,
                    &mut active_case,
                    &mut log,
                );
            }
            Ok(quick_xml::events::Event::Eof) | Err(_) => break,
            _ => {}
        }
    }

    log
}

/// What `log` records for `target`: an exact identifier, or, when `strip`
/// and the log has no binary headers, the identifier without its leading
/// crate segment.
fn lookup(log: &ResultLog, target: &str, strip: bool) -> Lookup {
    let key = if log.outcomes.contains_key(target) {
        target
    } else {
        match target.split_once("::") {
            Some((_, rest))
                if strip
                    && log.binaries.is_empty()
                    && log.outcomes.contains_key(rest) =>
            {
                rest
            }
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
        .and_then(|m| m.result_artifact.as_deref().or(m.result_dir.as_deref()))
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
    if covered && lookup(log, target, true) == Lookup::Absent {
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
    Artifact { method, results }: Artifact<'_>,
    target: &str,
    ctx: &mut StatusCtx<'_>,
) -> ItemOutcome {
    let (strip, method) = (method.strip, method.name);
    let artifact = artifact_name(ctx.config, method);
    let outcome = |passed: bool, message: Option<String>| ItemOutcome {
        target: target.to_string(),
        passed,
        message,
    };
    let Some(results) = results else {
        ctx.defects.push(Defect::at(
            row,
            format!(
                "condition {} target '{target}' not found: {artifact} is missing",
                row.id
            ),
        ));
        return outcome(false, Some("result log missing".to_string()));
    };
    match lookup(results, target, strip) {
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

/// The status of a condition `row` of the artifact method `method`.
fn check_artifact(
    row: &Row,
    ctx: &mut StatusCtx<'_>,
    method: &ArtifactMethod,
) -> ConditionStatus {
    let results = ctx.results.remove(method.name).unwrap_or_else(|| {
        ctx.config
            .method
            .get(method.name)
            .and_then(|m| m.result_artifact.as_ref())
            .and_then(|path| read_text(&ctx.base.join(path)).ok())
            .map(|text| (method.parse)(&text))
    });
    let items: Vec<ItemOutcome> = row
        .targets
        .iter()
        .map(|target| {
            let artifact = Artifact {
                method,
                results: results.as_ref(),
            };
            evaluate_target(row, artifact, target, ctx)
        })
        .collect();
    ctx.results.insert(method.name, results);
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
    condition(row, derived, items)
}

fn check_libtest(row: &Row, ctx: &mut StatusCtx<'_>) -> ConditionStatus {
    let method = ArtifactMethod {
        name: "libtest",
        parse: parse_libtest_results,
        strip: true,
    };
    check_artifact(row, ctx, &method)
}

fn check_kani(row: &Row, ctx: &mut StatusCtx<'_>) -> ConditionStatus {
    let method = ArtifactMethod {
        name: "kani",
        parse: parse_kani_results,
        strip: true,
    };
    check_artifact(row, ctx, &method)
}

fn check_compare(row: &Row, ctx: &mut StatusCtx<'_>) -> ConditionStatus {
    let method = ArtifactMethod {
        name: "compare",
        parse: parse_compare_results,
        strip: false,
    };
    check_artifact(row, ctx, &method)
}

fn check_junit(row: &Row, ctx: &mut StatusCtx<'_>) -> ConditionStatus {
    let method = ArtifactMethod {
        name: "junit",
        parse: parse_junit_results,
        strip: false,
    };
    check_artifact(row, ctx, &method)
}

fn check_gate(row: &Row, ctx: &mut StatusCtx<'_>) -> ConditionStatus {
    let (items, statuses): GateTargets = row
        .targets
        .iter()
        .map(|target| gate_target(row, target, ctx))
        .unzip();
    let derived = statuses.into_iter().max().unwrap_or(Status::Uncovered);
    condition(row, derived, items)
}

fn load_gate_outcome(target: &str, ctx: &mut StatusCtx<'_>) -> GateLoadOutcome {
    let artifact = format!("{target}{RESULT_SUFFIX}");
    ctx.gates
        .entry(target.to_string())
        .or_insert_with(|| {
            let result_dir = ctx
                .config
                .method
                .get("gate")
                .and_then(|m| m.result_dir.as_deref())
                .unwrap_or("target/ci-artifacts");
            let path = ctx.base.join(result_dir).join(&artifact);
            match GateOutcome::load_from_file(&path) {
                Ok(gate_outcome) => {
                    if gate_outcome.schema == crate::gate::OUTCOME_SCHEMA {
                        GateLoadOutcome::Loaded(gate_outcome)
                    } else {
                        GateLoadOutcome::Invalid(format!(
                            "unsupported outcome schema version: {}",
                            gate_outcome.schema
                        ))
                    }
                }
                Err(crate::error::GateError::Io(err))
                    if err.kind() == std::io::ErrorKind::NotFound =>
                {
                    GateLoadOutcome::Missing
                }
                Err(err) => GateLoadOutcome::Invalid(err.to_string()),
            }
        })
        .clone()
}

fn evaluate_gate_verdict(
    row: &Row,
    target: &str,
    ctx: &mut StatusCtx<'_>,
    verdict: Verdict,
) -> GateTargetEval {
    let outcome = |passed: bool, message: Option<String>| ItemOutcome {
        target: target.to_string(),
        passed,
        message,
    };
    match verdict {
        Verdict::Pass => (outcome(true, None), Status::Pass),
        Verdict::Fail => {
            ctx.defects.push(Defect::at(
                row,
                format!(
                    "condition {} target '{target}' failed in quality gate",
                    row.id
                ),
            ));
            (
                outcome(false, Some("gate outcome failed".to_string())),
                Status::Fail,
            )
        }
        Verdict::Skipped => {
            ctx.defects.push(Defect::at(
                row,
                format!(
                    "condition {} target '{target}' gate outcome was skipped",
                    row.id
                ),
            ));
            (
                outcome(false, Some("gate outcome was skipped".to_string())),
                Status::Fail,
            )
        }
        Verdict::Warn => {
            ctx.defects.push(Defect::at(
                row,
                format!(
                    "condition {} target '{target}' gate outcome has warnings",
                    row.id
                ),
            ));
            (
                outcome(false, Some("gate outcome has warnings".to_string())),
                Status::Fail,
            )
        }
    }
}

fn gate_target(
    row: &Row,
    target: &str,
    ctx: &mut StatusCtx<'_>,
) -> GateTargetEval {
    let outcome = |passed: bool, message: Option<String>| ItemOutcome {
        target: target.to_string(),
        passed,
        message,
    };
    let artifact = format!("{target}{RESULT_SUFFIX}");
    match load_gate_outcome(target, ctx) {
        GateLoadOutcome::Missing => {
            ctx.defects.push(Defect::at(
                row,
                format!(
                    "condition {} target '{target}' not found: {artifact} is missing",
                    row.id
                ),
            ));
            (
                outcome(
                    false,
                    Some("gate outcome artifact missing".to_string()),
                ),
                Status::Unrun,
            )
        }
        GateLoadOutcome::Invalid(err) => {
            ctx.defects.push(Defect::at(
                row,
                format!(
                    "condition {} target '{target}' in {artifact} is invalid: {err}",
                    row.id
                ),
            ));
            (
                outcome(
                    false,
                    Some("gate outcome artifact invalid".to_string()),
                ),
                Status::Fail,
            )
        }
        GateLoadOutcome::Loaded(gate_outcome) => {
            evaluate_gate_verdict(row, target, ctx, gate_outcome.verdict)
        }
    }
}

/// The derived graph gives every link condition it interned its status, so a
/// link condition reaches this check only when the graph did not intern it.
fn check_link(row: &Row, ctx: &mut StatusCtx<'_>) -> ConditionStatus {
    ctx.defects.push(Defect::at(
        row,
        format!("condition {} is not in the link graph", row.id),
    ));
    condition(row, Status::Fail, Vec::new())
}

/// The status of a `doc-status` condition: the worst status of its targets.
fn check_doc_status(row: &Row, ctx: &mut StatusCtx<'_>) -> ConditionStatus {
    let (items, statuses): DocTargets = row
        .targets
        .iter()
        .map(|target| doc_target(row, target, ctx))
        .unzip();
    let derived = statuses.into_iter().max().unwrap_or(Status::Uncovered);
    condition(row, derived, items)
}

fn find_doc_or_decision_status(
    target: &str,
    ctx: &StatusCtx<'_>,
) -> Option<TargetDocState> {
    if ctx
        .decisions
        .as_ref()
        .is_some_and(|rules| rules.names_decision(target))
    {
        decision_status(target, ctx).map(|status| {
            let accepted = status.as_deref().is_some_and(|s| {
                ctx.decisions.as_ref().is_some_and(|r| r.accepts(s))
            });
            TargetDocState {
                status,
                passed: accepted,
                draft: false,
            }
        })
    } else {
        document_status(target, ctx).map(|status| {
            let accepted =
                matches!(status.as_deref(), Some("Ready" | "Shipped"));
            let draft = status.as_deref() == Some("Draft");
            TargetDocState {
                status,
                passed: accepted || draft,
                draft,
            }
        })
    }
}

/// The outcome of one `doc-status` target: a decision ID or a traced
/// document name.
fn doc_target(
    row: &Row,
    target: &str,
    ctx: &mut StatusCtx<'_>,
) -> DocTargetEval {
    let outcome = |passed: bool, message: Option<String>| ItemOutcome {
        target: target.to_string(),
        passed,
        message,
    };
    let row_doc = row.id.split(QUALIFIER).next().unwrap_or_default();
    if !row_doc.is_empty() && target == row_doc {
        let message = "cannot target its own document".to_string();
        ctx.defects.push(Defect::at(
            row,
            format!("condition {} target '{target}' {message}", row.id),
        ));
        return (outcome(false, Some(message)), Status::Fail);
    }
    let Some(doc_state) = find_doc_or_decision_status(target, ctx) else {
        let message = "names no traced document or decision".to_string();
        ctx.defects.push(Defect::at(
            row,
            format!("condition {} target '{target}' {message}", row.id),
        ));
        return (outcome(false, Some(message)), Status::Unrun);
    };
    if doc_state.draft {
        ctx.warnings.push(InterpreterWarning {
            code: "W-4".to_string(),
            file: row.file.clone(),
            line: row.line,
            item: target.to_string(),
            message: "document status is Draft".to_string(),
        });
    }
    if doc_state.passed {
        return (outcome(true, None), Status::Pass);
    }
    let message = doc_state.status.map_or_else(
        || "has no status".to_string(),
        |s| format!("has status {s}"),
    );
    ctx.defects.push(Defect::at(
        row,
        format!("condition {} target '{target}' {message}", row.id),
    ));
    (outcome(false, Some(message)), Status::Fail)
}

/// The status of decision `id`: `None` when no selected decision record
/// defines it.
fn decision_status(id: &str, ctx: &StatusCtx<'_>) -> RecordStatus {
    let rules = ctx.decisions.as_ref()?;
    let config = ctx.config.decisions.as_ref()?;
    let files =
        select(ctx.base, &config.files, &ctx.config.doc_suffixes()).ok()?;
    files.iter().find_map(|file| {
        let source = read_text(&ctx.base.join(file)).ok()?;
        rules.record_status(file, &source, id)
    })
}

/// The status badge of traced document `doc`: `None` when no row names the
/// document or its file cannot be read.
fn document_status(doc: &str, ctx: &StatusCtx<'_>) -> RecordStatus {
    let file = ctx.documents.get(doc)?;
    let source = read_text(&ctx.base.join(file)).ok()?;
    Some(ctx.patterns.status.as_ref().and_then(|re| {
        re.captures(&source)
            .and_then(|c| c.name("status"))
            .map(|m| m.as_str().to_string())
    }))
}

/// Condition `row` with `status` and `items`.
fn condition(
    row: &Row,
    status: Status,
    items: Vec<ItemOutcome>,
) -> ConditionStatus {
    ConditionStatus {
        id: row.id.clone(),
        parents: row.parents.clone(),
        method: row.method.clone().unwrap_or_default(),
        status,
        items,
    }
}

/// The status of condition `row` by its method; `Uncovered` with no target
/// and `Fail` for an unknown method.
fn condition_status(row: &Row, ctx: &mut StatusCtx<'_>) -> ConditionStatus {
    let method = row.method.as_deref().unwrap_or_default();
    let Some(entry) = METHODS.iter().find(|m| m.name == method) else {
        ctx.defects.push(Defect::at(
            row,
            format!("condition {} names unknown method `{method}`", row.id),
        ));
        return condition(row, Status::Fail, Vec::new());
    };
    if row.targets.is_empty() {
        ctx.defects.push(Defect::at(
            row,
            format!("condition {} names no target", row.id),
        ));
        return condition(row, Status::Uncovered, Vec::new());
    }
    (entry.check)(row, ctx)
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
/// ID and the worse of the link-graph rollup and those conditions. A condition
/// the graph omitted (for example an unqualified id) still contributes through
/// the children, so the requirement cannot stay green while a child fails.
fn requirement_statuses(
    order: &[&str],
    evaluated: &[ConditionStatus],
    rolled: &RequirementMap<'_>,
) -> Vec<RequirementStatus> {
    let mut by_parent = ByParent::new();
    for condition in evaluated {
        for parent in &condition.parents {
            by_parent
                .entry(parent.as_str())
                .or_default()
                .push(condition.clone());
        }
    }
    order
        .iter()
        .map(|&id| {
            let mut children = by_parent.remove(id).unwrap_or_default();
            children.sort_by(|a, b| a.id.cmp(&b.id));
            let leaf = worst(&children);
            RequirementStatus {
                id: id.to_string(),
                status: rolled.get(id).copied().map_or(leaf, |g| g.max(leaf)),
                conditions: children,
            }
        })
        .collect()
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
    for (method, entry) in &config.method {
        let Some(artifact) = entry.result_artifact.as_deref() else {
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
    [Status::Pass, Status::Uncovered, Status::Unrun, Status::Fail]
        .into_iter()
        .map(|status| (status, 0))
        .collect()
}

fn evaluate_all_conditions<'a>(
    graph: &LinkGraph<'a, Acyclic>,
    cond_idx: &Indexed<'_>,
    ctx: &mut StatusCtx<'_>,
    counts: &mut StatusCounts,
) -> ConditionEvalResult<'a> {
    let (conditions, condition_order) = cond_idx;
    let mut evaluated = Vec::new();

    let LinkStatuses {
        conditions: mut derived,
        requirements,
    } = graph.derive(|row| condition_status(row, ctx));

    for row in condition_order.iter().filter_map(|id| conditions.get(id)) {
        // `LinkGraph::resolve` only interns `<doc>#…` ids. A row the graph
        // omitted must still take its leaf status so the gate cannot pass by
        // skipping it.
        let status = derived
            .remove(row.id.as_str())
            .unwrap_or_else(|| condition_status(row, ctx));

        if let Some(count) = counts.get_mut(&status.status) {
            *count = count.saturating_add(1);
        }
        evaluated.push(status);
    }

    (evaluated, requirements)
}

/// The link view of `graph` as text in `format`, with the condition statuses
/// of `report`.
#[must_use]
pub fn render_graph(
    graph: &LinkGraph<'_, Acyclic>,
    report: &TraceReport,
    format: GraphFormat,
) -> String {
    let statuses = report
        .requirements
        .iter()
        .flat_map(|r| &r.conditions)
        .map(|c| (c.id.as_str(), c.clone()))
        .collect();
    graph.render(format, Some(&statuses))
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
    let (graph, link_defects) = link_graph(reqs);
    derive_with_graph(&graph, link_defects, config, base)
}

/// Derives condition and requirement status relative to `base` from the rows
/// of `graph`, their acyclic link graph, and its `link_defects`.
#[must_use]
pub fn derive_with_graph(
    graph: &LinkGraph<'_, Acyclic>,
    link_defects: Defects,
    config: &TraceConfig,
    base: &Path,
) -> Derived {
    let reqs = graph.rows();
    let (_, order) = first_by_id(reqs, DEFINITION);
    let cond_idx = first_by_id(reqs, CONDITION);

    let mut defects = link_defects;
    let mut warnings = Vec::new();
    let interpreter_results = load_interpreter_log(config, base);
    let mut counts = initial_status_counts();

    let mut documents = DocFiles::new();
    for row in reqs {
        if let Some((doc, _)) = row.id.split_once(QUALIFIER) {
            documents.entry(doc).or_insert(row.file.as_str());
        }
    }
    let mut ctx = StatusCtx {
        config,
        base,
        results: ResultCache::new(),
        gates: GateCache::new(),
        interpreter_results: interpreter_results.as_ref(),
        documents,
        decisions: config
            .decisions
            .as_ref()
            .and_then(|d| d.rules(Path::new("trace.toml")).ok()),
        patterns: DocumentMetaPatterns {
            status: regex::Regex::new(
                r"(?:Design|Doc)%20Status-(?P<status>[A-Za-z_]+)-",
            )
            .ok(),
        },
        defects: &mut defects,
        warnings: &mut warnings,
    };

    let (evaluated, rolled) =
        evaluate_all_conditions(graph, &cond_idx, &mut ctx, &mut counts);

    let requirements = requirement_statuses(&order, &evaluated, &rolled);
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
        warnings,
    };

    (report, defects)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use crate::trace::reqs::{DecisionConfig, MethodConfig};

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
            schema: SCHEMA,
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
            warnings: Vec::new(),
        }
    }

    #[test]
    fn a_report_passes_only_without_fail_unrun_or_uncovered() {
        assert!(
            report(
                &[(Status::Pass, 3), (Status::Fail, 0)],
                &[Status::Pass, Status::Pass]
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
                parse_kani_results(&log).outcomes.get("h::bad"),
                Some(&false),
                "{marker}"
            );
        }
        let ok = "Checking harness h::ok...\n** 2 of 2 cover properties satisfied\nVERIFICATION:- SUCCESSFUL\n";
        assert_eq!(parse_kani_results(ok).outcomes.get("h::ok"), Some(&true));
    }

    #[test]
    fn each_result_format_has_a_parser() {
        let lt = parse_libtest_results("test a::t ... ok\n");
        assert_eq!(lt.outcomes.get("a::t"), Some(&true));
        let kn = parse_kani_results(
            "Checking harness h::ok...\nVERIFICATION:- SUCCESSFUL\n",
        );
        assert_eq!(kn.outcomes.get("h::ok"), Some(&true));
        let cp = parse_compare_results(
            r#"{"suites":[{"name":"mat","status":"Pass","comparisons":[]}]}"#,
        );
        assert_eq!(cp.outcomes.get("mat"), Some(&true));
        let ju = parse_junit_results(
            r#"<testsuite><testcase name="c1" classname="pkg.Cls"/></testsuite>"#,
        );
        assert_eq!(ju.outcomes.get("c1"), Some(&true));
    }

    #[test]
    fn the_method_table_names_each_method_once() {
        let names: BTreeSet<&str> = METHODS.iter().map(|m| m.name).collect();
        assert_eq!(names.len(), METHODS.len());
        for name in names {
            assert!(super::super::reqs::is_method(name), "{name}");
        }
        assert!(!super::super::reqs::is_method("review"));
        assert!(!super::super::reqs::is_method("deferred"));
    }

    #[test]
    fn compare_results_are_parsed() {
        let text = r#"{
            "suites": [
                {
                    "name": "matrix_dense",
                    "status": "Pass",
                    "comparisons": [
                        {
                            "key": "matrix.inv/scipy.rust",
                            "signal": "matrix.inv",
                            "verdict": "pass"
                        },
                        {
                            "key": "matrix.det/scipy.rust",
                            "signal": "matrix.det",
                            "verdict": "fail"
                        }
                    ]
                }
            ]
        }"#;
        let log = parse_compare_results(text);
        assert_eq!(log.outcomes.get("matrix_dense"), Some(&true));
        assert_eq!(log.outcomes.get("matrix.inv"), Some(&true));
        assert_eq!(log.outcomes.get("matrix_dense::matrix.inv"), Some(&true));
        assert_eq!(log.outcomes.get("matrix.inv/scipy.rust"), Some(&true));
        assert_eq!(
            log.outcomes.get("matrix_dense::matrix.inv/scipy.rust"),
            Some(&true)
        );
        assert_eq!(log.outcomes.get("matrix.det"), Some(&false));
        assert_eq!(log.outcomes.get("matrix_dense::matrix.det"), Some(&false));
        assert_eq!(log.outcomes.get("matrix.det/scipy.rust"), Some(&false));
        assert_eq!(
            log.outcomes.get("matrix_dense::matrix.det/scipy.rust"),
            Some(&false)
        );
    }

    #[test]
    fn compare_status_variants_are_parsed() {
        let text = r#"{
            "suites": [
                {
                    "name": "s_passed",
                    "status": "passed",
                    "comparisons": [
                        {"key": "k1", "signal": "sig_ok", "verdict": "ok"},
                        {"key": "k2", "signal": "sig_passed", "verdict": "passed"},
                        {"key": "k3", "signal": "sig_err", "verdict": "error"},
                        {"key": "k4", "signal": "sig_failed", "verdict": "failed"}
                    ]
                },
                {"name": "s_ok", "status": "ok", "comparisons": []}
            ]
        }"#;
        let log = parse_compare_results(text);
        assert_eq!(log.outcomes.get("s_passed"), Some(&true));
        assert_eq!(log.outcomes.get("s_ok"), Some(&true));
        assert_eq!(log.outcomes.get("sig_ok"), Some(&true));
        assert_eq!(log.outcomes.get("sig_passed"), Some(&true));
        assert_eq!(log.outcomes.get("sig_err"), Some(&false));
        assert_eq!(log.outcomes.get("sig_failed"), Some(&false));
    }

    #[test]
    fn compare_conflicting_records_are_ambiguous() {
        let text = r#"{
            "suites": [
                {
                    "name": "s",
                    "status": "Pass",
                    "comparisons": [
                        {"key": "k", "signal": "sig", "verdict": "pass"},
                        {"key": "k", "signal": "sig", "verdict": "fail"}
                    ]
                }
            ]
        }"#;
        let log = parse_compare_results(text);
        assert!(log.ambiguous.contains("sig"));
        assert!(log.ambiguous.contains("s::sig"));
    }

    #[test]
    fn compare_missing_required_name_or_verdict_fails_deserialization() {
        // Missing "name" on a suite fails deserialization, resulting in an empty log.
        let missing_name = r#"{"suites":[
            {"name":"suite_a","status":"Pass","comparisons":[{"signal":"torque","verdict":"pass"}]},
            {"comparisons":[{"signal":"torque","verdict":"fail"}]}
        ]}"#;
        let log = parse_compare_results(missing_name);
        assert_eq!(lookup(&log, "torque", false), Lookup::Absent);

        // Missing "verdict" on a comparison fails deserialization, resulting in an empty log.
        let missing_verdict = r#"{"suites":[
            {"name":"suite_a","status":"Pass","comparisons":[
                {"signal":"torque","verdict":"pass"},
                {"signal":"torque"}
            ]}
        ]}"#;
        let log = parse_compare_results(missing_verdict);
        assert_eq!(lookup(&log, "torque", false), Lookup::Absent);
    }

    #[test]
    fn compare_unknown_verdict_or_empty_suite_name_fails_closed() {
        let text = r#"{"suites":[
            {"name":"suite_a","status":"Pass","comparisons":[{"signal":"torque","verdict":"pass"}]},
            {"name":"","status":"Pass","comparisons":[{"signal":"torque","verdict":"fail"}]}
        ]}"#;
        let log = parse_compare_results(text);
        assert!(log.ambiguous.contains("torque"));
        assert_eq!(log.outcomes.get(""), None);

        let empty_status = r#"{"suites":[
            {"name":"suite_empty_status","status":"","comparisons":[{"signal":"power","verdict":"pass"}]}
        ]}"#;
        let log_empty_status = parse_compare_results(empty_status);
        assert_eq!(log_empty_status.outcomes.get("suite_empty_status"), None);
        assert_eq!(log_empty_status.outcomes.get("power"), Some(&true));

        let unknown_verdict = r#"{"suites":[
            {"name":"suite_a","status":"Pass","comparisons":[
                {"signal":"torque","verdict":"pass"},
                {"signal":"torque","verdict":"unknown"}
            ]}
        ]}"#;
        let log = parse_compare_results(unknown_verdict);
        assert!(log.ambiguous.contains("torque"));
        assert!(log.ambiguous.contains("suite_a::torque"));

        let single_unknown = r#"{"suites":[
            {"name":"suite_a","status":"Pass","comparisons":[
                {"signal":"speed","verdict":"pending"}
            ]}
        ]}"#;
        let log_single = parse_compare_results(single_unknown);
        assert_eq!(log_single.outcomes.get("speed"), Some(&false));
        assert_eq!(log_single.outcomes.get("suite_a::speed"), Some(&false));
    }

    #[test]
    fn junit_results_are_parsed() {
        let text = r#"<?xml version="1.0" encoding="UTF-8"?>
        <testsuites>
            <testsuite name="SuiteA">
                <testcase name="test_ok" classname="pkg.ClassA"/>
                <testcase name='test_single' classname='pkg.ClassSingle'/>
                <testcase name="test_fail" classname="pkg.ClassA">
                    <failure message="assertion failed">stack trace</failure>
                </testcase>
                <testcase name="test_error" classname="pkg.ClassA">
                    <error message="runtime error"/>
                </testcase>
                <testcase name="test_skip" classname="pkg.ClassA">
                    <skipped message="ignored"/>
                </testcase>
                <testcase name="test_skip_start" classname="pkg.ClassA">
                    <skipped>reason with start/end tag</skipped>
                </testcase>
            </testsuite>
        </testsuites>"#;
        let log = parse_junit_results(text);
        assert_eq!(log.outcomes.get("test_ok"), Some(&true));
        assert_eq!(log.outcomes.get("test_single"), Some(&true));
        assert_eq!(log.outcomes.get("pkg.ClassA::test_ok"), Some(&true));
        assert_eq!(log.outcomes.get("pkg.ClassA.test_ok"), Some(&true));
        assert_eq!(log.outcomes.get("SuiteA::test_ok"), Some(&true));
        assert_eq!(
            log.outcomes.get("SuiteA::pkg.ClassA::test_ok"),
            Some(&true)
        );

        assert_eq!(log.outcomes.get("test_fail"), Some(&false));
        assert_eq!(log.outcomes.get("pkg.ClassA::test_fail"), Some(&false));

        assert_eq!(log.outcomes.get("test_error"), Some(&false));
        assert_eq!(log.outcomes.get("pkg.ClassA::test_error"), Some(&false));

        // Skipped test is omitted (remains `Unrun`)
        assert_eq!(log.outcomes.get("test_skip"), None);
        assert_eq!(log.outcomes.get("test_skip_start"), None);
    }

    #[test]
    fn junit_conflicting_records_are_ambiguous() {
        let text = r#"<testsuite>
            <testcase name="dup_test" classname="pkg.Cls"/>
            <testcase name="dup_test" classname="pkg.Cls"><failure/></testcase>
        </testsuite>"#;
        let log = parse_junit_results(text);
        assert!(log.ambiguous.contains("dup_test"));
        assert!(log.ambiguous.contains("pkg.Cls::dup_test"));
    }

    /// `classname` before `name` is the usual producer order. A bare search
    /// for `name=` binds the trailing `name=` of `classname=` and can
    /// record a passing `classname` while dropping a failed case of that name.
    #[test]
    fn junit_classname_before_name_does_not_false_pass() {
        let text = r#"<testsuite>
            <testcase classname="safety_check" name="setup"/>
            <testcase classname="other" name="safety_check"><failure/></testcase>
        </testsuite>"#;
        let log = parse_junit_results(text);
        assert_eq!(log.outcomes.get("setup"), Some(&true));
        assert_eq!(log.outcomes.get("safety_check::setup"), Some(&true));
        assert_eq!(log.outcomes.get("safety_check"), Some(&false));
        assert_eq!(log.outcomes.get("other::safety_check"), Some(&false));
        assert_ne!(log.outcomes.get("safety_check"), Some(&true));
    }

    /// `system-out` CDATA often embeds log markers. A bare `find("</testcase>")`
    /// closes on the CDATA text, drops the real `<failure>`, and records Pass.
    #[test]
    fn junit_cdata_close_marker_does_not_false_pass() {
        let text = r#"<testsuite>
            <testcase classname="pkg" name="safety_check">
                <system-out><![CDATA[log contains </testcase> marker]]></system-out>
                <failure message="assertion failed"/>
            </testcase>
            <testcase classname="pkg" name="mentions_failure">
                <system-out><![CDATA[look for <failure in logs]]></system-out>
            </testcase>
        </testsuite>"#;
        let log = parse_junit_results(text);
        assert_eq!(log.outcomes.get("safety_check"), Some(&false));
        assert_eq!(log.outcomes.get("pkg::safety_check"), Some(&false));
        assert_eq!(log.outcomes.get("mentions_failure"), Some(&true));
        assert_ne!(log.outcomes.get("safety_check"), Some(&true));
    }

    /// Comments and suite-level CDATA can embed a literal `<testcase .../>`.
    /// The XML reader must skip those as tokens and not record ghost outcomes.
    #[test]
    fn junit_comment_or_cdata_testcase_does_not_false_pass() {
        let text = r#"<?xml version="1.0"?>
        <testsuite name="SuiteA">
            <!-- <testcase classname="pkg" name="ghost_pass"/> -->
            <system-out><![CDATA[<testcase classname="pkg" name="cdata_ghost"/>]]></system-out>
            <testcase classname="pkg" name="real_test"/>
            <!-- </testcase> in comment before failure -->
            <testcase classname="pkg" name="real_fail">
                <!-- </testcase> -->
                <failure message="boom"/>
            </testcase>
        </testsuite>"#;
        let log = parse_junit_results(text);
        assert_eq!(log.outcomes.get("ghost_pass"), None);
        assert_eq!(log.outcomes.get("cdata_ghost"), None);
        assert_eq!(log.outcomes.get("real_test"), Some(&true));
        assert_eq!(log.outcomes.get("real_fail"), Some(&false));
    }

    /// Entity decoding and multi-line attributes are handled according to XML standards.
    #[test]
    fn junit_entities_and_multiline_attributes() {
        let text = r#"<testsuite name="SuiteA">
            <testcase
                classname="pkg.ClassA"
                name="test_with_&amp;_and_&quot;quotes&quot;"
            />
            <testcase
                name="test_fail"
                classname="pkg.ClassA"
            >
                <failure message="fail"/>
            </testcase>
        </testsuite>"#;
        let log = parse_junit_results(text);
        assert_eq!(log.outcomes.get("test_with_&_and_\"quotes\""), Some(&true));
        assert_eq!(
            log.outcomes.get("pkg.ClassA::test_with_&_and_\"quotes\""),
            Some(&true)
        );
        assert_eq!(log.outcomes.get("test_fail"), Some(&false));
    }

    fn write_log(dir: &Path, name: &str, text: &str) {
        fs::write(dir.join("target/ci-artifacts").join(name), text).unwrap();
    }

    fn default_config() -> TraceConfig {
        let method = METHODS
            .iter()
            .map(|m| m.name)
            .filter(|m| !matches!(*m, "link" | "doc-status"))
            .map(|m| {
                if m == "gate" {
                    return (
                        m.to_string(),
                        MethodConfig {
                            result_artifact: None,
                            result_dir: Some("target/ci-artifacts".to_string()),
                            interpreter_artifact: None,
                        },
                    );
                }
                let file = match m {
                    "compare" => "cross-val-report.json".to_string(),
                    "junit" => "junit.xml".to_string(),
                    other => format!("{other}.log"),
                };
                (
                    m.to_string(),
                    MethodConfig {
                        result_artifact: Some(format!(
                            "target/ci-artifacts/{file}"
                        )),
                        result_dir: None,
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
    fn kani_conflicting_records_are_ambiguous() {
        let kani = "Checking harness h::dup...\nVERIFICATION:- FAILED\n\
                    Checking harness h::dup...\nVERIFICATION:- SUCCESSFUL\n\
                    ** 1 of 1 cover properties satisfied\n";
        let kani_run =
            run("kani_ambiguous", kani, condition("kani", &["h::dup"]));
        assert_eq!(status(&kani_run), Some(Status::Fail));
        assert!(
            kani_run
                .1
                .first()
                .is_some_and(|d| d.message.contains("is ambiguous in")),
            "{:?}",
            kani_run.1
        );
    }

    #[test]
    fn automated_condition_without_targets_is_uncovered() {
        let derived = run("uncovered", "", condition("libtest", &[]));
        assert_eq!(status(&derived), Some(Status::Uncovered));
        assert!(!derived.0.passes());
        assert_eq!(
            derived.1.first().map(|d| d.message.as_str()),
            Some("condition w#VC-1.1 names no target")
        );
        assert_eq!(derived.0.counts.get(&Status::Uncovered), Some(&1));
    }

    #[test]
    fn doc_status_condition_with_accepted_adr_passes() {
        let dir = test_dir("doc_status_adr");
        let adr_dir = dir.join("documentation/adr");
        fs::create_dir_all(&adr_dir).unwrap();
        fs::write(
            adr_dir.join("0001.md"),
            "# ADR-0001: Size\n\n![Status](https://img.shields.io/badge/ADR%20Status-Accepted-brightgreen)\n",
        ).unwrap();
        let config = TraceConfig {
            decisions: Some(DecisionConfig {
                files: vec!["documentation/adr".to_string()],
                id: "ADR-[0-9]{4}".to_string(),
                definition: r"^#\s+ADR-[0-9]{4}:".to_string(),
                status: "ADR%20Status-(?P<status>[A-Za-z]+)-".to_string(),
                accepted: vec!["Accepted".to_string()],
            }),
            ..default_config()
        };
        let derived = derive_with_base(
            &[definition(), condition("doc-status", &["ADR-0001"])],
            &config,
            &dir,
        );
        assert_eq!(status(&derived), Some(Status::Pass));
        assert!(derived.0.passes());
        assert!(derived.1.is_empty(), "{:?}", derived.1);
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
    fn libtest_lines_that_are_not_results_are_skipped() {
        let log = parse_libtest_results(
            "running 3 tests\ntest a ... ignored\ntest b\ntest c ... ok\n",
        );
        assert_eq!(log.outcomes.len(), 1);
        assert_eq!(log.outcomes.get("c"), Some(&true));
        assert_eq!(without_doctest_line("f - i"), "f - i");
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
    fn compare_and_junit_targets_match_exactly() {
        let dir = test_dir("exact");
        write_log(
            &dir,
            "cross-val-report.json",
            r#"{"suites":[{"name":"suite_a","status":"Pass","comparisons":[{"signal":"torque","verdict":"pass"}]}]}"#,
        );
        write_log(
            &dir,
            "junit.xml",
            r#"<testsuite name="s"><testcase name="test_ok" classname="A"/></testsuite>"#,
        );
        let derive = |method: &str, target: &str| {
            status(&derive_with_base(
                &[definition(), condition(method, &[target])],
                &default_config(),
                &dir,
            ))
        };
        assert_eq!(derive("compare", "torque"), Some(Status::Pass));
        assert_eq!(derive("compare", "suite_a::torque"), Some(Status::Pass));
        assert_eq!(derive("compare", "suite_b::torque"), Some(Status::Unrun));
        assert_eq!(derive("junit", "A::test_ok"), Some(Status::Pass));
        assert_eq!(derive("junit", "B::test_ok"), Some(Status::Unrun));
    }

    /// Derives one `doc-status` condition naming `targets`, with document `w`
    /// written to `docs/w-design.md` carrying `badge` when given.
    fn doc_status_run(
        name: &str,
        badge: Option<&str>,
        targets: &[&str],
    ) -> Derived {
        let dir = test_dir(name);
        fs::create_dir_all(dir.join("docs")).unwrap();
        let badge = badge.map_or_else(String::new, |b| {
            format!("![Design Status Badge](https://img.shields.io/badge/Design%20Status-{b}-blue)\n")
        });
        fs::write(
            dir.join("docs/w-design.md"),
            "# W (w)\n\n![Design Status Badge](https://img.shields.io/badge/Design%20Status-Ready-blue)\n",
        )
        .unwrap();
        fs::write(
            dir.join("docs/other-design.md"),
            format!("# Other (other)\n\n{badge}"),
        )
        .unwrap();
        let other_cond = Row {
            schema: SCHEMA,
            id: "other#VC-1.1".to_string(),
            kind: CONDITION.to_string(),
            parents: vec!["w#FR-1".to_string()],
            method: Some("doc-status".to_string()),
            targets: vec!["w".to_string()],
            file: "docs/other-design.md".to_string(),
            line: 2,
            text: String::new(),
        };
        derive_with_base(
            &[definition(), other_cond, condition("doc-status", targets)],
            &default_config(),
            &dir,
        )
    }

    #[test]
    fn doc_status_passes_on_a_ready_or_shipped_document() {
        for badge in ["Ready", "Shipped"] {
            let derived = doc_status_run("doc_ready", Some(badge), &["other"]);
            assert_eq!(status(&derived), Some(Status::Pass), "{badge}");
            assert!(derived.1.is_empty(), "{:?}", derived.1);
            assert!(derived.0.warnings.is_empty(), "{:?}", derived.0.warnings);
        }
    }

    #[test]
    fn doc_status_warns_on_a_draft_document() {
        let derived = doc_status_run("doc_draft", Some("Draft"), &["other"]);
        assert_eq!(status(&derived), Some(Status::Pass));
        assert!(derived.0.passes());
        assert!(derived.1.is_empty(), "{:?}", derived.1);
        let codes: Vec<_> = derived
            .0
            .warnings
            .iter()
            .map(|w| (w.code.as_str(), w.item.as_str()))
            .collect();
        assert_eq!(codes, [("W-4", "other")]);
    }

    #[test]
    fn doc_status_fails_on_any_other_status() {
        for badge in [Some("Stale"), None] {
            let derived = doc_status_run("doc_stale", badge, &["other"]);
            assert_eq!(status(&derived), Some(Status::Fail), "{badge:?}");
            assert!(!derived.0.passes());
            assert_eq!(derived.1.len(), 1, "{:?}", derived.1);
        }
    }

    #[test]
    fn doc_status_target_naming_no_traced_document_is_unrun() {
        let derived =
            doc_status_run("doc_untraced", Some("Ready"), &["w-design"]);
        assert_eq!(status(&derived), Some(Status::Unrun));
        assert_eq!(
            derived.1.first().map(|d| d.message.as_str()),
            Some(
                "condition w#VC-1.1 target 'w-design' names no traced \
                 document or decision"
            )
        );
    }

    #[test]
    fn doc_status_self_targeting_fails_as_defect() {
        let derived = doc_status_run("doc_self", Some("Ready"), &["w"]);
        assert_eq!(status(&derived), Some(Status::Fail));
        assert_eq!(
            derived.1.first().map(|d| d.message.as_str()),
            Some(
                "condition w#VC-1.1 target 'w' cannot target its own document"
            )
        );
    }

    #[test]
    fn doc_status_reads_decisions_with_the_configured_patterns() {
        let dir = test_dir("doc_decisions");
        fs::create_dir_all(dir.join("decisions")).unwrap();
        fs::write(dir.join("decisions/1.md"), "# DEC-1\nState: approved\n")
            .unwrap();
        fs::write(dir.join("decisions/2.md"), "# DEC-2\nState: rejected\n")
            .unwrap();
        let config = TraceConfig {
            decisions: Some(DecisionConfig {
                files: vec!["decisions".to_string()],
                id: "DEC-[0-9]+".to_string(),
                definition: "^# DEC-".to_string(),
                status: "State: (?P<status>[a-z]+)".to_string(),
                accepted: vec!["approved".to_string()],
            }),
            ..default_config()
        };
        let derive = |target: &str| {
            derive_with_base(
                &[definition(), condition("doc-status", &[target])],
                &config,
                &dir,
            )
        };
        assert_eq!(status(&derive("DEC-1")), Some(Status::Pass));
        assert_eq!(status(&derive("DEC-2")), Some(Status::Fail));
        assert_eq!(status(&derive("DEC-3")), Some(Status::Unrun));
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

    /// A definition of `id` in the document of `id`.
    fn definition_of(id: &str) -> Row {
        Row {
            id: id.to_string(),
            ..definition()
        }
    }

    /// A condition `id` of `parent` with `method` naming `targets`.
    fn condition_of(
        id: &str,
        parent: &str,
        method: &str,
        targets: &[&str],
    ) -> Row {
        Row {
            id: id.to_string(),
            parents: vec![parent.to_string()],
            ..condition(method, targets)
        }
    }

    fn condition_status_of<'a>(
        derived: &'a Derived,
        id: &str,
    ) -> Option<&'a ConditionStatus> {
        derived
            .0
            .requirements
            .iter()
            .flat_map(|r| &r.conditions)
            .find(|c| c.id == id)
    }

    #[test]
    fn link_takes_the_worst_status_of_the_named_documents() {
        let dir = test_dir("link_worst");
        write_log(&dir, "libtest.log", "test t::a ... FAILED\n");
        let rows = [
            definition_of("w#FR-1"),
            condition_of("w#VC-1.1", "w#FR-1", "libtest", &["t::a"]),
            definition_of("l#FR-1"),
            condition_of("l#VC-1.1", "l#FR-1", "link", &["w"]),
        ];
        let derived = derive_with_base(&rows, &default_config(), &dir);
        let linked = condition_status_of(&derived, "l#VC-1.1").unwrap();
        assert_eq!(linked.status, Status::Fail);
        assert_eq!(linked.items.len(), 1);
        assert!(!derived.0.passes());
    }

    #[test]
    fn link_takes_the_worst_status_of_the_named_requirements() {
        let dir = test_dir("link_req");
        write_log(&dir, "libtest.log", "test t::a ... FAILED\n");
        let rows = [
            definition_of("w#FR-1"),
            condition_of("w#VC-1.1", "w#FR-1", "libtest", &["t::a"]),
            definition_of("l#FR-1"),
            condition_of("l#VC-1.1", "l#FR-1", "link", &["w#FR-1"]),
        ];
        let derived = derive_with_base(&rows, &default_config(), &dir);
        let linked = condition_status_of(&derived, "l#VC-1.1").unwrap();
        assert_eq!(linked.status, Status::Fail);
        assert_eq!(linked.items.len(), 1);
        assert!(!derived.0.passes());
    }

    #[test]
    fn condition_without_doc_qualifier_still_fails_the_gate() {
        let dir = test_dir("unqualified_condition");
        write_log(&dir, "libtest.log", "test t::ok ... ok\n");
        let rows = [
            definition_of("w#FR-1"),
            condition_of("w#VC-1.1", "w#FR-1", "libtest", &["t::ok"]),
            // Hand-authored / legacy `jsonl` may omit `<doc>#`; the graph skips
            // that id, but the gate must still count the leaf `Unrun`.
            condition_of("VC-9.9", "w#FR-1", "libtest", &["missing::t"]),
        ];
        let derived = derive_with_base(&rows, &default_config(), &dir);
        let omitted = condition_status_of(&derived, "VC-9.9").unwrap();
        assert_eq!(omitted.status, Status::Unrun);
        let parent = derived
            .0
            .requirements
            .iter()
            .find(|r| r.id == "w#FR-1")
            .unwrap();
        assert_eq!(parent.status, Status::Unrun);
        assert!(!derived.0.passes());
    }

    #[test]
    fn kani_target_passes_with_crate_qualified_or_bare_name() {
        let log = "Checking harness proofs::h...\n - Status: SUCCESS\n** 1 of 1 cover properties satisfied\nVERIFICATION:- SUCCESSFUL\n";
        for target in ["proofs::h", "my_crate::proofs::h"] {
            let derived = run("kani_strip", log, condition("kani", &[target]));
            assert_eq!(status(&derived), Some(Status::Pass), "{target}");
            assert!(derived.0.passes());
            assert!(derived.1.is_empty(), "{:?}", derived.1);
        }
    }

    fn gate_run(
        name: &str,
        outcomes: &[GateOutcome],
        targets: &[&str],
    ) -> Derived {
        let dir = test_dir(name);
        let out_dir = dir.join("target/ci-artifacts");
        fs::create_dir_all(&out_dir).unwrap();
        for o in outcomes {
            o.save_to_dir(&out_dir).unwrap();
        }
        derive_with_base(
            &[definition(), condition("gate", targets)],
            &default_config(),
            &dir,
        )
    }

    #[test]
    fn gate_target_passes_on_verdict_pass() {
        let outcome = GateOutcome {
            schema: crate::gate::OUTCOME_SCHEMA,
            gate: "fmt".to_string(),
            verdict: Verdict::Pass,
            exit_code: Some(0),
            duration_secs: 0.1,
            summary: Some("Formatted cleanly".to_string()),
            log_file: "fmt.log".to_string(),
        };
        let derived = gate_run("gate_pass", &[outcome], &["fmt"]);
        assert_eq!(status(&derived), Some(Status::Pass));
        assert!(derived.0.passes());
        assert!(derived.1.is_empty(), "{:?}", derived.1);
    }

    #[test]
    fn gate_target_fails_on_fail_skipped_or_warn() {
        for (name, verdict) in [
            ("gate_fail", Verdict::Fail),
            ("gate_skip", Verdict::Skipped),
            ("gate_warn", Verdict::Warn),
        ] {
            let outcome = GateOutcome {
                schema: crate::gate::OUTCOME_SCHEMA,
                gate: "clippy".to_string(),
                verdict,
                exit_code: Some(1),
                duration_secs: 0.5,
                summary: None,
                log_file: "clippy.log".to_string(),
            };
            let derived = gate_run(name, &[outcome], &["clippy"]);
            assert_eq!(status(&derived), Some(Status::Fail), "{verdict:?}");
            assert!(!derived.0.passes());
            assert_eq!(derived.1.len(), 1, "{:?}", derived.1);
        }
    }

    #[test]
    fn gate_target_missing_artifact_is_unrun() {
        let derived = gate_run("gate_missing", &[], &["clippy"]);
        assert_eq!(status(&derived), Some(Status::Unrun));
        assert!(!derived.0.passes());
        assert_eq!(derived.1.len(), 1);
        assert!(
            derived
                .1
                .first()
                .unwrap()
                .message
                .contains("clippy.result.json is missing")
        );
    }

    #[test]
    fn gate_target_invalid_schema_or_corrupt_fails() {
        let dir = test_dir("gate_invalid");
        let out_dir = dir.join("target/ci-artifacts");
        fs::create_dir_all(&out_dir).unwrap();
        fs::write(out_dir.join("corrupt.result.json"), "not json").unwrap();
        let wrong_schema = GateOutcome {
            schema: 999,
            gate: "bad_schema".to_string(),
            verdict: Verdict::Pass,
            exit_code: Some(0),
            duration_secs: 0.1,
            summary: None,
            log_file: "bad.log".to_string(),
        };
        wrong_schema.save_to_dir(&out_dir).unwrap();

        let derived = derive_with_base(
            &[definition(), condition("gate", &["corrupt", "bad_schema"])],
            &default_config(),
            &dir,
        );
        assert_eq!(status(&derived), Some(Status::Fail));
        assert!(!derived.0.passes());
        assert_eq!(derived.1.len(), 2);
    }

    #[test]
    fn derive_with_graph_and_render_graph_share_link_graph() {
        let rows = [
            definition(),
            Row {
                schema: SCHEMA,
                id: "w#VC-1.1".to_string(),
                kind: CONDITION.to_string(),
                file: "w.md".to_string(),
                line: 2,
                text: "link".to_string(),
                parents: vec!["w#FR-1".to_string()],
                method: Some("link".to_string()),
                targets: vec!["w#FR-1".to_string()],
            },
        ];
        let (graph, defects) = link_graph(&rows);
        assert_eq!(graph.rows().len(), 2);
        let derived = derive_with_graph(
            &graph,
            defects,
            &default_config(),
            Path::new("."),
        );
        assert_eq!(derived.0.requirements.len(), 1);
        let rendered = render_graph(&graph, &derived.0, GraphFormat::Mermaid);
        assert!(rendered.starts_with("flowchart TD\n"));
    }
}
