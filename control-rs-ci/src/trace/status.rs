//! Requirement and condition status derivation.
//!
//! Each condition takes a status:
//! - `Pass`: method in `marked_methods`, marked items exist, all passed.
//! - `Fail`: method in `marked_methods`, any marked item failed (fails gate).
//! - `Unrun`: method in `marked_methods`, no result recorded for item (fails gate).
//! - `Uncovered`: method in `marked_methods` and no marker in source text (fails gate).
//! - `Review`: method outside `marked_methods`; awaits sign-off.
//!
//! A requirement takes the worst status among its conditions in order:
//! `Fail` > `Unrun` > `Uncovered` > `Review` > `Pass`.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};

use super::reqs::TraceConfig;
use super::{
    CONDITION, DEFINITION, Defect, Defects, MARKER, Row, SCHEMA, read_text,
    write_file,
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

/// Marker rows by the qualified ID they name.
type MarkersById<'a> = BTreeMap<&'a str, Vec<&'a Row>>;

/// Satisfied and total cover properties count.
type CoverSummary = (usize, usize);

/// Map from method name to test/harness outcomes.
type MethodResultMap = HashMap<String, HashMap<String, bool>>;

/// Cache of file lines for marked item extraction.
type FileLineCache = HashMap<String, Vec<String>>;

/// Map of test/harness identifier to pass/fail.
type TestResultMap = HashMap<String, bool>;

/// Map of verification method to compiled item regex rule.
type ItemRules = HashMap<String, Regex>;

/// Evaluated conditions and pending reviews.
type ConditionEvalResult = (Vec<ConditionStatus>, Vec<ReviewCondition>);

/// Resolved item identifier and metadata.
type ResolvedItem = (String, Option<MarkedItemInfo>);

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
    /// All marked items passed verification.
    Pass,
    /// Method outside `marked_methods`; awaits sign-off.
    Review,
    /// Method in `marked_methods` and no marker; fails the gate.
    Uncovered,
    /// Method in `marked_methods` and no result recorded; fails the gate.
    Unrun,
    /// Method in `marked_methods` and verification failed; fails the gate.
    Fail,
}

/// Location of a test marker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkerLocation {
    /// Path relative to the working directory.
    pub file: String,
    /// 1-based line number of the marker span.
    pub line: usize,
}

/// Verification outcome of one marked item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItemOutcome {
    /// Source file of the marked item.
    pub file: String,
    /// 1-based line number of the marker.
    pub line: usize,
    /// Item identifier / function name.
    pub item: String,
    /// Whether the item passed verification.
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
    /// Number of matching test markers found in source text.
    pub marker_count: usize,
    /// Locations of matching markers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub markers: Vec<MarkerLocation>,
    /// Per-item verification outcomes.
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
    /// Warning code (`W-1` or `W-2`).
    pub code: String,
    /// Source file path.
    pub file: String,
    /// 1-based line number.
    pub line: usize,
    /// Item identifier.
    pub item: String,
    /// Warning description.
    pub message: String,
}

/// Extracted metadata from the source item marked by `#[req]`.
#[derive(Debug, Clone)]
struct MarkedItemInfo {
    name: String,
    attributes: String,
    has_miri_ignore: bool,
}

/// Target marked item to evaluate.
struct EvalTarget<'a> {
    mark: &'a Row,
    method: &'a str,
    row_id: &'a str,
    has_artifact: bool,
}

/// Context for condition status derivation across a trace run.
struct StatusCtx<'a> {
    config: &'a TraceConfig,
    base: &'a Path,
    file_cache: &'a mut FileLineCache,
    result_cache: &'a MethodResultMap,
    miri_results: Option<&'a TestResultMap>,
    item_rules: &'a ItemRules,
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
    /// Marker rows that could not be resolved or name a requirement.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unresolved_markers: Vec<Row>,
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
    /// Whether no condition/requirement is `Uncovered`, `Unrun`, or `Fail`,
    /// and all markers are valid.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.unresolved_markers.is_empty()
            && self.counts.get(&Status::Fail).copied().unwrap_or(0) == 0
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

fn inspect_marked_item(
    lines: &[&str],
    line_1_based: usize,
) -> Option<MarkedItemInfo> {
    if lines.is_empty() || line_1_based == 0 || line_1_based > lines.len() {
        return None;
    }
    let start_idx = line_1_based.saturating_sub(1);
    let mut bal = 0isize;
    let mut end_idx = start_idx;
    for (i, &line) in lines.iter().enumerate().skip(start_idx).take(16) {
        end_idx = i;
        bal = bal.saturating_add(line.chars().fold(0, |b, c| match c {
            '(' => b.saturating_add(1),
            ')' => b.saturating_sub(1),
            _ => b,
        }));
        if bal <= 0 && i >= start_idx {
            break;
        }
    }

    let mut attrs = collect_preceding_attrs(lines, start_idx);
    let fn_name = find_fn_and_trailing_attrs(lines, end_idx, &mut attrs);

    let all_attrs = attrs.join("\n");
    let has_miri_ignore = all_attrs.contains("miri, ignore")
        || all_attrs.contains("cfg_attr(miri, ignore)");
    fn_name.map(|name| MarkedItemInfo {
        name,
        attributes: all_attrs,
        has_miri_ignore,
    })
}

fn collect_preceding_attrs<'a>(
    lines: &'a [&str],
    start_idx: usize,
) -> Vec<&'a str> {
    let mut attrs = Vec::new();
    let mut above_idx = start_idx;
    while above_idx > 0 {
        let prev_idx = above_idx.saturating_sub(1);
        let Some(prev_line) = lines.get(prev_idx) else {
            break;
        };
        let prev = prev_line.trim();
        if prev.starts_with("#[") || prev.ends_with(']') {
            attrs.insert(0, prev);
            above_idx = prev_idx;
        } else {
            break;
        }
    }
    attrs
}

fn find_fn_and_trailing_attrs<'a>(
    lines: &'a [&str],
    end_idx: usize,
    attrs: &mut Vec<&'a str>,
) -> Option<String> {
    for &line in lines.iter().skip(end_idx.saturating_add(1)).take(32) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("#[") || trimmed.ends_with(']') {
            attrs.push(trimmed);
            continue;
        }
        if let Some(idx) = trimmed.find("fn ") {
            let after_idx = idx.saturating_add(3);
            let after_fn = trimmed.get(after_idx..).unwrap_or("").trim_start();
            let ident: String = after_fn
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() {
                return Some(ident);
            }
        }
    }
    None
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

/// The markers that name a requirement or no defined condition, with one
/// defect each.
fn marker_defects(
    marks: &[Row],
    definitions: &ById<'_>,
    conditions: &ById<'_>,
    defects: &mut Defects,
) -> Vec<Row> {
    let mut unresolved = Vec::new();
    for mark in marks.iter().filter(|m| m.kind == MARKER) {
        let message = if definitions.contains_key(mark.id.as_str()) {
            format!(
                "marker {} names a requirement; markers name verification \
                 conditions",
                mark.id
            )
        } else if conditions.contains_key(mark.id.as_str()) {
            continue;
        } else {
            format!("marker {} names no defined condition", mark.id)
        };
        defects.push(Defect::at(mark, message));
        unresolved.push(mark.clone());
    }
    unresolved
}

fn validate_item_rule(
    target: &EvalTarget<'_>,
    item_name: &str,
    item_info: Option<&MarkedItemInfo>,
    ctx: &mut StatusCtx<'_>,
) -> bool {
    let Some(rule) = ctx.item_rules.get(target.method) else {
        return true;
    };
    let attrs = item_info.map_or("", |info| info.attributes.as_str());
    if rule.is_match(attrs) {
        true
    } else {
        ctx.defects.push(Defect::at(
            target.mark,
            format!(
                "marked item '{item_name}' does not match item rule for method '{}'",
                target.method
            ),
        ));
        false
    }
}

fn check_miri_log_coverage(
    target: &EvalTarget<'_>,
    item_name: &str,
    ctx: &mut StatusCtx<'_>,
) {
    if target.method != "test" {
        return;
    }
    let Some(miri_map) = ctx.miri_results else {
        return;
    };
    let in_miri = miri_map.iter().any(|(k, _)| {
        k.as_str() == item_name || k.ends_with(&format!("::{item_name}"))
    });
    if !in_miri {
        ctx.warnings.push(InterpreterWarning {
            code: "W-2".to_string(),
            file: target.mark.file.clone(),
            line: target.mark.line,
            item: item_name.to_string(),
            message: "marked test item was executed under test.log but is absent from miri.log"
                .to_string(),
        });
    }
}

fn check_log_outcome(
    target: &EvalTarget<'_>,
    item_name: &str,
    results: &TestResultMap,
    ctx: &mut StatusCtx<'_>,
) -> ItemOutcome {
    let found = results.iter().find(|(k, _)| {
        k.as_str() == item_name || k.ends_with(&format!("::{item_name}"))
    });

    match found {
        Some((_, true)) => {
            check_miri_log_coverage(target, item_name, ctx);
            ItemOutcome {
                file: target.mark.file.clone(),
                line: target.mark.line,
                item: item_name.to_string(),
                passed: true,
                message: None,
            }
        }
        Some((_, false)) => {
            ctx.defects.push(Defect::at(
                target.mark,
                format!(
                    "{} marked item '{item_name}' failed verification",
                    target.row_id
                ),
            ));
            ItemOutcome {
                file: target.mark.file.clone(),
                line: target.mark.line,
                item: item_name.to_string(),
                passed: false,
                message: Some("verification failed in result log".to_string()),
            }
        }
        None => {
            ctx.defects.push(Defect::at(
                target.mark,
                format!(
                    "{} marked item '{item_name}' has not run",
                    target.row_id
                ),
            ));
            ItemOutcome {
                file: target.mark.file.clone(),
                line: target.mark.line,
                item: item_name.to_string(),
                passed: false,
                message: Some("no result recorded in result log".to_string()),
            }
        }
    }
}

fn resolve_marked_item(
    target: &EvalTarget<'_>,
    ctx: &mut StatusCtx<'_>,
) -> ResolvedItem {
    let mark = target.mark;
    let lines_owned =
        ctx.file_cache.entry(mark.file.clone()).or_insert_with(|| {
            let path = ctx.base.join(&mark.file);
            read_text(&path)
                .map(|text| text.lines().map(String::from).collect())
                .unwrap_or_default()
        });
    let line_refs: Vec<&str> = lines_owned.iter().map(String::as_str).collect();

    let item_info = inspect_marked_item(&line_refs, mark.line).or_else(|| {
        let mark_lines: Vec<&str> = mark.text.lines().collect();
        inspect_marked_item(&mark_lines, 1)
    });

    let item_name = item_info.as_ref().map_or_else(
        || format!("item_at_line_{}", mark.line),
        |info| info.name.clone(),
    );

    (item_name, item_info)
}

fn evaluate_marked_item(
    target: &EvalTarget<'_>,
    ctx: &mut StatusCtx<'_>,
) -> ItemOutcome {
    let mark = target.mark;
    let (item_name, item_info) = resolve_marked_item(target, ctx);
    let rule_matches =
        validate_item_rule(target, &item_name, item_info.as_ref(), ctx);

    if target.method == "test"
        && item_info.as_ref().is_some_and(|info| info.has_miri_ignore)
    {
        ctx.warnings.push(InterpreterWarning {
            code: "W-1".to_string(),
            file: mark.file.clone(),
            line: mark.line,
            item: item_name.clone(),
            message: "marked test item contains cfg_attr(miri, ignore)"
                .to_string(),
        });
    }

    if !rule_matches {
        return ItemOutcome {
            file: mark.file.clone(),
            line: mark.line,
            item: item_name.clone(),
            passed: false,
            message: Some(format!(
                "marked item '{item_name}' does not match item rule for method '{}'",
                target.method
            )),
        };
    }

    if !target.has_artifact {
        return ItemOutcome {
            file: mark.file.clone(),
            line: mark.line,
            item: item_name,
            passed: true,
            message: None,
        };
    }

    let Some(results) = ctx.result_cache.get(target.method) else {
        ctx.defects.push(Defect::at(
            mark,
            format!(
                "{} marked item '{item_name}' has not run (log missing)",
                target.row_id
            ),
        ));
        return ItemOutcome {
            file: mark.file.clone(),
            line: mark.line,
            item: item_name,
            passed: false,
            message: Some("result log missing".to_string()),
        };
    };

    check_log_outcome(target, &item_name, results, ctx)
}

fn review_condition_status(
    row: &Row,
    markers: &[&Row],
    method: String,
) -> ConditionStatus {
    ConditionStatus {
        id: row.id.clone(),
        parents: row.parents.clone(),
        method,
        status: Status::Review,
        marker_count: markers.len(),
        markers: markers
            .iter()
            .map(|m| MarkerLocation {
                file: m.file.clone(),
                line: m.line,
            })
            .collect(),
        items: Vec::new(),
    }
}

fn uncovered_condition_status(
    row: &Row,
    method: String,
    ctx: &mut StatusCtx<'_>,
) -> ConditionStatus {
    let msg = if method == "proof" {
        format!("{} has no marked proof harness", row.id)
    } else {
        format!("{} has no marked test", row.id)
    };
    ctx.defects.push(Defect::at(row, msg));
    ConditionStatus {
        id: row.id.clone(),
        parents: row.parents.clone(),
        method,
        status: Status::Uncovered,
        marker_count: 0,
        markers: Vec::new(),
        items: Vec::new(),
    }
}

fn condition_status(
    row: &Row,
    markers: &[&Row],
    ctx: &mut StatusCtx<'_>,
) -> ConditionStatus {
    let method = row.method.clone().unwrap_or_default();
    if !ctx.config.marked_methods.contains(&method) {
        return review_condition_status(row, markers, method);
    }

    if markers.is_empty() {
        return uncovered_condition_status(row, method, ctx);
    }

    let has_artifact = ctx
        .config
        .method
        .get(&method)
        .and_then(|m| m.result_artifact.as_ref())
        .is_some();

    let items: Vec<ItemOutcome> = markers
        .iter()
        .map(|mark| {
            let target = EvalTarget {
                mark,
                method: &method,
                row_id: &row.id,
                has_artifact,
            };
            evaluate_marked_item(&target, ctx)
        })
        .collect();

    let status = if items.iter().any(|i| {
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

    ConditionStatus {
        id: row.id.clone(),
        parents: row.parents.clone(),
        method,
        status,
        marker_count: markers.len(),
        markers: markers
            .iter()
            .map(|m| MarkerLocation {
                file: m.file.clone(),
                line: m.line,
            })
            .collect(),
        items,
    }
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

fn load_method_artifacts(config: &TraceConfig, base: &Path) -> MethodResultMap {
    let mut result_cache = HashMap::new();
    for (method, mcfg) in &config.method {
        let Some(artifact_rel) = &mcfg.result_artifact else {
            continue;
        };
        let artifact_path = base.join(artifact_rel);
        if !artifact_path.exists() {
            result_cache.insert(method.clone(), HashMap::new());
            continue;
        }
        let Ok(text) = read_text(&artifact_path) else {
            continue;
        };
        if method == "test" {
            result_cache.insert(method.clone(), parse_libtest_results(&text));
        } else if method == "proof" {
            result_cache.insert(method.clone(), parse_kani_results(&text));
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

fn compile_item_rules(config: &TraceConfig) -> ItemRules {
    let mut item_rules = HashMap::new();
    for (method, mcfg) in &config.method {
        if let Some(re) = mcfg
            .item_rule
            .as_deref()
            .and_then(|pat| Regex::new(pat).ok())
        {
            item_rules.insert(method.clone(), re);
        }
    }
    item_rules
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
    markers: &MarkersById<'_>,
    ctx: &mut StatusCtx<'_>,
    counts: &mut StatusCounts,
) -> ConditionEvalResult {
    let (conditions, condition_order) = cond_idx;
    let mut evaluated = Vec::new();
    let mut review = Vec::new();

    for row in condition_order.iter().filter_map(|id| conditions.get(id)) {
        let found = markers.get(row.id.as_str()).map_or(&[][..], Vec::as_slice);
        let status = condition_status(row, found, ctx);

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

/// Derives condition and requirement status from the rows of `trace-reqs`
/// and `trace-marks`, using current working directory as base.
#[must_use]
pub fn derive(reqs: &[Row], marks: &[Row], config: &TraceConfig) -> Derived {
    derive_with_base(reqs, marks, config, Path::new("."))
}

/// Derives condition and requirement status relative to `base`.
#[must_use]
pub fn derive_with_base(
    reqs: &[Row],
    marks: &[Row],
    config: &TraceConfig,
    base: &Path,
) -> Derived {
    let (definitions, order) = first_by_id(reqs, DEFINITION);
    let cond_idx = first_by_id(reqs, CONDITION);
    let mut markers = MarkersById::new();
    for mark in marks.iter().filter(|m| m.kind == MARKER) {
        markers.entry(mark.id.as_str()).or_default().push(mark);
    }

    let mut defects = Vec::new();
    let mut warnings = Vec::new();
    let unresolved_markers =
        marker_defects(marks, &definitions, &cond_idx.0, &mut defects);

    let result_cache = load_method_artifacts(config, base);
    let miri_results = load_miri_log(base);
    let item_rules = compile_item_rules(config);
    let mut counts = initial_status_counts();
    let mut file_cache: FileLineCache = HashMap::new();

    let mut ctx = StatusCtx {
        config,
        base,
        file_cache: &mut file_cache,
        result_cache: &result_cache,
        miri_results: miri_results.as_ref(),
        item_rules: &item_rules,
        defects: &mut defects,
        warnings: &mut warnings,
    };

    let (evaluated, review) =
        evaluate_all_conditions(&cond_idx, &markers, &mut ctx, &mut counts);

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
        unresolved_markers,
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

    const PFX: &str = concat!("#[", "req(",);

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
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn default_config() -> TraceConfig {
        let mut method = BTreeMap::new();
        method.insert(
            "test".to_string(),
            MethodConfig {
                item_rule: Some(r"#\[test\]|#\[tokio::test\]".to_string()),
                result_artifact: None,
            },
        );
        method.insert(
            "proof".to_string(),
            MethodConfig {
                item_rule: Some(
                    r"#\[kani::(?:proof|proof_for_contract)\]".to_string(),
                ),
                result_artifact: None,
            },
        );
        TraceConfig {
            id: r"(?:FR|NFR|C)-[0-9]+[a-z]?".to_string(),
            condition: r"VC-[0-9]+(?:\.[0-9]+[a-z]?)?".to_string(),
            doc: "[a-z0-9-]+".to_string(),
            files: vec!["docs".to_string()],
            doc_suffix: "-design".to_string(),
            definition: r"^- \*\*(?:FR|NFR|C)-".to_string(),
            verification: r"^\| *(?:[a-z0-9-]+#)?VC-".to_string(),
            methods: vec![
                "test".to_string(),
                "proof".to_string(),
                "analysis".to_string(),
                "inspection".to_string(),
                "review".to_string(),
            ],
            marked_methods: vec!["test".to_string(), "proof".to_string()],
            method,
            retired: vec![],
            markers: None,
        }
    }

    fn row(id: &str, kind: &str, text: &str) -> Row {
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: kind.to_string(),
            parents: Vec::new(),
            method: None,
            file: "docs/w-design.md".to_string(),
            line: 1,
            text: text.to_string(),
        }
    }

    fn condition_row(id: &str, parent: &str, method: &str, text: &str) -> Row {
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: CONDITION.to_string(),
            parents: vec![parent.to_string()],
            method: Some(method.to_string()),
            file: "docs/w-design.md".to_string(),
            line: 1,
            text: text.to_string(),
        }
    }

    #[test]
    fn marked_test_condition_is_pass_when_no_artifact_configured() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let mark_text =
            format!("{PFX}\"w#VC-1.1\")]\n#[test]\nfn test_foo() {{}}");
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let config = default_config();
        let (report, defects) = derive(&reqs, &marks, &config);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Pass)
        );
        assert!(report.passes());
        assert!(defects.is_empty());
        assert_eq!(report.counts.get(&Status::Pass), Some(&1));
    }

    #[test]
    fn marked_test_condition_passes_with_libtest_ok_artifact() {
        let dir = test_dir("libtest_ok");
        let log_dir = dir.join("target/ci-artifacts");
        fs::create_dir_all(&log_dir).unwrap();
        fs::write(
            log_dir.join("test.log"),
            "running 1 test\ntest widget::tests::test_foo ... ok\n\ntest result: ok. 1 passed;\n",
        )
        .unwrap();

        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let mark_text =
            format!("{PFX}\"w#VC-1.1\")]\n#[test]\nfn test_foo() {{}}");
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let mut config = default_config();
        config.method.get_mut("test").unwrap().result_artifact =
            Some("target/ci-artifacts/test.log".to_string());

        let (report, defects) = derive_with_base(&reqs, &marks, &config, &dir);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Pass)
        );
        assert!(report.passes());
        assert!(defects.is_empty());
        assert_eq!(report.counts.get(&Status::Pass), Some(&1));
    }

    #[test]
    fn marked_test_condition_fails_when_libtest_reports_failed() {
        let dir = test_dir("libtest_fail");
        let log_dir = dir.join("target/ci-artifacts");
        fs::create_dir_all(&log_dir).unwrap();
        fs::write(
            log_dir.join("test.log"),
            "running 1 test\ntest widget::tests::test_foo ... FAILED\n\ntest result: FAILED. 0 passed; 1 failed;\n",
        )
        .unwrap();

        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let mark_text =
            format!("{PFX}\"w#VC-1.1\")]\n#[test]\nfn test_foo() {{}}");
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let mut config = default_config();
        config.method.get_mut("test").unwrap().result_artifact =
            Some("target/ci-artifacts/test.log".to_string());

        let (report, defects) = derive_with_base(&reqs, &marks, &config, &dir);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Fail)
        );
        assert!(!report.passes());
        assert!(!defects.is_empty());
        assert_eq!(report.counts.get(&Status::Fail), Some(&1));
    }

    #[test]
    fn marked_test_condition_unrun_when_not_in_test_log() {
        let dir = test_dir("libtest_unrun");
        let log_dir = dir.join("target/ci-artifacts");
        fs::create_dir_all(&log_dir).unwrap();
        fs::write(
            log_dir.join("test.log"),
            "running 1 test\ntest other_test ... ok\n",
        )
        .unwrap();

        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let mark_text =
            format!("{PFX}\"w#VC-1.1\")]\n#[test]\nfn test_foo() {{}}");
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let mut config = default_config();
        config.method.get_mut("test").unwrap().result_artifact =
            Some("target/ci-artifacts/test.log".to_string());

        let (report, defects) = derive_with_base(&reqs, &marks, &config, &dir);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Unrun)
        );
        assert!(!report.passes());
        assert!(!defects.is_empty());
        assert_eq!(report.counts.get(&Status::Unrun), Some(&1));
    }

    #[test]
    fn kani_proof_condition_passes_when_verification_successful() {
        let dir = test_dir("kani_ok");
        let log_dir = dir.join("target/ci-artifacts");
        fs::create_dir_all(&log_dir).unwrap();
        fs::write(
            log_dir.join("kani.log"),
            "Checking harness check_saturating...\nRESULTS:\nCheck 1: ok\n - Status: SUCCESS\n** 1 of 1 cover properties satisfied\nVERIFICATION:- SUCCESSFUL\n",
        )
        .unwrap();

        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "proof",
                "| VC-1.1 | FR-1 | `proof` | Kani proof |",
            ),
        ];
        let mark_text = format!(
            "{PFX}\"w#VC-1.1\")]\n#[kani::proof]\nfn check_saturating() {{}}"
        );
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let mut config = default_config();
        config.method.get_mut("proof").unwrap().result_artifact =
            Some("target/ci-artifacts/kani.log".to_string());

        let (report, defects) = derive_with_base(&reqs, &marks, &config, &dir);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Pass)
        );
        assert!(report.passes());
        assert!(defects.is_empty());
        assert_eq!(report.counts.get(&Status::Pass), Some(&1));
    }

    #[test]
    fn kani_proof_condition_fails_when_cover_unsatisfied() {
        let dir = test_dir("kani_unsat");
        let log_dir = dir.join("target/ci-artifacts");
        fs::create_dir_all(&log_dir).unwrap();
        fs::write(
            log_dir.join("kani.log"),
            "Checking harness check_saturating...\nRESULTS:\nCheck 1: cover\n - Status: UNSATISFIABLE\n** 0 of 1 cover properties satisfied\nVERIFICATION:- SUCCESSFUL\n",
        )
        .unwrap();

        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "proof",
                "| VC-1.1 | FR-1 | `proof` | Kani proof |",
            ),
        ];
        let mark_text = format!(
            "{PFX}\"w#VC-1.1\")]\n#[kani::proof]\nfn check_saturating() {{}}"
        );
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let mut config = default_config();
        config.method.get_mut("proof").unwrap().result_artifact =
            Some("target/ci-artifacts/kani.log".to_string());

        let (report, defects) = derive_with_base(&reqs, &marks, &config, &dir);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Fail)
        );
        assert!(!report.passes());
        assert!(!defects.is_empty());
    }

    #[test]
    fn item_rule_mismatch_produces_defect() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let mark_text = format!("{PFX}\"w#VC-1.1\")]\nfn regular_fn() {{}}");
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let config = default_config();
        let (report, defects) = derive(&reqs, &marks, &config);

        assert!(!report.passes());
        assert!(defects.iter().any(|d| d.message.contains("item rule")));
    }

    #[test]
    fn miri_warnings_w1_and_w2_emitted() {
        let dir = test_dir("miri_warn");
        let log_dir = dir.join("target/ci-artifacts");
        fs::create_dir_all(&log_dir).unwrap();
        fs::write(
            log_dir.join("test.log"),
            "running 1 test\ntest widget::tests::test_foo ... ok\n",
        )
        .unwrap();
        // miri.log exists but does not have `test_foo`
        fs::write(log_dir.join("miri.log"), "running 0 tests\n").unwrap();

        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let mark_text = format!(
            "{PFX}\"w#VC-1.1\")]\n#[cfg_attr(miri, ignore)]\n#[test]\nfn test_foo() {{}}"
        );
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let mut config = default_config();
        config.method.get_mut("test").unwrap().result_artifact =
            Some("target/ci-artifacts/test.log".to_string());

        let (report, _) = derive_with_base(&reqs, &marks, &config, &dir);

        assert!(report.warnings.iter().any(|w| w.code == "W-1"));
        assert!(report.warnings.iter().any(|w| w.code == "W-2"));
    }

    #[test]
    fn unmarked_test_condition_is_uncovered_and_fails() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let config = default_config();
        let (report, defects) = derive(&reqs, &[], &config);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Uncovered)
        );
        assert!(!report.passes());
        assert_eq!(defects.len(), 1);
        assert_eq!(
            defects.first().map(|d| d.message.as_str()),
            Some("w#VC-1.1 has no marked test")
        );
        assert_eq!(report.counts.get(&Status::Uncovered), Some(&1));
    }

    #[test]
    fn review_condition_takes_review_status() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "review",
                "| VC-1.1 | FR-1 | `review` | Manual inspection |",
            ),
        ];
        let config = default_config();
        let (report, defects) = derive(&reqs, &[], &config);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Review)
        );
        assert!(report.passes());
        assert!(defects.is_empty());
        assert_eq!(report.counts.get(&Status::Review), Some(&1));
        assert_eq!(report.review.len(), 1);
    }

    #[test]
    fn marker_naming_requirement_is_a_defect() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let marks = [row("w#FR-1", MARKER, "fn t")];
        let config = default_config();
        let (report, defects) = derive(&reqs, &marks, &config);

        assert!(!report.passes());
        assert!(
            defects
                .iter()
                .any(|d| d.message.contains("names a requirement"))
        );
    }

    #[test]
    fn marker_naming_undefined_condition_is_a_defect() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let marks = [row("w#VC-9.9", MARKER, "fn t")];
        let config = default_config();
        let (report, defects) = derive(&reqs, &marks, &config);

        assert!(!report.passes());
        assert!(
            defects
                .iter()
                .any(|d| { d.message.contains("names no defined condition") })
        );
    }
}
