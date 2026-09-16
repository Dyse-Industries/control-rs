//! Requirement linker associating design document requirements with source annotations.
//!
//! Evaluates bidirectional association between Section 2 requirement declarations and
//! Section 6.4 verification evidence, resolving symbolic locators against function doc comments
//! (`/// Trace: ...`) and test execution results.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;
use std::path::Path;

use super::doc_parser::{
    DeclaredRequirement, ParsedDesignDoc, TraceabilityRow, doc_slug,
};
use super::results::TestRunOutcome;
use super::source_scanner::{OrphanTest, SourceAnnotation, SourceScanResult};

type OutcomesDyn<'a> = &'a dyn OutcomesLookup;
type OutcomesRef<'a> = Option<OutcomesDyn<'a>>;
type ReqSetIndex = HashMap<String, HashSet<String>>;
type TraceByReq<'a> = HashMap<String, Vec<&'a TraceabilityRow>>;

/// Requirement verification status according to the Section 6.3 lattice.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum RequirementStatus {
    /// Requirement is traced and all associated verification artifacts succeeded.
    Verified,
    /// Requirement is traced but an associated test or gate failed.
    Failed,
    /// Requirement is traced but no execution outcome was recorded in this run.
    NotRun,
    /// Requirement is explicitly deferred or listed under Section 6.7 Not verified.
    Unverified,
    /// Requirement artifact locator could not be resolved to a valid repository artifact.
    Unresolved,
    /// Requirement is declared in Section 2 but missing from Section 6.4 and 6.7.
    Missing,
}

/// A specific defect detected in the traceability matrix.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TraceDefect {
    /// A requirement declared in §2 appears in neither §6.4 nor §6.7.
    MissingRequirement {
        /// Owning document.
        doc: String,
        /// Requirement ID.
        req_id: String,
        /// Requirement title.
        title: String,
        /// Whether owning document has Approved or Reviewed status.
        is_approved: bool,
    },
    /// A requirement ID referenced in §6.4 or source comments is not declared in §2.
    UndeclaredRequirement {
        /// Owning document or file.
        location: String,
        /// Undeclared requirement ID.
        req_id: String,
    },
    /// An artifact locator in §6.4 could not be resolved.
    UnresolvedArtifact {
        /// Owning document.
        doc: String,
        /// Requirement ID.
        req_id: String,
        /// The unresolved artifact string.
        locator: String,
        /// Whether owning document has Approved or Reviewed status.
        is_approved: bool,
    },
    /// A test function lacking requirement doc comment annotations.
    OrphanTest(OrphanTest),
}

/// Trace report entry for an individual requirement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TracedRequirementItem {
    /// Owning design document.
    pub doc_path: String,
    /// Requirement identifier, e.g. "FR-1".
    pub req_id: String,
    /// Requirement title from Section 2.
    pub title: String,
    /// Declared verification methods.
    pub methods: Vec<String>,
    /// Artifact locators from Section 6.4.
    pub locators: Vec<String>,
    /// Matching source functions found in the codebase.
    pub source_functions: Vec<String>,
    /// Final verification status.
    pub status: RequirementStatus,
}

/// Complete summary of the traceability analysis.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TraceMatrixSummary {
    /// Total number of unique requirements across all design docs.
    pub total_requirements: usize,
    /// Total number of design documents scanned.
    pub total_documents: usize,
    /// Requirements verified by passing tests or valid inspections.
    pub verified_count: usize,
    /// Requirements where associated tests failed.
    pub failed_count: usize,
    /// Requirements explicitly declared unverified in Section 6.7.
    pub unverified_count: usize,
    /// Requirements traced but not executed in this run.
    pub not_run_count: usize,
    /// Requirements with unresolved locators.
    pub unresolved_count: usize,
    /// Unresolved requirements in Approved or Reviewed documents.
    pub approved_unresolved_count: usize,
    /// Requirements declared in §2 but missing from §6.4 and §6.7.
    pub missing_count: usize,
    /// Missing requirements in Approved or Reviewed documents.
    pub approved_missing_count: usize,
    /// Total number of recorded test outcomes parsed.
    pub recorded_outcomes_count: usize,
    /// Tests lacking requirement doc comments.
    pub orphan_test_count: usize,
    /// All detected defects.
    pub defects: Vec<TraceDefect>,
    /// Per-requirement item details.
    pub items: Vec<TracedRequirementItem>,
}

struct AnalysisCtx<'a> {
    scan: &'a SourceScanResult,
    repo_root: &'a Path,
    test_outcomes: OutcomesRef<'a>,
}

struct LocatorCtx<'a> {
    all_annotations: &'a [SourceAnnotation],
    orphan_tests: &'a [OrphanTest],
    repo_root: &'a Path,
    test_outcomes: OutcomesRef<'a>,
}

struct ReqLinkCtx<'a> {
    doc: &'a ParsedDesignDoc,
    analysis: &'a AnalysisCtx<'a>,
    is_approved: bool,
    unverified_text_lower: &'a str,
    doc_slug: &'a str,
}

struct StatusInputs<'a> {
    locators: &'a [String],
    matching_funcs: &'a [String],
    is_unverified: bool,
    all_resolved: bool,
    test_outcomes: OutcomesRef<'a>,
}

struct LocatorBatch<'a> {
    locators: &'a [String],
    matching: &'a [String],
    is_unverified: bool,
}

struct MethodsLocators {
    methods: Vec<String>,
    locators: Vec<String>,
}

struct OutcomeAcc {
    any_run: bool,
    any_failed: bool,
    all_passed: bool,
}

struct RunnableClass<'a> {
    locators: Vec<&'a str>,
    all_static: bool,
}

/// Lookup interface for recorded test execution outcomes.
pub trait OutcomesLookup {
    /// Number of recorded named outcomes.
    fn outcome_count(&self) -> usize;
    /// Outcome for a test name, if present.
    fn outcome(&self, key: &str) -> Option<&TestRunOutcome>;
    /// Whether a test name is present.
    fn has_outcome(&self, key: &str) -> bool;
}

impl<S: BuildHasher> OutcomesLookup for HashMap<String, TestRunOutcome, S> {
    fn outcome_count(&self) -> usize {
        self.len()
    }

    fn outcome(&self, key: &str) -> Option<&TestRunOutcome> {
        self.get(key)
    }

    fn has_outcome(&self, key: &str) -> bool {
        self.contains_key(key)
    }
}

/// Links and associates parsed design documents with source annotations and test outcomes.
#[must_use]
pub fn link_requirements(
    docs: &[ParsedDesignDoc],
    scan: &SourceScanResult,
    repo_root: &Path,
    test_outcomes: OutcomesRef<'_>,
) -> TraceMatrixSummary {
    let mut summary = TraceMatrixSummary {
        total_documents: docs.len(),
        recorded_outcomes_count: test_outcomes
            .map_or(0, OutcomesLookup::outcome_count),
        ..Default::default()
    };

    let corpus = index_declared_reqs(docs);
    let ctx = AnalysisCtx {
        scan,
        repo_root,
        test_outcomes,
    };
    for doc in docs {
        link_document(doc, &ctx, &corpus, &mut summary);
    }

    summary.orphan_test_count = scan.orphan_tests.len();
    for orphan in &scan.orphan_tests {
        summary
            .defects
            .push(TraceDefect::OrphanTest(orphan.clone()));
    }

    summary
}

fn index_declared_reqs(docs: &[ParsedDesignDoc]) -> ReqSetIndex {
    let mut corpus = ReqSetIndex::new();
    for doc in docs {
        let mut doc_req_set = HashSet::new();
        for req in &doc.requirements {
            doc_req_set.insert(req.id.clone());
        }
        corpus.insert(doc.path.clone(), doc_req_set);
    }
    corpus
}

fn link_document(
    doc: &ParsedDesignDoc,
    ctx: &AnalysisCtx<'_>,
    corpus: &ReqSetIndex,
    summary: &mut TraceMatrixSummary,
) {
    let doc_slug_str = doc_slug(&doc.path);
    let is_approved = doc.status.eq_ignore_ascii_case("approved")
        || doc.status.eq_ignore_ascii_case("reviewed");
    let unverified_text_lower = doc.unverified_text.to_lowercase();

    let mut trace_by_req: TraceByReq<'_> = HashMap::new();
    for row in &doc.trace_rows {
        trace_by_req
            .entry(row.req_id.clone())
            .or_default()
            .push(row);
    }

    let declared_set = corpus.get(&doc.path).cloned().unwrap_or_default();
    for req_id in trace_by_req.keys() {
        if !declared_set.contains(req_id) {
            summary.defects.push(TraceDefect::UndeclaredRequirement {
                location: doc.path.clone(),
                req_id: req_id.clone(),
            });
        }
    }

    let req_ctx = ReqLinkCtx {
        doc,
        analysis: ctx,
        is_approved,
        unverified_text_lower: &unverified_text_lower,
        doc_slug: &doc_slug_str,
    };
    for req in &doc.requirements {
        let empty: &[&TraceabilityRow] = &[];
        let rows = trace_by_req.get(&req.id).map_or(empty, Vec::as_slice);
        link_one_requirement(req, rows, &req_ctx, summary);
    }
}

fn link_one_requirement(
    req: &DeclaredRequirement,
    rows: &[&TraceabilityRow],
    ctx: &ReqLinkCtx<'_>,
    summary: &mut TraceMatrixSummary,
) {
    summary.total_requirements = summary.total_requirements.saturating_add(1);
    let is_unverified =
        ctx.unverified_text_lower.contains(&req.id.to_lowercase());
    if rows.is_empty() {
        record_absent_requirement(req, ctx, is_unverified, summary);
        return;
    }

    let collected = collect_methods_locators(rows);
    let matching =
        matching_source_funcs(ctx.analysis.scan, ctx.doc_slug, &req.id);
    let all_resolved = resolve_all_locators(
        req,
        &LocatorBatch {
            locators: &collected.locators,
            matching: &matching,
            is_unverified,
        },
        ctx,
        summary,
    );
    let status = determine_status(&StatusInputs {
        locators: &collected.locators,
        matching_funcs: &matching,
        is_unverified,
        all_resolved,
        test_outcomes: ctx.analysis.test_outcomes,
    });
    record_status_count(summary, status, ctx.is_approved);
    summary.items.push(TracedRequirementItem {
        doc_path: ctx.doc.path.clone(),
        req_id: req.id.clone(),
        title: req.title.clone(),
        methods: collected.methods,
        locators: collected.locators,
        source_functions: matching,
        status,
    });
}

fn resolve_all_locators(
    req: &DeclaredRequirement,
    batch: &LocatorBatch<'_>,
    ctx: &ReqLinkCtx<'_>,
    summary: &mut TraceMatrixSummary,
) -> bool {
    let locator_ctx = LocatorCtx {
        all_annotations: &ctx.analysis.scan.annotations,
        orphan_tests: &ctx.analysis.scan.orphan_tests,
        repo_root: ctx.analysis.repo_root,
        test_outcomes: ctx.analysis.test_outcomes,
    };
    let mut all_resolved = true;
    for loc in batch.locators {
        if !resolve_locator(loc, batch.matching, &locator_ctx)
            && !batch.is_unverified
        {
            all_resolved = false;
            summary.defects.push(TraceDefect::UnresolvedArtifact {
                doc: ctx.doc.path.clone(),
                req_id: req.id.clone(),
                locator: loc.clone(),
                is_approved: ctx.is_approved,
            });
        }
    }
    all_resolved
}

fn record_absent_requirement(
    req: &DeclaredRequirement,
    ctx: &ReqLinkCtx<'_>,
    is_unverified: bool,
    summary: &mut TraceMatrixSummary,
) {
    if is_unverified {
        summary.unverified_count = summary.unverified_count.saturating_add(1);
        summary.items.push(empty_item(
            ctx.doc,
            req,
            RequirementStatus::Unverified,
        ));
        return;
    }
    summary.missing_count = summary.missing_count.saturating_add(1);
    if ctx.is_approved {
        summary.approved_missing_count =
            summary.approved_missing_count.saturating_add(1);
    }
    summary.defects.push(TraceDefect::MissingRequirement {
        doc: ctx.doc.path.clone(),
        req_id: req.id.clone(),
        title: req.title.clone(),
        is_approved: ctx.is_approved,
    });
    summary
        .items
        .push(empty_item(ctx.doc, req, RequirementStatus::Missing));
}

fn empty_item(
    doc: &ParsedDesignDoc,
    req: &DeclaredRequirement,
    status: RequirementStatus,
) -> TracedRequirementItem {
    TracedRequirementItem {
        doc_path: doc.path.clone(),
        req_id: req.id.clone(),
        title: req.title.clone(),
        methods: Vec::new(),
        locators: Vec::new(),
        source_functions: Vec::new(),
        status,
    }
}

fn collect_methods_locators(rows: &[&TraceabilityRow]) -> MethodsLocators {
    let mut methods = Vec::new();
    let mut locators = Vec::new();
    for r in rows {
        if !methods.contains(&r.method) {
            methods.push(r.method.clone());
        }
        for loc in &r.locators {
            if !locators.contains(loc) {
                locators.push(loc.clone());
            }
        }
    }
    MethodsLocators { methods, locators }
}

fn matching_source_funcs(
    scan: &SourceScanResult,
    doc_slug_str: &str,
    req_id: &str,
) -> Vec<String> {
    let mut matching = Vec::new();
    for ann in &scan.annotations {
        if ann.matches_req(doc_slug_str, req_id) {
            matching.push(format!("{}::{}", ann.file_path, ann.fn_name));
        }
    }
    matching
}

const fn record_status_count(
    summary: &mut TraceMatrixSummary,
    status: RequirementStatus,
    is_approved: bool,
) {
    match status {
        RequirementStatus::Verified => {
            summary.verified_count = summary.verified_count.saturating_add(1);
        }
        RequirementStatus::Failed => {
            summary.failed_count = summary.failed_count.saturating_add(1);
        }
        RequirementStatus::NotRun => {
            summary.not_run_count = summary.not_run_count.saturating_add(1);
        }
        RequirementStatus::Unverified => {
            summary.unverified_count =
                summary.unverified_count.saturating_add(1);
        }
        RequirementStatus::Unresolved => {
            summary.unresolved_count =
                summary.unresolved_count.saturating_add(1);
            if is_approved {
                summary.approved_unresolved_count =
                    summary.approved_unresolved_count.saturating_add(1);
            }
        }
        RequirementStatus::Missing => {
            summary.missing_count = summary.missing_count.saturating_add(1);
            if is_approved {
                summary.approved_missing_count =
                    summary.approved_missing_count.saturating_add(1);
            }
        }
    }
}

/// Check whether a locator string resolves to a valid source function, known gate, or existing file.
#[must_use]
fn resolve_locator(
    locator: &str,
    matching_funcs: &[String],
    ctx: &LocatorCtx<'_>,
) -> bool {
    let clean = locator.trim();
    if clean.is_empty() {
        return false;
    }
    if let Some(resolved) = resolve_scheme(clean, ctx.repo_root) {
        return resolved;
    }
    resolve_test_target(clean, matching_funcs, ctx)
}

fn resolve_scheme(clean: &str, repo_root: &Path) -> Option<bool> {
    if let Some(gate_name) = clean.strip_prefix("gate:") {
        return Some(!gate_name.trim().is_empty());
    }
    if clean.starts_with("cargo ") {
        return Some(true);
    }
    if let Some(budget) = clean.strip_prefix("budget:") {
        return Some(!budget.trim().is_empty());
    }
    if let Some(target) = clean.strip_prefix("inspection:") {
        return Some(path_target_exists(target, true, repo_root));
    }
    if let Some(target) = clean.strip_prefix("oracle:") {
        return Some(path_target_exists(target, true, repo_root));
    }
    if let Some(target) = clean.strip_prefix("compile_fail:") {
        return Some(path_target_exists(target, false, repo_root));
    }
    if let Some(target) = clean.strip_prefix("doctest:") {
        return Some(path_target_exists(target, false, repo_root));
    }
    None
}

fn path_target_exists(
    target: &str,
    split_hash: bool,
    repo_root: &Path,
) -> bool {
    let mut path_part = target;
    if split_hash {
        path_part = path_part.split('#').next().unwrap_or("");
    }
    let path_part = path_part.split("::").next().unwrap_or("").trim();
    !path_part.is_empty() && repo_root.join(path_part).exists()
}

fn resolve_test_target(
    clean: &str,
    matching_funcs: &[String],
    ctx: &LocatorCtx<'_>,
) -> bool {
    let test_target = clean.strip_prefix("test:").unwrap_or(clean).trim();
    if !rs_file_exists(test_target, ctx.repo_root) {
        return false;
    }
    let fn_short = test_target.split("::").last().unwrap_or(test_target).trim();
    matching_funcs
        .iter()
        .any(|func| func.ends_with(fn_short) || func.contains(test_target))
        || ctx
            .all_annotations
            .iter()
            .any(|ann| ann.fn_name == fn_short)
        || ctx
            .orphan_tests
            .iter()
            .any(|orphan| orphan.fn_name == fn_short)
        || outcomes_contain(ctx.test_outcomes, fn_short, test_target)
        || ctx
            .repo_root
            .join(test_target.split("::").next().unwrap_or(test_target))
            .exists()
}

fn rs_file_exists(test_target: &str, repo_root: &Path) -> bool {
    let Some(rs_idx) = test_target.find(".rs") else {
        return true;
    };
    let end = rs_idx.saturating_add(3);
    let Some(file_candidate) = test_target.get(..end) else {
        return false;
    };
    repo_root.join(file_candidate).exists()
}

fn outcomes_contain(
    test_outcomes: OutcomesRef<'_>,
    fn_short: &str,
    test_target: &str,
) -> bool {
    test_outcomes.is_some_and(|outcomes| {
        outcomes.has_outcome(fn_short) || outcomes.has_outcome(test_target)
    })
}

fn determine_status(inputs: &StatusInputs<'_>) -> RequirementStatus {
    if inputs.is_unverified {
        return RequirementStatus::Unverified;
    }
    if !inputs.all_resolved {
        return RequirementStatus::Unresolved;
    }
    if inputs.locators.is_empty() {
        return RequirementStatus::NotRun;
    }
    let classified = classify_runnable(inputs.locators);
    if classified.all_static {
        return RequirementStatus::Verified;
    }
    let Some(outcomes) = inputs.test_outcomes else {
        return RequirementStatus::NotRun;
    };
    evaluate_outcomes(&classified.locators, inputs.matching_funcs, outcomes)
}

fn classify_runnable(locators: &[String]) -> RunnableClass<'_> {
    let mut runnable = Vec::new();
    let mut all_static = true;
    for loc in locators {
        let clean = loc.trim();
        if is_static_locator(clean) {
            continue;
        }
        all_static = false;
        runnable.push(clean);
    }
    RunnableClass {
        locators: runnable,
        all_static,
    }
}

fn is_static_locator(clean: &str) -> bool {
    clean.starts_with("gate:")
        || clean.starts_with("inspection:")
        || clean.starts_with("compile_fail:")
        || clean.starts_with("budget:")
        || clean.starts_with("cargo clippy")
        || clean.starts_with("cargo lint")
        || clean.starts_with("cargo fmt")
        || clean.starts_with("cargo check")
}

fn evaluate_outcomes(
    runnable: &[&str],
    matching_funcs: &[String],
    outcomes: OutcomesDyn<'_>,
) -> RequirementStatus {
    let mut acc = OutcomeAcc {
        any_run: false,
        any_failed: false,
        all_passed: true,
    };
    apply_locator_outcomes(runnable, outcomes, &mut acc);
    apply_func_outcomes(matching_funcs, outcomes, &mut acc);
    if acc.any_failed {
        RequirementStatus::Failed
    } else if acc.any_run && acc.all_passed {
        RequirementStatus::Verified
    } else {
        RequirementStatus::NotRun
    }
}

fn apply_locator_outcomes(
    runnable: &[&str],
    outcomes: OutcomesDyn<'_>,
    acc: &mut OutcomeAcc,
) {
    for loc in runnable {
        let clean = loc.strip_prefix("test:").unwrap_or(loc);
        let fn_short = clean.split("::").last().unwrap_or(clean);
        let outcome = outcomes
            .outcome(fn_short)
            .or_else(|| outcomes.outcome(clean));
        match outcome {
            Some(o) if o.is_passed() => {
                acc.any_run = true;
            }
            Some(o) if o.is_failed() => {
                acc.any_run = true;
                acc.any_failed = true;
                acc.all_passed = false;
            }
            Some(_) | None => {
                acc.all_passed = false;
            }
        }
    }
}

fn apply_func_outcomes(
    matching_funcs: &[String],
    outcomes: OutcomesDyn<'_>,
    acc: &mut OutcomeAcc,
) {
    for func in matching_funcs {
        let fn_short = func.split("::").last().unwrap_or(func);
        let outcome = outcomes
            .outcome(fn_short)
            .or_else(|| outcomes.outcome(func));
        match outcome {
            Some(o) if o.is_passed() => {
                acc.any_run = true;
            }
            Some(o) if o.is_failed() => {
                acc.any_run = true;
                acc.any_failed = true;
            }
            Some(_) | None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(id: &str, title: &str, doc: &str) -> DeclaredRequirement {
        DeclaredRequirement {
            id: id.to_string(),
            title: title.to_string(),
            doc_path: doc.to_string(),
            doc_status: "Approved".to_string(),
        }
    }

    fn gate_row(req_id: &str, doc: &str) -> TraceabilityRow {
        TraceabilityRow {
            req_raw: req_id.to_string(),
            req_id: req_id.to_string(),
            method: "Requirements-based test".to_string(),
            artifact_raw: "gate:clippy-ci".to_string(),
            locators: vec!["gate:clippy-ci".to_string()],
            doc_path: doc.to_string(),
            doc_status: "Approved".to_string(),
        }
    }

    fn approved_doc(
        path: &str,
        requirements: Vec<DeclaredRequirement>,
        trace_rows: Vec<TraceabilityRow>,
    ) -> ParsedDesignDoc {
        ParsedDesignDoc {
            path: path.to_string(),
            requirements,
            trace_rows,
            unverified_text: String::new(),
            status: "Approved".to_string(),
        }
    }

    fn item_status(summary: &TraceMatrixSummary) -> RequirementStatus {
        summary.items.first().unwrap().status
    }

    /// # Verification
    /// Trace: FR-2
    /// Method: Requirements-based test
    #[test]
    fn test_link_missing_requirement() {
        let doc = approved_doc(
            "doc1.md",
            vec![
                declared("FR-1", "First", "doc1.md"),
                declared("FR-2", "Second", "doc1.md"),
            ],
            vec![gate_row("FR-1", "doc1.md")],
        );
        let scan = SourceScanResult::default();
        let summary = link_requirements(&[doc], &scan, Path::new("."), None);

        assert_eq!(summary.total_requirements, 2);
        assert_eq!(summary.missing_count, 1);
        assert_eq!(summary.approved_missing_count, 1);
        assert_eq!(summary.verified_count, 1);
        assert!(summary.defects.iter().any(|d| matches!(
            d,
            TraceDefect::MissingRequirement { req_id, is_approved: true, .. } if req_id == "FR-2"
        )));
    }

    /// # Verification
    /// Trace: FR-3
    /// Method: Requirements-based test
    #[test]
    fn test_link_undeclared_requirement() {
        let doc = approved_doc(
            "doc1.md",
            vec![declared("FR-1", "First", "doc1.md")],
            vec![TraceabilityRow {
                req_raw: "FR-99".to_string(),
                req_id: "FR-99".to_string(),
                method: "Inspection".to_string(),
                artifact_raw: "gate:clippy-ci".to_string(),
                locators: vec!["gate:clippy-ci".to_string()],
                doc_path: "doc1.md".to_string(),
                doc_status: "Approved".to_string(),
            }],
        );
        let scan = SourceScanResult::default();
        let summary = link_requirements(&[doc], &scan, Path::new("."), None);
        assert!(summary.defects.iter().any(|d| matches!(
            d,
            TraceDefect::UndeclaredRequirement { req_id, .. } if req_id == "FR-99"
        )));
    }

    /// # Verification
    /// Trace: FR-1
    /// Method: Requirements-based test
    #[test]
    fn test_unresolved_locator_prevents_verified() {
        let path = "math/storage-design.md";
        let doc = approved_doc(
            path,
            vec![declared("FR-1", "Layout", path)],
            vec![TraceabilityRow {
                req_raw: "FR-1".to_string(),
                req_id: "FR-1".to_string(),
                method: "Requirements-based test".to_string(),
                artifact_raw: "test:nonexistent_file.rs::nonexistent_test"
                    .to_string(),
                locators: vec![
                    "test:nonexistent_file.rs::nonexistent_test".to_string(),
                ],
                doc_path: path.to_string(),
                doc_status: "Approved".to_string(),
            }],
        );
        let mut outcomes = HashMap::new();
        outcomes.insert("nonexistent_test".to_string(), TestRunOutcome::Passed);
        let scan = SourceScanResult::default();
        let summary = link_requirements(
            &[doc],
            &scan,
            Path::new("."),
            Some(&outcomes as &dyn OutcomesLookup),
        );
        assert_eq!(
            summary.verified_count, 0,
            "Unresolved locator must never be Verified"
        );
        assert_eq!(summary.unresolved_count, 1);
        assert_eq!(summary.approved_unresolved_count, 1);
        assert_eq!(item_status(&summary), RequirementStatus::Unresolved);
    }

    fn fr2_storage_fixture() -> (ParsedDesignDoc, SourceScanResult) {
        let path = "math/storage-design.md";
        let doc = approved_doc(
            path,
            vec![declared("FR-2", "Bounds", path)],
            vec![TraceabilityRow {
                req_raw: "FR-2".to_string(),
                req_id: "FR-2".to_string(),
                method: "Requirements-based test".to_string(),
                artifact_raw: "test:t1, test:t2".to_string(),
                locators: vec!["test:t1".to_string(), "test:t2".to_string()],
                doc_path: path.to_string(),
                doc_status: "Approved".to_string(),
            }],
        );
        let scan = SourceScanResult {
            annotations: vec![
                annotated("t1", "storage-design#FR-2", 1),
                annotated("t2", "storage-design#FR-2", 10),
            ],
            orphan_tests: Vec::new(),
        };
        (doc, scan)
    }

    fn annotated(fn_name: &str, req: &str, line: usize) -> SourceAnnotation {
        SourceAnnotation {
            requirements: vec![req.to_string()],
            method: Some("Requirements-based test".to_string()),
            fn_name: fn_name.to_string(),
            file_path: "src/math/tests/storage_tests.rs".to_string(),
            line_number: line,
            is_test: true,
        }
    }

    /// # Verification
    /// Trace: FR-6
    /// Method: Requirements-based test
    #[test]
    fn test_universal_quantification_multi_test() {
        let (doc, scan) = fr2_storage_fixture();
        let mut outcomes_partial = HashMap::new();
        outcomes_partial.insert("t1".to_string(), TestRunOutcome::Passed);
        let summary_partial = link_requirements(
            std::slice::from_ref(&doc),
            &scan,
            Path::new("."),
            Some(&outcomes_partial as &dyn OutcomesLookup),
        );
        assert_eq!(item_status(&summary_partial), RequirementStatus::NotRun);

        let mut outcomes_failed = HashMap::new();
        outcomes_failed.insert("t1".to_string(), TestRunOutcome::Passed);
        outcomes_failed.insert("t2".to_string(), TestRunOutcome::Failed);
        let summary_failed = link_requirements(
            std::slice::from_ref(&doc),
            &scan,
            Path::new("."),
            Some(&outcomes_failed as &dyn OutcomesLookup),
        );
        assert_eq!(item_status(&summary_failed), RequirementStatus::Failed);

        let mut outcomes_all = HashMap::new();
        outcomes_all.insert("t1".to_string(), TestRunOutcome::Passed);
        outcomes_all.insert("t2".to_string(), TestRunOutcome::Passed);
        let summary_all = link_requirements(
            std::slice::from_ref(&doc),
            &scan,
            Path::new("."),
            Some(&outcomes_all as &dyn OutcomesLookup),
        );
        assert_eq!(item_status(&summary_all), RequirementStatus::Verified);
    }

    /// # Verification
    /// Trace: FR-7
    /// Method: Requirements-based test
    #[test]
    fn test_document_scoping_prevents_cross_contamination() {
        let path = "documentation/math/storage-design.md";
        let doc_math = approved_doc(
            path,
            vec![declared("FR-1", "Storage Layout", path)],
            vec![TraceabilityRow {
                req_raw: "FR-1".to_string(),
                req_id: "FR-1".to_string(),
                method: "Requirements-based test".to_string(),
                artifact_raw: "test:test_ci_orphan".to_string(),
                locators: vec!["test:test_ci_orphan".to_string()],
                doc_path: path.to_string(),
                doc_status: "Approved".to_string(),
            }],
        );
        let scan = SourceScanResult {
            annotations: vec![SourceAnnotation {
                requirements: vec!["FR-1".to_string()],
                method: Some("Requirements-based test".to_string()),
                fn_name: "test_ci_orphan".to_string(),
                file_path: "control-rs-ci/src/trace/requirement_linker.rs"
                    .to_string(),
                line_number: 100,
                is_test: true,
            }],
            orphan_tests: Vec::new(),
        };
        let summary =
            link_requirements(&[doc_math], &scan, Path::new("."), None);
        assert!(
            summary.items.first().unwrap().source_functions.is_empty(),
            "Cross-crate bare FR-1 must not contaminate storage-design"
        );
    }
}
