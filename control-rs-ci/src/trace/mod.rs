//! Requirement traceability subsystem for `control-rs`.
//!
//! Provides bidirectional traceability between design document requirements (§2)
//! and verification evidence (§6.4 / §6.7), resolved against function doc comments
//! (`/// Trace: ...`) and test execution results.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

pub use doc_parser::{
    DeclaredRequirement, ParsedDesignDoc, find_design_docs, parse_design_doc,
};
pub use requirement_linker::{
    RequirementStatus, TraceDefect, TraceMatrixSummary, TracedRequirementItem,
    link_requirements,
};
pub use source_scanner::{
    OrphanTest, SourceAnnotation, find_rust_sources, scan_sources,
};

pub mod doc_parser;
pub mod requirement_linker;
pub mod results;
pub mod source_scanner;

type SubsystemCounts = Vec<(String, usize)>;

/// Run complete traceability analysis across the workspace documentation and source code.
/// Discover workspace source roots by inspecting `Cargo.toml` without searching external directories.
#[must_use]
pub fn discover_workspace_src_roots(
    repo_root: &Path,
) -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();

    let root_src = repo_root.join("src");
    if root_src.is_dir() {
        roots.push(root_src);
    }
    let root_tests = repo_root.join("tests");
    if root_tests.is_dir() {
        roots.push(root_tests);
    }

    let cargo_toml = repo_root.join("Cargo.toml");
    if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
        for member in parse_workspace_members(&content) {
            if member == "." || member.is_empty() {
                continue;
            }
            let member_dir = repo_root.join(member);
            let member_src = member_dir.join("src");
            if member_src.is_dir() {
                roots.push(member_src);
            }
            let member_tests = member_dir.join("tests");
            if member_tests.is_dir() {
                roots.push(member_tests);
            }
        }
    }

    // The host-validation member's tests discharge design-doc requirements
    // whose locators are repo-relative. Scan it from the root corpus so Trace
    // annotations participate in the live gate even when `trace.toml` omits
    // it.
    let validation = repo_root.join("control-rs-validation/src");
    if validation.is_dir() {
        roots.push(validation);
    }

    roots
}

/// Parse workspace members list from root `Cargo.toml`.
#[must_use]
pub fn parse_workspace_members(cargo_toml: &str) -> Vec<String> {
    let mut members = Vec::new();
    let mut in_workspace = false;
    let mut in_members = false;

    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            if trimmed == "[workspace]" {
                in_workspace = true;
            } else {
                in_workspace = false;
                in_members = false;
            }
            continue;
        }

        if in_workspace
            && trimmed.starts_with("members")
            && trimmed.contains('=')
        {
            in_members = true;
        }

        if in_members {
            parse_quoted_members(trimmed, &mut members);
            if trimmed.contains(']') {
                in_members = false;
            }
        }
    }

    members
}

fn parse_quoted_members(trimmed: &str, members: &mut Vec<String>) {
    let mut remaining = trimmed;
    while let Some(start) = remaining.find('"') {
        let Some(rest) = remaining.get(start.saturating_add(1)..) else {
            break;
        };
        if let Some(end) = rest.find('"') {
            let Some(member) = rest.get(..end) else {
                break;
            };
            members.push(member.to_string());
            remaining = rest.get(end.saturating_add(1)..).unwrap_or("");
        } else {
            break;
        }
    }
}

/// Run complete traceability analysis across the workspace documentation and source code.
#[must_use]
pub fn run_traceability_analysis(
    repo_root: &Path,
    cargo_test_output: Option<&str>,
    ets_results_json: Option<&str>,
) -> TraceMatrixSummary {
    let doc_dir = repo_root.join("documentation");
    let doc_paths = if doc_dir.is_dir() {
        find_design_docs(&doc_dir)
    } else {
        Vec::new()
    };

    let mut parsed_docs = Vec::new();
    for path in &doc_paths {
        if let Some(doc) = parse_design_doc(path, repo_root) {
            parsed_docs.push(doc);
        }
    }

    let src_roots = discover_workspace_src_roots(repo_root);
    let existing_roots: Vec<&Path> =
        src_roots.iter().map(std::path::PathBuf::as_path).collect();

    let rust_files = find_rust_sources(&existing_roots);
    let scan_results = scan_sources(&rust_files, repo_root);

    let mut combined_outcomes = HashMap::new();
    if let Some(test_out) = cargo_test_output {
        combined_outcomes.extend(results::parse_cargo_test_output(test_out));
    }

    if let Some(ets_json) = ets_results_json {
        combined_outcomes.extend(results::parse_ets_results(ets_json));
    }

    let test_outcomes_opt = if combined_outcomes.is_empty() {
        None
    } else {
        Some(&combined_outcomes as &dyn requirement_linker::OutcomesLookup)
    };

    link_requirements(&parsed_docs, &scan_results, repo_root, test_outcomes_opt)
}

/// Render a high-level executive brief suitable for inclusion in `ci-report.md`.
#[must_use]
pub fn render_trace_brief(summary: &TraceMatrixSummary) -> String {
    let mut out = String::new();
    out.push_str("### Requirement Traceability Brief\n\n");
    render_metrics_table(&mut out, summary);
    render_defects_section(&mut out, summary);
    render_orphans_section(&mut out, summary);
    out
}

const fn brief_status(summary: &TraceMatrixSummary) -> &'static str {
    if summary.approved_missing_count == 0
        && summary.approved_unresolved_count == 0
        && summary.failed_count == 0
    {
        "Pass"
    } else {
        "Defects Found"
    }
}

fn render_metrics_table(out: &mut String, summary: &TraceMatrixSummary) {
    let status_str = brief_status(summary);
    let draft_gaps = summary
        .missing_count
        .saturating_sub(summary.approved_missing_count);
    let _ = write!(
        out,
        "| Metric | Value |\n\
         | :--- | :--- |\n\
         | **Status** | {status_str} |\n\
         | **Total Requirements** | {} across {} design documents |\n\
         | **Verified** | {} |\n\
         | **Failed** | {} |\n\
         | **Not Run / Untested** | {} |\n\
         | **Deferred (Unverified §6.7)** | {} |\n\
         | **Unresolved Locators** | {} (Approved: {}) |\n\
         | **Missing Requirements** | {} (Approved: {}) |\n\
         | **Draft Gaps (Advisory)** | {draft_gaps} missing in draft documents |\n\
         | **Orphan Tests** | {} tests lacking doc annotations |\n\
         | **Full Report** | See `trace-report.md` & `trace-report.json` |\n\n",
        summary.total_requirements,
        summary.total_documents,
        summary.verified_count,
        summary.failed_count,
        summary.not_run_count,
        summary.unverified_count,
        summary.unresolved_count,
        summary.approved_unresolved_count,
        summary.missing_count,
        summary.approved_missing_count,
        summary.orphan_test_count,
    );
}

const fn defect_sort_key(defect: &TraceDefect) -> u8 {
    match defect {
        TraceDefect::MissingRequirement {
            is_approved: true, ..
        } => 0,
        TraceDefect::UnresolvedArtifact {
            is_approved: true, ..
        } => 1,
        TraceDefect::UndeclaredRequirement { .. } => 2,
        TraceDefect::MissingRequirement {
            is_approved: false, ..
        } => 3,
        TraceDefect::UnresolvedArtifact {
            is_approved: false, ..
        } => 4,
        TraceDefect::OrphanTest(_) => 5,
    }
}

fn render_defects_section(out: &mut String, summary: &TraceMatrixSummary) {
    let mut sorted_defects: Vec<&TraceDefect> = summary
        .defects
        .iter()
        .filter(|d| !matches!(d, TraceDefect::OrphanTest(_)))
        .collect();
    if sorted_defects.is_empty() {
        return;
    }
    sorted_defects.sort_by_key(|d| defect_sort_key(d));

    out.push_str("<details>\n<summary>Traceability Defects (");
    let _ = write!(out, "{} document findings", sorted_defects.len());
    out.push_str(")</summary>\n\n");
    out.push_str("| Type | Location | Requirement | Detail |\n");
    out.push_str("| :--- | :--- | :--- | :--- |\n");
    for defect in sorted_defects.iter().take(30) {
        render_defect_row(out, defect);
    }
    if sorted_defects.len() > 30 {
        let extra = sorted_defects.len().saturating_sub(30);
        let _ = write!(
            out,
            "\n*... and {extra} additional findings recorded in `trace-report.md`.*\n",
        );
    }
    out.push_str("\n</details>\n\n");
}

fn render_defect_row(out: &mut String, defect: &TraceDefect) {
    match defect {
        TraceDefect::MissingRequirement {
            doc,
            req_id,
            title,
            is_approved,
        } => {
            let tag = if *is_approved {
                "**Approved Missing**"
            } else {
                "Draft Missing"
            };
            let _ = writeln!(
                out,
                "| {tag} | `{doc}` | `{req_id}` | Declared in §2 but absent from §6.4/§6.7: {title} |"
            );
        }
        TraceDefect::UndeclaredRequirement { location, req_id } => {
            let _ = writeln!(
                out,
                "| Undeclared | `{location}` | `{req_id}` | Cited in §6.4 or doc comments but not declared in §2 |"
            );
        }
        TraceDefect::UnresolvedArtifact {
            doc,
            req_id,
            locator,
            is_approved,
        } => {
            let tag = if *is_approved {
                "**Approved Unresolved**"
            } else {
                "Draft Unresolved"
            };
            let _ = writeln!(
                out,
                "| {tag} | `{doc}` | `{req_id}` | Artifact locator cannot be resolved: `{locator}` |"
            );
        }
        TraceDefect::OrphanTest(_) => {}
    }
}

fn render_orphans_section(out: &mut String, summary: &TraceMatrixSummary) {
    if summary.orphan_test_count == 0 {
        return;
    }

    let mut orphans_by_subsystem: HashMap<String, usize> = HashMap::new();
    for defect in &summary.defects {
        if let TraceDefect::OrphanTest(orphan) = defect {
            let subsystem = orphan
                .file_path
                .split('/')
                .take(2)
                .collect::<Vec<&str>>()
                .join("/");
            let count = orphans_by_subsystem.entry(subsystem).or_default();
            *count = count.saturating_add(1);
        }
    }

    let mut sorted_subsystems: SubsystemCounts =
        orphans_by_subsystem.into_iter().collect();
    sorted_subsystems.sort_by_key(|b| std::cmp::Reverse(b.1));

    out.push_str("<details>\n<summary>Orphan Tests Roll-up (");
    let _ = write!(
        out,
        "{} tests lacking `/// Trace:` annotations",
        summary.orphan_test_count
    );
    out.push_str(")</summary>\n\n");
    out.push_str("| Subsystem / Directory | Orphan Test Count |\n");
    out.push_str("| :--- | :--- |\n");
    for (subsystem, count) in &sorted_subsystems {
        let _ = writeln!(out, "| `{subsystem}` | {count} |");
    }
    out.push_str(
        "\n*Full list of orphan tests available in `trace-report.md`.*\n",
    );
    out.push_str("\n</details>\n\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_workspace_members() {
        let toml = r#"
[workspace]
members = [
    ".",
    "control-rs-ets",
    "control-rs-macros",
    "control-rs-ets-host",
    "control-rs-tui",
    "control-rs-ci",
]
"#;
        let members = parse_workspace_members(toml);
        assert_eq!(members.len(), 6);
        assert_eq!(members.first().unwrap(), ".");
        assert_eq!(members.get(1).unwrap(), "control-rs-ets");
        assert_eq!(members.get(5).unwrap(), "control-rs-ci");
    }

    #[test]
    fn test_parse_workspace_members_inline() {
        let toml = r#"[workspace]
members = ["crate-a", "crate-b"]
"#;
        let members = parse_workspace_members(toml);
        assert_eq!(members, vec!["crate-a", "crate-b"]);
    }
}
