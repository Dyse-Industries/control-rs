//! Rust source code doc-comment scanner for requirement traceability.

use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};

use super::doc_parser::extract_req_id;

/// An annotated function or method in Rust source code.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceAnnotation {
    /// Requirements traced by this function, e.g. `FR-1`, `NFR-2`.
    pub requirements: Vec<String>,
    /// Optional method specified in doc comment, e.g. "Requirements-based test".
    pub method: Option<String>,
    /// Function or test name, e.g. "`test_roots_satisfy_characteristic_equation`".
    pub fn_name: String,
    /// Relative path to source file.
    pub file_path: String,
    /// Line number where the function starts.
    pub line_number: usize,
    /// Whether this function is a test or verification suite.
    pub is_test: bool,
}

/// A test function that does not declare any requirement it traces.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OrphanTest {
    /// Function or test name.
    pub fn_name: String,
    /// Relative path to source file.
    pub file_path: String,
    /// Line number.
    pub line_number: usize,
}

/// Results of scanning source trees.
#[derive(Debug, Clone, Default)]
pub struct SourceScanResult {
    /// All annotations found on functions.
    pub annotations: Vec<SourceAnnotation>,
    /// All test functions lacking requirement annotations.
    pub orphan_tests: Vec<OrphanTest>,
}

struct ScanFnCtx<'a> {
    lines: &'a [&'a str],
    rel_path: &'a str,
    is_in_test_dir: bool,
    trace_re: &'a Regex,
    method_re: &'a Regex,
}

struct FnPrelude<'a> {
    doc_lines: Vec<&'a str>,
    has_test_attr: bool,
}

impl SourceAnnotation {
    /// Checks whether this annotation matches the specified document slug and requirement ID.
    #[must_use]
    pub fn matches_req(&self, doc_slug: &str, req_id: &str) -> bool {
        let qual = format!("{doc_slug}#{req_id}");
        if self.requirements.contains(&qual) {
            return true;
        }
        if self.requirements.contains(&req_id.to_string()) {
            return self.is_contextually_relevant(doc_slug);
        }
        false
    }

    fn is_contextually_relevant(&self, doc_slug: &str) -> bool {
        let path = self.file_path.as_str();
        math_relevant(doc_slug, path)
            || numerical_relevant(doc_slug, path)
            || classical_relevant(doc_slug, path)
            || crate_relevant(doc_slug, path)
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

fn math_relevant(slug: &str, path: &str) -> bool {
    contains_any(
        slug,
        &["math", "storage", "subprograms", "error", "fixed", "num"],
    ) && path.contains("math")
}

fn numerical_relevant(slug: &str, path: &str) -> bool {
    (contains_any(slug, &["matrix", "numerical"])
        && contains_any(path, &["matrix", "numerical"]))
        || (contains_any(slug, &["polynomial", "numerical"])
            && contains_any(path, &["polynomial", "numerical"]))
        || (contains_any(slug, &["state-space", "state_space", "numerical"])
            && contains_any(path, &["state_space", "state-space", "numerical"]))
        || (contains_any(
            slug,
            &["transfer-function", "transfer_function", "numerical"],
        ) && contains_any(
            path,
            &["transfer_function", "transfer-function", "numerical"],
        ))
        || (contains_any(slug, &["tensor", "numerical"])
            && contains_any(path, &["tensor", "numerical"]))
}

fn classical_relevant(slug: &str, path: &str) -> bool {
    (contains_any(slug, &["classical-tools-examples", "buck", "motor"])
        && contains_any(path, &["buck", "dc-motor"]))
        || (slug.contains("classical-tools")
            && contains_any(path, &["classical", "control"]))
}

fn crate_relevant(slug: &str, path: &str) -> bool {
    (slug.contains("ci") && path.contains("control-rs-ci"))
        || (slug.contains("trace") && path.contains("trace"))
        || (slug.contains("ets-host") && path.contains("control-rs-ets-host"))
        || (slug.contains("ets") && path.contains("control-rs-ets"))
        || (slug.contains("macros") && path.contains("control-rs-macros"))
        || (slug.contains("tui") && path.contains("control-rs-tui"))
}

/// Recursively find all `*.rs` files under `roots`.
#[must_use]
pub fn find_rust_sources(roots: &[&Path]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for root in roots {
        find_rs_recursive(root, &mut files);
    }
    files.sort();
    files
}

fn find_rs_recursive(dir: &Path, acc: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.') && name != "target" && name != "scratch" {
                find_rs_recursive(&path, acc);
            }
        } else if path.is_file()
            && path.extension().and_then(|e| e.to_str()) == Some("rs")
        {
            acc.push(path);
        }
    }
}

/// Scan a set of Rust source files relative to `repo_root`.
#[must_use]
pub fn scan_sources(files: &[PathBuf], repo_root: &Path) -> SourceScanResult {
    let mut result = SourceScanResult::default();

    for file in files {
        let rel_path = file
            .strip_prefix(repo_root)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();

        let Ok(content) = fs::read_to_string(file) else {
            continue;
        };

        scan_file_content(&content, &rel_path, &mut result);
    }

    result
}

fn collect_fn_prelude<'a>(lines: &'a [&str], fn_idx: usize) -> FnPrelude<'a> {
    let mut doc_lines = Vec::new();
    let mut has_test_attr = false;
    let mut j = fn_idx;
    while j > 0 {
        j = j.saturating_sub(1);
        let Some(prev_line) = lines.get(j) else {
            break;
        };
        let prev = prev_line.trim();
        if prev.starts_with("///") {
            doc_lines.push(prev.trim_start_matches("///").trim());
        } else if prev.starts_with("#[test]")
            || prev.starts_with("#[ets_test]")
            || prev.starts_with("#[tokio::test]")
        {
            has_test_attr = true;
        } else if !(prev.starts_with("#[") || prev.is_empty()) {
            break;
        }
    }
    FnPrelude {
        doc_lines,
        has_test_attr,
    }
}

fn parse_trace_requirements(doc_text: &str, trace_re: &Regex) -> Vec<String> {
    let mut reqs = Vec::new();
    for cap in trace_re.captures_iter(doc_text) {
        let Some(raw_list) = cap.get(1).map(|m| m.as_str()) else {
            continue;
        };
        for part in raw_list.split(',') {
            push_req_from_part(part.trim(), &mut reqs);
        }
    }
    reqs
}

fn push_req_from_part(trimmed: &str, reqs: &mut Vec<String>) {
    if trimmed.contains('#') {
        let mut sub = trimmed.split('#');
        if let (Some(slug), Some(raw_req)) = (sub.next(), sub.next())
            && let Some(id) = extract_req_id(raw_req)
        {
            let qual = format!("{}#{}", slug.trim(), id);
            if !reqs.contains(&qual) {
                reqs.push(qual);
            }
        }
    } else if let Some(id) = extract_req_id(trimmed)
        && !reqs.contains(&id)
    {
        reqs.push(id);
    }
}

/// Scan the text content of a single Rust file.
pub fn scan_file_content(
    content: &str,
    rel_path: &str,
    result: &mut SourceScanResult,
) {
    let lines: Vec<&str> = content.lines().collect();
    let is_in_test_dir =
        rel_path.contains("/tests/") || rel_path.starts_with("tests/");
    let Ok(fn_re) = Regex::new(
        r"^\s*(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+([A-Za-z0-9_]+)",
    ) else {
        return;
    };
    let Ok(trace_re) = Regex::new(
        r"(?i)(?:trace|traces|discharges|implements|verifies)\s*:\s*([A-Za-z0-9_,\s\.\-#]+)",
    ) else {
        return;
    };
    let Ok(method_re) = Regex::new(r"(?i)method\s*:\s*([^#\n\r]+)") else {
        return;
    };

    let ctx = ScanFnCtx {
        lines: &lines,
        rel_path,
        is_in_test_dir,
        trace_re: &trace_re,
        method_re: &method_re,
    };
    let mut i = 0;
    while i < lines.len() {
        if let Some(line) = lines.get(i)
            && let Some(fn_cap) = fn_re.captures(line)
        {
            let fn_name = fn_cap
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            record_scanned_fn(&ctx, i, fn_name, result);
        }
        i = i.saturating_add(1);
    }
}

fn record_scanned_fn(
    ctx: &ScanFnCtx<'_>,
    i: usize,
    fn_name: String,
    result: &mut SourceScanResult,
) {
    let fn_line = i.saturating_add(1);
    let prelude = collect_fn_prelude(ctx.lines, i);
    let is_test = prelude.has_test_attr
        || ctx.is_in_test_dir
        || fn_name.starts_with("test_")
        || fn_name.starts_with("prop_");
    let mut doc_lines = prelude.doc_lines;
    doc_lines.reverse();
    let doc_text = doc_lines.join("\n");
    let reqs = parse_trace_requirements(&doc_text, ctx.trace_re);
    let method = ctx
        .method_re
        .captures(&doc_text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string());

    if !reqs.is_empty() {
        result.annotations.push(SourceAnnotation {
            requirements: reqs,
            method,
            fn_name,
            file_path: ctx.rel_path.to_string(),
            line_number: fn_line,
            is_test,
        });
    } else if is_test {
        result.orphan_tests.push(OrphanTest {
            fn_name,
            file_path: ctx.rel_path.to_string(),
            line_number: fn_line,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_file_content_annotated() {
        let content = r"
/// Computes the roots of the polynomial.
///
/// # Verification
/// Trace: FR-1, NFR-2
/// Method: Requirements-based test
#[test]
fn test_roots_converge() {
    assert!(true);
}

#[test]
fn test_unannotated_helper() {
    assert!(true);
}
";
        let mut res = SourceScanResult::default();
        scan_file_content(content, "tests/my_test.rs", &mut res);
        assert_annotated_scan(&res);
    }

    fn assert_annotated_scan(res: &SourceScanResult) {
        assert_eq!(res.annotations.len(), 1);
        let ann = res.annotations.first().unwrap();
        assert_eq!(ann.fn_name, "test_roots_converge");
        assert_eq!(ann.requirements, vec!["FR-1", "NFR-2"]);
        assert_eq!(ann.method.as_deref(), Some("Requirements-based test"));
        assert!(ann.is_test);
        assert_eq!(res.orphan_tests.len(), 1);
        assert_eq!(
            res.orphan_tests.first().unwrap().fn_name,
            "test_unannotated_helper"
        );
    }

    #[test]
    fn test_scan_file_content_qualified_annotation() {
        let content = r"
/// Storage bounds tests.
///
/// # Verification
/// Trace: error-design#FR-1, storage-design#FR-2
#[test]
fn test_storage_bounds() {
    assert!(true);
}
";
        let mut res = SourceScanResult::default();
        scan_file_content(content, "src/math/tests/storage_tests.rs", &mut res);
        assert_qualified_scan(&res);
    }

    fn assert_qualified_scan(res: &SourceScanResult) {
        assert_eq!(res.annotations.len(), 1);
        let ann = res.annotations.first().unwrap();
        assert_eq!(ann.fn_name, "test_storage_bounds");
        assert_eq!(
            ann.requirements,
            vec!["error-design#FR-1", "storage-design#FR-2"]
        );
        assert!(ann.matches_req("error-design", "FR-1"));
        assert!(ann.matches_req("storage-design", "FR-2"));
        assert!(!ann.matches_req("classical-tools", "FR-1"));
    }
}
