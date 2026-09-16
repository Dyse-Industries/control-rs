//! Markdown design document parser for requirement traceability.

use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};

/// A requirement declared in Section 2 of a design document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredRequirement {
    /// Identifier, e.g. "FR-1", "NFR-2", "C-3".
    pub id: String,
    /// Title or short summary.
    pub title: String,
    /// Relative path of the owning document.
    pub doc_path: String,
    /// Document review status ("Approved", "Reviewed", "Draft").
    pub doc_status: String,
}

/// A row in the Section 6.4 Traceability table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceabilityRow {
    /// Raw requirement string or ID, e.g. "FR-1 — Routh-Hurwitz Stability".
    pub req_raw: String,
    /// Normalized requirement ID, e.g. "FR-1".
    pub req_id: String,
    /// Method from the V&V catalogue, e.g. "Requirements-based test".
    pub method: String,
    /// Raw artifact string from the table, e.g. "`routh_tests.rs::test_stable`".
    pub artifact_raw: String,
    /// Individual artifact locators parsed from the artifact cell.
    pub locators: Vec<String>,
    /// Relative path of the owning document.
    pub doc_path: String,
    /// Document review status ("Approved", "Reviewed", "Draft").
    pub doc_status: String,
}

/// Parsed verification content of a single design document.
#[derive(Debug, Clone, Default)]
pub struct ParsedDesignDoc {
    /// Relative path of the document.
    pub path: String,
    /// Requirements declared in Section 2.
    pub requirements: Vec<DeclaredRequirement>,
    /// Mappings declared in Section 6.4.
    pub trace_rows: Vec<TraceabilityRow>,
    /// Requirement IDs or keywords explicitly mentioned as unverified in Section 6.7.
    pub unverified_text: String,
    /// Document status ("Draft", "Reviewed", "Approved", or unknown).
    pub status: String,
}

impl DeclaredRequirement {
    /// Fully qualified requirement identifier, e.g. "error-design#FR-1".
    #[must_use]
    #[cfg(test)]
    pub fn qualified_id(&self) -> String {
        format!("{}#{}", doc_slug(&self.doc_path), self.id)
    }
}

impl TraceabilityRow {
    /// Fully qualified requirement identifier, e.g. "error-design#FR-1".
    #[must_use]
    #[cfg(test)]
    pub fn qualified_id(&self) -> String {
        format!("{}#{}", doc_slug(&self.doc_path), self.req_id)
    }
}

/// Extracts a canonical document slug from a path (e.g. "documentation/math/error-design.md" -> "error-design").
#[must_use]
pub fn doc_slug(doc_path: &str) -> String {
    let filename = doc_path.split('/').next_back().unwrap_or(doc_path);
    filename.trim_end_matches(".md").to_string()
}

/// Recursively find all `*.md` files under `dir`, skipping hidden directories and `target/`.
#[must_use]
pub fn find_design_docs(dir: &Path) -> Vec<PathBuf> {
    let mut docs = Vec::new();
    find_docs_recursive(dir, &mut docs);
    docs.sort();
    docs
}

fn find_docs_recursive(dir: &Path, acc: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.') && name != "target" && name != "scratch" {
                find_docs_recursive(&path, acc);
            }
        } else if path.is_file()
            && path.extension().and_then(|e| e.to_str()) == Some("md")
        {
            let filename =
                path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Exclude templates or general indexes if they don't declare requirements
            if filename != "design-template.md" {
                acc.push(path);
            }
        }
    }
}

/// Normalize requirement ID, e.g. "FR-1 — Routh Stability" -> "FR-1", "NFR-2" -> "NFR-2".
#[must_use]
pub fn extract_req_id(raw: &str) -> Option<String> {
    let Ok(re) = Regex::new(r"\b((?:FR|NFR|C|VAL)-\d+[a-z]?)\b") else {
        return None;
    };
    re.captures(raw)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_uppercase())
}

/// Parse a single Markdown design document into [`ParsedDesignDoc`].
#[must_use]
pub fn parse_design_doc(path: &Path, root: &Path) -> Option<ParsedDesignDoc> {
    let content = fs::read_to_string(path).ok()?;
    let rel_path = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();

    let status = parse_doc_status(&content);
    let requirements = parse_requirements_section(&content, &rel_path, &status);
    let trace_rows = parse_traceability_table(&content, &rel_path, &status);
    let unverified_text = parse_unverified_section(&content);

    // Only consider documents that have either declared requirements or a trace table
    if requirements.is_empty() && trace_rows.is_empty() {
        return None;
    }

    Some(ParsedDesignDoc {
        path: rel_path,
        requirements,
        trace_rows,
        unverified_text,
        status,
    })
}

fn parse_doc_status(content: &str) -> String {
    let lower = content.to_lowercase();
    if lower.contains("status-approved")
        || lower.contains("doc%20status-approved")
    {
        "Approved".to_string()
    } else if lower.contains("status-reviewed")
        || lower.contains("doc%20status-reviewed")
    {
        "Reviewed".to_string()
    } else if lower.contains("status-draft")
        || lower.contains("doc%20status-draft")
    {
        "Draft".to_string()
    } else {
        "Unknown".to_string()
    }
}

fn parse_requirements_section(
    content: &str,
    doc_path: &str,
    doc_status: &str,
) -> Vec<DeclaredRequirement> {
    let mut reqs = Vec::new();
    let Ok(req_header_re) = Regex::new(r"(?m)^#{2,4}\s+2\.\s+Requirements")
    else {
        return reqs;
    };
    let Some(header_match) = req_header_re.find(content) else {
        return reqs;
    };

    let section_start = header_match.end();
    // Section 2 ends at the next major section header, e.g. `## 3.` or `### 3.`
    let section_end = Regex::new(r"(?m)^#{2,4}\s+3\.")
        .ok()
        .and_then(|re| re.find(content.get(section_start..).unwrap_or("")))
        .map_or(content.len(), |m| section_start.saturating_add(m.start()));

    let req_content = content.get(section_start..section_end).unwrap_or("");
    let Ok(bullet_re) = Regex::new(
        r"(?m)^\s*[-*]\s+\*\*([A-Za-z0-9_-]+)\s*(?:[—–-]\s*([^:*]+))?\*\*\s*:\s*(.*)$",
    ) else {
        return reqs;
    };

    for cap in bullet_re.captures_iter(req_content) {
        let Some(raw_id) = cap.get(1).map(|m| m.as_str()) else {
            continue;
        };
        if let Some(id) = extract_req_id(raw_id) {
            let title = requirement_title(&cap);
            reqs.push(DeclaredRequirement {
                id,
                title,
                doc_path: doc_path.to_string(),
                doc_status: doc_status.to_string(),
            });
        }
    }

    reqs
}

fn requirement_title(cap: &regex::Captures<'_>) -> String {
    cap.get(2).map_or_else(
        || {
            cap.get(3)
                .map(|m| {
                    let s = m.as_str().trim();
                    s.split('.').next().unwrap_or(s).to_string()
                })
                .unwrap_or_default()
        },
        |m| m.as_str().trim().to_string(),
    )
}

fn parse_traceability_table(
    content: &str,
    doc_path: &str,
    doc_status: &str,
) -> Vec<TraceabilityRow> {
    let mut rows = Vec::new();
    let Ok(table_header_re) = Regex::new(r"(?m)^#{2,4}\s+6\.4\s+Traceability")
    else {
        return rows;
    };
    let Some(header_match) = table_header_re.find(content) else {
        return rows;
    };

    let section_start = header_match.end();
    let section_end = Regex::new(r"(?m)^#{2,4}\s+6\.[5-9]")
        .ok()
        .and_then(|re| re.find(content.get(section_start..).unwrap_or("")))
        .map_or(content.len(), |m| section_start.saturating_add(m.start()));

    let table_content = content.get(section_start..section_end).unwrap_or("");
    for line in table_content.lines() {
        rows.extend(parse_trace_table_line(line.trim(), doc_path, doc_status));
    }

    rows
}

fn parse_trace_table_line(
    trimmed: &str,
    doc_path: &str,
    doc_status: &str,
) -> Vec<TraceabilityRow> {
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
        return Vec::new();
    }
    let cells: Vec<&str> = trimmed
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect();

    let Some(raw_req) = cells.first().copied() else {
        return Vec::new();
    };
    let Some(method) = cells.get(1).map(ToString::to_string) else {
        return Vec::new();
    };
    let Some(raw_artifact) = cells.get(2).copied() else {
        return Vec::new();
    };

    if raw_req.eq_ignore_ascii_case("requirement")
        || raw_req.starts_with(":-")
        || raw_req.starts_with("--")
    {
        return Vec::new();
    }

    let req_ids = parse_requirement_list_or_range(raw_req);
    let locators = parse_locators_from_cell(raw_artifact);
    let mut rows = Vec::new();
    for req_id in req_ids {
        rows.push(TraceabilityRow {
            req_raw: raw_req.to_string(),
            req_id,
            method: method.clone(),
            artifact_raw: raw_artifact.to_string(),
            locators: locators.clone(),
            doc_path: doc_path.to_string(),
            doc_status: doc_status.to_string(),
        });
    }
    rows
}

/// Parse requirement references from a cell, handling ranges like "FR-1 .. FR-8" or "FR-1, FR-2".
fn parse_requirement_list_or_range(raw: &str) -> Vec<String> {
    let mut results = Vec::new();

    let Ok(range_re) =
        Regex::new(r"([A-Za-z]+)-(\d+)\s*\.\.\s*([A-Za-z]+)-(\d+)")
    else {
        if let Some(id) = extract_req_id(raw) {
            results.push(id);
        }
        return results;
    };

    if expand_requirement_range(raw, &range_re, &mut results) {
        return results;
    }

    let Ok(re) = Regex::new(r"\b((?:FR|NFR|C|VAL)-\d+[a-z]?)\b") else {
        return results;
    };
    for cap in re.captures_iter(raw) {
        let Some(id) = cap.get(1).map(|m| m.as_str().to_uppercase()) else {
            continue;
        };
        if !results.contains(&id) {
            results.push(id);
        }
    }

    results
}

fn expand_requirement_range(
    raw: &str,
    range_re: &Regex,
    results: &mut Vec<String>,
) -> bool {
    let Some(caps) = range_re.captures(raw) else {
        return false;
    };
    let Some(prefix1) = caps.get(1).map(|m| m.as_str()) else {
        return false;
    };
    let start: usize = caps
        .get(2)
        .and_then(|m| m.as_str().parse().ok())
        .unwrap_or(0);
    let Some(prefix2) = caps.get(3).map(|m| m.as_str()) else {
        return false;
    };
    let end: usize = caps
        .get(4)
        .and_then(|m| m.as_str().parse().ok())
        .unwrap_or(0);

    if prefix1.eq_ignore_ascii_case(prefix2)
        && start <= end
        && end.saturating_sub(start) <= 50
    {
        for i in start..=end {
            results.push(format!("{}-{}", prefix1.to_uppercase(), i));
        }
        return true;
    }
    false
}

fn parse_locators_from_cell(raw: &str) -> Vec<String> {
    let mut locators = Vec::new();

    let clean = raw.replace('`', "");
    let normalized = clean.replace("<br>", "\n").replace("<br/>", "\n");
    for line in normalized.lines() {
        for part in line.split([',', ';']) {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                locators.push(trimmed.to_string());
            }
        }
    }

    locators
}

fn parse_unverified_section(content: &str) -> String {
    let Ok(header_re) = Regex::new(r"(?m)^#{2,4}\s+6\.7\s+Not\s+verified")
    else {
        return String::new();
    };
    let Some(header_match) = header_re.find(content) else {
        return String::new();
    };

    let start = header_match.end();
    let end = Regex::new(r"(?m)^#{2,4}\s+[7-9]\.")
        .ok()
        .and_then(|re| re.find(content.get(start..).unwrap_or("")))
        .map_or(content.len(), |m| start.saturating_add(m.start()));

    content.get(start..end).unwrap_or("").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_req_id() {
        assert_eq!(extract_req_id("FR-1"), Some("FR-1".to_string()));
        assert_eq!(extract_req_id("FR-1 — Title"), Some("FR-1".to_string()));
        assert_eq!(extract_req_id("NFR-23"), Some("NFR-23".to_string()));
        assert_eq!(extract_req_id("C-4"), Some("C-4".to_string()));
        assert_eq!(extract_req_id("Not a req"), None);
    }

    #[test]
    fn test_parse_requirements_section() {
        let doc = r"
# Sample Doc

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Native Numerical Math**: Must do math.
- **FR-2 — State Space Models**: Must support state space.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Allocation**: Must not allocate.
- **C-1 — Native Rust**: Must be native.

### 3. Technical Overview
";
        let reqs = parse_requirements_section(doc, "sample.md", "Approved");
        assert_eq!(reqs.len(), 4);
        assert_eq!(reqs.first().unwrap().id, "FR-1");
        assert_eq!(reqs.first().unwrap().qualified_id(), "sample#FR-1");
        assert_eq!(reqs.first().unwrap().title, "Native Numerical Math");
        assert_eq!(reqs.get(2).unwrap().id, "NFR-1");
        assert_eq!(reqs.get(3).unwrap().id, "C-1");
    }

    #[test]
    fn test_parse_traceability_table() {
        let doc = r"
### 6.4 Traceability

| Requirement | Method | Artifact |
|:------------|:-------|:---------|
| FR-1 | Requirements-based test | `tests/test_math.rs::test_add`, `test_sub` |
| FR-2 .. FR-3 | Property-based test | `tests/prop_tests.rs::prop_check` |
| NFR-1 | Static analysis | `cargo clippy-ci` |

### 6.5 Coverage
";
        let rows = parse_traceability_table(doc, "sample.md", "Approved");
        assert_eq!(rows.len(), 4); // FR-1, FR-2, FR-3, NFR-1
        assert_eq!(rows.first().unwrap().req_id, "FR-1");
        assert_eq!(rows.first().unwrap().qualified_id(), "sample#FR-1");
        assert_eq!(rows.first().unwrap().locators.len(), 2);
        assert_eq!(rows.get(1).unwrap().req_id, "FR-2");
        assert_eq!(rows.get(2).unwrap().req_id, "FR-3");
        assert_eq!(rows.get(3).unwrap().req_id, "NFR-1");
    }
}
