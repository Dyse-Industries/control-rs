//! Source markers, found by reading source text.
//!
//! A line that contains the configured marker text starts a marker span,
//! extended while a parenthesis opened after the marker text is unclosed. The
//! span yields one `marker` row per qualified ID `<doc>#<id>` in it, with the
//! span as its text. The scan reads text only, so markers work in any source language and
//! do not depend on what a build compiles.

use std::path::Path;

use super::reqs::{MarkedId, MarkerConfig, Rules, row};
use super::select::select;
use super::{Defect, Defects, MARKER, Rows, read_text, sort_rows};
use crate::error::GateResult;

const ATTR_OPEN: &str = concat!("#", "[");
const RUST_REQ: &str = concat!("#", "[", "req(");

/// Maximum number of lines in one marker span.
const SPAN_LINES: usize = 16;

/// Marker rows and the defects of marker spans.
pub type Scan = (Rows, Defects);

/// A marker span: its text, the index of its last line and whether it closed.
type Span = (String, usize, bool);

/// Matched marker: slice after the delimiter and start position.
type MarkerMatch<'a> = (&'a str, usize);

/// Open minus close parentheses in `text`.
fn paren_balance(text: &str) -> isize {
    text.chars().fold(0, |bal: isize, c| match c {
        '(' => bal.saturating_add(1),
        ')' => bal.saturating_sub(1),
        _ => bal,
    })
}

/// Finds the start of the marker and the slice following its opening delimiter.
fn find_marker<'a>(line: &'a str, marker: &str) -> Option<MarkerMatch<'a>> {
    if marker == RUST_REQ {
        let mut search_from = 0;
        while let Some(open_bracket) = line[search_from..].find(ATTR_OPEN) {
            let start = search_from.saturating_add(open_bracket);
            let after_bracket = &line[start.saturating_add(2)..];
            let trimmed = after_bracket.trim_start();
            let rest = trimmed.rfind("::").map_or(trimmed, |path_end| {
                &trimmed[path_end.saturating_add(2)..]
            });
            if let Some(req_rest) = rest.strip_prefix("req") {
                let req_trimmed = req_rest.trim_start();
                if let Some(paren_rest) = req_trimmed.strip_prefix('(') {
                    let match_end = line.len().saturating_sub(paren_rest.len());
                    return Some((&line[match_end..], start));
                }
            }
            search_from = start.saturating_add(2);
        }
        None
    } else {
        line.find(marker).map(|pos| {
            let after = line
                .get(pos.saturating_add(marker.len())..)
                .unwrap_or_default();
            (after, pos)
        })
    }
}

/// The marker span that starts at `lines[idx]`: its text, the index of its
/// last line and whether it closed within [`SPAN_LINES`].
///
/// The span extends past the marker line only while a parenthesis opened
/// after the marker text is unclosed.
fn span(lines: &[&str], idx: usize, marker: &str) -> Span {
    let first = lines.get(idx).copied().unwrap_or_default();
    let mut text = first.to_string();
    let (after, _) = find_marker(first, marker).unwrap_or_default();
    let has_paren = marker.contains('(') || marker == RUST_REQ;
    let mut balance =
        paren_balance(after).saturating_add(isize::from(has_paren));
    let mut last = idx;
    while balance > 0 {
        let next = last.saturating_add(1);
        let Some(line) = lines.get(next) else {
            return (text, last, false);
        };
        if next.saturating_sub(idx) >= SPAN_LINES {
            return (text, last, false);
        }
        text.push('\n');
        text.push_str(line);
        balance = balance.saturating_add(paren_balance(line));
        last = next;
    }
    (text, last, true)
}

/// The marker rows of one source file and the defects of its marker spans.
///
/// A span with no ID, an ID written without `<doc>#`, an ID with a tag and a
/// span still unclosed at 16 lines are defects.
#[must_use]
pub fn scan_source(
    file: &str,
    source: &str,
    marker: &str,
    rules: &Rules,
) -> Scan {
    let mut rows = Vec::new();
    let mut defects = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut idx = 0;
    while let Some(&line) = lines.get(idx) {
        if find_marker(line, marker).is_none() {
            idx = idx.saturating_add(1);
            continue;
        }
        let at = (file, idx.saturating_add(1));
        let (text, last, closed) = span(&lines, idx, marker);
        let defect = |message: String| Defect {
            file: file.to_string(),
            line: at.1,
            message,
        };
        if !closed {
            defects.push(defect(format!(
                "marker span is unclosed at {SPAN_LINES} lines"
            )));
        }
        let mut ids = rules.marked_ids(&text);
        if ids.is_empty() {
            defects.push(defect("marker names no requirement ID".to_string()));
        }
        let mut seen = std::collections::BTreeSet::new();
        ids.retain(|id| seen.insert(id.clone()));
        for id in ids {
            match id {
                MarkedId::Qualified(id) => {
                    rows.push(row(id, MARKER, at, &text));
                }
                MarkedId::Unqualified(id) => defects.push(defect(format!(
                    "marker ID {id} is not qualified as <doc>#<id>"
                ))),
                MarkedId::Tagged(id) => defects.push(defect(format!(
                    "marker ID {id} carries a tag, reserved in this revision"
                ))),
            }
        }
        idx = last.saturating_add(1);
    }
    (rows, defects)
}

/// The marker rows and defects of every source file that `config` selects
/// below `base`, rows sorted by file, then line.
///
/// # Errors
/// `GateError::Config` for a root that is not a file or directory and
/// `GateError::Io` if a file cannot be read.
pub fn collect(
    base: &Path,
    config: &MarkerConfig,
    rules: &Rules,
) -> GateResult<Scan> {
    let mut rows = Vec::new();
    let mut defects = Vec::new();
    for file in select(base, &config.files, &config.suffixes)? {
        let source = read_text(&base.join(&file))?;
        let (found, wrong) = scan_source(&file, &source, &config.marker, rules);
        rows.extend(found);
        defects.extend(wrong);
    }
    sort_rows(&mut rows);
    Ok((rows, defects))
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_rs_trace_macros::req;

    use crate::trace::reqs::TraceConfig;

    const CONFIG: &str = r"id = '(?:FR|NFR|C)-[0-9]+[a-z]?'
condition = 'VC-[0-9]+(?:\.[0-9]+[a-z]?)?'
doc = '[a-z0-9-]+'
files = []
definition = '^- \*\*'
verification = '^\|'
methods = ['test']
marked_methods = ['test']
retired = []
";

    const PFX: &str = concat!("#[", "req(",);

    fn rules() -> Rules {
        let config: TraceConfig = toml::from_str(CONFIG).unwrap();
        config.rules(Path::new("trace.toml")).unwrap()
    }

    #[test]
    fn a_marker_line_yields_one_row_per_qualified_id() {
        let source =
            format!("\n{PFX}\"widget#FR-1\", \"widget#NFR-2\")]\n#[test]\n");
        let (rows, defects) = scan_source("src/a.rs", &source, PFX, &rules());
        let ids: Vec<_> = rows.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["widget#FR-1", "widget#NFR-2"]);
        assert!(rows.iter().all(|r| r.kind == MARKER && r.line == 2));
        assert_eq!(
            rows.first().map(|r| r.text.as_str()),
            Some(format!("{PFX}\"widget#FR-1\", \"widget#NFR-2\")]").as_str())
        );
        assert!(defects.is_empty());
    }

    #[test]
    fn any_language_can_carry_markers() {
        let source = "int f(void);\n// req: widget#FR-3\nint g(void);\n";
        let (rows, _) = scan_source("src/g.c", source, "// req:", &rules());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows.first().map(|r| r.line), Some(2));
    }

    #[test]
    fn unqualified_or_missing_ids_on_a_marker_line_are_defects() {
        let source = format!("{PFX}\"FR-1\")]\n{PFX})]\nfn f() {{}}\n");
        let (rows, defects) = scan_source("src/a.rs", &source, PFX, &rules());
        assert!(rows.is_empty());
        let messages: Vec<_> =
            defects.iter().map(ToString::to_string).collect();
        assert_eq!(
            messages,
            [
                "src/a.rs:1: marker ID FR-1 is not qualified as <doc>#<id>",
                "src/a.rs:2: marker names no requirement ID",
            ]
        );
    }

    #[test]
    fn lines_without_the_marker_text_are_ignored() {
        let source = "// widget#FR-1 is mentioned here\nfn f() {}\n";
        let (rows, defects) = scan_source("src/a.rs", source, PFX, &rules());
        assert!(rows.is_empty() && defects.is_empty());
    }

    #[req("requirement-traceability#VC-7.1")]
    #[test]
    fn comment_markers_stop_at_their_line_and_attributes_span_to_close() {
        let c = "// req: widget#VC-1.1\nint g(int x) { return f(x); }\n";
        let (rows, defects) = scan_source("src/g.c", c, "// req:", &rules());
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows.first().map(|r| r.text.as_str()),
            Some("// req: widget#VC-1.1")
        );
        assert!(defects.is_empty());
        let py = "# req: widget#VC-1.1\ndef g(x):\n    return f(x)\n";
        let (rows, _) = scan_source("src/g.py", py, "# req:", &rules());
        assert_eq!(
            rows.first().map(|r| r.text.as_str()),
            Some("# req: widget#VC-1.1")
        );
        let rust = format!(
            "{PFX}\n    \"widget#VC-1.1\",\n    \"widget#VC-2.1\"\n)]\nfn t(x: (u8, u8)) {{}}\n"
        );
        let (rows, defects) = scan_source("src/a.rs", &rust, PFX, &rules());
        let ids: Vec<_> = rows.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["widget#VC-1.1", "widget#VC-2.1"]);
        assert!(rows.iter().all(|r| r.line == 1 && !r.text.contains("fn t")));
        assert!(defects.is_empty());
    }

    #[req("requirement-traceability#VC-7.2")]
    #[test]
    fn tagged_ids_and_unclosed_spans_are_defects() {
        let tagged = format!("{PFX}\"widget#VC-1.1=true\")]\n");
        let (rows, defects) = scan_source("src/a.rs", &tagged, PFX, &rules());
        assert!(rows.is_empty());
        let messages: Vec<_> =
            defects.iter().map(ToString::to_string).collect();
        assert_eq!(
            messages,
            [
                "src/a.rs:1: marker ID widget#VC-1.1 carries a tag, reserved in \
              this revision"
            ]
        );
        let open = format!("{PFX}\"widget#VC-1.1\",\n{}", "//\n".repeat(20));
        let (_, defects) = scan_source("src/a.rs", &open, PFX, &rules());
        let messages: Vec<_> =
            defects.iter().map(ToString::to_string).collect();
        assert_eq!(
            messages,
            ["src/a.rs:1: marker span is unclosed at 16 lines"]
        );
    }

    #[test]
    fn duplicate_id_in_same_span_is_deduplicated() {
        let source = format!(
            "{PFX}\"widget#VC-1.1\", \"widget#VC-1.1\")]\nfn test_a() {{}}\n"
        );
        let (rows, defects) = scan_source("src/a.rs", &source, PFX, &rules());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows.first().map(|r| r.id.as_str()), Some("widget#VC-1.1"));
        assert!(defects.is_empty());
    }

    #[test]
    fn qualified_req_macro_and_spaces_are_recognized() {
        let hash = "#";
        let cases = [
            format!(
                "{hash}[control_rs_trace_macros::req(\"widget#VC-1.1\")]\nfn t1() {{}}\n"
            ),
            format!("{hash}[req (\"widget#VC-1.1\")]\nfn t2() {{}}\n"),
            format!("{hash}[crate::req(\"widget#VC-1.1\")]\nfn t3() {{}}\n"),
        ];
        for source in cases {
            let (rows, defects) =
                scan_source("src/a.rs", &source, PFX, &rules());
            assert_eq!(rows.len(), 1, "failed for: {source}");
            assert_eq!(
                rows.first().map(|r| r.id.as_str()),
                Some("widget#VC-1.1")
            );
            assert!(defects.is_empty());
        }
    }

    #[test]
    fn paren_balance_handles_nested_and_unbalanced_parentheses() {
        assert_eq!(paren_balance(""), 0);
        assert_eq!(paren_balance("abc"), 0);
        assert_eq!(paren_balance("(a (b) c)"), 0);
        assert_eq!(paren_balance("((a)"), 1);
        assert_eq!(paren_balance("(((a)"), 2);
        assert_eq!(paren_balance(")"), -1);
        assert_eq!(paren_balance("))"), -2);

        let nested = format!("{PFX}(\"widget#VC-1.1\"))]\nfn t() {{}}\n");
        let (rows, defects) = scan_source("src/a.rs", &nested, PFX, &rules());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows.first().map(|r| r.id.as_str()), Some("widget#VC-1.1"));
        assert!(defects.is_empty());
    }
}
