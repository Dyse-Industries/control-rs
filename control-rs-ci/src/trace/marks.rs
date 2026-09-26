//! Source markers, found by reading source text.
//!
//! A line that contains the configured marker text is a marker. It yields one
//! `marker` row per qualified ID `<doc>#<id>` on it, with the line as its
//! text. The scan reads text only, so markers work in any source language and
//! do not depend on what a build compiles.

use std::path::Path;

use super::reqs::{MarkerConfig, Rules, row};
use super::select::select;
use super::{Defect, Defects, MARKER, Rows, read_text, sort_rows};
use crate::error::GateResult;

/// Marker rows and the defects of marker lines.
pub type Scan = (Rows, Defects);

/// The marker rows of one source file and the defects of its marker lines.
///
/// A marker line with an ID written without `<doc>#`, or with no ID at all,
/// is a defect.
#[must_use]
pub fn scan_source(
    file: &str,
    source: &str,
    marker: &str,
    rules: &Rules,
) -> Scan {
    let mut rows = Vec::new();
    let mut defects = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        if !line.contains(marker) {
            continue;
        }
        let at = (file, idx.saturating_add(1));
        let ids = rules.marked_ids(line);
        let defect = |message: String| Defect {
            file: file.to_string(),
            line: at.1,
            message,
        };
        if ids.is_empty() {
            defects.push(defect("marker names no requirement ID".to_string()));
        }
        for id in ids {
            match id {
                Ok(id) => rows.push(row(id, MARKER, at, line)),
                Err(id) => defects.push(defect(format!(
                    "marker ID {id} is not qualified as <doc>#<id>"
                ))),
            }
        }
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
    use crate::trace::reqs::TraceConfig;

    const CONFIG: &str = r"id = '(?:FR|NFR|C)-[0-9]+[a-z]?'
doc = '[a-z0-9-]+'
files = []
definition = '^- \*\*'
retired = []

[references]
";

    fn rules() -> Rules {
        let config: TraceConfig = toml::from_str(CONFIG).unwrap();
        config.rules(Path::new("trace.toml")).unwrap()
    }

    #[test]
    fn a_marker_line_yields_one_row_per_qualified_id() {
        let source = "\n#[req(\"widget#FR-1\", \"widget#NFR-2\")]\n#[test]\n";
        let (rows, defects) =
            scan_source("src/a.rs", source, "#[req(", &rules());
        let ids: Vec<_> = rows.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["widget#FR-1", "widget#NFR-2"]);
        assert!(rows.iter().all(|r| r.kind == MARKER && r.line == 2));
        assert_eq!(
            rows.first().map(|r| r.text.as_str()),
            Some("#[req(\"widget#FR-1\", \"widget#NFR-2\")]")
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
        let source = "#[req(\"FR-1\")]\n#[req()]\nfn f() {}\n";
        let (rows, defects) =
            scan_source("src/a.rs", source, "#[req(", &rules());
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
        let (rows, defects) =
            scan_source("src/a.rs", source, "#[req(", &rules());
        assert!(rows.is_empty() && defects.is_empty());
    }
}
