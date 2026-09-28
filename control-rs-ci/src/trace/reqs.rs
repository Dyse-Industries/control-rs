//! Requirement definitions and verification conditions in Markdown.
//!
//! The matching rules are the whole parser. Lines inside fenced code blocks
//! are skipped. A line that matches `definition` defines the first
//! requirement ID on it, and its text runs on through the indented lines that
//! follow. A line that matches `verification` defines the condition ID on it;
//! every other requirement ID on the line is a parent and the code span that
//! names a method is its method.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use regex::Regex;
use serde::Deserialize;

use super::{CONDITION, DEFINITION, Defect, Row, SCHEMA, normalize, read_text};
use crate::error::{GateError, GateResult};

/// A file and line.
pub(super) type At<'a> = (&'a str, usize);

/// Definition rows by ID.
type Definitions<'a> = BTreeMap<&'a str, Vec<&'a Row>>;

/// A fence: its character and length.
type Fence = (char, usize);

/// An open fence: its fence specification and 1-based start line.
type OpenFence = (Fence, usize);

/// A line outside fenced code blocks: its 1-based number and text.
type Line<'a> = (usize, &'a str);

/// Visible lines and an optional unclosed fence start line.
type VisibleLines<'a> = (Vec<Line<'a>>, Option<usize>);

/// Scanned rows and scan defects of a Markdown document.
pub type MarkdownScan = (Vec<Row>, Vec<Defect>);

/// One ID occurrence in a marker span.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MarkedId {
    /// A qualified ID, `<doc>#<id>`.
    Qualified(String),
    /// An ID written without `<doc>#`.
    Unqualified(String),
    /// An ID followed by `=<tag>`, reserved for the extension path.
    Tagged(String),
}

/// A `trace.toml`: the patterns that find requirements in Markdown.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceConfig {
    /// Pattern for one requirement ID.
    pub id: String,
    /// Pattern for one verification condition ID (for example, `VC-1.1`).
    pub condition: String,
    /// Pattern for the document name in a qualified ID `<doc>#<id>`.
    pub doc: String,
    /// Markdown files, and directories whose `<doc_suffix>.md` files are
    /// read.
    pub files: Vec<String>,
    /// Text removed from a file stem to form the document name.
    #[serde(default)]
    pub doc_suffix: String,
    /// Pattern for a line that defines a requirement.
    pub definition: String,
    /// Pattern for a line that defines a verification condition.
    pub verification: String,
    /// Verification methods a condition may name.
    pub methods: Vec<String>,
    /// The subset of methods whose conditions need at least one marker.
    pub marked_methods: Vec<String>,
    /// Item rules and result artifacts per method.
    #[serde(default)]
    pub method: BTreeMap<String, MethodConfig>,
    /// Qualified IDs that must not be defined or referenced again.
    pub retired: Vec<String>,
    /// Where `trace-marks` finds markers; none when absent.
    #[serde(default)]
    pub markers: Option<MarkerConfig>,
}

/// The `[method.<name>]` table of a `trace.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodConfig {
    /// Regex pattern matching the item attribute or declaration (for example, `#[test]`).
    #[serde(default)]
    pub item_rule: Option<String>,
    /// Relative path to the verification result log artifact (for example, `target/ci-artifacts/test.log`).
    #[serde(default)]
    pub result_artifact: Option<String>,
}

/// The `[markers]` table of a `trace.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkerConfig {
    /// Source files, and directories whose files ending in a suffix are read.
    pub files: Vec<String>,
    /// File-name endings of the source files to read, such as `.rs`.
    pub suffixes: Vec<String>,
    /// Text that makes a line a marker attribute or prefix.
    pub marker: String,
}

/// The compiled patterns of a [`TraceConfig`].
#[derive(Debug, Clone)]
pub struct Rules {
    /// One ID occurrence: `trace_doc` is the optional document and
    /// `trace_id` the ID.
    occurrence: Regex,
    /// Pattern matching condition IDs.
    condition_occurrence: Regex,
    /// A defining line.
    definition: Regex,
    /// A verification condition line.
    verification: Regex,
    /// Verification methods a condition may name.
    methods: Vec<String>,
    /// Text removed from a file stem to form its document name.
    doc_suffix: String,
    /// IDs that must not appear.
    retired: BTreeSet<String>,
}

struct ScanContext<'a> {
    doc: &'a str,
    rules: &'a Rules,
}

impl TraceConfig {
    /// The file-name ending of the Markdown files a directory root selects.
    #[must_use]
    pub fn doc_suffixes(&self) -> Vec<String> {
        vec![format!("{}.md", self.doc_suffix)]
    }

    /// Reads and parses a `trace.toml`.
    ///
    /// # Errors
    /// `GateError::Io` if the file cannot be read and `GateError::Config` if
    /// it does not parse.
    pub fn load(path: &Path) -> GateResult<Self> {
        let text = read_text(path)?;
        toml::from_str(&text).map_err(|e| GateError::Config {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
    }

    /// Compiles the patterns; `path` names the configuration in errors.
    ///
    /// # Errors
    /// `GateError::Config` for an invalid pattern.
    pub fn rules(&self, path: &Path) -> GateResult<Rules> {
        let error = |message: String| GateError::Config {
            path: path.to_path_buf(),
            message,
        };
        let compile = |key: &str, pattern: &str| {
            Regex::new(pattern).map_err(|e| {
                error(format!("`{key}` is not a valid pattern: {e}"))
            })
        };
        if let Some(m) = self
            .marked_methods
            .iter()
            .find(|m| !self.methods.contains(m))
        {
            return Err(error(format!(
                "marked method `{m}` is not in `methods`"
            )));
        }
        compile("doc", &self.doc)?;
        compile("id", &self.id)?;
        compile("condition", &self.condition)?;
        let occurrence = format!(
            r"(?:(?P<trace_doc>{})#)?\b(?P<trace_id>{})\b",
            self.doc, self.id
        );
        let condition_occurrence = format!(
            r"(?:(?P<trace_doc>{})#)?\b(?P<trace_id>{})\b",
            self.doc, self.condition
        );
        Ok(Rules {
            occurrence: compile("id", &occurrence)?,
            condition_occurrence: compile("condition", &condition_occurrence)?,
            definition: compile("definition", &self.definition)?,
            verification: compile("verification", &self.verification)?,
            methods: self.methods.clone(),
            doc_suffix: self.doc_suffix.clone(),
            retired: self.retired.iter().cloned().collect(),
        })
    }
}

impl Rules {
    /// The qualified requirement IDs on `line`, left to right; an ID written
    /// without `<doc>#` belongs to `doc`.
    pub fn ids<'a>(
        &'a self,
        line: &'a str,
        doc: &'a str,
    ) -> impl Iterator<Item = String> + 'a {
        self.occurrence.captures_iter(line).filter_map(move |caps| {
            let id = caps.name("trace_id")?.as_str();
            let owner = caps.name("trace_doc").map_or(doc, |m| m.as_str());
            Some(format!("{owner}#{id}"))
        })
    }

    /// Whether `name` is one of the configured verification methods.
    #[must_use]
    pub fn is_method(&self, name: &str) -> bool {
        self.methods.iter().any(|m| m == name)
    }

    /// The ID occurrences in a marker span, left to right.
    #[must_use]
    pub fn marked_ids(&self, text: &str) -> Vec<MarkedId> {
        let cond_matches = self.condition_occurrence.captures_iter(text);
        let req_matches = self.occurrence.captures_iter(text);
        let mut matches: Vec<_> = req_matches.chain(cond_matches).collect();
        matches.sort_by_key(|caps| caps.get(0).map_or(0, |m| m.start()));
        matches
            .into_iter()
            .filter_map(|caps| {
                let whole = caps.get(0)?;
                let id = caps.name("trace_id")?.as_str();
                let written = whole.as_str();
                let tagged = text
                    .get(whole.end()..)
                    .is_some_and(|rest| rest.starts_with('='));
                Some(if tagged {
                    MarkedId::Tagged(written.to_string())
                } else if let Some(doc) = caps.name("trace_doc") {
                    MarkedId::Qualified(format!("{}#{id}", doc.as_str()))
                } else {
                    MarkedId::Unqualified(id.to_string())
                })
            })
            .collect()
    }
}

/// The document name of `file`: its stem without `suffix`.
#[must_use]
pub fn doc_name(file: &str, suffix: &str) -> String {
    let stem = Path::new(file)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    stem.strip_suffix(suffix)
        .map_or_else(|| stem.clone(), str::to_owned)
}

/// The first cell of a Markdown table line.
fn first_cell(line: &str) -> &str {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('|').unwrap_or(trimmed);
    rest.split('|').next().unwrap_or(rest).trim()
}

fn scan_condition_line(
    line: &str,
    cell: &str,
    at: At<'_>,
    ctx: &ScanContext<'_>,
) -> Option<Row> {
    let cond_caps = ctx.rules.condition_occurrence.captures(cell)?;
    let c_id = cond_caps.name("trace_id")?.as_str();
    let c_doc = cond_caps.name("trace_doc").map_or(ctx.doc, |m| m.as_str());
    let cond_qualified = format!("{c_doc}#{c_id}");

    let mut parents: Vec<String> = ctx
        .rules
        .occurrence
        .captures_iter(line)
        .filter_map(|caps| {
            let id = caps.name("trace_id")?.as_str();
            let owner = caps.name("trace_doc").map_or(ctx.doc, |d| d.as_str());
            Some(format!("{owner}#{id}"))
        })
        .collect();

    let mut seen = BTreeSet::new();
    parents.retain(|p| seen.insert(p.clone()));

    // The method is the code span whose content is in `methods`; failing
    // that, the first code span that holds no ID, so that `check` reports it
    // as outside `methods`.
    let spans = super::code_spans(line);
    let method = spans
        .iter()
        .find(|s| ctx.rules.is_method(s.content))
        .or_else(|| {
            spans.iter().find(|s| {
                !ctx.rules.occurrence.is_match(s.content)
                    && !ctx.rules.condition_occurrence.is_match(s.content)
            })
        })
        .map(|s| s.content.to_string());

    let mut condition = row(cond_qualified, CONDITION, at, line);
    condition.parents = parents;
    condition.method = method;
    Some(condition)
}

/// The definition and condition rows of one Markdown document, and any scan defects.
///
/// `file` is the path recorded in each row; its stem, less the configured
/// suffix, names the document.
#[must_use]
pub fn scan_markdown(file: &str, source: &str, rules: &Rules) -> MarkdownScan {
    let doc = doc_name(file, &rules.doc_suffix);
    let ctx = ScanContext { doc: &doc, rules };
    let (lines, unclosed) = visible_lines(source);
    let mut rows = Vec::new();
    let mut defects = Vec::new();

    if let Some(open_line) = unclosed {
        defects.push(Defect {
            file: file.to_string(),
            line: open_line,
            message: "fenced code block is unclosed".to_string(),
        });
    }

    for (idx, &(number, line)) in lines.iter().enumerate() {
        let at = (file, number);
        if rules.definition.is_match(line)
            && let Some(id) = rules.ids(line, &doc).next()
        {
            rows.push(row(id, DEFINITION, at, &definition_text(&lines, idx)));
        } else if rules.verification.is_match(line) {
            let cell = first_cell(line);
            match scan_condition_line(line, cell, at, &ctx) {
                Some(row) => rows.push(row),
                None => defects.push(Defect {
                    file: file.to_string(),
                    line: number,
                    message: "verification row names no condition ID"
                        .to_string(),
                }),
            }
        }
    }

    if !rows.iter().any(|r| r.kind == DEFINITION) {
        defects.push(Defect {
            file: file.to_string(),
            line: 1,
            message: format!("document {doc} defines no requirements"),
        });
    }

    (rows, defects)
}

fn check_duplicate_definitions(
    definitions: &Definitions<'_>,
    conditions: &[&Row],
    defects: &mut Vec<Defect>,
) {
    for (&id, found) in definitions {
        if let [first, second, ..] = found.as_slice() {
            defects.push(Defect::at(
                second,
                format!(
                    "{id} is defined more than once; first definition at {}:{}",
                    first.file, first.line
                ),
            ));
        }
        if let Some(first) = found.first() {
            let has_cond =
                conditions.iter().any(|c| c.parents.iter().any(|p| p == id));
            if !has_cond {
                defects.push(Defect::at(
                    first,
                    format!("{id} has no verification condition"),
                ));
            }
        }
    }
}

fn check_duplicate_conditions(
    condition_defs: &Definitions<'_>,
    defects: &mut Vec<Defect>,
) {
    for (&id, found) in condition_defs {
        let mut seen_sites = std::collections::BTreeSet::new();
        let mut first_row: Option<&Row> = None;
        for row in found {
            if seen_sites.insert((&row.file, row.line)) {
                if first_row.is_none() {
                    first_row = Some(row);
                } else if let Some(first) = first_row {
                    defects.push(Defect::at(
                        row,
                        format!(
                            "condition {id} is defined more than once; first definition at {}:{}",
                            first.file, first.line
                        ),
                    ));
                    break;
                }
            }
        }
    }
}

fn check_condition_parents(
    conditions: &[&Row],
    definitions: &Definitions<'_>,
    defects: &mut Vec<Defect>,
) {
    for condition in conditions {
        if condition.parents.is_empty() {
            defects.push(Defect::at(
                condition,
                format!("condition {} has no parent requirement", condition.id),
            ));
        } else {
            for parent in &condition.parents {
                if !definitions.contains_key(parent.as_str()) {
                    defects.push(Defect::at(
                        condition,
                        format!(
                            "condition {} references undefined requirement {parent}",
                            condition.id
                        ),
                    ));
                }
            }
        }
    }
}

fn check_condition_methods(
    conditions: &[&Row],
    rules: &Rules,
    defects: &mut Vec<Defect>,
) {
    for condition in conditions {
        let id = &condition.id;
        let ids = rules
            .condition_occurrence
            .find_iter(&condition.text)
            .count();
        if ids > 1 {
            defects.push(Defect::at(
                condition,
                format!(
                    "verification row of {id} names more than one condition ID"
                ),
            ));
        }
        let methods = super::code_spans(&condition.text)
            .iter()
            .filter(|s| rules.is_method(s.content))
            .count();
        let message = match &condition.method {
            None => {
                Some(format!("condition {id} names no verification method"))
            }
            Some(m) if !rules.is_method(m) => {
                Some(format!("condition {id} names unknown method `{m}`"))
            }
            Some(_) if methods > 1 => {
                Some(format!("condition {id} names more than one method"))
            }
            Some(_) => None,
        };
        if let Some(message) = message {
            defects.push(Defect::at(condition, message));
        }
    }
}

/// The defects in `rows`, sorted by file, then line.
#[must_use]
pub fn check(rows: &[Row], rules: &Rules) -> Vec<Defect> {
    let mut definitions = Definitions::new();
    let mut condition_defs = Definitions::new();
    for row in rows.iter().filter(|r| r.kind == DEFINITION) {
        definitions.entry(row.id.as_str()).or_default().push(row);
    }
    for row in rows.iter().filter(|r| r.kind == CONDITION) {
        condition_defs.entry(row.id.as_str()).or_default().push(row);
    }
    let conditions: Vec<&Row> =
        rows.iter().filter(|r| r.kind == CONDITION).collect();
    let mut defects = Vec::new();

    check_duplicate_definitions(&definitions, &conditions, &mut defects);
    check_duplicate_conditions(&condition_defs, &mut defects);
    check_condition_parents(&conditions, &definitions, &mut defects);
    check_condition_methods(&conditions, rules, &mut defects);

    for row in rows {
        for id in std::iter::once(&row.id).chain(&row.parents) {
            if rules.retired.contains(id) {
                defects.push(Defect::at(row, format!("{id} is retired")));
            }
        }
    }

    defects.sort_by(|a, b| {
        (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line))
    });
    defects
}

/// The lines outside fenced code blocks, with their 1-based numbers, and
/// whether an unclosed fence was reached at EOF.
fn visible_lines(source: &str) -> VisibleLines<'_> {
    let mut fence: Option<OpenFence> = None;
    let mut lines = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        let line_num = idx.saturating_add(1);
        match fence {
            Some((open, _)) => {
                if closes(line, open) {
                    fence = None;
                }
            }
            None => match opens(line) {
                Some(open) => fence = Some((open, line_num)),
                None => lines.push((line_num, line)),
            },
        }
    }
    let unclosed = fence.map(|(_, line_num)| line_num);
    (lines, unclosed)
}

/// The fence character and length when `line` opens a fenced code block.
fn opens(line: &str) -> Option<Fence> {
    let text = line.trim_start();
    let fence = text.chars().next().filter(|&c| matches!(c, '`' | '~'))?;
    let len = text.chars().take_while(|&c| c == fence).count();
    let info = text.get(len..).unwrap_or_default();
    (len >= 3 && !(fence == '`' && info.contains('`'))).then_some((fence, len))
}

/// Whether `line` closes the block that `open` opened.
fn closes(line: &str, open: Fence) -> bool {
    let (fence, len) = open;
    let text = line.trim();
    text.chars().count() >= len && text.chars().all(|c| c == fence)
}

/// The text of the definition at `lines[idx]`: its line and the indented,
/// non-blank lines that follow it.
fn definition_text(lines: &[Line<'_>], idx: usize) -> String {
    let mut parts = Vec::new();
    for (pos, &(_, line)) in
        lines.get(idx..).unwrap_or_default().iter().enumerate()
    {
        let continues =
            line.starts_with([' ', '\t']) && !line.trim().is_empty();
        if pos > 0 && !continues {
            break;
        }
        parts.push(line);
    }
    parts.join("\n")
}

/// A row of the current schema at `at`, a file and line, with its text
/// normalized.
pub(super) fn row(id: String, kind: &str, at: At<'_>, text: &str) -> Row {
    Row {
        schema: SCHEMA,
        id,
        kind: kind.to_string(),
        parents: Vec::new(),
        method: None,
        file: at.0.to_string(),
        line: at.1,
        text: normalize(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_rs_trace_macros::req;

    const CLEAN: &str = "\
# Widget

- **FR-1 — Size**: The widget shall report its size.

| Condition | Requirement | Method | Criterion |
|:----------|:------------|:-------|:----------|
| VC-1.1    | FR-1        | `test` | Exact size match |
";

    const FILE: &str = "docs/widget-design.md";

    const KEYS: &str = r#"id = '(?:FR|NFR|C)-[0-9]+[a-z]?'
condition = 'VC-[0-9]+(?:\.[0-9]+[a-z]?)?'
doc = '[a-z0-9-]+'
files = ["docs/*-design.md"]
doc_suffix = "-design"
definition = '^- \*\*(?:FR|NFR|C)-'
verification = '^\| *(?:[a-z0-9-]+#)?VC-'
methods = ["test", "analysis", "inspection", "review"]
marked_methods = ["test"]
"#;

    /// A row's kind, ID and line.
    type Found = (String, String, usize);

    fn rules_with(keys: &str) -> Rules {
        let text = format!("{KEYS}{keys}");
        let config: TraceConfig = toml::from_str(&text).unwrap();
        config.rules(Path::new("trace.toml")).unwrap()
    }

    fn rules() -> Rules {
        rules_with("retired = []")
    }

    fn scan(source: &str, rules: &Rules) -> Vec<Found> {
        scan_markdown(FILE, source, rules)
            .0
            .into_iter()
            .map(|r| (r.kind, r.id, r.line))
            .collect()
    }

    fn messages(source: &str, rules: &Rules) -> Vec<String> {
        let (rows, mut defects) = scan_markdown(FILE, source, rules);
        defects.extend(check(&rows, rules));
        defects.sort_by(|a, b| {
            (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line))
        });
        defects.iter().map(|d| d.message.clone()).collect()
    }

    fn expect(kind: &str, id: &str, line: usize) -> Found {
        (kind.to_string(), id.to_string(), line)
    }

    #[req("requirement-traceability#VC-1.1", "requirement-traceability#VC-2.1")]
    #[test]
    fn template_rows_are_found() {
        assert_eq!(
            scan(CLEAN, &rules()),
            [
                expect("definition", "widget#FR-1", 3),
                expect("condition", "widget#VC-1.1", 7),
            ]
        );
    }

    #[test]
    fn fenced_blocks_are_skipped() {
        let source = "\
- **FR-1 — A**: The unit shall work.

```markdown
- **FR-2 — B**: inside a fence
```

   ```text
   | VC-3.1 | FR-3 | `test` | indented fence |
   ```

~~~
- **FR-4 — C**: inside a tilde fence
~~~
";
        assert_eq!(
            scan(source, &rules()),
            [expect("definition", "widget#FR-1", 1)]
        );
    }

    #[test]
    fn definitions_take_the_first_id_and_indented_lines() {
        let source = "\
- **FR-1 — A**: The unit shall match FR-2 and
  keep going.

  Not this paragraph.
";
        let (rows, _) = scan_markdown(FILE, source, &rules());
        assert_eq!(rows.len(), 1);
        let definition = rows.first().unwrap();
        assert_eq!(definition.id, "widget#FR-1");
        assert_eq!(
            definition.text,
            "- **FR-1 — A**: The unit shall match FR-2 and keep going."
        );
    }

    #[req("requirement-traceability#VC-2.1", "requirement-traceability#VC-3.1")]
    #[test]
    fn reference_lines_record_every_local_and_qualified_id() {
        let source =
            "| VC-1.1 | FR-1, storage#FR-3 | `test` | Criterion FR-2 |\n";
        let (rows, _) = scan_markdown(FILE, source, &rules());
        assert_eq!(rows.len(), 1);
        let cond = rows.first().unwrap();
        assert_eq!(cond.id, "widget#VC-1.1");
        assert_eq!(
            cond.parents,
            ["widget#FR-1", "storage#FR-3", "widget#FR-2"]
        );
        assert_eq!(cond.method, Some("test".to_string()));
    }

    #[test]
    fn rows_carry_normalized_text() {
        let (rows, _) = scan_markdown(FILE, CLEAN, &rules());
        let plan = rows.get(1).unwrap();
        assert_eq!(plan.text, "| VC-1.1 | FR-1 | `test` | Exact size match |");
    }

    #[req("requirement-traceability#VC-3.1")]
    #[test]
    fn doc_name_is_the_stem_without_the_suffix() {
        assert_eq!(doc_name("a/storage-design.md", "-design"), "storage");
        assert_eq!(doc_name("a/ets-overview.md", "-design"), "ets-overview");
        assert_eq!(doc_name("a/x-design.md", ""), "x-design");
    }

    #[test]
    fn clean_document_has_no_defects() {
        assert!(messages(CLEAN, &rules()).is_empty());
    }

    #[test]
    fn duplicate_definition_is_reported_once_at_the_second() {
        let source = format!("{CLEAN}\n- **FR-1 — Again**: It shall repeat.\n");
        let (rows, _) = scan_markdown(FILE, &source, &rules());
        let defects = check(&rows, &rules());
        assert_eq!(defects.len(), 1);
        let defect = defects.first().unwrap();
        assert_eq!(defect.line, 9);
        assert_eq!(
            defect.message,
            "widget#FR-1 is defined more than once; first definition at \
             docs/widget-design.md:3"
        );
    }

    #[test]
    fn reference_to_an_undefined_id_is_unresolved() {
        let source = format!("{CLEAN}| VC-7.1 | FR-7 | `test` | Other |\n");
        assert_eq!(
            messages(&source, &rules()),
            [
                "condition widget#VC-7.1 references undefined requirement widget#FR-7"
            ]
        );
    }

    #[test]
    fn definition_without_a_kind_of_reference_is_missing_it() {
        let source = "\
- **FR-1 — Size**: The widget shall report its size.
";
        assert_eq!(
            messages(source, &rules()),
            ["widget#FR-1 has no verification condition"]
        );
    }

    #[test]
    fn retired_ids_are_reported_where_they_appear() {
        let retired = rules_with("retired = [\"widget#FR-1\"]");
        let defects = messages(CLEAN, &retired);
        assert_eq!(
            defects.iter().filter(|m| m.ends_with("is retired")).count(),
            2
        );
        assert!(
            messages(CLEAN, &rules_with("retired = [\"widget#FR-2\"]"))
                .is_empty()
        );
    }

    #[test]
    fn unknown_keys_and_reserved_kinds_are_rejected() {
        for key in [
            "extra = 1",
            "exclude_phrases = []",
            "require_phrases = []",
            "references = {}",
        ] {
            let unknown = format!("{KEYS}retired = []\n{key}");
            assert!(toml::from_str::<TraceConfig>(&unknown).is_err(), "{key}");
        }
    }

    #[test]
    fn invalid_patterns_are_rejected() {
        let bad = format!("{KEYS}retired = []")
            .replace("definition = '^- \\*\\*", "definition = '(");
        let config: TraceConfig = toml::from_str(&bad).unwrap();
        assert!(matches!(
            config.rules(Path::new("trace.toml")),
            Err(GateError::Config { .. })
        ));
    }

    #[test]
    fn scan_condition_finds_parents() {
        let source = "| VC-1.1 | FR-1, FR-2 | `test` |";
        let (found, _) = scan_markdown(FILE, source, &rules());
        assert_eq!(found.len(), 1);

        let first = found.first().unwrap();
        assert_eq!(first.id, "widget#VC-1.1");
        assert_eq!(
            first.parents,
            ["widget#FR-1".to_string(), "widget#FR-2".to_string()]
        );
        assert_eq!(first.method, Some("test".to_string()));
    }

    #[test]
    fn condition_without_parents_is_an_orphan() {
        let source = "| VC-1.1 | | `test` |";
        let (found, mut defects) = scan_markdown(FILE, source, &rules());
        defects.extend(check(&found, &rules()));
        assert!(defects.iter().any(|d| d.message.contains("has no parent")));
    }

    #[req("requirement-traceability#VC-3.1")]
    #[test]
    fn ids_inside_longer_tokens_or_condition_ids_are_not_parents() {
        let source = "| VC-1.1 | FR-1 | `test` | Per IEC-61508 and XFR-2 |\n";
        let (rows, _) = scan_markdown(FILE, source, &rules());
        assert_eq!(
            rows.first().map(|r| r.parents.as_slice()),
            Some(["widget#FR-1".to_string()].as_slice())
        );
    }

    #[req("requirement-traceability#VC-4.1")]
    #[test]
    fn row_and_method_defects_are_reported() {
        let cases = [
            (
                "| VC-1.1 | VC-1.2 | FR-1 | `test` | Two |",
                "verification row of widget#VC-1.1 names more than one \
                 condition ID",
            ),
            (
                "| VC-1.1 | FR-1 | Plain |",
                "condition widget#VC-1.1 names no verification method",
            ),
            (
                "| VC-1.1 | FR-1 | `demo` | Other |",
                "condition widget#VC-1.1 names unknown method `demo`",
            ),
            (
                "| VC-1.1 | FR-1 | `test` | `review` |",
                "condition widget#VC-1.1 names more than one method",
            ),
        ];
        for (line, message) in cases {
            let source = format!(
                "- **FR-1 — Size**: The widget shall report.\n\n{line}\n"
            );
            assert_eq!(messages(&source, &rules()), [message], "{line}");
        }
    }

    #[test]
    fn a_marked_method_outside_methods_is_rejected() {
        let text = format!("{KEYS}retired = []").replace(
            "marked_methods = [\"test\"]",
            "marked_methods = [\"fuzz\"]",
        );
        let config: TraceConfig = toml::from_str(&text).unwrap();
        assert!(matches!(
            config.rules(Path::new("trace.toml")),
            Err(GateError::Config { .. })
        ));
    }

    #[req("requirement-traceability#VC-2.1")]
    #[test]
    fn a_parent_named_twice_is_recorded_once() {
        let source = "| VC-1.1 | FR-1 | `test` | FR-1 holds iff both hold |\n";
        let (rows, _) = scan_markdown(FILE, source, &rules());
        assert_eq!(
            rows.first().map(|r| r.parents.as_slice()),
            Some(["widget#FR-1".to_string()].as_slice())
        );
    }

    #[test]
    fn condition_pattern_is_required() {
        let missing =
            KEYS.replace("condition = 'VC-[0-9]+(?:\\.[0-9]+[a-z]?)?'\n", "");
        let toml_str = format!("{missing}retired = []");
        assert!(toml::from_str::<TraceConfig>(&toml_str).is_err());
    }

    #[test]
    fn unclosed_fence_is_reported_as_defect() {
        let source =
            "- **FR-1 — Size**: shall report.\n\n```text\nunclosed fence\n";
        let (rows, defects) = scan_markdown(FILE, source, &rules());
        assert_eq!(rows.len(), 1);
        assert_eq!(defects.len(), 1);
        assert_eq!(defects.first().map(|d| d.line), Some(3));
        assert_eq!(
            defects.first().map(|d| d.message.as_str()),
            Some("fenced code block is unclosed")
        );
    }

    #[test]
    fn document_with_no_definitions_is_a_defect() {
        let source = "| VC-1.1 | FR-1 | `test` | Orphan |\n";
        let (rows, defects) = scan_markdown(FILE, source, &rules());
        assert_eq!(rows.len(), 1);
        assert_eq!(defects.len(), 1);
        assert_eq!(defects.first().map(|d| d.line), Some(1));
        assert_eq!(
            defects.first().map(|d| d.message.as_str()),
            Some("document widget defines no requirements")
        );
    }

    #[test]
    fn verification_row_first_cell_validation() {
        let bad = "- **FR-1 — Size**: shall report.\n\n| VC-X | FR-1 | `test` | see VC-9.1 |\n";
        let defects = messages(bad, &rules());
        assert!(
            defects
                .iter()
                .any(|d| d == "verification row names no condition ID")
        );

        let good = "- **FR-1 — Size**: shall report.\n\n| VC-1.1 | FR-1 | `test` | Valid |\n";
        let (rows, defects) = scan_markdown(FILE, good, &rules());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.get(1).map(|r| r.id.as_str()), Some("widget#VC-1.1"));
        assert!(defects.is_empty());
    }

    #[test]
    fn fence_opens_and_closes_edge_cases() {
        assert_eq!(opens("```"), Some(('`', 3)));
        assert_eq!(opens("```markdown"), Some(('`', 3)));
        assert_eq!(opens("~~~"), Some(('~', 3)));
        assert_eq!(opens("~~~~"), Some(('~', 4)));
        assert_eq!(opens("``"), None);
        assert_eq!(opens("```foo`bar"), None);
        assert_eq!(opens("~~~foo`bar"), Some(('~', 3)));
        assert_eq!(opens("   ```"), Some(('`', 3)));

        assert!(closes("```", ('`', 3)));
        assert!(closes("````", ('`', 3)));
        assert!(!closes("``", ('`', 3)));
        assert!(!closes("~~~", ('`', 3)));
        assert!(!closes("``` extra", ('`', 3)));
        assert!(closes("   ```   ", ('`', 3)));
    }

    #[test]
    fn definition_text_indentation_boundaries() {
        let lines: Vec<Line<'_>> = vec![
            (1, "- **FR-1**: first line"),
            (2, "  continuation with spaces"),
            (3, "\tcontinuation with tab"),
            (4, ""),
            (5, "  after blank line"),
        ];
        let text = definition_text(&lines, 0);
        assert_eq!(
            text,
            "- **FR-1**: first line\n  continuation with spaces\n\tcontinuation with tab"
        );

        let single: Vec<Line<'_>> =
            vec![(1, "- **FR-2**: solo line"), (2, "next paragraph")];
        assert_eq!(definition_text(&single, 0), "- **FR-2**: solo line");
    }
}
