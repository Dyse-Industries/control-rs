//! Requirement definitions and references in Markdown.
//!
//! The matching rules are the whole parser. Lines inside fenced code blocks
//! are skipped. A line that matches `definition` defines the first ID on it,
//! and its text runs on through the indented lines that follow. A line that
//! matches a reference pattern references every ID on it, once per matching
//! pattern.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use regex::Regex;
use serde::Deserialize;

use super::{
    CONDITION, DEFINITION, Defect, MARKER, Row, SCHEMA, normalize, read_text,
    strip_code_spans,
};
use crate::error::{GateError, GateResult};

/// A file and line.
pub(super) type At<'a> = (&'a str, usize);

/// Definition rows by ID.
type Definitions<'a> = BTreeMap<&'a str, Vec<&'a Row>>;

/// A fence: its character and length.
type Fence = (char, usize);

/// A line outside fenced code blocks: its 1-based number and text.
type Line<'a> = (usize, &'a str);

/// ID occurrences on a marker line: qualified, or written without `<doc>#`.
pub type MarkedIds = Vec<Result<String, String>>;

/// Reference kinds, by name, with the pattern of their lines.
type References = Vec<(String, Regex)>;

/// A `trace.toml`: the patterns that find requirements in Markdown.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceConfig {
    /// Pattern for one requirement ID.
    pub id: String,
    /// Pattern for one verification condition ID (for example, `VC-1.1`).
    #[serde(default = "default_condition_pattern")]
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
    /// Reference kinds by name, each with the pattern of its lines.
    pub references: BTreeMap<String, String>,
    /// Qualified IDs that must not be defined or referenced again.
    pub retired: Vec<String>,
    /// Each match of one of these patterns in definition text is a defect.
    #[serde(default)]
    pub exclude_phrases: Vec<String>,
    /// Where `trace-marks` finds markers; none when absent.
    #[serde(default)]
    pub markers: Option<MarkerConfig>,
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
    /// Reference kinds, by name, with the pattern of their lines.
    references: References,
    /// Text removed from a file stem to form its document name.
    doc_suffix: String,
    /// IDs that must not appear.
    retired: BTreeSet<String>,
    /// Phrases that must not appear in definition text.
    exclude: Vec<Regex>,
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
    /// `GateError::Config` for an invalid pattern or a reference kind named
    /// `definition` or `marker`.
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
        let mut references = Vec::new();
        for (kind, pattern) in &self.references {
            if kind == DEFINITION || kind == MARKER {
                return Err(error(format!(
                    "reference kind `{kind}` is reserved"
                )));
            }
            references.push((kind.clone(), compile(kind, pattern)?));
        }
        compile("doc", &self.doc)?;
        compile("id", &self.id)?;
        compile("condition", &self.condition)?;
        let occurrence = format!(
            "(?:(?P<trace_doc>{})#)?(?P<trace_id>{})",
            self.doc, self.id
        );
        let condition_occurrence = format!(
            "(?:(?P<trace_doc>{})#)?(?P<trace_id>{})",
            self.doc, self.condition
        );
        Ok(Rules {
            occurrence: compile("id", &occurrence)?,
            condition_occurrence: compile("condition", &condition_occurrence)?,
            definition: compile("definition", &self.definition)?,
            references,
            doc_suffix: self.doc_suffix.clone(),
            retired: self.retired.iter().cloned().collect(),
            exclude: self
                .exclude_phrases
                .iter()
                .map(|p| compile("exclude_phrases", p))
                .collect::<GateResult<_>>()?,
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

    /// The qualified condition IDs on `line`, left to right.
    pub fn condition_ids<'a>(
        &'a self,
        line: &'a str,
        doc: &'a str,
    ) -> impl Iterator<Item = String> + 'a {
        self.condition_occurrence
            .captures_iter(line)
            .filter_map(move |caps| {
                let id = caps.name("trace_id")?.as_str();
                let owner = caps.name("trace_doc").map_or(doc, |m| m.as_str());
                Some(format!("{owner}#{id}"))
            })
    }

    /// The ID occurrences on a marker line, left to right: `Ok` with the
    /// qualified ID, or `Err` with an ID written without `<doc>#`.
    #[must_use]
    pub fn marked_ids(&self, line: &str) -> MarkedIds {
        let cond_matches: Vec<_> =
            self.condition_occurrence.captures_iter(line).collect();
        let cond_spans: Vec<_> = cond_matches
            .iter()
            .filter_map(|c| c.get(0).map(|m| m.range()))
            .collect();

        let req_matches = self.occurrence.captures_iter(line).filter(|caps| {
            caps.get(0).is_none_or(|m| {
                !cond_spans
                    .iter()
                    .any(|cs| m.start() < cs.end && m.end() > cs.start)
            })
        });

        let mut matches: Vec<_> = req_matches.chain(cond_matches).collect();
        matches.sort_by_key(|caps| caps.get(0).map_or(0, |m| m.start()));
        matches
            .into_iter()
            .filter_map(|caps| {
                let id = caps.name("trace_id")?.as_str();
                Some(caps.name("trace_doc").map_or_else(
                    || Err(id.to_string()),
                    |doc| Ok(format!("{}#{id}", doc.as_str())),
                ))
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

fn scan_condition_reference(
    line: &str,
    c_caps: &regex::Captures<'_>,
    at: At<'_>,
    ctx: &ScanContext<'_>,
) -> Vec<Row> {
    let mut rows = Vec::new();
    let Some(c_id) = c_caps.name("trace_id").map(|m| m.as_str()) else {
        return rows;
    };
    let c_doc = c_caps.name("trace_doc").map_or(ctx.doc, |m| m.as_str());
    let cond_qualified = format!("{c_doc}#{c_id}");
    let c_span = c_caps.get(0).map_or(0..0, |m| m.range());

    let parent_ids: Vec<String> = ctx
        .rules
        .occurrence
        .captures_iter(line)
        .filter_map(|caps| {
            let m = caps.get(0)?;
            if m.start() >= c_span.end || m.end() <= c_span.start {
                let id = caps.name("trace_id")?.as_str();
                let owner =
                    caps.name("trace_doc").map_or(ctx.doc, |d| d.as_str());
                Some(format!("{owner}#{id}"))
            } else {
                None
            }
        })
        .collect();

    if parent_ids.is_empty() {
        rows.push(condition_row(cond_qualified, None, at, line));
    } else {
        for pid in parent_ids {
            rows.push(condition_row(
                cond_qualified.clone(),
                Some(pid),
                at,
                line,
            ));
        }
    }
    rows
}

/// The definition, condition and reference rows of one Markdown document.
///
/// `file` is the path recorded in each row; its stem, less the configured
/// suffix, names the document.
#[must_use]
pub fn scan_markdown(file: &str, source: &str, rules: &Rules) -> Vec<Row> {
    let doc = doc_name(file, &rules.doc_suffix);
    let ctx = ScanContext { doc: &doc, rules };
    let lines = visible_lines(source);
    let mut rows = Vec::new();
    for (idx, &(number, line)) in lines.iter().enumerate() {
        let at = (file, number);
        if rules.definition.is_match(line)
            && let Some(id) = rules.ids(line, &doc).next()
        {
            rows.push(row(id, DEFINITION, at, &definition_text(&lines, idx)));
        }
        for (kind, pattern) in &rules.references {
            if pattern.is_match(line) {
                if let Some(c_caps) = rules.condition_occurrence.captures(line)
                {
                    rows.extend(scan_condition_reference(
                        line, &c_caps, at, &ctx,
                    ));
                } else {
                    rows.extend(
                        rules.ids(line, &doc).map(|id| row(id, kind, at, line)),
                    );
                }
            }
        }
    }
    rows
}

fn check_duplicate_definitions(
    definitions: &Definitions<'_>,
    conditions: &[&Row],
    references: &[&Row],
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
                conditions.iter().any(|c| c.parent.as_deref() == Some(id));
            let has_ref = references.iter().any(|r| r.id == id);
            if !has_cond && !has_ref {
                defects.push(Defect::at(
                    first,
                    format!("{id} has no verification condition or reference"),
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
        if let Some(parent) = &condition.parent {
            if !definitions.contains_key(parent.as_str()) {
                defects.push(Defect::at(
                    condition,
                    format!(
                        "condition {} references undefined requirement {parent}",
                        condition.id
                    ),
                ));
            }
        } else {
            defects.push(Defect::at(
                condition,
                format!("condition {} has no parent requirement", condition.id),
            ));
        }
    }
}

fn check_dangling_references(
    references: &[&Row],
    definitions: &Definitions<'_>,
    defects: &mut Vec<Defect>,
) {
    for reference in references {
        if !definitions.contains_key(reference.id.as_str()) {
            defects.push(Defect::at(
                reference,
                format!("{} is not defined", reference.id),
            ));
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
    let references: Vec<&Row> = rows
        .iter()
        .filter(|r| r.kind != DEFINITION && r.kind != CONDITION)
        .collect();
    let mut defects = Vec::new();

    check_duplicate_definitions(
        &definitions,
        &conditions,
        &references,
        &mut defects,
    );
    check_duplicate_conditions(&condition_defs, &mut defects);
    check_condition_parents(&conditions, &definitions, &mut defects);
    check_dangling_references(&references, &definitions, &mut defects);

    for row in rows.iter().filter(|r| rules.retired.contains(&r.id)) {
        defects.push(Defect::at(row, format!("{} is retired", row.id)));
    }

    for definition in rows.iter().filter(|r| r.kind == DEFINITION) {
        defects.extend(phrase_defects(definition, rules));
    }
    defects.sort_by(|a, b| {
        (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line))
    });
    defects
}

/// The phrase-rule defects of one definition row: one per match of each
/// excluded phrase.
fn phrase_defects(definition: &Row, rules: &Rules) -> Vec<Defect> {
    let text = strip_code_spans(&definition.text);
    let mut defects = Vec::new();
    for pattern in &rules.exclude {
        for found in pattern.find_iter(&text) {
            defects.push(Defect::at(
                definition,
                format!(
                    "{} contains the excluded phrase \"{}\"",
                    definition.id,
                    found.as_str()
                ),
            ));
        }
    }
    defects
}

/// The lines outside fenced code blocks, with their 1-based numbers.
fn visible_lines(source: &str) -> Vec<Line<'_>> {
    let mut fence = None;
    let mut lines = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        match fence {
            Some(open) => {
                if closes(line, open) {
                    fence = None;
                }
            }
            None => match opens(line) {
                Some(open) => fence = Some(open),
                None => lines.push((idx.saturating_add(1), line)),
            },
        }
    }
    lines
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

fn default_condition_pattern() -> String {
    r"VC-(?:[A-Z0-9-]+|[0-9]+)(?:\.[0-9]+[a-z]?)?".to_string()
}

/// A row of the current schema at `at`, a file and line, with its text
/// normalized.
pub(super) fn row(id: String, kind: &str, at: At<'_>, text: &str) -> Row {
    Row {
        schema: SCHEMA,
        id,
        kind: kind.to_string(),
        parent: None,
        file: at.0.to_string(),
        line: at.1,
        text: normalize(text),
    }
}

/// A condition row of the current schema at `at`, a file and line, with its text
/// normalized.
pub(super) fn condition_row(
    id: String,
    parent: Option<String>,
    at: At<'_>,
    text: &str,
) -> Row {
    Row {
        schema: SCHEMA,
        id,
        kind: CONDITION.to_string(),
        parent,
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

| Requirements | Gates  | Criterion        |
|:-------------|:-------|:-----------------|
| FR-1         | `test` | Exact size match |
";

    const FILE: &str = "docs/widget-design.md";

    const KEYS: &str = r#"id = '(?:FR|NFR|C)-[0-9]+[a-z]?'
doc = '[a-z0-9-]+'
files = ["docs/*-design.md"]
doc_suffix = "-design"
definition = '^- \*\*(?:FR|NFR|C)-'
"#;

    const REFERENCES: &str = r"[references]
verification = '^\| *(?:[a-z0-9-]+#)?(?:FR|NFR|C)-'
";

    /// A row's kind, ID and line.
    type Found = (String, String, usize);

    fn rules_with(keys: &str) -> Rules {
        let text = format!("{KEYS}{keys}\n{REFERENCES}");
        let config: TraceConfig = toml::from_str(&text).unwrap();
        config.rules(Path::new("trace.toml")).unwrap()
    }

    fn rules() -> Rules {
        rules_with("retired = []")
    }

    fn scan(source: &str, rules: &Rules) -> Vec<Found> {
        scan_markdown(FILE, source, rules)
            .into_iter()
            .map(|r| (r.kind, r.id, r.line))
            .collect()
    }

    fn messages(source: &str, rules: &Rules) -> Vec<String> {
        check(&scan_markdown(FILE, source, rules), rules)
            .iter()
            .map(|d| d.message.clone())
            .collect()
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
                expect("verification", "widget#FR-1", 7),
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
   | FR-3 | `test` | `test` | indented fence |
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
        let rows = scan_markdown(FILE, source, &rules());
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
        let source = "| FR-1, storage#FR-3 | `test` | Criterion FR-2 |\n";
        assert_eq!(
            scan(source, &rules()),
            [
                expect("verification", "widget#FR-1", 1),
                expect("verification", "storage#FR-3", 1),
                expect("verification", "widget#FR-2", 1),
            ]
        );
    }

    #[req("requirement-traceability#VC-2.1")]
    #[test]
    fn a_line_matching_two_patterns_yields_one_reference_per_pattern() {
        let text = format!("{KEYS}retired = []\n{REFERENCES}any = '^\\|'\n");
        let config: TraceConfig = toml::from_str(&text).unwrap();
        let rules = config.rules(Path::new("trace.toml")).unwrap();
        assert_eq!(
            scan("| FR-1 | `test` | Criterion |\n", &rules),
            [
                expect("any", "widget#FR-1", 1),
                expect("verification", "widget#FR-1", 1)
            ]
        );
    }

    #[test]
    fn rows_carry_normalized_text() {
        let rows = scan_markdown(FILE, CLEAN, &rules());
        let plan = rows.get(1).unwrap();
        assert_eq!(plan.text, "| FR-1 | `test` | Exact size match |");
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
        let defects = check(&scan_markdown(FILE, &source, &rules()), &rules());
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
        let source = format!("{CLEAN}| FR-7 | `test` | Other |\n");
        assert_eq!(messages(&source, &rules()), ["widget#FR-7 is not defined"]);
    }

    #[test]
    fn definition_without_a_kind_of_reference_is_missing_it() {
        let source = "\
- **FR-1 — Size**: The widget shall report its size.
";
        assert_eq!(
            messages(source, &rules()),
            ["widget#FR-1 has no verification condition or reference"]
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

    #[req("requirement-traceability#VC-5.1")]
    #[test]
    fn every_excluded_phrase_match_is_reported_outside_code_spans() {
        let exclude = rules_with(
            "retired = []\nexclude_phrases = ['\\bshould\\b', '\\bmay\\b']",
        );
        let quoted = CLEAN.replace("shall report", "shall `should` report");
        assert!(messages(&quoted, &exclude).is_empty());
        let source = CLEAN
            .replace("shall report its size", "should report, may be should");
        assert_eq!(
            messages(&source, &exclude),
            [
                "widget#FR-1 contains the excluded phrase \"should\"",
                "widget#FR-1 contains the excluded phrase \"should\"",
                "widget#FR-1 contains the excluded phrase \"may\"",
            ]
        );
    }

    #[test]
    fn an_empty_exclude_list_disables_the_rule() {
        let source = CLEAN.replace("shall report", "should report");
        assert!(messages(&source, &rules()).is_empty());
    }

    #[test]
    fn unknown_keys_and_reserved_kinds_are_rejected() {
        for key in ["extra = 1", "require_phrases = []"] {
            let unknown = format!("{KEYS}retired = []\n{key}\n{REFERENCES}");
            assert!(toml::from_str::<TraceConfig>(&unknown).is_err(), "{key}");
        }
        let reserved =
            format!("{KEYS}retired = []\n{REFERENCES}marker = '^x'\n");
        let config: TraceConfig = toml::from_str(&reserved).unwrap();
        assert!(matches!(
            config.rules(Path::new("trace.toml")),
            Err(GateError::Config { .. })
        ));
    }

    #[test]
    fn invalid_patterns_are_rejected() {
        let bad = format!("{KEYS}retired = []\n{REFERENCES}")
            .replace("definition = '^- \\*\\*", "definition = '(");
        let config: TraceConfig = toml::from_str(&bad).unwrap();
        assert!(matches!(
            config.rules(Path::new("trace.toml")),
            Err(GateError::Config { .. })
        ));
    }
}
