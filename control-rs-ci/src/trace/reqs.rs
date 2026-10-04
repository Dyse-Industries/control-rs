//! Requirement definitions and verification conditions in Markdown.
//!
//! The matching rules are the whole parser. Lines inside fenced code blocks
//! are skipped. A line that matches `definition` defines the first
//! requirement ID on it, and its text runs on through the indented lines that
//! follow. A line that matches `verification` defines the condition ID on it;
//! every other requirement ID on the line is a parent, the code span that
//! names a method is its method and the code spans of the fourth cell are its
//! targets.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use regex::Regex;
use serde::Deserialize;

use super::{CONDITION, DEFINITION, Defect, Row, SCHEMA, normalize, read_text};
use crate::error::{GateError, GateResult};

/// A file and line.
pub(super) type At<'a> = (&'a str, usize);

/// First row declaring a document ID by document name.
type DocFirstFile<'a> = BTreeMap<&'a str, &'a Row>;

/// A document ID and file path pair for reporting collisions.
type DocFilePair<'a> = (&'a str, &'a str);

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

/// A file path and its text.
pub type Source<'a> = (&'a str, &'a str);

/// Decision records by ID.
type Decisions<'a> = BTreeMap<&'a str, &'a Decision>;

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
    /// Markdown files, and directories whose Markdown files are read.
    pub files: Vec<String>,
    /// Pattern matching the document ID declaration in a Markdown document.
    #[serde(default = "default_doc_id")]
    pub doc_id: String,
    /// Pattern for a line that defines a requirement.
    pub definition: String,
    /// Pattern for a line that defines a verification condition.
    pub verification: String,
    /// Verification methods a condition may name.
    pub methods: Vec<String>,
    /// The subset of methods whose conditions name targets that a result
    /// artifact must record as passed.
    pub automated_methods: Vec<String>,
    /// Result artifact per method.
    #[serde(default)]
    pub method: BTreeMap<String, MethodConfig>,
    /// Qualified IDs that must not be defined or referenced again.
    pub retired: Vec<String>,
    /// Decision records and their citations; absent, the decision checks
    /// do not run.
    #[serde(default)]
    pub decisions: Option<DecisionConfig>,
}

/// The `[decisions]` table of a `trace.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionConfig {
    /// Decision record files, and directories whose Markdown files are read.
    pub files: Vec<String>,
    /// Pattern for one decision ID.
    pub id: String,
    /// Pattern for the line that defines a decision record's ID.
    pub definition: String,
    /// Pattern whose `status` capture is a decision record's status.
    pub status: String,
    /// Decision statuses a requirement may cite.
    pub accepted: Vec<String>,
}

/// The `[method.<name>]` table of a `trace.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodConfig {
    /// Relative path to the verification result log artifact (for example, `target/ci-artifacts/test.log`).
    #[serde(default)]
    pub result_artifact: Option<String>,
    /// Relative path to the log of a secondary interpreter run of the same
    /// tests (for example, `target/ci-artifacts/miri.log`); a passing target
    /// of a crate the log covers but does not record is a W-2 warning.
    #[serde(default)]
    pub interpreter_artifact: Option<String>,
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
    /// Pattern matching the document ID declaration in Markdown.
    doc_id: Regex,
    /// IDs that must not appear.
    retired: BTreeSet<String>,
    /// The decision patterns, when the configuration has `[decisions]`.
    decisions: Option<DecisionRules>,
}

/// The compiled patterns of a [`DecisionConfig`].
#[derive(Debug, Clone)]
pub struct DecisionRules {
    /// One decision ID, word-bounded.
    id: Regex,
    /// A line that defines a decision record.
    definition: Regex,
    /// A decision record's status, in its `status` capture.
    status: Regex,
    /// Decision statuses a requirement may cite.
    accepted: BTreeSet<String>,
}

/// A decision record: its ID, status and defining line.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Decision {
    /// File of the record.
    file: String,
    /// The decision ID.
    id: String,
    /// 1-based line of the definition.
    line: usize,
    /// The captured status, if any.
    status: Option<String>,
}

struct ScanContext<'a> {
    doc: &'a str,
    rules: &'a Rules,
}

impl TraceConfig {
    /// The file-name endings of the Markdown files a directory root selects.
    #[must_use]
    pub fn doc_suffixes(&self) -> Vec<String> {
        vec![".md".to_string()]
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
        for m in &self.automated_methods {
            if !self.methods.contains(m) {
                return Err(error(format!(
                    "automated method `{m}` is not in `methods`"
                )));
            }
            let has_artifact = self
                .method
                .get(m)
                .is_some_and(|c| c.result_artifact.is_some());
            if !has_artifact {
                return Err(error(format!(
                    "automated method `{m}` has no `result_artifact`"
                )));
            }
        }
        compile("doc", &self.doc)?;
        compile("id", &self.id)?;
        compile("condition", &self.condition)?;
        compile("doc_id", &self.doc_id)?;
        let decisions =
            self.decisions.as_ref().map(|d| d.rules(path)).transpose()?;
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
            doc_id: compile("doc_id", &self.doc_id)?,
            retired: self.retired.iter().cloned().collect(),
            decisions,
        })
    }
}

impl DecisionConfig {
    /// Compiles the patterns; `path` names the configuration in errors.
    ///
    /// # Errors
    /// `GateError::Config` for an invalid pattern, or a status pattern
    /// without a `status` capture.
    pub fn rules(&self, path: &Path) -> GateResult<DecisionRules> {
        let compile = |key: &str, pattern: &str, capture: bool| {
            let error = |message: String| GateError::Config {
                path: path.to_path_buf(),
                message: format!("`decisions.{key}` {message}"),
            };
            let regex = Regex::new(pattern)
                .map_err(|e| error(format!("is not a valid pattern: {e}")))?;
            if capture && !regex.capture_names().any(|n| n == Some("status")) {
                return Err(error("has no `status` capture".to_string()));
            }
            Ok(regex)
        };
        Ok(DecisionRules {
            id: compile("id", &format!(r"\b(?:{})\b", self.id), false)?,
            definition: compile("definition", &self.definition, false)?,
            status: compile("status", &self.status, true)?,
            accepted: self.accepted.iter().cloned().collect(),
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
}

fn default_doc_id() -> String {
    r"^#\s+.*\((?P<doc>[a-z0-9-]+)\)".to_string()
}

/// Finds the declared document ID in `lines` using `rules.doc_id`, returning
/// the declared ID or reporting scan defects.
fn extract_doc_id<'a>(
    file: &str,
    lines: &'a [Line<'_>],
    rules: &Rules,
    defects: &mut Vec<Defect>,
) -> Option<&'a str> {
    let mut declared: Option<At<'a>> = None;
    for &(number, line) in lines {
        if let Some(caps) = rules.doc_id.captures(line) {
            let id = caps
                .name("doc")
                .map(|m| m.as_str())
                .or_else(|| caps.get(1).map(|m| m.as_str()))
                .unwrap_or_else(|| caps.get(0).map_or("", |m| m.as_str()));
            if let Some((existing_id, _)) = declared {
                if existing_id != id {
                    defects.push(Defect {
                        file: file.to_string(),
                        line: number,
                        message: format!(
                            "duplicate document ID declaration `{id}`"
                        ),
                    });
                }
            } else {
                declared = Some((id, number));
            }
        }
    }
    if declared.is_none() {
        defects.push(Defect {
            file: file.to_string(),
            line: 1,
            message: "missing document ID declaration matching doc_id"
                .to_string(),
        });
    }
    declared.map(|(id, _)| id)
}

/// The first cell of a Markdown table line.
fn first_cell(line: &str) -> &str {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('|').unwrap_or(trimmed);
    rest.split('|').next().unwrap_or(rest).trim()
}

/// The code spans of the fourth cell of a table line with five or more cells.
fn targets(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let inner = trimmed.strip_prefix('|').unwrap_or(trimmed);
    let inner = inner.strip_suffix('|').unwrap_or(inner);
    let cells: Vec<&str> = inner.split('|').collect();
    if cells.len() < 5 {
        return Vec::new();
    }
    cells
        .get(3)
        .map(|cell| {
            super::code_spans(cell)
                .iter()
                .map(|s| s.content.trim().to_string())
                .collect()
        })
        .unwrap_or_default()
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
    condition.targets = targets(line);
    Some(condition)
}

/// The definition and condition rows of one Markdown document, and any scan defects.
///
/// `file` is the path recorded in each row; its document ID is extracted from
/// the first line matching `rules.doc_id`.
#[must_use]
pub fn scan_markdown(file: &str, source: &str, rules: &Rules) -> MarkdownScan {
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

    let fallback_stem = Path::new(file)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let declared_doc = extract_doc_id(file, &lines, rules, &mut defects);
    let doc = declared_doc.unwrap_or(&fallback_stem);
    let ctx = ScanContext { doc, rules };

    for (idx, &(number, line)) in lines.iter().enumerate() {
        let at = (file, number);
        if rules.definition.is_match(line)
            && let Some(id) = rules.ids(line, doc).next()
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

fn check_duplicate_document_ids(rows: &[Row], defects: &mut Vec<Defect>) {
    let mut doc_first_file = DocFirstFile::new();
    let mut reported: BTreeSet<DocFilePair<'_>> = BTreeSet::new();
    for row in rows.iter().filter(|r| r.kind == DEFINITION) {
        if let Some(doc) = row.id.split('#').next() {
            if let Some(first) = doc_first_file.get(doc) {
                if first.file != row.file
                    && reported.insert((doc, row.file.as_str()))
                {
                    defects.push(Defect::at(
                        row,
                        format!(
                            "document ID `{doc}` is defined in multiple files; also defined in {}",
                            first.file
                        ),
                    ));
                }
            } else {
                doc_first_file.insert(doc, row);
            }
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
    check_duplicate_document_ids(rows, &mut defects);

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

/// The decision defects of the decision `records` and of the requirement
/// definitions in `rows`, sorted by file, then line. Without `[decisions]`
/// there are none.
#[must_use]
pub fn check_decisions(
    rules: &Rules,
    records: &[Source<'_>],
    rows: &[Row],
) -> Vec<Defect> {
    let Some(decision_rules) = rules.decisions.as_ref() else {
        return Vec::new();
    };
    let found: Vec<Decision> = records
        .iter()
        .filter_map(|&(file, source)| {
            decision_record(file, source, decision_rules)
        })
        .collect();
    let mut by_id = Decisions::new();
    let mut defects = Vec::new();
    for decision in &found {
        let at = |message: String| Defect {
            file: decision.file.clone(),
            line: decision.line,
            message,
        };
        if decision.status.is_none() {
            defects.push(at(format!("decision {} has no status", decision.id)));
        }
        if let Some(first) = by_id.get(decision.id.as_str()) {
            defects.push(at(format!(
                "decision {} is defined more than once; first definition at {}:{}",
                decision.id, first.file, first.line
            )));
        } else {
            by_id.insert(decision.id.as_str(), decision);
        }
    }
    for row in rows.iter().filter(|r| r.kind == DEFINITION) {
        check_citations(row, decision_rules, &by_id, &mut defects);
    }
    defects.sort_by(|a, b| {
        (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line))
    });
    defects
}

/// Reports each decision that the text of the requirement `row` cites and
/// that has no record, or whose status is not accepted.
fn check_citations(
    row: &Row,
    rules: &DecisionRules,
    by_id: &Decisions<'_>,
    defects: &mut Vec<Defect>,
) {
    let cited: BTreeSet<&str> =
        rules.id.find_iter(&row.text).map(|m| m.as_str()).collect();
    for id in cited {
        let Some(decision) = by_id.get(id) else {
            defects.push(Defect::at(
                row,
                format!(
                    "requirement {} cites decision {id}, which has no decision record",
                    row.id
                ),
            ));
            continue;
        };
        let status = decision.status.as_deref().unwrap_or("none");
        if !rules.accepted.contains(status) {
            defects.push(Defect::at(
                row,
                format!(
                    "requirement {} cites decision {id} with status {status}; requirements cite only accepted decisions",
                    row.id
                ),
            ));
        }
    }
}

/// The `status` capture of the first line in `lines` that `pattern` matches.
fn captured_status(lines: &[Line<'_>], pattern: &Regex) -> Option<String> {
    lines.iter().find_map(|&(_, line)| {
        pattern
            .captures(line)
            .and_then(|caps| caps.name("status"))
            .map(|m| m.as_str().to_string())
    })
}

/// The decision record in `source`: the first decision ID on the first line
/// outside fenced code blocks that matches `rules.definition`.
fn decision_record(
    file: &str,
    source: &str,
    rules: &DecisionRules,
) -> Option<Decision> {
    let (lines, _) = visible_lines(source);
    let (line, id) = lines.iter().find_map(|&(number, text)| {
        rules
            .definition
            .is_match(text)
            .then(|| rules.id.find(text))
            .flatten()
            .map(|m| (number, m.as_str().to_string()))
    })?;
    Some(Decision {
        file: file.to_string(),
        id,
        line,
        status: captured_status(&lines, &rules.status),
    })
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
        targets: Vec::new(),
        file: at.0.to_string(),
        line: at.1,
        text: normalize(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLEAN: &str = "\
# Widget (widget)

- **FR-1 — Size**: The widget shall report its size.

| Condition | Requirement | Method | Criterion |
|:----------|:------------|:-------|:----------|
| VC-1.1    | FR-1        | `test` | Exact size match |
";

    const DECISIONS: &str = r#"retired = []

[decisions]
files = ["docs/adr"]
id = 'ADR-[0-9]{4}'
definition = '^#\s+ADR-[0-9]{4}:'
status = 'ADR%20Status-(?P<status>[A-Za-z]+)-'
accepted = ["Accepted"]
"#;

    const FILE: &str = "docs/widget-design.md";

    const KEYS: &str = r#"id = '(?:FR|NFR|C)-[0-9]+[a-z]?'
condition = 'VC-[0-9]+(?:\.[0-9]+[a-z]?)?'
doc = '[a-z0-9-]+'
files = ["docs/*-design.md"]
doc_id = '^#\s+.*\((?P<doc>[a-z0-9-]+)\)'
definition = '^- \*\*(?:FR|NFR|C)-'
verification = '^\| *(?:[a-z0-9-]+#)?VC-'
methods = ["test", "analysis", "inspection", "review"]
automated_methods = []
"#;

    /// A row's kind, ID and line.
    type Found = (String, String, usize);

    /// A fixture file path and its text.
    type Fixture<'a> = (&'a str, String);

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
# Widget (widget)

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
            [expect("definition", "widget#FR-1", 3)]
        );
    }

    #[test]
    fn definitions_take_the_first_id_and_indented_lines() {
        let source = "\
# Widget (widget)

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

    #[test]
    fn reference_lines_record_every_local_and_qualified_id() {
        let source = "# Widget (widget)\n\n| VC-1.1 | FR-1, storage#FR-3 | `test` | Criterion FR-2 |\n";
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

    #[test]
    fn doc_id_is_extracted_from_the_title_declaration() {
        let source = "# Storage Backends (storage)\n\n- **FR-1**: Desc\n\n| VC-1.1 | FR-1 | `test` | C |\n";
        let (rows, defects) =
            scan_markdown("docs/storage.md", source, &rules());
        assert!(defects.is_empty(), "{defects:?}");
        assert_eq!(rows.first().unwrap().id, "storage#FR-1");
    }

    #[test]
    fn missing_doc_id_declaration_is_reported_as_defect() {
        let source = "# Storage Backends\n\n- **FR-1**: Desc\n\n| VC-1.1 | FR-1 | `test` | C |\n";
        let (rows, defects) =
            scan_markdown("docs/storage.md", source, &rules());
        assert_eq!(defects.len(), 1);
        assert_eq!(defects.first().map(|d| d.line), Some(1));
        assert_eq!(
            defects.first().map(|d| d.message.as_str()),
            Some("missing document ID declaration matching doc_id")
        );
        assert_eq!(rows.first().unwrap().id, "storage#FR-1");
    }

    #[test]
    fn duplicate_doc_id_across_files_is_reported() {
        let (rows1, _) = scan_markdown(
            "docs/a.md",
            "# Doc A (widget)\n\n- **FR-1**: Desc\n\n| VC-1.1 | FR-1 | `test` | C |\n",
            &rules(),
        );
        let (rows2, _) = scan_markdown(
            "docs/b.md",
            "# Doc B (widget)\n\n- **FR-2**: Desc\n\n| VC-2.1 | FR-2 | `test` | C |\n",
            &rules(),
        );
        let all_rows = [rows1, rows2].concat();
        let defects = check(&all_rows, &rules());
        assert!(
            defects.iter().any(|d| d
                .message
                .contains("document ID `widget` is defined in multiple files"))
        );
    }

    #[test]
    fn the_default_doc_id_pattern_reads_a_heading_suffix() {
        let keys: String = KEYS
            .lines()
            .filter(|l| !l.starts_with("doc_id"))
            .collect::<Vec<_>>()
            .join("\n");
        let config: TraceConfig =
            toml::from_str(&format!("{keys}\nretired = []")).unwrap();
        let rules = config.rules(Path::new("trace.toml")).unwrap();
        assert_eq!(
            scan(CLEAN, &rules).first(),
            Some(&expect(DEFINITION, "widget#FR-1", 3))
        );
    }

    #[test]
    fn a_different_second_document_id_is_reported_and_a_repeat_is_not() {
        let rows = "\n- **FR-1 — Size**: It shall.\n";
        let different = format!("# Widget (widget)\n# Other (other){rows}");
        let (_, defects) = scan_markdown(FILE, &different, &rules());
        assert!(defects.iter().any(|d| {
            d.message == "duplicate document ID declaration `other`"
        }));
        let same = format!("# Widget (widget)\n# Again (widget){rows}");
        let (_, defects) = scan_markdown(FILE, &same, &rules());
        assert!(
            !defects
                .iter()
                .any(|d| d.message.contains("duplicate document ID"))
        );
    }

    #[test]
    fn targets_come_from_the_fourth_of_at_least_five_cells() {
        assert_eq!(
            targets("| VC-1.1 | FR-1 | `libtest` | `a::b` | note |"),
            ["a::b"]
        );
        let got = targets("| VC-1.1 | FR-1 | `libtest` | `a::b` |");
        assert!(got.is_empty(), "{got:?}");
    }

    #[test]
    fn only_definitions_make_a_document_id_shared_between_files() {
        let (defined, _) = scan_markdown(
            "docs/a.md",
            "# Doc A (widget)\n\n- **FR-1**: Desc\n\n| VC-1.1 | FR-1 | `test` | C |\n",
            &rules(),
        );
        let (only_conditions, _) = scan_markdown(
            "docs/b.md",
            "# Doc B (widget)\n\n| VC-2.1 | FR-1 | `test` | C |\n",
            &rules(),
        );
        let rows = [defined, only_conditions].concat();
        assert!(
            !check(&rows, &rules())
                .iter()
                .any(|d| d.message.contains("defined in multiple files"))
        );
    }

    #[test]
    fn clean_document_has_no_defects() {
        let got = messages(CLEAN, &rules());
        assert!(got.is_empty(), "{got:?}");
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
# Widget (widget)

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
        let got = messages(CLEAN, &rules_with("retired = [\"widget#FR-2\"]"));
        assert!(got.is_empty(), "{got:?}");
    }

    #[test]
    fn unknown_keys_and_reserved_kinds_are_rejected() {
        for key in [
            "extra = 1",
            "doc_suffix = \"-design\"",
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
        let source = "# Widget (widget)\n\n| VC-1.1 | FR-1, FR-2 | `test` |";
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

    #[test]
    fn ids_inside_longer_tokens_or_condition_ids_are_not_parents() {
        let source = "# Widget (widget)\n\n| VC-1.1 | FR-1 | `test` | Per IEC-61508 and XFR-2 |\n";
        let (rows, _) = scan_markdown(FILE, source, &rules());
        assert_eq!(
            rows.first().map(|r| r.parents.as_slice()),
            Some(["widget#FR-1".to_string()].as_slice())
        );
    }

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
                "# Widget (widget)\n\n- **FR-1 — Size**: The widget shall report.\n\n{line}\n"
            );
            assert_eq!(messages(&source, &rules()), [message], "{line}");
        }
    }

    #[test]
    fn an_automated_method_needs_a_listed_name_and_a_result_artifact() {
        let rejected = |method: &str, tail: &str| {
            let text = format!("{KEYS}retired = []{tail}").replace(
                "automated_methods = []",
                &format!("automated_methods = [\"{method}\"]"),
            );
            let config: TraceConfig = toml::from_str(&text).unwrap();
            matches!(
                config.rules(Path::new("trace.toml")),
                Err(GateError::Config { .. })
            )
        };
        assert!(rejected("fuzz", ""));
        assert!(rejected("test", ""));
        assert!(!rejected(
            "test",
            "\n[method.test]\nresult_artifact = \"t.log\"\n"
        ));
    }

    #[test]
    fn targets_are_the_code_spans_of_the_fourth_cell() {
        let source = "# Widget (widget)\n\n\
            | VC-1.1 | FR-1 | `test` | `a::t::x`, `a::t::y` | `z` Crit |\n\
            | VC-1.2 | FR-1 | `review` | — | Crit |\n\
            | VC-1.3 | FR-1 | `test` | Crit |\n";
        let (rows, _) = scan_markdown(FILE, source, &rules());
        let targets: Vec<_> = rows.iter().map(|r| r.targets.clone()).collect();
        assert_eq!(
            targets,
            [
                vec!["a::t::x".to_string(), "a::t::y".to_string()],
                vec![],
                vec![]
            ]
        );
    }

    #[test]
    fn a_parent_named_twice_is_recorded_once() {
        let source = "# Widget (widget)\n\n| VC-1.1 | FR-1 | `test` | FR-1 holds iff both hold |\n";
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
        let source = "# Widget (widget)\n\n- **FR-1 — Size**: shall report.\n\n```text\nunclosed fence\n";
        let (rows, defects) = scan_markdown(FILE, source, &rules());
        assert_eq!(rows.len(), 1);
        assert_eq!(defects.len(), 1);
        assert_eq!(defects.first().map(|d| d.line), Some(5));
        assert_eq!(
            defects.first().map(|d| d.message.as_str()),
            Some("fenced code block is unclosed")
        );
    }

    #[test]
    fn document_with_no_definitions_is_a_defect() {
        let source =
            "# Widget (widget)\n\n| VC-1.1 | FR-1 | `test` | Orphan |\n";
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
        let bad = "# Widget (widget)\n\n- **FR-1 — Size**: shall report.\n\n| VC-X | FR-1 | `test` | see VC-9.1 |\n";
        let defects = messages(bad, &rules());
        assert!(
            defects
                .iter()
                .any(|d| d == "verification row names no condition ID")
        );

        let good = "# Widget (widget)\n\n- **FR-1 — Size**: shall report.\n\n| VC-1.1 | FR-1 | `test` | Valid |\n";
        let (rows, defects) = scan_markdown(FILE, good, &rules());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.get(1).map(|r| r.id.as_str()), Some("widget#VC-1.1"));
        assert!(defects.is_empty(), "{defects:?}");
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

    fn decision(id: &str, status: Option<&str>) -> String {
        let badge = status.map_or_else(String::new, |s| {
            format!("![Status](https://img.shields.io/badge/ADR%20Status-{s}-orange)\n")
        });
        format!("# {id}: Title\n\n{badge}\n## Context\n")
    }

    fn document(cites: &str) -> String {
        format!(
            "# Widget (widget)\n\n- **FR-1 — Size**: The widget shall follow\n  {cites}.\n\nSee also ADR-0002 and ADR-0009.\n"
        )
    }

    fn decision_messages(
        records: &[Fixture<'_>],
        documents: &[Fixture<'_>],
        rules: &Rules,
    ) -> Vec<String> {
        let records: Vec<Source<'_>> =
            records.iter().map(|(f, s)| (*f, s.as_str())).collect();
        let rows: Vec<Row> = documents
            .iter()
            .flat_map(|(f, s)| scan_markdown(f, s, rules).0)
            .collect();
        check_decisions(rules, &records, &rows)
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    #[test]
    fn cited_decision_without_record_is_reported() {
        let rules = rules_with(DECISIONS);
        let records =
            [("docs/adr/0001.md", decision("ADR-0001", Some("Accepted")))];
        let documents = [(FILE, document("ADR-0001 and ADR-0003"))];
        assert_eq!(
            decision_messages(&records, &documents, &rules),
            [format!(
                "{FILE}:3: requirement widget#FR-1 cites decision ADR-0003, which has no decision record"
            )]
        );
    }

    #[test]
    fn decision_defined_twice_is_a_duplicate() {
        let rules = rules_with(DECISIONS);
        let records = [
            ("docs/adr/0001-a.md", decision("ADR-0001", Some("Accepted"))),
            ("docs/adr/0001-b.md", decision("ADR-0001", Some("Accepted"))),
        ];
        assert_eq!(
            decision_messages(&records, &[], &rules),
            [
                "docs/adr/0001-b.md:1: decision ADR-0001 is defined more than once; first definition at docs/adr/0001-a.md:1"
            ]
        );
    }

    #[test]
    fn requirement_citing_unaccepted_decision_is_reported() {
        let rules = rules_with(DECISIONS);
        let records = [
            ("docs/adr/0001.md", decision("ADR-0001", Some("Proposed"))),
            ("docs/adr/0002.md", decision("ADR-0002", Some("Superseded"))),
            ("docs/adr/0003.md", decision("ADR-0003", Some("Accepted"))),
        ];
        let documents = [(FILE, document("ADR-0001, ADR-0002 and ADR-0003"))];
        assert_eq!(
            decision_messages(&records, &documents, &rules),
            [
                format!(
                    "{FILE}:3: requirement widget#FR-1 cites decision ADR-0001 with status Proposed; requirements cite only accepted decisions"
                ),
                format!(
                    "{FILE}:3: requirement widget#FR-1 cites decision ADR-0002 with status Superseded; requirements cite only accepted decisions"
                ),
            ]
        );
    }

    #[test]
    fn citations_outside_requirement_text_are_ignored() {
        let rules = rules_with(DECISIONS);
        let records =
            [("docs/adr/0002.md", decision("ADR-0002", Some("Superseded")))];
        let documents = [(FILE, document("the size rule"))];
        assert_eq!(
            decision_messages(&records, &documents, &rules),
            Vec::<String>::new()
        );
    }

    #[test]
    fn decision_record_without_status_is_reported() {
        let rules = rules_with(DECISIONS);
        let records = [("docs/adr/0001.md", decision("ADR-0001", None))];
        assert_eq!(
            decision_messages(&records, &[], &rules),
            ["docs/adr/0001.md:1: decision ADR-0001 has no status"]
        );
        let documents = [(FILE, document("ADR-0001"))];
        assert_eq!(
            decision_messages(&records, &documents, &rules),
            [
                "docs/adr/0001.md:1: decision ADR-0001 has no status"
                    .to_string(),
                format!(
                    "{FILE}:3: requirement widget#FR-1 cites decision ADR-0001 with status none; requirements cite only accepted decisions"
                ),
            ]
        );
    }

    #[test]
    fn absent_decisions_table_disables_decision_checks() {
        let rules = rules();
        let documents = [(FILE, document("ADR-0009"))];
        assert_eq!(
            decision_messages(&[], &documents, &rules),
            Vec::<String>::new()
        );
    }

    #[test]
    fn decision_patterns_need_a_status_capture() {
        let keys = DECISIONS.replace("(?P<status>[A-Za-z]+)", "[A-Za-z]+");
        let config: TraceConfig =
            toml::from_str(&format!("{KEYS}{keys}")).unwrap();
        let error = config.rules(Path::new("trace.toml")).unwrap_err();
        assert!(
            error.to_string().contains("has no `status` capture"),
            "{error}"
        );
    }
}
