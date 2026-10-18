//! Requirement traceability.
//!
//! `trace-reqs` records the requirement definitions and verification
//! conditions it finds in Markdown, including the fully qualified test or
//! harness targets each condition names, and `trace-check` derives condition
//! status from those targets and the result logs of the gates. Each definition
//! and condition is one [`Row`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::{Deserialize, Serialize};

use crate::error::{GateError, GateResult};
use crate::ui;

pub mod reqs;
pub mod select;
pub mod status;

/// Row kind of a verification condition.
pub const CONDITION: &str = "condition";

/// Row kind of a requirement definition.
pub const DEFINITION: &str = "definition";

/// Row and report format version, incremented on every incompatible change.
pub const SCHEMA: u32 = 5;

/// Exit code of a usage, configuration or I/O error.
pub const USAGE_ERROR: u8 = 2;

/// Outgoing neighbor vertex indices per vertex.
type Adjacency = Vec<Vec<usize>>;

/// Defects in report order.
pub type Defects = Vec<Defect>;

/// Values of the required flags, in declared order.
pub type FlagValues<const N: usize> = [PathBuf; N];

/// A directed edge between interned vertex indices.
type IndexedEdge = (usize, usize);

/// A link from one node of the link graph to another, by ID.
pub type Link<'a> = (&'a str, &'a str);

/// Rows of one artifact.
pub type Rows = Vec<Row>;

/// Offset and length of a run of backticks.
type Run = (usize, usize);

/// A flag value while parsing, `None` until the flag is seen.
type Slot = Option<PathBuf>;

/// Interned vertex index by node identifier.
type VertexIds<'a> = BTreeMap<&'a str, usize>;

/// A GFM code span: its byte range, delimiters included, and its content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeSpan<'a> {
    /// Byte offset of the opening backticks.
    pub start: usize,
    /// Byte offset just past the closing backticks.
    pub end: usize,
    /// Text between the delimiters.
    pub content: &'a str,
}

/// A defect, reported as `path:line: message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Defect {
    /// File the defect points at.
    pub file: String,
    /// 1-based line number.
    pub line: usize,
    /// What is wrong.
    pub message: String,
}

/// One occurrence of one requirement or condition ID: a line of `reqs.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// Row format version, [`SCHEMA`].
    pub schema: u32,
    /// Qualified ID, `<doc>#<id>`.
    pub id: String,
    /// [`DEFINITION`] or [`CONDITION`].
    pub kind: String,
    /// Parent requirement IDs for conditions, `<doc>#<id>`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parents: Vec<String>,
    /// Verification method for conditions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Test or harness identifiers a condition names in its `Target` cell.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<String>,
    /// Path relative to the working directory.
    pub file: String,
    /// 1-based line number.
    pub line: usize,
    /// The matched text, each run of whitespace replaced by one space.
    pub text: String,
}

/// Depth-first search state for Tarjan's strongly connected components pass.
struct Tarjan {
    /// Outgoing neighbor indices per vertex.
    adj: Adjacency,
    /// Discovery number per vertex (`0` until visited).
    index: Vec<usize>,
    /// Lowest discovery number reachable from each vertex.
    lowlink: Vec<usize>,
    /// Whether each vertex is on the active recursion stack.
    on_stack: Vec<bool>,
    /// Active recursion stack of vertices.
    stack: Vec<usize>,
    /// Strongly connected component index per vertex.
    comp: Vec<usize>,
    /// Number of strongly connected components assigned so far.
    comp_count: usize,
    /// Next `1`-based discovery number.
    clock: usize,
}

impl Defect {
    /// A defect at the file and line of `row`.
    #[must_use]
    pub fn at(row: &Row, message: impl Into<String>) -> Self {
        Self {
            file: row.file.clone(),
            line: row.line,
            message: message.into(),
        }
    }
}

impl fmt::Display for Defect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.file, self.line, self.message)
    }
}

impl Tarjan {
    /// Initializes Tarjan's state over `adj`.
    fn new(adj: Adjacency) -> Self {
        let n = adj.len();
        Self {
            adj,
            index: vec![0; n],
            lowlink: vec![0; n],
            on_stack: vec![false; n],
            stack: Vec::with_capacity(n),
            comp: vec![0; n],
            comp_count: 0,
            clock: 1,
        }
    }

    /// Visits `v` and assigns component IDs in reverse topological order.
    fn visit(&mut self, v: usize) {
        if let Some(slot) = self.index.get_mut(v) {
            *slot = self.clock;
        }
        if let Some(slot) = self.lowlink.get_mut(v) {
            *slot = self.clock;
        }
        self.clock = self.clock.saturating_add(1);
        self.stack.push(v);
        if let Some(slot) = self.on_stack.get_mut(v) {
            *slot = true;
        }
        let neighbors = self.adj.get(v).cloned().unwrap_or_default();
        for w in neighbors {
            let w_idx = self.index.get(w).copied().unwrap_or(0);
            if w_idx == 0 {
                self.visit(w);
                let w_low = self.lowlink.get(w).copied().unwrap_or(0);
                if let Some(v_low) = self.lowlink.get_mut(v) {
                    *v_low = (*v_low).min(w_low);
                }
            } else if self.on_stack.get(w).copied().unwrap_or(false)
                && let Some(v_low) = self.lowlink.get_mut(v)
            {
                *v_low = (*v_low).min(w_idx);
            }
        }
        if self.lowlink.get(v) == self.index.get(v) {
            let cid = self.comp_count;
            self.comp_count = self.comp_count.saturating_add(1);
            while let Some(w) = self.stack.pop() {
                if let Some(slot) = self.on_stack.get_mut(w) {
                    *slot = false;
                }
                if let Some(slot) = self.comp.get_mut(w) {
                    *slot = cid;
                }
                if w == v {
                    break;
                }
            }
        }
    }
}

/// `text` with every run of whitespace, line breaks included, replaced by one
/// space, and none left at either end.
#[must_use]
pub fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The GFM code spans of `text`, in order.
///
/// A run of backticks opens a span that the next run of the same length
/// closes. A run without a closing partner is literal text.
#[must_use]
pub fn code_spans(text: &str) -> Vec<CodeSpan<'_>> {
    let mut spans = Vec::new();
    let mut pos = 0;
    while let Some((open, len)) = backtick_run(text, pos) {
        let body = open.saturating_add(len.max(1));
        let mut search = body;
        let mut close = None;
        while let Some((at, run)) = backtick_run(text, search) {
            if run == len {
                close = Some(at);
                break;
            }
            search = at.saturating_add(run.max(1));
        }
        let Some(at) = close else {
            pos = body.max(open.saturating_add(1));
            continue;
        };
        let end = at.saturating_add(len.max(1));
        spans.push(CodeSpan {
            start: open,
            end,
            content: text.get(body..at).unwrap_or_default(),
        });
        pos = end.max(open.saturating_add(1));
    }
    spans
}

/// The document of a qualified ID `<doc>#<id>`.
#[must_use]
pub fn doc_of(id: &str) -> &str {
    id.split('#').next().unwrap_or(id)
}

/// The indices of the links in `edges` that lie on a cycle.
///
/// Computed in a single Tarjan strongly connected components pass: an edge
/// `(u, v)` lies on a cycle when `u` and `v` belong to the same component.
#[must_use]
pub fn links_on_cycles(edges: &[Link<'_>]) -> BTreeSet<usize> {
    let mut ids = VertexIds::new();
    let mut adj = Adjacency::new();
    let mut indexed: Vec<IndexedEdge> = Vec::with_capacity(edges.len());
    for &(from, to) in edges {
        let u = intern_vertex(from, &mut ids, &mut adj);
        let v = intern_vertex(to, &mut ids, &mut adj);
        if let Some(neighbors) = adj.get_mut(u) {
            neighbors.push(v);
        }
        indexed.push((u, v));
    }
    let n = adj.len();
    let mut tarjan = Tarjan::new(adj);
    for v in 0..n {
        if tarjan.index.get(v).copied().unwrap_or(0) == 0 {
            tarjan.visit(v);
        }
    }
    indexed
        .into_iter()
        .enumerate()
        .filter(|(_, (u, v))| tarjan.comp.get(*u) == tarjan.comp.get(*v))
        .map(|(index, _)| index)
        .collect()
}

/// Interns `node` into `ids` and `adj`, returning its vertex index.
fn intern_vertex<'a>(
    node: &'a str,
    ids: &mut VertexIds<'a>,
    adj: &mut Adjacency,
) -> usize {
    if let Some(&id) = ids.get(node) {
        id
    } else {
        let id = adj.len();
        ids.insert(node, id);
        adj.push(Vec::new());
        id
    }
}

/// Sorts rows by `file`, then `line`, keeping the order of rows that tie.
pub fn sort_rows(rows: &mut [Row]) {
    rows.sort_by(|a, b| {
        (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line))
    });
}

/// Reads the rows of a JSON Lines file.
///
/// # Errors
/// `GateError::Io` if the file cannot be read, `GateError::Json` for a
/// malformed row and `GateError::Config` for a row whose `schema` is not
/// [`SCHEMA`].
pub fn read_rows(path: &Path) -> GateResult<Rows> {
    let file = fs::File::open(path).map_err(|e| with_path(path, &e))?;
    let mut rows = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| with_path(path, &e))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(&line)?;
        if row.schema != SCHEMA {
            return Err(GateError::Config {
                path: path.to_path_buf(),
                message: format!(
                    "line {}: row schema {} is not {SCHEMA}",
                    idx.saturating_add(1),
                    row.schema
                ),
            });
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Writes `rows` as JSON Lines, creating the parent directory.
///
/// # Errors
/// `GateError::Io` or `GateError::Json` on failure.
pub fn write_rows(path: &Path, rows: &[Row]) -> GateResult<()> {
    let mut out = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut out, row)?;
        out.push(b'\n');
    }
    write_file(path, &out)
}

/// Writes `bytes` to `path`, creating the parent directory.
///
/// # Errors
/// `GateError::Io` on failure.
pub fn write_file(path: &Path, bytes: &[u8]) -> GateResult<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| with_path(parent, &e))?;
    }
    fs::write(path, bytes).map_err(|e| with_path(path, &e))
}

/// Reads a UTF-8 text file.
///
/// # Errors
/// `GateError::Io` naming `path` on failure.
pub fn read_text(path: &Path) -> GateResult<String> {
    fs::read_to_string(path).map_err(|e| with_path(path, &e))
}

/// Prints each defect to stdout as one `path:line: message` line.
pub fn print_defects(defects: &[Defect]) {
    let mut out = io::stdout().lock();
    for defect in defects {
        let _ = writeln!(out, "{defect}");
    }
}

/// Prints the defects and returns the exit code: a failure when any exists.
/// `what` names the defects in the summary line.
#[must_use]
pub fn finish(defects: &[Defect], what: &str) -> ExitCode {
    print_defects(defects);
    if defects.is_empty() {
        ui::status("Finished", format!("no {what} defects"));
        ExitCode::SUCCESS
    } else {
        ui::failure("Failed", format!("{} {what} defects", defects.len()));
        ExitCode::FAILURE
    }
}

/// Values of the required `flags`, in order, from `--flag value` pairs.
///
/// # Errors
/// A usage message for an unknown, repeated or missing flag, or a flag
/// without a value.
pub fn parse_flags<const N: usize>(
    args: impl IntoIterator<Item = String>,
    flags: [&str; N],
) -> Result<FlagValues<N>, String> {
    use lexopt::prelude::*;
    let mut values: [Slot; N] = std::array::from_fn(|_| None);
    let mut parser =
        lexopt::Parser::from_iter(std::iter::once(String::new()).chain(args));
    while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
        match arg {
            Long(name) => {
                let full = format!("--{name}");
                let slot = flags
                    .iter()
                    .position(|flag| *flag == full)
                    .and_then(|idx| values.get_mut(idx))
                    .ok_or_else(|| format!("unknown argument `{full}`"))?;
                if slot.is_some() {
                    return Err(format!("`{full}` is given more than once"));
                }
                let val = parser
                    .value()
                    .map_err(|_| format!("`{full}` needs a value"))?;
                *slot = Some(PathBuf::from(val));
            }
            Short(c) => return Err(format!("unknown argument `-{c}`")),
            Value(val) => {
                return Err(format!(
                    "unknown argument `{}`",
                    val.to_string_lossy()
                ));
            }
        }
    }
    if let Some((flag, _)) =
        flags.iter().zip(&values).find(|(_, value)| value.is_none())
    {
        return Err(format!("missing `{flag} <path>`"));
    }
    Ok(values.map(Option::unwrap_or_default))
}

/// Parses the required `flags` from the process arguments.
///
/// `-h` or `--help` prints `usage` to stdout.
///
/// # Errors
/// The exit code `main` returns instead of running: success after help,
/// [`USAGE_ERROR`] after a usage error, which is printed with `usage`.
pub fn cli_args<const N: usize>(
    usage: &str,
    flags: [&str; N],
) -> Result<FlagValues<N>, ExitCode> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        anstream::println!("usage: {usage}");
        return Err(ExitCode::SUCCESS);
    }
    parse_flags(args, flags).map_err(|message| {
        ui::error(message);
        anstream::eprintln!("usage: {usage}");
        ExitCode::from(USAGE_ERROR)
    })
}

/// The first run of backticks at or after byte `from`: its offset and length.
fn backtick_run(text: &str, from: usize) -> Option<Run> {
    let offset = text.get(from..)?.find('`')?;
    let start = from.saturating_add(offset);
    let len = text
        .get(start..)?
        .bytes()
        .take_while(|&b| b == b'`')
        .count();
    if len == 0 {
        return None;
    }
    Some((start, len))
}

/// An I/O error whose message names `path`.
fn with_path(path: &Path, error: &io::Error) -> GateError {
    GateError::Io(io::Error::new(
        error.kind(),
        format!("{}: {error}", path.display()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(file: &str, line: usize) -> Row {
        Row {
            schema: SCHEMA,
            id: "doc#FR-1".to_string(),
            kind: DEFINITION.to_string(),
            parents: Vec::new(),
            method: None,
            targets: Vec::new(),
            file: file.to_string(),
            line,
            text: String::new(),
        }
    }

    #[test]
    fn normalize_collapses_whitespace_and_line_breaks() {
        assert_eq!(normalize("  a\n   b\t c  "), "a b c");
    }

    #[test]
    fn code_spans_follow_gfm_backtick_runs() {
        let text = "a `b` ``c ` d`` `e";
        let contents: Vec<_> =
            code_spans(text).iter().map(|s| s.content).collect();
        assert_eq!(contents, ["b", "c ` d"]);

        for text in ["", "plain text", "`", "``", "```"] {
            let got = code_spans(text);
            assert!(got.is_empty(), "{text:?}: {got:?}");
        }

        let mixed = "`open ``other`";
        let mixed_contents: Vec<_> =
            code_spans(mixed).iter().map(|s| s.content).collect();
        assert_eq!(mixed_contents, ["open ``other"]);

        let triple = "```outer `nested` outer```";
        let triple_contents: Vec<_> =
            code_spans(triple).iter().map(|s| s.content).collect();
        assert_eq!(triple_contents, ["outer `nested` outer"]);
    }

    #[test]
    fn defect_displays_as_path_line_message() {
        let defect =
            Defect::at(&row("a.md", 7), "doc#FR-1 is defined more than once");
        assert_eq!(
            defect.to_string(),
            "a.md:7: doc#FR-1 is defined more than once"
        );
    }

    #[test]
    fn rows_sort_by_file_then_line_and_keep_ties() {
        let mut first = row("b.md", 2);
        first.kind = "plan".to_string();
        let mut rows = vec![row("b.md", 2), row("a.md", 9), first.clone()];
        sort_rows(&mut rows);
        assert_eq!(rows.first().map(|r| r.file.as_str()), Some("a.md"));
        assert_eq!(rows.get(1).map(|r| r.kind.as_str()), Some(DEFINITION));
        assert_eq!(rows.get(2), Some(&first));
    }

    #[test]
    fn rows_round_trip_through_json_lines() {
        let dir = std::env::temp_dir().join(format!(
            "control_rs_ci_trace_unit_rows_{}",
            std::process::id()
        ));
        let path = dir.join("rows.jsonl");
        let rows = vec![row("a.md", 1), row("a.md", 2)];
        write_rows(&path, &rows).unwrap();
        assert_eq!(read_rows(&path).unwrap(), rows);
    }

    #[test]
    fn rows_of_another_schema_are_rejected() {
        let dir = std::env::temp_dir().join(format!(
            "control_rs_ci_trace_unit_schema_{}",
            std::process::id()
        ));
        let path = dir.join("rows.jsonl");
        let mut other = row("a.md", 1);
        other.schema = SCHEMA.saturating_add(1);
        write_rows(&path, &[other]).unwrap();
        assert!(matches!(read_rows(&path), Err(GateError::Config { .. })));
    }

    #[test]
    fn flags_parse_in_declared_order() {
        let args = ["--out", "o.jsonl", "--config", "c.toml"].map(String::from);
        let [config, out] = parse_flags(args, ["--config", "--out"]).unwrap();
        assert_eq!(config, PathBuf::from("c.toml"));
        assert_eq!(out, PathBuf::from("o.jsonl"));
    }

    #[test]
    fn flag_errors_name_the_problem() {
        let flags = ["--config", "--out"];
        let missing = parse_flags(["--config", "c"].map(String::from), flags);
        assert_eq!(missing.unwrap_err(), "missing `--out <path>`");
        let unknown = parse_flags(["--bad".to_string()], flags);
        assert_eq!(unknown.unwrap_err(), "unknown argument `--bad`");
        let repeated =
            parse_flags(["--out", "a", "--out", "b"].map(String::from), flags);
        assert_eq!(repeated.unwrap_err(), "`--out` is given more than once");
        let bare = parse_flags(["--out".to_string()], flags);
        assert_eq!(bare.unwrap_err(), "`--out` needs a value");
    }
}
