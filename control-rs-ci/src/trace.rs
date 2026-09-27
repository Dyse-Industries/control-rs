//! Requirement traceability.
//!
//! `trace-reqs` records the requirement definitions and references it finds
//! in Markdown, `trace-marks` records the markers it finds in source text, and
//! `trace-check` joins both with gate results. Every occurrence of a requirement ID is one [`Row`].

use std::fmt;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::{Deserialize, Serialize};

use crate::error::{GateError, GateResult};
use crate::ui;

pub mod marks;
pub mod reqs;
pub mod select;
pub mod status;

/// Row kind of a verification condition / decision.
pub const CONDITION: &str = "condition";

/// Row kind of a requirement definition.
pub const DEFINITION: &str = "definition";

/// Row kind of a source marker.
pub const MARKER: &str = "marker";

/// Row and report format version, incremented on every incompatible change.
pub const SCHEMA: u32 = 2;

/// Exit code of a usage, configuration or I/O error.
pub const USAGE_ERROR: u8 = 2;

/// Defects in report order.
pub type Defects = Vec<Defect>;

/// Values of the required flags, in declared order.
pub type FlagValues<const N: usize> = [PathBuf; N];

/// Rows of one artifact.
pub type Rows = Vec<Row>;

/// Offset and length of a run of backticks.
type Run = (usize, usize);

/// A flag value while parsing, `None` until the flag is seen.
type Slot = Option<PathBuf>;

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

/// One occurrence of one requirement or condition ID: a line of `reqs.jsonl`
/// or `marks.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// Row format version, [`SCHEMA`].
    pub schema: u32,
    /// Qualified ID, `<doc>#<id>`.
    pub id: String,
    /// [`DEFINITION`], [`CONDITION`], a reference kind or [`MARKER`].
    pub kind: String,
    /// Parent requirement ID for conditions, `<doc>#<id>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// Path relative to the working directory.
    pub file: String,
    /// 1-based line number.
    pub line: usize,
    /// The matched text, each run of whitespace replaced by one space.
    pub text: String,
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
        let body = open.saturating_add(len);
        let mut search = body;
        let mut close = None;
        while let Some((at, run)) = backtick_run(text, search) {
            if run == len {
                close = Some(at);
                break;
            }
            search = at.saturating_add(run);
        }
        let Some(at) = close else {
            pos = body;
            continue;
        };
        let end = at.saturating_add(len);
        spans.push(CodeSpan {
            start: open,
            end,
            content: text.get(body..at).unwrap_or_default(),
        });
        pos = end;
    }
    spans
}

/// `text` with each code span deleted, backticks included.
#[must_use]
pub fn strip_code_spans(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pos = 0;
    for span in code_spans(text) {
        out.push_str(text.get(pos..span.start).unwrap_or_default());
        pos = span.end;
    }
    out.push_str(text.get(pos..).unwrap_or_default());
    out
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
    let mut values: [Slot; N] = std::array::from_fn(|_| None);
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let slot = flags
            .iter()
            .position(|flag| *flag == arg)
            .and_then(|idx| values.get_mut(idx))
            .ok_or_else(|| format!("unknown argument `{arg}`"))?;
        if slot.is_some() {
            return Err(format!("`{arg}` is given more than once"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("`{arg}` needs a value"))?;
        *slot = Some(PathBuf::from(value));
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
    use control_rs_trace_macros::req;

    fn row(file: &str, line: usize) -> Row {
        Row {
            schema: SCHEMA,
            id: "doc#FR-1".to_string(),
            kind: DEFINITION.to_string(),
            parent: None,
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
    }

    #[test]
    fn strip_code_spans_removes_spans_and_backticks() {
        assert_eq!(
            strip_code_spans("shall return `None` from `value(i, j)` now"),
            "shall return  from  now"
        );
        assert_eq!(strip_code_spans("an `unclosed span"), "an `unclosed span");
    }

    #[test]
    fn defect_displays_as_path_line_message() {
        let defect = Defect::at(&row("a.md", 7), "doc#FR-1 is retired");
        assert_eq!(defect.to_string(), "a.md:7: doc#FR-1 is retired");
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
        let dir = std::env::temp_dir().join("control_rs_ci_trace_unit_rows");
        let path = dir.join("rows.jsonl");
        let rows = vec![row("a.md", 1), row("a.md", 2)];
        write_rows(&path, &rows).unwrap();
        assert_eq!(read_rows(&path).unwrap(), rows);
    }

    #[req("requirement-traceability#VC-14.1")]
    #[test]
    fn rows_of_another_schema_are_rejected() {
        let dir = std::env::temp_dir().join("control_rs_ci_trace_unit_schema");
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
