//! Requirement definition and reference checker (`cargo trace-reqs`).
//!
//! Reads the Markdown files that a `trace.toml` selects, writes one row per
//! definition and reference and prints each defect as `path:line: message`.

use std::path::Path;
use std::process::ExitCode;

use control_rs_ci::GateResult;
use control_rs_ci::trace::reqs::{
    Rules, Source, TraceConfig, check, check_decisions, scan_markdown,
};
use control_rs_ci::trace::select::select;
use control_rs_ci::trace::{
    Defects, Row, USAGE_ERROR, cli_args, finish, read_text, sort_rows,
    write_rows,
};
use control_rs_ci::ui;

/// Command-line synopsis.
const USAGE: &str = "trace-reqs --config <trace.toml> --out <reqs.jsonl>";

/// A file path and its text, owned.
type OwnedSource = (String, String);

fn main() -> ExitCode {
    let [config, out] = match cli_args(USAGE, ["--config", "--out"]) {
        Ok(args) => args,
        Err(code) => return code,
    };
    match run(&config, &out) {
        Ok(defects) => finish(&defects, "requirement"),
        Err(e) => {
            ui::error(e);
            ExitCode::from(USAGE_ERROR)
        }
    }
}

/// Scans the selected files, writes their rows to `out` and returns the
/// defects.
fn run(config_path: &Path, out: &Path) -> GateResult<Defects> {
    if out.exists() {
        let _ = std::fs::remove_file(out);
    }
    let config = TraceConfig::load(config_path)?;
    let rules = config.rules(config_path)?;
    let files = select(Path::new("."), &config.files, &config.doc_suffixes())?;
    if files.is_empty() {
        return Err(control_rs_ci::GateError::Config {
            path: config_path.to_path_buf(),
            message: "no Markdown files selected by `files`".to_string(),
        });
    }
    let mut rows = Vec::new();
    let mut defects = Vec::new();
    for file in &files {
        let source = read_text(Path::new(file))?;
        let (file_rows, file_defects) = scan_markdown(file, &source, &rules);
        rows.extend(file_rows);
        defects.extend(file_defects);
    }
    defects.extend(decision_defects(&config, &rules, &rows)?);
    sort_rows(&mut rows);
    write_rows(out, &rows)?;
    ui::status(
        "Writing",
        format!(
            "{} ({} rows from {} files)",
            out.display(),
            rows.len(),
            files.len()
        ),
    );
    defects.extend(check(&rows, &rules));
    defects.sort_by(|a, b| {
        (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line))
    });
    Ok(defects)
}

/// The decision defects of the decision records and of the requirement
/// `rows`; none when the configuration has no `[decisions]` table.
fn decision_defects(
    config: &TraceConfig,
    rules: &Rules,
    rows: &[Row],
) -> GateResult<Defects> {
    let Some(decisions) = &config.decisions else {
        return Ok(Defects::new());
    };
    let files =
        select(Path::new("."), &decisions.files, &config.doc_suffixes())?;
    let mut records = Vec::new();
    for file in files {
        let source = read_text(Path::new(&file))?;
        records.push((file, source));
    }
    Ok(check_decisions(rules, &pairs(&records), rows))
}

/// Borrowed file and text pairs of `list`.
fn pairs(list: &[OwnedSource]) -> Vec<Source<'_>> {
    list.iter().map(|(f, s)| (f.as_str(), s.as_str())).collect()
}
