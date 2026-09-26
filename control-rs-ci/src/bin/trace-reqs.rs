//! Requirement definition and reference checker (`cargo trace-reqs`).
//!
//! Reads the Markdown files that a `trace.toml` selects, writes one row per
//! definition and reference and prints each defect as `path:line: message`.

use std::path::Path;
use std::process::ExitCode;

use control_rs_ci::GateResult;
use control_rs_ci::trace::reqs::{TraceConfig, check, scan_markdown};
use control_rs_ci::trace::select::select;
use control_rs_ci::trace::{
    Defects, USAGE_ERROR, cli_args, finish, read_text, sort_rows, write_rows,
};
use control_rs_ci::ui;

/// Command-line synopsis.
const USAGE: &str = "trace-reqs --config <trace.toml> --out <reqs.jsonl>";

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
    let config = TraceConfig::load(config_path)?;
    let rules = config.rules(config_path)?;
    let files = select(Path::new("."), &config.files, &config.doc_suffixes())?;
    let mut rows = Vec::new();
    for file in &files {
        let source = read_text(Path::new(file))?;
        rows.extend(scan_markdown(file, &source, &rules));
    }
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
    Ok(check(&rows, &rules))
}
