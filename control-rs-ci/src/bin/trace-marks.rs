//! Marker scanner (`cargo trace-marks`).
//!
//! Reads the source files that the `[markers]` table of a `trace.toml`
//! selects, writes one row per marked requirement ID to `marks.jsonl` and
//! prints each defect as `path:line: message`. Without a `[markers]` table
//! it writes an empty `marks.jsonl`.

use std::path::Path;
use std::process::ExitCode;

use control_rs_ci::GateResult;
use control_rs_ci::trace::reqs::TraceConfig;
use control_rs_ci::trace::{
    Defects, USAGE_ERROR, cli_args, finish, marks, write_rows,
};
use control_rs_ci::ui;

/// Command-line synopsis.
const USAGE: &str = "trace-marks --config <trace.toml> --out <marks.jsonl>";

fn main() -> ExitCode {
    let [config, out] = match cli_args(USAGE, ["--config", "--out"]) {
        Ok(args) => args,
        Err(code) => return code,
    };
    match run(&config, &out) {
        Ok(defects) => finish(&defects, "marker"),
        Err(e) => {
            ui::error(e);
            ExitCode::from(USAGE_ERROR)
        }
    }
}

/// Scans the selected source files, writes their rows to `out` and returns
/// the defects.
fn run(config_path: &Path, out: &Path) -> GateResult<Defects> {
    if out.exists() {
        let _ = std::fs::remove_file(out);
    }
    let config = TraceConfig::load(config_path)?;
    let rules = config.rules(config_path)?;
    let (rows, defects) = match &config.markers {
        Some(markers) => marks::collect(Path::new("."), markers, &rules)?,
        None => (Vec::new(), Vec::new()),
    };
    write_rows(out, &rows)?;
    ui::status(
        "Writing",
        format!("{} ({} markers)", out.display(), rows.len()),
    );
    Ok(defects)
}
