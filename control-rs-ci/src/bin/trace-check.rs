//! Requirement trace (`cargo trace-check`, gate `trace`).
//!
//! Checks each verification condition of `reqs.jsonl` against the result logs
//! of its method, writes `trace-report.json` and fails when a condition is
//! `Uncovered`, `Unrun` or `Fail`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use control_rs_ci::GateResult;
use control_rs_ci::trace::link::{GraphFormat, link_graph};
use control_rs_ci::trace::reqs::TraceConfig;
use control_rs_ci::trace::status::{self, Derived, TraceReport};
use control_rs_ci::trace::{
    Defect, FlagValues, USAGE_ERROR, cli_args_with, print_defects, read_rows,
    write_file,
};
use control_rs_ci::ui;

/// Command-line synopsis.
const USAGE: &str = "trace-check --config <trace.toml> --reqs <reqs.jsonl> --out <trace-report.json> [--graph <links.mmd|links.dot>]";

/// A graph file and the format its name selects.
type GraphFile = (PathBuf, GraphFormat);

fn main() -> ExitCode {
    let required = ["--config", "--reqs", "--out"];
    let (paths, [graph]) = match cli_args_with(USAGE, required, ["--graph"]) {
        Ok(parsed) => parsed,
        Err(code) => return code,
    };
    let format = match graph.as_deref().map(graph_format).transpose() {
        Ok(format) => format,
        Err(message) => {
            ui::error(message);
            anstream::eprintln!("usage: {USAGE}");
            return ExitCode::from(USAGE_ERROR);
        }
    };
    match run(&paths, graph.zip(format)) {
        Ok((report, defects)) => finish(&report, &defects),
        Err(e) => {
            ui::error(e);
            ExitCode::from(USAGE_ERROR)
        }
    }
}

/// The graph format a file name selects: `.mmd` is Mermaid, `.dot` is DOT.
fn graph_format(path: &Path) -> Result<GraphFormat, String> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("mmd") => Ok(GraphFormat::Mermaid),
        Some("dot") => Ok(GraphFormat::Dot),
        _ => Err(format!(
            "`--graph {}` needs a `.mmd` or `.dot` file name",
            path.display()
        )),
    }
}

/// Derives the report from the recorded artifacts and writes it, then the
/// link graph when `graph` names one.
fn run(
    [config, reqs, out]: &FlagValues<3>,
    graph: Option<GraphFile>,
) -> GateResult<Derived> {
    if out.exists() {
        let _ = std::fs::remove_file(out);
    }
    let reqs = read_rows(reqs)?;
    let trace_cfg = TraceConfig::load(config)?;
    let (links, link_defects) = link_graph(&reqs);
    let (report, defects) = status::derive_with_graph(
        &links,
        link_defects,
        &trace_cfg,
        Path::new("."),
    );
    report.write(out)?;
    ui::status("Writing", out.display());
    if let Some((path, format)) = graph {
        let text = status::render_graph(&links, &report, format);
        write_file(&path, text.as_bytes())?;
        ui::status("Writing", path.display());
    }
    Ok((report, defects))
}

/// Prints the defects and the count per status; the exit code is a failure
/// unless the trace passes.
fn finish(report: &TraceReport, defects: &[Defect]) -> ExitCode {
    for warning in &report.warnings {
        ui::warning("Warning", warning.to_string());
    }
    print_defects(defects);
    let counts = report
        .counts
        .iter()
        .map(|(status, count)| format!("{count} {status:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    if report.passes() {
        ui::status("Finished", counts);
        ExitCode::SUCCESS
    } else {
        ui::failure("Failed", counts);
        ExitCode::FAILURE
    }
}
