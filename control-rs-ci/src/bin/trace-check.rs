//! Requirement trace (`cargo trace-check`, gate `trace`).
//!
//! Joins `reqs.jsonl` and `marks.jsonl` with the recorded gate results,
//! writes `trace-report.json` and fails when a requirement is `Failed` or
//! `Unverified` or a marker names no defined requirement.

use std::process::ExitCode;

use control_rs_ci::GateResult;
use control_rs_ci::trace::status::{self, Derived, TraceReport};
use control_rs_ci::trace::{
    Defect, FlagValues, USAGE_ERROR, cli_args, print_defects, read_rows,
};
use control_rs_ci::ui;

/// Command-line synopsis.
const USAGE: &str = "trace-check --config <trace.toml> --reqs <reqs.jsonl> --marks <marks.jsonl> \
                     --gates <gate.toml> --results <dir> \
                     --out <trace-report.json>";

fn main() -> ExitCode {
    let flags = [
        "--config",
        "--reqs",
        "--marks",
        "--gates",
        "--results",
        "--out",
    ];
    let paths = match cli_args(USAGE, flags) {
        Ok(paths) => paths,
        Err(code) => return code,
    };
    match run(&paths) {
        Ok((report, defects)) => finish(&report, &defects),
        Err(e) => {
            ui::error(e);
            ExitCode::from(USAGE_ERROR)
        }
    }
}

/// Derives the report from the recorded artifacts and writes it.
fn run(
    [config, reqs, marks, gates, results, out]: &FlagValues<6>,
) -> GateResult<Derived> {
    let reqs = read_rows(reqs)?;
    let marks = read_rows(marks)?;
    let trace_cfg = control_rs_ci::trace::reqs::TraceConfig::load(config)?;
    let names = status::gate_names(gates)?;
    let verdicts = status::load_verdicts(results, &names);
    let gates_ctx =
        status::GateContext::new(&names, &verdicts, &trace_cfg.test_gates);
    let (report, defects) = status::derive(&reqs, &marks, &gates_ctx);
    report.write(out)?;
    ui::status("Writing", out.display());
    Ok((report, defects))
}

/// Prints the defects and the count per status; the exit code is a failure
/// unless the trace passes.
fn finish(report: &TraceReport, defects: &[Defect]) -> ExitCode {
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
