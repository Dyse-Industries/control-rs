//! [`QualityGate`] trait and concrete implementations for the CI pipeline.
//!
//! One [`run_gate`] function owns skip detection, wall-clock timing,
//! warn-vs-fail policy, and counter updates for every gate family.

use std::time::Instant;

use crate::gates;
use crate::report::GateVerdict;
use crate::runner::{Gate, GateMode, PipelineCtx};

// ── Types ────────────────────────────────────────────────────────────────────

/// `(key, value)` environment overrides passed to a spawned `cargo` command.
type EnvPairs<'a> = &'a [(&'a str, &'a str)];

/// `(passed, output, details)` returned by host-tool dispatch.
type HostToolOutcome = (bool, String, String);

/// Outcome returned by every [`QualityGate::execute`] call.
pub(crate) struct GateOutcome {
    /// `true` when the gate command exited successfully.
    pub(crate) passed: bool,
    /// `true` when the gate could not apply on this host and ran nothing.
    /// A gate that did not run reports Skip, never Pass: a green verdict
    /// must mean the check executed.
    pub(crate) skipped: bool,
    /// Captured combined stdout + stderr (ANSI-stripped).
    pub(crate) output: String,
    /// One-line verdict detail for `ci-report.md`.
    pub(crate) details: String,
}

/// Result of [`run_gate`]: verdict plus the raw output/details/timing that
/// callers store into `ci-report.md` fields.
pub(crate) struct GateRunResult {
    /// Pass / fail / skip / warn verdict.
    pub(crate) verdict: GateVerdict,
    /// Captured combined stdout + stderr (ANSI-stripped).
    pub(crate) output: String,
    /// One-line verdict detail for `ci-report.md`.
    pub(crate) details: String,
    /// Wall-clock time taken by the gate, in seconds.
    pub(crate) time: f32,
}

/// A quality gate that can be executed inside a [`PipelineCtx`].
pub(crate) trait QualityGate {
    /// Execute the gate, optionally writing summary data into `ctx`.
    fn execute(&self, ctx: &mut PipelineCtx<'_>) -> GateOutcome;
}

/// A standard gate that runs `cargo <subcommand> [args…]` and returns pass/fail.
///
/// Covers: clean, fmt, clippy, check, build, test.
pub(crate) struct CargoArgvGate<'a> {
    /// Label shown in status lines (e.g. `` `cargo clippy …` ``).
    pub(crate) label: &'a str,
    /// Subcommand + arguments (e.g. `&["clippy", "--workspace", …]`).
    pub(crate) argv: &'a [&'a str],
    /// Environment overrides.
    pub(crate) envs: EnvPairs<'a>,
}

/// A gate that wraps an off-the-shelf host verification tool.
///
/// Covers: mutants, miri, valgrind, fuzz, lockbud, deny, audit, kani.
pub(crate) struct HostToolGate {
    /// The gate variant to dispatch (must be a host-tool gate).
    pub(crate) gate: Gate,
}

/// Gate that runs tarpaulin and stores coverage summary in [`PipelineCtx`].
pub struct CoverageGate;

/// Gate that runs requirement traceability and stores the summary.
pub struct TraceGate;

/// Gate that runs cross-validation oracle comparison and stores the summary.
pub struct ValidateGate;

/// Gate that runs the embedded target matrix and stores results.
pub struct EtsGate;

// ── Impls ────────────────────────────────────────────────────────────────────

impl QualityGate for CargoArgvGate<'_> {
    fn execute(&self, _ctx: &mut PipelineCtx<'_>) -> GateOutcome {
        let (passed, output) = run_cargo(self.label, self.argv, self.envs);
        GateOutcome {
            passed,
            skipped: false,
            output,
            details: String::new(),
        }
    }
}

impl QualityGate for HostToolGate {
    fn execute(&self, ctx: &mut PipelineCtx<'_>) -> GateOutcome {
        if let Some(reason) = gates::host_tool_inapplicable(self.gate) {
            return GateOutcome {
                passed: false,
                skipped: true,
                output: String::new(),
                details: reason,
            };
        }
        let (passed, output, details) = run_host_tool(self.gate, ctx);
        GateOutcome {
            passed,
            skipped: false,
            output,
            details,
        }
    }
}

impl QualityGate for CoverageGate {
    fn execute(&self, ctx: &mut PipelineCtx<'_>) -> GateOutcome {
        let start = Instant::now();
        let (res, output) = gates::run_tarpaulin();
        ctx.state.coverage.time = start.elapsed().as_secs_f32();
        ctx.state.tarp_output.clone_from(&output);
        let passed = res.is_ok();
        if let Ok(summary) = res {
            ctx.state.tarp_summary = summary;
        }
        GateOutcome {
            passed,
            skipped: false,
            output,
            details: String::new(),
        }
    }
}

impl QualityGate for TraceGate {
    fn execute(&self, ctx: &mut PipelineCtx<'_>) -> GateOutcome {
        let start = Instant::now();
        let mut combined_ets_tests = Vec::new();
        for (name, res) in &ctx.state.matrix_results {
            if let Ok(run_result) = res {
                for t in &run_result.results {
                    let mut tagged = t.clone();
                    tagged.suite_name =
                        format!("{} ({})", tagged.suite_name, name);
                    combined_ets_tests.push(tagged);
                }
            }
        }
        let ets_json_sample = serde_json::to_string(&combined_ets_tests).ok();
        let test_output_opt = if ctx.state.test_cmd.output.trim().is_empty() {
            None
        } else {
            Some(ctx.state.test_cmd.output.as_str())
        };
        let (passed, summary, _) = gates::run_traceability(
            &ctx.config.repo_root,
            test_output_opt,
            ets_json_sample.as_deref(),
        );
        ctx.state.trace.time = start.elapsed().as_secs_f32();
        ctx.state.trace_summary = Some(summary);
        GateOutcome {
            passed,
            skipped: false,
            output: String::new(),
            details: String::new(),
        }
    }
}

impl QualityGate for ValidateGate {
    fn execute(&self, ctx: &mut PipelineCtx<'_>) -> GateOutcome {
        if ctx.example_targets.is_empty() {
            return GateOutcome {
                passed: false,
                skipped: false,
                output: String::new(),
                details: "no suites configured in toml".to_string(),
            };
        }
        let start = Instant::now();
        let (passed, summary, _) = gates::run_cross_comparison(
            ctx.example_targets,
            &ctx.config.repo_root,
            ctx.timeout_secs,
        );
        ctx.state.cross_val.time = start.elapsed().as_secs_f32();
        ctx.state.cross_val_summary = Some(summary);
        GateOutcome {
            passed,
            skipped: false,
            output: String::new(),
            details: String::new(),
        }
    }
}

impl QualityGate for EtsGate {
    // run_ets_gate (runner.rs) guarantees ctx.targets_to_run is non-empty
    // before this is ever called -- an empty target list is handled there
    // as an implicit skip, not a failure.
    fn execute(&self, ctx: &mut PipelineCtx<'_>) -> GateOutcome {
        use std::time::Duration;
        crate::report::status("Running", Gate::Ets.command_label());
        let start = Instant::now();
        ctx.state.matrix_results = crate::target_matrix::execute_target_matrix(
            ctx.targets_to_run,
            Duration::from_secs(ctx.timeout_secs),
        );
        ctx.state.ets_time = start.elapsed().as_secs_f32();
        let passed = crate::runner::ets_all_passed(&ctx.state.matrix_results);
        GateOutcome {
            passed,
            skipped: false,
            output: String::new(),
            details: String::new(),
        }
    }
}

// ── Functions ────────────────────────────────────────────────────────────────

/// Run one quality gate, handling skip, timing, warn-vs-fail, and counters.
///
/// Returns the final verdict plus captured output/details/timing so callers
/// can store them in `PipelineState`.
pub(crate) fn run_gate(
    id: Gate,
    mode: GateMode,
    job: &impl QualityGate,
    ctx: &mut PipelineCtx<'_>,
) -> GateRunResult {
    use crate::report;

    if mode == GateMode::Skip {
        ctx.state.counters.skipped_gates =
            ctx.state.counters.skipped_gates.saturating_add(1);
        report::status("Skipped", id.command_label());
        return GateRunResult {
            verdict: GateVerdict::Skip,
            output: String::new(),
            details: String::new(),
            time: 0.0,
        };
    }

    let start = Instant::now();
    let outcome = job.execute(ctx);
    let time = start.elapsed().as_secs_f32();
    if outcome.skipped {
        ctx.state.counters.skipped_gates =
            ctx.state.counters.skipped_gates.saturating_add(1);
        report::status("Skipped", id.command_label());
        return GateRunResult {
            verdict: GateVerdict::Skip,
            output: outcome.output,
            details: outcome.details,
            time,
        };
    }
    let verdict = verdict_from(outcome.passed, mode);
    apply_gate_counters(id, mode, outcome.passed, ctx);
    GateRunResult {
        verdict,
        output: outcome.output,
        details: outcome.details,
        time,
    }
}

pub(crate) fn verdict_from(passed: bool, mode: GateMode) -> GateVerdict {
    if passed {
        GateVerdict::Pass
    } else if mode == GateMode::Warn {
        GateVerdict::Warn
    } else {
        GateVerdict::Fail
    }
}

pub(crate) fn apply_gate_counters(
    id: Gate,
    mode: GateMode,
    passed: bool,
    ctx: &mut PipelineCtx<'_>,
) {
    use crate::report;
    if passed {
        ctx.state.counters.passed_gates =
            ctx.state.counters.passed_gates.saturating_add(1);
        return;
    }
    if mode == GateMode::Warn {
        ctx.state.counters.warned_gate_names.push(id.name());
        report::status_warn("Warned", id.command_label());
        return;
    }
    ctx.state.counters.failed_gate_names.push(id.name());
    ctx.state.counters.ci_success = false;
}

fn run_cargo(label: &str, argv: &[&str], envs: EnvPairs<'_>) -> (bool, String) {
    use crate::report;
    use std::process::{Command, Stdio};

    report::status("Running", label);
    let mut cmd = Command::new("cargo");
    cmd.arg("--color").arg(gates::cargo_color());
    cmd.args(argv);
    gates::apply_cargo_term_color(&mut cmd);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    match cmd.spawn() {
        Ok(mut child) => {
            let tee = gates::StdioTee::attach(&mut child);
            match child.wait() {
                Ok(status) => (status.success(), tee.join()),
                Err(e) => (false, format!("Failed to execute {label}: {e}")),
            }
        }
        Err(e) => (false, format!("Failed to execute {label}: {e}")),
    }
}

fn labeled_host(gate: Gate, (ok, output): (bool, String)) -> HostToolOutcome {
    (ok, output, gate.command_label().to_string())
}

/// Dispatch host-tool execution.
fn run_host_tool(gate: Gate, ctx: &PipelineCtx<'_>) -> HostToolOutcome {
    match gate {
        Gate::Mutants => {
            let (res, output) = gates::run_mutants(ctx.out_path);
            match res {
                Ok(ref summary) => {
                    (true, output, gates::mutants_details(summary))
                }
                Err(_) => (
                    false,
                    output,
                    "`cargo mutants` failed: surviving or timeout mutants"
                        .to_string(),
                ),
            }
        }
        Gate::Miri => labeled_host(gate, gates::run_miri()),
        Gate::Valgrind => labeled_host(gate, gates::run_valgrind()),
        Gate::Fuzz => labeled_host(
            gate,
            gates::run_fuzz(ctx.workspace_dir, ctx.timeout_secs),
        ),
        Gate::Lockbud => labeled_host(gate, gates::run_lockbud()),
        Gate::Deny => labeled_host(gate, gates::run_deny()),
        Gate::Audit => labeled_host(gate, gates::run_audit()),
        Gate::Kani => labeled_host(gate, gates::run_kani()),
        Gate::Clean
        | Gate::Fmt
        | Gate::Clippy
        | Gate::Check
        | Gate::Build
        | Gate::Test
        | Gate::Coverage
        | Gate::Trace
        | Gate::Ets
        | Gate::Validate => (
            false,
            String::new(),
            format!(
                "host-tool dispatch received non-host gate {}",
                gate.command_label()
            ),
        ),
    }
}
