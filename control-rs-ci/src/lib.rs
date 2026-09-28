//! # `control-rs-ci`
//!
//! Continuous Integration & Quality Gate Infrastructure for `control-rs`.
//!
//! Provides a modular quality gate execution engine, standardized process outcome
//! records, decentralized artifact aggregation, and automated Markdown report rendering.

#![deny(missing_docs)]

pub use cli::run_cli;
pub use config::{GateConfig, GatePolicy, Stage};
pub use error::{GateError, GateResult};
pub use gate::{Gate, GateContext, GateList, GateOutcome, SharedGate, Verdict};
pub use report::{ReportAggregator, WrittenReport};

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub mod allow_audit;
pub mod cli;

pub mod config;

pub mod error;

pub mod ets;

pub mod gate;

pub mod report;

pub mod trace;

pub mod ui;

/// Workspace-relative directory that holds one Cargo target directory per
/// execution group (`target/ci-groups/<group>`).
///
/// Concurrent groups that share one target directory serialize on Cargo's
/// build-directory lock. A dedicated directory per group removes that
/// contention. `--clean` deletes this directory, so every group rebuilds from
/// an empty target directory.
pub const GROUP_TARGET_ROOT: &str = "target/ci-groups";

/// Concurrent groups in declaration order, each with its gates in pipeline order.
type GroupedGates<'a> = Vec<(&'a str, GateList)>;

/// Selected gates split by stage.
#[derive(Debug, Default)]
struct Schedule<'a> {
    /// `exclusive.pre` gates, run first on the calling thread.
    pre: GateList,
    /// Concurrent groups.
    groups: GroupedGates<'a>,
    /// `exclusive.post` and unscheduled gates, run last on the calling thread.
    post: GateList,
}

/// State shared by every group of one pipeline run.
#[derive(Debug, Clone, Copy)]
pub struct RunEnv<'a> {
    /// Execution context: workspace root, artifact directory, timeout.
    pub ctx: &'a GateContext,
    /// Serializes status lines from concurrent groups.
    pub ui_lock: &'a Mutex<()>,
    /// Echo each gate's output behind its group tag.
    pub verbose: bool,
}

/// One execution group: its name, color and optional Cargo target directory.
#[derive(Debug, Clone, Copy)]
pub struct GroupSpec<'a> {
    /// Group name printed in the tag (`pre`, `post` or a declared group).
    pub name: &'a str,
    /// Tag color.
    pub style: anstyle::Style,
    /// Dedicated `CARGO_TARGET_DIR`, or `None` for the shared one.
    pub target_dir: Option<&'a Path>,
}

/// Result of running one gate list.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupRun {
    /// Elapsed wall-clock time in seconds.
    pub duration_secs: f64,
    /// Whether any gate recorded `Verdict::Fail` or could not record a result.
    pub failed: bool,
}

/// Gate selection and behavior switches for one pipeline run.
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineOptions<'a> {
    /// Run only these gates, when set.
    pub only_gates: GateFilter<'a>,
    /// Never run these gates, when set.
    pub skip_gates: GateFilter<'a>,
    /// Stop after this gate.
    pub up_to_gate: Option<&'a str>,
    /// Without `only_gates`, also select `default = false` gates.
    pub all: bool,
    /// Clean artifacts before running.
    pub clean: bool,
    /// Echo gate output.
    pub verbose: bool,
    /// Arguments appended to the selected gate's `args` (FR-14). Non-empty
    /// only when exactly one gate is selected.
    pub extra_args: &'a [String],
    /// Bounded group concurrency limit (FR-13); overrides `[execution] max_jobs`.
    pub max_jobs: Option<usize>,
}

/// Optional list of gate names used to filter a run.
pub type GateFilter<'a> = Option<&'a [String]>;

/// Returns `gate` with `CARGO_TARGET_DIR` set to `target_dir`, unless the gate
/// already declares its own `CARGO_TARGET_DIR` in `gate.toml`.
fn with_target_dir(gate: &SharedGate, target_dir: &Path) -> SharedGate {
    if gate.env.contains_key("CARGO_TARGET_DIR") {
        return Arc::clone(gate);
    }
    let mut scoped = (**gate).clone();
    scoped.env.insert(
        "CARGO_TARGET_DIR".to_string(),
        target_dir.display().to_string(),
    );
    Arc::new(scoped)
}

/// Returns `gate` with `extra` appended after its configured `args`.
fn with_extra_args(gate: &SharedGate, extra: &[String]) -> SharedGate {
    if extra.is_empty() {
        return Arc::clone(gate);
    }
    let mut extended = (**gate).clone();
    extended.args.extend_from_slice(extra);
    Arc::new(extended)
}

/// Cleans up the CI artifacts directory, the per-group target directories and
/// any leftover workspace root artifacts.
///
/// Removes the configured output directory (`out_dir`, typically `target/ci-artifacts`),
/// the per-group Cargo target directories under [`GROUP_TARGET_ROOT`] and any
/// stray legacy CI artifacts found in the workspace root.
///
/// # Errors
/// Returns `GateError` if configuration loading or directory deletion fails.
pub fn clean_artifacts(
    workspace_root: &Path,
    config_path: &Path,
) -> GateResult<PathBuf> {
    let config = GateConfig::load_from_path(config_path)?;
    let out_dir = workspace_root.join(&config.runner.out_dir);

    if out_dir.exists() {
        std::fs::remove_dir_all(&out_dir)?;
    }

    let group_targets = workspace_root.join(GROUP_TARGET_ROOT);
    if group_targets.exists() {
        std::fs::remove_dir_all(&group_targets)?;
    }

    // Remove any stray or legacy CI artifacts from the workspace root
    let stray_root_artifacts = [
        "ci-report.md",
        "mutants.out",
        "mutants.out.old",
        "tarpaulin-report.html",
        "tarpaulin-report.json",
    ];
    for artifact in &stray_root_artifacts {
        let p = workspace_root.join(artifact);
        if p.is_dir() {
            let _ = std::fs::remove_dir_all(&p);
        } else if p.is_file() {
            let _ = std::fs::remove_file(&p);
        }
    }

    Ok(out_dir)
}

/// Runs one gate and prints its status lines. Returns whether it failed:
/// `Verdict::Fail`, or no result could be recorded.
fn execute_gate(gate: &Gate, tag: &str, env: &RunEnv<'_>) -> bool {
    let name = gate.name();
    {
        let _guard = env.ui_lock.lock();
        ui::status("Running", format!("{tag}{}", gate.command_display()));
    }
    let echo = env.verbose.then(|| ui::format_echo_prefix(tag, name));
    match gate.execute_with_echo(env.ctx, echo.as_deref()) {
        Ok(outcome) => {
            let _guard = env.ui_lock.lock();
            report_outcome(&outcome, tag, name);
            outcome.verdict == Verdict::Fail
        }
        Err(e) => {
            let _guard = env.ui_lock.lock();
            ui::error(format!("{tag}Error executing gate '{name}': {e}"));
            true
        }
    }
}

/// Prints the status line for a finished gate.
fn report_outcome(outcome: &GateOutcome, tag: &str, name: &str) {
    let summary = outcome
        .summary
        .as_ref()
        .map_or_else(String::new, |s| format!(" ({s})"));
    let secs = outcome.duration_secs;
    match outcome.verdict {
        Verdict::Pass => {
            ui::status("Passed", format!("{tag}{name} in {secs:.2}s"));
        }
        Verdict::Warn => {
            ui::warning(
                "Warning",
                format!("{tag}{name} in {secs:.2}s{summary}"),
            );
        }
        Verdict::Fail => {
            ui::failure(
                "Failed",
                format!("{tag}{name} in {secs:.2}s{summary}"),
            );
        }
        Verdict::Skipped => {
            ui::warning("Skipped", format!("{tag}{name}"));
        }
    }
}

/// Runs `gates` one at a time on the calling thread.
///
/// With `group.target_dir`, the directory is created if missing and every
/// gate runs with `CARGO_TARGET_DIR` set to it, except a gate that declares
/// its own `CARGO_TARGET_DIR` in `gate.toml`. Without it, gates inherit the
/// caller's environment and build in the shared target directory.
///
/// Each pool worker calls this for the groups it takes from the queue; the
/// calling thread calls it for the `pre` and `post` stages.
///
/// # Errors
/// Returns `GateError::Io` if the target directory cannot be created.
pub fn run_group(
    group: &GroupSpec<'_>,
    gates: &[SharedGate],
    env: &RunEnv<'_>,
) -> GateResult<GroupRun> {
    let start = Instant::now();
    if let Some(dir) = group.target_dir {
        std::fs::create_dir_all(dir)?;
    }
    let tag = ui::format_group_tag(Some(group.name), Some(group.style));
    let mut failed = false;
    for gate in gates {
        let gate = group
            .target_dir
            .map_or_else(|| Arc::clone(gate), |dir| with_target_dir(gate, dir));
        failed |= execute_gate(&gate, &tag, env);
    }
    Ok(GroupRun {
        duration_secs: start.elapsed().as_secs_f64(),
        failed,
    })
}

/// Concurrency limit: `--max-jobs`, then `[execution] max_jobs`, then the
/// number of groups.
fn resolve_max_jobs(
    options: &PipelineOptions<'_>,
    config: &GateConfig,
    config_path: &Path,
    group_count: usize,
) -> GateResult<usize> {
    match options.max_jobs.or(config.execution.max_jobs) {
        Some(0) => Err(GateError::Config {
            path: config_path.to_path_buf(),
            message: "max_jobs must be greater than zero".to_string(),
        }),
        Some(n) => Ok(n),
        None => Ok(group_count.max(1)),
    }
}

fn record_disabled_gates(
    disabled: &[SharedGate],
    ctx: &GateContext,
) -> GateResult<()> {
    for gate in disabled {
        let outcome = gate.record_disabled(ctx)?;
        report_outcome(&outcome, "", gate.name());
    }
    Ok(())
}

fn display_pipeline_summary(report_path: &Path, passed: bool, duration: f64) {
    ui::status("Writing", format!("{}", report_path.display()));
    if passed {
        ui::status("Finished", format!("ci in {duration:.2}s"));
    } else {
        ui::failure(
            "Failed",
            format!("ci in {duration:.2}s (see report for details)"),
        );
    }
}

/// Runs the complete CI quality gate pipeline or filtered subset.
///
/// Selected gates run in three stages: `exclusive.pre` gates one at a time,
/// then the groups on the bounded worker pool, then (after every group has
/// joined) `exclusive.post` and unscheduled gates one at a time. A `pre` gate
/// that fails stops the run before any group starts; the gates left unexecuted
/// have no result, so the report fails them.
///
/// # Arguments
/// * `workspace_root` - Root directory of the Cargo workspace.
/// * `config_path` - Path to `gate.toml`.
/// * `options` - Gate filters (`only_gates`, `skip_gates`, `up_to_gate`),
///   whether to clean the artifacts directory first and whether to echo
///   gate output.
///
/// # Errors
/// Returns `GateError` if configuration loading, gate execution, or report rendering fails.
pub fn run_pipeline(
    workspace_root: &Path,
    config_path: &Path,
    options: &PipelineOptions<'_>,
) -> GateResult<bool> {
    let pipeline_start = Instant::now();
    if options.clean {
        let _ = clean_artifacts(workspace_root, config_path)?;
    }
    let config = GateConfig::load_from_path(config_path)?;
    let out_dir = workspace_root.join(&config.runner.out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let ctx = GateContext {
        workspace_root: workspace_root.to_path_buf(),
        out_dir: out_dir.clone(),
        default_timeout: Duration::from_secs(config.runner.timeout_secs),
    };

    let selected = selected_with_passthrough(&config, config_path, options)?;
    if selected.is_empty() {
        ui::warn_diag("no gates selected; `--list` shows the registered gates");
        return Ok(false);
    }
    let executed_gate_names: Vec<String> =
        selected.iter().map(|g| g.name().to_string()).collect();
    // Every result is rewritten or absent after this run, so a result from
    // an earlier run can never stand in for a gate that did not run now.
    remove_results(&out_dir);

    // A disabled gate selected by name records `Skipped` and does not run.
    let (disabled, active_gates): (GateList, GateList) = selected
        .into_iter()
        .partition(|g| g.mode() == GatePolicy::Skip);
    record_disabled_gates(&disabled, &ctx)?;

    let schedule = partition_gates(active_gates, &config);
    let ui_lock = Mutex::new(());
    let env = RunEnv {
        ctx: &ctx,
        ui_lock: &ui_lock,
        verbose: options.verbose,
    };
    let max_jobs =
        resolve_max_jobs(options, &config, config_path, schedule.groups.len())?;

    run_stages(
        &schedule,
        &env,
        &workspace_root.join(GROUP_TARGET_ROOT),
        max_jobs,
    )?;

    let aggregator =
        ReportAggregator::new(out_dir, workspace_root.to_path_buf());
    let report = aggregator
        .write_report(&config, Some(executed_gate_names.as_slice()))?;
    let total_duration = pipeline_start.elapsed().as_secs_f64();

    display_pipeline_summary(&report.path, report.pass, total_duration);
    Ok(report.pass)
}

/// Runs the `pre` stage, then (unless a `pre` gate failed) the groups on the
/// worker pool and, after the barrier join, the `post` stage.
///
/// # Errors
/// Returns `GateError::Io` if a target directory cannot be created.
fn run_stages(
    schedule: &Schedule<'_>,
    env: &RunEnv<'_>,
    group_targets: &Path,
    max_jobs: usize,
) -> GateResult<()> {
    let stage = |name| GroupSpec {
        name,
        style: ui::exclusive_style(),
        target_dir: None,
    };
    if run_group(&stage("pre"), &schedule.pre, env)?.failed {
        let not_run = schedule
            .groups
            .iter()
            .map(|(_, gates)| gates.len())
            .sum::<usize>()
            .saturating_add(schedule.post.len());
        ui::failure(
            "Aborted",
            format!("a pre gate failed; {not_run} selected gates not run"),
        );
        return Ok(());
    }
    run_concurrent_groups(&schedule.groups, group_targets, env, max_jobs)?;
    // Barrier passed: post gates run on the calling thread, one at a time,
    // in the shared target directory.
    run_group(&stage("post"), &schedule.post, env)?;
    Ok(())
}

/// Applies the `only`/`skip`/`up_to` filters and default selection to the
/// configured gates, preserving pipeline order.
///
/// Without `only_gates`, a gate is selected when its policy is not `skip`
/// and, unless `all` is set, its definition does not set `default = false`.
/// Gates named in `only_gates` are selected whatever their policy; the caller
/// records disabled ones as `Skipped` instead of running them.
fn select_gates(
    all_gates: GateList,
    config: &GateConfig,
    options: &PipelineOptions<'_>,
) -> GateList {
    let mut active_gates = GateList::new();
    for gate in all_gates {
        let name = gate.name();
        let selected = options.only_gates.map_or_else(
            || {
                config.policy_for(name) != GatePolicy::Skip
                    && (options.all || gate.default)
            },
            |only| only.iter().any(|g| g == name),
        );
        let skipped = options
            .skip_gates
            .is_some_and(|skip| skip.iter().any(|g| g == name));
        if !selected || skipped {
            continue;
        }

        let is_up_to = options.up_to_gate.is_some_and(|up_to| up_to == name);
        active_gates.push(gate);
        if is_up_to {
            break;
        }
    }
    active_gates
}

/// Selected gates for this run, with `options.extra_args` appended to the
/// single selected gate (FR-14).
///
/// # Errors
/// Returns `GateError::Config` when extra arguments are given and the
/// selection is not exactly one gate.
fn selected_with_passthrough(
    config: &GateConfig,
    config_path: &Path,
    options: &PipelineOptions<'_>,
) -> GateResult<GateList> {
    let selected = select_gates(gate::build_all_gates(config), config, options);
    if !options.extra_args.is_empty() && selected.len() != 1 {
        return Err(GateError::Config {
            path: config_path.to_path_buf(),
            message: format!(
                "argument passthrough needs exactly one selected gate, found {}",
                selected.len()
            ),
        });
    }
    Ok(selected
        .iter()
        .map(|g| with_extra_args(g, options.extra_args))
        .collect())
}

/// Removes every `<gate>.result.json` in `out_dir`.
fn remove_results(out_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(out_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|name| name.ends_with(gate::RESULT_SUFFIX))
        {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Splits gates into the `pre` stage, the concurrent groups (in declaration
/// order) and the `post` stage. A gate scheduled nowhere runs in `post`.
fn partition_gates(
    active_gates: GateList,
    config: &GateConfig,
) -> Schedule<'_> {
    let mut schedule = Schedule::default();
    for name in config.execution.groups.names() {
        schedule.groups.push((name, GateList::new()));
    }
    for gate in active_gates {
        match config.stage_of(gate.name()) {
            Stage::Pre => schedule.pre.push(gate),
            Stage::Post => schedule.post.push(gate),
            Stage::Group(name) => {
                if let Some((_, gates)) =
                    schedule.groups.iter_mut().find(|(group, _)| *group == name)
                {
                    gates.push(gate);
                }
            }
        }
    }
    schedule.groups.retain(|(_, gates)| !gates.is_empty());
    schedule
}

/// Runs groups concurrently on a bounded worker pool (FR-13), each with its
/// own Cargo target directory, so concurrent groups never wait on each other's
/// build-directory lock. Workers take groups in declaration order.
fn run_concurrent_groups(
    groups: &GroupedGates<'_>,
    group_targets: &Path,
    env: &RunEnv<'_>,
    max_jobs: usize,
) -> GateResult<()> {
    if groups.is_empty() {
        return Ok(());
    }
    let worker_count = max_jobs.min(groups.len());
    let group_queue =
        Mutex::new(groups.iter().enumerate().collect::<VecDeque<_>>());
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(s.spawn(|| -> GateResult<()> {
                loop {
                    let next = match group_queue.lock() {
                        Ok(mut q) => q.pop_front(),
                        Err(poisoned) => poisoned.into_inner().pop_front(),
                    };
                    let Some((idx, (group_name, group_gates))) = next else {
                        break;
                    };
                    let target_dir = group_targets.join(group_name);
                    let style = ui::group_style(idx);
                    let group = GroupSpec {
                        name: group_name,
                        style,
                        target_dir: Some(&target_dir),
                    };
                    let run = run_group(&group, group_gates, env)?;
                    let _guard = env.ui_lock.lock();
                    let tag =
                        ui::format_group_tag(Some(group_name), Some(style));
                    ui::status(
                        "Joined",
                        format!("{tag}in {:.2}s", run.duration_secs),
                    );
                }
                Ok(())
            }));
        }
        handles.into_iter().try_for_each(|h| {
            h.join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn gate_with_env(env: HashMap<String, String>) -> Arc<Gate> {
        Arc::new(Gate::new("build", "cargo build", Vec::new()).with_env(env))
    }

    #[test]
    fn group_target_dir_is_injected() {
        let dir = Path::new("/ws/target/ci-groups/lint");
        let scoped = with_target_dir(&gate_with_env(HashMap::new()), dir);
        assert_eq!(
            scoped.env.get("CARGO_TARGET_DIR").map(String::as_str),
            Some("/ws/target/ci-groups/lint")
        );
    }

    #[test]
    fn extra_args_follow_configured_args() {
        let gate = Arc::new(Gate::new(
            "mutants",
            "cargo mutants",
            vec!["--json".to_string()],
        ));
        let extra = ["--jobs".to_string(), "8".to_string()];
        let extended = with_extra_args(&gate, &extra);
        assert_eq!(extended.args, ["--json", "--jobs", "8"]);
        assert_eq!(
            extended.command_display(),
            "`cargo mutants --json --jobs 8`"
        );
        assert!(Arc::ptr_eq(&with_extra_args(&gate, &[]), &gate));
    }

    #[test]
    fn declared_target_dir_is_kept() {
        let env = HashMap::from([(
            "CARGO_TARGET_DIR".to_string(),
            "target/geiger".to_string(),
        )]);
        let scoped = with_target_dir(
            &gate_with_env(env),
            Path::new("/ws/target/ci-groups/audit"),
        );
        assert_eq!(
            scoped.env.get("CARGO_TARGET_DIR").map(String::as_str),
            Some("target/geiger")
        );
    }
}
