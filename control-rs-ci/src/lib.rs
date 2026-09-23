//! # `control-rs-ci`
//!
//! Continuous Integration & Quality Gate Infrastructure for `control-rs`.
//!
//! Provides a modular quality gate execution engine, standardized process outcome
//! records, decentralized artifact aggregation, and automated Markdown report rendering.

#![deny(missing_docs)]

pub use cli::run_cli;
pub use config::{GateConfig, GatePolicy};
pub use error::{GateError, GateResult};
pub use gate::{Gate, GateContext, GateList, GateOutcome, SharedGate, Verdict};
pub use report::{ReportAggregator, WrittenReport};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub mod allow_audit;
pub mod cli;

pub mod config;

pub mod error;

pub mod gate;

pub mod report;

pub mod ui;

/// Workspace-relative directory that holds one Cargo target directory per
/// execution group (`target/ci-groups/<group>`).
///
/// Concurrent groups that share one target directory serialize on Cargo's
/// build-directory lock. A dedicated directory per group removes that
/// contention. `--clean` deletes this directory, so every group rebuilds from
/// an empty target directory.
pub const GROUP_TARGET_ROOT: &str = "target/ci-groups";

/// Concurrent groups keyed by name, each with its gates in pipeline order.
type GroupedGates<'a> = BTreeMap<&'a str, GateList>;

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
    /// Group name printed in the tag (or `"exclusive"`).
    pub name: &'a str,
    /// Tag color.
    pub style: anstyle::Style,
    /// Dedicated `CARGO_TARGET_DIR`, or `None` for the shared one.
    pub target_dir: Option<&'a Path>,
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
    /// Clean artifacts before running.
    pub clean: bool,
    /// Echo gate output.
    pub verbose: bool,
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

fn execute_gate(gate: &Gate, tag: &str, env: &RunEnv<'_>) {
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
        }
        Err(e) => {
            let _guard = env.ui_lock.lock();
            ui::error(format!("{tag}Error executing gate '{name}': {e}"));
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

/// Runs `gates` one at a time on the calling thread and returns the elapsed
/// wall-clock time in seconds.
///
/// With `group.target_dir`, the directory is created if missing and every
/// gate runs with `CARGO_TARGET_DIR` set to it, except a gate that declares
/// its own `CARGO_TARGET_DIR` in `gate.toml`. Without it, gates inherit the
/// caller's environment and build in the shared target directory.
///
/// The scheduler spawns one thread per group and calls this function on each.
///
/// # Errors
/// Returns `GateError::Io` if the target directory cannot be created.
pub fn run_group(
    group: &GroupSpec<'_>,
    gates: &[SharedGate],
    env: &RunEnv<'_>,
) -> GateResult<f64> {
    let start = Instant::now();
    if let Some(dir) = group.target_dir {
        std::fs::create_dir_all(dir)?;
    }
    let tag = ui::format_group_tag(Some(group.name), Some(group.style));
    for gate in gates {
        let gate = group
            .target_dir
            .map_or_else(|| Arc::clone(gate), |dir| with_target_dir(gate, dir));
        execute_gate(&gate, &tag, env);
    }
    Ok(start.elapsed().as_secs_f64())
}

/// Runs the complete CI quality gate pipeline or filtered subset.
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

    let active_gates =
        select_gates(gate::build_all_gates(&config)?, &config, options);
    let executed_gate_names: Vec<String> =
        active_gates.iter().map(|g| g.name().to_string()).collect();
    remove_stale_results(&out_dir, &executed_gate_names);

    let (groups, exclusive) = partition_gates(active_gates, &config);
    let ui_lock = Mutex::new(());
    let env = RunEnv {
        ctx: &ctx,
        ui_lock: &ui_lock,
        verbose: options.verbose,
    };

    run_concurrent_groups(
        &groups,
        &workspace_root.join(GROUP_TARGET_ROOT),
        &env,
    )?;

    // Barrier passed: exclusive gates run on the calling thread, one at a time,
    // in the shared target directory.
    let exclusive_group = GroupSpec {
        name: "exclusive",
        style: ui::exclusive_style(),
        target_dir: None,
    };
    run_group(&exclusive_group, &exclusive, &env)?;

    let aggregator =
        ReportAggregator::new(out_dir, workspace_root.to_path_buf());
    let report = aggregator
        .write_report(&config, Some(executed_gate_names.as_slice()))?;
    let total_duration = pipeline_start.elapsed().as_secs_f64();

    ui::status("Writing", format!("{}", report.path.display()));
    if report.pass {
        ui::status("Finished", format!("ci in {total_duration:.2}s"));
    } else {
        ui::failure(
            "Failed",
            format!("ci in {total_duration:.2}s (see report for details)"),
        );
    }

    Ok(report.pass)
}

/// Applies the `only`/`skip`/`up_to` filters and `skip` policies to the
/// configured gates, preserving pipeline order.
fn select_gates(
    all_gates: GateList,
    config: &GateConfig,
    options: &PipelineOptions<'_>,
) -> GateList {
    let mut active_gates = GateList::new();
    for gate in all_gates {
        let name = gate.name();
        let selected = options.only_gates.map_or_else(
            || config.policy_for(name) != GatePolicy::Skip,
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

/// Removes result artifacts of gates that will not run this time.
fn remove_stale_results(out_dir: &Path, executed_gate_names: &[String]) {
    let Ok(entries) = std::fs::read_dir(out_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && let Some(gate_name) = name.strip_suffix(gate::RESULT_SUFFIX)
            && !executed_gate_names.iter().any(|g| g == gate_name)
        {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Splits gates into per-group lists and the exclusive list.
///
/// A gate is exclusive if explicitly declared in `exclusive_gates` OR if it is
/// unassigned to any group in `execution.groups`. All unassigned gates safely
/// default to sequential execution with full processor authority.
fn partition_gates(
    active_gates: GateList,
    config: &GateConfig,
) -> (GroupedGates<'_>, GateList) {
    let groups_cfg = &config.execution.groups;
    let mut groups = GroupedGates::new();
    let mut exclusive = GateList::new();
    for gate in active_gates {
        let declared_exclusive = config
            .execution
            .exclusive_gates
            .iter()
            .any(|e| e == gate.name());
        let group = groups_cfg
            .iter()
            .find(|(_, members)| members.iter().any(|m| m == gate.name()));
        match group {
            Some((name, _)) if !declared_exclusive => {
                groups.entry(name.as_str()).or_default().push(gate);
            }
            _ => exclusive.push(gate),
        }
    }
    (groups, exclusive)
}

/// Runs every group on its own thread with its own Cargo target directory,
/// so concurrent groups never wait on each other's build-directory lock.
fn run_concurrent_groups(
    groups: &GroupedGates<'_>,
    group_targets: &Path,
    env: &RunEnv<'_>,
) -> GateResult<()> {
    std::thread::scope(|s| {
        // Spawn every group before joining any, so the groups overlap.
        let mut handles = Vec::with_capacity(groups.len());
        for (idx, (group_name, group_gates)) in groups.iter().enumerate() {
            let target_dir = group_targets.join(group_name);
            let style = ui::group_style(idx);
            handles.push(s.spawn(move || {
                let group = GroupSpec {
                    name: group_name,
                    style,
                    target_dir: Some(&target_dir),
                };
                let duration = run_group(&group, group_gates, env)?;
                let _guard = env.ui_lock.lock();
                let tag = ui::format_group_tag(Some(group_name), Some(style));
                ui::status("Joined", format!("{tag}in {duration:.2}s"));
                Ok::<(), GateError>(())
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
