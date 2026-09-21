//! # `control-rs-ci`
//!
//! Continuous Integration & Quality Gate Infrastructure for `control-rs`.
//!
//! Provides a modular quality gate execution engine, standardized process outcome
//! records, decentralized artifact aggregation, and automated Markdown report rendering.

#![deny(missing_docs)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::shadow_unrelated,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::wildcard_imports,
    clippy::similar_names,
    clippy::cognitive_complexity,
    clippy::match_wildcard_for_single_variants,
    clippy::type_complexity,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::arbitrary_source_item_ordering,
    clippy::multiple_crate_versions,
    clippy::equatable_if_let,
    clippy::nursery,
    clippy::cargo,
    clippy::collapsible_if,
    clippy::single_match,
    clippy::format_push_string,
    clippy::map_unwrap_or,
    clippy::if_not_else,
    clippy::unreadable_literal,
    clippy::redundant_closure_for_method_calls,
    clippy::single_match_else,
    clippy::items_after_statements,
    clippy::too_many_arguments,
    clippy::unused_self,
    clippy::unnecessary_wraps,
    clippy::unnecessary_literal_bound,
    clippy::doc_markdown,
    clippy::bool_to_int_with_if,
    clippy::branches_sharing_code,
    clippy::or_fun_call
)]

pub mod cli;
pub mod config;
pub mod error;
pub mod gate;
pub mod report;
pub mod ui;

pub use cli::run_cli;
pub use config::{GateConfig, GatePolicy};
pub use error::GateError;
pub use gate::{Gate, GateContext, GateOutcome, Verdict};
pub use report::ReportAggregator;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Cleans up the CI artifacts directory and any leftover workspace root artifacts.
///
/// Removes the configured output directory (`out_dir`, typically `target/ci-artifacts`)
/// and removes any stray legacy CI artifacts found in the workspace root.
///
/// # Errors
/// Returns `GateError` if configuration loading or directory deletion fails.
pub fn clean_artifacts(
    workspace_root: &Path,
    config_path: &Path,
) -> Result<PathBuf, GateError> {
    let config = GateConfig::load_from_path(config_path)?;
    let out_dir = workspace_root.join(&config.runner.out_dir);

    if out_dir.exists() {
        std::fs::remove_dir_all(&out_dir)?;
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

fn execute_gate(
    gate: &Arc<Gate>,
    group_id: Option<&str>,
    style: Option<anstyle::Style>,
    ctx: &GateContext,
    ui_lock: &Mutex<()>,
) {
    let name = gate.name();
    let tag = ui::format_group_tag(group_id, style);
    {
        let _guard = ui_lock.lock();
        ui::status("Running", format!("{tag}{}", gate.command_display()));
    }
    match gate.execute(ctx) {
        Ok(outcome) => {
            let _guard = ui_lock.lock();
            let summary = outcome
                .summary
                .as_ref()
                .map_or(String::new(), |s| format!(" ({s})"));
            match outcome.verdict {
                Verdict::Pass => ui::status(
                    "Passed",
                    format!("{tag}{name} in {:.2}s", outcome.duration_secs),
                ),
                Verdict::Warn => ui::warning(
                    "Warning",
                    format!(
                        "{tag}{name} in {:.2}s{summary}",
                        outcome.duration_secs
                    ),
                ),
                Verdict::Fail => ui::failure(
                    "Failed",
                    format!(
                        "{tag}{name} in {:.2}s{summary}",
                        outcome.duration_secs
                    ),
                ),
                Verdict::Skipped => {
                    ui::warning("Skipped", format!("{tag}{name}"));
                }
            }
        }
        Err(e) => {
            let _guard = ui_lock.lock();
            ui::error(format!("{tag}Error executing gate '{name}': {e}"));
        }
    }
}

/// Runs the complete CI quality gate pipeline or filtered subset.
///
/// # Arguments
/// * `workspace_root` - Root directory of the Cargo workspace.
/// * `config_path` - Path to `gate.toml`.
/// * `only_gates` - Optional whitelist of gates to execute.
/// * `skip_gates` - Optional blacklist of gates to skip.
/// * `up_to_gate` - Optional gate name to stop execution after.
/// * `clean` - Whether to clean artifacts directory before running.
///
/// # Errors
/// Returns `GateError` if configuration loading, gate execution, or report rendering fails.
pub fn run_pipeline(
    workspace_root: &Path,
    config_path: &Path,
    only_gates: Option<&[String]>,
    skip_gates: Option<&[String]>,
    up_to_gate: Option<&str>,
    clean: bool,
) -> Result<bool, GateError> {
    let pipeline_start = Instant::now();
    if clean {
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

    let all_gates = gate::build_all_gates(&config)?;
    let mut active_gates = Vec::new();
    let mut executed_gate_names = Vec::new();

    for gate in all_gates {
        let name = gate.name().to_string();

        if let Some(only) = only_gates {
            if !only.iter().any(|g| g == &name) {
                continue;
            }
        } else if config.policy_for(&name) == GatePolicy::Skip {
            continue;
        }

        if let Some(skip) = skip_gates {
            if skip.iter().any(|g| g == &name) {
                continue;
            }
        }

        let is_up_to = up_to_gate.is_some_and(|up_to| up_to == name);
        executed_gate_names.push(name);
        active_gates.push(gate);

        if is_up_to {
            break;
        }
    }

    // Remove stale results for inactive or skipped gates
    if let Ok(entries) = std::fs::read_dir(&out_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".result.json") {
                    let gate_name = name.trim_end_matches(".result.json");
                    if !executed_gate_names.iter().any(|g| g == gate_name) {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
        }
    }

    // A gate is exclusive if explicitly declared in `exclusive_gates` OR if it is
    // unassigned to any group in `execution.groups`. All unassigned gates safely
    // default to sequential execution with full processor authority.
    let (exclusive, concurrent): (Vec<_>, Vec<_>) =
        active_gates.into_iter().partition(|g| {
            config
                .execution
                .exclusive_gates
                .iter()
                .any(|e| e == g.name())
                || !config
                    .execution
                    .groups
                    .values()
                    .any(|members| members.iter().any(|m| m == g.name()))
        });

    let ui_lock = Mutex::new(());

    if config.execution.parallel {
        let mut groups: BTreeMap<&str, Vec<Arc<Gate>>> = BTreeMap::new();

        for gate in concurrent {
            if let Some((group, _)) =
                config.execution.groups.iter().find(|(_, members)| {
                    members.iter().any(|m| m == gate.name())
                })
            {
                groups.entry(group.as_str()).or_default().push(gate);
            }
        }

        std::thread::scope(|s| {
            for (idx, (group_name, group_gates)) in
                groups.into_iter().enumerate()
            {
                let ctx_ref = &ctx;
                let lock_ref = &ui_lock;
                let style = ui::group_style(idx);
                s.spawn(move || {
                    let group_start = Instant::now();
                    for g in &group_gates {
                        execute_gate(
                            g,
                            Some(group_name),
                            Some(style),
                            ctx_ref,
                            lock_ref,
                        );
                    }
                    let duration = group_start.elapsed().as_secs_f64();
                    let _guard = lock_ref.lock();
                    let tag =
                        ui::format_group_tag(Some(group_name), Some(style));
                    ui::status("Joined", format!("{tag}in {duration:.2}s"));
                });
            }
        });
    } else {
        for gate in concurrent {
            execute_gate(&gate, None, None, &ctx, &ui_lock);
        }
    }

    for gate in exclusive {
        let style = ui::exclusive_style();
        execute_gate(&gate, Some("exclusive"), Some(style), &ctx, &ui_lock);
    }

    let aggregator =
        ReportAggregator::new(out_dir, workspace_root.to_path_buf());
    let (is_pass, report_path) = aggregator
        .write_report(&config, Some(executed_gate_names.as_slice()))?;
    let total_duration = pipeline_start.elapsed().as_secs_f64();

    ui::status("Writing", format!("{}", report_path.display()));
    if is_pass {
        ui::status("Finished", format!("ci in {:.2}s", total_duration));
    } else {
        ui::failure(
            "Failed",
            format!("ci in {:.2}s (see report for details)", total_duration),
        );
    }

    Ok(is_pass)
}
