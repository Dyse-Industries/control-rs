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
pub mod gates;
pub mod quality_gate;
pub mod report;
pub mod ui;

pub use cli::run_cli;
pub use config::{GateConfig, GatePolicy};
pub use error::GateError;
pub use quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};
pub use report::ReportAggregator;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn execute_gate(
    gate: &Arc<dyn QualityGate>,
    ctx: &GateContext,
    ui_lock: &Mutex<()>,
) {
    let name = gate.name();
    {
        let _guard = ui_lock.lock();
        ui::status("Running", gate.command_display());
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
                    format!("{name} in {:.2}s", outcome.duration_secs),
                ),
                Verdict::Warn => ui::warning(
                    "Warning",
                    format!("{name} in {:.2}s{summary}", outcome.duration_secs),
                ),
                Verdict::Fail => ui::failure(
                    "Failed",
                    format!("{name} in {:.2}s{summary}", outcome.duration_secs),
                ),
                Verdict::Skipped => ui::warning("Skipped", name.to_string()),
            }
        }
        Err(e) => {
            let _guard = ui_lock.lock();
            ui::error(format!("Error executing gate '{name}': {e}"));
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
///
/// # Errors
/// Returns `GateError` if configuration loading, gate execution, or report rendering fails.
pub fn run_pipeline(
    workspace_root: &Path,
    config_path: &Path,
    only_gates: Option<&[String]>,
    skip_gates: Option<&[String]>,
    up_to_gate: Option<&str>,
) -> Result<bool, GateError> {
    let pipeline_start = Instant::now();
    let config = GateConfig::load_from_path(config_path)?;
    let out_dir = workspace_root.join(&config.runner.out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let ctx = GateContext {
        workspace_root: workspace_root.to_path_buf(),
        out_dir: out_dir.clone(),
        default_timeout: Duration::from_secs(config.runner.timeout_secs),
    };

    let all_gates = gates::build_all_gates(&config);
    let mut active_gates = Vec::new();
    let mut executed_gate_names = Vec::new();

    for gate in all_gates {
        let name = gate.name().to_string();

        if let Some(only) = only_gates {
            if !only.iter().any(|g| g == &name) {
                continue;
            }
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

    let (exclusive, concurrent): (Vec<_>, Vec<_>) =
        active_gates.into_iter().partition(|g| {
            config
                .execution
                .exclusive_gates
                .iter()
                .any(|e| e == g.name())
        });

    let ui_lock = Mutex::new(());

    if config.execution.parallel {
        let mut lanes: BTreeMap<&str, Vec<Arc<dyn QualityGate>>> =
            BTreeMap::new();
        for gate in concurrent {
            let lane_name = config
                .execution
                .lanes
                .iter()
                .find(|(_, members)| members.iter().any(|m| m == gate.name()))
                .map_or("cargo", |(lane, _)| lane.as_str());
            lanes.entry(lane_name).or_default().push(gate);
        }

        std::thread::scope(|s| {
            for (_lane, lane_gates) in lanes {
                let ctx_ref = &ctx;
                let lock_ref = &ui_lock;
                s.spawn(move || {
                    for g in lane_gates {
                        execute_gate(&g, ctx_ref, lock_ref);
                    }
                });
            }
        });
    } else {
        for gate in concurrent {
            execute_gate(&gate, &ctx, &ui_lock);
        }
    }

    for gate in exclusive {
        execute_gate(&gate, &ctx, &ui_lock);
    }

    let is_partial =
        only_gates.is_some() || skip_gates.is_some() || up_to_gate.is_some();
    let subset_filter = if is_partial {
        Some(executed_gate_names.as_slice())
    } else {
        None
    };

    let aggregator =
        ReportAggregator::new(out_dir, workspace_root.to_path_buf());
    let (is_pass, report_path) =
        aggregator.write_report(&config, subset_filter)?;
    let total_duration = pipeline_start.elapsed().as_secs_f64();

    ui::status("Writing", format!("{}", report_path.display()));
    if is_pass {
        ui::status(
            "Finished",
            format!("ci-pipeline passed in {:.2}s", total_duration),
        );
    } else {
        ui::error(format!(
            "ci-pipeline failed in {:.2}s (see report for details)",
            total_duration
        ));
    }

    Ok(is_pass)
}
