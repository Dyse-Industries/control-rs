//! Integration test running the full cross-comparison suite during `cargo test --workspace`.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::time::Duration;

use control_rs_compare::compare::{ComparatorOptions, run_comparison};
use control_rs_compare::config::CompareConfigFile;
use control_rs_compare::runner::{RunnerOptions, execute_master_plan};

#[test]
fn test_workspace_cross_compare() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from);

    let config_path = workspace_root.join("compare.toml");
    if !config_path.exists() {
        eprintln!(
            "Warning: compare.toml not found at workspace root, skipping test"
        );
        return;
    }

    let config_file = CompareConfigFile::load_from_file(&config_path)
        .expect("Failed to load compare.toml");
    let master_plan = config_file
        .resolve_master_plan(&config_path)
        .expect("Failed to resolve master plan from compare.toml");

    let results_dir = workspace_root.join(&master_plan.general.out_dir);

    let runner_opts = RunnerOptions {
        workspace_root,
        out_dir: results_dir.clone(),
        timeout: Duration::from_secs(master_plan.general.timeout_secs),
        quiet: true,
    };

    // 1. Execute variants
    execute_master_plan(&master_plan, &runner_opts, &[])
        .expect("Failed executing cross-compare master plan variants");

    // 2. Run numerical comparisons
    let comparator_opts = ComparatorOptions {
        results_dir,
        suite_filter: None,
        oracle_override: None,
        signals: None,
        strict: true,
        quiet: true,
        num_threads: None,
    };

    let report = run_comparison(Some(&master_plan), &comparator_opts)
        .expect("Failed running HDF5 dataset comparison");

    assert_eq!(
        report.summary.verdict, "Pass",
        "Cross-compare verification failed: {:#?}",
        report.summary
    );
}
