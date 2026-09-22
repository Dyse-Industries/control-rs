//! Criterion performance regression evaluator and benchmark budget harness.
//!
//! Evaluates Criterion benchmark outputs against real-time latency budgets
//! (e.g. <= 10 µs jitter for flight control loops) and statistical regression
//! baselines. Always executes benchmarks prior to evaluation unless `--skip-bench`
//! is explicitly provided.

#![allow(
    missing_docs,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct EstimateEntry {
    point_estimate: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct CriterionEstimates {
    mean: Option<EstimateEntry>,
    median: Option<EstimateEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct BenchmarkMeta {
    full_id: Option<String>,
    title: Option<String>,
}

#[derive(Debug, Clone)]
struct BenchmarkEvaluation {
    id: String,
    median_ns: f64,
    base_median_ns: Option<f64>,
    delta_pct: Option<f64>,
    budget_ns: f64,
    budget_passed: bool,
    regression_passed: bool,
    failure_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct CliOptions {
    bench_target: Option<String>,
    run_all: bool,
    skip_bench: bool,
    threshold_pct: f64,
    criterion_dir: Option<PathBuf>,
}

fn parse_cli_args() -> Result<CliOptions, String> {
    let mut args = std::env::args().skip(1);
    let mut bench_target = None;
    let mut run_all = false;
    let mut skip_bench = false;
    let mut threshold_pct = 15.0; // 15% default noise tolerance
    let mut criterion_dir = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--all" => {
                run_all = true;
            }
            "--skip-bench" => {
                skip_bench = true;
            }
            "--bench" => {
                if let Some(target) = args.next() {
                    bench_target = Some(target);
                } else {
                    return Err("Missing argument for --bench".to_string());
                }
            }
            "--threshold" => {
                if let Some(val_str) = args.next() {
                    threshold_pct = val_str.parse::<f64>().map_err(|e| {
                        format!("Invalid float for --threshold: {e}")
                    })?;
                } else {
                    return Err("Missing argument for --threshold".to_string());
                }
            }
            "--criterion-dir" => {
                if let Some(dir_str) = args.next() {
                    criterion_dir = Some(PathBuf::from(dir_str));
                } else {
                    return Err(
                        "Missing argument for --criterion-dir".to_string()
                    );
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => {
                return Err(format!("Unknown option: {unknown}"));
            }
        }
    }

    Ok(CliOptions {
        bench_target,
        run_all,
        skip_bench,
        threshold_pct,
        criterion_dir,
    })
}

fn print_help() {
    println!(
        "Usage: cargo regression [OPTIONS]\n\n\
         Options:\n  \
           --bench <NAME>        Run only the specified benchmark target (e.g. jitter, scaling)\n  \
           --all                 Run all workspace benchmark targets\n  \
           --skip-bench          Skip running cargo bench and analyze existing Criterion artifacts\n  \
           --threshold <PCT>     Maximum acceptable performance regression percentage against baseline (default: 15.0)\n  \
           --criterion-dir <DIR> Custom Criterion output directory (defaults to target/criterion)\n  \
           -h, --help            Print help information"
    );
}

fn find_workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(parent) = manifest_dir.parent()
        && parent.join("Cargo.toml").exists()
    {
        return parent.to_path_buf();
    }
    manifest_dir
}

fn budget_for_benchmark(id: &str) -> f64 {
    match id {
        "jitter/state_space_step_response"
        | "jitter/ekf_8x8_covariance_update" => 10_000.0, // 10 µs
        "jitter/hilbert_32x32_solve" => 100_000.0, // 100 µs
        s if s.starts_with("jitter/") => 50_000.0, // 50 µs
        s if s.starts_with("state_space_scaling/zoh_dim/128") => 100_000_000.0, // 100 ms
        s if s.starts_with("state_space_scaling/") => 50_000_000.0, // 50 ms
        s if s.starts_with("matrix_inversion_scaling/") => 10_000_000.0, // 10 ms
        s if s.starts_with("tensor_contraction_scaling/") => 10_000_000.0, // 10 ms
        s if s.starts_with("polynomial_evaluation_scaling/") => 1_000_000.0, // 1 ms
        _ => 100_000_000.0, // 100 ms default budget
    }
}

fn format_duration(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.2} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000.0)
    }
}

fn run_benchmarks(root: &Path, opts: &CliOptions) -> Result<(), String> {
    if opts.skip_bench {
        println!(
            "Notice: Skipping benchmark execution (--skip-bench specified). Evaluating existing artifacts in target/criterion."
        );
        return Ok(());
    }

    let mut cmd = Command::new("cargo");
    cmd.current_dir(root);
    cmd.arg("bench");

    if opts.run_all {
        println!("Executing all Criterion benchmarks (cargo bench)...");
    } else if let Some(ref target) = opts.bench_target {
        println!(
            "Executing Criterion benchmark target '{target}' (cargo bench --bench {target})..."
        );
        cmd.args(["--bench", target]);
    } else {
        println!(
            "Executing default Criterion benchmark target 'jitter' (cargo bench --bench jitter)..."
        );
        cmd.args(["--bench", "jitter"]);
    }

    let status = cmd
        .status()
        .map_err(|e| format!("Failed to spawn cargo bench: {e}"))?;

    if status.success() {
        println!("Benchmark execution completed successfully.\n");
        Ok(())
    } else {
        Err(format!(
            "cargo bench failed with exit code {:?}",
            status.code()
        ))
    }
}

fn find_estimates_files(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                find_estimates_files(&path, files);
            } else if path.file_name().and_then(|n| n.to_str())
                == Some("estimates.json")
                && let Some(parent) = path.parent()
                && parent.file_name().and_then(|n| n.to_str()) == Some("new")
            {
                files.push(path);
            }
        }
    }
}

fn read_benchmark_id(bench_dir: &Path, root_criterion: &Path) -> String {
    let meta_file = bench_dir.join("new").join("benchmark.json");
    if let Ok(content) = fs::read_to_string(&meta_file)
        && let Ok(meta) = serde_json::from_str::<BenchmarkMeta>(&content)
    {
        if let Some(full_id) = meta.full_id {
            return full_id;
        }
        if let Some(title) = meta.title {
            return title;
        }
    }

    // Fallback: derive ID from relative path
    if let Ok(rel) = bench_dir.strip_prefix(root_criterion) {
        return rel.to_string_lossy().to_string();
    }

    bench_dir.file_name().map_or_else(
        || "unknown".to_string(),
        |s| s.to_string_lossy().to_string(),
    )
}

fn evaluate_benchmark(
    new_estimates_path: &Path,
    root_criterion: &Path,
    threshold_pct: f64,
) -> Result<BenchmarkEvaluation, String> {
    let new_dir = new_estimates_path
        .parent()
        .ok_or_else(|| "Invalid estimates path".to_string())?;
    let bench_dir = new_dir
        .parent()
        .ok_or_else(|| "Invalid benchmark directory".to_string())?;

    let id = read_benchmark_id(bench_dir, root_criterion);

    // Read new estimates
    let new_content = fs::read_to_string(new_estimates_path).map_err(|e| {
        format!("Failed to read {}: {e}", new_estimates_path.display())
    })?;
    let new_est: CriterionEstimates = serde_json::from_str(&new_content)
        .map_err(|e| {
            format!("Failed to parse {}: {e}", new_estimates_path.display())
        })?;

    let median_entry = new_est
        .median
        .or(new_est.mean)
        .ok_or_else(|| format!("No median or mean estimate found in {id}"))?;

    let median_ns = median_entry.point_estimate;

    // Check baseline if present
    let base_file = bench_dir.join("base").join("estimates.json");
    let (base_median_ns, delta_pct, regression_passed, reg_err) = if base_file
        .exists()
    {
        if let Ok(base_content) = fs::read_to_string(&base_file)
            && let Ok(base_est) =
                serde_json::from_str::<CriterionEstimates>(&base_content)
            && let Some(base_entry) = base_est.median.or(base_est.mean)
        {
            let base_median = base_entry.point_estimate;
            let delta = if base_median > 0.0 {
                ((median_ns - base_median) / base_median) * 100.0
            } else {
                0.0
            };

            // Avoid false alarms on sub-microsecond timer jitter: require delta > 50 ns
            let is_reg =
                delta > threshold_pct && (median_ns - base_median) > 50.0;
            if is_reg {
                (
                    Some(base_median),
                    Some(delta),
                    false,
                    Some(format!(
                        "Regression of +{delta:.2}% exceeds threshold of +{threshold_pct:.1}%"
                    )),
                )
            } else {
                (Some(base_median), Some(delta), true, None)
            }
        } else {
            (None, None, true, None)
        }
    } else {
        (None, None, true, None)
    };

    let budget_ns = budget_for_benchmark(&id);
    let budget_passed = median_ns <= budget_ns;
    let budget_err = if budget_passed {
        None
    } else {
        Some(format!(
            "Latency {} exceeded cycle budget of {}",
            format_duration(median_ns),
            format_duration(budget_ns)
        ))
    };

    let failure_reason = match (budget_err, reg_err) {
        (Some(b), Some(r)) => Some(format!("{b}; {r}")),
        (Some(b), None) => Some(b),
        (None, Some(r)) => Some(r),
        (None, None) => None,
    };

    Ok(BenchmarkEvaluation {
        id,
        median_ns,
        base_median_ns,
        delta_pct,
        budget_ns,
        budget_passed,
        regression_passed,
        failure_reason,
    })
}

fn main() {
    println!("=== control-rs Performance Regression & Budget Harness ===");

    let opts = match parse_cli_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Error: {e}\n");
            print_help();
            std::process::exit(1);
        }
    };

    let root = find_workspace_root();
    println!("Workspace root: {}", root.display());

    if let Err(e) = run_benchmarks(&root, &opts) {
        eprintln!("Error executing benchmarks: {e}");
        std::process::exit(1);
    }

    let criterion_dir = opts
        .criterion_dir
        .unwrap_or_else(|| root.join("target/criterion"));

    if !criterion_dir.exists() {
        eprintln!(
            "Error: Criterion directory not found at {}",
            criterion_dir.display()
        );
        std::process::exit(1);
    }

    let mut estimate_files = Vec::new();
    find_estimates_files(&criterion_dir, &mut estimate_files);

    if estimate_files.is_empty() {
        eprintln!(
            "Error: No Criterion benchmark estimates found in {}",
            criterion_dir.display()
        );
        std::process::exit(1);
    }

    let mut evaluations: BTreeMap<String, BenchmarkEvaluation> =
        BTreeMap::new();
    let mut parse_errors = Vec::new();

    for path in estimate_files {
        match evaluate_benchmark(&path, &criterion_dir, opts.threshold_pct) {
            Ok(eval) => {
                evaluations.insert(eval.id.clone(), eval);
            }
            Err(e) => {
                parse_errors.push(e);
            }
        }
    }

    if !parse_errors.is_empty() {
        eprintln!("Encountered errors parsing benchmark artifacts:");
        for err in &parse_errors {
            eprintln!("  - {err}");
        }
        std::process::exit(1);
    }

    println!("--- Benchmark Performance & Regression Matrix ---");
    println!(
        "{:<42} {:>14} {:>14} {:>10} {:>14} {:>8}",
        "Benchmark Identifier",
        "Median Latency",
        "Baseline",
        "Change",
        "Cycle Budget",
        "Status"
    );
    println!("{}", "-".repeat(108));

    let mut failure_count = 0;

    for eval in evaluations.values() {
        let median_str = format_duration(eval.median_ns);
        let base_str = eval
            .base_median_ns
            .map_or_else(|| "N/A (base)".to_string(), format_duration);
        let delta_str = eval.delta_pct.map_or_else(
            || "-".to_string(),
            |d| {
                if d >= 0.0 {
                    format!("+{d:.2}%")
                } else {
                    format!("{d:.2}%")
                }
            },
        );
        let budget_str = format_duration(eval.budget_ns);

        let status = if eval.budget_passed && eval.regression_passed {
            "PASS"
        } else {
            failure_count += 1;
            "FAIL"
        };

        println!(
            "{:<42} {:>14} {:>14} {:>10} {:>14} {:>8}",
            eval.id, median_str, base_str, delta_str, budget_str, status
        );
    }

    println!("{}", "-".repeat(108));

    if failure_count > 0 {
        eprintln!(
            "\nPerformance verification FAILED: {failure_count}/{} benchmark(s) breached latency budgets or regression tolerances.",
            evaluations.len()
        );
        for eval in evaluations.values() {
            if let Some(ref reason) = eval.failure_reason {
                eprintln!("  - '{}': {reason}", eval.id);
            }
        }
        std::process::exit(1);
    }

    println!(
        "\nPerformance verification PASSED: All {} benchmark(s) satisfied latency budgets and regression thresholds.",
        evaluations.len()
    );
    std::process::exit(0);
}
