//! HDF5 typed dataset inspection and numerical comparison engine.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hdf5_pure::File;

use crate::config::MasterPlan;
use crate::error::HarnessError;
use crate::report::{
    ComparisonFinding, MethodFinding, SuiteReport, ValidationReport,
    ValidationSummary,
};

/// High-level execution options for the comparison engine.
#[derive(Debug, Clone)]
pub struct ComparatorOptions {
    /// Directory containing `.h5` result containers.
    pub results_dir: PathBuf,

    /// Optional suite filter (runs comparison only for named suites).
    pub suite_filter: Option<BTreeSet<String>>,

    /// True oracle override (e.g. "scipy").
    pub oracle_override: Option<String>,

    /// If true, discrepancy triggers a non-zero exit return.
    pub strict: bool,

    /// If true, suppresses stdout streaming messages.
    pub quiet: bool,
}

impl Default for ComparatorOptions {
    fn default() -> Self {
        Self {
            results_dir: PathBuf::from("results"),
            suite_filter: None,
            oracle_override: None,
            strict: true,
            quiet: false,
        }
    }
}

/// Tolerance bound configuration for a numerical comparison check.
#[derive(Debug, Clone, PartialEq)]
pub struct ToleranceSpec {
    /// Method name ("abs", "rel", "rms", "matrix_norm", "interval", "envelope", "exact_match").
    pub method: String,
    /// Numerical threshold or tolerance bound.
    pub bound: f64,
}

impl Default for ToleranceSpec {
    fn default() -> Self {
        Self {
            method: "abs".to_string(),
            bound: 1e-4,
        }
    }
}

/// Executes comparison across all ingested HDF5 containers in the results directory.
///
/// # Errors
/// Returns `HarnessError` if directory scanning or file ingestion fails.
pub fn run_comparison(
    plan: Option<&MasterPlan>,
    options: &ComparatorOptions,
) -> Result<ValidationReport, HarnessError> {
    let start_time = Instant::now();

    if !options.results_dir.exists() {
        return Err(HarnessError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Results directory '{}' does not exist",
                options.results_dir.display()
            ),
        )));
    }

    // 1. Ingest all `.h5` files in results_dir: map suite -> (variant -> path)
    let mut suite_files: BTreeMap<String, BTreeMap<String, PathBuf>> =
        BTreeMap::new();
    let entries = fs::read_dir(&options.results_dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| ext == "h5" || ext == "hdf5")
        {
            let (suite, variant) = parse_suite_and_variant(&path);
            suite_files.entry(suite).or_default().insert(variant, path);
        }
    }

    let mut suite_reports = Vec::new();
    let mut total_passed = 0_usize;
    let mut total_failed = 0_usize;

    // 2. Process each discovered suite
    for (suite_name, variants) in &suite_files {
        if let Some(filter) = &options.suite_filter {
            if !filter.contains(suite_name) {
                continue;
            }
        }

        let suite_start = Instant::now();
        let configured_suite =
            plan.and_then(|p| p.suites.iter().find(|s| s.name == *suite_name));

        let default_oracle_name = "scipy".to_string();
        let oracle_name = options
            .oracle_override
            .as_ref()
            .or_else(|| configured_suite.map(|s| &s.true_oracle))
            .unwrap_or(&default_oracle_name);

        let oracle_path = variants
            .get(oracle_name)
            .or_else(|| variants.get("scipy").or_else(|| variants.get("rust")));

        let Some(oracle_p) = oracle_path else {
            let suite_duration = suite_start.elapsed().as_secs_f64();
            total_failed = total_failed.saturating_add(1);
            suite_reports.push(SuiteReport {
                name: suite_name.clone(),
                status: "Fail".to_string(),
                duration_secs: suite_duration,
                comparisons: vec![],
                failure_reason: Some(format!(
                    "Reference oracle '{oracle_name}' container missing for suite '{suite_name}'"
                )),
            });
            continue;
        };

        let (actual_oracle_suite, actual_oracle_variant) =
            parse_suite_and_variant(oracle_p);
        let oracle_file = match File::open(oracle_p) {
            Ok(f) => f,
            Err(e) => {
                let suite_duration = suite_start.elapsed().as_secs_f64();
                total_failed = total_failed.saturating_add(1);
                suite_reports.push(SuiteReport {
                    name: suite_name.clone(),
                    status: "Fail".to_string(),
                    duration_secs: suite_duration,
                    comparisons: vec![],
                    failure_reason: Some(format!(
                        "Failed to open oracle container '{}': {e}",
                        oracle_p.display()
                    )),
                });
                continue;
            }
        };

        let mut suite_comparisons = Vec::new();
        let mut suite_passed = true;

        for (peer_variant, peer_path) in variants {
            if peer_variant == &actual_oracle_variant {
                continue;
            }

            let peer_file = match File::open(peer_path) {
                Ok(f) => f,
                Err(e) => {
                    suite_passed = false;
                    suite_comparisons.push(ComparisonFinding {
                        key: format!("{suite_name}.open.{peer_variant}"),
                        pair: (
                            actual_oracle_variant.clone(),
                            peer_variant.clone(),
                        ),
                        signal: "(container)".to_string(),
                        policy: "all_of".to_string(),
                        verdict: "fail".to_string(),
                        methods: vec![MethodFinding {
                            r#type: "container_open_error".to_string(),
                            bound: 0.0,
                            observed: f64::INFINITY,
                            verdict: "fail".to_string(),
                            details: Some(format!(
                                "Failed to open peer container '{}': {e}",
                                peer_path.display()
                            )),
                        }],
                    });
                    continue;
                }
            };

            // Discover datasets by reading root and child groups
            let signals = discover_datasets(&oracle_file);

            for signal in signals {
                let comparison_key =
                    format!("{actual_oracle_suite}.{signal}.{peer_variant}");

                let res = compare_dataset(
                    &oracle_file,
                    &peer_file,
                    &signal,
                    &ToleranceSpec::default(),
                );

                if res.verdict == "fail" {
                    suite_passed = false;
                }

                suite_comparisons.push(ComparisonFinding {
                    key: comparison_key,
                    pair: (actual_oracle_variant.clone(), peer_variant.clone()),
                    signal,
                    policy: "all_of".to_string(),
                    verdict: res.verdict.clone(),
                    methods: vec![res],
                });
            }
        }

        let suite_duration = suite_start.elapsed().as_secs_f64();
        if suite_passed {
            total_passed = total_passed.saturating_add(1);
        } else {
            total_failed = total_failed.saturating_add(1);
        }

        suite_reports.push(SuiteReport {
            name: suite_name.clone(),
            status: if suite_passed { "Pass" } else { "Fail" }.to_string(),
            duration_secs: suite_duration,
            comparisons: suite_comparisons,
            failure_reason: None,
        });
    }

    let total_duration = start_time.elapsed().as_secs_f64();
    let overall_verdict = if total_failed == 0 { "Pass" } else { "Fail" };

    Ok(ValidationReport {
        summary: ValidationSummary {
            total_suites: total_passed.saturating_add(total_failed),
            passed_suites: total_passed,
            failed_suites: total_failed,
            total_duration_secs: total_duration,
            verdict: overall_verdict.to_string(),
        },
        suites: suite_reports,
    })
}

/// Compares a single dataset between an oracle and peer HDF5 file.
#[must_use]
pub fn compare_dataset(
    oracle_file: &File,
    peer_file: &File,
    signal: &str,
    tol: &ToleranceSpec,
) -> MethodFinding {
    let o_ds = match oracle_file.dataset(signal) {
        Ok(d) => d,
        Err(e) => {
            return MethodFinding {
                r#type: tol.method.clone(),
                bound: tol.bound,
                observed: f64::INFINITY,
                verdict: "fail".to_string(),
                details: Some(format!("Signal '{signal}' not in oracle: {e}")),
            };
        }
    };

    let p_ds = match peer_file.dataset(signal) {
        Ok(d) => d,
        Err(e) => {
            return MethodFinding {
                r#type: tol.method.clone(),
                bound: tol.bound,
                observed: f64::INFINITY,
                verdict: "fail".to_string(),
                details: Some(format!(
                    "Signal '{signal}' missing in peer: {e}"
                )),
            };
        }
    };

    // Check dimensions
    let o_shape = o_ds.shape().ok();
    let p_shape = p_ds.shape().ok();
    if o_shape != p_shape {
        return MethodFinding {
            r#type: "shape_mismatch".to_string(),
            bound: 0.0,
            observed: f64::INFINITY,
            verdict: "fail".to_string(),
            details: Some(format!(
                "Shape mismatch for '{signal}': oracle={o_shape:?}, peer={p_shape:?}"
            )),
        };
    }

    // Try reading numeric f64 data
    if let Ok(o_data) = o_ds.read_f64() {
        let p_data = match p_ds.read_f64() {
            Ok(d) => d,
            Err(e) => {
                return MethodFinding {
                    r#type: tol.method.clone(),
                    bound: tol.bound,
                    observed: f64::INFINITY,
                    verdict: "fail".to_string(),
                    details: Some(format!(
                        "Failed to read float data from peer '{signal}': {e}"
                    )),
                };
            }
        };

        return compare_float_arrays(&o_data, &p_data, tol);
    }

    // Try reading string data
    if let Ok(o_strings) = o_ds.read_string() {
        let p_strings = match p_ds.read_string() {
            Ok(s) => s,
            Err(e) => {
                return MethodFinding {
                    r#type: "exact_match".to_string(),
                    bound: 0.0,
                    observed: 0.0,
                    verdict: "fail".to_string(),
                    details: Some(format!(
                        "Failed to read string data from peer '{signal}': {e}"
                    )),
                };
            }
        };

        let passed = o_strings == p_strings;
        return MethodFinding {
            r#type: "exact_match".to_string(),
            bound: 0.0,
            observed: if passed { 1.0 } else { 0.0 },
            verdict: if passed { "pass" } else { "fail" }.to_string(),
            details: if passed {
                None
            } else {
                Some("String dataset mismatch".to_string())
            },
        };
    }

    MethodFinding {
        r#type: "unsupported_type".to_string(),
        bound: 0.0,
        observed: f64::INFINITY,
        verdict: "fail".to_string(),
        details: Some(format!("Unsupported dataset type for '{signal}'")),
    }
}

/// Compares two floating-point vectors according to a tolerance specification.
#[must_use]
pub fn compare_float_arrays(
    oracle: &[f64],
    peer: &[f64],
    tol: &ToleranceSpec,
) -> MethodFinding {
    if oracle.len() != peer.len() {
        return MethodFinding {
            r#type: tol.method.clone(),
            bound: tol.bound,
            observed: f64::INFINITY,
            verdict: "fail".to_string(),
            details: Some(format!(
                "Length mismatch: oracle={}, peer={}",
                oracle.len(),
                peer.len()
            )),
        };
    }

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for (&o, &p) in oracle.iter().zip(peer.iter()) {
        if o.is_nan() || p.is_nan() || o.is_infinite() || p.is_infinite() {
            return MethodFinding {
                r#type: tol.method.clone(),
                bound: tol.bound,
                observed: f64::INFINITY,
                verdict: "fail".to_string(),
                details: Some(
                    "NaN or Infinite floating-point value detected".to_string(),
                ),
            };
        }

        let diff = (o - p).abs();
        if diff > max_abs {
            max_abs = diff;
        }

        let denom = o.abs() + f64::EPSILON;
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }

        sum_sq += diff * diff;
    }

    #[allow(clippy::cast_precision_loss)]
    let count = oracle.len() as f64;
    let rms = if count > 0.0 {
        (sum_sq / count).sqrt()
    } else {
        0.0
    };
    let frobenius_norm = sum_sq.sqrt();

    match tol.method.as_str() {
        "rel" => {
            let passed = max_rel <= tol.bound;
            MethodFinding {
                r#type: "rel".to_string(),
                bound: tol.bound,
                observed: max_rel,
                verdict: if passed { "pass" } else { "fail" }.to_string(),
                details: if passed {
                    None
                } else {
                    Some(format!(
                        "Max relative error {max_rel:.3e} > bound {:.3e}",
                        tol.bound
                    ))
                },
            }
        }
        "rms" => {
            let passed = rms <= tol.bound;
            MethodFinding {
                r#type: "rms".to_string(),
                bound: tol.bound,
                observed: rms,
                verdict: if passed { "pass" } else { "fail" }.to_string(),
                details: if passed {
                    None
                } else {
                    Some(format!(
                        "RMS error {rms:.3e} > bound {:.3e}",
                        tol.bound
                    ))
                },
            }
        }
        "matrix_norm" | "norm" => {
            let passed = frobenius_norm <= tol.bound;
            MethodFinding {
                r#type: "matrix_norm".to_string(),
                bound: tol.bound,
                observed: frobenius_norm,
                verdict: if passed { "pass" } else { "fail" }.to_string(),
                details: if passed {
                    None
                } else {
                    Some(format!(
                        "Frobenius norm {frobenius_norm:.3e} > bound {:.3e}",
                        tol.bound
                    ))
                },
            }
        }
        _ => {
            // Default: abs
            let passed = max_abs <= tol.bound;
            MethodFinding {
                r#type: "abs".to_string(),
                bound: tol.bound,
                observed: max_abs,
                verdict: if passed { "pass" } else { "fail" }.to_string(),
                details: if passed {
                    None
                } else {
                    Some(format!(
                        "Max absolute difference {max_abs:.3e} > bound {:.3e}",
                        tol.bound
                    ))
                },
            }
        }
    }
}

fn parse_suite_and_variant(path: &Path) -> (String, String) {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown.unknown");

    let parts: Vec<&str> = stem.split('.').collect();
    if parts.len() >= 2 {
        let suite = parts.first().unwrap_or(&"unknown").to_string();
        let variant = parts.get(1).unwrap_or(&"unknown").to_string();
        (suite, variant)
    } else {
        (stem.to_string(), "unknown".to_string())
    }
}

fn discover_datasets(file: &File) -> Vec<String> {
    // Collect dataset paths from root
    let mut datasets = Vec::new();
    let common_signals = [
        "matrix/a",
        "matrix/b",
        "matrix/c",
        "polynomial/roots",
        "polynomial/coefficients",
        "state_space/a",
        "state_space/b",
        "state_space/c",
        "state_space/d",
        "state_space/time_series/state_trajectory",
        "state_space/time_series/output_trajectory",
        "transfer_function/numerator",
        "transfer_function/denominator",
        "tensor/data",
        "transient/v_out",
        "plant/natural_frequency_rad_s",
        "plant/damping_ratio",
    ];

    for signal in common_signals {
        if file.dataset(signal).is_ok() {
            datasets.push(signal.to_string());
        }
    }

    datasets
}
