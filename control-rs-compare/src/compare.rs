//! HDF5 typed dataset inspection and numerical comparison engine.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hdf5_pure::{AttrValue, Dataset, File, Group};

use crate::config::{MasterPlan, ToleranceTable};
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

    /// Explicit signals to verify (if provided, overrides dynamic discovery).
    pub signals: Option<Vec<String>>,

    /// If true, discrepancy triggers a non-zero exit return.
    pub strict: bool,

    /// If true, suppresses stdout streaming messages.
    pub quiet: bool,

    /// Optional thread count override for parallel chunked evaluation.
    pub num_threads: Option<usize>,
}

impl Default for ComparatorOptions {
    fn default() -> Self {
        Self {
            results_dir: PathBuf::from("results"),
            suite_filter: None,
            oracle_override: None,
            signals: None,
            strict: true,
            quiet: false,
            num_threads: None,
        }
    }
}

/// Tolerance bound configuration for a numerical comparison check.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToleranceSpec {
    /// Method name ("abs", "rel", "rms", "matrix_norm", "exact_match").
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

/// Composite multi-method tolerance policy for evaluating a signal.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignalTolerancePolicy {
    /// List of numerical tolerance methods to evaluate.
    pub methods: Vec<ToleranceSpec>,
    /// Satisfaction policy ("all_of" or "any_of").
    pub policy: String,
}

impl Default for SignalTolerancePolicy {
    fn default() -> Self {
        Self {
            methods: vec![ToleranceSpec::default()],
            policy: "all_of".to_string(),
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

        // Load external tolerance table if configured
        let tolerance_table = configured_suite
            .and_then(|s| s.tolerance_table.as_deref())
            .map(Path::new)
            .and_then(|p| {
                if p.exists() {
                    ToleranceTable::load_from_file(p).ok()
                } else {
                    None
                }
            });

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

            // Determine signals: explicit provided signals take precedence over dynamic discovery
            let explicit_signals = options
                .signals
                .as_ref()
                .or_else(|| configured_suite.and_then(|s| s.signals.as_ref()));

            let signals = if let Some(explicit) = explicit_signals {
                explicit.clone()
            } else {
                discover_datasets(&oracle_file)
            };

            for signal in signals {
                let comparison_key =
                    format!("{actual_oracle_suite}.{signal}.{peer_variant}");

                let oracle_ds = oracle_file.dataset(&signal).ok();
                let policy = resolve_signal_tolerances(
                    oracle_ds.as_ref(),
                    &signal,
                    peer_variant,
                    tolerance_table.as_ref(),
                );

                let mut method_findings = Vec::new();
                let mut method_passes = Vec::new();

                for spec in &policy.methods {
                    let threads = options.num_threads.unwrap_or_else(|| {
                        std::thread::available_parallelism()
                            .map_or(1, std::num::NonZero::get)
                    });
                    let res = compare_dataset_with_threads(
                        &oracle_file,
                        &peer_file,
                        &signal,
                        spec,
                        threads,
                    );
                    let passed = res.verdict == "pass";
                    method_passes.push(passed);
                    method_findings.push(res);
                }

                let signal_passed = if policy.policy == "any_of" {
                    method_passes.iter().any(|&p| p)
                } else {
                    method_passes.iter().all(|&p| p)
                };

                if !signal_passed {
                    suite_passed = false;
                }

                suite_comparisons.push(ComparisonFinding {
                    key: comparison_key,
                    pair: (actual_oracle_variant.clone(), peer_variant.clone()),
                    signal,
                    policy: policy.policy,
                    verdict: if signal_passed { "pass" } else { "fail" }
                        .to_string(),
                    methods: method_findings,
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

/// Compares a single dataset between an oracle and peer HDF5 file using default available concurrency.
#[must_use]
pub fn compare_dataset(
    oracle_file: &File,
    peer_file: &File,
    signal: &str,
    tol: &ToleranceSpec,
) -> MethodFinding {
    let threads =
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    compare_dataset_with_threads(oracle_file, peer_file, signal, tol, threads)
}

/// Compares a single dataset between an oracle and peer HDF5 file with explicit concurrency.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn compare_dataset_with_threads(
    oracle_file: &File,
    peer_file: &File,
    signal: &str,
    tol: &ToleranceSpec,
    num_threads: usize,
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

        return compare_float_arrays_parallel(
            &o_data,
            &p_data,
            tol,
            num_threads,
        );
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

/// Minimum element count threshold to trigger multi-threaded chunked evaluation.
pub const CHUNK_THRESHOLD: usize = 65_536;

/// Partial summary statistics accumulated over a chunk slice of dataset elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartialChunkStats {
    /// Maximum absolute pointwise difference observed in the chunk.
    pub max_abs: f64,
    /// Maximum relative pointwise error observed in the chunk.
    pub max_rel: f64,
    /// Sum of squared residuals observed in the chunk.
    pub sum_sq: f64,
    /// Set to true if any NaN or Infinity value was encountered.
    pub has_invalid: bool,
}

impl Default for PartialChunkStats {
    fn default() -> Self {
        Self {
            max_abs: 0.0,
            max_rel: 0.0,
            sum_sq: 0.0,
            has_invalid: false,
        }
    }
}

/// Computes partial comparison statistics over a single slice of oracle and peer elements.
#[must_use]
pub fn compute_chunk_stats(oracle: &[f64], peer: &[f64]) -> PartialChunkStats {
    let mut stats = PartialChunkStats::default();

    for (&o, &p) in oracle.iter().zip(peer.iter()) {
        if o.is_nan() || p.is_nan() || o.is_infinite() || p.is_infinite() {
            stats.has_invalid = true;
            return stats;
        }

        let diff = (o - p).abs();
        if diff > stats.max_abs {
            stats.max_abs = diff;
        }

        let denom = o.abs() + f64::EPSILON;
        let rel = diff / denom;
        if rel > stats.max_rel {
            stats.max_rel = rel;
        }

        stats.sum_sq += diff * diff;
    }

    stats
}

/// Reduces an array of chunk statistics and evaluates against a tolerance specification.
#[must_use]
pub fn reduce_and_evaluate(
    total_len: usize,
    partials: &[PartialChunkStats],
    tol: &ToleranceSpec,
) -> MethodFinding {
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for chunk in partials {
        if chunk.has_invalid {
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

        if chunk.max_abs > max_abs {
            max_abs = chunk.max_abs;
        }
        if chunk.max_rel > max_rel {
            max_rel = chunk.max_rel;
        }
        sum_sq += chunk.sum_sq;
    }

    #[allow(clippy::cast_precision_loss)]
    let count = total_len as f64;
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

/// Compares two floating-point vectors sequentially according to a tolerance specification.
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

    let stats = compute_chunk_stats(oracle, peer);
    reduce_and_evaluate(oracle.len(), &[stats], tol)
}

/// Compares two floating-point vectors in parallel across worker threads.
#[must_use]
pub fn compare_float_arrays_parallel(
    oracle: &[f64],
    peer: &[f64],
    tol: &ToleranceSpec,
    num_threads: usize,
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

    let threads = num_threads.max(1);
    if oracle.len() < CHUNK_THRESHOLD || threads <= 1 {
        return compare_float_arrays(oracle, peer, tol);
    }

    let max_chunks = oracle.len().div_ceil(CHUNK_THRESHOLD);
    let actual_threads = threads.min(max_chunks);
    let chunk_size = oracle.len().div_ceil(actual_threads);
    let mut partials = vec![PartialChunkStats::default(); actual_threads];

    std::thread::scope(|s| {
        for (k, out_stat) in partials.iter_mut().enumerate() {
            let start = k * chunk_size;
            let end = (start + chunk_size).min(oracle.len());
            if start < end {
                let o_slice = &oracle[start..end];
                let p_slice = &peer[start..end];
                s.spawn(move || {
                    *out_stat = compute_chunk_stats(o_slice, p_slice);
                });
            }
        }
    });

    reduce_and_evaluate(oracle.len(), &partials, tol)
}

fn parse_suite_and_variant(path: &Path) -> (String, String) {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown.unknown");

    let parts: Vec<&str> = stem.split('.').collect();
    if parts.len() >= 2 {
        let suite = (*parts.first().unwrap_or(&"unknown")).to_string();
        let variant = (*parts.get(1).unwrap_or(&"unknown")).to_string();
        (suite, variant)
    } else {
        (stem.to_string(), "unknown".to_string())
    }
}

/// Discovers all datasets in an HDF5 container by recursively traversing the group hierarchy.
///
/// Starts at the root group and visits all child groups recursively (like running `ls`),
/// accumulating fully qualified dataset paths (e.g. `"matrix/a"`, `"transient/v_out"`).
/// Groups starting with an underscore (such as `_meta`) are skipped per multi-modal container conventions.
/// Discovered dataset paths are sorted deterministically.
#[must_use]
pub fn discover_datasets(file: &File) -> Vec<String> {
    let mut datasets = Vec::new();
    let root = file.root();
    collect_group_datasets(&root, "", &mut datasets);
    datasets.sort();
    datasets
}

fn collect_group_datasets(group: &Group, prefix: &str, out: &mut Vec<String>) {
    if let Ok(ds_names) = group.datasets() {
        for name in ds_names {
            if name.starts_with('_') {
                continue;
            }
            let path = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            out.push(path);
        }
    }

    if let Ok(subgroups) = group.groups() {
        for grp_name in subgroups {
            if grp_name.starts_with('_') {
                continue;
            }
            let next_prefix = if prefix.is_empty() {
                grp_name.clone()
            } else {
                format!("{prefix}/{grp_name}")
            };
            if let Ok(subgroup) = group.group(&grp_name) {
                collect_group_datasets(&subgroup, &next_prefix, out);
            }
        }
    }
}

/// Resolves the tolerance policy for a dataset signal from:
/// 1. External TOML tolerance table (`suite.tolerance_table`).
/// 2. HDF5 dataset attributes on the oracle container.
/// 3. Default fallback tolerance (`abs` <= 1e-4, `policy = "all_of"`).
#[must_use]
pub fn resolve_signal_tolerances(
    oracle_dataset: Option<&Dataset>,
    signal: &str,
    peer_variant: &str,
    tolerance_table: Option<&ToleranceTable>,
) -> SignalTolerancePolicy {
    // 1. External tolerance table check
    if let Some(table) = tolerance_table {
        if let Some(cfg) = table.find_signal(signal) {
            let policy_str = if cfg.policy.is_empty() {
                "all_of".to_string()
            } else {
                cfg.policy.clone()
            };

            let mut methods = Vec::new();
            if !cfg.methods.is_empty() {
                for m in &cfg.methods {
                    let bound = cfg
                        .peer_bounds
                        .get(peer_variant)
                        .copied()
                        .unwrap_or(m.bound);
                    methods.push(ToleranceSpec {
                        method: m.r#type.clone(),
                        bound,
                    });
                }
            } else if let Some(m) = &cfg.method {
                let bound = cfg
                    .peer_bounds
                    .get(peer_variant)
                    .copied()
                    .or(cfg.bound)
                    .unwrap_or(1e-4);
                methods.push(ToleranceSpec {
                    method: m.clone(),
                    bound,
                });
            } else if let Some(b) = cfg.bound {
                let bound =
                    cfg.peer_bounds.get(peer_variant).copied().unwrap_or(b);
                methods.push(ToleranceSpec {
                    method: "abs".to_string(),
                    bound,
                });
            }

            if !methods.is_empty() {
                return SignalTolerancePolicy {
                    methods,
                    policy: policy_str,
                };
            }
        }
    }

    // 2. Oracle dataset HDF5 attributes check
    if let Some(ds) = oracle_dataset {
        if let Ok(attrs) = ds.attrs() {
            if let Some(policy) =
                extract_policy_from_attrs(&attrs, peer_variant)
            {
                return policy;
            }
        }
    }

    // 3. Fallback default
    SignalTolerancePolicy::default()
}

fn extract_policy_from_attrs(
    attrs: &HashMap<String, AttrValue>,
    peer_variant: &str,
) -> Option<SignalTolerancePolicy> {
    let peer_override_key = format!("bound.{peer_variant}");
    let peer_bound_override =
        attrs.get(&peer_override_key).and_then(attr_to_f64);

    // Multi-method JSON string attribute
    if let Some(methods_attr) = attrs.get("methods") {
        if let Some(json_str) = attr_to_string(methods_attr) {
            if let Ok(parsed_methods) =
                serde_json::from_str::<Vec<serde_json::Value>>(&json_str)
            {
                let mut specs = Vec::new();
                for item in parsed_methods {
                    let method_name = item
                        .get("type")
                        .or_else(|| item.get("method"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("abs")
                        .to_string();
                    let bound = peer_bound_override
                        .or_else(|| {
                            item.get("bound")
                                .and_then(serde_json::Value::as_f64)
                        })
                        .unwrap_or(1e-4);
                    specs.push(ToleranceSpec {
                        method: method_name,
                        bound,
                    });
                }

                if !specs.is_empty() {
                    let policy_str = attrs
                        .get("policy")
                        .and_then(attr_to_string)
                        .unwrap_or_else(|| "all_of".to_string());
                    return Some(SignalTolerancePolicy {
                        methods: specs,
                        policy: policy_str,
                    });
                }
            }
        }
    }

    // Single method attribute
    let method_name = attrs
        .get("measure")
        .or_else(|| attrs.get("method"))
        .and_then(attr_to_string);

    let bound_val = peer_bound_override
        .or_else(|| attrs.get("bound").and_then(attr_to_f64));

    if let (Some(m), Some(b)) = (method_name.as_ref(), bound_val) {
        return Some(SignalTolerancePolicy {
            methods: vec![ToleranceSpec {
                method: m.clone(),
                bound: b,
            }],
            policy: "all_of".to_string(),
        });
    }

    if let Some(b) = bound_val {
        return Some(SignalTolerancePolicy {
            methods: vec![ToleranceSpec {
                method: method_name.unwrap_or_else(|| "abs".to_string()),
                bound: b,
            }],
            policy: "all_of".to_string(),
        });
    }

    if let Some(m) = method_name {
        return Some(SignalTolerancePolicy {
            methods: vec![ToleranceSpec {
                method: m,
                bound: 1e-4,
            }],
            policy: "all_of".to_string(),
        });
    }

    None
}

fn attr_to_f64(attr: &AttrValue) -> Option<f64> {
    match attr {
        AttrValue::F64(v) => Some(*v),
        AttrValue::F32(v) => Some(f64::from(*v)),
        AttrValue::F64Array(arr) => arr.first().copied(),
        AttrValue::F32Array(arr) => arr.first().map(|&v| f64::from(v)),
        AttrValue::I32(v) => Some(f64::from(*v)),
        #[allow(clippy::cast_precision_loss)]
        AttrValue::I64(v) => Some(*v as f64),
        AttrValue::U32(v) => Some(f64::from(*v)),
        #[allow(clippy::cast_precision_loss)]
        AttrValue::U64(v) => Some(*v as f64),
        AttrValue::String(s)
        | AttrValue::AsciiString(s)
        | AttrValue::VarLenString(s)
        | AttrValue::VarLenAsciiString(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn attr_to_string(attr: &AttrValue) -> Option<String> {
    match attr {
        AttrValue::String(s)
        | AttrValue::AsciiString(s)
        | AttrValue::VarLenString(s)
        | AttrValue::VarLenAsciiString(s) => Some(s.clone()),
        AttrValue::StringArray(arr)
        | AttrValue::AsciiStringArray(arr)
        | AttrValue::VarLenStringArray(arr)
        | AttrValue::VarLenAsciiStringArray(arr)
        | AttrValue::VarLenAsciiCharArray(arr) => arr.first().cloned(),
        _ => None,
    }
}
