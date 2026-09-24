//! HDF5 typed dataset inspection and numerical comparison engine.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hdf5_pure::{AttrValue, Dataset, File, Group};

use crate::config::{
    MasterPlan, SignalToleranceConfig, SuiteConfig, ToleranceTable,
};
use crate::error::HarnessError;
use crate::report::{
    ComparisonFinding, MethodFinding, SuiteReport, ValidationReport,
    ValidationSummary,
};

/// Minimum element count threshold to trigger multi-threaded chunked evaluation.
pub const CHUNK_THRESHOLD: usize = 65_536;

/// Names of the suites to compare.
pub type SuiteNames = BTreeSet<String>;

/// Dataset paths to compare.
pub type SignalNames = Vec<String>;

/// Container paths of one suite keyed by variant name.
type VariantFiles = BTreeMap<String, PathBuf>;

/// Variant containers keyed by suite name.
type SuiteFiles = BTreeMap<String, VariantFiles>;

/// HDF5 attributes of one dataset.
type Attrs = HashMap<String, AttrValue>;

/// High-level execution options for the comparison engine.
#[derive(Debug, Clone)]
pub struct ComparatorOptions {
    /// Directory containing `.h5` result containers.
    pub results_dir: PathBuf,

    /// Optional suite filter (runs comparison only for named suites).
    pub suite_filter: Option<SuiteNames>,

    /// True oracle override (for example, `"scipy"`).
    pub oracle_override: Option<String>,

    /// Explicit signals to verify (if provided, overrides dynamic discovery).
    pub signals: Option<SignalNames>,

    /// If true, discrepancy triggers a non-zero exit return.
    pub strict: bool,

    /// If true, suppresses stdout streaming messages.
    pub quiet: bool,

    /// Optional thread count override for parallel chunked evaluation.
    pub num_threads: Option<usize>,
}

/// Tolerance bound configuration for a numerical comparison check.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToleranceSpec {
    /// Method name (`"abs"`, `"rel"`, `"rms"`, `"matrix_norm"`, `"exact_match"`).
    pub method: String,
    /// Numerical threshold or tolerance bound.
    pub bound: f64,
}

/// Composite multi-method tolerance policy for evaluating a signal.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignalTolerancePolicy {
    /// List of numerical tolerance methods to evaluate.
    pub methods: Vec<ToleranceSpec>,
    /// Satisfaction policy (`"all_of"` or `"any_of"`).
    pub policy: String,
}

/// Oracle side of one suite comparison and the settings shared by its peers.
struct SuiteOracle<'a> {
    suite_name: &'a str,
    /// Suite name parsed from the oracle container's file name.
    oracle_suite: String,
    /// Variant name of the oracle container.
    oracle_variant: String,
    oracle_file: File,
    /// Explicit signal list, overriding discovery, when configured.
    explicit_signals: Option<&'a SignalNames>,
    tolerance_table: Option<ToleranceTable>,
    num_threads: usize,
}

/// The same dataset path in an oracle and a peer container.
struct DatasetPair<'a> {
    oracle: &'a File,
    peer: &'a File,
    signal: &'a str,
}

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
    /// Number of element pairs accumulated, as `f64` for the RMS mean
    /// (exact up to 2^53 elements).
    pub count: f64,
}

/// A peer container and its variant name.
struct Peer<'a> {
    file: &'a File,
    variant: &'a str,
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

impl Default for ToleranceSpec {
    fn default() -> Self {
        Self {
            method: "abs".to_string(),
            bound: 1e-4,
        }
    }
}

impl Default for SignalTolerancePolicy {
    fn default() -> Self {
        Self {
            methods: vec![ToleranceSpec::default()],
            policy: "all_of".to_string(),
        }
    }
}

impl<'a> SuiteOracle<'a> {
    /// Opens the oracle container and resolves the suite's signal list,
    /// tolerance table and thread count. The error is the failure reason.
    fn open(
        suite_name: &'a str,
        oracle_path: &Path,
        configured_suite: Option<&'a SuiteConfig>,
        options: &'a ComparatorOptions,
    ) -> Result<Self, String> {
        let (oracle_suite, oracle_variant) =
            parse_suite_and_variant(oracle_path);
        let oracle_file = File::open(oracle_path).map_err(|e| {
            format!(
                "Failed to open oracle container '{}': {e}",
                oracle_path.display()
            )
        })?;

        // Load external tolerance table if configured
        let tolerance_table = configured_suite
            .and_then(|s| s.tolerance_table.as_deref())
            .map(Path::new)
            .filter(|p| p.exists())
            .and_then(|p| ToleranceTable::load_from_file(p).ok());

        Ok(Self {
            suite_name,
            oracle_suite,
            oracle_variant,
            oracle_file,
            // Explicit signals take precedence over dynamic discovery.
            explicit_signals: options
                .signals
                .as_ref()
                .or_else(|| configured_suite.and_then(|s| s.signals.as_ref())),
            tolerance_table,
            num_threads: options.num_threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map_or(1, std::num::NonZero::get)
            }),
        })
    }

    /// Finding for a whole-container failure of `peer_variant`.
    fn container_failure(
        &self,
        peer_variant: &str,
        kind: &str,
        method: MethodFinding,
    ) -> ComparisonFinding {
        ComparisonFinding {
            key: format!("{}.{kind}.{peer_variant}", self.suite_name),
            pair: (self.oracle_variant.clone(), peer_variant.to_string()),
            signal: "(container)".to_string(),
            policy: "all_of".to_string(),
            verdict: "fail".to_string(),
            methods: vec![method],
        }
    }

    /// Compares every oracle signal with one peer container.
    ///
    /// A peer may omit a signal only where the oracle dataset carries
    /// `missing_ok.<peer>`; every other omission fails the comparison, and a
    /// peer that provides no signal at all fails its coverage check.
    fn compare_peer(
        &self,
        peer_variant: &str,
        peer_path: &Path,
    ) -> Vec<ComparisonFinding> {
        let peer_file = match File::open(peer_path) {
            Ok(f) => f,
            Err(e) => {
                return vec![self.container_failure(
                    peer_variant,
                    "open",
                    MethodFinding {
                        r#type: "container_open_error".to_string(),
                        bound: 0.0,
                        observed: f64::INFINITY,
                        verdict: "fail".to_string(),
                        details: Some(format!(
                            "Failed to open peer container '{}': {e}",
                            peer_path.display()
                        )),
                    },
                )];
            }
        };

        let signals = self
            .explicit_signals
            .map_or_else(|| discover_datasets(&self.oracle_file), Clone::clone);
        let signal_count = signals.len();

        let mut findings = Vec::new();
        for signal in signals {
            let oracle_ds = self.oracle_file.dataset(&signal).ok();
            if peer_file.dataset(&signal).is_err()
                && missing_ok(oracle_ds.as_ref(), peer_variant)
            {
                continue;
            }
            let peer = Peer {
                file: &peer_file,
                variant: peer_variant,
            };
            findings.push(self.compare_signal(
                &peer,
                signal,
                oracle_ds.as_ref(),
            ));
        }

        if signal_count > 0 && findings.is_empty() {
            findings.push(self.container_failure(
                peer_variant,
                "coverage",
                MethodFinding {
                    r#type: "signal_coverage".to_string(),
                    bound: 1.0,
                    observed: 0.0,
                    verdict: "fail".to_string(),
                    details: Some(format!(
                        "Variant '{peer_variant}' provides no oracle signal"
                    )),
                },
            ));
        }
        findings
    }

    /// Evaluates every tolerance method configured for one signal.
    fn compare_signal(
        &self,
        peer: &Peer<'_>,
        signal: String,
        oracle_ds: Option<&Dataset>,
    ) -> ComparisonFinding {
        let peer_variant = peer.variant;
        let policy = resolve_signal_tolerances(
            oracle_ds,
            &signal,
            peer_variant,
            self.tolerance_table.as_ref(),
        );

        let pair = DatasetPair {
            oracle: &self.oracle_file,
            peer: peer.file,
            signal: &signal,
        };
        let methods: Vec<MethodFinding> = policy
            .methods
            .iter()
            .map(|spec| pair.compare(spec, self.num_threads))
            .collect();

        let mut passes = methods.iter().map(|m| m.verdict == "pass");
        let signal_passed = if policy.policy == "any_of" {
            passes.any(|p| p)
        } else {
            passes.all(|p| p)
        };

        ComparisonFinding {
            key: format!("{}.{signal}.{peer_variant}", self.oracle_suite),
            pair: (self.oracle_variant.clone(), peer_variant.to_string()),
            signal,
            policy: policy.policy,
            verdict: if signal_passed { "pass" } else { "fail" }.to_string(),
            methods,
        }
    }
}

impl DatasetPair<'_> {
    /// Compares the two datasets under `tol` using up to `num_threads`
    /// worker threads for large numeric arrays.
    fn compare(
        &self,
        tol: &ToleranceSpec,
        num_threads: usize,
    ) -> MethodFinding {
        let signal = self.signal;
        let o_ds = match self.oracle.dataset(signal) {
            Ok(d) => d,
            Err(e) => {
                return failed_method(
                    &tol.method,
                    tol.bound,
                    format!("Signal '{signal}' not in oracle: {e}"),
                );
            }
        };
        let p_ds = match self.peer.dataset(signal) {
            Ok(d) => d,
            Err(e) => {
                return failed_method(
                    &tol.method,
                    tol.bound,
                    format!("Signal '{signal}' missing in peer: {e}"),
                );
            }
        };

        // Check dimensions
        let o_shape = o_ds.shape().ok();
        let p_shape = p_ds.shape().ok();
        if o_shape != p_shape {
            return failed_method(
                "shape_mismatch",
                0.0,
                format!(
                    "Shape mismatch for '{signal}': oracle={o_shape:?}, peer={p_shape:?}"
                ),
            );
        }

        // Try reading numeric f64 data
        if let Ok(o_data) = o_ds.read_f64() {
            return match p_ds.read_f64() {
                Ok(p_data) => compare_float_arrays_parallel(
                    &o_data,
                    &p_data,
                    tol,
                    num_threads,
                ),
                Err(e) => failed_method(
                    &tol.method,
                    tol.bound,
                    format!(
                        "Failed to read float data from peer '{signal}': {e}"
                    ),
                ),
            };
        }

        // Try reading string data
        if let Ok(o_strings) = o_ds.read_string() {
            return compare_string_datasets(signal, &o_strings, &p_ds);
        }

        failed_method(
            "unsupported_type",
            0.0,
            format!("Unsupported dataset type for '{signal}'"),
        )
    }
}

impl Default for PartialChunkStats {
    fn default() -> Self {
        Self {
            max_abs: 0.0,
            max_rel: 0.0,
            sum_sq: 0.0,
            has_invalid: false,
            count: 0.0,
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
    let suite_files = ingest_results(&options.results_dir)?;

    let suite_reports: Vec<SuiteReport> = suite_files
        .iter()
        .filter(|(suite_name, _)| {
            options
                .suite_filter
                .as_ref()
                .is_none_or(|filter| filter.contains(*suite_name))
        })
        .map(|(suite_name, variants)| {
            compare_suite(plan, options, suite_name, variants)
        })
        .collect();

    let total_passed =
        suite_reports.iter().filter(|r| r.status == "Pass").count();
    let total_failed = suite_reports.len().saturating_sub(total_passed);
    let overall_verdict = if total_failed == 0 { "Pass" } else { "Fail" };

    Ok(ValidationReport {
        summary: ValidationSummary {
            total_suites: suite_reports.len(),
            passed_suites: total_passed,
            failed_suites: total_failed,
            total_duration_secs: start_time.elapsed().as_secs_f64(),
            verdict: overall_verdict.to_string(),
        },
        suites: suite_reports,
    })
}

/// Maps every `.h5`/`.hdf5` file in `results_dir` to `suite -> variant -> path`.
fn ingest_results(results_dir: &Path) -> Result<SuiteFiles, HarnessError> {
    if !results_dir.exists() {
        return Err(HarnessError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Results directory '{}' does not exist",
                results_dir.display()
            ),
        )));
    }

    let mut suite_files = SuiteFiles::new();
    for entry in fs::read_dir(results_dir)? {
        let path = entry?.path();
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
    Ok(suite_files)
}

/// A suite that failed before any comparison ran.
fn failed_suite(name: &str, duration_secs: f64, reason: String) -> SuiteReport {
    SuiteReport {
        name: name.to_string(),
        status: "Fail".to_string(),
        duration_secs,
        comparisons: vec![],
        failure_reason: Some(reason),
    }
}

/// Compares every peer variant of one suite with its oracle container.
fn compare_suite(
    plan: Option<&MasterPlan>,
    options: &ComparatorOptions,
    suite_name: &str,
    variants: &VariantFiles,
) -> SuiteReport {
    let suite_start = Instant::now();
    let configured_suite =
        plan.and_then(|p| p.suites.iter().find(|s| s.name == suite_name));

    let default_oracle_name = "scipy".to_string();
    let oracle_name = options
        .oracle_override
        .as_ref()
        .or_else(|| configured_suite.map(|s| &s.true_oracle))
        .unwrap_or(&default_oracle_name);

    let Some(oracle_p) = variants
        .get(oracle_name)
        .or_else(|| variants.get("scipy").or_else(|| variants.get("rust")))
    else {
        return failed_suite(
            suite_name,
            suite_start.elapsed().as_secs_f64(),
            format!(
                "Reference oracle '{oracle_name}' container missing for suite '{suite_name}'"
            ),
        );
    };

    let oracle = match SuiteOracle::open(
        suite_name,
        oracle_p,
        configured_suite,
        options,
    ) {
        Ok(oracle) => oracle,
        Err(reason) => {
            return failed_suite(
                suite_name,
                suite_start.elapsed().as_secs_f64(),
                reason,
            );
        }
    };

    let mut comparisons = Vec::new();
    for (peer_variant, peer_path) in variants {
        if *peer_variant != oracle.oracle_variant {
            comparisons.extend(oracle.compare_peer(peer_variant, peer_path));
        }
    }

    let suite_passed = comparisons.iter().all(|c| c.verdict == "pass");
    SuiteReport {
        name: suite_name.to_string(),
        status: if suite_passed { "Pass" } else { "Fail" }.to_string(),
        duration_secs: suite_start.elapsed().as_secs_f64(),
        comparisons,
        failure_reason: None,
    }
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
    DatasetPair {
        oracle: oracle_file,
        peer: peer_file,
        signal,
    }
    .compare(tol, threads)
}

/// A failed method finding with an infinite observation.
fn failed_method(method: &str, bound: f64, details: String) -> MethodFinding {
    MethodFinding {
        r#type: method.to_string(),
        bound,
        observed: f64::INFINITY,
        verdict: "fail".to_string(),
        details: Some(details),
    }
}

/// Exact-match comparison of a string dataset with the peer's copy.
fn compare_string_datasets(
    signal: &str,
    oracle: &[String],
    peer_ds: &Dataset,
) -> MethodFinding {
    let passed = match peer_ds.read_string() {
        Ok(p_strings) => oracle == p_strings.as_slice(),
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
    MethodFinding {
        r#type: "exact_match".to_string(),
        bound: 0.0,
        observed: if passed { 1.0 } else { 0.0 },
        verdict: if passed { "pass" } else { "fail" }.to_string(),
        details: (!passed).then(|| "String dataset mismatch".to_string()),
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

        stats.sum_sq = diff.mul_add(diff, stats.sum_sq);
        stats.count += 1.0;
    }

    stats
}

/// Reduces an array of chunk statistics and evaluates against a tolerance specification.
#[must_use]
pub fn reduce_and_evaluate(
    partials: &[PartialChunkStats],
    tol: &ToleranceSpec,
) -> MethodFinding {
    if partials.iter().any(|chunk| chunk.has_invalid) {
        return failed_method(
            &tol.method,
            tol.bound,
            "NaN or Infinite floating-point value detected".to_string(),
        );
    }

    let max_abs = partials.iter().fold(0.0_f64, |m, c| m.max(c.max_abs));
    let max_rel = partials.iter().fold(0.0_f64, |m, c| m.max(c.max_rel));
    let mut sum_sq = 0.0_f64;
    let mut count = 0.0_f64;
    for chunk in partials {
        sum_sq += chunk.sum_sq;
        count += chunk.count;
    }

    match tol.method.as_str() {
        "rel" => {
            bounded_finding("rel", max_rel, tol.bound, "Max relative error")
        }
        "rms" => {
            let rms = if count > 0.0 {
                (sum_sq / count).sqrt()
            } else {
                0.0
            };
            bounded_finding("rms", rms, tol.bound, "RMS error")
        }
        "matrix_norm" | "norm" => bounded_finding(
            "matrix_norm",
            sum_sq.sqrt(),
            tol.bound,
            "Frobenius norm",
        ),
        // Default: abs
        _ => bounded_finding(
            "abs",
            max_abs,
            tol.bound,
            "Max absolute difference",
        ),
    }
}

/// Finding for `observed <= bound`, naming `label` in the failure detail.
fn bounded_finding(
    method: &str,
    observed: f64,
    bound: f64,
    label: &str,
) -> MethodFinding {
    let passed = observed <= bound;
    MethodFinding {
        r#type: method.to_string(),
        bound,
        observed,
        verdict: if passed { "pass" } else { "fail" }.to_string(),
        details: (!passed)
            .then(|| format!("{label} {observed:.3e} > bound {bound:.3e}")),
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
    reduce_and_evaluate(&[stats], tol)
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

    // `chunk_size >= 1` because `oracle.len() >= CHUNK_THRESHOLD`.
    std::thread::scope(|s| {
        let chunks = oracle.chunks(chunk_size).zip(peer.chunks(chunk_size));
        for (out_stat, (o_slice, p_slice)) in partials.iter_mut().zip(chunks) {
            s.spawn(move || {
                *out_stat = compute_chunk_stats(o_slice, p_slice);
            });
        }
    });

    reduce_and_evaluate(&partials, tol)
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
/// accumulating fully qualified dataset paths (for example, `"matrix/a"`, `"transient/v_out"`).
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
    if let Some(policy) = tolerance_table
        .and_then(|table| table.find_signal(signal))
        .and_then(|cfg| policy_from_table(cfg, peer_variant))
    {
        return policy;
    }

    // 2. Oracle dataset HDF5 attributes check
    if let Some(policy) = oracle_dataset
        .and_then(|ds| ds.attrs().ok())
        .and_then(|attrs| extract_policy_from_attrs(&attrs, peer_variant))
    {
        return policy;
    }

    // 3. Fallback default
    SignalTolerancePolicy::default()
}

/// Policy from a tolerance-table entry: its method list, else its single
/// method, else a bare bound as `abs`. `peer_bounds` overrides each bound.
fn policy_from_table(
    cfg: &SignalToleranceConfig,
    peer_variant: &str,
) -> Option<SignalTolerancePolicy> {
    let peer_bound = cfg.peer_bounds.get(peer_variant).copied();
    let methods: Vec<ToleranceSpec> = if cfg.methods.is_empty() {
        match (&cfg.method, cfg.bound) {
            (Some(m), _) => vec![ToleranceSpec {
                method: m.clone(),
                bound: peer_bound.or(cfg.bound).unwrap_or(1e-4),
            }],
            (None, Some(b)) => vec![ToleranceSpec {
                method: "abs".to_string(),
                bound: peer_bound.unwrap_or(b),
            }],
            (None, None) => Vec::new(),
        }
    } else {
        cfg.methods
            .iter()
            .map(|m| ToleranceSpec {
                method: m.r#type.clone(),
                bound: peer_bound.unwrap_or(m.bound),
            })
            .collect()
    };

    (!methods.is_empty()).then(|| SignalTolerancePolicy {
        methods,
        policy: if cfg.policy.is_empty() {
            "all_of".to_string()
        } else {
            cfg.policy.clone()
        },
    })
}

fn extract_policy_from_attrs(
    attrs: &Attrs,
    peer_variant: &str,
) -> Option<SignalTolerancePolicy> {
    let peer_override_key = format!("bound.{peer_variant}");
    let peer_bound_override =
        attrs.get(&peer_override_key).and_then(attr_to_f64);

    // Multi-method JSON string attribute
    if let Some(policy) = policy_from_methods_attr(attrs, peer_bound_override) {
        return Some(policy);
    }

    // Single method attribute
    let method_name = attrs
        .get("measure")
        .or_else(|| attrs.get("method"))
        .and_then(attr_to_string);

    let bound_val = peer_bound_override
        .or_else(|| attrs.get("bound").and_then(attr_to_f64));

    if method_name.is_none() && bound_val.is_none() {
        return None;
    }
    Some(SignalTolerancePolicy {
        methods: vec![ToleranceSpec {
            method: method_name.unwrap_or_else(|| "abs".to_string()),
            bound: bound_val.unwrap_or(1e-4),
        }],
        policy: "all_of".to_string(),
    })
}

/// Policy from a JSON `methods` attribute (a list of `{type|method, bound}`)
/// and an optional `policy` attribute.
fn policy_from_methods_attr(
    attrs: &Attrs,
    peer_bound_override: Option<f64>,
) -> Option<SignalTolerancePolicy> {
    let json_str = attrs.get("methods").and_then(attr_to_string)?;
    let parsed_methods: Vec<serde_json::Value> =
        serde_json::from_str(&json_str).ok()?;
    let specs: Vec<ToleranceSpec> = parsed_methods
        .iter()
        .map(|item| ToleranceSpec {
            method: item
                .get("type")
                .or_else(|| item.get("method"))
                .and_then(|v| v.as_str())
                .unwrap_or("abs")
                .to_string(),
            bound: peer_bound_override
                .or_else(|| {
                    item.get("bound").and_then(serde_json::Value::as_f64)
                })
                .unwrap_or(1e-4),
        })
        .collect();

    (!specs.is_empty()).then(|| SignalTolerancePolicy {
        methods: specs,
        policy: attrs
            .get("policy")
            .and_then(attr_to_string)
            .unwrap_or_else(|| "all_of".to_string()),
    })
}

/// Returns `true` when the oracle dataset marks the signal as optional for `peer_variant`
/// through a non-zero `missing_ok.<peer>` attribute.
fn missing_ok(oracle_dataset: Option<&Dataset>, peer_variant: &str) -> bool {
    let key = format!("missing_ok.{peer_variant}");
    oracle_dataset
        .and_then(|ds| ds.attrs().ok())
        .and_then(|attrs| attrs.get(&key).and_then(attr_to_f64))
        .is_some_and(|v| v != 0.0)
}

fn attr_to_f64(attr: &AttrValue) -> Option<f64> {
    match attr {
        AttrValue::F64(v) => Some(*v),
        AttrValue::F32(v) => Some(f64::from(*v)),
        AttrValue::F64Array(arr) => arr.first().copied(),
        AttrValue::F32Array(arr) => arr.first().map(|&v| f64::from(v)),
        AttrValue::I32(v) => Some(f64::from(*v)),
        // Parsing the decimal form is the correctly rounded conversion.
        AttrValue::I64(v) => v.to_string().parse().ok(),
        AttrValue::U32(v) => Some(f64::from(*v)),
        AttrValue::U64(v) => v.to_string().parse().ok(),
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
