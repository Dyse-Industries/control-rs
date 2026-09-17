//! True-oracle 1:1 HDF5 comparison against dataset attributes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::h5::{DiscoveredDataset, DiscoveredDatasets, H5Container};
use super::tolerance::ToleranceBound;

/// Comparison records paired with the overall validation result.
pub type CompareOutcome = (Vec<ComparisonRecord>, ValidationResult);

/// Peer variant name and HDF5 path.
pub type PeerPath = (String, PathBuf);

/// Result of a cross-validation pass: `Ok(())` or the list of discrepancies.
pub type ValidationResult = Result<(), Vec<String>>;

type OptionalF64Dataset = Option<super::h5::ShapeAndData<f64>>;

/// Structured record of an individual cross-validation check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonRecord {
    /// Unique metric token from the true-oracle dataset path.
    pub key: String,
    /// Verified pair (e.g. `["rust", "scipy"]`).
    pub pair: [String; 2],
    /// Error metric: "abs" | "rel" | "`rel_l2`" | "residual" | "exact" | "interval" | "lt".
    pub measure: String,
    /// Declared numeric bound from true-oracle HDF5 attributes.
    pub bound: f64,
    /// Observed discrepancy or error.
    pub observed: f64,
    /// Verdict: "pass" | "fail".
    pub verdict: String,
    /// Detailed diagnostic message on breach.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
    /// False when the oracle transcribes the design under test rather than
    /// computing an independent reference.
    #[serde(default = "default_independent")]
    pub independent: bool,
    /// HDF5 signal path (without the variant prefix).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<String>,
    /// Worst-element index for array comparisons.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
}

/// Accumulates discrepancies between a Rust payload and an oracle payload.
#[derive(Debug, Default)]
pub struct CrossValidation {
    errs: Vec<String>,
    records: Vec<ComparisonRecord>,
}

struct DatasetCompare<'a> {
    oracle: &'a OracleView,
    oracle_variant: &'a str,
    peer_variant: &'a str,
    peer: &'a H5Container,
    ds: &'a DiscoveredDataset,
}

struct EvalState {
    observed: f64,
    passed: bool,
    max_diff_pair: (f64, f64),
    max_diff_idx: usize,
}

struct NumericEval<'a> {
    key: &'a str,
    var_a: &'a str,
    var_b: &'a str,
    signal: &'a str,
    data_a: &'a [f64],
    data_b: &'a [f64],
    bound: &'a ToleranceBound,
}

struct OracleView {
    container: H5Container,
    datasets: DiscoveredDatasets,
    paths: BTreeSet<String>,
}

impl Default for ComparisonRecord {
    fn default() -> Self {
        Self {
            key: String::new(),
            pair: Default::default(),
            measure: String::new(),
            bound: 0.0,
            observed: 0.0,
            verdict: String::new(),
            details: None,
            independent: true,
            signal: None,
            index: None,
        }
    }
}

impl CrossValidation {
    /// Creates an empty cross-validation tracker.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            errs: Vec::new(),
            records: Vec::new(),
        }
    }

    /// Records a discrepancy directly.
    pub fn push(&mut self, msg: impl Into<String>) {
        self.errs.push(msg.into());
    }

    /// Records a structured comparison record directly.
    pub fn record(&mut self, record: ComparisonRecord) {
        self.records.push(record);
    }

    /// Returns the accumulated structured comparison records.
    #[must_use]
    pub fn comparisons(&self) -> &[ComparisonRecord] {
        &self.records
    }

    /// Consumes the tracker and returns the comparison records and validation result.
    pub fn into_comparisons(self) -> CompareOutcome {
        let res = if self.errs.is_empty() {
            Ok(())
        } else {
            Err(self.errs)
        };
        (self.records, res)
    }

    /// Returns true when no discrepancy has been recorded.
    #[must_use]
    pub const fn is_clean(&self) -> bool {
        self.errs.is_empty()
    }

    /// Consumes the comparison, yielding the accumulated discrepancies.
    ///
    /// # Errors
    /// Returns every recorded discrepancy when the comparison is not clean.
    pub fn finish(self) -> ValidationResult {
        if self.errs.is_empty() {
            Ok(())
        } else {
            Err(self.errs)
        }
    }
}

/// Compares each peer file 1:1 against the true-oracle file.
pub fn compare_h5_files(
    oracle_path: &Path,
    oracle_variant: &str,
    peers: &[PeerPath],
) -> CompareOutcome {
    let mut cv = CrossValidation::new();
    let Some(oracle) = load_oracle(&mut cv, oracle_path) else {
        return cv.into_comparisons();
    };
    if peers.is_empty() {
        cv.push(
            "no peer variant files to compare against the true oracle"
                .to_string(),
        );
        return cv.into_comparisons();
    }
    for peer in peers {
        compare_peer(&mut cv, &oracle, oracle_variant, peer);
    }
    if cv.comparisons().is_empty() && cv.is_clean() {
        cv.push(
            "true-oracle comparison produced zero comparison records"
                .to_string(),
        );
    }
    cv.into_comparisons()
}

fn compare_one_dataset(cv: &mut CrossValidation, ctx: &DatasetCompare<'_>) {
    let spec = match ctx.oracle.container.read_dataset_tolerance(&ctx.ds.path) {
        Ok(s) => s,
        Err(e) => {
            cv.push(e);
            return;
        }
    };
    let bound = spec.as_bound(ctx.peer_variant);
    let Some((shape_a, data_a)) =
        read_f64_or_push(cv, &ctx.oracle.container, &ctx.ds.path)
    else {
        return;
    };
    let Some((shape_b, data_b)) = read_f64_or_push(cv, ctx.peer, &ctx.ds.path)
    else {
        return;
    };
    if shape_a != shape_b {
        cv.push(format!(
            "{} shape mismatch: {} {shape_a:?} vs {} {shape_b:?}",
            ctx.ds.signal, ctx.oracle_variant, ctx.peer_variant
        ));
        return;
    }
    if let Some(idx) = non_numeric_index(&data_a, &data_b) {
        cv.push(format!("{}[{idx}]: non-numeric entry", ctx.ds.signal));
        return;
    }
    evaluate_numeric_dataset(
        cv,
        &NumericEval {
            key: &ctx.ds.signal,
            var_a: ctx.oracle_variant,
            var_b: ctx.peer_variant,
            signal: &ctx.ds.signal,
            data_a: &data_a,
            data_b: &data_b,
            bound: &bound,
        },
    );
}

fn compare_peer(
    cv: &mut CrossValidation,
    oracle: &OracleView,
    oracle_variant: &str,
    peer: &PeerPath,
) {
    let (peer_variant, peer_path) = peer;
    let Some(peer_container) = open_peer_container(cv, peer_path) else {
        return;
    };
    let Some(peer_ds) = traverse_or_push(cv, &peer_container, peer_path) else {
        return;
    };
    let peer_paths: BTreeSet<String> =
        peer_ds.iter().map(|d| d.signal.clone()).collect();
    let mut has_extra = false;
    for extra in peer_paths.difference(&oracle.paths) {
        has_extra = true;
        cv.push(format!(
            "signal '{extra}' present in '{peer_variant}' missing from true oracle '{oracle_variant}'"
        ));
    }
    if has_extra {
        return;
    }
    for ds in &oracle.datasets {
        if !peer_paths.contains(&ds.signal) {
            continue;
        }
        compare_one_dataset(
            cv,
            &DatasetCompare {
                oracle,
                oracle_variant,
                peer_variant,
                peer: &peer_container,
                ds,
            },
        );
    }
}

const fn default_independent() -> bool {
    true
}

fn eval_abs(eval: &NumericEval<'_>, state: &mut EvalState) {
    for (idx, (&a, &b)) in
        eval.data_a.iter().zip(eval.data_b.iter()).enumerate()
    {
        note_diff(state, (a - b).abs(), (a, b), idx);
    }
    state.passed = state.observed <= eval.bound.bound;
}

fn eval_exact(eval: &NumericEval<'_>, state: &mut EvalState) {
    for (idx, (&a, &b)) in
        eval.data_a.iter().zip(eval.data_b.iter()).enumerate()
    {
        if a.to_bits() != b.to_bits() {
            state.observed = 1.0;
            state.passed = false;
            state.max_diff_pair = (a, b);
            state.max_diff_idx = idx;
            break;
        }
    }
}

fn eval_interval(
    cv: &mut CrossValidation,
    eval: &NumericEval<'_>,
    state: &mut EvalState,
) -> bool {
    let Some([min, max]) = eval.bound.interval else {
        cv.push(format!(
            "{}: measure=interval requires an interval attribute",
            eval.key
        ));
        return false;
    };
    for (idx, &b) in eval.data_b.iter().enumerate() {
        if b < min {
            note_diff(state, min - b, (b, min), idx);
        } else if b > max {
            note_diff(state, b - max, (b, max), idx);
        }
    }
    state.passed = state.observed <= 0.0;
    true
}

fn eval_lt(eval: &NumericEval<'_>, state: &mut EvalState) {
    let mut mag_b: f64 = 0.0;
    for (idx, (&a, &b)) in
        eval.data_a.iter().zip(eval.data_b.iter()).enumerate()
    {
        mag_b = mag_b.max(b.abs());
        if b >= a {
            note_diff(state, b - a, (a, b), idx);
            state.passed = false;
        }
    }
    if eval.bound.bound > 0.0 && mag_b > eval.bound.bound {
        state.passed = false;
        if mag_b > state.observed {
            state.observed = mag_b;
        }
    } else if state.passed {
        state.observed = mag_b;
    }
}

fn eval_rel(eval: &NumericEval<'_>, state: &mut EvalState) {
    for (idx, (&a, &b)) in
        eval.data_a.iter().zip(eval.data_b.iter()).enumerate()
    {
        let diff = (a - b).abs() / a.abs().max(1e-12);
        note_diff(state, diff, (a, b), idx);
    }
    state.passed = state.observed <= eval.bound.bound;
}

fn eval_rel_l2(eval: &NumericEval<'_>, state: &mut EvalState) {
    let mut sum_sq_diff = 0.0;
    let mut sum_sq_a = 0.0;
    for (&a, &b) in eval.data_a.iter().zip(eval.data_b.iter()) {
        sum_sq_diff += (a - b).powi(2);
        sum_sq_a += a.powi(2);
    }
    let denom = sum_sq_a.sqrt().max(1e-12);
    state.observed = sum_sq_diff.sqrt() / denom;
    state.passed = state.observed <= eval.bound.bound;
    if let Some((&a, &b)) = eval.data_a.first().zip(eval.data_b.first()) {
        state.max_diff_pair = (a, b);
    }
}

/// Evaluates numeric arrays against an authoritative tolerance bound.
fn evaluate_numeric_dataset(cv: &mut CrossValidation, eval: &NumericEval<'_>) {
    let mut state = EvalState {
        observed: 0.0,
        passed: true,
        max_diff_pair: (0.0, 0.0),
        max_diff_idx: 0,
    };
    match eval.bound.measure.as_str() {
        "rel_l2" => eval_rel_l2(eval, &mut state),
        "rel" => eval_rel(eval, &mut state),
        "exact" => eval_exact(eval, &mut state),
        "lt" => eval_lt(eval, &mut state),
        "interval" => {
            if !eval_interval(cv, eval, &mut state) {
                return;
            }
        }
        "abs" => eval_abs(eval, &mut state),
        other => {
            cv.push(format!("{}: unrecognized measure '{other}'", eval.key));
            return;
        }
    }
    record_numeric_result(cv, eval, &state);
}

fn load_oracle(
    cv: &mut CrossValidation,
    oracle_path: &Path,
) -> Option<OracleView> {
    let container = match H5Container::open(oracle_path) {
        Ok(c) => c,
        Err(e) => {
            cv.push(format!(
                "Cannot open true-oracle HDF5 at '{}': {e}",
                oracle_path.display()
            ));
            return None;
        }
    };
    let datasets = traverse_or_push(cv, &container, oracle_path)?;
    let paths: BTreeSet<String> =
        datasets.iter().map(|d| d.signal.clone()).collect();
    if paths.is_empty() {
        cv.push(format!(
            "true-oracle '{}' contains zero compared datasets",
            oracle_path.display()
        ));
        return None;
    }
    Some(OracleView {
        container,
        datasets,
        paths,
    })
}

fn non_numeric_index(data_a: &[f64], data_b: &[f64]) -> Option<usize> {
    data_a
        .iter()
        .chain(data_b.iter())
        .enumerate()
        .find(|(_, v)| !v.is_finite())
        .map(|(i, _)| {
            if i < data_a.len() {
                i
            } else {
                i.saturating_sub(data_a.len())
            }
        })
}

fn note_diff(state: &mut EvalState, diff: f64, pair: (f64, f64), idx: usize) {
    if diff > state.observed {
        state.observed = diff;
        state.max_diff_pair = pair;
        state.max_diff_idx = idx;
    }
}

fn open_peer_container(
    cv: &mut CrossValidation,
    peer_path: &Path,
) -> Option<H5Container> {
    match H5Container::open(peer_path) {
        Ok(c) => Some(c),
        Err(e) => {
            cv.push(format!(
                "Cannot open peer HDF5 at '{}': {e}",
                peer_path.display()
            ));
            None
        }
    }
}

fn read_f64_or_push(
    cv: &mut CrossValidation,
    container: &H5Container,
    path: &str,
) -> OptionalF64Dataset {
    match container.read_dataset_f64(path) {
        Ok(v) => Some(v),
        Err(e) => {
            cv.push(e);
            None
        }
    }
}

fn record_numeric_result(
    cv: &mut CrossValidation,
    eval: &NumericEval<'_>,
    state: &EvalState,
) {
    let measure = eval.bound.measure.as_str();
    let coupled = if eval.bound.independent {
        ""
    } else {
        "; coupled transcription"
    };
    let details = format!(
        "{} {}[{}]: {} {:.6e} vs {} {:.6e} ({measure} diff {:.2e} > tol {:.2e}){coupled}",
        eval.key,
        eval.signal,
        state.max_diff_idx,
        eval.var_a,
        state.max_diff_pair.0,
        eval.var_b,
        state.max_diff_pair.1,
        state.observed,
        eval.bound.bound
    );
    if !state.passed {
        cv.push(details.clone());
    }
    cv.record(ComparisonRecord {
        key: eval.key.to_string(),
        pair: [eval.var_a.to_string(), eval.var_b.to_string()],
        measure: eval.bound.measure.clone(),
        bound: eval.bound.bound,
        observed: state.observed,
        verdict: if state.passed {
            "pass".to_string()
        } else {
            "fail".to_string()
        },
        details: if state.passed { None } else { Some(details) },
        independent: eval.bound.independent,
        signal: Some(eval.signal.to_string()),
        index: Some(state.max_diff_idx),
    });
}

fn traverse_or_push(
    cv: &mut CrossValidation,
    container: &H5Container,
    path: &Path,
) -> Option<DiscoveredDatasets> {
    match container.traverse_datasets() {
        Ok(d) => Some(d),
        Err(e) => {
            cv.push(format!("Error traversing '{}': {e}", path.display()));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::h5::{DatasetTolerance, H5Container};

    fn spec(measure: &str, bound: f64) -> DatasetTolerance {
        DatasetTolerance {
            measure: measure.to_string(),
            bound,
            interval: None,
            independent: true,
            peer_bounds: Vec::new(),
        }
    }

    fn write_file(
        path: &Path,
        values: &[f64],
        attrs: Option<DatasetTolerance>,
    ) {
        let c = H5Container::create(path).unwrap();
        c.write_dataset_1d("/step", values).unwrap();
        if let Some(s) = attrs {
            c.write_dataset_tolerance("/step", &s).unwrap();
        }
        c.close().unwrap();
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#FR-6
    /// Method: Requirements-based test
    fn test_path_sets_must_match() {
        let temp_dir = std::env::temp_dir().join("control_rs_ci_cmp_paths");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let oracle = temp_dir.join("o.h5");
        let peer = temp_dir.join("p.h5");
        write_file(&oracle, &[1.0], Some(spec("abs", 1e-5)));
        let c = H5Container::create(&peer).unwrap();
        c.write_dataset_1d("/step", &[1.0]).unwrap();
        c.write_dataset_1d("/extra", &[2.0]).unwrap();
        c.close().unwrap();
        let (_, result) =
            compare_h5_files(&oracle, "scipy", &[("rust".to_string(), peer)]);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#FR-6
    /// Method: Requirements-based test
    fn test_peer_bound_override() {
        let temp_dir = std::env::temp_dir().join("control_rs_ci_cmp_override");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let oracle = temp_dir.join("o.h5");
        let jax = temp_dir.join("j.h5");
        let rust = temp_dir.join("r.h5");
        let mut s = spec("abs", 1e-9);
        s.peer_bounds.push(("jax".to_string(), 1.0));
        write_file(&oracle, &[1.0], Some(s));
        write_file(&jax, &[1.5], None);
        write_file(&rust, &[1.000_000_000_1], None);
        let (records, result) = compare_h5_files(
            &oracle,
            "scipy",
            &[("jax".to_string(), jax), ("rust".to_string(), rust)],
        );
        assert!(result.is_ok(), "{result:?}");
        let jax_rec = records
            .iter()
            .find(|r| r.pair.get(1).is_some_and(|p| p == "jax"))
            .unwrap();
        assert_eq!(jax_rec.verdict, "pass");
        assert_eq!(jax_rec.bound.to_bits(), 1.0_f64.to_bits());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#NFR-2
    /// Method: Requirements-based test
    fn test_numeric_determinism() {
        let temp_dir = std::env::temp_dir().join("control_rs_ci_cmp_det");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let oracle = temp_dir.join("o.h5");
        let peer = temp_dir.join("p.h5");
        write_file(&oracle, &[1.0, 2.0, 3.0], Some(spec("abs", 1e-5)));
        write_file(&peer, &[1.0, 2.0, 3.000_000_1], None);
        let peers = vec![("rust".to_string(), peer)];
        let (a, va) = compare_h5_files(&oracle, "scipy", &peers);
        let (b, vb) = compare_h5_files(&oracle, "scipy", &peers);
        assert_eq!(va.is_ok(), vb.is_ok());
        assert_eq!(a, b);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_missing_attributes_fail() {
        let temp_dir = std::env::temp_dir().join("control_rs_ci_cmp_attrs");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let oracle = temp_dir.join("o.h5");
        let peer = temp_dir.join("p.h5");
        write_file(&oracle, &[1.0], None);
        write_file(&peer, &[1.0], None);
        let (_, result) =
            compare_h5_files(&oracle, "scipy", &[("rust".to_string(), peer)]);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#FR-6
    /// Method: Requirements-based test
    fn test_interval_evaluates_peer_not_oracle() {
        let temp_dir = std::env::temp_dir().join("control_rs_ci_cmp_interval");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let oracle = temp_dir.join("o.h5");
        let peer_ok = temp_dir.join("p_ok.h5");
        let peer_bad = temp_dir.join("p_bad.h5");
        let mut s = spec("interval", 0.0);
        s.interval = Some([0.0, 1.0]);
        write_file(&oracle, &[5.0], Some(s));
        write_file(&peer_ok, &[0.5], None);
        write_file(&peer_bad, &[2.0], None);
        let (_, ok) = compare_h5_files(
            &oracle,
            "scipy",
            &[("rust".to_string(), peer_ok)],
        );
        assert!(ok.is_ok(), "{ok:?}");
        let (_, bad) = compare_h5_files(
            &oracle,
            "scipy",
            &[("rust".to_string(), peer_bad)],
        );
        assert!(bad.is_err());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#FR-3
    /// Method: Requirements-based test
    fn test_unknown_measure_fails_closed() {
        let temp_dir = std::env::temp_dir().join("control_rs_ci_cmp_measure");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let oracle = temp_dir.join("o.h5");
        let peer = temp_dir.join("p.h5");
        write_file(&oracle, &[1.0], Some(spec("residual", 1e-3)));
        write_file(&peer, &[2.0], None);
        let (_, result) =
            compare_h5_files(&oracle, "scipy", &[("rust".to_string(), peer)]);
        assert!(
            result.is_err(),
            "unrecognized measure must not evaluate as abs, got {result:?}"
        );
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    /// Relative measures divide by the true-oracle magnitude, so a huge
    /// wrong peer cannot shrink the error.
    ///
    /// # Verification
    /// Trace: oracle-harness#FR-7
    /// Method: Requirements-based test
    fn test_rel_normalizes_by_true_oracle() {
        let temp_dir = std::env::temp_dir().join("control_rs_ci_cmp_rel_den");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let oracle = temp_dir.join("o.h5");
        let peer = temp_dir.join("p.h5");
        write_file(&oracle, &[1.0], Some(spec("rel", 0.1)));
        write_file(&peer, &[100.0], None);
        let (records, result) =
            compare_h5_files(&oracle, "scipy", &[("rust".to_string(), peer)]);
        assert!(result.is_err(), "peer 100 vs oracle 1 must fail rel 0.1");
        let rec = records
            .iter()
            .find(|r| r.measure == "rel")
            .expect("rel record");
        assert!(
            (rec.observed - 99.0).abs() < 1e-9,
            "observed = {}, expected |100-1|/|1| = 99",
            rec.observed
        );
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    /// A peer that only produced a subset of the true-oracle paths is still
    /// compared on the overlap. Padding the peer with oracle arrays is not
    /// required for the path sets to be usable.
    ///
    /// # Verification
    /// Trace: oracle-harness#FR-8
    /// Method: Requirements-based test
    fn test_independent_peer_subset_is_compared() {
        let temp_dir = std::env::temp_dir().join("control_rs_ci_cmp_subset");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let oracle = temp_dir.join("o.h5");
        let peer = temp_dir.join("p.h5");
        let c = H5Container::create(&oracle).unwrap();
        c.write_dataset_1d("/full", &[1.0, 2.0]).unwrap();
        c.write_dataset_tolerance("/full", &spec("abs", 1e-5))
            .unwrap();
        c.write_dataset_1d("/partial", &[10.0]).unwrap();
        c.write_dataset_tolerance("/partial", &spec("abs", 1e-5))
            .unwrap();
        c.close().unwrap();
        let p = H5Container::create(&peer).unwrap();
        p.write_dataset_1d("/partial", &[10.000_000_000_1]).unwrap();
        p.close().unwrap();
        let (records, result) =
            compare_h5_files(&oracle, "numpy", &[("flint".to_string(), peer)]);
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(records.len(), 1);
        assert_eq!(records.first().map(|r| r.key.as_str()), Some("partial"));
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#FR-9
    /// Method: Requirements-based test
    fn test_bound_only_in_meta_is_not_compared() {
        let temp_dir = std::env::temp_dir().join("control_rs_ci_cmp_meta");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let oracle = temp_dir.join("o.h5");
        let peer = temp_dir.join("p.h5");
        let c = H5Container::create(&oracle).unwrap();
        c.write_dataset_1d("/step", &[1.0]).unwrap();
        c.write_dataset_tolerance("/step", &spec("abs", 1e-5))
            .unwrap();
        c.write_dataset_1d("/_meta/spice_bound", &[0.05]).unwrap();
        c.close().unwrap();
        write_file(&peer, &[1.000_000_000_1], None);
        let (records, result) =
            compare_h5_files(&oracle, "scipy", &[("rust".to_string(), peer)]);
        assert!(result.is_ok(), "{result:?}");
        assert!(
            records.iter().all(|r| !r.key.contains("_meta")),
            "bounds that live only under /_meta must not appear as comparison records"
        );
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
