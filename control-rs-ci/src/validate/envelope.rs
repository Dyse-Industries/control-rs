//! Unified result envelope validation and regression store.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub use super::comparator::ComparisonRecord;
pub use super::tolerance::{ToleranceBound, ToleranceTable};

/// Validation verdict for a result container or envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerValidationReport {
    /// Path of the container or envelope file.
    pub file_path: String,
    /// Subject identifier.
    pub subject: String,
    /// Sources present in the container.
    pub sources: Vec<String>,
    /// Comparisons present in the container.
    #[serde(default)]
    pub comparisons: Vec<ComparisonRecord>,
    /// Whether the container adheres to the validation contract.
    pub is_valid: bool,
    /// Any structural discrepancies or schema errors.
    pub errors: Vec<String>,
}

/// Envelope metadata header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeMetadata {
    /// Subject identifier, e.g. "`buck_converter`".
    #[serde(default)]
    pub subject: String,
    /// UTC timestamp string.
    #[serde(default)]
    pub timestamp: String,
    /// Optional git commit SHA.
    #[serde(default)]
    pub git_commit: Option<String>,
    /// Host compilation target triple.
    #[serde(default)]
    pub rust_target: Option<String>,
    /// Profile (debug/release).
    #[serde(default)]
    pub profile: Option<String>,
    /// Requirements verified by this envelope.
    #[serde(default)]
    pub requirements: Vec<String>,
}

/// Backward-compatible alias for [`ContainerValidationReport`].
pub type EnvelopeValidationReport = ContainerValidationReport;

/// Constraints an envelope must satisfy to count as evidence for this run.
///
/// `results/` is gitignored but persists between runs, so without these an
/// envelope written at an earlier revision validates as current evidence.
#[derive(Debug, Clone, Default)]
pub struct FreshnessPolicy {
    /// Short SHA of the revision under test.
    pub head_commit: Option<String>,
    /// Earliest modification time an envelope of this run can carry.
    pub written_after: Option<SystemTime>,
}

/// Unified multi-source result envelope conforming to `oracle-harness-design.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultEnvelope {
    /// Execution metadata.
    pub metadata: EnvelopeMetadata,
    /// Result payload per source ("rust", "scipy", "ngspice", etc.).
    pub sources: HashMap<String, serde_json::Value>,
    /// Individual comparison records against authoritative tolerances.
    #[serde(default)]
    pub comparisons: Vec<ComparisonRecord>,
}

/// Compare two bounds allowing only round-trip representation error.
fn bounds_agree(a: f64, b: f64) -> bool {
    if a.to_bits() == b.to_bits() {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= 1e-12 * scale
}

fn check_comparison_record(
    comp: &ComparisonRecord,
    tolerances: Option<&ToleranceTable>,
    errors: &mut Vec<String>,
) {
    let declared_pass = comp.verdict == "pass";

    if !declared_pass {
        let msg = comp.details.clone().unwrap_or_else(|| {
            format!(
                "Comparison failed for '{}': observed {} exceeded bound {}",
                comp.key, comp.observed, comp.bound
            )
        });
        errors.push(msg);
    }

    let Some(table) = tolerances else {
        return;
    };
    let Some(bound) = table.get(&comp.key) else {
        errors.push(format!(
            "Comparison key '{}' is declared in no tolerance table",
            comp.key
        ));
        return;
    };
    check_record_against_bound(comp, bound, declared_pass, errors);
}

fn check_envelope_sources(envelope: &ResultEnvelope, errors: &mut Vec<String>) {
    if !envelope.sources.contains_key("rust") {
        errors
            .push("Missing required 'rust' source in sources map".to_string());
    }
    if envelope.sources.len() < 2 {
        errors.push(
            "Expected at least one external oracle source in addition to 'rust'".to_string(),
        );
    }
}

fn check_freshness_policy(
    path: &Path,
    envelope: &ResultEnvelope,
    freshness: Option<&FreshnessPolicy>,
    errors: &mut Vec<String>,
) {
    let Some(policy) = freshness else {
        return;
    };
    if let (Some(head), Some(stamped)) = (
        policy.head_commit.as_deref(),
        envelope.metadata.git_commit.as_deref(),
    ) {
        // Either may be abbreviated, so accept a common prefix.
        if !head.starts_with(stamped) && !stamped.starts_with(head) {
            errors.push(format!(
                "Envelope records git_commit '{stamped}', this run is at \
                 '{head}'"
            ));
        }
    }
    let Some(floor) = policy.written_after else {
        return;
    };
    match fs::metadata(path).and_then(|m| m.modified()) {
        Ok(modified) if modified >= floor => {}
        Ok(_) => errors.push(format!(
            "Envelope {} predates this run and is stale",
            path.display()
        )),
        Err(e) => errors.push(format!(
            "Cannot determine modification time of {}: {e}",
            path.display()
        )),
    }
}

fn check_record_against_bound(
    comp: &ComparisonRecord,
    bound: &ToleranceBound,
    declared_pass: bool,
    errors: &mut Vec<String>,
) {
    if !comp.measure.is_empty() && comp.measure != bound.measure {
        errors.push(format!(
            "Comparison '{}' reports measure '{}', table declares '{}'",
            comp.key, comp.measure, bound.measure
        ));
    }

    let table_bound = expected_bound(bound);
    if !bounds_agree(comp.bound, table_bound) {
        errors.push(format!(
            "Comparison '{}' reports bound {:e}, table declares {:e}",
            comp.key, comp.bound, table_bound
        ));
    }

    if recompute_verdict(comp, bound) != declared_pass {
        errors.push(format!(
            "Comparison '{}' reports '{}', but observed {:e} against \
             bound {:e} re-derives to '{}'",
            comp.key,
            comp.verdict,
            comp.observed,
            table_bound,
            if declared_pass { "fail" } else { "pass" }
        ));
    }
}

/// Discovers all `results/*.json` envelopes under an example directory.
#[must_use]
pub fn discover_result_envelopes(example_dir: &Path) -> Vec<PathBuf> {
    let results_dir = example_dir.join("results");
    let mut envelopes = Vec::new();

    let Ok(entries) = fs::read_dir(results_dir) else {
        return envelopes;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path.extension().and_then(|e| e.to_str()) == Some("json")
        {
            let file_name =
                path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Disallow legacy sidecar dumps from matching as envelopes
            if file_name.ends_with("_oracle.json") {
                continue;
            }
            envelopes.push(path);
        }
    }

    envelopes.sort();
    envelopes
}

fn envelope_subject(envelope: &ResultEnvelope, path: &Path) -> String {
    if envelope.metadata.subject.trim().is_empty() {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    } else {
        envelope.metadata.subject.clone()
    }
}

/// Bound a record must satisfy, accounting for interval enclosures.
fn expected_bound(bound: &ToleranceBound) -> f64 {
    match bound.interval {
        Some([_, max]) if bound.measure == "interval" => max,
        _ => bound.bound,
    }
}

/// Re-derive a verdict from the observed value and the authoritative bound.
fn recompute_verdict(comp: &ComparisonRecord, bound: &ToleranceBound) -> bool {
    if !comp.observed.is_finite() {
        return false;
    }
    if bound.measure == "interval" {
        let [min, max] = bound.interval.unwrap_or([0.0, bound.bound]);
        comp.observed >= min && comp.observed <= max
    } else {
        comp.observed <= expected_bound(bound)
    }
}

/// Validate that a JSON file is a valid unified multi-source result envelope.
///
/// # Errors
///
/// Returns an error string when `path` cannot be read or does not parse as
/// a result envelope. Envelope defects are reported in the returned
/// report, not as an error.
pub fn validate_envelope_file(
    path: &Path,
    tolerances: Option<&ToleranceTable>,
    freshness: Option<&FreshnessPolicy>,
) -> Result<ContainerValidationReport, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    let envelope =
        serde_json::from_str::<ResultEnvelope>(&content).map_err(|e| {
            format!(
                "JSON file {} does not conform to unified envelope schema: {e}",
                path.display()
            )
        })?;

    let mut errors = Vec::new();
    let subject = envelope_subject(&envelope, path);
    check_envelope_sources(&envelope, &mut errors);
    if envelope.comparisons.is_empty() {
        errors.push("Envelope contains zero comparison records".to_string());
    }
    for comp in &envelope.comparisons {
        check_comparison_record(comp, tolerances, &mut errors);
    }
    check_freshness_policy(path, &envelope, freshness, &mut errors);

    let sources = envelope.sources.keys().cloned().collect();
    Ok(ContainerValidationReport {
        file_path: path.to_string_lossy().to_string(),
        subject,
        sources,
        comparisons: envelope.comparisons,
        is_valid: errors.is_empty(),
        errors,
    })
}
