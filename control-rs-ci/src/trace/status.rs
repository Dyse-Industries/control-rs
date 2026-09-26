//! Requirement status from gate results.
//!
//! A requirement's gates are the gate names that appear in code spans on its
//! reference rows. The first matching condition sets its status: any gate
//! failed, any gate without a result or skipped, every gate passed or warned,
//! and no gate named.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;

use super::{
    DEFINITION, Defect, Defects, MARKER, Row, SCHEMA, code_spans, write_file,
};
use crate::config::GateConfig;
use crate::error::{GateError, GateResult};
use crate::gate::{GateOutcome, RESULT_SUFFIX, Verdict};

/// A report and its defects.
pub type Derived = (TraceReport, Defects);

/// Gate names.
pub type GateNames = BTreeSet<String>;

/// Verdict of each named gate; `None` when the gate has no result.
pub type GateVerdicts = BTreeMap<String, Option<Verdict>>;

/// Gates named by each requirement, by qualified ID.
pub type NamedGates = BTreeMap<String, GateNames>;

/// Recorded verdicts by gate name. A gate without a usable result is absent.
pub type Verdicts = BTreeMap<String, Verdict>;

/// Status of one requirement.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize,
)]
pub enum Status {
    /// A gate verdict is `fail`.
    Failed,
    /// A gate has no result, or its verdict is `skipped`.
    Unverified,
    /// At least one gate, and every gate passed or warned.
    Verified,
    /// No gate is named.
    Unchecked,
}

/// One requirement's entry in `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RequirementStatus {
    /// Qualified ID.
    pub id: String,
    /// Derived status.
    pub status: Status,
    /// Verdict of each named gate; `None` when the gate has no result.
    pub gates: GateVerdicts,
}

/// `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TraceReport {
    /// Report format version.
    pub schema: u32,
    /// Number of requirements per status.
    pub counts: BTreeMap<Status, usize>,
    /// One entry per requirement, in the order of their first definition.
    pub requirements: Vec<RequirementStatus>,
    /// Marker rows whose ID has no definition.
    pub unresolved_markers: Vec<Row>,
}

impl TraceReport {
    /// Whether no requirement is `Failed` or `Unverified` and every marker
    /// resolves to a definition.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.unresolved_markers.is_empty()
            && self.requirements.iter().all(|r| {
                !matches!(r.status, Status::Failed | Status::Unverified)
            })
    }

    /// Writes the report as pretty-printed JSON, creating the parent
    /// directory.
    ///
    /// # Errors
    /// `GateError::Io` or `GateError::Json` on failure.
    pub fn write(&self, path: &Path) -> GateResult<()> {
        let mut json = serde_json::to_vec_pretty(self)?;
        json.push(b'\n');
        write_file(path, &json)
    }
}

/// Names of the gates that `gate.toml` defines.
///
/// # Errors
/// `GateError::Config` if the file is missing or does not parse.
pub fn gate_names(path: &Path) -> GateResult<GateNames> {
    if !path.is_file() {
        return Err(GateError::Config {
            path: path.to_path_buf(),
            message: "file not found".to_string(),
        });
    }
    Ok(GateConfig::load_from_path(path)?
        .gate_definitions
        .into_keys()
        .collect())
}

/// The recorded verdict of each gate in `gates` that has a readable
/// `<gate>.result.json` in `dir`.
#[must_use]
pub fn load_verdicts(dir: &Path, gates: &GateNames) -> Verdicts {
    gates
        .iter()
        .filter_map(|gate| {
            let path = dir.join(format!("{gate}{RESULT_SUFFIX}"));
            GateOutcome::load_from_file(&path)
                .ok()
                .map(|outcome| (gate.clone(), outcome.verdict))
        })
        .collect()
}

/// The gates named in code spans on the reference rows of each requirement.
#[must_use]
pub fn named_gates(reqs: &[Row], gate_names: &GateNames) -> NamedGates {
    let mut named = NamedGates::new();
    for row in reqs
        .iter()
        .filter(|r| r.kind != DEFINITION && r.kind != MARKER)
    {
        let gates = named.entry(row.id.clone()).or_default();
        for span in code_spans(&row.text) {
            let name = span.content.trim();
            if gate_names.contains(name) {
                gates.insert(name.to_string());
            }
        }
    }
    named
}

/// The report and its defects: each `Failed` or `Unverified` requirement at
/// its first definition, and each marker whose ID has no definition.
#[must_use]
pub fn derive(
    reqs: &[Row],
    marks: &[Row],
    gate_names: &GateNames,
    verdicts: &Verdicts,
) -> Derived {
    let named = named_gates(reqs, gate_names);
    let mut counts: BTreeMap<Status, usize> = [
        Status::Failed,
        Status::Unverified,
        Status::Verified,
        Status::Unchecked,
    ]
    .into_iter()
    .map(|status| (status, 0))
    .collect();
    let mut defined = BTreeSet::new();
    let mut requirements = Vec::new();
    let mut defects = Vec::new();
    for definition in reqs.iter().filter(|r| r.kind == DEFINITION) {
        if !defined.insert(definition.id.as_str()) {
            continue;
        }
        let gates: GateVerdicts = named
            .get(&definition.id)
            .into_iter()
            .flatten()
            .map(|gate| (gate.clone(), verdicts.get(gate).copied()))
            .collect();
        let status = status_of(&gates);
        if let Some(count) = counts.get_mut(&status) {
            *count = count.saturating_add(1);
        }
        if let Some(reason) = failure(status, &gates) {
            defects.push(Defect::at(
                definition,
                format!("{} is {status:?}: {reason}", definition.id),
            ));
        }
        requirements.push(RequirementStatus {
            id: definition.id.clone(),
            status,
            gates,
        });
    }
    let unresolved_markers: Vec<Row> = marks
        .iter()
        .filter(|m| m.kind == MARKER && !defined.contains(m.id.as_str()))
        .cloned()
        .collect();
    for marker in &unresolved_markers {
        defects.push(Defect::at(
            marker,
            format!("{} has no definition", marker.id),
        ));
    }
    let report = TraceReport {
        schema: SCHEMA,
        counts,
        requirements,
        unresolved_markers,
    };
    (report, defects)
}

/// The status the first matching condition gives a requirement's gates.
fn status_of(gates: &GateVerdicts) -> Status {
    if gates.is_empty() {
        Status::Unchecked
    } else if gates.values().any(|v| *v == Some(Verdict::Fail)) {
        Status::Failed
    } else if gates
        .values()
        .any(|v| matches!(v, None | Some(Verdict::Skipped)))
    {
        Status::Unverified
    } else {
        Status::Verified
    }
}

/// Why a `Failed` or `Unverified` requirement fails the trace.
fn failure(status: Status, gates: &GateVerdicts) -> Option<String> {
    let reasons: Vec<String> = gates
        .iter()
        .filter_map(|(gate, verdict)| match (status, verdict) {
            (Status::Failed, Some(Verdict::Fail)) => {
                Some(format!("gate {gate} failed"))
            }
            (Status::Unverified, None) => {
                Some(format!("gate {gate} has no result"))
            }
            (Status::Unverified, Some(Verdict::Skipped)) => {
                Some(format!("gate {gate} was skipped"))
            }
            _ => None,
        })
        .collect();
    (!reasons.is_empty()).then(|| reasons.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, kind: &str, text: &str) -> Row {
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: kind.to_string(),
            file: "docs/w-design.md".to_string(),
            line: 1,
            text: text.to_string(),
        }
    }

    /// A definition with one plan row naming `gate`, or none when empty. The
    /// kind cell, `example`, is not a gate name.
    fn requirement(id: &str, gate: &str) -> Vec<Row> {
        let cell = if gate.is_empty() {
            String::new()
        } else {
            format!("`{gate}`")
        };
        vec![
            row(id, DEFINITION, "- **FR-1 — A**: It shall work."),
            row(id, "plan", &format!("| FR-1 | `example` | {cell} | Step |")),
        ]
    }

    fn names() -> GateNames {
        ["test", "lint"].map(String::from).into()
    }

    fn status(reqs: &[Row], verdicts: &Verdicts) -> (Status, bool) {
        let (report, defects) = derive(reqs, &[], &names(), verdicts);
        let status = report.requirements.first().map(|r| r.status);
        (status.unwrap(), defects.is_empty() && report.passes())
    }

    #[test]
    fn a_failed_gate_fails_the_requirement() {
        let verdicts = Verdicts::from([("test".to_string(), Verdict::Fail)]);
        let reqs = requirement("w#FR-1", "test");
        assert_eq!(status(&reqs, &verdicts), (Status::Failed, false));
    }

    #[test]
    fn a_gate_without_result_or_skipped_leaves_it_unverified() {
        let reqs = requirement("w#FR-1", "lint");
        assert_eq!(
            status(&reqs, &Verdicts::new()),
            (Status::Unverified, false)
        );
        let skipped = Verdicts::from([("lint".to_string(), Verdict::Skipped)]);
        assert_eq!(status(&reqs, &skipped), (Status::Unverified, false));
    }

    #[test]
    fn passing_or_warning_gates_verify_it() {
        let verdicts = Verdicts::from([
            ("test".to_string(), Verdict::Pass),
            ("lint".to_string(), Verdict::Warn),
        ]);
        let mut reqs = requirement("w#FR-1", "test");
        reqs.push(row("w#FR-1", "acceptance", "| FR-1 | `lint` | a | b | c |"));
        assert_eq!(status(&reqs, &verdicts), (Status::Verified, true));
    }

    #[test]
    fn a_requirement_naming_no_gate_is_unchecked() {
        let reqs = requirement("w#FR-1", "");
        assert_eq!(status(&reqs, &Verdicts::new()), (Status::Unchecked, true));
    }

    #[test]
    fn a_marker_without_definition_fails_the_trace() {
        let reqs = requirement("w#FR-1", "");
        let marks = [row("w#FR-9", MARKER, "fn t")];
        let (report, defects) =
            derive(&reqs, &marks, &names(), &Verdicts::new());
        assert!(!report.passes());
        assert_eq!(report.unresolved_markers.len(), 1);
        let messages: Vec<_> =
            defects.iter().map(ToString::to_string).collect();
        assert_eq!(messages, ["docs/w-design.md:1: w#FR-9 has no definition"]);
    }

    #[test]
    fn counts_cover_every_status_and_duplicates_count_once() {
        let mut reqs = requirement("w#FR-1", "");
        reqs.extend(requirement("w#FR-1", ""));
        let (report, _) = derive(&reqs, &[], &names(), &Verdicts::new());
        assert_eq!(report.requirements.len(), 1);
        assert_eq!(report.counts.len(), 4);
        assert_eq!(report.counts.get(&Status::Unchecked), Some(&1));
    }

    #[test]
    fn failure_messages_name_the_gate() {
        let reqs = requirement("w#FR-1", "test");
        let (_, defects) = derive(&reqs, &[], &names(), &Verdicts::new());
        let messages: Vec<_> =
            defects.iter().map(|d| d.message.as_str()).collect();
        assert_eq!(messages, ["w#FR-1 is Unverified: gate test has no result"]);
    }
}
