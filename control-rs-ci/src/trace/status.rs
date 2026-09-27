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
    CONDITION, DEFINITION, Defect, Defects, MARKER, Row, SCHEMA, code_spans,
    write_file,
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

/// Gates named by each requirement or condition, by qualified ID.
pub type NamedGates = BTreeMap<String, GateNames>;

/// Recorded verdicts by gate name. A gate without a usable result is absent.
pub type Verdicts = BTreeMap<String, Verdict>;

/// Conditions grouped by their parent requirement ID.
type ConditionsByParent<'a> = BTreeMap<&'a str, Vec<&'a Row>>;

/// Derived evaluation outcome: status, gate verdicts, and child condition statuses.
type Evaluation = (Status, GateVerdicts, Vec<ConditionStatus>);

/// Count of occurrences for each marker ID.
type MarkerCounts<'a> = BTreeMap<&'a str, usize>;

/// Status of one requirement or condition.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize,
)]
pub enum Status {
    /// A gate verdict is `fail`.
    Failed,
    /// A gate has no result, or its verdict is `skipped`, or a required marker is missing.
    Unverified,
    /// At least one gate/marker, and every gate passed or warned.
    Verified,
    /// No gate is named.
    Unchecked,
}

/// One condition's status in `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConditionStatus {
    /// Qualified Condition ID.
    pub id: String,
    /// Parent requirement ID.
    pub parent: String,
    /// Derived status.
    pub status: Status,
    /// Verdict of each named gate; `None` when the gate has no result.
    pub gates: GateVerdicts,
    /// Number of matching test markers found in source text.
    pub marker_count: usize,
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
    /// Child verification conditions, if defined.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<ConditionStatus>,
}

/// Count of requirements per status.
pub type StatusCounts = BTreeMap<Status, usize>;

struct Accumulator<'a> {
    counts: &'a mut StatusCounts,
    defects: &'a mut Defects,
}

struct Context<'a> {
    named: &'a NamedGates,
    verdicts: &'a Verdicts,
    marker_counts: &'a MarkerCounts<'a>,
}

/// `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TraceReport {
    /// Report format version.
    pub schema: u32,
    /// Number of requirements per status.
    pub counts: StatusCounts,
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

/// The gates named in code spans on the reference rows of each requirement or condition.
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

fn count_markers<'a>(marks: &'a [Row]) -> MarkerCounts<'a> {
    let mut marker_counts: MarkerCounts<'a> = BTreeMap::new();
    for mark in marks.iter().filter(|m| m.kind == MARKER) {
        let count = marker_counts.entry(mark.id.as_str()).or_default();
        *count = count.saturating_add(1);
    }
    marker_counts
}

fn group_conditions_by_parent<'a>(reqs: &'a [Row]) -> ConditionsByParent<'a> {
    let mut conditions_by_parent: ConditionsByParent<'a> = BTreeMap::new();
    for row in reqs.iter().filter(|r| r.kind == CONDITION) {
        if let Some(parent) = &row.parent {
            conditions_by_parent
                .entry(parent.as_str())
                .or_default()
                .push(row);
        }
    }
    conditions_by_parent
}

fn evaluate_conditions(
    definition: &Row,
    cond_rows: &[&Row],
    ctx: &Context<'_>,
) -> Evaluation {
    let mut cond_statuses = Vec::new();
    let mut all_gates = BTreeMap::new();
    for cond_row in cond_rows {
        let c_gates: GateVerdicts = ctx
            .named
            .get(&cond_row.id)
            .into_iter()
            .flatten()
            .map(|gate| (gate.clone(), ctx.verdicts.get(gate).copied()))
            .collect();
        for (g, v) in &c_gates {
            all_gates.entry(g.clone()).or_insert(*v);
        }
        let m_count = ctx
            .marker_counts
            .get(cond_row.id.as_str())
            .copied()
            .unwrap_or(0)
            .saturating_add(
                ctx.marker_counts
                    .get(definition.id.as_str())
                    .copied()
                    .unwrap_or(0),
            );
        let c_status = condition_status_of(&c_gates, m_count);
        cond_statuses.push(ConditionStatus {
            id: cond_row.id.clone(),
            parent: definition.id.clone(),
            status: c_status,
            gates: c_gates,
            marker_count: m_count,
        });
    }

    let req_status = aggregate_condition_status(&cond_statuses);
    (req_status, all_gates, cond_statuses)
}

fn evaluate_direct(
    definition: &Row,
    named: &NamedGates,
    verdicts: &Verdicts,
) -> Evaluation {
    let direct_gates: GateVerdicts = named
        .get(&definition.id)
        .into_iter()
        .flatten()
        .map(|gate| (gate.clone(), verdicts.get(gate).copied()))
        .collect();
    let req_status = status_of(&direct_gates);
    (req_status, direct_gates, Vec::new())
}

fn condition_unverified_reason(c: &ConditionStatus) -> String {
    let reason = failure(c.status, &c.gates).unwrap_or_else(|| {
        if c.marker_count == 0 && c.gates.keys().any(|g| g.contains("test")) {
            "no test marker".to_string()
        } else {
            "unverified".to_string()
        }
    });
    format!("{}: {reason}", c.id)
}

fn process_definition(
    definition: &Row,
    conditions_by_parent: &ConditionsByParent<'_>,
    ctx: &Context<'_>,
    acc: &mut Accumulator<'_>,
) -> RequirementStatus {
    let (status, gates, conditions) = conditions_by_parent
        .get(definition.id.as_str())
        .map_or_else(
            || evaluate_direct(definition, ctx.named, ctx.verdicts),
            |cond_rows| evaluate_conditions(definition, cond_rows, ctx),
        );

    if let Some(count) = acc.counts.get_mut(&status) {
        *count = count.saturating_add(1);
    }

    if status == Status::Failed || status == Status::Unverified {
        if !conditions.is_empty() {
            let unverified_reasons: Vec<String> = conditions
                .iter()
                .filter(|c| {
                    c.status == Status::Failed || c.status == Status::Unverified
                })
                .map(condition_unverified_reason)
                .collect();
            acc.defects.push(Defect::at(
                definition,
                format!(
                    "{} is {status:?}: [{}]",
                    definition.id,
                    unverified_reasons.join("; ")
                ),
            ));
        } else if let Some(reason) = failure(status, &gates) {
            acc.defects.push(Defect::at(
                definition,
                format!("{} is {status:?}: {reason}", definition.id),
            ));
        }
    }

    RequirementStatus {
        id: definition.id.clone(),
        status,
        gates,
        conditions,
    }
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
    let marker_counts = count_markers(marks);
    let conditions_by_parent = group_conditions_by_parent(reqs);
    let ctx = Context {
        named: &named,
        verdicts,
        marker_counts: &marker_counts,
    };

    let mut counts: StatusCounts = [
        Status::Failed,
        Status::Unverified,
        Status::Verified,
        Status::Unchecked,
    ]
    .into_iter()
    .map(|status| (status, 0))
    .collect();

    let defined: BTreeSet<&str> = reqs
        .iter()
        .filter(|r| r.kind == DEFINITION || r.kind == CONDITION)
        .map(|r| r.id.as_str())
        .collect();

    let mut seen_reqs = BTreeSet::new();
    let mut requirements = Vec::new();
    let mut defects = Vec::new();
    let mut acc = Accumulator {
        counts: &mut counts,
        defects: &mut defects,
    };

    for definition in reqs.iter().filter(|r| r.kind == DEFINITION) {
        if seen_reqs.insert(definition.id.as_str()) {
            requirements.push(process_definition(
                definition,
                &conditions_by_parent,
                &ctx,
                &mut acc,
            ));
        }
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

/// The status derived for a condition from its gates and markers.
fn condition_status_of(gates: &GateVerdicts, marker_count: usize) -> Status {
    if gates.is_empty() {
        Status::Unchecked
    } else if gates.values().any(|v| *v == Some(Verdict::Fail)) {
        Status::Failed
    } else if gates
        .values()
        .any(|v| matches!(v, None | Some(Verdict::Skipped)))
        || (gates.keys().any(|g| g.contains("test")) && marker_count == 0)
    {
        Status::Unverified
    } else {
        Status::Verified
    }
}

/// The status aggregated across child conditions.
fn aggregate_condition_status(conditions: &[ConditionStatus]) -> Status {
    if conditions.is_empty() {
        Status::Unchecked
    } else if conditions.iter().any(|c| c.status == Status::Failed) {
        Status::Failed
    } else if conditions.iter().any(|c| c.status == Status::Unverified) {
        Status::Unverified
    } else if conditions.iter().all(|c| c.status == Status::Unchecked) {
        Status::Unchecked
    } else {
        Status::Verified
    }
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

    const PFX: &str = concat!("#[", "req(",);

    fn row(id: &str, kind: &str, text: &str) -> Row {
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: kind.to_string(),
            parent: None,
            file: "docs/w-design.md".to_string(),
            line: 1,
            text: text.to_string(),
        }
    }

    fn condition_row(id: &str, parent: &str, text: &str) -> Row {
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: CONDITION.to_string(),
            parent: Some(parent.to_string()),
            file: "docs/w-design.md".to_string(),
            line: 1,
            text: text.to_string(),
        }
    }

    /// A definition with one verification row naming `gate`, or none when
    /// empty. The kind cell, `example`, is not a gate name.
    fn requirement(id: &str, gate: &str) -> Vec<Row> {
        let cell = if gate.is_empty() {
            String::new()
        } else {
            format!("`{gate}`")
        };
        vec![
            row(id, DEFINITION, "- **FR-1 — A**: It shall work."),
            row(id, "verification", &format!("| FR-1 | {cell} | Step |")),
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
        reqs.push(row("w#FR-1", "verification", "| FR-1 | `lint` | a |"));
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

    #[test]
    fn verification_condition_requires_marker_for_test_gate() {
        let verdicts = Verdicts::from([("test".to_string(), Verdict::Pass)]);
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        // Without marker -> Unverified
        let (report, defects) = derive(&reqs, &[], &names(), &verdicts);
        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Unverified)
        );
        assert!(!report.passes());
        assert_eq!(
            defects.first().map(|d| d.message.as_str()),
            Some("w#FR-1 is Unverified: [w#VC-1.1: no test marker]")
        );

        // With marker -> Verified
        let mark_text = format!("{PFX}\"w#VC-1.1\")]");
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let (report_pass, defects_pass) =
            derive(&reqs, &marks, &names(), &verdicts);
        assert_eq!(
            report_pass.requirements.first().map(|r| r.status),
            Some(Status::Verified)
        );
        assert!(report_pass.passes());
        assert!(defects_pass.is_empty());
    }

    #[test]
    fn multiple_conditions_conjunction_determines_requirement_status() {
        let verdicts = Verdicts::from([("test".to_string(), Verdict::Pass)]);
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "| VC-1.1 | FR-1 | `test` | Step 1 |",
            ),
            condition_row(
                "w#VC-1.2",
                "w#FR-1",
                "| VC-1.2 | FR-1 | `test` | Step 2 |",
            ),
        ];
        // Only VC-1.1 marked -> FR-1 remains Unverified
        let m1 = format!("{PFX}\"w#VC-1.1\")]");
        let m2 = format!("{PFX}\"w#VC-1.2\")]");
        let marks = [row("w#VC-1.1", MARKER, &m1)];
        let (report, defects) = derive(&reqs, &marks, &names(), &verdicts);
        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Unverified)
        );
        assert_eq!(
            defects.first().map(|d| d.message.as_str()),
            Some("w#FR-1 is Unverified: [w#VC-1.2: no test marker]")
        );

        // Both marked -> FR-1 Verified
        let marks_both =
            [row("w#VC-1.1", MARKER, &m1), row("w#VC-1.2", MARKER, &m2)];
        let (report_both, defects_both) =
            derive(&reqs, &marks_both, &names(), &verdicts);
        assert_eq!(
            report_both.requirements.first().map(|r| r.status),
            Some(Status::Verified)
        );
        assert!(report_both.passes());
        assert!(defects_both.is_empty());
    }
}
