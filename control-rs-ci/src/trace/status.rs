//! Requirement and condition status derivation.
//!
//! Each condition takes a status:
//! - `Covered`: method in `marked_methods` and at least one matching marker exists.
//! - `Uncovered`: method in `marked_methods` and no matching marker exists (fails gate).
//! - `Review`: method is outside `marked_methods` (passes gate).
//!
//! A requirement takes the worst status among its conditions in order
//! `Uncovered` > `Review` > `Covered`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::reqs::TraceConfig;
use super::{
    CONDITION, DEFINITION, Defect, Defects, MARKER, Row, SCHEMA, write_file,
};
use crate::error::GateResult;

/// A report and its defects.
pub type Derived = (TraceReport, Defects);

/// Rows by qualified ID, first occurrence kept.
type ById<'a> = BTreeMap<&'a str, &'a Row>;

/// Rows of one kind by ID, and the IDs in order of first occurrence.
type Indexed<'a> = (ById<'a>, Vec<&'a str>);

/// Marker rows by the qualified ID they name.
type MarkersById<'a> = BTreeMap<&'a str, Vec<&'a Row>>;

/// Status of one requirement or condition.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub enum Status {
    /// Method in `marked_methods` and at least one marker.
    Covered,
    /// Method outside `marked_methods`; awaits sign-off.
    Review,
    /// Method in `marked_methods` and no marker; fails the gate.
    Uncovered,
}

/// Location of a test marker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MarkerLocation {
    /// Path relative to the working directory.
    pub file: String,
    /// 1-based line number of the marker span.
    pub line: usize,
}

/// One condition's status in `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConditionStatus {
    /// Qualified Condition ID.
    pub id: String,
    /// Parent requirement IDs.
    pub parents: Vec<String>,
    /// Verification method.
    pub method: String,
    /// Derived status.
    pub status: Status,
    /// Number of matching test markers found in source text.
    pub marker_count: usize,
    /// Locations of matching markers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub markers: Vec<MarkerLocation>,
}

/// One requirement's entry in `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RequirementStatus {
    /// Qualified ID.
    pub id: String,
    /// Derived status.
    pub status: Status,
    /// Child verification conditions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<ConditionStatus>,
}

/// A review condition awaiting sign-off.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReviewCondition {
    /// Qualified condition ID.
    pub id: String,
    /// Parent requirement IDs.
    pub parents: Vec<String>,
    /// Verification method.
    pub method: String,
    /// Path of the defining document.
    pub file: String,
    /// 1-based line number of the condition row.
    pub line: usize,
}

/// Count of conditions per status.
pub type StatusCounts = BTreeMap<Status, usize>;

/// `trace-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TraceReport {
    /// Report format version.
    pub schema: u32,
    /// Number of conditions per status.
    pub counts: StatusCounts,
    /// One entry per requirement, in the order of their first definition.
    pub requirements: Vec<RequirementStatus>,
    /// Review conditions awaiting sign-off.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub review: Vec<ReviewCondition>,
    /// Marker rows that could not be resolved or name a requirement.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unresolved_markers: Vec<Row>,
}

impl TraceReport {
    /// Whether no condition/requirement is `Uncovered` and all markers are valid.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.unresolved_markers.is_empty()
            && self.counts.get(&Status::Uncovered).copied().unwrap_or(0) == 0
            && self
                .requirements
                .iter()
                .all(|r| r.status != Status::Uncovered)
    }

    /// Writes the report as pretty-printed JSON, creating the parent directory.
    ///
    /// # Errors
    /// `GateError::Io` or `GateError::Json` on failure.
    pub fn write(&self, path: &Path) -> GateResult<()> {
        let mut json = serde_json::to_vec_pretty(self)?;
        json.push(b'\n');
        write_file(path, &json)
    }
}

/// The first row of each ID among `rows` of `kind`, and the IDs in order of
/// first occurrence.
fn first_by_id<'a>(rows: &'a [Row], kind: &str) -> Indexed<'a> {
    let mut by_id = ById::new();
    let mut order = Vec::new();
    for row in rows.iter().filter(|r| r.kind == kind) {
        if !by_id.contains_key(row.id.as_str()) {
            order.push(row.id.as_str());
            by_id.insert(row.id.as_str(), row);
        }
    }
    (by_id, order)
}

/// The markers that name a requirement or no defined condition, with one
/// defect each.
fn marker_defects(
    marks: &[Row],
    definitions: &ById<'_>,
    conditions: &ById<'_>,
    defects: &mut Defects,
) -> Vec<Row> {
    let mut unresolved = Vec::new();
    for mark in marks.iter().filter(|m| m.kind == MARKER) {
        let message = if definitions.contains_key(mark.id.as_str()) {
            format!(
                "marker {} names a requirement; markers name verification \
                 conditions",
                mark.id
            )
        } else if conditions.contains_key(mark.id.as_str()) {
            continue;
        } else {
            format!("marker {} names no defined condition", mark.id)
        };
        defects.push(Defect::at(mark, message));
        unresolved.push(mark.clone());
    }
    unresolved
}

/// The status of one condition row given the markers that name it.
fn condition_status(
    row: &Row,
    markers: &MarkersById<'_>,
    config: &TraceConfig,
) -> ConditionStatus {
    let found = markers.get(row.id.as_str()).map_or(&[][..], Vec::as_slice);
    let method = row.method.clone().unwrap_or_default();
    let status = if !config.marked_methods.contains(&method) {
        Status::Review
    } else if found.is_empty() {
        Status::Uncovered
    } else {
        Status::Covered
    };
    ConditionStatus {
        id: row.id.clone(),
        parents: row.parents.clone(),
        method,
        status,
        marker_count: found.len(),
        markers: found
            .iter()
            .map(|m| MarkerLocation {
                file: m.file.clone(),
                line: m.line,
            })
            .collect(),
    }
}

/// The worst status of `conditions`; `Uncovered` when there are none.
fn worst(conditions: &[ConditionStatus]) -> Status {
    conditions
        .iter()
        .map(|c| c.status)
        .max()
        .unwrap_or(Status::Uncovered)
}

/// One entry per requirement in `order`, each with its conditions sorted by
/// ID and the worst of their statuses.
fn requirement_statuses(
    order: &[&str],
    evaluated: &[ConditionStatus],
) -> Vec<RequirementStatus> {
    order
        .iter()
        .map(|&id| {
            let mut children: Vec<ConditionStatus> = evaluated
                .iter()
                .filter(|c| c.parents.iter().any(|p| p == id))
                .cloned()
                .collect();
            children.sort_by(|a, b| a.id.cmp(&b.id));
            RequirementStatus {
                id: id.to_string(),
                status: worst(&children),
                conditions: children,
            }
        })
        .collect()
}

/// Derives condition and requirement status from the rows of `trace-reqs`
/// and `trace-marks`.
#[must_use]
pub fn derive(reqs: &[Row], marks: &[Row], config: &TraceConfig) -> Derived {
    let (definitions, order) = first_by_id(reqs, DEFINITION);
    let (conditions, condition_order) = first_by_id(reqs, CONDITION);
    let mut markers = MarkersById::new();
    for mark in marks.iter().filter(|m| m.kind == MARKER) {
        markers.entry(mark.id.as_str()).or_default().push(mark);
    }

    let mut defects = Vec::new();
    let unresolved_markers =
        marker_defects(marks, &definitions, &conditions, &mut defects);

    let mut counts: StatusCounts =
        [Status::Covered, Status::Review, Status::Uncovered]
            .into_iter()
            .map(|status| (status, 0))
            .collect();
    let mut evaluated = Vec::new();
    let mut review = Vec::new();
    for row in condition_order.iter().filter_map(|id| conditions.get(id)) {
        let status = condition_status(row, &markers, config);
        if let Some(count) = counts.get_mut(&status.status) {
            *count = count.saturating_add(1);
        }
        match status.status {
            Status::Uncovered => defects.push(Defect::at(
                row,
                format!("{} has no marked test", row.id),
            )),
            Status::Review => review.push(ReviewCondition {
                id: row.id.clone(),
                parents: row.parents.clone(),
                method: status.method.clone(),
                file: row.file.clone(),
                line: row.line,
            }),
            Status::Covered => {}
        }
        evaluated.push(status);
    }

    let requirements = requirement_statuses(&order, &evaluated);

    defects.sort_by(|a, b| {
        (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line))
    });
    let report = TraceReport {
        schema: SCHEMA,
        counts,
        requirements,
        review,
        unresolved_markers,
    };
    (report, defects)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PFX: &str = concat!("#[", "req(",);

    fn default_config() -> TraceConfig {
        TraceConfig {
            id: r"(?:FR|NFR|C)-[0-9]+[a-z]?".to_string(),
            condition: r"VC-[0-9]+(?:\.[0-9]+[a-z]?)?".to_string(),
            doc: "[a-z0-9-]+".to_string(),
            files: vec!["docs".to_string()],
            doc_suffix: "-design".to_string(),
            definition: r"^- \*\*(?:FR|NFR|C)-".to_string(),
            verification: r"^\| *(?:[a-z0-9-]+#)?VC-".to_string(),
            methods: vec![
                "test".to_string(),
                "analysis".to_string(),
                "inspection".to_string(),
                "review".to_string(),
            ],
            marked_methods: vec!["test".to_string()],
            retired: vec![],
            exclude_phrases: vec![],
            markers: None,
        }
    }

    fn row(id: &str, kind: &str, text: &str) -> Row {
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: kind.to_string(),
            parents: Vec::new(),
            method: None,
            file: "docs/w-design.md".to_string(),
            line: 1,
            text: text.to_string(),
        }
    }

    fn condition_row(id: &str, parent: &str, method: &str, text: &str) -> Row {
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: CONDITION.to_string(),
            parents: vec![parent.to_string()],
            method: Some(method.to_string()),
            file: "docs/w-design.md".to_string(),
            line: 1,
            text: text.to_string(),
        }
    }

    #[test]
    fn marked_test_condition_is_covered() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let mark_text = format!("{PFX}\"w#VC-1.1\")]");
        let marks = [row("w#VC-1.1", MARKER, &mark_text)];
        let config = default_config();
        let (report, defects) = derive(&reqs, &marks, &config);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Covered)
        );
        assert!(report.passes());
        assert!(defects.is_empty());
        assert_eq!(report.counts.get(&Status::Covered), Some(&1));
    }

    #[test]
    fn unmarked_test_condition_is_uncovered_and_fails() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let config = default_config();
        let (report, defects) = derive(&reqs, &[], &config);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Uncovered)
        );
        assert!(!report.passes());
        assert_eq!(defects.len(), 1);
        assert_eq!(
            defects.first().map(|d| d.message.as_str()),
            Some("w#VC-1.1 has no marked test")
        );
        assert_eq!(report.counts.get(&Status::Uncovered), Some(&1));
    }

    #[test]
    fn review_condition_takes_review_status() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "review",
                "| VC-1.1 | FR-1 | `review` | Manual inspection |",
            ),
        ];
        let config = default_config();
        let (report, defects) = derive(&reqs, &[], &config);

        assert_eq!(
            report.requirements.first().map(|r| r.status),
            Some(Status::Review)
        );
        assert!(report.passes());
        assert!(defects.is_empty());
        assert_eq!(report.counts.get(&Status::Review), Some(&1));
        assert_eq!(report.review.len(), 1);
    }

    #[test]
    fn marker_naming_requirement_is_a_defect() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let marks = [row("w#FR-1", MARKER, "fn t")];
        let config = default_config();
        let (report, defects) = derive(&reqs, &marks, &config);

        assert!(!report.passes());
        assert!(
            defects
                .iter()
                .any(|d| d.message.contains("names a requirement"))
        );
    }

    #[test]
    fn marker_naming_undefined_condition_is_a_defect() {
        let reqs = vec![
            row("w#FR-1", DEFINITION, "- **FR-1 — A**: Work."),
            condition_row(
                "w#VC-1.1",
                "w#FR-1",
                "test",
                "| VC-1.1 | FR-1 | `test` | Step |",
            ),
        ];
        let marks = [row("w#VC-9.9", MARKER, "fn t")];
        let config = default_config();
        let (report, defects) = derive(&reqs, &marks, &config);

        assert!(!report.passes());
        assert!(
            defects
                .iter()
                .any(|d| { d.message.contains("names no defined condition") })
        );
    }
}
