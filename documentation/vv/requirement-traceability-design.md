# Requirement Traceability Infrastructure (requirement-traceability)

![Date Badge](https://img.shields.io/badge/Date-October_23,_2026-blue)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)
![Design Status Badge](https://img.shields.io/badge/Design%20Status-Ready-brightgreen)

---

## Introduction

Requirements in `control-rs` are written first, in the Requirements section of
each design document, and the design follows from them. This document
specifies the tooling that checks every requirement is decomposed into
verification conditions and that every condition verified by test or formal
proof names the test or proof harness that discharges it and that the harness
passed.

The decomposition follows the vocabulary of MC/DC: a decision is a Boolean
expression composed of conditions and Boolean operators, and a condition
contains no Boolean operator (Hayhurst et al., 2001). A requirement statement
is the decision; its verification conditions are the conditions. Coverage
criteria of this shape can be defined directly on requirements rather than on
code, which keeps the resulting verification traceable to the requirements they
exercise (Whalen et al., 2006). The weakest such criterion asks for at least
one test per requirement (Staats et al., 2010); this design asks for at least
one passing target per automated condition and leaves the stronger criteria open
(see
[Extension Path](#extension-path)).

The tooling rests on three concepts. A **definition** is the one line where a
requirement ID is introduced. A **condition** is a row of a Verification table
that introduces a condition ID, names its parent requirements, states how it is
verified and, for an automated method, names the fully qualified **targets**
(tests or proof harnesses) that discharge it. A condition of the `gate` method
names a quality gate slug (such as `clippy`, `deny`, or `fmt`) and verifies
that the gate completed successfully via its serialized outcome record. A
condition of the `link` method names design documents, requirements or
conditions instead and takes their status. A condition of the `doc-status`
method names a decision record or another traced design document (excluding
the document in which the condition is defined) and takes its recorded status.
Two small binaries in `control-rs-ci` handle one step each:

- `trace-reqs` finds definitions and conditions in Markdown, checks them and
  writes `reqs.jsonl`;
- `trace-check` reads `reqs.jsonl` and the result logs of the gates, and writes
  `trace-report.json` and a coverage verdict.

**Concept of operations**

| Actor                     | Action                                                                           | Touches                                        |
|:--------------------------|:---------------------------------------------------------------------------------|:-----------------------------------------------|
| Author                    | Writes requirements and their verification conditions first, then the design     | One Markdown file, in any editor               |
| Verification author       | Names each test or proof harness in the Target cell of the condition it verifies | The same Markdown file                         |
| Reviewer                  | Reads requirements, conditions and criteria in a pull request                    | Rendered Markdown on GitHub                    |
| Linter (local or CI)      | Reports missing, duplicate or unresolved IDs with `path:line: message`           | The same Markdown file                         |
| Tracer (CI)               | Reports per-target verification status of each condition                         | `reqs.jsonl`, result logs, `trace-report.json` |
| Systems engineer (future) | Builds hierarchies, allocations and verification matrices                        | External tools that read the `.jsonl` files    |

Three principles follow from the table:

1. **Plain text is the source of truth.** A requirement is readable and
   writable without any tool.
2. **The linter is the only tool an author needs.** Editor support comes from
   `path:line: message` diagnostics, which every editor already consumes from a
   terminal, a run configuration or a problem matcher. No plugin, schema,
   snippet or language server is part of the design.
3. **Everything beyond lint and trace is optional and external.** Systems-level
   features consume the generated `.jsonl` files and never enter the CI path of
   this workspace.

**No template.** The tracer assumes no document layout beyond the patterns in
its configuration. `control-rs` and external users of `control-rs-ci` each
state their definition and verification-row grammar there (see
[Configuration](#configuration)).

**Scope.** In scope: definitions, conditions, targets, the checks over them,
the Requirement → Condition → Target hierarchy inside one design document,
target status derivation, decision references (FR-12), links (FR-13), quality
gate verification (`gate`, FR-14), document status verification (`doc-status`),
and the generated artifacts. Out of
scope:
test adequacy; condition, decision and MC/DC coverage of requirements (see
[Extension Path](#extension-path)); code-level structural coverage;
requirement-to-requirement hierarchies; citation, reference and general
document style, which an external, unpublished review tool owns; requirement
editing tools; and any systems-engineering view.

---

## Requirements

### Functional Requirements

- **FR-1 — Definitions**: `trace-reqs` shall record one definition for the
  first requirement ID on each line that matches the `definition` pattern,
  with the definition text formed per [Matching Rules](#matching-rules).
- **FR-2 — Conditions**: `trace-reqs` shall record one condition row for each
  line that matches the `verification` pattern, with the condition ID on the
  line as `id`, every requirement ID outside condition IDs as `parents`, the
  method named in a code span as `method` and the code spans of the `Target`
  cell as `targets`.
- **FR-3 — Qualified IDs**: `trace-reqs` shall qualify every recorded ID as
  `<doc>#<id>` per [Matching Rules](#matching-rules), with `<doc>` extracted by
  the `doc_id` pattern, so that equal IDs in different documents never collide,
  and shall recognize an ID only at word boundaries.
- **FR-4 — Checks**: `trace-reqs` shall report as a defect: a missing document
  ID declaration matching `doc_id`; a document, requirement or condition defined
  more than once; a condition with no parent or with an undefined parent; a
  verification row with no condition ID or with more than one; a condition row
  with no method, with more than one, with one outside the method set of
  [Methods](#methods), with no target, or a `doc-status` condition whose target
  names its own document; a defined requirement with no condition; and a retired
  ID that is defined or referenced.
- **FR-6 — Requirement Rows**: `trace-reqs` shall write `reqs.jsonl` with one
  row per definition and per condition, in the row schema of
  [Artifacts](#artifacts).
- **FR-8 — Coverage Verdict**: `trace-check` shall derive condition and
  requirement status per [Status Derivation](#status-derivation), write
  `trace-report.json` and exit 0 iff no condition is `Uncovered`, `Unrun` or
  `Fail`.
- **FR-9 — Diagnostic Format**: `trace-reqs` and `trace-check`
  shall report each defect as one `path:line: message` line and shall exit
  non-zero when any defect exists.
- **FR-10 — File Selection**: `trace-reqs` shall select its input files per
  [File Selection](#file-selection).
- **FR-11 — Targets**: `trace-check` shall resolve each target of a condition
  whose method is automated against the result log or outcome record of that
  method, and shall report a target that is absent from the log or directory,
  failed, or ambiguous (for example conflicting outcomes across repeats) as a
  defect at the condition row.
- **FR-12 — Decision References**: When the configuration has a `[decisions]`
  table, `trace-reqs` shall record the ID and status of each decision record
  it selects, and shall report as a defect: a decision record with no status
  or defined more than once; and a decision ID cited in the text of a
  requirement definition that no decision record defines, or whose status is
  not in `accepted` (ADR-0001).

- **FR-13 — Links**: For a condition whose method is `link`,
  `trace-reqs` shall resolve each target as a design document, a requirement or
  a condition per [Matching Rules](#matching-rules), and shall report as a
  defect a target that names nothing defined, a document
  that defines no requirement and a condition that lies on a link cycle.
  `trace-check` shall give the condition the status of what it names per
  [Status Derivation](#status-derivation), using the graph of
  `tracer-link-graph-design.md`.
- **FR-14 — Quality Gate References**: For a condition whose method is `gate`,
  `trace-check` shall resolve each target as a quality gate slug against
  `<result_dir>/<gate>.result.json` using `GateOutcome::load_from_file`, and
  shall report as a defect: a target naming a gate with no outcome artifact, an
  outcome record with an unsupported schema or unreadable format, or an outcome
  record whose verdict is not `Pass` (such as `Fail` or `Skipped`).

### Non-Functional Requirements

- **NFR-1 — Audit Latency**: Each of the two binaries shall complete on the
  `control-rs` workspace in under 1 s of wall-clock time, excluding compilation.
- **NFR-3 — Dependency Budget**: The two binaries shall add no dependency to
  `control-rs-ci` beyond its current set, workspace crates
  `control-rs-compare`, and the `regex` crate.
- **NFR-4 — Schema Versioning**: Every row and `trace-report.json` shall carry
  an integer `schema` field, incremented on every incompatible change.

### Constraints

- **C-1 — Plain-Text Source**: Requirements and their verification shall live in
  the Markdown files the configuration lists; no generated file shall be
  committed or required for reading or writing a requirement.
- **C-2 — No Editor Integration**: The design shall deliver diagnostics only; it
  shall ship no editor plugin, schema, snippet or language server.
- **C-3 — Paths from Configuration**: Every input and output path of the two
  binaries shall arrive from a command-line argument or the configuration file.
- **C-4 — Read-Only Inputs**: The tracer shall run no test, build or gate; it
  reads Markdown, the configuration, its own artifacts and per-target test and
  proof result logs from `target/ci-artifacts/`.
- **C-5 — External Systems Tooling**: Hierarchy views, allocation, verification
  matrices and model export shall consume the `.jsonl` files outside this
  workspace's CI path.
- **C-6 — Artifact Location**: The two binaries shall write under
  `target/ci-artifacts/` when run as gates.
- **C-7 — Document Style Exclusion**: `trace-reqs` shall check no citation,
  reference or general document style.
- **C-8 — Per-Target Verification Status**: The tracer shall derive condition
  status from per-target test and proof results; it shall never infer condition
  status from aggregate gate verdicts.

---

## Technical Overview

### Author Workflow

1. **Define the requirement** in the Requirements section:

   ```markdown
   - **FR-3 — Packed Storage**: Packed symmetric, Hermitian and triangular storage
     shall return `None` from `value(i, j)` when `i` or `j` is out of bounds.
   ```

2. **Decompose it into conditions** in the Verification table. Each row
   introduces one condition, names its parent requirements, gives one method
   and names the target that discharges it. The Criterion of the first condition
   states how the conditions combine into the requirement:

   ```markdown
   | Condition | Requirement | Method    | Target                                                          | Criterion                                                      |
   |:----------|:------------|:----------|:----------------------------------------------------------------|:---------------------------------------------------------------|
   | VC-3.1    | FR-3        | `libtest` | `control_rs::math::storage::tests::packed_value_out_of_bounds_is_none` | Row index out of bounds gives `None`; FR-3 holds iff both hold |
   | VC-3.2    | FR-3        | `libtest` | `control_rs::math::storage::tests::packed_value_out_of_bounds_is_none` | Column index out of bounds gives `None`                        |
   | VC-3.3    | FR-3        | `kani`    | `control_rs::math::storage::proofs::prove_saturating_div_no_panic`     | Saturating division never panics over the valid input domain   |
   ```

3. **Name the targets.** The `Method` states the framework and the `Target`
   cell holds the fully qualified test or harness path in code spans, as the
   result log prints it. Conditions may share a target, and one
   condition may name multiple:

   | Method            | Target format                                  | Result log (`target/ci-artifacts/`)                                      |
            |:------------------|:-----------------------------------------------|:-------------------------------------------------------------------------|
   | `libtest`         | `crate::module::tests::test_name`              | `test.log` (`test <path> ... ok`)                                        |
   | `libtest` doctest | `crate::<file> - <item>`                       | `test.log` (`test <file> - <item> (line N) ... ok`)                      |
   | `kani`            | `crate::module::proofs::harness_name`          | `kani.log` (`Checking harness <path>...`)                                |
   | `compare`         | `suite`, `suite::signal`, `signal`, or `key`   | `cross-val-report.json`                                                  |
   | `junit`           | `name`, `classname::name`, or `classname.name` | `junit.xml` (`<testcase ...>`)                                           |
   | `gate`            | `<gate-slug>` (for example `clippy`, `deny`, `fmt`)   | `<gate>.result.json` (`GateOutcome`)                                     |
   | `link`            | `<doc>`, `<doc>#<id>`                          | none; resolved against traced requirements/documents                     |
   | `doc-status`      | `ADR-NNNN`, `<doc>`, `roadmap#<item>`          | none; resolved from decision records or design docs (cannot be own doc) |


4. **Run `cargo trace-reqs` and `cargo trace-check`.** Each problem prints as
   `path:line: message`, which the editor turns into a clickable link:

   ```text
   documentation/math/storage-design.md:212: condition storage#VC-3.2 target 'control_rs::math::storage::tests::column_out_of_bounds' not found in test.log
   ```

To withdraw a requirement or condition, delete it and add its qualified ID to
`retired` in the configuration. The ID is never reused.

### Reading the Output

Every line of `reqs.jsonl` is one occurrence of one ID:

```json lines
{
  "schema": 5,
  "id": "storage#FR-3",
  "kind": "definition",
  "file": "documentation/math/storage-design.md",
  "line": 42,
  "text": "- **FR-3 — Packed Storage**: Packed symmetric ... out of bounds."
}
{
  "schema": 5,
  "id": "storage#VC-3.1",
  "kind": "condition",
  "parents": [
    "storage#FR-3"
  ],
  "method": "libtest",
  "targets": [
    "control_rs::math::storage::tests::packed_value_out_of_bounds_is_none"
  ],
  "file": "documentation/math/storage-design.md",
  "line": 211,
  "text": "| VC-3.1 | FR-3 | `libtest` | `control_rs::math::storage::tests::packed_value_out_of_bounds_is_none` | Row index out of bounds gives `None`; FR-3 holds iff both hold |"
}
```

Grouping rows by `id` gives everything known about a condition; following
`parents` gives its requirements. No field depends on configuration to be
understood.

### Data Flow

```mermaid
flowchart LR
    Cfg["trace.toml"] --> TR
    Cfg --> TC
    Docs["Markdown files"] --> TR["trace-reqs"]
    Dec["decision records<br/><i>optional</i>"] --> TR
    TR --> RJ["reqs.jsonl<br/><i>condition, method, targets</i>"]
    RJ --> TC["trace-check"]
    Logs["target/ci-artifacts/<br/>test, kani, compare, junit, gate"] --> TC
    Docs -.->|" doc-status "| TC
    Dec -.->|" doc-status "| TC
    TC --> Rep["trace-report.json"]
    TC --> Exit["exit status"]
    RJ -.-> Ext["external systems tool<br/><i>outside CI (C-5)</i>"]
```

### Traceability Model

```mermaid
flowchart LR
    R["Requirement (decision)<br/>FR-n, NFR-n, C-n"] -->|" 1..n "| V["Verification condition<br/>VC-x.y, method"]
    V -->|" 1..n "| T["Target<br/>test, harness, document or decision"]
```

A condition may have more than one parent, such as one `doc-status` condition
covering
every constraint. Verification outcomes are derived from per-target results: a
target evaluates to a boolean pass or fail in its result log, and `trace-check`
aggregates target outcomes into condition statuses (C-8).

**Where requirements live.** Requirements stay in the design documents.
Measured on 2026-09-26, the corpus holds 268 requirements in 23
`*-design.md` files, 244 of them in 20 Approved documents. Keeping them in place
needs no migration, keeps each requirement next to its rationale and renders on
GitHub. The `-design.md` suffix already serves as a file-type pattern for editor
rules (`files.associations` in VS Code, file-type patterns in JetBrains IDEs).

---

## Architecture

### Configuration

The method set is fixed in [Methods](#methods). The configuration does not
list methods.

Both binaries read one TOML file named by `--config`. The `control-rs`
configuration:

```toml
id = '(?:FR|NFR|C)-[0-9]+[a-z]?'
condition = 'VC-[0-9]+(?:\.[0-9]+[a-z]?)?'
doc = '[a-z0-9-]+'
files = [
    "documentation/vv/requirement-traceability-design.md",
    "documentation/vv/tracer-link-graph-design.md",
    "documentation/math/fixed-num-design.md",
    "documentation/math/num-traits-design.md",
    "documentation/control-toolboxes/classical-control-design.md",
]
doc_id = '^#\s+.*\((?P<doc>[a-z0-9-]+)\)'
definition = '^- \*\*(?:FR|NFR|C)-'
verification = '^\| *(?:[a-z0-9-]+#)?VC-'
retired = [
    "requirement-traceability#FR-5",
    "requirement-traceability#VC-5.1",
    "requirement-traceability#FR-7",
    "requirement-traceability#VC-7.1",
    "requirement-traceability#VC-7.2",
    "requirement-traceability#NFR-2",
    "requirement-traceability#VC-12.1",
]

[method.libtest]
result_artifact = "target/ci-artifacts/test.log"
interpreter_artifact = "target/ci-artifacts/miri.log"

[method.kani]
result_artifact = "target/ci-artifacts/kani.log"

[method.compare]
result_artifact = "target/ci-artifacts/cross-val-report.json"

[method.junit]
result_artifact = "target/ci-artifacts/junit.xml"

[method.gate]
result_dir = "target/ci-artifacts"

[decisions]
files = ["documentation/adr"]
id = 'ADR-[0-9]{4}'
definition = '^#\s+ADR-[0-9]{4}:'
status = 'ADR%20Status-(?P<status>[A-Za-z]+)-'
accepted = ["Accepted"]
```

| Key                                   | Meaning                                                                                                                                                                            |
|:--------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `id`                                  | Regex for one requirement ID                                                                                                                                                       |
| `condition`                           | Regex for one condition ID; optional (defaults to `VC-[0-9]+(?:\.[0-9]+[a-z]?)?`)                                                                                                  |
| `doc`                                 | Regex for the document name in a qualified ID `<doc>#<id>`                                                                                                                         |
| `files`                               | Markdown files, and directories whose `.md` files are read (see [File Selection](#file-selection))                                                                                 |
| `doc_id`                              | Regex for the document ID declaration; optional (defaults to `^#\s+.*\((?P<doc>[a-z0-9-]+)\)`)                                                                                     |
| `definition`                          | Regex for a line that defines a requirement                                                                                                                                        |
| `verification`                        | Regex for a line that defines a condition                                                                                                                                          |
| `method.<m>`                          | Result artifact path for method `<m>`; a `<m>` that names no method of `METHODS` is a configuration error, and a condition of a result-artifact method with no artifact is `Unrun` |
| `method.gate.result_dir`              | Directory where gate outcome files `<gate>.result.json` reside; optional (defaults to `target/ci-artifacts`)                                                                       |
| `method.libtest.interpreter_artifact` | Log of the same tests run under an interpreter; optional, and absent it disables W-2                                                                                               |
| `retired`                             | Qualified IDs that must not be defined or referenced again                                                                                                                         |
| `decisions.files`                     | Roots of decision records, selected per [File Selection](#file-selection); the table is optional and its absence disables FR-12                                                    |
| `decisions.id`                        | Regex for one decision ID                                                                                                                                                          |
| `decisions.definition`                | Regex for the line that defines a decision record's ID                                                                                                                             |
| `decisions.status`                    | Regex whose `<status>` capture is a decision record's status                                                                                                                       |
| `decisions.accepted`                  | Decision statuses a requirement may cite                                                                                                                                           |

### Methods

Serves: FR-4, FR-8, FR-11, FR-13, FR-14

Each method is one `TraceCheckMethod` instance in `METHODS`, in
`control_rs_ci::trace::status`. `METHODS` is the only list of method names:
`trace-reqs` checks a row's method against it and `trace-check` calls the
method's `check`.

```rust
/// A verification method a condition row may name.
pub(crate) struct TraceCheckMethod {
    /// The method as a condition row writes it.
    pub name: &'static str,
    /// The status of a condition row of this method.
    pub check: fn(&Row, &mut StatusCtx<'_>) -> ConditionStatus,
}

pub(crate) const METHODS: [TraceCheckMethod; 7] = [/* libtest, kani, compare, junit, gate, link, doc-status */];
```

- `libtest`, `kani`, `compare` and `junit` look each target up in the result
  artifact that `[method.<name>]` names. Target matching for `libtest` and
  `kani` drops the leading crate segment of a target (`strip: true`). `compare`
  decodes `cross-val-report.json` using
  `control_rs_compare::report::ValidationReport`.
- `gate` resolves each target as a gate slug against
  `<result_dir>/<target>.result.json`,
  deserializing the typed `GateOutcome` via
  `control_rs_ci::gate::GateOutcome::load_from_file`.
- `doc-status` reads a document target's status from the file the rows give
  for that document, as the first match of `DocumentMetaPatterns` (the fixed
  pattern `(?:Design|Doc)%20Status-(?P<status>[A-Za-z_]+)-`), and a decision
  target's status with `[decisions]`. A `doc-status` condition cannot target its
  own document. Its targets add no link edge.
- `link` returns the status that `LinkGraph::derive` gave the row; `derive`
  takes the `check` of each other row's method as `leaf`.

### File Selection

`files` lists roots relative to the working directory.

- A root that is a file is selected.
- A root that is a directory selects every file below it whose name ends in a
  `.md` suffix.
- Symbolic links are never followed, so a walk cannot loop.
- A root that is missing, or is neither a file nor a directory, is an error.

### Matching Rules

The rules below are the whole parser:

1. Lines inside fenced code blocks, including indented fences, are skipped,
   so examples are inert.
2. An ID occurrence is a match of `id` or `condition` bounded by word
   boundaries, optionally preceded by a match of `doc` and `#`. A requirement
   ID inside a condition ID, or inside a longer token such as `IEC-61508`, is
   not an occurrence.
3. A line that matches `definition` defines the first requirement ID on it.
   Its text is the line plus the indented lines that follow it.
4. A line that matches `verification` defines the condition ID on it. Every
   requirement ID on the line is a parent, the one code span whose content
   is in the method set is the method and the code spans of the fourth cell of
   a row
   with five or more cells are the targets.
5. The document name is the `<doc>` captured by the `doc_id` pattern on the
   first matching line: `# Title (storage)` yields `storage`. An ID written
   without `<doc>#` belongs to the document it appears in. A document with no
   line matching `doc_id` is a defect.
6. A decision record is a file selected by `decisions.files` with a line
   matching `decisions.definition`; that line defines the decision ID on it,
   and the `<status>` of the first `decisions.status` match is its status. A
   selected file with no such line is not a decision record. Every
   occurrence of `decisions.id` in the text of a requirement definition (rule 3)
   is a citation; an occurrence anywhere else is not.
7. A target of a `link` condition is a document
   name `<doc>` or a qualified ID `<doc>#<id>` of a requirement or a condition.
   A target is not a parent: the IDs in the `Target` cell of such a row are
   excluded from the parents of rule 4.
8. A target of a `doc-status` condition is a decision ID or a document name
   `<doc>` other than the document defining the condition. A condition naming
   its own document is a defect.
9. A target of a `gate` condition is a quality gate slug corresponding to a
   gate outcome record `<target>.result.json`.

### Checks

| Check             | Defect                                                                                                                                               |
|:------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `doc_id`          | A document contains no line matching `doc_id`, or multiple conflicting declarations                                                                  |
| `duplicate`       | A requirement, condition, document or decision ID is defined more than once                                                                          |
| `unresolved`      | A condition names a parent that is not defined                                                                                                       |
| `orphan`          | A condition names no parent                                                                                                                          |
| `row`             | A verification row names no condition ID, or more than one                                                                                           |
| `method`          | A condition names no method, more than one, or an unknown method                                                                                     |
| `target`          | A condition names no target                                                                                                                          |
| `self_target`     | A `doc-status` condition names the document in which it is defined                                                                                   |
| `missing`         | A defined requirement has no condition                                                                                                               |
| `retired`         | A retired ID is defined or referenced                                                                                                                |
| `decision`        | A requirement cites a decision ID that has no decision record                                                                                        |
| `decision_status` | A decision record has no status, or a requirement cites a decision whose status is not in `accepted`                                                 |
| `link`            | A link condition names a target that is no document, requirement or condition, names a document that defines no requirement, or lies on a link cycle |

### Status Derivation

Condition status is derived from the targets of the condition and the result
logs in `target/ci-artifacts/`:

| Condition                                                             | Status                        | Fails gate           |
|:----------------------------------------------------------------------|:------------------------------|:---------------------|
| Any method, no target named                                           | `Uncovered`                   | yes                  |
| `Artifact` method, every target recorded as passed                    | `Pass`                        | no                   |
| `Artifact` method, any target recorded as failed                      | `Fail`                        | yes                  |
| `Artifact` method, any target recorded with ambiguous outcomes        | `Fail`                        | yes                  |
| `Artifact` method, any target absent from its log                     | `Unrun`                       | yes                  |
| Method `gate`, target outcome artifact missing                        | `Unrun`                       | yes                  |
| Method `gate`, target outcome verdict is `Pass`                       | `Pass`                        | no                   |
| Method `gate`, target outcome verdict is `Fail` or `Skipped`          | `Fail`                        | yes                  |
| Method `gate`, target outcome unreadable or invalid schema            | `Fail`                        | yes                  |
| Method `link`, a target naming nothing defined                        | `Unrun`                       | yes                  |
| Method `link`, condition on a link cycle                              | `Fail`                        | yes                  |
| Method `link`, condition not in the link graph                        | `Fail`                        | yes                  |
| Method `link`, otherwise                                              | Worst status of what it names | iff that status does |
| Method `doc-status`, condition targets its own document               | `Fail`                        | yes                  |
| Method `doc-status`, a target naming no traced document or decision   | `Unrun`                       | yes                  |
| Method `doc-status`, decision target with a status in `accepted`      | `Pass`                        | no                   |
| Method `doc-status`, document target with status `Ready` or `Shipped` | `Pass`                        | no                   |
| Method `doc-status`, document target with status `Draft`              | `Pass`, warning W-4           | no                   |
| Method `doc-status`, any other status or none                         | `Fail`                        | yes                  |
| Method `doc-status`, multiple targets                                 | Worst status among targets    | iff that status does |

`trace-check` reads per-target results using deterministic framework-specific
rules. A target matches a logged identifier exactly:

- **`libtest` (`test.log`)**: A line `test <path> ... ok` maps to a passing
  target; `test <path> ... FAILED` maps to a failed target. The identifier is
  `<binary>::<path>`, the binary taken from the preceding `Running` or
  `Doc-tests` header, so the first segment of a target is the test binary
  (`control_rs`, `trace_tests`). A log without headers records bare paths,
  which a target also matches after its first segment is dropped. A doctest
  is identified by its file and item, without the `(line N)` that changes on
  every edit. A path recorded with different outcomes, or two doctests
  on one item, is ambiguous: the target is a `Fail` defect.
- **`kani` (`kani.log`)**: A harness `Checking harness <path>...` maps to a
  passing target iff Kani reports `VERIFICATION:- SUCCESSFUL` and all evaluated
  cover properties are `SATISFIED`. The target may match with or without the
  leading crate path segment (`strip: true`). An assertion failure, unwinding
  failure or unsatisfiable cover witness maps to a failed target. A harness
  recorded with different outcomes is ambiguous: the target is a `Fail`
  defect.
- **`compare` (`cross-val-report.json`)**: A cross-validation comparison report
  JSON deserialized via `control_rs_compare::report::ValidationReport` maps
  suite names, `<suite>::<signal>`, `<suite>::<key>`, signal names, and
  comparison keys to passing targets when the suite status is
  `SuiteStatus::Pass`
  or the comparison verdict is `ComparisonVerdict::Pass`. Discrepancies or
  failures map to a failed target. Conflicting outcomes across repeats are
  ambiguous: the target is a `Fail` defect.
- **`junit` (`junit.xml`)**: A JUnit XML report maps test case names,
  `<classname>::<name>`, and `<classname>.<name>` to passing targets when the
  `<testcase>` element contains no `<failure>` or `<error>` child elements. A
  test case with failure or error maps to a failed target. Skipped test cases
  are omitted (remaining `Unrun`). Conflicting outcomes across repeats are
  ambiguous: the target is a `Fail` defect.
  Attributes on `<testcase>` (`name`, `classname`) are parsed via the
  `quick-xml` reader so that attribute order, XML entity decoding, multi-line
  tags and whitespace are handled according to XML token standards. Test case
  close tags (`</testcase>`), comments (`<!-- ... -->`), and judgment elements
  (`<failure`, `<error`, `<skipped`) are handled by reader event tokens outside
  `<![CDATA[...]]>` blocks, preventing literal tags in embedded log streams
  (`<system-out>`, `<system-err>`) from closing the case early or falsifying
  case
  outcome.
- **`gate` (`<gate>.result.json`)**: A gate slug maps to a passing target iff
  `<result_dir>/<gate>.result.json` exists, deserializes into a valid
  `GateOutcome` record via `GateOutcome::load_from_file`, and its `verdict` is
  `Verdict::Pass`. A verdict of `Verdict::Fail` or `Verdict::Skipped`, or an
  invalid or unreadable record, maps to a failed target. A missing result file
  maps to `Unrun`.

Only `libtest` drops a target segment. A `compare` or `junit` target that names
another suite or class than the one logged is absent, even when its bare
signal, key or name is logged.

**Draft warning.** `trace-check` emits the non-fatal warning **W-4 (Draft
document)** when a `doc-status` target is a document whose status is `Draft`.
The warning points at the condition row and names the document.

**Interpreter warning.** Miri serves as a secondary interpreter for test
execution rather than a distinct verification method. `trace-check` emits the
non-fatal warning **W-2 (Unexecuted under Miri)** when a `libtest` target
passed in `test.log` is absent from the `interpreter_artifact` log, provided
that log ran the target's test binary. Doctest targets are exempt.

**Staleness warning.** `trace-check` emits the non-fatal warning **W-3 (Stale
result log)** when the result log of an automated method is older than the
most recently modified document that `reqs.jsonl` names. A document edited
after the run may name targets the log never ran. The warning points at that
document and names the log.

A requirement takes the worst status of its conditions, in the order
`Fail`, `Unrun`, `Uncovered`, `Pass`. Each defect prints
at the
condition row, as `condition <id> target '<target>' not found in <log>` or
`... failed in <log>`.

**Links.** A link condition takes the worst status of the documents,
requirements and conditions it names, a document taking the worst status of its
requirements. The graph, its cycle detection and its evaluation order are
specified in `tracer-link-graph-design.md`.
The per-status counts include link conditions.
Defects print at the condition row, as
`condition <id> target '<target>' names no selected document, requirement or
condition`, `... defines no requirement` or `condition <id> lies on a link
cycle`.

`trace-report.json` holds `schema` (5), summary counts per status, one entry
per requirement (ID, status and child conditions with method, status and
target outcomes) and the warnings W-2, W-3 and W-4. `Status` has the variants
`Pass`, `Uncovered`, `Unrun` and `Fail`; no status awaits sign-off or a later
phase.

### Artifacts

| Field     | Meaning                                                                                                                      |
|:----------|:-----------------------------------------------------------------------------------------------------------------------------|
| `schema`  | Row format version, `5` (NFR-4)                                                                                              |
| `id`      | Qualified ID, `<doc>#<id>`                                                                                                   |
| `kind`    | `definition` or `condition`                                                                                                  |
| `parents` | Condition rows only: qualified parent requirement IDs                                                                        |
| `method`  | Condition rows only: the verification method                                                                                 |
| `targets` | Condition rows only: the code spans of the `Target` cell, if any; documents, requirements or conditions for a link condition |
| `file`    | Path relative to the working directory                                                                                       |
| `line`    | 1-based line number                                                                                                          |
| `text`    | The matched line; for a definition, its full text                                                                            |

In `text`, each run of whitespace, line breaks included, becomes one space, and
none remains at either end. A wrapped definition therefore reads as one line.

Rows are sorted by `file`, then `line`. Examples are in
[Reading the Output](#reading-the-output).

### Invocation and Gate Wiring

| Binary        | Gate and placement              | Arguments                                                                                                                                                  |
|:--------------|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `trace-reqs`  | `trace-reqs`, `lint` group      | `--config .cargo/trace/trace.toml --out target/ci-artifacts/reqs.jsonl`                                                                                    |
| `trace-check` | `trace`, `post` exclusive stage | `--config .cargo/trace/trace.toml --reqs target/ci-artifacts/reqs.jsonl --out target/ci-artifacts/trace-report.json --graph target/ci-artifacts/links.mmd` |

`trace-check` also accepts an optional `--graph <path>` that writes the link
graph of `tracer-link-graph-design.md` as Mermaid (`.mmd`) or Graphviz DOT
(`.dot`); any other extension is a usage error. The gate passes
`--graph target/ci-artifacts/links.mmd`.

`trace-reqs` runs early in the concurrent `lint` group to extract requirement
definitions and targets. `trace-check` runs in the `post` exclusive stage after
test execution and `kani` formal verification complete, ensuring all result logs
are present in `target/ci-artifacts/` before condition status derivation.

### Adoption

Existing verification tables carry no condition IDs or methods, so every
existing requirement would report a `missing` defect. `files` is therefore
the adoption mechanism: it lists documents one by one as each is migrated and
becomes the directory `documentation` when the last one is done. A document
outside `files` is not checked at all; a document inside it is checked in full.

### Extension Path

Coverage criteria defined on requirements form a ladder. Requirements coverage
asks for one test per requirement that causes it to be met (Staats et al.,
2010). Unique-first-cause coverage asks that every basic condition take all
outcomes and be shown to affect the outcome of its formula independently, and
is adapted from MC/DC (Staats et al., 2010). MC/DC over a decision with n
inputs needs, in general, at least n + 1 test cases (Hayhurst et al., 2001).

| Level              | Obligation                                   | This revision                      |
|:-------------------|:---------------------------------------------|:-----------------------------------|
| Linked             | One passing target per automated condition   | Enforced                           |
| Condition coverage | A target driving each condition each way     | Not representable in a Target cell |
| Decision coverage  | The requirement observed both true and false | Not representable in a Target cell |
| Unique cause       | Independence pairs per condition             | Review of the Criterion            |

Four choices keep the stronger levels additive:

1. A later per-target outcome tag is a schema increment (NFR-4) that leaves the
   `Target` cell grammar intact.
2. Condition IDs are never reused (`retired`), so later tags attach to stable
   IDs.
3. The `.jsonl` files stay the interface for external tools (C-5).
4. The Criterion states how the conditions combine into the requirement, so a
   reviewer can check independence pairs now.

---

## Alternatives

| Alternative                                                           | Rejected Because                                                                                                                                                                                                                                                                                                  | Reference                      |
|:----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------|
| **Requirement status from gate verdicts**                             | A gate verdict aggregates many tests; its failure says nothing about one requirement, and reading results ties the tracer to gate order and to which gates a trigger ran.                                                                                                                                         | n/a                            |
| **Direct requirement-to-test links**                                  | Skips the condition decomposition; one link on a requirement would satisfy every way the requirement can hold or fail.                                                                                                                                                                                            | Whalen et al. (2006)           |
| **Condition coverage through outcome-tagged targets**                 | Deferred: buys condition coverage but not independence, at least doubles the targets per condition during migration, and fixes a model before the corpus is decomposed.                                                                                                                                           | Staats et al. (2010)           |
| **Mechanical MC/DC over requirements**                                | Needs every condition's value in every test, from instrumentation or execution traces; a text tracer cannot observe it (C-4).                                                                                                                                                                                     | Hayhurst et al. (2001)         |
| **Code-level MC/DC in the tracer**                                    | Measures code structure, not requirements, and the unstable MC/DC instrumentation in `rustc` has been removed; code-level MC/DC belongs to a separate gate.                                                                                                                                                       | Rust Project Developers (2025) |
| **Test-level verdicts from libtest JSON**                             | The format is unstable and requires `-Z unstable-options`; test outcomes belong to the gates (C-8).                                                                                                                                                                                                               | Rust Project Developers (2026) |
| **Named captures, requirement blocks and link attachment**            | Meaning hid in capture-group names and position-dependent parsing state; the generated rows mirrored parser internals and could not be read without the configuration.                                                                                                                                            | n/a                            |
| **Fixed grammar with configured section, table and column names**     | Many keys with distinct semantics, still bound to headings and pipe tables.                                                                                                                                                                                                                                       | n/a                            |
| **Adoption baseline file and previous-artifact ID check**             | Two baselines with different formats and a workflow restore step; listing migrated documents in `files` ratchets adoption with no new concept, and `retired` covers ID reuse.                                                                                                                                     | n/a                            |
| **Structured JSON or TOML requirement files**                         | Each step toward better editing (per-area schemas, snippets, IDE-specific workarounds) added machinery an author must understand before writing a requirement; TOML adds worse multi-line text.                                                                                                                   | n/a                            |
| **Source markers (attribute or comment) linking tests to conditions** | A second copy of the link in source text, found by scanning, that the table does not show; it needs a scanner, an item-rule check and, for attributes, a proc-macro crate, and drifts from the Verification table. A qualified path in the table is read beside the condition and checked against the result log. | Hatzl (2026b)                  |
| **Globs with `!` exclusions and `depth`**                             | A glob engine and a symbolic-link rule table to select a handful of directories; roots plus a suffix select the same files.                                                                                                                                                                                       | n/a                            |
| **Hand-written matcher instead of the `regex` crate**                 | Users would learn a bespoke pattern language; regular expressions are already known. The cost is one dependency (NFR-3).                                                                                                                                                                                          | n/a                            |
| **Requirement prose and smell linting in the tracer**                 | Requirement prose quality belongs to Vale and compiler linting to Clippy; the tracer focuses strictly on traceability.                                                                                                                                                                                            | n/a                            |
| **Editor plugins, schemas or a language server**                      | Adds per-editor machinery; `path:line: message` works everywhere (C-2).                                                                                                                                                                                                                                           | n/a                            |
| **OpenFastTrace (Java)**                                              | Adds a Java runtime to CI.                                                                                                                                                                                                                                                                                        | OpenFastTrace (2026a, 2026b)   |
| **StrictDoc (Python)**                                                | Adds a Python toolchain and its own `.sdoc` grammar.                                                                                                                                                                                                                                                              | StrictDoc (2026)               |
| **mantra (Rust)**                                                     | Relies on a database and line-coverage pairing rather than conditions.                                                                                                                                                                                                                                            | Hatzl (2026a)                  |
| **Doorstop**                                                          | One YAML file per requirement; loses requirements-beside-rationale.                                                                                                                                                                                                                                               | Doorstop (2026)                |
| **`doc-status` targets as link edges**                                | A recorded status is not a verification rollup, and a row naming its own document would lie on a link cycle; a leaf that reads the document's status needs no edge.                                                                                                                                               | n/a                            |
| **A non-failing pending status (`Review`, `Deferred`)**               | Lets a condition rest unverified with a green gate; a `Draft` document passes with W-4 in view, and every other unaccepted status fails.                                                                                                                                                                          | n/a                            |
| **Immutability of accepted decision records**                         | Needs the base branch as input, outside C-4; review of the diff is the control, as for retired IDs.                                                                                                                                                                                                               | n/a                            |
| **Requirement lists in decision records**                             | A second, hand-kept link in the other direction; citations in the documents already give it, and a search recovers it.                                                                                                                                                                                            | n/a                            |
| **Hard-coding quality gate methods in the tracer**                    | Adding bespoke methods for each gate (for example, `clippy`, `deny`, `fmt`) hard-codes gate names into the tracer and requires code changes whenever gates change; a generic `gate` method checking `<gate>.result.json` handles any gate flexibly and with scalability.                                          | n/a                            |
| **Self-targeting `doc-status` conditions**                            | Allowing conditions in a document to target their own document ID creates circular verification, causing all such conditions to fail whenever the document is in `Draft` or `Stale`; conditions must verify external documents, decisions, or quality gates.                                                      | n/a                            |

---

## Verification & Validation

### Verification

| Condition | Requirement                            | Method    | Target                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Criterion                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|:----------|:---------------------------------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VC-1.1    | FR-1                                   | `libtest` | `control_rs_ci::trace::reqs::tests::definitions_take_the_first_id_and_indented_lines`, `control_rs_ci::trace::reqs::tests::rows_carry_normalized_text`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Definition fixtures yield exactly the expected rows, with normalized text and indented continuation lines; FR-1 holds iff both hold                                                                                                                                                                                                                                                                                                            |
| VC-2.1    | FR-2                                   | `libtest` | `control_rs_ci::trace::reqs::tests::reference_lines_record_every_local_and_qualified_id`, `control_rs_ci::trace::reqs::tests::a_parent_named_twice_is_recorded_once`, `control_rs_ci::trace::reqs::tests::scan_condition_finds_parents`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | A condition row yields one row with every parent and its method; a row with more than one parent yields one row                                                                                                                                                                                                                                                                                                                                |
| VC-3.1    | FR-3                                   | `libtest` | `control_rs_ci::trace::reqs::tests::doc_id_is_extracted_from_the_title_declaration`, `control_rs_ci::trace::reqs::tests::ids_inside_longer_tokens_or_condition_ids_are_not_parents`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Local and qualified IDs resolve to `<doc>#<id>`; an ID inside a longer token or inside a condition ID is not recorded                                                                                                                                                                                                                                                                                                                          |
| VC-4.1    | FR-4                                   | `libtest` | `control_rs_ci::trace::reqs::tests::row_and_method_defects_are_reported`, `control_rs_ci::trace::reqs::tests::duplicate_definition_is_reported_once_at_the_second`, `control_rs_ci::trace::reqs::tests::reference_to_an_undefined_id_is_unresolved`, `control_rs_ci::trace::reqs::tests::definition_without_a_kind_of_reference_is_missing_it`, `control_rs_ci::trace::reqs::tests::retired_ids_are_reported_where_they_appear`, `control_rs_ci::trace::reqs::tests::missing_doc_id_declaration_is_reported_as_defect`, `control_rs_ci::trace::reqs::tests::duplicate_doc_id_across_files_is_reported`, `control_rs_ci::trace::reqs::tests::condition_without_parents_is_an_orphan`, `control_rs_ci::trace::reqs::tests::a_condition_of_any_method_without_a_target_is_reported`, `control_rs_ci::trace::reqs::tests::doc_status_targeting_own_document_is_a_defect`, `control_rs_ci::trace::status::tests::the_method_table_names_each_method_once` | One fixture per defect kind of the Checks table yields exactly that defect; a row of each method with no target gives one `target` defect; `METHODS` holds seven distinct names and `method` finds each                                                                                                                                                                                                                                        |
| VC-6.1    | FR-6                                   | `libtest` | `trace_tests::cli::rows_have_the_schema_fields_and_reruns_are_identical`, `trace_tests::cli::a_clean_document_passes_and_writes_its_rows`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Rows carry the schema fields and two runs on an unchanged corpus produce byte-identical `reqs.jsonl`                                                                                                                                                                                                                                                                                                                                           |
| VC-8.1    | FR-8                                   | `libtest` | `trace_tests::cli::passed_unrun_and_uncovered_conditions_take_their_status`, `trace_tests::cli::every_automated_target_passed_passes_and_doc_status_does_not_fail`, `control_rs_ci::trace::status::tests::automated_condition_without_targets_is_uncovered`, `control_rs_ci::trace::status::tests::missing_log_is_unrun`, `control_rs_ci::trace::status::tests::condition_without_doc_qualifier_still_fails_the_gate`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | A fixture of passed, unexecuted and uncovered conditions gives the specified statuses and exit code; an unqualified condition still fails the gate                                                                                                                                                                                                                                                                                             |
| VC-8.2    | FR-8                                   | `libtest` | `control_rs_ci::trace::status::tests::doc_status_condition_with_accepted_adr_passes`, `control_rs_ci::trace::status::tests::doc_status_passes_on_a_ready_or_shipped_document`, `control_rs_ci::trace::status::tests::doc_status_warns_on_a_draft_document`, `control_rs_ci::trace::status::tests::doc_status_fails_on_any_other_status`, `control_rs_ci::trace::status::tests::doc_status_self_targeting_fails_as_defect`, `control_rs_ci::trace::status::tests::doc_status_reads_decisions_with_the_configured_patterns`, `control_rs_ci::trace::status::tests::doc_status_target_naming_no_traced_document_is_unrun`, `control_rs_ci::trace::link::tests::a_doc_status_condition_adds_no_edge`                                                                                                                                                                                                                                                     | A doc-status condition passes on an accepted decision and a `Ready` or `Shipped` document, passes with one W-4 on a `Draft` document, fails on any other or no status, rejects self-targeting as a defect, reads decision statuses with the `[decisions]` patterns, is `Unrun` for a target naming no traced document or decision, and naming its own document adds no edge and no cycle                                                       |
| VC-9.1    | FR-9                                   | `libtest` | `trace_tests::cli::defects_print_as_path_line_message_and_fail_the_run`, `trace_tests::cli::usage_and_configuration_errors_exit_with_code_two`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Every diagnostic line matches `^[^:]+:[0-9]+: .+$`; any defect gives a non-zero exit                                                                                                                                                                                                                                                                                                                                                           |
| VC-10.1   | FR-10                                  | `libtest` | `trace_tests::cli::roots_select_suffixed_files_below_directories_and_named_files`, `trace_tests::cli::a_missing_root_is_a_configuration_error`, `trace_tests::cli::symbolic_links_are_never_followed`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Selection fixtures: file root, suffix filter, symbolic link and missing root                                                                                                                                                                                                                                                                                                                                                                   |
| VC-11.1   | NFR-1                                  | `gate`    | `trace-reqs`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | The `trace-reqs` quality gate outcome reports `Pass` and completes execution within latency bounds; NFR-1 holds                                                                                                                                                                                                                                                                                                                                |
| VC-13.1   | NFR-3                                  | `gate`    | `deny`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | The `deny` quality gate outcome reports `Pass`, enforcing that dependencies and advisories satisfy workspace budgets; NFR-3 holds                                                                                                                                                                                                                                                                                                              |
| VC-14.1   | NFR-4                                  | `libtest` | `trace_tests::cli::every_row_and_the_report_carry_the_schema_version`, `trace_tests::cli::trace_check_rejects_rows_of_another_schema`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Every row and the report carry `schema` 5; a row of another schema is rejected                                                                                                                                                                                                                                                                                                                                                                 |
| VC-15.1   | C-1, C-2, C-3, C-4, C-5, C-6, C-7, C-8 | `gate`    | `clippy`, `fmt`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Workspace quality gates `clippy` and `fmt` report `Pass`, verifying codebase linting and format constraints without violating architectural bounds; C-1 through C-8 hold                                                                                                                                                                                                                                                                       |
| VC-16.1   | C-4, C-8                               | `kani`    | `control_rs::math::fixed_num::proofs::prove_fixed_saturating_div`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Bounded model checking verifies that the linked proof harnesses satisfy safety and non-vacuity contracts without panic                                                                                                                                                                                                                                                                                                                         |
| VC-17.1   | FR-11                                  | `libtest` | `control_rs_ci::trace::reqs::tests::targets_are_the_code_spans_of_the_fourth_cell`, `control_rs_ci::trace::status::tests::libtest_target_passes_with_crate_qualified_or_bare_name`, `control_rs_ci::trace::status::tests::libtest_target_fails_when_the_log_reports_failed`, `control_rs_ci::trace::status::tests::target_absent_from_the_log_is_unrun`, `control_rs_ci::trace::status::tests::every_listed_target_must_pass`, `control_rs_ci::trace::status::tests::kani_target_passes_and_unsatisfied_cover_fails`, `control_rs_ci::trace::status::tests::compare_results_are_parsed`, `control_rs_ci::trace::status::tests::junit_results_are_parsed`, `trace_tests::cli::a_failed_target_fails_the_trace_with_a_located_defect`, `control_rs_ci::trace::reqs::tests::a_method_table_names_an_automated_method`                                                                                                                                   | Targets are recorded per condition; a target that passed, failed, is ambiguous or is absent from its log yields `Pass`, `Fail` or `Unrun`, and every listed target must pass                                                                                                                                                                                                                                                                   |
| VC-17.2   | FR-11                                  | `libtest` | `control_rs_ci::trace::status::tests::libtest_results_are_keyed_by_binary`, `control_rs_ci::trace::status::tests::a_path_recorded_with_different_outcomes_is_ambiguous`, `control_rs_ci::trace::status::tests::kani_conflicting_records_are_ambiguous`, `control_rs_ci::trace::status::tests::compare_conflicting_records_are_ambiguous`, `control_rs_ci::trace::status::tests::junit_conflicting_records_are_ambiguous`, `control_rs_ci::trace::status::tests::junit_classname_before_name_does_not_false_pass`, `control_rs_ci::trace::status::tests::junit_cdata_close_marker_does_not_false_pass`, `control_rs_ci::trace::status::tests::doctests_match_on_file_and_item_without_the_line`, `control_rs_ci::trace::status::tests::binary_names_drop_the_extension_and_hash`, `control_rs_ci::trace::status::tests::compare_and_junit_targets_match_exactly`                                                                                      | The same path in two test binaries keeps its own outcome, a wrong crate segment is `Unrun`, a compare or junit target of another suite or class is `Unrun` though its bare name passed, conflicting records (libtest, kani, compare, junit) and shared doctest items are ambiguous, junit attribute lookups match at token boundaries, CDATA blocks do not close test cases or trigger failures, and a doctest matches without its line number |
| VC-17.3   | FR-11                                  | `libtest` | `control_rs_ci::trace::status::tests::target_missing_from_miri_log_warns`, `control_rs_ci::trace::status::tests::interpreter_warnings_cover_only_the_crates_the_log_runs`, `control_rs_ci::trace::status::tests::no_interpreter_artifact_means_no_w2`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | W-2 follows `interpreter_artifact`, covers only the crates its log ran and is absent without the key                                                                                                                                                                                                                                                                                                                                           |
| VC-18.1   | FR-12                                  | `libtest` | `control_rs_ci::trace::reqs::tests::cited_decision_without_record_is_reported`, `control_rs_ci::trace::reqs::tests::decision_defined_twice_is_a_duplicate`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | A citation with no decision record and a decision ID defined twice are each reported once at the citing or second defining line; FR-12 holds iff all four conditions hold                                                                                                                                                                                                                                                                      |
| VC-18.2   | FR-12                                  | `libtest` | `control_rs_ci::trace::reqs::tests::requirement_citing_unaccepted_decision_is_reported`, `control_rs_ci::trace::reqs::tests::citations_outside_requirement_text_are_ignored`, `control_rs_ci::trace::reqs::tests::decision_record_without_status_is_reported`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | A requirement citing a Proposed and a Superseded decision gives two defects; decision IDs outside requirement text give none; a record with no status gives one                                                                                                                                                                                                                                                                                |
| VC-18.3   | FR-12                                  | `libtest` | `control_rs_ci::trace::reqs::tests::absent_decisions_table_disables_decision_checks`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | With no `[decisions]` table, a corpus citing undefined decision IDs gives no decision defect                                                                                                                                                                                                                                                                                                                                                   |
| VC-18.4   | FR-12                                  | `libtest` | `trace_negative_gates::test_negative_trace_reqs_gate_fails_on_unaccepted_and_missing_decisions`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Negative example: the `trace-reqs` gate fails on a fixture workspace whose requirement cites a Proposed decision and an unrecorded one, and its log reports both at the requirement                                                                                                                                                                                                                                                            |
| VC-19.1   | FR-13                                  | `libtest` | `control_rs_ci::trace::reqs::tests::link_targets_are_not_parents`, `control_rs_ci::trace::reqs::tests::link_conditions_report_resolution_and_cycle_defects`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Targets of a link row are not parents, a target naming nothing, an empty document and a cycle are each a defect at the row; FR-13 holds iff all four conditions hold                                                                                                                                                                                                                                                                           |
| VC-19.2   | FR-13                                  | `libtest` | `control_rs_ci::trace::status::tests::link_takes_the_worst_status_of_the_named_documents`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | A link condition takes the worst status of what it names and fails on a cycle                                                                                                                                                                                                                                                                                                                                                                  |
| VC-19.3   | FR-13                                  | `libtest` | `trace_tests::cli::a_failing_linked_document_fails_the_linking_document`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | End to end, a failed target in a linked document fails the linking condition and the run                                                                                                                                                                                                                                                                                                                                                       |
| VC-19.4   | FR-13                                  | `link`    | `tracer-link-graph`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | The link graph that resolution, cycle detection and status derivation use holds per its own conditions                                                                                                                                                                                                                                                                                                                                         |
| VC-20.1   | FR-14                                  | `libtest` | `control_rs_ci::trace::status::tests::gate_target_passes_on_verdict_pass`, `control_rs_ci::trace::status::tests::gate_target_fails_on_fail_skipped_or_warn`, `control_rs_ci::trace::status::tests::gate_target_missing_artifact_is_unrun`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | A condition of method *gate* resolves its target gate slug against `<result_dir>/<gate>.result.json`, passing on `Pass`, failing on `Fail` or `Skipped`, and reporting `Unrun` when absent; FR-14 holds                                                                                                                                                                                                                                        |

### Limits

- The tracer shows that each automated condition names targets that passed in
  the result logs. It does not show that the targets drive the condition both
  ways or show its independent effect on the requirement; the Criterion states
  the combination and review checks it.
- A target's claim is not checked against what the test or proof harness does.
- A target that is compiled out or ignored is absent from its log and reports
  `Unrun`.
- A condition has no link to the code that implements it, so a surviving
  mutant cannot be attributed to a requirement.
- Every requirement ID on a verification row is a parent, including one
  mentioned in the Criterion, and a second condition ID or a second method
  code span is a defect. Criteria therefore name neither.
- A requirement deleted without being added to `retired` is not detected;
  review of the diff is the control.
- W-3 compares file modification times with the traced documents only. A
  code or test edit after the run leaves the log stale without a warning.
- Decision checks read status badges and the decision IDs in requirement
  text. A mention elsewhere, or in a document outside `files`, is not
  checked, and whether a design follows the decisions it cites is left to
  review.
- A `doc-status` condition shows the recorded status of what it names, not
  that its Criterion holds; the review that set the status carries the claim.
  A document outside `files` is not traced, so a target naming it is `Unrun`.

---

## Performance & Resource Considerations

- Both binaries are single-pass and line-oriented; each line is tested
  against a handful of compiled patterns. Cost is dominated by file I/O.
  Measured on 2026-09-27 against rev 1.0 of this document: `trace-reqs` 16 ms.
- Artifacts are small: one row per definition and condition.
- `regex` brings `aho-corasick`, `memchr`, `regex-automata` and
  `regex-syntax`, with no proc-macro crate. `control-rs-compare` already
  depends on it, so the lockfile gains no crate.

---

## Risks & Open Questions

- **Migration effort.** Each of the 23 documents needs condition IDs and
  methods in its Verification table before it joins `files`.
- **Decomposition quality.** Coverage is only as good as the conditions an
  author writes; the tracer cannot tell whether a requirement is fully
  decomposed. Review of the Verification table carries that.
- **Deferred criteria.** Outcome tags are reconsidered after the first areas
  migrate, or when an adopter needs condition coverage evidence (see
  [Extension Path](#extension-path)).
- **Document renames.** Renaming a design document changes every qualified ID
  in it, which breaks `retired` entries that name the old document. They are
  updated in the same pull request.
- **Open: DO-178C wording.** The research record quotes the DO-178B MC/DC
  definition through Hayhurst et al. (2001); the DO-178C text of the
  requirements-based and structural coverage analyses has not been quoted.

---

## Development Plan

| Phase / Task                                             | Description                                                                                                                                                                                                                                                                                | Estimated Effort |
|:---------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: `trace-reqs`**                                | Word-bounded IDs, condition rows with `parents` and `method`, the Checks table, schema 2, configuration keys `verification`, `methods` and `marked_methods`.                                                                                                                               | 2                |
| **Phase 2: Targets**                                     | `Target` column with fully qualified test and harness paths in condition rows; `trace-marks`, source markers and `control-rs-trace-macros` removed.                                                                                                                                        | 1                |
| **Phase 3: `trace-check`**                               | Coverage derivation per Status Derivation, report schema 2, `--config` in place of gate inputs.                                                                                                                                                                                            | 1                |
| **Phase 4: Gate wiring**                                 | `trace` joins the `lint` group; the report job's trace step and the runner's retention of other gates' results go (`ci-design.md`).                                                                                                                                                        | 1                |
| **Phase 5: Corpus migration**                            | Add condition IDs and methods to each document's Verification table and add the document to `files`. One pull request per area.                                                                                                                                                            | 3                |
| **Phase 6: Framework Methods (PR3-6)**                   | Framework-qualified methods `libtest`, `kani`, `compare` and `junit`; per-target result evaluation in `trace-check` from `test.log`, `kani.log`, `cross-val-report.json` and `junit.xml`; Miri warning W-2; schema 4; `trace` moved to `post` stage.                                       | 2                |
| **Phase 7: Decision references**                         | `[decisions]` table, decision records and citations per Matching Rules, the `decision` and `decision_status` checks and decision IDs in `duplicate`; `.cargo/trace/trace.toml` gains the table.                                                                                            | 1                |
| **Phase 8: Links**                                       | The `link` method, Matching Rule 7, the `link` check and link status derivation over the graph of `tracer-link-graph-design.md`; `--graph` for `trace-check`.                                                                                                                              | 1                |
| **Phase 9: Method objects**                              | `METHODS` of `TraceCheckMethod` for every method; the `target` check; `DocStatus` reading traced documents and decision records with W-4; `Review` and `Deferred` removed, report schema 5; exact `compare` and `junit` lookup; `doc-status` rows of traced documents name document names. | 2                |
| **Phase 10: Quality Gate Method and Typed Dependencies** | Generic `gate` verification method reading typed `GateOutcome`; rejection of self-targeting `doc-status` conditions; typed `ValidationReport` deserialization via `control-rs-compare`; `kani` prefix stripping; token-bounded and CDATA-safe `junit` XML parsing.                         | 2                |

---

### 10. Revision History

| Revision | Date               | Author                                              | Description                                                                                                                                                        |
|:---------|:-------------------|:----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | September 27, 2026 | @MitchellDScott                                     | Definitions and references; verification conditions and condition coverage.                                                                                        |
| 2.0      | September 28, 2026 | @MitchellDScott                                     | Proof method and per-item verification status.                                                                                                                     |
| 3.0      | September 30, 2026 | @MitchellDScott                                     | Framework-qualified `Target` cells replace source markers (FR-11).                                                                                                 |
| 4.0      | October 3, 2026    | @MitchellDScott                                     | Decision references (FR-12).                                                                                                                                       |
| 5.0      | October 4, 2026    | @MitchellDScott                                     | Tracer gap closures: libtest binaries, doctests and the `deferred` method.                                                                                         |
| 6.0      | October 16, 2026   | @MitchellDScott, Claude Opus 5.5                    | No document template is assumed.                                                                                                                                   |
| 7.0      | October 21, 2026   | @MitchellDScott, Claude Sonnet 5.5, Claude Opus 5.5 | Links (FR-13) and the `--graph` argument.                                                                                                                          |
| 8.0      | October 23, 2026   | @MitchellDScott, Antigravity, Claude Opus 5.5       | Structured compare and junit methods in one METHODS table; doc-status reads document status; report schema 5.                                                      |
| 8.1      | October 23, 2026   | @MitchellDScott, Antigravity, Cursor Agent          | Quality gate method (gate), self-targeting condition guard on doc-status, typed compare and gate dependencies, and JUnit token-bounded and CDATA-safe XML parsing. |

---

## References

[1] K. J. Hayhurst, D. S. Veerhusen, J. J. Chilenski and L. K. Rierson, "A
Practical Tutorial on Modified Condition/Decision Coverage," NASA Langley
Research Center, Hampton, VA, USA, Rep. no. NASA/TM-2001-210876, 2001.

[2] M. W. Whalen, A. Rajan, M. P. E. Heimdahl and S. P. Miller, "Coverage
metrics for requirements-based testing," in *Proc. ACM/SIGSOFT Int. Symp.
Software Testing and Analysis (ISSTA 2006)*, Portland, ME, USA, 2006,
pp. 25–36, doi: 10.1145/1146238.1146242.

[3] M. Staats, M. W. Whalen, M. P. E. Heimdahl and A. Rajan, "Coverage Metrics
for Requirements-Based Testing: Evaluation of Effectiveness," in *Proc. 2nd
NASA Formal Methods Symp. (NFM 2010)*, Washington, DC, USA, 2010.

[4] M. Hatzl, *mantra-rust-macros* (Version 0.7.8). [Online]. Available:
https://docs.rs/crate/mantra-rust-macros/latest. Accessed: Sep. 11, 2026.

[5] Rust Project Developers, "coverage: Remove all unstable support for MC/DC
instrumentation (#144999)," in *rust-lang/rust*. [Online]. Available:
https://github.com/rust-lang/rust/commit/562222b73765a326fa800a075814deaf627874df.
Accessed: Sep. 27, 2026.

[6] The Rust Project Developers, "3558-libtest-json," *The Rust RFC Book*.
[Online]. Available: https://rust-lang.github.io/rfcs/3558-libtest-json.html.
Accessed: Sep. 11, 2026.

[7] OpenFastTrace, "doc/user_guide.md," in *itsallcode/openfasttrace*.
[Online]. Available:
https://github.com/itsallcode/openfasttrace/blob/main/doc/user_guide.md.
Accessed: Sep. 11, 2026.

[8] OpenFastTrace, "doc/spec/system_requirements.md," in
*itsallcode/openfasttrace*. [Online]. Available:
https://github.com/itsallcode/openfasttrace/blob/main/doc/spec/system_requirements.md.
Accessed: Sep. 11, 2026.

[9] StrictDoc, "User Guide," *StrictDoc Documentation*. [Online]. Available:
https://strictdoc.readthedocs.io/en/stable/stable/docs/strictdoc_01_user_guide.html.
Accessed: Sep. 11, 2026.

[10] M. Hatzl, "README.md," in *mhatzl/mantra*. [Online]. Available:
https://github.com/mhatzl/mantra. Accessed: Sep. 11, 2026.

[11] Doorstop, "Validating Requirements," *Doorstop Documentation*. [Online].
Available: https://doorstop.readthedocs.io/en/latest/cli/validation.html.
Accessed: Sep. 11, 2026.

[12] Kani Rust Verifier Contributors, "The Kani Rust Verifier Documentation and
Source," *model-checking/kani GitHub repository*. [Online]. Available:
https://github.com/model-checking/kani. Accessed: Sep. 28, 2026.

[13] Y. Moy, E. Ledinot, H. Delseny, V. Wiels, and B. Monate, "Testing or
Formal Verification: DO-178C Alternatives and Industrial Experience," *IEEE
Software*, vol. 30, no. 3, pp. 50–57, 2013, doi: 10.1109/MS.2013.43.

[14] D. Cofer and S. P. Miller, "Formal Methods Case Studies for DO-333," NASA
Langley Research Center, Hampton, VA, USA, Rep. no. NASA/CR-2014-218244, 2014.
[Online]. Available:
https://shemesh.larc.nasa.gov/people/bld/ftp/NASA-CR-2014-218244.pdf. Accessed:
Sep. 28, 2026.