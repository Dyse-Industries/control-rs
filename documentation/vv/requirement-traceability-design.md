# Requirement Traceability Infrastructure (Design Document)

![Date Badge](https://img.shields.io/badge/Date-September_26,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

## Introduction

Requirements in `control-rs` are written first, in the Requirements section of
each design document, and the design follows from them. This document
specifies the tooling that checks every requirement has a verification row
with a gate and criterion, and links it to the gate results that verify it.

The tooling rests on two concepts. A **definition** is the one line where a
requirement ID is introduced. A **reference** is any other line that names the
ID for a stated purpose, such as a verification-table row.
Three small binaries in `control-rs-ci` handle one step each:

- `trace-reqs` finds definitions and references in Markdown, checks them and
  writes `reqs.jsonl`;
- `trace-marks` finds requirement markers in source text, in any language,
  and writes `marks.jsonl`;
- `trace-check` joins both with gate results and writes `trace-report.json` and
  a gate verdict.

**Concept of operations**

| Actor                     | Action                                                                          | Touches                                           |
|:--------------------------|:--------------------------------------------------------------------------------|:--------------------------------------------------|
| Author                    | Writes requirements and their verification first, then the architecture         | One Markdown file, in any editor                  |
| Reviewer                  | Reads requirements and verification in a pull request                           | Rendered Markdown on GitHub                       |
| Linter (local or CI)      | Reports missing, duplicate or unresolved requirements with `path:line: message` | The same Markdown file                            |
| Tracer (CI)               | Links requirements to gate evidence and test markers                            | `reqs.jsonl`, `marks.jsonl`, `<gate>.result.json` |
| Systems engineer (future) | Builds hierarchies, allocations and verification matrices                       | External tools that read the `.jsonl` files       |

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

**Two audiences.** `control-rs` always follows
`documentation/design-template.md`, and its configuration encodes that
template. External users of `control-rs-ci` describe their own layout with the
same few patterns (see [Configuration](#configuration)).

**Scope.** In scope: definitions, references, markers, the checks over them,
status derivation and the three artifacts. Out of scope: test adequacy (the
tracer shows that linked evidence exists and passed, not that it is
sufficient); requirement hierarchies; citation, reference and general document
style, which an external, unpublished review tool owns; requirement editing
tools; and any systems-engineering view.

---

## Requirements

### Functional Requirements

- **FR-1 — Definitions**: `trace-reqs` shall record one definition for the
  first requirement ID on each line that matches the `definition` pattern,
  with the definition text formed per [Matching Rules](#matching-rules).
- **FR-2 — References**: `trace-reqs` shall record one reference of kind `K`
  for every requirement ID on each line that matches the reference pattern
  named `K`.
- **FR-3 — Qualified IDs**: `trace-reqs` shall qualify every recorded ID as
  `<doc>#<id>` per [Matching Rules](#matching-rules), so that equal IDs in
  different documents never collide.
- **FR-4 — Checks**: `trace-reqs` shall report as a defect each ID defined
  more than once, each reference to an undefined ID, each defined ID that lacks
  a reference of any configured kind and each retired ID that is defined or
  referenced.
- **FR-5 — Phrase Rules**: `trace-reqs` shall report as a defect each match of
  an `exclude_phrases` pattern in definition text.
- **FR-6 — Requirement Rows**: `trace-reqs` shall write `reqs.jsonl` with one
  row per definition and per reference, in the row schema of
  [Artifacts](#artifacts).
- **FR-7 — Marker Rows**: `trace-marks` shall write `marks.jsonl` with one row
  of kind `marker` for every qualified ID on each marker line, and shall report
  as a defect each marker line with no ID and each ID on a marker line written
  without `<doc>#`, per [Source Markers](#source-markers).
- **FR-8 — Trace Verdict**: `trace-check` shall derive each requirement status
  per [Status Derivation](#status-derivation), write `trace-report.json` and
  exit 0 iff no requirement is `Failed` or `Unverified` and every marker
  resolves to a definition.
- **FR-9 — Diagnostic Format**: `trace-reqs`, `trace-marks` and `trace-check`
  shall report
  each defect as one `path:line: message` line and shall exit non-zero when any
  defect exists.

### Non-Functional Requirements

- **NFR-1 — Audit Latency**: Each of the three binaries shall complete on the
  `control-rs` workspace in under 1 s of wall-clock time, excluding compilation.
- **NFR-2 — Zero Runtime Intrusion**: A marker shall expand to the item it
  marks unchanged, so that no marker changes compiled output for any target.
- **NFR-3 — Dependency Budget**: The three binaries shall add no dependency to
  `control-rs-ci` beyond its current set and the `regex` crate; the `#[req]`
  crate shall depend only on `syn` and `proc-macro2`.
- **NFR-4 — Schema Versioning**: Every row and `trace-report.json` shall carry
  an integer `schema` field, incremented on every incompatible change.

### Constraints

- **C-1 — Plain-Text Source**: Requirements and their verification shall live in
  the Markdown files the configuration lists; no generated file shall be
  committed or required for reading or writing a requirement.
- **C-2 — No Editor Integration**: The design shall deliver diagnostics only; it
  shall ship no editor plugin, schema, snippet or language server.
- **C-3 — Paths from Configuration**: Every input and output path of the three
  binaries shall arrive from a command-line argument or the configuration file.
- **C-4 — Read-Only Evidence**: The tracer shall run no test, build or gate; it
  shall read recorded artifacts only and remove no artifact but its own.
- **C-5 — External Systems Tooling**: Hierarchy views, allocation, verification
  matrices and model export shall consume the `.jsonl` files outside this
  workspace's CI path.
- **C-6 — Artifact Location**: The three binaries shall write under
  `target/ci-artifacts/` when run as gates; `#[req]` shall write nothing.
- **C-7 — Document Style Exclusion**: `trace-reqs` shall check no citation,
  reference or general document style.

---

## Technical Overview

### Author Workflow

1. **Define the requirement** in the Requirements section:

   ```markdown
   - **FR-3 — Packed Storage**: Packed symmetric, Hermitian and triangular storage
     shall return `None` from `value(i, j)` when `i` or `j` is out of bounds.
   ```

2. **State its verification** with a row in the Verification table. The `Gates`
   cell names the gate whose result is the evidence; `Criterion` is one sentence
   stating the pass condition, including any bound:

   ```markdown
   | Requirements | Gates  | Criterion                                                          |
   |:-------------|:-------|:-------------------------------------------------------------------|
   | FR-3         | `test` | `value(i, j)` is `None` iff i = N or j = N over [0, N]², SP/HP/TP |
   ```

3. **Mark the test** that verifies it, optionally, so the link survives in
   code. Rust uses the `#[req]` attribute; any other language uses the marker
   text the configuration names, such as `// req: storage#FR-3`:

   ```rust
   #[req("storage#FR-3")]
   #[test]
   fn packed_value_out_of_bounds_is_none() { /* ... */ }
   ```

4. **Run `cargo trace-reqs`.** Each problem prints as `path:line: message`,
   which the editor turns into a clickable link:

   ```text
   documentation/math/storage-design.md:42: storage#FR-3 has no verification reference
   ```

To withdraw a requirement, delete its definition and add its qualified ID to
`retired` in the configuration. The ID is never reused.

### Reading the Output

Every line of `reqs.jsonl` and `marks.jsonl` is one occurrence of one ID, with
the same six fields:

```json lines
{
  "schema": 1,
  "id": "storage#FR-3",
  "kind": "definition",
  "file": "documentation/math/storage-design.md",
  "line": 42,
  "text": "- **FR-3 — Packed Storage**: Packed symmetric ... out of bounds."
}
{
  "schema": 1,
  "id": "storage#FR-3",
  "kind": "verification",
  "file": "documentation/math/storage-design.md",
  "line": 210,
  "text": "| FR-3 | `test` | `value(i, j)` is `None` iff i = N or j = N over [0, N]², SP/HP/TP |"
}
{
  "schema": 1,
  "id": "storage#FR-3",
  "kind": "marker",
  "file": "src/math/storage/tests.rs",
  "line": 119,
  "text": "#[req(\"storage#FR-3\")]"
}
```

Grouping rows by `id` gives everything known about a requirement; filtering by
`kind` gives every verification row or every marker. No field
depends on configuration to be understood.

### Data Flow

```mermaid
flowchart LR
    Cfg["trace.toml"] --> TR
    Docs["Markdown files"] --> TR["trace-reqs"]
    TR --> RJ["reqs.jsonl"]
    Cfg --> TM
    Src["Source files"] --> TM["trace-marks"]
    TM --> MJ["marks.jsonl"]
    Gates[".cargo/gate.toml"] --> TC
    Res["&lt;gate&gt;.result.json"] --> TC["trace-check"]
    RJ --> TC
    MJ --> TC
    TC --> Rep["trace-report.json"]
    TC --> Exit["exit status"]
    RJ -.-> Ext["external systems tool<br/><i>outside CI (C-5)</i>"]
```

**Where requirements live.** Requirements stay in the design documents.
Measured on 2026-09-26, the corpus holds 268 requirements in 23
`*-design.md` files, 244 of them in 20 Approved documents. Keeping them in place
needs no migration, keeps each requirement next to its rationale and renders on
GitHub. The `-design.md` suffix already serves as a file-type pattern for editor
rules (`files.associations` in VS Code, file-type patterns in JetBrains IDEs).

---

## Architecture

### Configuration

`trace-reqs` and `trace-marks` read one TOML file named by `--config`. The
`control-rs` configuration:

```toml
id = '(?:FR|NFR|C)-[0-9]+[a-z]?'
doc = '[a-z0-9-]+'
files = ["documentation/vv/requirement-traceability-design.md"]
doc_suffix = "-design"
definition = '^- \*\*(?:FR|NFR|C)-'
retired = []
exclude_phrases = []

[references]
verification = '^\| *(?:[a-z0-9-]+#)?(?:FR|NFR|C)-'

[markers]
files = ["src", "tests", "benches"]
suffixes = [".rs"]
marker = "#[req("
```

| Key                | Meaning                                                                                                        |
|:-------------------|:---------------------------------------------------------------------------------------------------------------|
| `id`               | Regex for one requirement ID                                                                                   |
| `doc`              | Regex for the document name in a qualified ID `<doc>#<id>`                                                     |
| `files`            | Markdown files, and directories whose `<doc_suffix>.md` files are read (see [File Selection](#file-selection)) |
| `doc_suffix`       | Text removed from the file stem to form the document name; optional                                            |
| `definition`       | Regex for a line that defines a requirement                                                                    |
| `references`       | Named regexes; each names one kind of reference, and every defined requirement needs at least one of each kind |
| `retired`          | Qualified IDs that must not be defined or referenced again                                                     |
| `exclude_phrases`  | Regexes; each match in definition text is a defect; optional                                                   |
| `markers.files`    | Source files, and directories whose files ending in a suffix are read; `[markers]` is optional                 |
| `markers.suffixes` | File-name endings of the source files to read, such as `.rs`                                                   |
| `markers.marker`   | Text that makes a source line a marker, such as `#[req(`                                                       |

In the template, a verification row is a pipe-table row whose first cell starts
with a requirement ID. `trace-marks` reads `id`, `doc` and `[markers]`; without
`[markers]` it writes an empty `marks.jsonl`.

### File Selection

`files` and `markers.files` list roots relative to the working directory.

- A root that is a file is selected.
- A root that is a directory selects every file below it whose name ends in a
  suffix: `<doc_suffix>.md` for `files`, each of `markers.suffixes` for
  `markers.files`.
- Symbolic links are never followed, so a walk cannot loop.
- A root that is missing, or is neither a file nor a directory, is an error.

### Matching Rules

The rules below are the whole parser:

1. Lines inside fenced code blocks, including indented fences, are skipped,
   so examples are inert.
2. An ID occurrence is a match of `id`, optionally preceded by a match of `doc`
   and `#`.
3. A line that matches `definition` defines the first ID on it. Its text is the
   line plus the indented lines that follow it.
4. A line that matches a reference pattern references every ID on it. A line
   may match more than one pattern and then yields one reference per pattern.
5. The document name is the file stem with `doc_suffix` removed:
   `storage-design.md` becomes `storage`. An ID written without `<doc>#` belongs
   to the document it appears in.
6. The phrase rule runs on definition text with backtick spans removed, so
   identifiers never match.

### Checks

| Check            | Defect                                                                                       |
|:-----------------|:---------------------------------------------------------------------------------------------|
| `duplicate`      | An ID is defined more than once                                                              |
| `unresolved`     | A reference names an ID that is not defined                                                  |
| `missing`        | A defined ID has no reference of a configured kind; the message names the kind               |
| `retired`        | A retired ID is defined or referenced                                                        |
| `exclude-phrase` | A match of an `exclude_phrases` pattern in definition text; one defect per match, quoting it |

The phrase check follows the lexical-smell approach of Femmer et al. [6]; the
patterns are the project's choice.

### Source Markers

A **marker line** is a source line that contains the `markers.marker` text.
Each qualified ID `<doc>#<id>` on it yields one `marker` row whose `text` is the
line. A marker line with no ID, or with an ID written without `<doc>#`, is a
defect. `trace-marks` reads text only, so markers work in any language and do
not depend on what a build compiles.

In Rust the marker is the `#[req]` attribute, placed on the item it marks, with
qualified IDs as string arguments. It lives in its own proc-macro crate,
`control-rs-trace-macros`, apart from the ETS macros of `control-rs-macros`
(`macros-design.md`). The compiler attaches it to that item, as mantra does
with its `req` attribute macro [8]:

```rust
#[req("storage#FR-3", "storage#FR-4")]
#[test]
fn packed_value_bounds() { /* ... */ }
```

The macro returns the item unchanged and writes nothing; an argument that is
not `<doc>#<id>` is a compile error.

Markers are optional. They are the mechanism for tracing mutation-gate repairs:
a test added to kill a mutant carries the marker of the requirement it
protects.

### Status Derivation

A requirement's **gates** are the gate names from `gate.toml` that appear in
backticks on any of its reference rows. Evidence is gate-level: a gate that
passes has passed every test it ran. `trace-check` reads each
`<gate>.result.json` `verdict` (`pass`, `warn`, `fail`, `skipped`) as defined in
`ci-design.md`.

| Condition (first match wins)                       | Status       | Fails gate |
|:---------------------------------------------------|:-------------|:-----------|
| Any gate verdict is `fail`                         | `Failed`     | yes        |
| Any gate has no result, or verdict `skipped`       | `Unverified` | yes        |
| At least one gate, and every gate `pass` or `warn` | `Verified`   | no         |
| No gate named                                      | `Unchecked`  | no         |

A `marker` row whose ID has no definition row fails the gate.
`trace-report.json` holds `schema`, a count per status, one entry per
requirement (ID, status, gate verdicts) and the unresolved markers.

### Artifacts

| Field    | Meaning                                                                                      |
|:---------|:---------------------------------------------------------------------------------------------|
| `schema` | Row format version (NFR-4)                                                                   |
| `id`     | Qualified ID, `<doc>#<id>`                                                                   |
| `kind`   | `definition`, a reference kind from `[references]`, or `marker`                              |
| `file`   | Path relative to the working directory                                                       |
| `line`   | 1-based line number                                                                          |
| `text`   | The matched line; for a definition, its full text; for a marker, the item's keyword and name |

In `text`, each run of whitespace, line breaks included, becomes one space, and
none remains at either end. A wrapped definition therefore reads as one line,
and a phrase pattern matches across the wrap.

Rows are sorted by `file`, then `line`. Examples are in
[Reading the Output](#reading-the-output).

### Invocation and Gate Wiring

| Binary        | Gate and placement                 | Arguments                                                                                                                                                                          |
|:--------------|:-----------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `trace-reqs`  | `trace-reqs`, `lint` group         | `--config .cargo/trace/trace.toml --out target/ci-artifacts/reqs.jsonl`                                                                                                            |
| `trace-marks` | `trace-marks`, `lint` group        | `--config .cargo/trace/trace.toml --out target/ci-artifacts/marks.jsonl`                                                                                                           |
| `trace-check` | `trace`, last in `exclusive_gates` | `--reqs target/ci-artifacts/reqs.jsonl --marks target/ci-artifacts/marks.jsonl --gates .cargo/gate.toml --results target/ci-artifacts --out target/ci-artifacts/trace-report.json` |

`trace` is the last gate to run and precedes the report: it reads every result
and artifact the run produced, and `ci-report.md` includes its verdict.
`cargo ci` runs the exclusive gates in list order after all groups join. In
GitHub Actions the lanes are separate jobs; the report job downloads their
artifacts, runs `cargo gate trace` and then runs `cargo report` whatever the
trace verdict. Each gate manages only its own artifacts: a run writes those of
the gates it runs and removes none of any other gate, so the downloaded results
stay in place.

### Adoption

Existing verification tables carry no Requirements column, so every
existing requirement would report a `missing` defect. `files` is therefore
the adoption mechanism: it lists documents one by one as each is migrated and
becomes the directory `documentation` when the last one is done. A document outside `files` is
not checked at all; a document inside it is checked in full.

---

## Alternatives

| Alternative                                                                       | Rejected Because                                                                                                                                                                                                         | Reference |
|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|
| **Named captures, requirement blocks and link attachment** (rev 1.3, 1.4)         | Meaning hid in capture-group names and position-dependent parsing state; `reqs.jsonl` mirrored parser internals and could not be read without the configuration.                                                         | n/a       |
| **Fixed grammar with configured section, table and column names** (rev 1.2)       | Many keys with distinct semantics, still bound to headings and pipe tables.                                                                                                                                              | n/a       |
| **Adoption baseline file and previous-artifact ID check** (rev 1.1 to 1.4)        | Two baselines with different formats and a workflow restore step; listing migrated documents in `files` ratchets adoption with no new concept, and `retired` covers ID reuse.                                            | n/a       |
| **`Parent` and `Status` lines**                                                   | Hierarchy belongs to the external systems tool (C-5); retirement is covered by `retired`; draft requirements are expected to carry verification from the start.                                                          | n/a       |
| **Structured JSON requirement files with per-area schemas** (prototype `4f4b70b`) | Each step toward better editing (per-area schemas, snippets, IDE-specific workarounds) added machinery an author must understand before writing a requirement. Retained: never-reused IDs, a generated record for tools. | n/a       |
| **TOML requirement files**                                                        | Rejected on 2026-09-25 for the same editing cost, with worse multi-line text.                                                                                                                                            | n/a       |
| **`// req:` comment markers in Rust**                                             | Nothing ties a comment to the item below it and nothing checks its IDs at compile time; the compiler attaches an attribute to its item. Other languages use comment markers through `markers.marker`.                    | n/a       |
| **Link records written by `#[req]` at compile time** (rev 1.6 to 1.8)             | Records depended on what a build compiled and on the build cache, went stale until a clean build, and editors that expand macros wrote records for unsaved edits. Reading source text has none of these failure modes.   | n/a       |
| **Globs with `!` exclusions and `depth`** (rev 1.6 to 1.8)                        | A glob engine and a symbolic-link rule table to select a handful of directories; roots plus a suffix select the same files.                                                                                              | n/a       |
| **Links written by the marked test at run time**                                  | Only functions could carry markers, and test binaries would change (NFR-2).                                                                                                                                              | n/a       |
| **Hand-written matcher instead of the `regex` crate**                             | Users would learn a bespoke pattern language; regular expressions are already known. The cost is one dependency (NFR-3).                                                                                                 | n/a       |
| **Requirement smells as a Vale style**                                            | Vale cannot scope rules to requirement text; the rules would fire on rationale prose. Vale remains the prose gate.                                                                                                       | n/a       |
| **Test-level verdicts from libtest JSON**                                         | The format is unstable and requires `-Z unstable-options`; gate-level verdicts are stable and already recorded.                                                                                                          | [7]       |
| **Editor plugins, schemas or a language server**                                  | Adds per-editor machinery; `path:line: message` works everywhere (C-2).                                                                                                                                                  | n/a       |
| **OpenFastTrace (Java)**                                                          | Adds a Java runtime to CI.                                                                                                                                                                                               | [1], [2]  |
| **StrictDoc (Python)**                                                            | Adds a Python toolchain and its own `.sdoc` grammar.                                                                                                                                                                     | [4]       |
| **mantra (Rust)**                                                                 | Relies on a database and line-coverage pairing rather than gate verdicts.                                                                                                                                                | [3]       |
| **Doorstop**                                                                      | One YAML file per requirement; loses requirements-beside-rationale.                                                                                                                                                      | [5]       |

---

## Verification & Validation

### Verification

| Requirements                                      | Gates                       | Criterion                                                                                                                                                                                                                                                                |
|:--------------------------------------------------|:----------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FR-1, FR-2, FR-3                                  | `test`                      | Matching fixtures produce exactly the expected rows: fenced blocks, first-ID definitions, indented continuation, multiple IDs per reference line, lines matching two patterns, local and qualified IDs; selection fixtures cover file root, directory root, missing root |
| FR-4, FR-9                                        | `test`                      | Every expected defect per positive check fixture, none per negative; diagnostic lines match `^[^:]+:[0-9]+: .+$`; exit 0 iff no defect                                                                                                                                  |
| FR-5                                              | `test`                      | Every match reported outside code spans, an empty list disables the rule                                                                                                                                                                                                 |
| FR-6, FR-7, NFR-4                                 | `test`                      | Two runs on an unchanged corpus are byte-identical; rows carry exactly the six Artifacts fields; marker fixtures: one row per qualified ID, unqualified and missing IDs reported                                                                                          |
| FR-8                                              | `test`                      | Status and exit status match exactly per Status Derivation row; an unresolved marker fails the trace                                                                                                                                                                     |
| NFR-1                                             | `trace-reqs`, `trace-marks` | Median of 5 runs < 1.0 s per binary                                                                                                                                                                                                                                     |
| NFR-2, NFR-3, C-1, C-2, C-3, C-4, C-5, C-6, C-7 |                             | Review: the diff adds no violation of any constraint                                                                                                                                                                                                                     |

### Limits

- Evidence is gate-level. A marked test compiled out by `cfg` under the named
  gate still counts as passed; `trace-check` cannot observe it.
- Markers are found by text. A marker line inside a comment or string literal
  counts like any other.
- A misspelled gate name is not a gate, so the requirement reads `Unchecked`
  rather than failing. The report lists `Unchecked` requirements for review.
- `trace` cannot serve as evidence: `trace-check` runs as that gate, so the
  current run's `trace` result does not exist while it reads results.
- Every ID on a verification row counts, including one mentioned in the
  criterion text, and so does every backticked gate name. Keeping IDs in the
  first cell and gate names in the Gates cell avoids unintended references.
- A requirement deleted without being added to `retired` is not detected;
  review of the diff is the control.
- Phrase checks are lexical. They catch configured patterns, not ambiguity in
  general.

---

## Performance & Resource Considerations

- All three binaries are single-pass and line-oriented; each line is tested
  against a handful of compiled patterns. Cost is dominated by file I/O.
- Artifacts are small: roughly 540 rows once all 268 requirements carry a
  verification reference.
- `regex` 1.13 brings `aho-corasick`, `memchr`, `regex-automata` and
  `regex-syntax`, with no proc-macro crate. `control-rs-compare` already
  depends on it, so the lockfile gains no crate.
- `control-rs-trace-macros` depends on `syn` and `proc-macro2`, which
  `control-rs-macros` already brings into the lockfile; a crate that carries
  markers takes it as a dev-dependency.

---

## Risks & Open Questions

- **Migration effort.** Each of the 23 documents needs a Requirements column in
  its Verification table before it joins `files`. Converting numbered
  headings (586) and `§n` cross-references (472) to the unnumbered template is
  separate editorial work that the tracer does not depend on.
- **Document renames.** Renaming a design document changes every qualified ID
  in it, which breaks markers and `retired` entries that name the old
  document. They are updated in the same pull request.

---

## Development Plan

| Phase / Task                            | Description                                                                                                                                                                                                                   | Estimated Effort |
|:----------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: `trace-reqs`**               | Configuration, file selection, matching rules, checks, `reqs.jsonl`. Remove `documentation/math/storage.req.json`.                                                                                                            | 2                |
| **Phase 2: `#[req]` and `trace-marks`** | The `control-rs-trace-macros` crate with `#[req]` and the source scan into `marks.jsonl`.                                                                                                                                     | 1                |
| **Phase 3: `trace-check`**              | Gate lookup, status derivation and `trace-report.json`.                                                                                                                                                                       | 1                |
| **Phase 4: Gate wiring**                | `.cargo/trace/trace.toml`, the `trace-reqs`, `trace-marks` and `trace` gates, Cargo aliases, the report-job step, a workflow path filter for `.cargo/trace/` and a runner that removes no artifact of a gate it does not run. | 1                |
| **Phase 5: Corpus migration**           | Add the Requirements column to each document's verification tables and add the document to `files`. One pull request per area.                                                                                                | 3                |

---

## Revision History

| Revision | Date               | Author          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:---------|:-------------------|:----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.5      | September 26, 2026 | @MitchellDScott | Simplified to definitions and references: one `definition` pattern, named reference patterns, six-field rows shared by `reqs.jsonl` and `marks.jsonl`. Removed named captures, requirement blocks, link attachment, `Parent` and `Status`, `malformed`, the adoption baseline and the previous-artifact ID check; adoption through `files`, ID reuse through `retired`. Added Author Workflow and Reading the Output.                                           |
| 1.6      | September 26, 2026 | @MitchellDScott | Globs with `!` exclusions and `depth`; `doc` pattern for qualified IDs; whitespace-normalized `text`; `#[req]` attribute markers in `control-rs-trace-macros` whose link records every build writes and `trace-marks` collects; `trace-check` runs as gate `trace` after every other gate and before the report; a gate run removes no other gate's artifacts; NFR-1 evidence from `trace-reqs` and `trace-marks`; C-3 citation and Phase 1 removals corrected. |
| 1.7      | September 26, 2026 | @MitchellDScott | Stated symbolic-link handling for glob walks: links to directories are not entered during a walk, links to files are selected under their own path, dangling links are ignored and links named in a glob's leading components are resolved. Added the double-selection limit and symbolic-link fixtures to the Plan.                                                                                                                                            |
| 1.8      | September 26, 2026 | @MitchellDScott | Dropped `require_phrases`; each match of an `exclude_phrases` pattern is a defect that quotes the match. The Acceptance bound for the checks counts every expected defect. Reference [6] follows the research record, and the ISO/IEC/IEEE 29148 citation is removed until a quoted source exists.                                                                                                                                                              |
| 1.9      | September 26, 2026 | @MitchellDScott | Markers are found in source text: `trace-marks` reads `[markers]` (roots, suffixes, marker text) and works for any language; `#[req]` checks its IDs and writes nothing. Globs replaced by roots plus suffix, with symbolic links never followed; `depth` removed. Removed the link-record Limits and the compiler-path and editor-expansion Risks; the gate-name overlap joins the reference-line Limit.                                                       |
| 1.10     | September 26, 2026 | @MitchellDScott | Merged Plan and Acceptance into a single Verification table (`Requirements \| Gates \| Criterion`). One reference kind `verification` replaces `plan` and `acceptance`; the pattern matches a row whose first cell starts with a requirement ID. Removed the Kind column, the gate-name overlap Limit, the row-shape Risk, and the acceptance-not-evaluated Limit. Updated template, `trace.toml`, tracer tests. |

---

## References

[1] OpenFastTrace Authors, "OpenFastTrace User Guide," *OpenFastTrace
Documentation*, 2026. [Online].
Available: https://itsallcode.github.io/openfasttrace/user_guide.html.

[2] OpenFastTrace Authors, "OpenFastTrace System Requirements," *OpenFastTrace
Specification*, 2026.

[3] M. Hatzl, "mantra: Tracing between requirements, implementation, and tests,"
*GitHub Repository*, 2026. [Online].
Available: https://github.com/mhatzl/mantra.

[4] StrictDoc Authors, "StrictDoc User Guide: Traceability between requirements
and source code," *StrictDoc Documentation*, 2026. [Online].
Available: https://strictdoc.readthedocs.io.

[5] Doorstop Authors, "Doorstop: Requirements management using version control,"
*Doorstop Documentation*, 2026. [Online].
Available: https://doorstop.readthedocs.io.

[6] H. Femmer, D. Méndez Fernández, S. Wagner, and S. Eder, "Rapid quality
assurance with Requirements Smells," *Journal of Systems and Software*, 2016,
doi: 10.1016/j.jss.2016.02.047.

[7] Rust Project, "RFC 3558: Libtest JSON output," *Rust RFCs*. [Online].
Available: https://rust-lang.github.io/rfcs/3558-libtest-json.html.

[8] M. Hatzl, *mantra-rust-macros* (Version 0.7.8). [Online]. Available:
https://docs.rs/crate/mantra-rust-macros/latest. Accessed: Sep. 11, 2026.
