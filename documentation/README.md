# Documentation

What each kind of document in `control-rs` is for and where it lives.
[Workspace](../README.md) · [Contributing](../CONTRIBUTING.md)

| Kind             | Where                                                         | Gives you                                                                |
|:-----------------|:--------------------------------------------------------------|:-------------------------------------------------------------------------|
| README           | `README.md` at the root, in every crate and under `examples/` | How to use that part of the code base: commands, aliases, runnable demos |
| Rustdoc          | Doc comments in source; `cargo doc --open`                    | The API reference for every public item                                  |
| Design documents | `documentation/<project>/<slug>-design.md`                    | What a component must do, how it is built and how it is verified         |
| Overviews        | `documentation/<project>/<project>-overview.md`               | How the design documents of one subsystem fit together                   |
| ADRs             | [`documentation/adr/`](adr/README.md)                         | Decisions that bind more than one crate or the whole workspace           |
| Guides           | `documentation/*.md`, `CONTRIBUTING.md`                       | Workspace-wide workflows, tools, standards and plans                     |

## README

README are the working reference for each part of the code base. Each one
is a cheat sheet for its directory: the cargo aliases and commands that
build, run and test it, and short demos with their expected output.
Start in the README of the crate you are working in.

- [`README.md`](../README.md): the library's numerical models, quickstart
  examples and the table of workspace crates.
- `control-rs-*/README.md`: one per crate. Tooling crates list their CLI;
  [`control-rs-ci`](../control-rs-ci/README.md) holds the gate commands and
  the requirement-tracing workflow.
- [`examples/README.md`](../examples/README.md): every example, firmware
  package and subprogram backend, with the command that runs it.

## Rustdoc

The API reference is generated from doc comments: run `cargo doc --open`, or
the `doc` gate in CI. Every public item states its arguments, return value,
errors, safety contract, panics and a runnable example. ETS suites carry a
lighter form. Both follow
[`documentation-standards.md`](documentation-standards.md).

## Design documents

A design document specifies one component before it is built: the
requirements it meets, its architecture, the alternatives it rejected and the
plan that verifies it. Reviewers and CI check the code against it.
Documents are grouped by project directory and indexed below.

Requirements (`FR-n`, `NFR-n`, `C-n`) and their verification conditions
(`VC-x.y` rows) are machine-read. For every document listed in
[`.cargo/trace/trace.toml`](../.cargo/trace/trace.toml), `cargo trace-reqs`
checks the IDs and `cargo trace-check` links each condition to the test or
proof result that discharges it
([Requirement Traceability Infrastructure](vv/requirement-traceability-design.md#author-workflow)).

The status badge under each title tells you how far to trust the document:

| Status    | Meaning                                       |
|:----------|:----------------------------------------------|
| `Draft`   | Being written or reviewed                     |
| `Ready`   | Reviewed; implementation may follow it        |
| `Shipped` | Implemented as written                        |
| `Stale`   | May no longer match the code; needs re-review |

## Overviews

An overview groups a set of component designs into a sequential set of
development phases.

## ADRs

An Architecture Decision Record holds one decision at the workspace, crate or
project level, such as a dependency policy or a format shared between crates.
Design documents cite it by number. The [ADR index](adr/README.md) lists every
record and its status.

## Guides

| Document                                                   | Purpose                                                     |
|:-----------------------------------------------------------|:------------------------------------------------------------|
| [`development-guide.md`](development-guide.md)             | Workspace architecture, cargo aliases, CI and ETS workflows |
| [`dependencies.md`](dependencies.md)                       | Every external tool, what needs it and how to install it    |
| [`documentation-standards.md`](documentation-standards.md) | Rustdoc and ETS documentation standard                      |
| [`roadmap.md`](roadmap.md)                                 | Planned pull requests and their order                       |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md)                    | How a change moves from design to merge                     |

---

## math

| Design                                                    | Status | Implementation                                     |
|:----------------------------------------------------------|:-------|:---------------------------------------------------|
| [Crate-Wide Error Module](math/error-design.md)           | Stale  | `src/math/`                                        |
| [Fixed-Point Scalar Type](math/fixed-num-design.md)       | Stale  | `src/math/fixed_num.rs`                            |
| [Numeric Trait Hierarchy](math/num-traits-design.md)      | Stale  | `src/math/num_traits.rs`                           |
| [Numeric Types](math/num-types-design.md)                 | Stale  | `src/math/num_types.rs`                            |
| [Storage Backends & Data Layouts](math/storage-design.md) | Stale  | `src/math/storage.rs`                              |
| [Linear Algebra Subprograms](math/subprograms-design.md)  | Stale  | `src/math/subprograms.rs`, `examples/subprograms/` |

## numerical-models

| Design                                                                                 | Status | Implementation             |
|:---------------------------------------------------------------------------------------|:-------|:---------------------------|
| [Numerical Models Integration & Examples](numerical-models/numerical-models-design.md) | Stale  | `control-rs-verification/` |
| [Matrix Type & Structural Specializations](numerical-models/matrix-design.md)          | Stale  | `src/matrix/`              |
| [Polynomial Type](numerical-models/polynomial-design.md)                               | Stale  | `src/polynomial/`          |
| [State-Space Model Type](numerical-models/state-space-design.md)                       | Stale  | `src/state_space/`         |
| [Tensor Type & Low-Cost Inference](numerical-models/tensor-design.md)                  | Stale  | `src/tensor/`              |
| [Transfer Function Type](numerical-models/transfer-function-design.md)                 | Stale  | `src/transfer_function/`   |

## ets

[Overview](ets/ets-overview.md)

| Design                                                                   | Status | Implementation                                           |
|:-------------------------------------------------------------------------|:-------|:---------------------------------------------------------|
| [Host ETS Library](ets/ets-host-design.md)                               | Stale  | `control-rs-ets-host/`                                   |
| [Embedded Test Server](ets/embedded-test-server-design.md)               | Stale  | `control-rs-ets/src/server.rs`                           |
| [ETS Executive](ets/ets-executive-design.md)                             | Stale  | `control-rs-ets/src/{exec,run,board,fault,cell}.rs`      |
| [ETS Link](ets/ets-link-design.md)                                       | Draft  | `control-rs-ets/src/link/`, `control-rs-ets/src/wire.rs` |
| [`CPUProfiler`](ets/cpu-profiler-design.md)                              | Stale  | `control-rs-ets/src/profiler.rs`                         |
| [Exportable Test Suites](ets/test-suite-design.md)                       | Stale  | `src/**/tests/` (`ets` feature)                          |
| [Procedural Macros for Distributed Test Discovery](ets/macros-design.md) | Stale  | `control-rs-macros/`                                     |

## tui

| Design                                       | Status | Implementation    |
|:---------------------------------------------|:-------|:------------------|
| [Terminal User Interface](tui/tui-design.md) | Stale  | `control-rs-tui/` |

## ci

| Design                                                                  | Status | Implementation                                                   |
| Design                                                                  | Status | Implementation                                                   |
|:------------------------------------------------------------------------|:-------|:-----------------------------------------------------------------|
| [Continuous Integration & Quality Gate Infrastructure](ci/ci-design.md) | Stale  | `control-rs-ci/`, `.cargo/gate.toml`, `.github/workflows/CI.yml` |

## vv

| Design                                                                           | Status | Implementation                                   |
|:---------------------------------------------------------------------------------|:-------|:-------------------------------------------------|
| [Cross-Compare Harness & HDF5 Comparison System](vv/cross-compare-design.md)     | Stale  | `control-rs-compare/`, `.cargo/compare.toml`     |
| [Range-Valued On-Target Property Tests](vv/ets-prop-test-design.md)              | Stale  | Not planned; property tests are written manually |
| [Requirement Traceability Infrastructure](vv/requirement-traceability-design.md) | Stale  | `control-rs-ci/src/trace.rs`                     |

## control-toolboxes

| Design                                                                     | Status    | Implementation           |
|:---------------------------------------------------------------------------|:----------|:-------------------------|
| [Classical Control Toolbox](control-toolboxes/classical-control-design.md) | In review | `src/classical_control/` |