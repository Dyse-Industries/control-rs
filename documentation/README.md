# Documentation

Design documents, standards and planning for the `control-rs` workspace.
[Workspace](../README.md) · [Contributing](../CONTRIBUTING.md)

| Document | Purpose |
|:--|:--|
| [`dependencies.md`](dependencies.md) | Every external tool, what needs it and how to install it |
| [`development-guide.md`](development-guide.md) | Cargo aliases, CI and ETS workflows |
| [`design-template.md`](design-template.md) | Template for new design documents |
| [`adr/`](adr/README.md) | Architecture Decision Records, their index and [template](adr-template.md) |
| [`doc-standards.md`](doc-standards.md) | Rustdoc, ETS and ADR documentation policy |
| [`roadmap.md`](roadmap.md) | Planned pull requests |

## math

| Design | Status | Implementation |
|:--|:--|:--|
| [Crate-Wide Error Module](math/error-design.md) | Stale | `src/math/` |
| [Fixed-Point Scalar Type](math/fixed-num-design.md) | Stale | `src/math/fixed_num.rs` |
| [Numeric Trait Hierarchy](math/num-traits-design.md) | Stale | `src/math/num_traits.rs` |
| [Numeric Types](math/num-types-design.md) | Stale | `src/math/num_types.rs` |
| [Storage Backends & Data Layouts](math/storage-design.md) | Stale | `src/math/storage.rs` |
| [Linear Algebra Subprograms](math/subprograms-design.md) | Stale | `src/math/subprograms.rs`, `examples/subprograms/` |

## numerical-models

| Design | Status | Implementation |
|:--|:--|:--|
| [Numerical Models Integration & Examples](numerical-models/numerical-models-design.md) | Stale | `control-rs-verification/` |
| [Matrix Type & Structural Specializations](numerical-models/matrix-design.md) | Stale | `src/matrix/` |
| [Polynomial Type](numerical-models/polynomial-design.md) | Stale | `src/polynomial/` |
| [State-Space Model Type](numerical-models/state-space-design.md) | Stale | `src/state_space/` |
| [Tensor Type & Low-Cost Inference](numerical-models/tensor-design.md) | Stale | `src/tensor/` |
| [Transfer Function Type](numerical-models/transfer-function-design.md) | Stale | `src/transfer_function/` |

## ets

[Overview](ets/ets-overview.md)

| Design | Status | Implementation |
|:--|:--|:--|
| [Embedded Test Server](ets/embedded-test-server-design.md) | Approved | `control-rs-ets/src/server.rs` |
| [ETS Link](ets/ets-link-design.md) | Draft | `control-rs-ets/src/link/`, `control-rs-ets/src/wire.rs` |
| [`CPUProfiler`](ets/cpu-profiler-design.md) | Approved | `control-rs-ets/src/profiler.rs` |
| [Exportable Test Suites](ets/test-suite-design.md) | Approved | `src/**/tests/` (`ets` feature) |

## macros

| Design | Status | Implementation |
|:--|:--|:--|
| [Procedural Macros for Distributed Test Discovery](macros/macros-design.md) | Approved | `control-rs-macros/` |

## ets-host

| Design | Status | Implementation |
|:--|:--|:--|
| [Host ETS Library](ets-host/ets-host-design.md) | Approved | `control-rs-ets-host/` |

## tui

| Design | Status | Implementation |
|:--|:--|:--|
| [Terminal User Interface](tui/tui-design.md) | Approved | `control-rs-tui/` |

## ci

| Design | Status | Implementation |
|:--|:--|:--|
| [Continuous Integration & Quality Gate Infrastructure](ci/ci-design.md) | Stale | `control-rs-ci/`, `.cargo/gate.toml`, `.github/workflows/CI.yml` |

## vv

| Design | Status | Implementation |
|:--|:--|:--|
| [Cross-Compare Harness & HDF5 Comparison System](vv/cross-compare-design.md) | Stale | `control-rs-compare/`, `.cargo/compare.toml` |
| [Range-Valued On-Target Property Tests](vv/ets-prop-test-design.md) | Deferred | Not planned; property tests are written manually |
| [Requirement Traceability Infrastructure](vv/requirement-traceability-design.md) | Stale | `control-rs-ci/src/trace.rs` |
