# `control-rs` Roadmap

[Documentation index](README.md)

- [x] **PR1**: Workspace Foundations & Strict Lint Baseline
- [x] **PR2**: ETS Host Extraction & Terminal Dashboard
  ([#55](https://github.com/Dyse-Industries/control-rs/pull/55); [ets-host](ets-host/ets-host-design.md), [tui](tui/tui-design.md))
- [x] **PR3**: CI Quality Gate Infrastructure (`control-rs-ci`)
  ([#58](https://github.com/Dyse-Industries/control-rs/pull/58); [ci](ci/ci-design.md))
    - [x] **PR3-1**: Multi-Oracle Differential Test Harness
      (`control-rs-compare`)
      ([#59](https://github.com/Dyse-Industries/control-rs/pull/59); [cross-compare](vv/cross-compare-design.md))
    - [x] **PR3-2**: CI Gate — Performance & Benchmarking (`regression`)
      ([#60](https://github.com/Dyse-Industries/control-rs/pull/60); [ci](ci/ci-design.md)
      §4.7)
    - [x] **PR3-3**: CI Cleanup & ETS Gate — Gate Selection vs Policy,
      Process-Tree Timeouts, Bare-Metal Target Build & QEMU ETS Execution
      (`control-rs-ci`)
      ([#63](https://github.com/Dyse-Industries/control-rs/pull/63); [ci](ci/ci-design.md))
    - [x] **PR3-4**: CI Gate — Requirement Traceability (`trace`) (
      [requirement-traceability](vv/requirement-traceability-design.md))
    - [x] **PR3-5**: Technical Debt & Verification Hardening
      ([#67](https://github.com/Dyse-Industries/control-rs/pull/67); [ci](ci/ci-design.md) rev 1.33)
        - [x] Exclusive `pre`/`post` stages with the `fetch` gate; groups in
          declaration order; bounded concurrency (`max_jobs`)
        - [x] Configuration validation, versioned gate outcomes, budget-safe
          report, fail-closed `valgrind`, retired schema and dead code removed
    - [x] **PR3-6**: Formal Verification & Dynamic Analysis Gates (`kani`, `miri`)
      ([#69](https://github.com/Dyse-Industries/control-rs/pull/69); gates failing at merge; repaired in PR3-7)
    - [x] **PR3-7**: Main Repair — Patch Failing Gates
        - Reproduce each failing gate on `main` with `cargo ci` and patch it
          minimally, with no redesign
    - [x] **PR3-8**: Requirement Traceability Redesign (`trace`)
        - `Target`-cell traceability replaces the `#[req]` macro; remove
          `control-rs-trace-macros`, `trace-marks` and `trace/marks.rs`
    - [x] **PR3-9**: Workspace Audit & Cleanup
        - Review PR3 against its design docs, reconcile docs, gate config and
          tests, and remove stray additions
- [ ] **PR4**: Classical Control Synthesis & Math Core (`src/classical_control`)
- [ ] **PR5**: Modern Control Toolbox (`src/modern_control`)
- [ ] **PR6**: Robust Control Analysis (`src/robust_control`)
- [ ] **PR7**: Nonlinear Control Synthesis & Estimators (`src/nonlinear_control`)
- [ ] **PR8**: System Identification (SysID) & Frequency Estimation
  (`src/sysid`)
- [ ] **PR9**: Safety Validation & Run-Time Assurance (`src/validation`)
- [ ] **PR10**: Design Hardening, Trace Wiring and Test Updates
    - Follows PR4 to PR9 so hardening covers every identified feature
- [ ] **PR11**: Hardware Acceleration & Architecture Subprograms (traits in
  `src/math/subprograms`; accelerated backends as examples in
  `examples/subprograms`) ([subprograms](math/subprograms-design.md))
- [ ] **PR12**: CI Workflow Hardening & Physical-Target ETS Runners (Teensy 4.1)
    - Teensy 4.1 `target-build` gate (`ci-design.md` FR-17; ITCM layout)
    - Workflow: build cache, pinned actions and tools, top-level
      `permissions:`, no duplicate `cargo doc` in the report job
- [ ] **PR13**: Automated Git Release System & Release Pipeline
- [ ] **PR14**: Flight Examples, Documentation Consolidation & Initial Release
  (`v0.1.0`)
