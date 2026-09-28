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
    - [ ] **PR3-5**: Technical Debt & Verification Hardening
      ([ci](ci/ci-design.md) rev 1.33)
        - [x] Exclusive `pre`/`post` stages with the `fetch` gate; groups in
          declaration order; bounded concurrency (`max_jobs`)
        - [x] Configuration validation, versioned gate outcomes, budget-safe
          report, fail-closed `valgrind`, retired schema and dead code removed
    - [ ] **PR3-6**: Formal Verification & Dynamic Analysis Gates (`kani`,
      `miri`)
      ([ci](ci/ci-design.md), [requirement-traceability](vv/requirement-traceability-design.md))
        - Revive `kani` and `miri` quality gates in `gate.toml` with dedicated
          verification harnesses
        - Requirement tracing integration: tracer derives condition status
          by reading per-item Kani and test results in post, verifying that
          formal proofs and dynamic analysis agree with declared conditions
- [ ] **PR4**: Classical Control Synthesis & Math Core (`src/classical_tools`)
- [ ] **PR5**: Modern Control Toolbox & State Observers (`src/modern_control`)
- [ ] **PR6**: System Identification (SysID) & Frequency Estimation
  (`src/sysid`)
- [ ] **PR7**: Safety Validation & Run-Time Assurance (`src/validation`)
- [ ] **PR8**: Hardware Acceleration & Architecture Subprograms (traits in
  `src/math/subprograms`; accelerated backends as examples in
  `examples/subprograms`) ([subprograms](math/subprograms-design.md))
- [ ] **PR9**: CI Workflow Hardening & Physical-Target ETS Runners (Teensy 4.1)
    - Teensy 4.1 `target-build` gate (`ci-design.md` FR-17; ITCM layout)
    - Workflow: build cache, pinned actions and tools, top-level
      `permissions:`, no duplicate `cargo doc` in the report job
- [ ] **PR10**: Automated Git Release System & Release Pipeline
- [ ] **PR11**: Flight Examples, Documentation Consolidation & Initial Release
  (`v0.1.0`)
