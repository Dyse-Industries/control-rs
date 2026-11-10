# `control-rs` Roadmap

[Documentation index](README.md)

Two tracks run in parallel. Track A (control toolboxes) is the priority. It is
a serial stack: a branch merges only after its design is approved, implemented
and green. Track B (ETS overhaul and examples) does not gate Track A. Release
work follows both tracks.

## Merged

- [x] **PR1**: Workspace Foundations & Strict Lint Baseline
- [x] **PR2**: ETS Host Extraction & Terminal Dashboard
  ([#55](https://github.com/Dyse-Industries/control-rs/pull/55); [ets-host](ets/ets-host-design.md), [tui](tui/tui-design.md))
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
      ([#67](https://github.com/Dyse-Industries/control-rs/pull/67); [ci](ci/ci-design.md)
      rev 1.33)
        - [x] Exclusive `pre`/`post` stages with the `fetch` gate; groups in
          declaration order; bounded concurrency (`max_jobs`)
        - [x] Configuration validation, versioned gate outcomes, budget-safe
          report, fail-closed `valgrind`, retired schema and dead code removed
    - [x] **PR3-6**: Formal Verification & Dynamic Analysis Gates (`kani`,
      `miri`)
      ([#69](https://github.com/Dyse-Industries/control-rs/pull/69); gates
      failing at merge; repaired in PR3-7)
    - [x] **PR3-7**: Main Repair — Patch Failing Gates
      ([#74](https://github.com/Dyse-Industries/control-rs/pull/74))
        - Reproduce each failing gate on `main` with `cargo ci` and patch it
          minimally, with no redesign
    - [x] **PR3-8**: Requirement Traceability Redesign (`trace`)
        - `Target`-cell traceability replaces the `#[req]` macro; remove
          `control-rs-trace-macros`, `trace-marks` and `trace/marks.rs`
    - [x] **PR3-9**: Workspace Audit & Cleanup
      ([#75](https://github.com/Dyse-Industries/control-rs/pull/75); mutant
      resolution
      in [#68](https://github.com/Dyse-Industries/control-rs/pull/68))
        - Review PR3 against its design docs, reconcile docs, gate config and
          tests and remove stray additions
    - [x] **PR3-10**: ADRs & Requirement Tracer Updates
      ([#76](https://github.com/Dyse-Industries/control-rs/pull/76),
      [#79](https://github.com/Dyse-Industries/control-rs/pull/79),
      [#80](https://github.com/Dyse-Industries/control-rs/pull/80))
        - ADRs for cross-cutting decisions, ADR checks in the tracer and the
          tracer gap patch
    - [x] **PR3-11**: Design Status & AI-Use Tracking
      ([#94](https://github.com/Dyse-Industries/control-rs/pull/94))
        - `Approval` phases and `Doc Status` badges on every design document
    - [x] **PR3-12**: CI & Tracer Maintenance
      ([#121](https://github.com/Dyse-Industries/control-rs/pull/121),
      [#122](https://github.com/Dyse-Industries/control-rs/pull/122),
      [#125](https://github.com/Dyse-Industries/control-rs/pull/125),
      [#127](https://github.com/Dyse-Industries/control-rs/pull/127);
      [#124](https://github.com/Dyse-Industries/control-rs/pull/124) reverted in
      [#128](https://github.com/Dyse-Industries/control-rs/pull/128))
        - Path-selective gate triggers, stale CI report cleanup, regression
          warning bound
        - Condition `link` method and link graph
          ([tracer-link-graph](vv/tracer-link-graph-design.md))
    - [x] **PR3-13**: Tracer Method Overhaul & Link Graph Optimizations
      ([#130](https://github.com/Dyse-Industries/control-rs/pull/130),
      [#131](https://github.com/Dyse-Industries/control-rs/pull/131),
      [#132](https://github.com/Dyse-Industries/control-rs/pull/132),
      [#133](https://github.com/Dyse-Industries/control-rs/pull/133);
      [#134](https://github.com/Dyse-Industries/control-rs/pull/134) reverted in
      [#138](https://github.com/Dyse-Industries/control-rs/pull/138))
        - Tracer method overhaul: unified method table, structured compare and
          JUnit methods
        - Link graph caching (built once per process), bounded SCC traversal
          and regression verdict floor
- [x] **PR4**: Classical Control Synthesis & Math Core (`src/classical_control`)
  ([#88](https://github.com/Dyse-Industries/control-rs/pull/88); stacked
  [#92](https://github.com/Dyse-Industries/control-rs/pull/92) and
  [#113](https://github.com/Dyse-Industries/control-rs/pull/113))
    - [x] **PR4-1**: Design (`control-toolboxes/classical-control-design.md`);
      in review against the implementation
    - [x] **PR4-2**: Implementation
    - [x] **PR4-3**: Fine-tuning and polish: `fixed-num-design.md` FR-8 and
      ADR-0003 for `MulAcc`; `margin`, `root_locus`, `Section`; phase-crossover
      bracketing, non-finite `Pid` gains
    - [x] Post-merge scope and correctness pass
      ([#119](https://github.com/Dyse-Industries/control-rs/pull/119))

## Track A: Control toolboxes (priority)

Stack order is PR5, PR6, PR7, PR8, PR9. Each design is a Draft on its
`mdsdev/prN-1-planning` branch until it is signed off. No stack branch merges
until its design is approved, implemented and `cargo ci` is green.

- [ ] **PR5**: Modern Control Toolbox (`src/modern_control`)
    - Design drafted (`control-toolboxes/modern-control-design.md`); needs the
      Gehrd, Orghr, Hseqr, Trsen and Trsyl kernels (see PR11)
- [ ] **PR6**: Robust Control Analysis (`src/robust_control`)
    - Design drafted (`control-toolboxes/robust-control-design.md`); needs
      `modern_control::schur` from PR5
- [ ] **PR7**: Nonlinear Control Synthesis & Estimators
  (`src/nonlinear_control`)
    - Design drafted (`control-toolboxes/nonlinear-control-design.md`)
- [ ] **PR8**: System Identification (SysID) & Frequency Estimation
  (`src/sysid`)
    - Design drafted (`control-toolboxes/sysid-design.md`); needs Gesvd and the
      PR5 Schur decomposition
- [ ] **PR9**: Safety Validation & Run-Time Assurance (`src/validation`)
    - Design drafted (`vv/validation-tools-design.md`)
- [ ] **PR10**: Design Hardening, Trace Wiring and Test Updates
    - Follows PR4 to PR9 so hardening covers every identified feature

## Track B: ETS overhaul and examples (parallel, non-blocking)

The authoritative lead design document for the target-side runtime is
`documentation/ets/embedded-test-server-design.md`, supported by the overall
subsystem development integration plan in `documentation/ets/ets-overview.md`.

- [ ] **ETS-1**: Target Suites, Macros and Profiler (`control-rs-ets`,
  `control-rs-macros`)
    - Branch: `mdsdev/ets-1-suites-profiler`
    - Design: `ets/embedded-test-server-design.md`,
      `ets/cpu-profiler-design.md`, `ets/test-suite-design.md`,
      `ets/macros-design.md`
- [ ] **ETS-2**: Target `HostComms` and Wire Protocol (`control-rs-ets`)
  (replaces [#103](https://github.com/Dyse-Industries/control-rs/pull/103))
    - Branch: `mdsdev/ets-2-host-comms` (base: `main`, parallel with ETS-1)
    - Design: `ets/host-comm-design.md`
- [ ] **ETS-3**: Target `Server` & Execution Platform (`control-rs-ets`)
  (replaces [#104](https://github.com/Dyse-Industries/control-rs/pull/104))
    - Branch: `mdsdev/ets-3-target-server` (base: `main` after ETS-1 and ETS-2)
    - Design: `ets/embedded-test-server-design.md` (lead design),
      `ets/server-runtime-design.md`
- [ ] **ETS-4**: Host `Session` and Headless Runner (`control-rs-ets-host`,
  `control-rs-ci`)
  (replaces [#105](https://github.com/Dyse-Industries/control-rs/pull/105))
    - Branch: `mdsdev/ets-4-host-session` (base: `main` after ETS-3)
    - Design: `ets-host/ets-host-design.md`
- [ ] **ETS-5**: Host Terminal Dashboard (`control-rs-tui`)
  (replaces [#106](https://github.com/Dyse-Industries/control-rs/pull/106))
    - Branch: `mdsdev/ets-5-tui` (base: `main` after ETS-4)
    - Design: `tui/tui-design.md`
- [ ] **ETS-6**: Board Examples
  ([#107](https://github.com/Dyse-Industries/control-rs/pull/107), open)
    - Branch: `mdsdev/ets-6-board-examples` (base: `main` after ETS-3, parallel
      with ETS-4 and ETS-5)
    - Design: `ets/board-examples-design.md`

## Independent

- [ ] **PR11**: Hardware Acceleration & Architecture Subprograms (traits in
  `src/math/subprograms`; accelerated backends as examples in
  `examples/subprograms`) ([subprograms](math/subprograms-design.md))
    - Traits and backends are independent of both tracks
    - Open: whether the kernels required by PR5, PR6 and PR8 land in those PRs
      or here

## Release

- [ ] **PR12**: CI Workflow Hardening & Physical-Target ETS Runners (Teensy 4.1)
    - Follows ETS-6 for the board list and ETS-3 for the runner API
    - Teensy 4.1 `target-build` gate (`ci-design.md` FR-17; ITCM layout)
    - Workflow: build cache, pinned actions and tools, top-level
      `permissions:`, no duplicate `cargo doc` in the report job
- [ ] **PR13**: Automated Git Release System & Release Pipeline
- [ ] **PR14**: Flight Examples, Documentation Consolidation & Initial Release
  (`v0.1.0`)
    - Follows PR10, PR11, PR12 and PR13

## Deferred

- Hosted-RTOS integration of the ETS
- Concurrent multi-task runs on one target
- HDF5 logging in the ETS host
- Layer 1/2+ hardware-in-the-loop harness design
- Range-valued property tests ([ets-prop-test](vv/ets-prop-test-design.md))
