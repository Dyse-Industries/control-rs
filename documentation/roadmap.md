# `control-rs` Roadmap

[Documentation index](README.md)

Two tracks run in parallel. Track A (control toolboxes) is the priority. It is
a serial stack: a branch merges only after its design is approved, implemented
and green. Track B (ETS overhaul and examples) does not gate Track A. Release
work follows both tracks.

## Merged

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
      ([#74](https://github.com/Dyse-Industries/control-rs/pull/74))
        - Reproduce each failing gate on `main` with `cargo ci` and patch it
          minimally, with no redesign
    - [x] **PR3-8**: Requirement Traceability Redesign (`trace`)
        - `Target`-cell traceability replaces the `#[req]` macro; remove
          `control-rs-trace-macros`, `trace-marks` and `trace/marks.rs`
    - [x] **PR3-9**: Workspace Audit & Cleanup
      ([#75](https://github.com/Dyse-Industries/control-rs/pull/75); mutant
      resolution in [#68](https://github.com/Dyse-Industries/control-rs/pull/68))
        - Review PR3 against its design docs, reconcile docs, gate config and
          tests, and remove stray additions
    - [x] **PR3-10**: ADRs & Requirement Tracer Updates
      ([#76](https://github.com/Dyse-Industries/control-rs/pull/76),
      [#79](https://github.com/Dyse-Industries/control-rs/pull/79),
      [#80](https://github.com/Dyse-Industries/control-rs/pull/80))
        - ADRs for cross-cutting decisions, ADR checks in the tracer and the
          tracer gap patch
    - [x] **PR3-11**: Design Status & AI-Use Tracking
      ([#94](https://github.com/Dyse-Industries/control-rs/pull/94))
        - `Approval` phases and `Doc Status` badges on every design document

## Track A: Control toolboxes (priority)

Stack order is PR4, PR5, PR6, PR7, PR8, PR9. Each design is a Draft until it is
signed off. No stack branch merges until its design is approved, implemented
and `cargo ci` is green.

- [ ] **PR4**: Classical Control Synthesis & Math Core (`src/classical_control`)
    - [ ] **PR4-1**: Design (`control-toolboxes/classical-control-design.md`);
      awaiting review and sign-off
    - [ ] **PR4-2**: Implementation; written on the stack, merges after PR4-1
- [ ] **PR5**: Modern Control Toolbox (`src/modern_control`)
    - Design drafted (`control-toolboxes/modern-control-design.md`); needs the
      Gehrd, Orghr, Hseqr, Trsen and Trsyl kernels (see PR11)
- [ ] **PR6**: Robust Control Analysis (`src/robust_control`)
    - Design drafted (`control-toolboxes/robust-control-design.md`); needs
      `modern_control::schur` from PR5
- [ ] **PR7**: Nonlinear Control Synthesis & Estimators (`src/nonlinear_control`)
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

The overhaul replaces the lifecycle-suite branch (`mdsdev/ets-loop-suite`),
which is archived instead of merged. Implementation PRs build from `main`.

- [ ] **ETS-1**: Design Overhaul (docs only)
    - Executive and runner object model, single wire revision 2, ADR-0002,
      ETS design documents consolidated under `documentation/ets/`
    - Gates ETS-2 to ETS-6 only
- [ ] **ETS-2**: Target Executive & Runners (`control-rs-ets`)
    - Dispatcher with runners, parameter groups, lifecycle tasks, fault path
- [ ] **ETS-3**: Host Session & Runs (`control-rs-ets-host`)
    - `Session` over a `Link`; case runs and task runs; observers
    - Starts in parallel with ETS-2; task runs need the ETS-2 task runner
- [ ] **ETS-4**: Frontends on `Session` (`control-rs-tui`, `control-rs-ci`)
    - Task row beside cases; Run all excludes tasks; no task runs in CI
- [ ] **ETS-5**: Overhaul Examples & Boundary Measurement
    - QEMU lockstep task and Teensy tasks; DWT cost of the task boundary
- [ ] **EX-1**: Board Examples
    - ESP32-C6 (standalone showcase), Arduino Uno R4 Minima, Seeed XIAO
      nRF52840, Seeed XIAO ESP32-S3 and Teensy 4.0 and 4.1
    - Independent of the overhaul by choice; moves onto ETS-2 as the API
      changes

## Independent

- [ ] **PR11**: Hardware Acceleration & Architecture Subprograms (traits in
  `src/math/subprograms`; accelerated backends as examples in
  `examples/subprograms`) ([subprograms](math/subprograms-design.md))
    - Traits and backends are independent of both tracks
    - Open: whether the kernels required by PR5, PR6 and PR8 land in those PRs
      or here

## Release

- [ ] **PR12**: CI Workflow Hardening & Physical-Target ETS Runners (Teensy 4.1)
    - Follows EX-1 for the board list and ETS-2 for the runner API
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
