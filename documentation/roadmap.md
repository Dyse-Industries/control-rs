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
which is archived instead of merged. Each PR follows one component of the
current separation and lands one or more complete design documents with their
implementation, `VC` tests and examples. A document is Approved and traced at
merge and no later PR adds requirements to it. No design-only PRs. PRs stack
in dependency order, ETS-1 first.

- [ ] **ETS-1**: Link and Wire Schema (`control-rs-ets`, `control-rs-ets-host`)
  ([#103](https://github.com/Dyse-Industries/control-rs/pull/103), open)
    - Design: `ets/ets-link-design.md`, `ets/ets-overview.md`
    - Whole-message `Link`; stream, USB bulk and CAN links behind features;
      byte stream trait and `embedded-io` adapter; host counterparts
    - Schema revision 2 (`PROTOCOL_VERSION = 2`): settings dictionary, task
      and supervision variants, panic capture; golden vectors
    - Existing server, session and TUI moved onto the link without new
      behavior; `link_discovery` example; QEMU and Teensy 4.1 on the stream link
- [ ] **ETS-2**: Target Executive and Suites (`control-rs-ets`, `control-rs-macros`)
  ([#104](https://github.com/Dyse-Industries/control-rs/pull/104), open)
    - Design: `ets/ets-executive-design.md`, `ets/embedded-test-server-design.md`,
      `ets/test-suite-design.md`, `ets/macros-design.md`,
      `ets/cpu-profiler-design.md`; ADR-0002
    - Dispatcher, `Context` and runner; `Atomic`, `Lockstep`, `FreeRunning`,
      `Periodic` and `Triggered`; `LifecycleTask`, `TaskSlot`, `Packet`
      derive; named settings (`SettingCell`, static dispatch); board binding;
      fault path and `safe_state`; step jitter profiling; atomic-free build
    - Examples: QEMU and Teensy 4.1 firmware with cases, settings and tasks;
      DWT boundary and jitter measurement
- [ ] **ETS-3**: Host Session and Runs (`control-rs-ets-host`, `control-rs-ci`)
  ([#105](https://github.com/Dyse-Industries/control-rs/pull/105), open)
    - Design: `ets/ets-host-design.md`
    - `Session`, case and task runs, `TaskSim`, `RunObserver`, device state
      tracking, unresponsive-run recovery, release-paced runs, settings over
      the session without the ELF; CI runs cases only
    - Examples: headless QEMU lockstep task with a host plant; Teensy 4.1
      periodic and triggered runs
- [ ] **ETS-4**: Terminal Dashboard (`control-rs-tui`)
  ([#106](https://github.com/Dyse-Industries/control-rs/pull/106), open)
    - Design: `ets/tui-design.md`
    - Tasks beside cases, setting console, fault and link status,
      release-paced controls, independent monitoring
    - Examples: TUI sessions against QEMU and Teensy 4.1
- [ ] **ETS-5**: Board Examples
  ([#107](https://github.com/Dyse-Industries/control-rs/pull/107), open)
    - Design: new `ets/board-examples-design.md` (per-board link, throughput
      and fault reporting requirements)
    - [ ] ESP32-C6 (standalone showcase), Arduino Uno R4 Minima and Seeed XIAO
      nRF52840 on the stream link
      ([#107](https://github.com/Dyse-Industries/control-rs/pull/107))
    - [ ] Seeed XIAO ESP32-S3 and Teensy 4.0 (not started)

## Independent

- [ ] **PR11**: Hardware Acceleration & Architecture Subprograms (traits in
  `src/math/subprograms`; accelerated backends as examples in
  `examples/subprograms`) ([subprograms](math/subprograms-design.md))
    - Traits and backends are independent of both tracks
    - Open: whether the kernels required by PR5, PR6 and PR8 land in those PRs
      or here
- [ ] **PR3-13**: Tracer & CI Follow-ups (open)
    - Tracer method overhaul: one method table, structured compare and JUnit
      methods ([#131](https://github.com/Dyse-Industries/control-rs/pull/131))
    - Link graph built once per process, tracer-link-graph FR-6
      ([#132](https://github.com/Dyse-Industries/control-rs/pull/132))
    - Bounded SCC traversal in the link graph
      ([#130](https://github.com/Dyse-Industries/control-rs/pull/130))
    - Regression verdict floor applied to the added time
      ([#133](https://github.com/Dyse-Industries/control-rs/pull/133))

## Release

- [ ] **PR12**: CI Workflow Hardening & Physical-Target ETS Runners (Teensy 4.1)
    - Follows ETS-5 for the board list and ETS-2 for the runner API
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
