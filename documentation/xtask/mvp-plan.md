# MVP Plan

This document outlines the suggested order and estimated time for implementing
the features described in the design documents.

Status (August 6, 2026): items 1–5 are substantially shipped — the
`CPUProfiler` trait and target implementations, the `HostComms` framing and
postcard schemas (`control-rs-hil/src/comms.rs`), the `SuiteDescriptor`/linker
mechanics, the `.hil_test_suites` discovery macros and the host-side
`ServerBridge`/headless orchestration. Remaining MVP work is concentrated in
the open steps of each design doc's Development Plan (watchdog multiplexing,
driver integration, TUI polish, CI Tiers 1–2).

| Order | Document                          | Task                                                                   | Estimated Time (days) | Estimated LOC | Confidence / Difficulty |
|-------|-----------------------------------|------------------------------------------------------------------------|-----------------------|---------------|-------------------------|
| 1     | `cpu-profile-utils-design-doc.md` | Implement the `CPUProfiler` trait.                                     | 0.5                   | 50            | High / Low              |
| 2     | `host-comm-design-doc.md`         | Implement the `HostComms` trait and command/telemetry data structures. | 2                     | 300           | Medium / Medium         |
| 3     | `test-suite-design-doc.md`        | Define `SuiteDescriptor` and implement linker script mechanics.        | 2                     | 250           | Medium / Medium         |
| 4     | `hil-server-design-doc.md`        | Build the on-target HIL Server.                                        | 3                     | 500           | Low / High              |
| 5     | `control-rs-macros-design-doc.md` | Develop `#[hil_suite]` and `#[hil_setup]` procedural macros.           | 3                     | 400           | Low / High              |
| 6     | `tui-design-doc.md`               | Develop the host-side Terminal User Interface.                         | 4                     | 800           | Medium / Medium         |
| 7     | `control-rs-hil-overview.md`      | Integrate the HIL Server, macros and TUI into the `control-rs` crate.  | 2                     | 200           | High / Medium           |
| 8     | `control-rs-ci-design-doc.md`     | Set up the CI pipeline for builtin HIL tests.                          | 2                     | 150           | High / Low              |
