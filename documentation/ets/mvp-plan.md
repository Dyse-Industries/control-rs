# MVP Plan

![Date Badge](https://img.shields.io/badge/Date-September_9,_2026-blue)
![Type Badge](https://img.shields.io/badge/Type-Plan-lightgrey)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

This document outlines the suggested order and estimated time for implementing
the features described in the design documents.

Status (September 9, 2026): items 1–5 are substantially shipped: the
`CPUProfiler` trait and target implementations, the `HostComms` framing and
postcard schemas (`control-rs-ets/src/comms.rs`), the `SuiteDescriptor` linker
mechanics, the `.ets_test_suites` discovery macros and the host-side
`ETSBridge` with headless orchestration. Items 6 and 7 exist as code inside
the deprecated `control-rs-xtask` and are pending extraction into their own
crates. Remaining work is the open steps of each design's Development Plan
(watchdog multiplexing, driver integration, TUI polish, CI Tiers 1 and 2).

| Order | Document                         | Task                                                                            | Estimated Time (days) | Estimated LOC | Confidence / Difficulty |
|-------|----------------------------------|---------------------------------------------------------------------------------|-----------------------|---------------|-------------------------|
| 1     | `cpu-profiler-design.md`         | Implement the `CPUProfiler` trait.                                              | 0.5                   | 50            | High / Low              |
| 2     | `host-comm-design.md`            | Implement the `HostComms` trait and command/telemetry data structures.          | 2                     | 300           | Medium / Medium         |
| 3     | `test-suite-design.md`           | Define `SuiteDescriptor` and implement linker script mechanics.                 | 2                     | 250           | Medium / Medium         |
| 4     | `embedded-test-server-design.md` | Build the on-target ETS.                                                        | 3                     | 500           | Low / High              |
| 5     | `../macros/macros-design.md`     | Develop `#[ets_suite]` and `#[ets_setup]` procedural macros.                    | 3                     | 400           | Low / High              |
| 6     | `../ets-host/ets-host-design.md` | Extract the host session library: transport, framing, session, headless runner. | 4                     | 900           | Medium / Medium         |
| 7     | `../tui/tui-design.md`           | Develop the host-side Terminal User Interface over `control-rs-ets-host`.       | 4                     | 800           | Medium / Medium         |
| 8     | `ets-overview.md`                | Integrate ETS, macros and TUI into the `control-rs` crate.                      | 2                     | 200           | High / Medium           |
| 9     | `../ci/ci-design.md`             | Set up the CI pipeline for builtin ETS tests.                                   | 2                     | 150           | High / Low              |
