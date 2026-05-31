# MVP Plan

This document outlines the suggested order and estimated time for implementing
the features described in the design documents.

| Order | Document                          | Task                                                                   | Estimated Time (days) | Estimated LOC | Confidence / Difficulty |
|-------|-----------------------------------|------------------------------------------------------------------------|-----------------------|---------------|-------------------------|
| 1     | `client-clock-design-doc.md`      | Implement the `ClientClock` trait.                                     | 0.5                   | 50            | High / Low              |
| 2     | `host-comm-design-doc.md`         | Implement the `HostComms` trait and command/telemetry data structures. | 2                     | 300           | Medium / Medium         |
| 3     | `test-suite-design-doc.md`        | Define `SuiteDescriptor` and implement linker script mechanics.        | 2                     | 250           | Medium / Medium         |
| 4     | `hil-runner-design-doc.md`        | Build the on-target HIL runner server.                                 | 3                     | 500           | Low / High              |
| 5     | `control-rs-macros-design-doc.md` | Develop `#[hil_suite]` and `#[hil_setup]` procedural macros.           | 3                     | 400           | Low / High              |
| 6     | `tui-design-doc.md`               | Develop the host-side Terminal User Interface.                         | 4                     | 800           | Medium / Medium         |
| 7     | `control-rs-hil-overview.md`      | Integrate the HIL runner, macros and TUI into the `control-rs` crate.  | 2                     | 200           | High / Medium           |
| 8     | `control-rs-ci-design-doc.md`     | Set up the CI pipeline for builtin HIL tests.                          | 2                     | 150           | High / Low              |
|       |                                   | **Total**                                                              | **18.5**              | **2650**      |                         |