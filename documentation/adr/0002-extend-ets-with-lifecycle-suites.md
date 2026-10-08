# ADR-0002: Extend the Embedded Test Server with lifecycle suites

![Date Badge](https://img.shields.io/badge/Date-October_7,_2026-blue)
![Status Badge](https://img.shields.io/badge/ADR%20Status-Proposed-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

## Context

`control-rs-ets` runs atomic cases: the server calls each case once inside a
critical section, profiles its cycles and stack, and reports the result when
it returns [1]. Suites are discovered from a linker section, their settings
are edited live, and `control-rs-ets-host` and `control-rs-tui` drive them
over one framed link. Hardware-in-the-loop and simulation-in-the-loop testing
needs code that runs continuously with interrupts live, exchanges one packet
each way with the host per step, can be stopped by the host and always tears
down the hardware it configured. An atomic case cannot express this. Does
continuous execution extend the existing server, or does it get a server of
its own?

---

## Decision

Extend the existing ETS server with lifecycle suites [2]. A lifecycle suite
is an `#[ets_suite]` module that, beside its settings and any number of atomic
cases, provides one lifecycle case: a setup, a step, a reset and a teardown,
run as a loop. Users deploy cases and lifecycle cases in one image, and the
TUI shows them side by side as rows of the same suite. The server gains only
the run states (setup, step, step boundary, reset and teardown) and the
commands and telemetry that drive them; atomic cases, suite descriptors,
indices and profiling are unchanged.

---

## Consequences

- Good: one firmware image, entrypoint, link, host library and console serve
  cases and loops, and a loop is tuned through the same suite settings as the
  cases that test the same hardware.
- Good: discovery, settings, framing, panic handling and reset recovery are
  reused rather than duplicated.
- Good: the command loop regains control after every run, so no run outlives
  its teardown.
- Bad: the wire protocol moves to revision 2, and `Command` gains a lifetime,
  which changes the `poll_command` signature of every `HostComms`
  implementor.
- Bad: the server owns a second mode; while a loop run is active it refuses
  atomic cases, discovery and a second run.
- Bad: loops inherit the server's cooperative control, so a step that
  never returns is stopped only by a reset.
- Follow-up: implementation, owned by `loop-suite-design.md` §9.
- Follow-up: lossless chunked command reception, a prerequisite, owned by
  `host-comm-design.md` §9 Step 7.

---

## Rejected Options

- A separate loop server or firmware: duplicates discovery, settings,
  framing, panic handling and host tooling, and cases and loops could not
  share one image or one TUI.
- A loop as a standalone suite with its own section and identifiers: its
  settings and tree node would be separate from the cases of the same suite.
- A host-owned loop over atomic cases: each step costs a host round trip and
  runs with interrupts masked.
- A kind flag on the existing suite descriptor: changes the descriptor layout
  and every existing suite index.
- Lifecycle and goal semantics in ETS: production runtime concerns rather
  than test concerns; loops add run states only.

---

## References

[1] `control-rs`, "Embedded Test Server,"
`documentation/ets/embedded-test-server-design.md`, Oct. 2026.

[2] `control-rs`, "Lifecycle Suites," `documentation/ets/loop-suite-design.md`,
rev. 1.7, Oct. 2026.
