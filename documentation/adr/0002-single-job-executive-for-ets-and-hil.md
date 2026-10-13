# ADR-0002: Single-Job Executive for ETS and HIL

![Date Badge](https://img.shields.io/badge/Date-October_12,_2026-blue)
![Status Badge](https://img.shields.io/badge/ADR%20Status-Proposed-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

## Context

`control-rs` verifies code across a layered hierarchy (Layer 0 host tests, Layer
0/1 target cases, Layer 1 HIL and SysId, Layer 1/2 firmware integration, Layer
1/2+ unmodified DUT; defined in `embedded-test-server-design.md`). The Embedded
Test Server (ETS) was hardened for Layer 0/1 atomic cases before its
architecture covered continuous tasks. Its server is one request-response loop,
its host session is an action and message state machine, and adding lifecycle
tasks by branching inside that loop couples every execution model to the
dispatcher. A separate HIL framework would duplicate board bring-up, drivers and
settings.


---

## Decision

Restructure the ETS as a single-job executive with an object model on both sides
of the link (`ets-executive-design.md`):

1. **One dispatcher, one runner.** The target `Server` resolves a command to a
   descriptor and a mode and constructs a `Runner<L, P, B, M: Mode>`. The runner
   owns the only execution loop. A plain case is a one-step `Atomic` executable.
   Lifecycle tasks run `Lockstep`, `FreeRunning`, `Periodic` or `Triggered`.
2. **Modes are types.** Per-run variants are type parameters or associated
   constants, not flags inside a loop. There are no trait objects on the target;
   the linker-section function pointer is the only indirection.
3. **Data-only descriptors.** Descriptors in flash name no server, transport or
   profiler type. Typed user contracts (`LifecycleTask`, `Packet`) are erased
   to buffers only inside the `#[ets_suite]` expansion; settings are typed
   primitive cells.
4. **Settings, not parameter groups.** Suite settings are named primitives in
   a dictionary, read by name or index and written by name with a typed value
   by the server at idle or at a step boundary; no atomics or sequence counters
   are used.
5. **Payload-level encoding.** The message schema has no channels. Each executable declares
   its packet encoding once at discovery; payloads stay opaque bytes with a step
   index.
6. **Fault containment without `Drop`.** A run phase and an associated
   `safe_state` function settle a fault. A board watchdog that the user starts
   in `#[ets_setup]` and the executive feeds through the board binding, or a
   host-triggered external reset, recovers a blocked step.
   The host tracks device state to detect unreported faults.
7. **One host session.** `Session<L: HostLink>` is the only writer on the link and
   lends it to a `CaseRun` or `TaskRun` per run. Run all is a sequence of
   `CaseRun`.
8. **Link-owned transport concerns.** The server exchanges whole messages
   through a `Link` that owns framing, bandwidth and any receive queue, using the
   transport's native framing (`ets-link-design.md`). Board resources and
   interrupts stay user configuration behind the board binding. The user owns
   fitting the ETS to constrained targets.
9. **Board binding.** `#[ets_setup]` passes an optional board binding with a
   watchdog feed, a wait hook and a release source. The user starts the
   watchdog and binds interrupts and timers; `Periodic` and `Triggered` runs
   take their releases from the binding.

Ground rules for review of every overhaul change: the dispatcher holds no
execution logic; modes are types; runners own loops; descriptors are data; there
are no trait objects; a new execution model is one descriptor kind, one runner
and one dispatcher arm; only `Session` writes to the link.

---

## Consequences

- Good: One firmware image and one host session serve unit profiling, lockstep
  HIL and free-running tasks.
- Good: Extending the ETS with an execution model leaves existing runners
  unchanged; the wire schema changes append-only after revision 2.
- Good: Fault, hang and unreported-fault paths are explicit and testable.
- Bad: Wire revision 2 breaks the schema once (merged run and state messages,
  settings as a typed dictionary, run verdicts in `ItemState::Done`) and
  requires updates in `control-rs-ets`, `control-rs-ets-host`,
  `control-rs-macros`, `control-rs-tui` and `control-rs-ci`.
- Bad: Execution is cooperative. A blocked step stops the executive; recovery
  needs a watchdog or an external reset.
- Bad: Interrupt handlers cannot read task state or settings; event-driven
  tasks are released only through a user-owned counter read by the board
  binding (`Triggered`).
- Follow-up: The documents under `documentation/ets/` follow this decision; the
  lifecycle-suite pull request is treated as a source of settled semantics, not
  merged.

---

## Rejected Options

- Lifecycle branches inside `Server`: couples execution models to the
  dispatcher.
- Channelized framing with a raw data plane: a payload-level encoding flag gives
  the same option without a framing break.
- Disjoint HIL and ETS frameworks: duplicates board bring-up and drivers.
- Separate case and task runners: duplicates arming, reporting and fault
  settlement.
- Trait-object descriptors and settings: fat pointers and associated types are
  not dyn-compatible in linker sections.
- Seqlock or field-addressed parameter groups: the server is the only writer,
  so plain cells need no reader retry.
- Text settings or struct settings with a target-reported schema: text makes
  the target format and parse floats; a schema needs a format, flash and
  host-side decoding.
- A typed transport trait with framing in each board implementation, or
  framing in the server: repeats reassembly per board or duplicates the
  framing USB and CAN already provide.
- An executive-started watchdog: board watchdogs differ in how they start and
  stop, and some cannot be stopped.
- `Drop`-based teardown: panic handlers do not return, so destructors do not run
  on a fault.
- Hosted integration into an RTOS and concurrent multi-task runs: out of scope
  for now.

---

## References

[1] `control-rs`, "Embedded Test Server,"
`documentation/ets/embedded-test-server-design.md`, Oct. 2026.

[2] `control-rs`, "ETS Executive," `documentation/ets/ets-executive-design.md`,
Oct. 2026.

[3] `control-rs`, "ETS Link," `documentation/ets/ets-link-design.md`, Oct. 2026.
