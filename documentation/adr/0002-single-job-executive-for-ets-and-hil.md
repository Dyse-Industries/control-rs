# ADR-0002: Single-Job Executive for ETS and HIL

![Date Badge](https://img.shields.io/badge/Date-October_31,_2026-blue)
![Status Badge](https://img.shields.io/badge/ADR%20Status-Proposed-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

## Context

`control-rs` verifies code across a layered hierarchy (Layer 0 host tests, Layer
0/1 target cases, Layer 1 HIL and SysId, Layer 1/2 firmware integration, Layer
1/2+ unmodified DUT; defined in `embedded-test-server-design.md`). The Embedded
Test Server (ETS) was originally structured for Layer 0/1 atomic cases over a
single peripheral before its architecture covered continuous lifecycle tasks or
multi-peripheral targets. Adding lifecycle tasks by branching inside the server
loop couples every execution model to the dispatcher, while maintaining a
separate HIL framework duplicates board bring-up, communication drivers and
online parameters. How should the target server and host session structure
execution, parameters, fault containment and multi-peripheral communication?

---

## Decision

Restructure the ETS as a single-job cyclic executive [1] with a static FIFO job
queue across one or more `HostComms` peripherals on the target
(`ets-executive-design.md`, `host-comm-design.md`) and a unified `Session` on
the host (`ets-host-design.md`):

1. **Multi-peripheral dispatcher, FIFO job queue and one runner.** The target
   `Server` polls one or more `HostComms` peripherals through a static
   `PeripheralSet` (single peripheral, heterogeneous tuple or fixed array),
   answers immediate queries (`ListSuites`, `Param` read) on the originating
   peripheral, broadcasts parameter updates (`Param` write) to all attached
   peripherals, and pushes execution requests (`RunExecutable`) tagged with
   their originating peripheral index into a fixed-capacity FIFO `JobQueue`.
   When idle, `Server` pops the next `Job` and constructs a
   `Runner<L, P, B, M: Mode>` that owns the only execution loop. Plain cases
   run as one-step `Atomic` executables; lifecycle tasks run `Lockstep`,
   `FreeRunning`, `Periodic` or `Triggered`.
2. **Modes are types.** Per-run variants are type parameters and associated
   constants on `Mode`, not runtime flags inside a loop. Target-side
   executables, runners, peripheral sets and board bindings use static dispatch
   with zero trait objects; the linker-section function pointer is the only
   indirection.
3. **Data-only descriptors.** Descriptors in flash name no server, peripheral or
   profiler type. Typed user contracts (`LifecycleTask`, `Packet`) are erased to
   byte slices only inside the `#[ets_suite]` expansion (`macros-design.md`).
4. **Named primitive parameters.** Suite parameters are named primitives (`u8`,
   `u16`, `u32`, `i32`, `f32`, `bool`) stored in `StateCell` entries in a
   `ParamTable`, read by name or index and written by name at idle or at a step
   boundary without atomics or sequence locks.
5. **Logical `HostComms` with binary framing and direct text compatibility.**
   Peripherals implement `HostComms` (`host-comm-design.md`) using either
   `FrameReader`/`FrameEncoder` binary framing, direct `Command::from_str` and
   `Telemetry::fmt` (`Display`) ASCII line compatibility for serial monitors, or
   a custom streaming protocol. During a run, `TaskSample` traffic routes to the
   job's originating peripheral while lifecycle transitions, statistics, logs
   and fault reports broadcast to all attached peripherals.
6. **Fault containment without `Drop`.** A run phase and an associated
   `safe_state` function settle a fault at most once. A board watchdog started
   in `#[ets_setup]` and fed through the `Board` binding, or a host-triggered
   reset, recovers a blocked step.
7. **Unified host `Session`.** `Session` (`ets-host-design.md`) directly owns
   the target connection (`Target::Subprocess` for QEMU or `Target::Serial` for
   hardware boards) without intermediate bridge threads, action queues or
   per-run wrapper objects, serving headless CI (`control-rs-ci`), scripted HIL
   loops and `control-rs-tui`.

---

## Consequences

- Good: One target firmware image supports unit profiling, lockstep HIL,
  free-running tasks, interactive serial-monitor control (`minicom`/`screen`)
  and simultaneous dual-endpoint USB CDC operation (`control-rs-tui` + serial
  monitor).
- Good: Copying `RunExecutable` requests into `JobQueue` releases the
  `poll_command` borrow immediately and lets companion peripherals queue
  follow-up jobs or tune parameters while a run is active.
- Good: Fault, hang and unreported-fault paths are explicit and testable without
  relying on unwinding or `Drop`.
- Bad: Wire revision 2 updates the command and telemetry schema across
  `control-rs-ets`, `control-rs-ets-host`, `control-rs-macros`, `control-rs-tui`
  and `control-rs-ci`.
- Bad: Execution is cooperative; a blocked step stops the executive until a
  board watchdog or host reset recovers the target.
- Bad: Interrupt handlers cannot access task state or parameters; event-driven
  tasks are released only through a user-owned counter read by the `Board`
  binding (`Triggered`).
- Follow-up: Target executive and multi-peripheral server implementation, owned
  by `ets-executive-design.md` and `embedded-test-server-design.md`.

---

## Rejected Options

- Lifecycle branches inside `Server`: couples execution models to the
  dispatcher.
- Byte-slice `Link` / `ByteStream` / `StreamLink` layer below `HostComms` or a
  separate `TextCodec` module: duplicates framing abstractions when `HostComms`
  with `FrameReader`/`FrameEncoder` and `FromStr`/`Display` covers both binary
  links and serial monitors directly.
- Blind broadcast of all telemetry across simultaneous peripherals: injects
  unsolicited discovery and high-rate binary `TaskSample` streams into idle
  companion terminals; origin-scoped replies plus state-change broadcast avoids
  contention.
- Separate `ETSBridge`, `CaseRun`, `TaskRun`, `TaskSim` and `RunObserver` layers
  on the host: forces callers to duplicate message-polling and reset loops that
  belong inside `Session`.
- Disjoint HIL and ETS frameworks: duplicates board bring-up and drivers.
- Trait-object descriptors, peripheral slices or settings: fat pointers and
  associated types are not dyn-compatible in linker sections and violate the
  static-dispatch constraint.
- Seqlock or field-addressed parameter groups: the server is the only writer at
  idle or step boundaries, so `StateCell` entries need no reader retry.
- `Drop`-based teardown: panic handlers do not return, so destructors never run
  on a fault.

---

## References

[1] P. E. Hood, "Ada and cyclic runtime scheduling," in *Proc. 1st Int. Conf.
Ada Programming Language Applications for the NASA Space Station, Vol. 1*,
Houston, TX, USA, 1986.
