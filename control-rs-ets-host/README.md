# `control-rs-ets-host`

Host-side counterpart to the [`control-rs-ets`](../control-rs-ets) Embedded
Test Server. It owns the connection to a target, the wire framing and the
command/telemetry session that drives test discovery and execution. It renders
nothing and runs no quality gates.
[Design](../documentation/ets/ets-host-design.md) · [Workspace](../README.md)

| Module    | Responsibility                                                               |
|:----------|:-----------------------------------------------------------------------------|
| `target`  | Target descriptors, QEMU architecture entries, ELF path resolution and build |
| `link`    | `HostLink` trait, message links over USB bulk and CAN                        |
| `bridge`  | `ETSBridge` (QEMU subprocess and serial), `OwnedTelemetry`                   |
| `catalog` | Suites, executables and settings reported at discovery                       |
| `session` | `Session<L>`: discovery, cases, settings, device state, recovery             |
| `task_run`| `Session::run_task`, `TaskSim`, `RunObserver`, `RunControl`                  |
| `headless`| `run_headless(target, options) -> Result<RunRecord, HostError>`              |
| `state`   | `SessionState`, the action machine of the terminal dashboard                 |

Two transports: a spawned subprocess (QEMU under `cargo run`) and a USB CDC
serial port. Both resolve to a `HostLink`, so `Session` is transport-agnostic.

The interactive consumer is [`control-rs-tui`](../control-rs-tui). Headless
execution is `run_headless`; automated quality gates run through
[`control-rs-ci`](../control-rs-ci).

## License

MIT OR Apache-2.0.
